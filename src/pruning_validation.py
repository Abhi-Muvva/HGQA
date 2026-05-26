"""
Tier-3 pruning validation helpers.

These checks are designed for medium/large grids where brute force is
infeasible and we need evidence that pruning preserves search quality well
enough before running expensive optimizers such as MPS-QAOA.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from cell_pruner import prune_cells
from helpers import chebyshev_distance


@dataclass(frozen=True)
class ScoreStats:
    best: float
    mean: float
    p5: float


def _build_dense_score_matrix(Q_obj: Dict[Tuple[int, int], float], N: int) -> np.ndarray:
    """
    Build a symmetric matrix S such that score(solution) = x^T S x.

    Q_obj stores diagonal terms directly and off-diagonal terms once in upper
    triangular form. We split each off-diagonal term equally across S[i, j]
    and S[j, i] so subset scoring can use a simple dense sum.
    """
    S = np.zeros((N, N), dtype=float)
    for (i, j), val in Q_obj.items():
        if i == j:
            S[i, i] += val
        else:
            half = 0.5 * val
            S[i, j] += half
            S[j, i] += half
    return S


def _score_solution(score_matrix: np.ndarray, solution: Sequence[int]) -> float:
    idx = np.asarray(sorted(solution), dtype=int)
    return float(score_matrix[np.ix_(idx, idx)].sum())


def _sequential_greedy(
    score_matrix: np.ndarray,
    candidates: Sequence[int],
    m: int,
) -> Tuple[List[int], float]:
    """
    Sequential greedy under the current objective over a candidate subset.
    """
    remaining = list(sorted(candidates))
    selected: List[int] = []
    current_score = 0.0

    for _ in range(m):
        best_cell = None
        best_delta = None

        for candidate in remaining:
            delta = score_matrix[candidate, candidate]
            if selected:
                delta += 2.0 * score_matrix[candidate, selected].sum()
            if best_delta is None or delta < best_delta:
                best_delta = float(delta)
                best_cell = candidate

        if best_cell is None:
            raise ValueError("Not enough candidate cells to complete greedy selection.")

        selected.append(best_cell)
        remaining.remove(best_cell)
        current_score += float(best_delta)

    return sorted(selected), float(current_score)


def _random_sample_scores(
    score_matrix: np.ndarray,
    candidates: Sequence[int],
    m: int,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    rng = np.random.default_rng(seed)
    population = np.asarray(sorted(candidates), dtype=int)
    scores = np.empty(n_samples, dtype=float)
    solutions: List[List[int]] = []

    for idx in range(n_samples):
        sample = np.sort(rng.choice(population, size=m, replace=False))
        scores[idx] = _score_solution(score_matrix, sample)
        solutions.append(sample.tolist())

    return scores, solutions


def _summarize_scores(scores: np.ndarray) -> ScoreStats:
    return ScoreStats(
        best=float(np.min(scores)),
        mean=float(np.mean(scores)),
        p5=float(np.percentile(scores, 5)),
    )


def _fitness_to_weights(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.min()
    return 1.0 / (1.0 + shifted)


def _repair_child(
    child: Sequence[int],
    candidates: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> List[int]:
    uniq = sorted(set(int(x) for x in child))
    if len(uniq) > m:
        rng.shuffle(uniq)
        uniq = sorted(uniq[:m])
    elif len(uniq) < m:
        pool = np.array([c for c in candidates if c not in uniq], dtype=int)
        add = rng.choice(pool, size=m - len(uniq), replace=False)
        uniq = sorted(uniq + add.tolist())
    return uniq


def _crossover(
    parent_a: Sequence[int],
    parent_b: Sequence[int],
    candidates: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> List[int]:
    set_a = set(parent_a)
    set_b = set(parent_b)
    common = sorted(set_a & set_b)
    child = list(common)

    mixed = sorted((set_a ^ set_b))
    rng.shuffle(mixed)
    child.extend(mixed[: max(0, m - len(child))])

    if len(child) < m:
        outside = np.array([c for c in candidates if c not in child], dtype=int)
        extra = rng.choice(outside, size=m - len(child), replace=False)
        child.extend(extra.tolist())

    return _repair_child(child, candidates, m, rng)


def _mutate(
    child: Sequence[int],
    candidates: np.ndarray,
    mutation_rate: float,
    rng: np.random.Generator,
) -> List[int]:
    updated = list(child)
    if rng.random() < mutation_rate:
        drop_idx = int(rng.integers(0, len(updated)))
        remaining = [c for c in candidates if c not in updated or c == updated[drop_idx]]
        replacement_pool = np.array(
            [c for c in remaining if c != updated[drop_idx]],
            dtype=int,
        )
        if replacement_pool.size > 0:
            updated[drop_idx] = int(rng.choice(replacement_pool))
    return sorted(updated)


def _generation_to_threshold(history: Sequence[float], final_score: float) -> int:
    start = history[0]
    improvement = start - final_score
    if improvement <= 0:
        return 0
    target = start - 0.95 * improvement
    for gen_idx, score in enumerate(history):
        if score <= target:
            return gen_idx
    return len(history) - 1


def _run_ga(
    score_matrix: np.ndarray,
    candidates: Sequence[int],
    m: int,
    population_size: int = 100,
    generations: int = 500,
    elite_count: int = 10,
    mutation_rate: float = 0.15,
    seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)
    candidate_arr = np.asarray(sorted(candidates), dtype=int)

    population = [
        sorted(rng.choice(candidate_arr, size=m, replace=False).tolist())
        for _ in range(population_size)
    ]

    best_solution: List[int] | None = None
    best_score = float("inf")
    history: List[float] = []
    t0 = time.perf_counter()

    for _ in range(generations):
        scores = np.array([_score_solution(score_matrix, sol) for sol in population], dtype=float)
        ranking = np.argsort(scores)
        population = [population[i] for i in ranking]
        scores = scores[ranking]

        if scores[0] < best_score:
            best_score = float(scores[0])
            best_solution = list(population[0])

        history.append(float(scores[0]))

        elites = population[:elite_count]
        weights = _fitness_to_weights(scores)
        probs = weights / weights.sum()

        next_population = [list(sol) for sol in elites]
        while len(next_population) < population_size:
            parent_ids = rng.choice(len(population), size=2, replace=False, p=probs)
            child = _crossover(
                population[int(parent_ids[0])],
                population[int(parent_ids[1])],
                candidate_arr,
                m,
                rng,
            )
            child = _mutate(child, candidate_arr, mutation_rate, rng)
            next_population.append(child)

        population = next_population

    runtime_sec = time.perf_counter() - t0
    final_score = history[-1]
    return {
        "best_score": float(best_score),
        "best_solution": best_solution,
        "history": history,
        "generations_to_95pct_final": _generation_to_threshold(history, final_score),
        "runtime_sec": runtime_sec,
    }


def _pct_gap(candidate: float, reference: float) -> float:
    denom = max(abs(reference), 1e-9)
    return 100.0 * abs(candidate - reference) / denom


def _status(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def _detect_knee(sweep_rows: Sequence[Dict], tolerance_pct: float = 1.0) -> int | None:
    """
    Return the smallest keep value after which greedy score is effectively flat.
    """
    if not sweep_rows:
        return None

    scores = [row["greedy_score"] for row in sweep_rows]
    best_future = np.minimum.accumulate(np.asarray(scores[::-1]))[::-1]

    for idx, row in enumerate(sweep_rows):
        future_best = float(best_future[idx])
        if _pct_gap(row["greedy_score"], future_best) <= tolerance_pct:
            return int(row["tier3_keep"])

    return None


def _perturbation_selection_frequency(
    score_matrix: np.ndarray,
    surviving_cells: Sequence[int],
    m: int,
    n_perturbations: int,
    sigma_scale: float,
    seed: int,
) -> Dict:
    """
    Run greedy selection N times with small diagonal noise; track per-cell
    selection frequency to identify low-evidence survivors.

    Diagonal noise ~ Normal(0, sigma_scale * diag_std). Off-diagonal terms are
    left unchanged so spatial interactions are not distorted.
    """
    rng = np.random.default_rng(seed)
    surviving_arr = list(sorted(surviving_cells))

    diag_vals = np.array([score_matrix[c, c] for c in surviving_arr], dtype=float)
    diag_std = float(np.std(diag_vals)) if len(diag_vals) > 1 else 1.0
    sigma = sigma_scale * diag_std

    # Save originals so we can restore exactly (avoids float accumulation drift)
    original_diag = {c: float(score_matrix[c, c]) for c in surviving_arr}
    selection_counts: Dict[int, int] = {c: 0 for c in surviving_arr}

    for _ in range(n_perturbations):
        noise = rng.normal(0.0, sigma, len(surviving_arr))
        for i, cell in enumerate(surviving_arr):
            score_matrix[cell, cell] = original_diag[cell] + noise[i]

        selected_cells, _ = _sequential_greedy(score_matrix, surviving_arr, m)
        for c in selected_cells:
            selection_counts[c] += 1

    # Restore exact originals
    for cell in surviving_arr:
        score_matrix[cell, cell] = original_diag[cell]

    freq = {c: selection_counts[c] / n_perturbations for c in surviving_arr}

    core       = sorted(c for c, f in freq.items() if f > 0.80)
    competitive = sorted(c for c, f in freq.items() if 0.01 < f <= 0.80)
    marginal   = sorted(c for c, f in freq.items() if f <= 0.01)
    K_effective = len(surviving_arr) - len(marginal)

    return {
        "status": "INFO",
        "n_perturbations": n_perturbations,
        "sigma_scale": sigma_scale,
        "diag_std": diag_std,
        "selection_freq": {c: round(freq[c], 4) for c in surviving_arr},
        "core_cells": core,
        "competitive_cells": competitive,
        "marginal_cells": marginal,
        "K": len(surviving_arr),
        "K_effective": K_effective,
        "marginal_fraction": len(marginal) / len(surviving_arr) if surviving_arr else 0.0,
    }


def run_pruning_validation(
    Q_obj: Dict[Tuple[int, int], float],
    N: int,
    m: int,
    num_cols: int,
    R4: int,
    surviving_cells: Sequence[int],
    tier3_keep_values: Sequence[int] | None = None,
    greedy_gap_threshold_pct: float = 5.0,
    random_samples: int = 10_000,
    ga_generations: int = 500,
    ga_population: int = 100,
    seed: int = 42,
    overkeeping_pass_ratio: float = 1.15,
    overkeeping_fail_ratio: float = 1.50,
    n_perturbations: int = 50,
    perturbation_sigma_scale: float = 0.10,
) -> Dict:
    """
    Run seven Tier-3 scale-validation checks and return a pass/fail report.

    Checks 1-5 detect over-pruning (quality loss from pruning too aggressively).
    Check 6 detects over-keeping (efficiency loss from pruning too little).
    Check 7 identifies low-evidence survivors via perturbation selection frequency.

    Notes
    -----
    - Full-vs-pruned scores are evaluated on the same objective. The pruned
      score is the restricted search-space objective, equivalent to scoring on
      the remapped pruned QUBO.
    - Lower scores are better throughout.
    - Check 6 status: PASS if current_K <= pass_ratio * K_at_knee,
      WARN if <= fail_ratio * K_at_knee, FAIL otherwise.
    - Check 7 is always INFO — it is a diagnostic, not a correctness gate.
      Set n_perturbations=0 to skip it.
    """
    score_matrix = _build_dense_score_matrix(Q_obj, N)
    all_cells = list(range(N))
    surviving_cells = sorted(surviving_cells)
    surviving_set = set(surviving_cells)

    if len(surviving_cells) < m:
        raise ValueError(f"Only {len(surviving_cells)} cells survived, but m={m}.")

    # Check 1: greedy comparison
    greedy_full_cells, greedy_full_score = _sequential_greedy(score_matrix, all_cells, m)
    greedy_pruned_cells, greedy_pruned_score = _sequential_greedy(score_matrix, surviving_cells, m)
    greedy_gap_pct = _pct_gap(greedy_pruned_score, greedy_full_score)
    same_cells = greedy_full_cells == greedy_pruned_cells
    check1_pass = greedy_gap_pct <= greedy_gap_threshold_pct

    # Check 2: random sampling
    full_scores, full_samples = _random_sample_scores(
        score_matrix, all_cells, m, random_samples, seed
    )
    pruned_scores, pruned_samples = _random_sample_scores(
        score_matrix, surviving_cells, m, random_samples, seed + 1
    )
    random_full = _summarize_scores(full_scores)
    random_pruned = _summarize_scores(pruned_scores)
    random_best_gap_pct = _pct_gap(random_pruned.best, random_full.best)
    random_mean_pass = random_pruned.mean <= random_full.mean
    random_p5_pass = random_pruned.p5 <= random_full.p5
    random_best_pass = random_pruned.best <= random_full.best or random_best_gap_pct <= greedy_gap_threshold_pct
    check2_pass = random_mean_pass and random_p5_pass and random_best_pass

    # Check 3: GA convergence
    ga_full = _run_ga(
        score_matrix,
        all_cells,
        m,
        population_size=ga_population,
        generations=ga_generations,
        seed=seed,
    )
    ga_pruned = _run_ga(
        score_matrix,
        surviving_cells,
        m,
        population_size=ga_population,
        generations=ga_generations,
        seed=seed + 1,
    )
    ga_gap_pct = _pct_gap(ga_pruned["best_score"], ga_full["best_score"])
    ga_score_pass = ga_pruned["best_score"] <= ga_full["best_score"] or ga_gap_pct <= greedy_gap_threshold_pct
    ga_speed_pass = ga_pruned["generations_to_95pct_final"] <= ga_full["generations_to_95pct_final"]
    check3_pass = ga_score_pass and ga_speed_pass

    # Check 4: neighbor coverage around full-Q greedy cells
    coverage_rows = []
    check4_pass = True
    for cell in greedy_full_cells:
        neighbors = [
            other for other in surviving_cells
            if other != cell and chebyshev_distance(cell, other, num_cols) <= 2
        ]
        row = {
            "cell": cell,
            "cell_survived": cell in surviving_set,
            "surviving_neighbors_d2": len(neighbors),
            "neighbors": neighbors,
        }
        coverage_rows.append(row)
        if len(neighbors) == 0:
            check4_pass = False

    # Check 5: sensitivity sweep
    if tier3_keep_values is None:
        tier3_keep_values = [2, 3, 4, 5, m]
    sweep_rows = []
    for keep in sorted(set(int(v) for v in tier3_keep_values)):
        sweep_survivors, sweep_report = prune_cells(
            Q_obj,
            N=N,
            m=m,
            num_cols=num_cols,
            R4=R4,
            tier3_keep=keep,
            enforce_tier3_keep_at_least_m=False,
            verbose=False,
        )
        sweep_greedy_cells, sweep_greedy_score = _sequential_greedy(score_matrix, sweep_survivors, m)
        sweep_rows.append({
            "tier3_keep": keep,
            "effective_keep": sweep_report["tier3"]["keep_per_cluster"],
            "K": len(sweep_survivors),
            "greedy_score": sweep_greedy_score,
            "greedy_cells": sweep_greedy_cells,
            "greedy_gap_vs_full_pct": _pct_gap(sweep_greedy_score, greedy_full_score),
        })

    knee_keep = _detect_knee(sweep_rows)
    check5_status = "PASS" if knee_keep is not None and knee_keep != max(row["tier3_keep"] for row in sweep_rows) else "INFO"

    # Check 6: over-keeping (derived from check_5 sweep data — no extra computation)
    current_K = len(surviving_cells)
    K_at_knee: int | None = None
    if knee_keep is not None:
        for row in sweep_rows:
            if row["tier3_keep"] == knee_keep:
                K_at_knee = row["K"]
                break

    if K_at_knee is None:
        check6_status = "INFO"
        check6_ratio: float | None = None
    else:
        check6_ratio = current_K / K_at_knee
        if check6_ratio <= overkeeping_pass_ratio:
            check6_status = "PASS"
        elif check6_ratio <= overkeeping_fail_ratio:
            check6_status = "WARN"
        else:
            check6_status = "FAIL"

    # Check 7: perturbation selection frequency (optional — skip if n_perturbations=0)
    if n_perturbations > 0:
        check7: Dict = _perturbation_selection_frequency(
            score_matrix, surviving_cells, m,
            n_perturbations=n_perturbations,
            sigma_scale=perturbation_sigma_scale,
            seed=seed + 10,
        )
    else:
        check7 = {"status": "SKIP", "K": current_K, "K_effective": current_K}

    results = {
        "summary": {
            "original_N": N,
            "surviving_K": len(surviving_cells),
            "reduction_pct": 100.0 * (1.0 - len(surviving_cells) / N),
            "m": m,
        },
        "check_1_greedy_comparison": {
            "status": _status(check1_pass),
            "full_greedy_cells": greedy_full_cells,
            "full_greedy_score": greedy_full_score,
            "pruned_greedy_cells": greedy_pruned_cells,
            "pruned_greedy_score": greedy_pruned_score,
            "same_cells": same_cells,
            "score_gap_pct": greedy_gap_pct,
            "threshold_pct": greedy_gap_threshold_pct,
        },
        "check_2_random_sampling": {
            "status": _status(check2_pass),
            "samples_per_space": random_samples,
            "full": random_full.__dict__,
            "pruned": random_pruned.__dict__,
            "best_gap_pct": random_best_gap_pct,
            "mean_pass": random_mean_pass,
            "p5_pass": random_p5_pass,
            "best_pass": random_best_pass,
            "best_full_solution": full_samples[int(np.argmin(full_scores))],
            "best_pruned_solution": pruned_samples[int(np.argmin(pruned_scores))],
        },
        "check_3_ga_convergence": {
            "status": _status(check3_pass),
            "full": ga_full,
            "pruned": ga_pruned,
            "best_gap_pct": ga_gap_pct,
            "best_score_pass": ga_score_pass,
            "speed_pass": ga_speed_pass,
        },
        "check_4_neighbor_coverage": {
            "status": _status(check4_pass),
            "rows": coverage_rows,
        },
        "check_5_sensitivity_sweep": {
            "status": check5_status,
            "rows": sweep_rows,
            "knee_keep": knee_keep,
            "note": (
                "INFO means the sweep is reported but no clear knee was detected "
                "before the largest keep value tested."
            ),
        },
        "check_6_overkeeping": {
            "status": check6_status,
            "current_K": current_K,
            "K_at_knee": K_at_knee,
            "knee_keep": knee_keep,
            "ratio": check6_ratio,
            "pass_ratio": overkeeping_pass_ratio,
            "fail_ratio": overkeeping_fail_ratio,
            "note": (
                "PASS/WARN/FAIL based on current_K / K_at_knee ratio. "
                "WARN = efficiency loss (too many qubits for solver). "
                "FAIL = severe over-keeping. Not a correctness failure."
            ),
        },
        "check_7_perturbation_frequency": check7,
    }
    return results


def print_pruning_validation_report(results: Dict) -> None:
    """
    Human-readable summary focused on pass/fail decisions.
    """
    summary = results["summary"]
    print("=" * 72)
    print("TIER-3 SCALE VALIDATION")
    print(
        f"N={summary['original_N']}  K={summary['surviving_K']}  "
        f"reduction={summary['reduction_pct']:.1f}%  m={summary['m']}"
    )
    print("=" * 72)

    c1 = results["check_1_greedy_comparison"]
    print(f"Check 1 - Greedy comparison: {c1['status']}")
    print(
        f"  full={c1['full_greedy_score']:.4f}  pruned={c1['pruned_greedy_score']:.4f}  "
        f"gap={c1['score_gap_pct']:.2f}%  same_cells={c1['same_cells']}"
    )

    c2 = results["check_2_random_sampling"]
    print(f"Check 2 - Random sampling: {c2['status']}")
    print(
        f"  best full/pruned={c2['full']['best']:.4f}/{c2['pruned']['best']:.4f}  "
        f"mean full/pruned={c2['full']['mean']:.4f}/{c2['pruned']['mean']:.4f}  "
        f"p5 full/pruned={c2['full']['p5']:.4f}/{c2['pruned']['p5']:.4f}"
    )

    c3 = results["check_3_ga_convergence"]
    print(f"Check 3 - GA convergence: {c3['status']}")
    print(
        f"  best full/pruned={c3['full']['best_score']:.4f}/{c3['pruned']['best_score']:.4f}  "
        f"95% gen full/pruned={c3['full']['generations_to_95pct_final']}/"
        f"{c3['pruned']['generations_to_95pct_final']}"
    )

    c4 = results["check_4_neighbor_coverage"]
    print(f"Check 4 - Neighbor coverage: {c4['status']}")
    zero_neighbors = [row["cell"] for row in c4["rows"] if row["surviving_neighbors_d2"] == 0]
    if zero_neighbors:
        print(f"  Greedy cells with zero surviving d<=2 neighbors: {zero_neighbors}")
    else:
        print("  Every full-greedy cell kept at least one surviving d<=2 alternative.")

    c5 = results["check_5_sensitivity_sweep"]
    print(f"Check 5 - Sensitivity sweep: {c5['status']}")
    print(f"  knee_keep={c5['knee_keep']}")
    for row in c5["rows"]:
        print(
            f"  keep={row['tier3_keep']:>2} -> K={row['K']:>4}  "
            f"greedy={row['greedy_score']:.4f}  gap_vs_full={row['greedy_gap_vs_full_pct']:.2f}%"
        )

    c6 = results.get("check_6_overkeeping", {})
    if c6:
        print(f"Check 6 - Over-keeping: {c6['status']}")
        if c6.get("K_at_knee") is not None:
            print(
                f"  current_K={c6['current_K']}  K_at_knee={c6['K_at_knee']}  "
                f"ratio={c6['ratio']:.2f}  "
                f"(PASS≤{c6['pass_ratio']}x  WARN≤{c6['fail_ratio']}x  FAIL>)"
            )
        else:
            print("  No sweep knee detected — over-keeping cannot be assessed.")

    c7 = results.get("check_7_perturbation_frequency", {})
    if c7 and c7.get("status") not in ("SKIP", None):
        print(f"Check 7 - Perturbation frequency ({c7.get('n_perturbations', '?')} runs): {c7['status']}")
        print(
            f"  K={c7['K']}  K_effective={c7['K_effective']}  "
            f"core={len(c7.get('core_cells', []))} (>80%)  "
            f"competitive={len(c7.get('competitive_cells', []))} (1-80%)  "
            f"marginal={len(c7.get('marginal_cells', []))} (≤1%)"
        )
        if c7.get("marginal_cells"):
            print(f"  Low-evidence survivors: {c7['marginal_cells']}")
    elif c7.get("status") == "SKIP":
        print("Check 7 - Perturbation frequency: SKIP")

    print("=" * 72)


def run_dataset_validation_case(
    dataset_file: str,
    N_qubits: int,
    scale_factor: float = 5.0,
    min_weight: float = 0.5,
    tier3_keep: int = 3,
    tier3_keep_values: Sequence[int] | None = None,
    random_samples: int = 10_000,
    ga_generations: int = 500,
    ga_population: int = 100,
    seed: int = 42,
) -> Dict:
    """
    Build the full QUBO pipeline for one dataset/grid-size pair, then run the
    Tier-3 validation checklist on that configuration.

    Imports are local so this module stays usable even if optional notebook
    dependencies such as pandas are unavailable in the current shell.
    """
    from data_loader import load_dataset
    from helpers import divide_graph_into_parts, calculate_cell_weights, suggest_parameters
    from qubo_builder import build_qubo, calibrate_alphas

    data = load_dataset(dataset_file)
    grid_details, plot_deets = divide_graph_into_parts(
        x_min=data["x_min"],
        x_max=data["x_max"],
        y_min=data["y_min"],
        y_max=data["y_max"],
        num_qubits=N_qubits,
        points_of_interest=data["pois"],
        existing_chargers=data["existing_chargers"],
        gas_stations=data["gas_stations"],
        grid_division="default",
    )

    cell_weights = calculate_cell_weights(
        grid_details,
        scale_factor=scale_factor,
        min_weight=min_weight,
    )
    params = suggest_parameters(grid_details, cell_weights, plot_deets, m=data["m"])
    a = params["alpha"]
    r = params["radii"]
    t = params["intra"]

    Q_obj, h5_params, diags = build_qubo(
        grid_details,
        cell_weights,
        plot_deets,
        m=data["m"],
        alpha1=a["a1"],
        alpha2=a["a2"],
        alpha3=a["a3"],
        alpha4=a["a4"],
        alpha5=a["a5"],
        alpha6=a["a6"],
        beta=t["beta"],
        gamma=t["gamma"],
        delta=t["delta"],
        epsilon=t["epsilon"],
        lam=params["lambda"],
        R1=r["R1"],
        Rs=r["Rs"],
        R3=r["R3"],
        R4=r["R4"],
        R6=r["R6"],
    )

    params_flat = {
        "alpha1": a["a1"],
        "alpha2": a["a2"],
        "alpha3": a["a3"],
        "alpha4": a["a4"],
        "alpha5": a["a5"],
        "alpha6": a["a6"],
        "beta": t["beta"],
        "gamma": t["gamma"],
        "delta": t["delta"],
        "epsilon": t["epsilon"],
        "lam": params["lambda"],
        "R1": r["R1"],
        "Rs": r["Rs"],
        "R3": r["R3"],
        "R4": r["R4"],
        "R6": r["R6"],
    }
    params_flat, needs_rebuild = calibrate_alphas(Q_obj, params_flat, diags, verbose=False)
    if needs_rebuild:
        Q_obj, h5_params, diags = build_qubo(
            grid_details,
            cell_weights,
            plot_deets,
            m=data["m"],
            **params_flat,
        )

    surviving_cells, prune_report = prune_cells(
        Q_obj,
        N=N_qubits,
        m=data["m"],
        num_cols=plot_deets["num_cols"],
        R4=params_flat["R4"],
        tier3_keep=tier3_keep,
        verbose=False,
    )

    validation = run_pruning_validation(
        Q_obj=Q_obj,
        N=N_qubits,
        m=data["m"],
        num_cols=plot_deets["num_cols"],
        R4=params_flat["R4"],
        surviving_cells=surviving_cells,
        tier3_keep_values=tier3_keep_values,
        random_samples=random_samples,
        ga_generations=ga_generations,
        ga_population=ga_population,
        seed=seed,
    )

    return {
        "dataset_file": dataset_file,
        "dataset_name": data.get("dataset_name", dataset_file),
        "N_qubits": N_qubits,
        "m": data["m"],
        "plot_deets": plot_deets,
        "params": params_flat,
        "h5_params": h5_params,
        "prune_report": prune_report,
        "validation": validation,
    }


def run_dataset_validation_suite(
    cases: Sequence[Dict],
    scale_factor: float = 5.0,
    min_weight: float = 0.5,
    tier3_keep: int = 3,
    tier3_keep_values: Sequence[int] | None = None,
    random_samples: int = 10_000,
    ga_generations: int = 500,
    ga_population: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Run the checklist across multiple dataset/grid-size configurations.

    Each case must contain:
      - dataset_file
      - N_qubits
    and may optionally include:
      - label
      - tier3_keep
      - tier3_keep_values
      - seed
    """
    results = []
    for idx, case in enumerate(cases):
        case_result = run_dataset_validation_case(
            dataset_file=case["dataset_file"],
            N_qubits=case["N_qubits"],
            scale_factor=scale_factor,
            min_weight=min_weight,
            tier3_keep=case.get("tier3_keep", tier3_keep),
            tier3_keep_values=case.get("tier3_keep_values", tier3_keep_values),
            random_samples=case.get("random_samples", random_samples),
            ga_generations=case.get("ga_generations", ga_generations),
            ga_population=case.get("ga_population", ga_population),
            seed=case.get("seed", seed + idx),
        )
        case_result["label"] = case.get(
            "label",
            f"{case_result['dataset_name']} @ N={case_result['N_qubits']}",
        )
        results.append(case_result)
    return results


def print_dataset_validation_suite(results: Sequence[Dict]) -> None:
    """
    Compact cross-case summary for Medium/Large/Metro comparisons.
    """
    print("=" * 96)
    print("TIER-3 MULTI-DATASET VALIDATION SUMMARY")
    print("=" * 96)
    for case in results:
        validation = case["validation"]
        prune_report = case["prune_report"]
        c1 = validation["check_1_greedy_comparison"]
        c2 = validation["check_2_random_sampling"]
        c3 = validation["check_3_ga_convergence"]
        c4 = validation["check_4_neighbor_coverage"]
        c5 = validation["check_5_sensitivity_sweep"]
        c6 = validation.get("check_6_overkeeping", {})
        c7 = validation.get("check_7_perturbation_frequency", {})

        print(case["label"])
        print(
            f"  prune: {prune_report['original_N']} -> {prune_report['pruned_N']} "
            f"({100.0 * (1.0 - prune_report['pruned_N'] / prune_report['original_N']):.1f}% cut)"
        )
        print(
            f"  checks: greedy={c1['status']}  random={c2['status']}  "
            f"ga={c3['status']}  neighbors={c4['status']}  sweep={c5['status']}  "
            f"over-keeping={c6.get('status', 'N/A')}"
        )
        print(
            f"  greedy gap={c1['score_gap_pct']:.2f}%  "
            f"ga gap={c3['best_gap_pct']:.2f}%  knee={c5['knee_keep']}"
            + (
                f"  K_at_knee={c6['K_at_knee']}  ratio={c6['ratio']:.2f}"
                if c6.get("K_at_knee") is not None else ""
            )
        )
        if c7.get("status") == "INFO":
            print(
                f"  perturbation: K_effective={c7['K_effective']}  "
                f"marginal={len(c7.get('marginal_cells', []))}  "
                f"({c7.get('marginal_fraction', 0)*100:.1f}% low-evidence)"
            )
        print("-" * 96)
