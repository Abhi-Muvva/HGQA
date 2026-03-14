"""
Cell Pruning Module
====================
Reduces QUBO variable count by eliminating grid cells that provably
cannot appear in any optimal solution (Tiers 1–2) or are redundant
neighbors of better candidates (Tier 3).

Pipeline position: AFTER build_qubo() on the full grid, BEFORE QAOA.

The pruner operates on the already-built Q_obj so that solo scores
exactly match the QUBO diagonals (including normalization).  For large
grids a pre-QUBO variant can be written later — the math is identical.

Key mathematical property used throughout:
  - All off-diagonal entries in Q_obj (H4, H6) are >= 0 (penalties).
  - Therefore solo(i) = Q_obj[(i,i)] is an UPPER BOUND on cell i's
    net contribution to any solution (pairwise terms can only add cost).
"""

import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple, Set, Optional

from helpers import chebyshev_distance


# ---------------------------------------------------------------------------
# Core: extract solo scores from Q_obj
# ---------------------------------------------------------------------------

def compute_solo_scores(Q_obj: Dict, N: int) -> np.ndarray:
    """
    Extract the diagonal of Q_obj as solo scores.

    solo(i) = Q_obj[(i,i)]  — the net reward/penalty of placing a single
    charger in cell i, ignoring all pairwise interactions.

    H5 is excluded (not in Q_obj by design).

    Returns
    -------
    np.ndarray of shape (N,)
    """
    solos = np.zeros(N)
    for i in range(N):
        solos[i] = Q_obj.get((i, i), 0.0)
    return solos


# ---------------------------------------------------------------------------
# Tier 1: Dead cell elimination
# ---------------------------------------------------------------------------

def _tier1_dead_cells(
    solos: np.ndarray,
    N: int,
    num_cols: int,
    R4: int,
) -> Tuple[Set[int], dict]:
    """
    Remove cells with solo = 0 that are also beyond R4 of any
    competitive cell (solo < 0).

    A zero-solo cell within R4 of a competitive cell is kept because
    it may provide pairwise "pressure relief" (reduce H4 spacing
    penalties by offering a distant placement option).

    Returns
    -------
    surviving : set of cell IDs that survive Tier 1
    report   : dict with pruning statistics
    """
    competitive = {i for i in range(N) if solos[i] < 0}
    zero_cells = {i for i in range(N) if solos[i] == 0.0}
    negative_cells = {i for i in range(N) if solos[i] < 0.0}

    pruned_t1 = set()
    kept_zero = set()

    for i in zero_cells:
        # Check if within R4 of ANY competitive cell
        near_competitive = any(
            chebyshev_distance(i, c, num_cols) <= R4
            for c in competitive
        )
        if near_competitive:
            kept_zero.add(i)
        else:
            pruned_t1.add(i)

    surviving = set(range(N)) - pruned_t1

    report = {
        'zero_cells': len(zero_cells),
        'pruned': len(pruned_t1),
        'kept_zero_near_competitive': len(kept_zero),
        'surviving': len(surviving),
    }
    return surviving, report


# ---------------------------------------------------------------------------
# Tier 2: Bound-based elimination
# ---------------------------------------------------------------------------

def _tier2_bound_elimination(
    solos: np.ndarray,
    Q_obj: Dict,
    surviving: Set[int],
    m: int,
) -> Tuple[Set[int], dict]:
    """
    Prune cells whose best-case inclusion cannot beat a known
    achievable reference score.

    For each cell i in the surviving set:
      LB(i) = solo(i) + sum of (m-1) best other solos
    This is a true lower bound (ignoring pairwise penalties >= 0).

    Reference score = greedy solution using m best solos, evaluated
    with actual pairwise penalties from Q_obj.

    If LB(i) > reference_score, cell i is provably not in any optimal
    solution.

    Returns
    -------
    surviving : updated set of surviving cell IDs
    report    : dict with pruning statistics
    """
    cells = sorted(surviving)

    # Sort surviving cells by solo score (most negative first)
    sorted_by_solo = sorted(cells, key=lambda c: solos[c])

    # Sequential greedy: pick cells one at a time, each time choosing
    # the cell that minimizes the TOTAL score (solo + pairwise with
    # already-selected cells).  This produces a much tighter reference
    # than just picking m best solos.
    greedy_m = []
    remaining = set(cells)
    for _ in range(m):
        best_cell = None
        best_total = float('inf')
        for candidate in remaining:
            trial = greedy_m + [candidate]
            trial_set = set(trial)
            score = 0.0
            for (i, j), val in Q_obj.items():
                if i == j:
                    if i in trial_set:
                        score += val
                else:
                    if i in trial_set and j in trial_set:
                        score += val
            if score < best_total:
                best_total = score
                best_cell = candidate
        greedy_m.append(best_cell)
        remaining.discard(best_cell)

    # Evaluate greedy reference WITH pairwise penalties
    ref_score = 0.0
    greedy_set = set(greedy_m)
    for (i, j), val in Q_obj.items():
        if i == j:
            if i in greedy_set:
                ref_score += val
        else:
            if i in greedy_set and j in greedy_set:
                ref_score += val

    # For each non-greedy cell, compute lower bound
    pruned_t2 = set()
    top_solos = sorted([solos[c] for c in cells])  # ascending (most negative first)

    for cell in cells:
        if cell in greedy_set:
            continue  # never prune greedy-selected cells

        # Best case for any solution containing this cell:
        # solo(cell) + (m-1) best solos from OTHER cells
        other_solos = sorted([solos[c] for c in cells if c != cell])
        best_m_minus_1 = sum(other_solos[:m - 1])
        lb = solos[cell] + best_m_minus_1

        if lb > ref_score:
            pruned_t2.add(cell)

    surviving = surviving - pruned_t2

    report = {
        'greedy_cells': greedy_m,
        'greedy_score': ref_score,
        'pruned': len(pruned_t2),
        'surviving': len(surviving),
    }
    return surviving, report


# ---------------------------------------------------------------------------
# Tier 3: Spatial deduplication
# ---------------------------------------------------------------------------

def _tier3_spatial_dedup(
    solos: np.ndarray,
    surviving: Set[int],
    num_cols: int,
    m: int,
    keep_per_cluster: int = 3,
    score_similarity: float = 0.05,
) -> Tuple[Set[int], dict]:
    """
    Among surviving cells, group cells that are BOTH adjacent (distance <= 1)
    AND have nearly identical solo scores into micro-clusters.  Keep the
    top-K by solo score per cluster.

    Score similarity gate: two adjacent cells are only merged if their
    solo scores differ by less than score_similarity × |solo_range|.
    Tighter default (0.05) prevents false merges between cells that
    happen to be adjacent but play different roles in the solution.

    keep_per_cluster scales with m: we keep max(keep_per_cluster, m)
    per cluster to ensure enough candidates survive for the optimizer.

    Minimum survival floor: never prune below 3*m cells.

    This is a heuristic — not provably optimal-preserving.
    Must be validated on small instances via brute-force.

    Returns
    -------
    surviving : updated set
    report    : dict
    """
    cells = sorted(surviving)
    effective_keep = max(keep_per_cluster, m)
    min_survive = max(3 * m, 6)

    # If already at or below floor, skip entirely
    if len(cells) <= min_survive:
        return surviving, {
            'num_clusters': 0, 'cluster_sizes': [],
            'pruned': 0, 'surviving': len(surviving),
            'keep_per_cluster': effective_keep,
            'skipped': 'below minimum survival floor',
        }

    # Score range for similarity gate
    solo_vals = [solos[c] for c in cells]
    solo_range = max(solo_vals) - min(solo_vals)
    if solo_range == 0:
        solo_range = 1.0
    threshold = score_similarity * solo_range

    # BFS clustering: adjacent AND very similar score
    visited = set()
    clusters = []

    for seed in cells:
        if seed in visited:
            continue
        cluster = []
        queue = [seed]
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            cluster.append(curr)
            for other in cells:
                if other not in visited:
                    if (chebyshev_distance(curr, other, num_cols) <= 1
                            and abs(solos[curr] - solos[other]) <= threshold):
                        queue.append(other)
        clusters.append(cluster)

    # Within each cluster, keep top-K by solo score
    pruned_t3 = set()
    cluster_report = []

    for cluster in clusters:
        if len(cluster) <= effective_keep:
            cluster_report.append((len(cluster), 0))
            continue

        ranked = sorted(cluster, key=lambda c: solos[c])
        keep = set(ranked[:effective_keep])
        drop = set(cluster) - keep
        pruned_t3.update(drop)
        cluster_report.append((len(cluster), len(drop)))

    # Enforce minimum survival floor
    if len(surviving) - len(pruned_t3) < min_survive:
        recoverable = sorted(pruned_t3, key=lambda c: solos[c])
        while len(surviving) - len(pruned_t3) < min_survive and recoverable:
            pruned_t3.discard(recoverable.pop(0))

    surviving = surviving - pruned_t3

    report = {
        'num_clusters': len(clusters),
        'cluster_sizes': [len(c) for c in clusters],
        'pruned': len(pruned_t3),
        'surviving': len(surviving),
        'keep_per_cluster': effective_keep,
        'score_threshold': threshold,
        'min_survival_floor': min_survive,
    }
    return surviving, report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prune_cells(
    Q_obj: Dict,
    N: int,
    m: int,
    num_cols: int,
    R4: int,
    tier3_keep: int = 3,
    verbose: bool = True,
) -> Tuple[List[int], Dict]:
    """
    Run all three pruning tiers and return surviving cell IDs.

    Parameters
    ----------
    Q_obj      : sparse Q from build_qubo() — full N-cell version
    N          : total grid cells
    m          : chargers to place
    num_cols   : grid columns (for Chebyshev distance)
    R4         : H4 spacing radius (used in Tier 1 safety check)
    tier3_keep : cells to keep per micro-cluster in Tier 3
    verbose    : print report

    Returns
    -------
    surviving_cells : sorted list of original cell IDs that survive
    full_report     : dict with per-tier details
    """
    solos = compute_solo_scores(Q_obj, N)

    if verbose:
        n_neg = np.sum(solos < 0)
        n_zero = np.sum(solos == 0)
        n_pos = np.sum(solos > 0)
        print("=" * 60)
        print("CELL PRUNING")
        print(f"  Starting cells: {N}")
        print(f"  Solo scores: {n_neg} negative, {n_zero} zero, {n_pos} positive")
        print(f"  Solo range: [{solos.min():.4f}, {solos.max():.4f}]")

    # Tier 1
    surviving, t1_report = _tier1_dead_cells(solos, N, num_cols, R4)
    if verbose:
        print(f"\n  Tier 1 (dead cells): {t1_report['pruned']} pruned, "
              f"{t1_report['surviving']} remain"
              f"  ({t1_report['kept_zero_near_competitive']} zero-solo cells "
              f"kept near competitive)")

    # Tier 2
    surviving, t2_report = _tier2_bound_elimination(solos, Q_obj, surviving, m)
    if verbose:
        print(f"  Tier 2 (bound elim): {t2_report['pruned']} pruned, "
              f"{t2_report['surviving']} remain"
              f"  (ref score={t2_report['greedy_score']:.4f})")

    # Tier 3
    surviving, t3_report = _tier3_spatial_dedup(
        solos, surviving, num_cols, m=m, keep_per_cluster=tier3_keep
    )
    if verbose:
        print(f"  Tier 3 (dedup):      {t3_report['pruned']} pruned, "
              f"{t3_report['surviving']} remain"
              f"  ({t3_report['num_clusters']} clusters, "
              f"sizes={t3_report['cluster_sizes']})")

    surviving_cells = sorted(surviving)

    if verbose:
        print(f"\n  RESULT: {N} → {len(surviving_cells)} cells "
              f"({100*(1 - len(surviving_cells)/N):.0f}% reduction)")
        print(f"  Surviving IDs: {surviving_cells}")
        print(f"  Solo scores: {[round(solos[c], 4) for c in surviving_cells]}")
        print("=" * 60)

    full_report = {
        'solos': solos,
        'tier1': t1_report,
        'tier2': t2_report,
        'tier3': t3_report,
        'surviving_cells': surviving_cells,
        'original_N': N,
        'pruned_N': len(surviving_cells),
    }
    return surviving_cells, full_report


# ---------------------------------------------------------------------------
# Q_obj remapping for QAOA
# ---------------------------------------------------------------------------

def remap_qubo_for_qaoa(
    Q_obj: Dict,
    surviving_cells: List[int],
) -> Tuple[Dict, Dict[int, int], Dict[int, int]]:
    """
    Filter Q_obj to surviving cells and remap keys to 0..K-1.

    Parameters
    ----------
    Q_obj           : full N-cell sparse Q
    surviving_cells : sorted list of original cell IDs

    Returns
    -------
    Q_pruned     : dict {(new_i, new_j): value} — K-variable QUBO
    cell_to_qubit: {original_cell_id: qubit_index}
    qubit_to_cell: {qubit_index: original_cell_id}
    """
    cell_to_qubit = {cell: idx for idx, cell in enumerate(surviving_cells)}
    qubit_to_cell = {idx: cell for idx, cell in enumerate(surviving_cells)}
    surviving_set = set(surviving_cells)

    Q_pruned = {}
    for (i, j), val in Q_obj.items():
        if i in surviving_set and j in surviving_set:
            new_i = cell_to_qubit[i]
            new_j = cell_to_qubit[j]
            # Maintain upper triangular: new_i <= new_j
            if new_i <= new_j:
                Q_pruned[(new_i, new_j)] = val
            else:
                Q_pruned[(new_j, new_i)] = Q_pruned.get((new_j, new_i), 0.0) + val

    return Q_pruned, cell_to_qubit, qubit_to_cell


def translate_results(
    results: List[Tuple[float, List[int], float]],
    qubit_to_cell: Dict[int, int],
) -> List[Tuple[float, List[int], float]]:
    """
    Convert QAOA results from qubit indices back to original cell IDs.

    Parameters
    ----------
    results       : list of (score, [qubit_ids], probability)
    qubit_to_cell : {qubit_index: original_cell_id}

    Returns
    -------
    list of (score, [original_cell_ids], probability)
    """
    translated = []
    for score, qubits, prob in results:
        cells = sorted([qubit_to_cell[q] for q in qubits])
        translated.append((score, cells, prob))
    return translated