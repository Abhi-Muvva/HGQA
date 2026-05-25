"""
Genetic Algorithm Solver
=========================
Implements the GA component of the HGQA hybrid pipeline.

Design principles (from DEVELOPMENT_NOTES.md §2):
  - Binary chromosome of length K (post-prune qubit count). Bit i = 1 means
    qubit i (= qubit_to_cell[i]) is selected.
  - Exactly m ones guaranteed by operator design — H5 never evaluated.
  - Fitness: evaluate against Q_pruned using chromosome indices directly as keys.
  - Crossover: uniform crossover with popcount repair (O(K), always m ones).
  - Mutation: swap mutation (one 1→0, one 0→1). Popcount unchanged.
  - Selection: tournament selection (works with negative fitness values).
  - Three seed modes: 'qaoa' (QAOA-seeded), 'random', 'greedy'.

Key invariant
-------------
Q_obj passed to run_ga must be Q_pruned from remap_qubo_for_qaoa(), whose keys
are qubit indices 0..K-1. Chromosome bit positions are also 0..K-1, so the
fitness evaluation looks up Q_pruned directly — no cell-ID conversion needed
inside the hot loop. qubit_to_cell is only used at the end to return the final
solution as original cell IDs.

Usage
-----
    from src.ga_solver import run_ga, qaoa_results_to_chromosomes

    # Convert QAOA results (qubit-space) to chromosomes
    seed_chroms = qaoa_results_to_chromosomes(qaoa_results_pruned, K)

    result = run_ga(
        Q_pruned, qubit_to_cell, K=K, m=M,
        seed_solutions=seed_chroms,
        seed_mode='qaoa',
        n_generations=200,
        verbose=True,
    )
    best_cells  = result['best_solution']   # original cell IDs
    best_score  = result['best_score']      # QUBO score (lower = better)
    convergence = result['convergence']     # best score per generation
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def _evaluate(Q_pruned: Dict, chromosome: List[int]) -> float:
    """
    Evaluate QUBO objective for a chromosome against Q_pruned.

    Q_pruned keys are qubit indices 0..K-1, matching chromosome bit positions.
    H5 is never evaluated here — the GA encoding guarantees exactly m ones.
    """
    s_set = {i for i, b in enumerate(chromosome) if b == 1}
    total = 0.0
    for (i, j), val in Q_pruned.items():
        if i == j:
            if i in s_set:
                total += val
        else:
            if i in s_set and j in s_set:
                total += val
    return total


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _crossover(
    parent_a: List[int],
    parent_b: List[int],
    m: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Uniform crossover with popcount repair (DEVELOPMENT_NOTES §2c).

      1. Positions where both agree: always_on (both 1) / always_off (both 0)
      2. Child inherits always_on. If len > m, randomly drop excess.
      3. Fill remaining m slots from disagreement positions (randomly chosen).

    Always produces exactly m ones.
    """
    K = len(parent_a)
    always_on = [i for i in range(K) if parent_a[i] == 1 and parent_b[i] == 1]
    disagree  = [i for i in range(K) if parent_a[i] != parent_b[i]]

    child = [0] * K

    if len(always_on) > m:
        chosen = rng.choice(len(always_on), size=m, replace=False)
        for idx in chosen:
            child[always_on[idx]] = 1
        return child

    for pos in always_on:
        child[pos] = 1

    needed = m - len(always_on)
    if needed > 0 and disagree:
        selected_disagree = rng.choice(
            len(disagree), size=min(needed, len(disagree)), replace=False
        )
        for idx in selected_disagree:
            child[disagree[idx]] = 1

    # Edge case: still short (disagree exhausted)
    ones_placed = sum(child)
    if ones_placed < m:
        zeros = [i for i in range(K) if child[i] == 0]
        extra = rng.choice(len(zeros), size=m - ones_placed, replace=False)
        for idx in extra:
            child[zeros[idx]] = 1

    return child


def _mutate(chromosome: List[int], rng: np.random.Generator) -> List[int]:
    """
    Swap mutation (DEVELOPMENT_NOTES §2d).

    Pick one position with value 1, one with value 0, swap them.
    Popcount is unchanged. Applied per-individual (not per-bit).
    """
    ones  = [i for i, b in enumerate(chromosome) if b == 1]
    zeros = [i for i, b in enumerate(chromosome) if b == 0]
    if not ones or not zeros:
        return chromosome[:]
    mutant = chromosome[:]
    mutant[int(rng.choice(ones))]  = 0
    mutant[int(rng.choice(zeros))] = 1
    return mutant


def _tournament_select(
    population: List[List[int]],
    scores: "np.ndarray",
    tournament_size: int,
    rng: np.random.Generator,
) -> List[int]:
    """Tournament selection — lower score wins (minimising)."""
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    winner  = indices[int(np.argmin(scores[indices]))]
    return population[winner]


# ---------------------------------------------------------------------------
# Population initialisation helpers
# ---------------------------------------------------------------------------

def _random_chromosome(K: int, m: int, rng: np.random.Generator) -> List[int]:
    """Random binary chromosome with exactly m ones."""
    chrom = [0] * K
    for i in rng.choice(K, size=m, replace=False):
        chrom[i] = 1
    return chrom


def _greedy_chromosome(
    Q_pruned: Dict,
    K: int,
    m: int,
) -> List[int]:
    """
    Greedy chromosome (DEVELOPMENT_NOTES §2g).

    Uses Q_pruned directly (qubit indices 0..K-1 as keys).
    Sequentially adds the qubit that minimises the partial-solution score.
    """
    selected: List[int] = []
    remaining = list(range(K))

    for _ in range(m):
        best_q     = None
        best_delta = float("inf")

        for q in remaining:
            # Diagonal contribution of adding q
            delta = Q_pruned.get((q, q), 0.0)
            # Off-diagonal contributions with already-selected qubits
            for s in selected:
                i, j = (q, s) if q <= s else (s, q)
                delta += Q_pruned.get((i, j), 0.0)
            if delta < best_delta:
                best_delta = delta
                best_q = q

        if best_q is None:
            best_q = remaining[0]
        selected.append(best_q)
        remaining.remove(best_q)

    chrom = [0] * K
    for q in selected:
        chrom[q] = 1
    return chrom


def _perturb(chromosome: List[int], n_swaps: int, rng: np.random.Generator) -> List[int]:
    """Apply n_swaps random swap mutations."""
    result = chromosome[:]
    for _ in range(n_swaps):
        result = _mutate(result, rng)
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ga(
    Q_pruned: Dict,
    qubit_to_cell: Dict[int, int],
    K: int,
    m: int,
    population_size: int = 50,
    n_generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    tournament_size: int = 3,
    seed_solutions: Optional[List[List[int]]] = None,
    seed_mode: str = "qaoa",
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Dict:
    """
    Run Genetic Algorithm on the pruned QUBO.

    Parameters
    ----------
    Q_pruned       : sparse QUBO from remap_qubo_for_qaoa() — keys are qubit
                     indices 0..K-1, NOT original cell IDs.
    qubit_to_cell  : {qubit_index: original_cell_id} — used only for final output
    K              : number of qubits (post-prune) = chromosome length
    m              : number of chargers to place = number of ones per chromosome
    population_size: GA population size
    n_generations  : number of generations to evolve
    crossover_rate : fraction of offspring produced via crossover
    mutation_rate  : per-individual mutation probability
    tournament_size: tournament size for selection (default 3)
    seed_solutions : list of binary chromosomes (0..K-1 indexed, length K) to
                     seed the initial population.
                     — For 'qaoa' mode: use qaoa_results_to_chromosomes() output.
                     — Ignored for 'random' and 'greedy' modes.
    seed_mode      : 'qaoa'   — seed_solutions fill part of population, rest random
                     'random' — entirely random initialisation
                     'greedy' — one greedy solution + random swap perturbations
    verbose        : print per-generation progress
    seed           : random seed for reproducibility

    Returns
    -------
    dict:
        'best_solution'    : list of m original cell IDs (best found)
        'best_score'       : float — QUBO objective value (lower = better)
        'convergence'      : list[float] — best score per generation
        'final_population' : list of chromosomes at end
        'n_generations_run': int
        'runtime_sec'      : float
    """
    if K < m:
        raise ValueError(f"K={K} < m={m}: not enough qubits after pruning.")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------
    population: List[List[int]] = []

    if seed_mode == "qaoa" and seed_solutions:
        for chrom in seed_solutions:
            if len(chrom) == K and sum(chrom) == m:
                population.append(list(chrom))
                if len(population) >= population_size:
                    break
        while len(population) < population_size:
            population.append(_random_chromosome(K, m, rng))

    elif seed_mode == "greedy":
        greedy_chrom = _greedy_chromosome(Q_pruned, K, m)
        population.append(greedy_chrom)
        n_swaps = max(1, m // 2)
        while len(population) < population_size:
            population.append(_perturb(greedy_chrom, n_swaps, rng))

    else:  # 'random' or 'qaoa' with no valid seeds
        while len(population) < population_size:
            population.append(_random_chromosome(K, m, rng))

    if verbose:
        print(f"GA [{seed_mode}]: K={K}, m={m}, pop={population_size}, gens={n_generations}")
        if seed_solutions and seed_mode == "qaoa":
            n_seeded = min(len(seed_solutions), population_size)
            print(f"  Seeded {n_seeded} QAOA chromosome(s) into population")

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------
    best_chrom  = population[0][:]
    best_score  = _evaluate(Q_pruned, best_chrom)
    convergence: List[float] = []
    t0 = time.perf_counter()

    report_every = max(1, n_generations // 10)

    for gen in range(n_generations):
        scores = np.array([_evaluate(Q_pruned, c) for c in population], dtype=float)

        gen_best_idx = int(np.argmin(scores))
        if scores[gen_best_idx] < best_score:
            best_score = float(scores[gen_best_idx])
            best_chrom = population[gen_best_idx][:]

        convergence.append(best_score)

        if verbose and (gen == 0 or (gen + 1) % report_every == 0):
            print(f"  Gen {gen+1:>4}/{n_generations}: best={best_score:.6f}")

        # Next generation — always keep elite (best individual)
        elite      = population[gen_best_idx][:]
        next_pop   = [elite]

        while len(next_pop) < population_size:
            if rng.random() < crossover_rate:
                p_a   = _tournament_select(population, scores, tournament_size, rng)
                p_b   = _tournament_select(population, scores, tournament_size, rng)
                child = _crossover(p_a, p_b, m, rng)
            else:
                child = _tournament_select(population, scores, tournament_size, rng)[:]

            if rng.random() < mutation_rate:
                child = _mutate(child, rng)

            next_pop.append(child)

        population = next_pop

    runtime_sec = time.perf_counter() - t0
    best_cells  = sorted([qubit_to_cell[i] for i, b in enumerate(best_chrom) if b == 1])

    if verbose:
        print(f"  Done in {runtime_sec:.2f}s | best={best_score:.6f} | cells={best_cells}")

    return {
        "best_solution":     best_cells,
        "best_score":        best_score,
        "convergence":       convergence,
        "final_population":  population,
        "n_generations_run": n_generations,
        "runtime_sec":       runtime_sec,
    }


# ---------------------------------------------------------------------------
# Chromosome conversion helpers
# ---------------------------------------------------------------------------

def qaoa_results_to_chromosomes(
    qaoa_results: List[Tuple],
    K: int,
) -> List[List[int]]:
    """
    Convert QAOA results to binary chromosomes for GA seeding.

    QAOA results from run_qaoa / run_qaoa_mps carry qubit indices 0..K-1
    in the solution list (they operate on Q_pruned, not original cell IDs).
    This converts each solution directly to a binary chromosome.

    Parameters
    ----------
    qaoa_results : list of (score, [qubit_ids], probability)
                   Output of run_qaoa() or run_qaoa_mps() on Q_pruned.
    K            : number of qubits (chromosome length)

    Returns
    -------
    list of binary chromosomes — one per QAOA solution, length K each.
    """
    chromosomes = []
    for _score, qubit_ids, _prob in qaoa_results:
        chrom = [0] * K
        for q in qubit_ids:
            if 0 <= q < K:
                chrom[q] = 1
        chromosomes.append(chrom)
    return chromosomes
