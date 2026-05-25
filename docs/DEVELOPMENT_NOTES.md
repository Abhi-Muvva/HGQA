# HGQA Development Notes
## Detailed Step-by-Step Guide — Immediate, Short Term, Medium Term

*Written against codebase state: qaoa_builder.py (debugged), cell_pruner.py (stress-tested),
qubo_builder.py (validated), helpers.py, datasets.py, data_loader.py (complete).
Primary missing module: ga_solver.py.*

---

## PART 1 — IMMEDIATE BLOCKERS

These must be completed in order. Each step is a hard dependency for the next.

---

### STEP 1 — Run small_city through the pruner and get K

**What to do:**
In the notebook, switch `DATASET_FILE` to the small_city `.xlsx` and run cells 1–10
(data load → grid setup → QUBO build → pruning). Record the final K printed by the pruner.

**Why this is first:**
K determines everything downstream. The entire QAOA path branches on K ≤ 25.
If small_city gives K > 25, you need to adjust dataset parameters before continuing.
This is a 2-minute run that unlocks or blocks weeks of work.

**Expected output:**
```
CELL PRUNING
  ...
  RESULT: X → K cells (Y% reduction)
```
K should be ≤ 25 for statevector QAOA to be feasible.

**What would be wrong:**
- K > 25: Tier 3 is not aggressive enough on this dataset, or the dataset has too many
  competitive cells. Options: increase Tier 3 cluster keep ratio, or adjust small_city
  parameters (fewer POIs, more concentrated clusters) to produce a denser pruning.
- K = 0 or very small (< m): Pruner is too aggressive. Check Tier 2 reference score
  and lower bound gap.
- Pruner crashes: Check that Q_pruned and h5_params_pruned are being built from the
  correct post-prune cell list, not the full grid.

**Also check — the N=102 discrepancy:**
In your medium_city run, the pruner reported K=90 but QAOA received N=102. Before
running anything else, find where K is set in the notebook before the QAOA cell.
It is probably one of:
- `K = grid_params['N']` (full grid, not pruned)
- `K = len(surviving_cells)` (correct)
- `K` set from `h5_params_pruned['N']` if that dict wasn't updated post-prune

Fix this so K equals `len(qubit_to_cell)` exactly.

---

### STEP 2 — Build ga_solver.py

This is the primary missing module. Nothing can be benchmarked without it.

#### 2a. Chromosome Representation

Each individual is a binary vector of length K (post-prune qubit count), where a 1
at position i means qubit i (= cell `qubit_to_cell[i]`) is selected for a new charger.

```
chromosome = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, ...]  # length K, exactly m ones
```

Popcount constraint (exactly m ones) is maintained by the operators, never violated.
This is enforced structurally — the operators never produce invalid chromosomes.
There is no penalty term in the GA fitness function for constraint violation — the
constraint is baked into representation and operators.

Why binary vector not list-of-cell-IDs:
- Direct compatibility with QUBO evaluation (evaluate_solution takes cell IDs, easily
  converted)
- Crossover and mutation are simpler on fixed-length binary with swap operators
- Numpy operations on arrays are faster than set operations

#### 2b. Fitness Function

```python
def fitness(chromosome, Q_obj, qubit_to_cell):
    selected_cells = [qubit_to_cell[i] for i, bit in enumerate(chromosome) if bit == 1]
    return evaluate_solution(Q_obj, selected_cells)  # from qubo_builder.py
```

We are MINIMIZING. Lower fitness = better solution. Tournament selection and
ranking must all treat smaller values as better. Do not accidentally maximize.

Important: fitness uses Q_obj only (H1–H4, H6), not H5. The GA never needs H5
because the constraint is maintained by operator design. This is correct — H5
exists in the QUBO only for the quantum solver which cannot enforce constraints
structurally.

#### 2c. Crossover — Uniform Crossover with Popcount Repair

Standard uniform crossover on binary chromosomes does not preserve popcount.
You must use a constrained crossover that guarantees exactly m ones in the child.

**Algorithm — Uniform Crossover with Repair:**
```
parent_a = [1, 0, 1, 0, 0, 1, 0, 1, ...]
parent_b = [0, 1, 0, 1, 1, 0, 1, 0, ...]

Step 1: Find positions where both parents agree:
  Both 1 (always_on):  positions where a[i]=1 AND b[i]=1
  Both 0 (always_off): positions where a[i]=0 AND b[i]=0
  Disagreement:        positions where a[i] != b[i]

Step 2: Child inherits all always_on positions as 1.
  If len(always_on) > m: randomly drop excess (rare, means parents are very similar)

Step 3: From disagreement positions, randomly pick enough to fill up to m ones.
  needed = m - len(always_on)
  child gets 1 at 'needed' randomly chosen disagreement positions

Step 4: All remaining positions = 0
```

This is O(K) and produces exactly m ones. No repair loop needed.

#### 2d. Mutation — Swap Mutation

Standard bit-flip mutation breaks popcount. Use swap mutation:

```
Step 1: Randomly pick one position where chromosome[i] = 1  (a selected cell)
Step 2: Randomly pick one position where chromosome[j] = 0  (an unselected cell)
Step 3: Swap: chromosome[i] = 0, chromosome[j] = 1
```

This moves one charger from cell i to cell j. Popcount is unchanged.
Mutation rate should be applied per-individual, not per-bit. Typical value: 0.1–0.3
(i.e., 10–30% of individuals undergo one swap mutation per generation).

Spatially-aware mutation (swap to a neighbor of i) is a possible enhancement but
not required for the initial implementation. Log as future work if needed.

#### 2e. Selection — Tournament Selection

```
Step 1: Randomly pick tournament_size individuals from the population (with replacement)
Step 2: Return the one with the lowest fitness score (we minimize)
```

tournament_size = 3 is a good default. Larger tournaments = more selection pressure
(faster convergence, more exploitation). Smaller = more exploration.

Do not use roulette wheel selection — it requires positive fitness values and breaks
when scores are negative (which they will be, since Q_obj terms are negative rewards).

#### 2f. Population Initialization Interface

The GA must accept three initialization modes via a common interface:

```python
def run_ga(
    Q_obj,          # sparse QUBO dict
    qubit_to_cell,  # list mapping qubit index → original cell ID
    K,              # number of qubits (post-prune)
    m,              # number of chargers to place
    population_size=50,
    n_generations=200,
    crossover_rate=0.8,
    mutation_rate=0.2,
    tournament_size=3,
    seed_solutions=None,   # list of chromosomes from QAOA — if None, use random init
    seed_mode='qaoa',      # 'qaoa', 'random', or 'greedy'
    verbose=True,
    seed=None,
) -> Tuple[List[int], float, List[float]]:
    # Returns: (best_solution_cell_ids, best_score, convergence_history)
```

`seed_solutions`: list of binary vectors (chromosomes) from QAOA top-k output.
These are inserted directly into the initial population. Remaining population slots
filled with random chromosomes (for qaoa mode) or all random (for random mode).

`convergence_history`: list of best fitness per generation, length = n_generations.
This is essential for the comparison plots.

#### 2g. Greedy Population Generator

For the greedy baseline:
```
Step 1: Score every cell individually (solo score = evaluate_solution(Q_obj, [cell]))
Step 2: Pick the cell with best (lowest) solo score → add to solution
Step 3: Score every remaining cell given current partial solution
Step 4: Pick the best marginal addition → add to solution
Step 5: Repeat until m chargers placed
```

This produces one greedy solution. To make a greedy-seeded population of size P:
generate 1 greedy solution, then produce P-1 variants via random swap mutations
(same swap operator as GA mutation, applied k times for perturbation).

The greedy solution is a strong baseline — it's what a human expert would likely
do by hand. If QAOA-seeded GA cannot beat greedy-seeded GA, the QAOA seeding
provides no value.

#### 2h. Return Format

```python
return {
    'best_solution': [cell_id_1, cell_id_2, ...],   # m cell IDs
    'best_score': float,                              # QUBO score (lower = better)
    'convergence': [float, ...],                      # best score per generation
    'final_population': [[chromosome], ...],          # full final population
    'n_generations_run': int,
}
```

Return the full final population so you can inspect diversity at convergence.
Return convergence history so plotting requires no extra work.

---

### STEP 3 — Wire up the hybrid pipeline in the notebook

After ga_solver.py exists, add these cells to the notebook:

**Cell 12 — GA with QAOA seed:**
```python
# Convert QAOA results to chromosomes for GA seeding
seed_chromosomes = []
if qaoa_results_cells is not None:
    for score, cells, prob in qaoa_results_cells:
        chrom = [0] * K
        for cell_id in cells:
            qubit_idx = cell_to_qubit[cell_id]  # need inverse map
            chrom[qubit_idx] = 1
        seed_chromosomes.append(chrom)

ga_qaoa_result = run_ga(
    Q_pruned, qubit_to_cell, K, M,
    seed_solutions=seed_chromosomes,
    seed_mode='qaoa',
    n_generations=200,
    verbose=True,
)
```

**Cell 13 — GA with random seed:**
```python
ga_random_result = run_ga(
    Q_pruned, qubit_to_cell, K, M,
    seed_solutions=None,
    seed_mode='random',
    n_generations=200,
    verbose=True,
)
```

**Cell 14 — GA with greedy seed:**
```python
ga_greedy_result = run_ga(
    Q_pruned, qubit_to_cell, K, M,
    seed_solutions=None,
    seed_mode='greedy',
    n_generations=200,
    verbose=True,
)
```

**Cell 15 — Comparison plots:**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(ga_qaoa_result['convergence'],   label='QAOA-seeded GA',   color='blue')
plt.plot(ga_random_result['convergence'], label='Random-seeded GA', color='red')
plt.plot(ga_greedy_result['convergence'], label='Greedy-seeded GA', color='green')
plt.xlabel('Generation')
plt.ylabel('Best QUBO Score (lower = better)')
plt.title(f'GA Convergence Comparison — {dataset_name}, K={K}, m={M}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'convergence_{dataset_name}.png', dpi=150)
plt.show()

# Score comparison table
print(f"\nFinal Scores:")
print(f"  QAOA-seeded:   {ga_qaoa_result['best_score']:.4f}")
print(f"  Random-seeded: {ga_random_result['best_score']:.4f}")
print(f"  Greedy-seeded: {ga_greedy_result['best_score']:.4f}")
```

---

### STEP 4 — Run end-to-end on small_city and verify

Run the full pipeline once with verbose=True on small_city. Verify:

1. QAOA statevector completes (K ≤ 25, should take seconds)
2. QAOA returns at least m feasible solutions (popcount = m) with nonzero probability
3. GA with QAOA seed converges — best score decreases monotonically (if not, check
   fitness function sign — are you accidentally maximizing?)
4. All three seeds produce different convergence curves (if all three curves are
   identical, initialization is being ignored somewhere)
5. QAOA-seeded GA reaches a good score in fewer generations than random-seeded
   (this is the result you need — it does not need to be dramatically better,
   just consistently earlier convergence across 5 runs)

---

## PART 2 — SHORT TERM

These steps run after the pipeline is end-to-end working on small_city.

---

### STEP 5 — Comparative Evaluation on small_city (5 runs each)

**What to do:**
Run each of the three GA variants (QAOA-seeded, random-seeded, greedy-seeded) five
times on small_city with different random seeds. Record for each run:
- Final best QUBO score
- Generation at which best score was first reached (convergence speed)
- Full convergence curve

**Why 5 runs:**
GA has randomness from population initialization, crossover, mutation, and selection.
A single run can be lucky or unlucky. 5 runs gives you mean ± std for each method,
which is sufficient for a research paper comparison at this scale.

**What to report:**
- Table: mean ± std of final score across 5 runs for each method
- Plot: mean convergence curve ± shaded std band for each method
- Note: whether QAOA-seeded reaches within 5% of its final score faster than random

**What would indicate a problem:**
- QAOA-seeded performs worse than random: either the QAOA solutions are low quality
  (check their individual QUBO scores against brute-force optimum), or the GA is
  discarding the seed too quickly (too-high mutation rate, too-small initial weight
  of seeded individuals vs random).
- All three methods converge to the same score: the problem is too easy (K too small,
  m too small). The GA finds the optimum regardless of starting point. This is not
  a failure of the method — it means small_city is below the difficulty threshold
  where seeding matters. Document this and move to medium_city for the main result.
- Greedy-seeded dominates QAOA-seeded: greedy is a strong competitor on small problems.
  The QAOA advantage is expected to be larger as K grows (more local minima in the
  landscape). Document this scaling argument explicitly.

---

### STEP 6 — Tier 3 Pruner Validation Checklist

The pruner has been stress-tested on synthetic edge cases but not on the actual
medium/large/metro datasets where brute force is infeasible. Run this checklist
for each of the three larger datasets:

**Test 1 — Greedy score comparison:**
Build Q on the pruned problem. Run greedy to get a score S_pruned.
Build Q on the full problem. Run greedy to get a score S_full.
Assert S_pruned ≈ S_full (within 5–10%). A large gap means Tier 2 eliminated cells
that the greedy algorithm would have selected — the pruner is too aggressive.

**Test 2 — 10,000 random solution sampling:**
Sample 10,000 random valid solutions (popcount = m) from the full cell set.
Sample 10,000 random valid solutions from the pruned cell set.
Compare the distribution of QUBO scores. The pruned distribution should have
lower (better) mean and tighter spread — pruning removed bad options.
If the pruned distribution is worse, Tier 2 eliminated good cells.

**Test 3 — GA convergence comparison:**
Run the GA on the full problem (no pruning) and on the pruned problem for the
same number of generations. Compare final scores. The pruned GA should match
or exceed the full GA quality while running faster (fewer K qubits = smaller
chromosome = faster fitness evaluation).

**Test 4 — Neighbor coverage check:**
For each pruned-out cell, verify that at least one surviving cell within Chebyshev
distance 2 covers the same set of POIs. This is the guarantee that Tier 2's
"no optimal solution contains this cell" claim holds approximately in practice.

**Test 5 — keep_per_cluster sensitivity (Tier 3):**
Re-run Tier 3 with keep_per_cluster × 0.5 and keep_per_cluster × 2.0.
Check how K changes. A large sensitivity means the score similarity threshold
or cluster definition is too tight. Aim for: doubling keep_per_cluster changes K
by less than 20%.

---

### STEP 7 — Fix the N=102 vs K=90 Discrepancy

The medium_city run showed QAOA receiving N=102 when the pruner reported K=90.

**Where to look in the notebook:**
Find the line that sets K (or N) before the QAOA cell. It should be:
```python
K = len(qubit_to_cell)
```

Also check that `h5_params_pruned` contains `'N': K` (the post-prune K), not
`'N': grid_params['N']` (the full grid size). If `build_qubo_pruned()` or however
you construct the pruned QUBO copies `h5_params` from the full build, the N inside
it may not be updated. Trace this through.

The 12-qubit difference (102 - 90 = 12) is suspicious — check if it corresponds
to something specific (e.g., the number of ancilla qubits Qiskit adds, or the
number of existing charger cells that were excluded from the candidate set but
included in the QUBO somehow).

**Why it matters:**
12 phantom qubits add 12×101/2 ≈ 606 extra ZZ terms to H5 (when included),
inflate the Hilbert space by 2^12 = 4096×, and produce 12 qubits that QAOA
assigns probability but that can never be validly selected. This wastes circuit
depth and sampling budget.

---

## PART 3 — MEDIUM TERM

These steps produce the paper-quality results and algorithmic contributions.

---

### STEP 8 — Implement Dicke State + XY Mixer

This is the correct fix for the H5 problem at large K. It is also a genuine
algorithmic contribution that distinguishes your work from Chandra et al.

#### What is the Dicke State?

The Dicke state |D(N, m)⟩ is the uniform superposition over all N-bit strings
with exactly m ones:

```
|D(N, m)⟩ = 1/sqrt(C(N,m)) × Σ_{|x|=m} |x⟩
```

For N=4, m=2:
```
|D(4,2)⟩ = 1/sqrt(6) × (|0011⟩ + |0101⟩ + |0110⟩ + |1001⟩ + |1010⟩ + |1100⟩)
```

This is the correct initial state for a constrained QAOA where feasible solutions
are exactly the bitstrings with popcount = m. By starting here instead of |+⟩^N,
you confine the quantum state to the feasible subspace from the beginning.

#### What is the XY Mixer?

The standard QAOA mixer is:
```
B = Σ_i X_i   (apply Pauli X to every qubit)
```
This mixer connects ALL basis states to each other, including states with different
popcount. After one mixer layer, amplitude leaks from the m-ones subspace to
(m±1)-ones subspace. This is why H5 is needed as a penalty — to push probability
back to the feasible subspace.

The XY mixer replaces B with:
```
B_XY = Σ_{i<j} (X_i X_j + Y_i Y_j)
```
The XX + YY term is a partial SWAP. Its action on two-qubit states:
- |01⟩ → |10⟩ and |10⟩ → |01⟩  (swaps a 0 and a 1)
- |00⟩ → |00⟩ and |11⟩ → |11⟩  (unchanged)

This means the XY mixer only moves amplitude between states that differ in exactly
two positions (one 0→1 swap). It PRESERVES popcount by construction.
The feasible subspace is an invariant subspace of B_XY. No amplitude ever leaks out.
H5 becomes completely unnecessary.

#### Circuit Implementation

**Step 1 — Prepare Dicke state |D(N, m)⟩:**

The Dicke state preparation circuit is described in Bartschi & Eidenbenz (2019),
"Deterministic Preparation of Dicke States." The circuit uses O(mN) gates.

Simplified approach for small m (your case: m=5):
```python
from qiskit import QuantumCircuit
import math

def prepare_dicke_state(N: int, m: int) -> QuantumCircuit:
    """
    Prepare |D(N,m)⟩ using the Bartschi-Eidenbenz construction.
    Places m ones in the first m qubits, then applies a sequence
    of Givens rotations to spread amplitude uniformly.
    """
    qc = QuantumCircuit(N)
    # Start with |11...100...0⟩ (m ones in first m positions)
    for i in range(m):
        qc.x(i)
    # Apply SCS (Splitting-and-Combining-Scheme) gates
    # See Bartschi & Eidenbenz for the recursive construction
    _dicke_helper(qc, N, m, 0)
    return qc

def _ry_angle(n: int, k: int) -> float:
    """Rotation angle for moving k excitations among n qubits."""
    return 2 * math.acos(math.sqrt(k / n))
```

For the full recursive construction, refer to:
Bartschi, A. & Eidenbenz, S. (2019). "Deterministic Preparation of Dicke States."
https://arxiv.org/abs/1904.07358

Alternative — use the `qiskit_optimization` or search for existing Dicke state
preparation utilities in recent Qiskit extensions.

**Step 2 — Build XY mixer circuit for one QAOA layer:**

```python
def xy_mixer_layer(N: int, beta: float) -> QuantumCircuit:
    """
    Apply exp(-i beta B_XY) where B_XY = Σ_{i<j} (XX + YY).

    The XX + YY gate on qubits i,j decomposes as:
      CNOT(i,j) → RX(2β) on j → RZ(π/2) on i → CNOT(i,j)
      (one common decomposition — verify against reference)
    
    For efficiency, only apply to pairs within a cutoff distance
    (long-range XY terms contribute little to cardinality enforcement
    since the constraint is already structural).
    """
    qc = QuantumCircuit(N)
    for i in range(N):
        for j in range(i + 1, N):
            # Apply partial iSWAP rotation parameterized by beta
            qc.rxx(2 * beta, i, j)
            qc.ryy(2 * beta, i, j)
    return qc
```

Note: RXX and RYY are available as standard gates in Qiskit 2.x.
The full XY mixer with all pairs is O(N²) gates — same count as H5, BUT these
are in the mixer layer which runs once per QAOA layer, not baked into the cost
Hamiltonian which is evaluated every optimization step. The structure is different.

**Step 3 — Full Dicke+XY QAOA function:**

```python
def run_qaoa_dicke(
    Q_obj: Dict,      # H5 NOT included — structurally unnecessary
    N: int,
    m: int,
    p: int = 1,
    ...
) -> ...:
    # 1. Build cost Hamiltonian from Q_obj only (no H5)
    # 2. Prepare Dicke state circuit
    # 3. For each QAOA layer: apply cost layer (RZZ gates) + XY mixer layer
    # 4. Add measurements
    # 5. Optimize beta/gamma parameters
    # 6. All sampled bitstrings have popcount = m (no post-selection needed)
```

**What changes vs run_qaoa_mps:**
- Initial state: Dicke preparation circuit replaces H^⊗N (Hadamard on all qubits)
- Mixer: XY layer replaces standard X mixer
- H5: completely removed from cost Hamiltonian
- Feasible fraction: 100% by construction (no post-selection filtering)
- Circuit depth: cost layer is cheaper (no H5 ZZ gates), mixer layer is about same
  cost as H5 was (O(N²) gates), but the mixer only applies p times while H5
  contributed to every cost layer AND the mixer in the old formulation

**Expected improvement:**
- Feasible fraction: 0% → 100%
- Circuit gates: from ~15,000 (with H5 in cost) to ~N² per layer (mixer only)
  For N=90, p=1: ~8,100 mixer gates vs ~15,000 old. Modest improvement in gate count
  but enormous improvement in result quality (from 0 feasible shots to all feasible).

#### Validation

Before deploying for the main experiment, validate against statevector QAOA on
small_city (K ≤ 25):
1. Run standard `run_qaoa()` (statevector, includes H5) → get top-k solutions
2. Run `run_qaoa_dicke()` (statevector, Dicke+XY, no H5) → get top-k solutions
3. Compare: do both return similar solutions? Do both beat random sampling?
4. Compare feasible fraction: standard should be ~10–40%, Dicke should be 100%

This is the MPS agreement test adapted for the Dicke construction.

---

### STEP 9 — Scale Evaluation Across All Four Datasets

Once the pipeline runs end-to-end on small_city, run it on all four datasets
and report pruning + GA performance numbers.

**What to collect per dataset:**

| Metric | How to measure |
|---|---|
| Full grid size N | From grid_params |
| Surviving cells K after pruning | From pruner output |
| Pruning reduction % | (N-K)/N × 100 |
| Tier 1 / Tier 2 / Tier 3 contribution | From pruner verbose output |
| QAOA feasible fraction | From run_qaoa verbose output |
| GA final score (QAOA-seeded) | From ga_result['best_score'] |
| GA final score (greedy-seeded) | From ga_result['best_score'] |
| GA final score (random-seeded) | From ga_result['best_score'] |
| Convergence generation (QAOA) | First generation where score within 1% of final |
| Runtime per method | time.time() wrappers |

**For large/metro where brute force is infeasible:**
Use the Tier 3 validation checklist (Step 6) to verify pruner correctness.
Do not claim brute-force optimality — claim "best found by GA after N generations"
and compare the three seeding methods against each other.

**The scaling argument for the paper:**
Show a table with all four datasets. Show that K/N (pruning ratio) stays high
(good reduction) as N grows. Show that GA scores improve or stay comparable as
K shrinks (pruning preserves solution quality). This is the empirical validation
of the pruner's correctness claim.

---

### STEP 10 — MPS Agreement Test

This is a correctness check on the MPS approximation, not a new experiment.

**What to do:**
Find a pruned instance where K is between 15 and 25 — small enough for exact
statevector but large enough to be non-trivial for MPS.

If small_city gives K=18 for example:
1. Run `run_qaoa()` (exact statevector) → get exact probability distribution
2. Run `run_qaoa_mps()` with bond_dim=8, 16, 32 → get approximate distributions
3. Compare: do the top-5 solutions from MPS match the top-5 from statevector?
4. Compute TVD (total variation distance) between the two probability distributions
5. Plot: TVD vs bond_dim to show how approximation quality improves with bond_dim

**Why this matters for the paper:**
If you claim MPS-QAOA is usable as a proof-of-concept at large K, you need evidence
that MPS is a faithful approximation at small K. This test provides that evidence.
The expected result: at bond_dim=32, TVD is small (< 0.1), meaning MPS recovers
the correct high-probability states even with truncation.

**Note on Dicke+XY and MPS:**
The Dicke initialization + XY mixer may actually be MORE compatible with MPS than
standard QAOA + H5. The Dicke state has bounded entanglement by construction
(it's related to W-states which have low entanglement entropy). The XY mixer
preserves this low-entanglement structure better than all-to-all ZZ gates.
This is worth noting in the paper as a theoretical observation even if not
experimentally validated.

---

## APPENDIX — Key Parameters and Their Current Status

| Parameter | Current value | How to tune | Status |
|---|---|---|---|
| GA population_size | TBD | 50 for small, 100 for large | Not set |
| GA n_generations | TBD | 200 for comparison runs | Not set |
| GA crossover_rate | TBD | Start at 0.8 | Not set |
| GA mutation_rate | TBD | Start at 0.2 | Not set |
| GA tournament_size | TBD | 3 is standard | Not set |
| QAOA p (depth) | 1 for MPS, 3 for SV | Higher p = better quality, slower | Set |
| QAOA n_restarts | 1 for MPS, 3 for SV | More = better, costs linearly | Set |
| MPS bond_dim | 32 (old), needs revisit after Dicke | Higher = more accurate, slower | Provisional |
| Tier 3 score_threshold | 5% | Sensitivity test in Step 6 | Set |
| Tier 3 keep_per_cluster | Scales with m | Sensitivity test in Step 6 | Set |

---

## APPENDIX — Future Additions Log

| ID | Description | Priority | Dependency |
|---|---|---|---|
| FA-001 | Adaptive radii scaling with grid dimensions | Low | After scale evaluation |
| FA-002 | Automatic parameter suggester | Low | After comparative eval |
| FA-003 | Dicke state + XY mixer (H5-free QAOA) | HIGH | Step 8 — needed for large K |
| FA-004 | D-Wave integration as QUBO solver for large K production use | Medium | After paper |
| FA-005 | MPS agreement test (TVD vs bond_dim) | Medium | Step 10 |
| FA-006 | Boltzmann sampler / continuous relaxation baseline generators | Low | After GA exists |
| FA-007 | Noisy QAOA via fake IBM backends | Low | After Dicke+XY |
| FA-008 | Spatially-aware mutation (swap to neighbor cell) | Low | After GA baseline |

---

*Notes version: 1.0 — March 2026*
*Author: Research notes compiled from development sessions*
*Next review: After Step 2 (ga_solver.py) is complete*
