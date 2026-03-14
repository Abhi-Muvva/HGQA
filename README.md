# Hybrid Quantum-Classical Genetic Algorithm for EV Charging Station Placement

<!-- ## Project Baseline Document -->

<!-- --- -->

## Project Overview

This project develops a hybrid quantum-classical algorithm that optimally places new electric vehicle (EV) charging stations in a city-like environment. The approach combines a **QUBO-based cell pruner**, the **Quantum Approximate Optimization Algorithm (QAOA)** via Qiskit, and a classical **Genetic Algorithm (GA)** to produce high-quality placement recommendations.

The core idea: after building a QUBO over the full grid, a cell pruner reduces the variable count by eliminating provably suboptimal cells, making the subsequent quantum optimization tractable. The quantum stage then explores the reduced combinatorial space to generate a strong set of candidate solutions, which seed the GA for final refinement. This hybrid approach is inspired by, and aims to improve upon, the methodology in the reference paper ([Chandra et al.](/References/TowardsanOptimalHybridAlgorithmforEVChargingStationsPlacementusingQuantumAnnealingandGeneticAlgorithms.pdf)).

The pipeline operates across three problem-size tiers, each using the most appropriate solver; from exact brute-force on tiny instances to MPS-based QAOA simulation for proof-of-concept at medium scale.

---

## Problem Definition

**Given**:
- A set of Points of Interest (POIs), each with a population density value (0 to 1)
- A set of existing EV charging stations
- A set of gas stations
- A number `n` of new charging stations to place

**Goal:**
Place `n` new EV charging stations such that:
- They are as close as possible to high-importance POIs (weighted by density)
- They optionally co-locate with gas stations (bonus)
- They avoid redundancy with existing chargers (penalty scales with cell importance)

**Output:**
A ranked shortlist of candidate grid locations (more than `n`, to give decision-makers flexibility).

---

## Grid-Based Discretization

### Why Grid-Based?
- Directly tied to qubit count (scalable with hardware)
- More realistic than pinpoint coordinates
- Output is a grid region, not an exact coordinate, which is more practical for planners

### Grid Construction
The entire geographic area is divided into a grid of **N** cells, where `N` equals the number of physical qubits available. Each qubit corresponds to exactly one grid cell — one binary variable in the QUBO. Each grid cell is assigned a unique ID from `0` to `N - 1`.

- For `N = 16` qubits → 16 grid cells (default: 4×4)
- For `N = 25` qubits → 25 grid cells (default: 5×5)
- For `N = 20` qubits → 20 grid cells (default: 4×5)

**Constraint — N must be composite (not prime):** A prime number of qubits can only be arranged as a 1×N strip, which is not a useful geographic grid. N must have at least two factors greater than 1 so it can be laid out as a proper rows×cols rectangle. Valid examples: 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25. Invalid examples: 2, 3, 5, 7, 11, 13, 17, 19, 23. The code enforces this at construction time and raises an error if a prime N is passed.

### Grid Layout Options
The grid layout is configurable:
- **Default:** Attempts the closest square grid (e.g., 4×4 for N=16, 5×5 for N=25). For non-square composites, picks the most square-like rectangle (e.g., 4×5 for N=20).
- **Show mode:** Displays all possible rectangular layouts (e.g., N=20 → 4×5, 2×10) and lets the user pick one

### Distance Metric
**Chebyshev distance** between grid cells:
- Same grid cell = 0
- Any adjacent cell (including diagonals) = 1
- Two cells apart = 2, and so on

This means 8-directional adjacency where moving diagonally costs the same as moving horizontally or vertically.


---

## Algorithm Pipeline

### Phase 1: Data Preparation
```
Raw Data (POIs with densities, chargers, gas stations)
        ↓
Grid Discretization (N cells where N = number of qubits, composite N required)
        ↓
Aggregate densities per cell → Normalize → Scale (linear with floor)
        ↓
Grid Data Table (summary per cell with continuous weights)
        ↓
Suggest QUBO Parameters (α weights, radii, λ — data-driven starting point)
```

### Phase 2: QUBO-Based Cell Pruning
```
Q_obj (full N-cell QUBO, H1-H4 + H6)
        |
        v
Tier 1: Remove zero-solo cells far from any competitive cell
        |
        v
Tier 2: Bound elimination -- prune cells provably not in any optimal solution
        |
        v
Tier 3: Spatial deduplication -- keep top-K per micro-cluster of near-identical cells
        |
        v
Surviving cell IDs (K << N)
        |
        v
Remap Q_obj keys to 0..K-1 for QAOA
```

<!-- See [QUBO-Based Cell Pruner](#qubo-based-cell-pruner) section for full details. -->

### Phase 3: Quantum Optimization (QAOA via Qiskit) — Three Execution Tiers

The quantum stage adapts to problem size. After pruning, K is the number of surviving cells (qubits fed to QAOA):



```
Pruned Q_obj (K variables)  +  H5 params (applied separately)
        |
        v
Build cost Hamiltonian (Ising form via QUBO-to-Ising conversion)
        |
        v
Run QAOA (statevector or MPS, COBYLA optimization, multiple restarts)
        |
        v
Collect top-k feasible solutions (popcount = m)
        |
        v
Translate qubit indices back to original cell IDs
        |
        v
Candidate Grid ID Sets (initial population for GA)
```


### Phase 4: Classical Genetic Algorithm Refinement
```
Initial Population (from QAOA / brute-force output, top-k solutions)
        |
        v
Evaluate Fitness using Q_obj (evaluate_solution -- same QUBO, no H5 needed)
        |
        v
Selection (tournament) -> Crossover (subset swap) -> Mutation (neighbor shift)
        |
        v
Repeat for G generations
        |
        v
Final Ranked Shortlist of original Grid IDs
```

### Phase 5: Evaluation and Visualization
```
Compare: Only GA (random seed) vs Hybrid (QAOA/MPS + GA)
        |
        v
Scoring metrics, convergence plots, grid visualizations
        |
        v
MPS agreement check: statevector vs MPS top-3 overlap (small instances only)
```


---

## Data Model

### Input Data
Three categories of data points, each placed on the grid:

| Data Type | Attributes | Notes |
|-----------|-----------|-------|
| **Points of Interest (POIs)** | Location (x, y), Population Density (0 to 1) | 1 = highest density, 0 = lowest density |
| **Existing Charging Stations** | Location (x, y) | Already operational EV chargers |
| **Gas Stations** | Location (x, y) | Potential co-location sites for new chargers |

### Continuous Density-Based Cell Weighting (Why Not Tiers)

This project uses a **continuous weighting system** instead of discrete tiers (Tier 1/2/3). The reasoning:

A discrete tier system assigns fixed labels (e.g., Tier 1, 2, 3) to POIs or grid cells. This creates several problems in a grid-based approach:

1. **Grid resolution dependency:** If tiers are assigned to individual POIs before gridding, the labels become meaningless after gridding. Changing the qubit count changes the grid resolution, which reshuffles which POIs land in which cells. A "Tier 1" POI alone in a large cell may matter less than three "Tier 3" POIs clustered in a small cell — but the tier labels don't capture this.

2. **Arbitrary boundary effects: (Initial plan: Tier-1: 5points, Tier-2: 3points, Tier-3: 1point)** With tiers, a cell at the 30th percentile gets weight 5, while a cell at the 31st percentile drops to weight 3. This cliff-edge creates discontinuities in the fitness landscape that can mislead the optimization — two nearly identical cells get treated very differently.

3. **Loss of granularity:** Three discrete weights (5, 3, 1) throw away the rich information contained in the actual density distribution. Two cells both labeled "Tier 2" could have very different real-world importance.

The continuous system solves all three problems by deriving cell weights from the data after gridding, producing a smooth gradient of importance with no arbitrary cutoffs.

### Cell Weight Calculation

**Step 1 — Aggregate:** After gridding, sum the population densities of all POIs in each cell.

```
Cell 47: POIs with densities 0.9, 0.7, 0.4  →  raw_score = 2.0
Cell 12: POIs with density 0.3              →  raw_score = 0.3
Cell 88: no POIs                            →  raw_score = 0.0
```

**Step 2 — Normalize:** Divide all cell scores by the maximum cell score across the grid, producing values in [0, 1].

```
If max raw_score = 2.5:
  Cell 47: normalized = 2.0 / 2.5 = 0.80
  Cell 12: normalized = 0.3 / 2.5 = 0.12
  Cell 88: normalized = 0.0 / 2.5 = 0.00
```

**Step 3 — Scale (Linear with Floor):** Apply linear scaling with a minimum weight floor to amplify differentiation at the top end while keeping low-density cells still relevant.

```
weight = max(normalized × scale_factor, min_weight)    [if cell has POIs]
weight = 0                                              [if cell has no POIs]
```

**Tunable Parameters:**
- `scale_factor` (default: 5) — controls how much the top end stretches
- `min_weight` (default: 0.5) — floor for the lowest-density cells that still have POIs

**Example with scale_factor = 5, min_weight = 0.5:**

```
Cell 47 (normalized 0.80):  weight = max(0.80 × 5, 0.5) = 4.0   ← high priority
Cell 23 (normalized 0.50):  weight = max(0.50 × 5, 0.5) = 2.5   ← medium priority
Cell 12 (normalized 0.12):  weight = max(0.12 × 5, 0.5) = 0.6   ← low but still relevant
Cell 05 (normalized 0.04):  weight = max(0.04 × 5, 0.5) = 0.5   ← floor, minimal but not zero
Cell 88 (no POIs):          weight = 0.0                          ← completely ignored
```

Key properties of this design:
- High-density cells get meaningfully large weights (up to `scale_factor`)
- Low-density cells with POIs are still considered (never fall below `min_weight`)
- Empty cells are completely ignored as attraction sources (weight = 0), but can still be selected as charger locations if they are near dense cells
- No arbitrary cutoff boundaries — the gradient is smooth
- Automatically adapts when grid resolution changes (different qubit count)

### Grid Data Table
After gridding and weight calculation, each cell contains:


| Field | Description |
|-------|-------------|
| Grid ID | Unique identifier (0 to N − 1) |
| Number of POIs | Count of POIs in this cell |
| Raw Density Score | Sum of population densities of POIs in this cell |
| Normalized Score | Raw score / max raw score across all cells |
| Cell Weight | Final weight after linear scaling with floor |
| Gas Station Count | Number of gas stations in this cell (0, 1, 2, ...). Used directly as `g_i` in H2 — more gas stations = stronger co-location bonus |
| Existing Charger Count | Number of existing chargers in this cell (0, 1, 2, ...). Used in H3 and H1's service gap factor `s_c` — more existing chargers = stronger repulsion and reduced POI attraction, reflecting saturation |


---

## Fitness Function — Unified QUBO Formulation

### Why QUBO Format

The fitness function is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem. This is a deliberate design choice that provides a critical advantage: the same mathematical formulation drives both the quantum and classical components of the hybrid algorithm.

### Binary Variables

We define `N` binary variables, one for each grid cell, where `N` is the number of physical qubits:

```
x_i = 1  if a new charger is placed in grid cell i
x_i = 0  otherwise
```

The solution vector `x` has exactly `m` entries equal to 1, where `m` is the number of new chargers to place.

### Notation Reference

Before defining the objective terms, here is the notation used throughout:


| Symbol | Meaning |
|--------|---------|
| `N` | Total number of grid cells (= number of physical qubits, must be composite) |
| `m` | Number of new chargers to place |
| `x_i` | Binary variable: 1 if charger placed in cell i, 0 otherwise |
| `w_c` | Cell weight of cell c (from Section 4.3, continuous, 0 to scale_factor) |
| `nw_c` | Normalized cell weight of cell c (= w_c / scale_factor, range [0, 1]) |
| `d(i, j)` | Chebyshev distance between cells i and j |
| `C_POI` | Set of cells that contain at least one POI |
| `E_all` | Set of all individual existing chargers (not cells — a cell with 3 chargers contributes 3 entries) |
| `g_i` | Number of gas stations in cell i (0, 1, 2, ...) |
| `s_c` | Service gap factor for POI cell c (how underserved it is by existing chargers) |
| `R₁` | H1 attraction radius — only POIs within R₁ of cell i contribute (default: 30% of grid side, min 2) |
| `Rₛ` | Service gap radius — only existing chargers within Rₛ of POI cell c contribute to s_c (default: = R₁) |
| `R₃` | H3 penalty radius — only existing chargers within R₃ of cell i contribute (default: 20% of grid side − 1, min 2) |
| `R₄` | H4 spacing radius — only pairs with d(i,j) ≤ R₄ get spacing penalty (default: 20% of grid side, min 2) |
| `R₆` | H6 redundancy radius — only POIs within R₆ of BOTH chargers contribute (default: 15% of grid side, min 2) |

---

### Objective Terms Overview

The QUBO objective consists of six terms, each handling a specific placement requirement. The full objective to **minimize**:

$$H_{final} = \alpha_1 H_1 + \alpha_2 H_2 + \alpha_3 H_3 + \alpha_4 H_4 + \alpha_5 H_5 + \alpha_6 H_6$$

| Term | Name | Type | Role |
|------|------|------|------|
| **H1** | POI Attraction | Reward (−) | Pull chargers toward dense, underserved POI areas |
| **H2** | Gas Station Bonus | Reward (−) | Prefer cells with gas stations for co-location |
| **H3** | Existing Charger Penalty | Penalty (+) | Push away from existing chargers, scaled by density: low-density areas get strong repulsion, high-density areas tolerate clustering |
| **H4** | New Charger Spacing | Penalty (+) | Spread new chargers apart in low-density areas, allow clustering in high-density areas |
| **H5** | Charger Count Constraint | Constraint | Force exactly `m` new chargers — hard constraint via large penalty |
| **H6** | Coverage Redundancy | Penalty (+) | Don't waste two chargers serving the same POI cluster |

All terms use **local distance radii** to enforce Q matrix sparsity. Each term only considers entities within its radius, making Q grid-local sparse rather than dense. Recommended ordering: R₁ ≥ R₄ > R₃ ≥ R₆.

<!-- **QUBO structure:** H1, H2, H3 are linear in `x_i` → Q matrix diagonal. H4, H5, H6 involve pairs `x_i × x_j` → Q matrix off-diagonal. H5 contributes to both. -->

*Detailed derivation of each term follows below.*


### Objective Term H1 — POI Attraction (Minimize → Place Chargers Near Underserved Dense Areas)

$$H_1 = -\sum_{i} x_i \sum_{\substack{c \in C_{POI} \\ d(i,c) \leq R_1}} \frac{w_c \cdot s_c}{1 + d(i, c)}$$


where the service gap factor `s_c` is a precomputed constant:

$$s_c = \frac{1}{1+ \sum_{\substack{e \in E_{all} \\ d(c,e) \leq R_s}} \frac{1}{1+d(c, e)}}$$


**What it does:** For each candidate cell `i`, it computes how attractive that cell is based on the weighted proximity to nearby POI cells (within radius R₁), discounted by how well those POIs are already served by existing chargers (within radius Rₛ). Cells closer to high-weight, underserved POI clusters get a more negative (better) score.

**Why designed this way:**

- The **negative sign** means minimizing H1 rewards placing chargers where the attraction is highest.

- The weight `w_c` comes from the **POI cell**, not the candidate cell. This is critical: an empty cell (weight = 0) adjacent to a dense cell (weight = 4.0) should still be attractive for charger placement. If we used the candidate cell's own weight, empty cells would never attract chargers, even when they're right next to areas that need them.

- The **service gap factor `s_c`** reduces attraction to POIs that are already well-served by existing chargers. It sums over all individual existing chargers (not just cells), so a cell with multiple existing chargers is recognized as more saturated. Examples:
  - POI cell `c` with 1 existing charger at distance 0: `s_c = 1/(1+1) = 0.5` — attraction halved
  - POI cell `c` with 3 existing chargers at distance 0: `s_c = 1/(1+3) = 0.25` — attraction drops to 25%
  - POI cell `c` with no existing chargers nearby: `s_c ≈ 1.0` — full attraction
  
  This prevents the algorithm from piling new chargers onto areas that already have good coverage, and naturally handles saturation. Importantly, `s_c` is precomputed from the data (it doesn't depend on `x`), so H1 remains linear in `x_i` and QUBO-compatible.

- The **`1 / (1 + d(i,c))`** decay function ensures closer POIs contribute more. At distance 0 (same cell), contribution is `w_c × s_c`. At distance 1 (adjacent), it's `w_c × s_c / 2`. At distance 5, it's `w_c × s_c / 6`. The decay is gradual — distant POIs still matter, just less.

- **Distance cutoff R₁:** Only POI cells within R₁ of cell `i` contribute. This enforces sparsity — at R₁ = 6 on a 16×16 grid, each cell considers ~168 neighbors instead of all 255, and the cutoff only discards contributions that are already less than 14% of same-cell strength. Rₛ applies the same logic to the `s_c` precomputation for consistency. The cutoff also has a desirable side effect: it makes H1 prefer locally central positions rather than globally central ones, which is more realistic for charger placement.

- **Summing over nearby POI cells** means the algorithm naturally prefers locations that are centrally positioned relative to local POI clusters.

**Scenario tests:**

| Scenario | Behavior |
|----------|----------|
| Cell right on high-weight POI (w=4.0), no existing charger (s≈1.0) | Strong attraction: -4.0 | 
| Cell right on high-weight POI (w=4.0), 1 existing charger at d=0 (s=0.5) | Reduced attraction: -2.0 | 
| Cell right on high-weight POI (w=4.0), 3 existing chargers at d=0 (s=0.25) | Heavily reduced: -1.0 | 
| Empty cell adjacent to dense POI (d=1) | Attracted: -w×s/2 | 
| Cell far from all POIs (d>R₁) | Zero — outside cutoff | 

**QUBO placement:** Linear in `x_i` → **diagonal** of Q matrix:

> $$Q_{ii} \mathrel{+}= \alpha_1 \left( -\sum_{\substack{c \in C_{POI} \\ d(i,c) \leq R_1}} \frac{w_c \cdot s_c}{1 + d(i, c)} \right)$$

---

### Objective Term H2 — Gas Station Co-location Bonus (Minimize → Prefer Gas Station Cells)

$$H_2 = - \sum_{i} x_i \beta g_i$$



**What it does:** Gives a bonus (negative contribution = better score) when a charger is placed in a cell that has gas stations. More gas stations in a cell = stronger bonus.

**Why designed this way:**

- The **negative sign** rewards gas station co-location.

- `g_i` is the **count** of gas stations in cell i (0, 1, 2, ...). A cell with 3 gas stations gets 3× the bonus of a cell with 1, because each gas station represents a separate co-location opportunity with its own land and infrastructure.
- `β` is the bonus weight: a tunable parameter that controls how important gas station co-location is relative to other objectives. If `β` is too large, the algorithm would force chargers onto gas stations even when better POI-serving locations exist. If too small, the bonus becomes negligible. This is deliberately a soft bonus, not a hard constraint.
- This reflects the real-world advantage: gas stations already have land, power infrastructure, and customer traffic patterns that make them natural candidates for EV charger installation.

**Scenario tests:**

| Scenario | Behavior |
|----------|----------|
| Cell has 1 gas station | Bonus: -β |
| Cell has 3 gas stations | Stronger bonus: -3β | 
| Cell has no gas station | No effect: 0 | 

**QUBO placement:** Linear in `x_i` → **diagonal**:

> $$Q_{ii} \mathrel{+}= \alpha_2 \left( -\beta \cdot g_i \right)$$

---

### Objective Term H3 — Existing Charger Penalty (Minimize → Avoid Redundancy in Low-Density Areas)

$$H_3 = + \sum_{i} x_i \gamma (1-nw_i) \sum_{\substack{e \in E_{all} \\ d(i,e) \leq R_3}} \frac{1}{1+d(i, e)}$$



**What it does:** Penalizes placing a new charger near an existing charger (within radius R₃), BUT the penalty is scaled by how low-density the candidate cell is. Sums over all individual existing chargers, so a cell with multiple existing chargers creates proportionally stronger repulsion.

**Why designed this way:**

- The **positive sign** means this adds to the cost — it's a penalty.

- **`(1 - nw_i)` is the key design element.** This uses the normalized weight of the candidate cell `i` (where the new charger would go):
  - High-density cell (nw_i = 0.8): penalty factor = 0.2 → almost no penalty. Rationale: busy areas can support multiple nearby chargers because charging takes time and queues form.
  - Medium-density cell (nw_i = 0.5): penalty factor = 0.5 → moderate penalty.
  - Low-density cell (nw_i = 0.1): penalty factor = 0.9 → strong penalty. Rationale: in sparse areas, spreading coverage is more valuable than clustering.
  - Empty cell (nw_i = 0.0): penalty factor = 1.0 → maximum penalty. No reason to cluster near existing chargers if there's no demand.

- The **sum over `E_all`** (individual chargers, not cells) means a cell with 3 existing chargers at d=0 contributes `3 × 1/(1+0) = 3` to the sum, versus just `1` for a cell with 1 charger. This naturally handles saturation.

- The **`1 / (1 + d(i,e))`** decay means the penalty is strongest when directly on top of an existing charger and fades with distance.

- `γ` is the **penalty factor** — a tunable parameter controlling overall penalty strength.

**Relationship with H1's service gap factor:** H1 (via `s_c`) reduces attraction to already-served POIs. H3 directly repels new chargers from existing charger locations. These are complementary, not redundant: a cell could be near an underserved POI (high `s_c` → H1 attracts) yet still close to an existing charger serving a different area (H3 repels). Both perspectives are needed.

**Scenario tests:**

| Scenario | Behavior |
|----------|----------|
| High-density cell (nw=0.9), 1 existing charger at d=0 | Tiny penalty: 0.1γ |
| High-density cell (nw=0.9), 3 existing chargers at d=0 | Larger penalty: 0.3γ (saturation pushes to adjacent) |
| Empty cell (nw=0), 1 existing charger at d=0 | Max penalty: γ |
| Any cell far from existing chargers (d>R₃) | Zero — outside cutoff |
| Medium cell (nw=0.5) at d=2 from 1 existing charger | Moderate: 0.167γ |

**QUBO placement:** Linear in `x_i` → **diagonal**:

> $$Q_{ii} \mathrel{+}= \alpha_3 \left( \gamma \cdot (1 - nw_i) \sum_{\substack{e \in E_{all} \\ d(i,e) \leq R_3}} \frac{1}{1 + d(i, e)} \right)$$

---

### Objective Term H4 — New Charger Spacing (Minimize → Spread New Chargers in Low-Density Areas)

$$H_4 = + \sum_{i}\sum_{\substack{j>i \\ d(i,j) \leq R_4}} x_i x_j \delta \frac{1- \max(nw_i, nw_j)}{1 + d(i, j)}$$


**What it does:** Penalizes placing two NEW chargers close to each other (within radius R₄), but only when neither cell is high-density.

**Why designed this way:**

- This is a **quadratic term** (involves pairs `x_i × x_j`), capturing the interaction between two placement decisions that linear terms cannot express.

- **`max(nw_i, nw_j)` instead of average:** If EITHER cell in the pair is high-density, the penalty drops i.e., if two chargers are close but one is in a high-demand area, that clustering is justified. Using average would penalize a high-density cell paired with a low-density neighbor, which doesn't make sense. Using max means: "if there's strong demand at either location, allow the clustering."

- The **`1 / (1 + d(i,j))`** decay means only nearby pairs get penalized. Two chargers on opposite ends of the grid don't interact.

- `δ` is the **spacing penalty factor** — controls how strongly we enforce spread.

- **Summing over `j > i`** (not `j ≠ i`) avoids double-counting each pair.

**Scenario tests:**

| Scenario | Behavior |
|----------|----------|
| Two chargers in adjacent high-density cells (nw=0.9, d=1) | Tiny: 0.05δ |
| Two chargers in adjacent empty cells (nw=0, d=1) | Strong: 0.5δ | 
| Two chargers far apart (d>R₄) | Zero — outside cutoff | 
| One high-density (0.9), one empty (0.0), adjacent | Low: 0.05δ (high density justifies) |


**QUBO placement:** Quadratic → **off-diagonal** (upper triangular, i < j):

> $$Q_{ij} \mathrel{+}= \alpha_4 \left( \delta \cdot \frac{1 - \max(nw_i, nw_j)}{1 + d(i, j)} \right) \qquad \text{for } d(i,j) \leq R_4$$

---

### Constraint Term H5 — Exact Number of Chargers

$$H_5 = \lambda (\sum_i x_i - m)^2$$

**What it does:** Forces the solution to place exactly `m` new chargers.

**Why designed this way:**

- This is a **hard constraint disguised as a penalty.** By making `λ` sufficiently large (much larger than the other terms), any solution that doesn't have exactly `m` chargers becomes so expensive that the optimizer avoids it.


**Expanding the square** (with careful handling of binary variables where $x_i^2 = x_i$):

$$\left(\sum_i x_i - m\right)^2 = \left(\sum_i x_i\right)^2 - 2m\left(\sum_i x_i\right) + m^2$$

$$\left(\sum_i x_i\right)^2 = \sum_i x_i^2 + \sum_{i \ne j} x_i x_j = \sum_i x_i + 2\sum_{i < j} x_i x_j$$

Therefore:

$$H_5 = \lambda \left(\sum_i x_i + 2\sum_{i < j} x_i x_j - 2m\sum_i x_i + m^2\right)$$

$$= \lambda \sum_i (1-2m) x_i + 2\lambda \sum_{i < j} x_i x_j + \lambda m^2$$


Which gives:
  - Diagonal: `λ(1 - 2m)` per cell (negative for m ≥ 1). It rewards selecting cells individually
  - Off-diagonal: `2λ` per pair. This penalizes every pair of selected cells, counterbalancing the diagonal reward
  - Constant: `λm²`. Doesn't affect optimization, can be ignored

**Important: The off-diagonal coefficient is `2λ`, not `λ`.** This comes from the fact that `(Σ x_i)²` produces `Σ_{i≠j} x_i x_j`, which when converted to upper triangular form (i < j only) doubles the coefficient. Getting this wrong would make the constraint only half as strong on pairwise terms, potentially allowing the optimizer to select too many or too few chargers.

- The **`λ` value** must be chosen carefully. Too small: the optimizer violates the constraint. Too large: it dominates the objective and the optimizer ignores placement quality. A common heuristic is to set `λ` to be ~2-5× the magnitude of the largest other term.

**H5 storage strategy — not stored in Q:**

H5's `2λ` off-diagonal term applies to every pair, which would destroy the sparsity gained from local radii. Instead:

- **GA:** Skips H5 entirely. GA individuals always have exactly `m` chargers by encoding, so H5 = 0 always. The GA evaluates fitness using only the sparse objective terms (H1–H4, H6).
- **QAOA (standard mixer):** H5 is applied as a separate Hamiltonian term during cost Hamiltonian construction, not stored in Q.
- **QAOA (XY-mixer, future):** A Hamming-weight preserving mixer restricts the search to states with exactly `m` ones, eliminating H5 entirely.

**Verification:**

| Chargers selected | H5 value | Penalty |
|-------------------|----------|---------|
| Exactly m | λ × 0 = 0 | ✓ No penalty |
| m + 1 | λ × 1 = λ | ✓ Penalty |
| m - 1 | λ × 1 = λ | ✓ Penalty |
| m + 2 | λ × 4 = 4λ | ✓ Quadratically increasing |


**QUBO placement:** Both **diagonal and off-diagonal** (applied separately, not stored in sparse Q):

> $$Q_{ii} \mathrel{+}= \alpha_5 \cdot \lambda \cdot (1 - 2m)$$

> $$Q_{ij} \mathrel{+}= \alpha_5 \cdot 2\lambda \qquad \text{for } i < j$$

---

### Objective Term H6 — Coverage Redundancy (Minimize → Don't Waste Chargers on Same Cluster)

$$H_6 = + \sum_i \sum_{j>i} x_i x_j \varepsilon \sum_{\substack{c \in C_{POI} \\ d(i,c) \leq R_6 \\ d(j,c) \leq R_6}} \frac {w_c}{(1+d(i,c))(1+d(j,c))}$$

**What it does:** Penalizes placing two new chargers such that they both serve the same POI cluster, wasting coverage.

**Why this term is needed (and why H4 alone is not enough):**

H4 penalizes chargers being physically close to each other. H6 penalizes chargers serving the same POIs — these are related but different situations:

| Situation | H4 catches it? | H6 catches it? |
|-----------|----------------|----------------|
| Two chargers adjacent, both near same POI | ✓ (but weakly if high density) | ✓ |
| Two chargers far apart, but both equidistant to same POI cluster (e.g., on opposite sides) | ✗ (they're far from each other) | ✓ |
| Two chargers adjacent, serving different POI clusters | ✓ (penalty, but negligible if both cells are high-density) | ✗ (correctly no penalty) |

H6 catches the second case that H4 misses entirely, and avoids the false positive in the third case.

**Why designed this way:**

- The **product `1/((1+d(i,c)) × (1+d(j,c)))`** is large only when BOTH chargers are close to the same POI cell `c`. If one is close and the other is far, the product is small. This precisely captures "both serving the same area."

- **`w_c` weighting** means redundancy near high-weight POIs is penalized more (a bigger waste of resources than redundancy near low-weight POIs).

- **Distance cutoff R₆:** The sum only includes POI cells where BOTH chargers are within R₆ cells. Without this cutoff, every pair of chargers would accumulate small penalty contributions from all POI cells across the entire grid, creating background noise that acts as a general "don't place any two chargers" penalty — which is not the intent. The cutoff keeps H6 focused on actual service redundancy. Default R₆ = 15% of grid side (min 2); tunable.

- `ε` is the **redundancy penalty factor** — controls how strongly we penalize coverage overlap.

**Scenario tests:**

| Scenario | Behavior | Correct? |
|----------|----------|----------|
| Charger i at POI cell c (d=0), charger j adjacent to c (d=1) | Penalty: ε × w_c / (1×2) = ε×w_c/2 | ✓ High penalty |
| Charger i near cluster A (d=1), charger j near different cluster B (d=1), clusters far apart | Cluster A terms: small (j far from A). Cluster B terms: small (i far from B). Total: ~0 | ✓ No penalty |
| Both chargers at d=3 from same POI, R₆=3 | ε × w_c / (4×4) = ε×w_c/16 | ✓ Moderate penalty |
| Both chargers at d=3 from same POI, R₆=2 | Not included (d > R₆) → 0 | ✓ Cutoff works |

**QUBO placement:** Quadratic → **off-diagonal** (upper triangular, i < j):

> $$Q_{ij} \mathrel{+}= \alpha_6 \left( \varepsilon \sum_{\substack{c \in C_{POI} \\ d(i,c) \leq R_6 \\ d(j,c) \leq R_6}} \frac {w_c}{(1+d(i,c))(1+d(j,c))} \right)$$

---

### Complete QUBO Formulation

The full objective to **minimize**:

$$H_{final} = \alpha_1 H_1 + \alpha_2 H_2 + \alpha_3 H_3 + \alpha_4 H_4 + \alpha_5 H_5 + \alpha_6 H_6$$

**Assembled Q matrix (upper triangular form):**

**DIAGONAL** ($Q_{ii}$) — encodes single-cell properties:

> $$Q_{ii} = \alpha_1 \left( -\sum_{\substack{c \in C_{POI} \\ d(i,c) \leq R_1}} \frac{w_c \cdot s_c}{1 + d(i,c)} \right) + \alpha_2 \left( -\beta \cdot g_i \right) + \alpha_3 \left( \gamma (1-nw_i) \sum_{\substack{e \in E_{all} \\ d(i,e) \leq R_3}} \frac{1}{1+d(i,e)} \right) + \alpha_5 \cdot \lambda (1 - 2m)$$

where $s_c = \frac{1}{1 + \sum_{\substack{e \in E_{all} \\ d(c,e) \leq R_s}} \frac{1}{1+d(c,e)}}$ is a precomputed constant.

**OFF-DIAGONAL** ($Q_{ij}$, $i < j$) — encodes pairwise interactions:

> $$Q_{ij} = \alpha_4 \left( \delta \cdot \frac{1 - \max(nw_i, nw_j)}{1 + d(i,j)} \right) + \alpha_6 \left( \varepsilon \sum_{\substack{c \in C_{POI} \\ d(i,c) \leq R_6 \\ d(j,c) \leq R_6}} \frac{w_c}{(1+d(i,c))(1+d(j,c))} \right) \qquad \text{for } d(i,j) \leq \max(R_4, 2R_6)$$

Note: H5 (`α₅ · 2λ`) is **not stored** in the sparse Q matrix. For QAOA it is applied as a separate Hamiltonian term. For the GA it is skipped entirely (always evaluates to 0).


### Verification

Every design requirement and identified edge case is handled by at least one term in the formulation:

| Requirement | Primary Term | Supporting Term |
|-------------|-------------|-----------------|
| Place chargers near dense POIs | H1 (attraction) | — |
| Don't over-serve already-covered POIs | H1 (service gap s_c) | H6 (redundancy) |
| Prefer gas station cells (soft bonus, scales with count) | H2 (bonus) | — |
| Avoid existing chargers in low-density areas | H3 (penalty) | H1 (reduced attraction via s_c) |
| Allow charger clustering in high-density areas | H3 (low penalty when nw high) | H4 (low penalty when nw high) |
| Spread chargers in low-density areas | H3 (high penalty when nw low) | H4 (high penalty when nw low) |
| Handle saturated cells (multiple existing chargers) | H3 (count-based sum over E_all) | H1 (s_c drops with more chargers) |
| Don't waste two chargers on same cluster | H6 (coverage redundancy) | H4 (spatial spread) |
| Allow multiple chargers per cluster when m is large | H6 (tunable via ε/α₆) | H1 (strong attraction overrides weak H6) |
| Place exactly m chargers | H5 (hard constraint) | — |
| Empty cells near dense areas are still viable for placement | H1 (uses POI cell weight, not candidate cell weight) | — |
| All terms QUBO-compatible | ✓ All linear or quadratic in binary x | — |
| Works for both QAOA and GA | ✓ Single Q matrix shared | — |

---


### Tunable Parameters Summary

All α weights, radii, and λ have data-driven starting values computed by `suggest_parameters`. The defaults below reflect those formulas; all values are overridable.

| Parameter | Role | Default | Tuning Guide |
|-----------|------|---------|-------------|
| `α₁` | Weight of H1 (POI attraction) | 3.0 | Primary driver — should be dominant. Increase to pull chargers closer to dense POIs |
| `α₂` | Weight of H2 (gas station bonus) | 0.5–1.0 (reduced if gas stations are far from dense POIs) | Meaningful but not dominant. Increase for stronger gas station preference |
| `α₃` | Weight of H3 (existing charger penalty) | 1.0–2.0 (increased when existing coverage is sparse) | Moderate influence. Decrease to allow more tolerance for placing near existing chargers |
| `α₄` | Weight of H4 (new charger spacing) | 1.0–1.5 (increased when POI clusters ≥ m) | Moderate influence. Decrease to allow charger clustering in dense areas |
| `α₅` | Weight of H5 (charger count constraint) | 1.0 | Must enforce constraint. Should always be high enough that violating Σx=m is never worth it |
| `α₆` | Weight of H6 (coverage redundancy) | 1.0–1.5 (increased when POI clusters ≥ m) | Moderate influence. Decrease when m is large relative to number of POI clusters (need multiple chargers per cluster). Increase for strict maximum coverage spread |
| `β` | Gas station bonus magnitude | 1.0 | Controls gas station attraction. Too high → forces chargers onto gas stations ignoring POIs |
| `γ` | Existing charger penalty magnitude | 1.0 | Controls existing charger repulsion. Too high → avoids existing chargers even when area needs more |
| `δ` | New charger spacing magnitude | 1.0 | Controls new charger mutual repulsion. Too high → forces spread even in areas that need clustering |
| `ε` | Coverage redundancy magnitude | 1.0 | Controls overlap penalty. Too high → prevents multiple chargers per cluster even when demand justifies it |
| `λ` | Constraint penalty multiplier | 5 × (α₁ + α₂) × m | 5× safety margin over max objective gain. Too low → wrong number of chargers. Too high → dominates and flattens objective |
| `R₁` | H1 attraction radius | 30% of grid side, min 2 | Largest radius — how far chargers "see" POIs. Too small → chargers placed only directly on POIs |
| `Rₛ` | Service gap radius for s_c | = R₁ | Always matches R₁ for consistency. Controls how far existing chargers reduce POI attraction |
| `R₃` | H3 existing charger penalty radius | 20% of grid side − 1, min 2 | How far existing chargers repel new ones. Must be < R₁ |
| `R₄` | H4 new charger spacing radius | 20% of grid side, min 2 | How far new chargers repel each other. Must be > R₃. Major impact on Q sparsity |
| `R₆` | H6 coverage redundancy radius | 15% of grid side, min 2 | Smallest radius — defines "serving the same area". Must be ≤ R₃ |
| `scale_factor` | Cell weight scaling | 5 | Controls weight differentiation. Increase → bigger gap between dense and sparse cells |
| `min_weight` | Cell weight floor | 0.5 | Floor for lowest-density cells with POIs. Increase → sparse areas stay more relevant |

Ordering constraint enforced by `suggest_parameters`: **R₁ ≥ R₄ > R₃ ≥ R₆**. Violating this degrades Q sparsity and makes terms interact incorrectly.

Note: `α` parameters and `β, γ, δ, ε` are technically redundant (e.g., `α₂ × β` could be a single parameter). They are kept separate for clarity — `α` values control the relative importance between objectives, while `β, γ, δ, ε` control magnitudes within each objective. During tuning, some may be collapsed.

Each term's raw contributions are **normalized** (divided by max absolute value across the grid) before applying `α` weights, so that `α₁ = 2, α₃ = 1` genuinely means "POI attraction is twice as important as existing charger penalty" regardless of dataset.


---

## QUBO-Based Cell Pruner

### Purpose and Pipeline Position

The cell pruner sits **after** `build_qubo()` on the full N-cell grid and **before** QAOA. It reduces the number of binary variables fed to the quantum stage by eliminating grid cells that provably (or very likely) cannot appear in any optimal solution. This is what makes QAOA tractable at medium scale without modifying the QUBO formulation itself.

The pruner is called "QUBO-based" because its correctness guarantees derive directly from the QUBO structure, specifically, the fact that all off-diagonal entries in Q_obj (H4 and H6) are non-negative (they are penalties). This gives us a key mathematical property:

> **solo(i) = Q_obj[(i,i)] is an upper bound on cell i's net contribution to any solution.** Pairwise terms can only add cost, never reduce it below the solo score.

### Three-Tier Pruning Architecture

The pruner runs three tiers in sequence. Each tier returns a surviving set passed to the next.

---

#### Tier 1 — Dead Cell Elimination (Provably Safe)

Removes cells with `solo(i) = 0` that are also beyond radius R₄ of any "competitive" cell (a cell with `solo < 0`).

**Rationale:** A zero-solo cell contributes nothing to the objective on its own. The only reason to keep a zero-solo cell is if it is within R₄ of a competitive cell, in that case it could provide a "spacing relief" option that reduces H4 penalties for the surrounding solution. Beyond R₄, it cannot interact with competitive cells and is safe to remove.

**Guarantee:** Provably safe. No optimal solution can contain a zero-solo cell that is also isolated from all competitive cells.

---

#### Tier 2 — Bound-Based Elimination (Provably Safe)

For each surviving cell i, computes a lower bound on the QUBO score of any solution containing i:

```
LB(i) = solo(i) + sum of (m-1) best solos from all other surviving cells
```

Since pairwise penalties are ≥ 0, this lower bound is valid, the true score of any solution containing i is at least LB(i). A reference score is obtained by running a sequential greedy search (picks cells one at a time minimizing total score including pairwise terms) on the surviving set.

If `LB(i) > reference_score`, cell i is provably suboptimal and removed. Cells selected by the greedy search itself are never pruned.

**Guarantee:** Provably safe. Any optimal solution scoring ≤ reference_score is preserved.

---

#### Tier 3 — Spatial Deduplication (Heuristic)

Groups surviving cells into micro-clusters: two cells are in the same cluster if they are adjacent (Chebyshev distance ≤ 1) **and** their solo scores differ by less than 5% of the overall solo score range. Within each cluster, keeps the top-K by solo score (default K = max(3, m)).

**Rationale:** Clusters of nearly identical adjacent cells are approximately interchangeable from the optimizer's perspective. Keeping only the top few retains the best representatives while reducing variables.

**Important caveats:**
- This is a heuristic — it is not provably optimal-preserving.
- The 5% score similarity gate prevents false merges between adjacent cells that play structurally different roles.
- A minimum survival floor of max(3m, 6) cells is enforced: pruning stops if the surviving count would fall below this floor.
<!-- - Must be validated on small instances via brute-force before trusting on larger problems. -->


---

### Expected Reduction

For EVCP(5,3,3) on a 16×16 grid (N=256):
- Most cells have `solo = 0` (no nearby POIs, no gas stations) → eliminated in Tier 1
- Tier 2 prunes cells with solo scores dominated by the greedy reference
- Typical result: 256 → 20–60 surviving cells depending on POI density distribution

This reduction is what makes statevector QAOA feasible on medium instances and MPS-based QAOA practical on larger ones.

---

## Automated Parameter Suggestion

Before running optimization, the pipeline automatically computes a data-driven starting point for all QUBO parameters. This is implemented in `suggest_parameters(grid_details, cell_weights, plot_deets, m)` and runs after grid construction and weight calculation, before Q matrix construction.

### What It Computes

**Radii** — derived as fractions of the grid side length, with the ordering constraint R₁ ≥ R₄ > R₃ ≥ R₆ enforced:

| Radius | Formula | Purpose |
|--------|---------|---------|
| `R₁` | 30% of grid side, min 2 | H1 POI attraction reach |
| `Rₛ` | = R₁ | Service gap radius, matches R₁ for consistency |
| `R₃` | 20% of grid side − 1, min 2 | H3 existing charger repulsion reach |
| `R₄` | 20% of grid side, min 2 | H4 new charger spacing reach |
| `R₆` | 15% of grid side, min 2 | H6 coverage redundancy reach |

**α weights** — set by magnitude-matching: each α is chosen so that its term contributes at a comparable scale to α₁ H₁ (the dominant term). Additional adjustments are made based on data diagnostics:

- `α₁ = 3.0` always (dominant driver)
- `α₂` reduced if gas stations are mostly away from high-density POI areas
- `α₃` increased if existing coverage is sparse (high average service gap `s_c`)
- `α₄`, `α₆` increased if the number of detected POI clusters ≥ m (spread matters more)

**λ** — set to `5 × (α₁ + α₂) × m`, giving a 5× safety margin over the maximum possible objective gain from placing m chargers. This ensures the constraint H5 always dominates any benefit from violating Σx = m.

**Intra-term magnitudes** — β, γ, δ, ε all default to 1.0. These are effectively absorbed by the α weights after per-term normalization and are left for manual tuning.

### Output

`suggest_parameters` returns a dict with four keys:

```python
{
  'radii':       {'R1': int, 'Rs': int, 'R3': int, 'R4': int, 'R6': int},
  'alpha':       {'a1': float, 'a2': float, 'a3': float, 'a4': float, 'a5': float, 'a6': float},
  'intra':       {'beta': float, 'gamma': float, 'delta': float, 'epsilon': float},
  'lambda':      float,
  '_magnitudes': {...},   # raw term magnitudes before α, for inspection
  '_diagnostics': {...},  # n_clusters, avg_service_gap, gas_poi_overlap, etc.
}
```

All values are overridable — the suggestion is a calibrated starting point, not a constraint. `print_parameter_suggestions(params)` prints a formatted summary with the diagnostic notes explaining each choice.

---

## Output Specification

### Number of Output Grid IDs
The algorithm outputs more candidate grids than the number of new stations needed:

```
num_output = n + BUFFER
```

Where `n` = number of new stations to place and `BUFFER` is a small fixed integer (default: 3). This gives planners a shortlist with flexibility — not every grid location may be feasible in practice, and the next-best options are immediately available without re-running the algorithm.

The buffer is fixed for now.
### Output Format
A ranked list of grid IDs with their fitness scores, showing why each was selected (proximity to which POIs, gas station presence, etc.).

---

## QAOA Component Details

**QAOA (Quantum Approximate Optimization Algorithm)** implemented via **Qiskit** (`qaoa_builder.py`).

### Role in the Hybrid

The quantum component's job is to produce a high-quality initial population for the GA by:
- Exploring the pruned combinatorial space (K surviving cells, not the full N-cell grid)
- Leveraging QAOA's variational structure to concentrate probability on low-energy states
- Returning multiple good candidate solutions (not just one) via the output buffer

### Execution Modes

Two execution modes are available depending on post-prune variable count K:

#### Mode 1 — Exact Statevector (`run_qaoa`)
- Uses `StatevectorEstimator` and `Statevector.from_instruction` for exact probability extraction
- COBYLA optimization of variational parameters, multiple random restarts
- Returns the exact statevector after optimization → all feasible basis states ranked by QUBO score
- **Use when:** K ≤ ~20. This is the ground-truth reference; no approximation.

#### Mode 2 — MPS Simulation (`run_qaoa` with Aer MPS backend, or `run_qaoa_noisy`)
- Uses Qiskit Aer's MPS simulator (`method='matrix_product_state'`)
- For low-p QAOA (p ≤ 3), circuits are shallow and build limited entanglement. MPS can represent these states exactly at sufficient bond dimension.
- **Use when:** K ≤ ~50–80. This is the primary PoC execution mode for medium-scale problems.
- **Bond dimension:** Should be set high enough that Mode 1 and Mode 2 agree on the top-3 solutions for small instances (K ≤ 20). If they disagree, bond dimension is too low.


### What "PoC Valid" Means for MPS

MPS simulation is classical: it gives no evidence of quantum speedup. The valid PoC claim is:

> "MPS-seeded GA outperforms random-seeded GA on medium-scale instances."

This demonstrates that the *seeding benefit* — better initial population → faster convergence → better final solution, is a real structural property of the hybrid approach. Because MPS accurately simulates low-p QAOA circuits, the same seeding benefit would be expected from actual QAOA hardware. This is the claim the experiments support; quantum speedup is out of scope for this PoC.

### QUBO → Ising Conversion

QAOA operates on an Ising Hamiltonian, not the QUBO directly. The conversion uses the substitution $x_i = (1 − z_i)/2$ where $z_i ∈ {−1, +1}$:

- **Diagonal Q[(i,i)]:** contributes $h[i] = −Q_{ii}/2$ (single-Z term) and offset $Q_{ii}/2$
- **Off-diagonal Q[(i,j)]:** contributes $J[(i,j)] = Q_{ij}/4$ (ZZ term), $h[i] += −Q_{ij}/4, h[j] += −Q_{ij}/4$, offset $Q_{ij}/4$
- **H5 (constraint):** converted separately via `h5_to_ising_coeffs` and merged — never stored in the sparse $Q_{obj}$

The key invariant: `QUBO_energy(x) = Ising_energy(z) + offset`, so evaluating solutions in either space gives consistent results.

### Output Format and Buffer

```python
results = [(score, [cell_ids], probability_or_frequency), ...]
```

Sorted ascending by score (best first). Length = `m + BUFFER` where `BUFFER = min(max(1, ceil(m/2)), 5)`. Cell IDs are in the original pre-pruning coordinate system (translation from qubit indices is applied automatically).

### Integration Approach

We use **Multi-shot seeding**: QAOA returns top-k solutions which become the GA's initial population.

---

## Genetic Algorithm Component Details

### Individual Encoding
Each individual = a list of grid IDs where new chargers are placed.

Example: `[42, 117, 203]` for placing 3 new chargers.

### Operators (To Be Explored)
- **Selection:** Tournament selection or roulette wheel (TBD)
- **Crossover:** Swap subsets of charger placements between parents (TBD — papers to review)
- **Mutation:** Shift a charger to a neighboring grid cell (spatially aware) (TBD — papers to review)


---

## Dataset Strategy

### Synthetic Data
We will generate synthetic city-like datasets since real-world data may be hard to obtain. The data will model realistic distributions:

- **POIs:** Clustered to simulate neighborhoods/districts
- **Gas stations:** Distributed along simulated main roads
- **Existing chargers:** Placed unevenly to simulate real-world coverage gaps

### Target Scale
Ballpark for a realistic mid-size city:
- 50–200 POIs with varying population densities
- 10–30 existing charging stations
- 20–50 gas stations
- 5–20 new stations to place

Exact numbers to be calibrated during development.



**Start small (paper-comparable datasets) → scale to mid-size city → push to see where the algorithm's limits are.**

---

## Open Items (To Be Resolved During Development)

| Item | Status |
|------|--------|
| **Genetic Algorithm module** (`ga_solver.py`) | **NEXT — blocker for hybrid integration** |
| Tier 3 brute-force validation (q=4, N=16, m=2) | To do before trusting pruner on larger instances |
| MPS agreement test (statevector vs MPS, K ≤ 20) | To do before medium-scale experiments |
| Output candidate count (`n + BUFFER`) | Fixed buffer of 3, to be validated against user needs |
| Crossover and mutation strategies | Papers to review, then decide |
| QUBO α weights (`α₁` through `α₆`) | Data-driven defaults from `suggest_parameters`; experimental tuning pending |
| Gas station bonus magnitude (`β`) | To be tuned experimentally |
| Existing charger penalty magnitude (`γ`) | To be tuned experimentally |
| New charger spacing magnitude (`δ`) | To be tuned experimentally |
| Coverage redundancy magnitude (`ε`) | To be tuned experimentally |
| Constraint penalty multiplier (`λ`) | Heuristic: ~5× max objective gain; validated by score decomposition |
| Local radii (`R₁, Rₛ, R₃, R₄, R₆`) | Defaults set, to be tuned experimentally |
| `scale_factor` for cell weighting | Default: 5, to be tuned experimentally |
| `min_weight` floor for cell weighting | Default: 0.5, to be tuned experimentally |
| Tier 2 greedy runtime on N=256 | Needs timing — O(m × K × |Q_obj|) can be slow; profile before scaling |
| Constrained mixers (XY-mixer) for QAOA | To explore if H5 causes performance issues; eliminates constraint term entirely |
| MPS bond dimension selection | Validated when statevector and MPS top-3 agree; default TBD |

---

## Key Innovations Over the Reference Paper

1. **Direct qubit-to-cell mapping** — N grid cells = N physical qubits. Grid resolution scales exactly with available hardware. N must be composite to allow a rectangular layout; the code enforces this constraint.
2. **Continuous density-based cell weighting** — eliminates arbitrary tier boundaries, adapts automatically to grid resolution changes, and preserves full granularity of the density distribution.
3. **QUBO-based cell pruner** — reduces the N-cell QUBO to K surviving cells (K << N) using provably safe Tier 1 and Tier 2 pruning plus heuristic Tier 3 spatial deduplication. Operates on the built Q_obj so solo scores exactly match QUBO diagonals. Makes QAOA tractable on medium-scale problems without modifying the formulation.
4. **Three-tier execution strategy** — brute-force for exact validation (K ≤ 20), statevector QAOA for small post-prune instances, MPS-based QAOA simulation as the PoC-valid path for medium scale. Adapts to available compute resources.
5. **Automated parameter suggestion** — `suggest_parameters` derives data-driven starting values for all α weights, radii, and λ from the grid data, using magnitude-matching and dataset diagnostics. Removes the need for blind manual tuning.
6. **Unified QUBO formulation for both QAOA and GA** — same Q matrix drives both algorithms, enabling fair comparison and consistent optimization landscape.
7. **Smooth charger clustering penalty** — inversely scaled by cell weight, allowing clustering in high-demand areas and penalizing in low-demand areas with no arbitrary cutoffs.
8. **Gas station co-location bonus** — leverages existing infrastructure as a soft incentive, scales with gas station count.
9. **Flexible output (shortlist > n)** — gives planners practical decision support.
10. **Chebyshev grid distance** — simpler and more intuitive than Euclidean for zonal planning.
11. **Local radii for Q matrix sparsity** — every term uses distance cutoffs, H5 stored separately, making Q grid-local sparse.
12. **Saturation-aware repulsion** — existing charger penalty and service gap factor sum over individual chargers, not cells, naturally handling multi-charger saturation.

---

*Document Version: 1.9 — March 13, 2026*