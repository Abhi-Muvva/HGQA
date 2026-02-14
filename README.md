# Hybrid Quantum-Classical Genetic Algorithm for EV Charging Station Placement

## Project Baseline Document

---

## 1. Project Overview

This project develops a hybrid quantum-classical algorithm that optimally places new electric vehicle (EV) charging stations in a city-like environment. The approach combines Quantum Approximate Optimization Algorithm (QAOA) via Qiskit with a classical Genetic Algorithm (GA) to produce high-quality placement recommendations.

The core idea: the quantum algorithm explores the discrete solution space to generate a strong set of candidate solutions, which then seed a classical genetic algorithm for further refinement. This hybrid approach is inspired by — and aims to improve upon — the methodology in the reference paper (Chandra et al.), which demonstrated a 42.89% improvement over vanilla quantum annealing by seeding a GA with quantum results.

---

## 2. Problem Definition

**Given:**
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

## 3. Grid-Based Discretization

### 3.1 Grid Construction
The entire geographic area is divided into a grid of **2^q** cells, where `q` is the number of qubits available. Each grid cell is assigned a unique ID from `0` to `2^q - 1`.

- For `q = 8` qubits → 256 grid cells (default: 16×16)
- For `q = 10` qubits → 1024 grid cells (default: 32×32)

### 3.2 Grid Layout Options
The grid layout is configurable:
- **Default:** Attempts the closest square grid (e.g., 16×16 for 256 cells)
- **Show mode:** Displays all possible rectangular layouts (e.g., 256 = 16×16, 32×8, 64×4, etc.) and lets the user pick one

### 3.3 Distance Metric
**Chebyshev distance** between grid cells:
- Same grid cell = 0
- Any adjacent cell (including diagonals) = 1
- Two cells apart = 2, and so on

This means 8-directional adjacency where moving diagonally costs the same as moving horizontally or vertically.

### 3.4 Why Grid-Based?
- Directly tied to qubit count (scalable with hardware)
- More realistic than pinpoint coordinates — real-world placement decisions are zonal
- Output is a grid region, not an exact coordinate, which is more practical for planners

---

## 4. Data Model

### 4.1 Input Data
Three categories of data points, each placed on the grid:

| Data Type | Attributes | Notes |
|-----------|-----------|-------|
| **Points of Interest (POIs)** | Location (x, y), Population Density (0 to 1) | 1 = highest density, 0 = lowest density |
| **Existing Charging Stations** | Location (x, y) | Already operational EV chargers |
| **Gas Stations** | Location (x, y) | Potential co-location sites for new chargers |

### 4.2 Continuous Density-Based Cell Weighting (Why Not Tiers)

This project uses a **continuous weighting system** instead of discrete tiers (Tier 1/2/3). The reasoning:

A discrete tier system assigns fixed labels (e.g., Tier 1, 2, 3) to POIs or grid cells. This creates several problems in a grid-based approach:

1. **Grid resolution dependency:** If tiers are assigned to individual POIs before gridding, the labels become meaningless after gridding. Changing the qubit count changes the grid resolution, which reshuffles which POIs land in which cells. A "Tier 1" POI alone in a large cell may matter less than three "Tier 3" POIs clustered in a small cell — but the tier labels don't capture this.

2. **Arbitrary boundary effects:** With tiers, a cell at the 30th percentile gets weight 5, while a cell at the 31st percentile drops to weight 3. This cliff-edge creates discontinuities in the fitness landscape that can mislead the optimization — two nearly identical cells get treated very differently.

3. **Loss of granularity:** Three discrete weights (5, 3, 1) throw away the rich information contained in the actual density distribution. Two cells both labeled "Tier 2" could have very different real-world importance.

The continuous system solves all three problems by deriving cell weights from the data after gridding, producing a smooth gradient of importance with no arbitrary cutoffs.

### 4.3 Cell Weight Calculation

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
- Empty cells are completely ignored (weight = 0)
- No arbitrary cutoff boundaries — the gradient is smooth
- Automatically adapts when grid resolution changes (different qubit count)

### 4.4 Grid Data Table
After gridding and weight calculation, each cell contains:

| Field | Description |
|-------|-------------|
| Grid ID | Unique identifier (0 to 2^q - 1) |
| Number of POIs | Count of POIs in this cell |
| Raw Density Score | Sum of population densities of POIs in this cell |
| Normalized Score | Raw score / max raw score across all cells |
| Cell Weight | Final weight after linear scaling with floor |
| Has Gas Station | Boolean (or count) |
| Has Existing Charger | Boolean (or count) |

---

## 5. Algorithm Pipeline

### Phase 1: Data Preparation
```
Raw Data (POIs with densities, chargers, gas stations)
        ↓
Grid Discretization (2^q cells, configurable layout)
        ↓
Aggregate densities per cell → Normalize → Scale (linear with floor)
        ↓
Grid Data Table (summary per cell with continuous weights)
```

### Phase 2: Quantum Optimization (QAOA via Qiskit)
```
Grid Data Table
        ↓
Formulate as optimization problem (cost Hamiltonian)
        ↓
Run QAOA (multiple shots / parameter variations)
        ↓
Collect top-k candidate solutions
        ↓
Candidate Grid ID Sets (initial population for GA)
```

### Phase 3: Classical Genetic Algorithm Refinement
```
Initial Population (from QAOA output)
        ↓
Evaluate Fitness (density-weighted proximity + bonuses + penalties)
        ↓
Selection → Crossover → Mutation (spatially aware)
        ↓
Repeat for G generations
        ↓
Final Ranked Shortlist of Grid IDs
```

### Phase 4: Evaluation & Visualization
```
Compare: Only QAOA vs Only GA (random seed) vs Hybrid (QAOA + GA)
        ↓
Scoring metrics, convergence plots, grid visualizations
```

---

## 6. Fitness Function — Unified QUBO Formulation

### 6.1 Why QUBO Format

The fitness function is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem. This is a deliberate design choice that provides a critical advantage: **the same mathematical formulation drives both the quantum and classical components of the hybrid algorithm.**

A QUBO problem takes the form:

```
f(x) = x^T Q x = Σ_i Σ_j  Q_ij × x_i × x_j
```

Where `x` is a vector of binary variables and `Q` is a square matrix of weights.

**Why this matters for our hybrid approach:**

1. **QAOA requires it.** QAOA works by encoding the optimization objective as a cost Hamiltonian, which is derived directly from the QUBO matrix Q. Without a QUBO formulation, we cannot use QAOA at all.

2. **The GA can use it directly.** Given an individual (a list of grid IDs), we convert it to a binary vector `x` (1s at selected grid positions, 0s elsewhere) and evaluate `f(x) = x^T Q x`. This is a simple matrix operation — fast and exact.

3. **Unified objective = fair comparison.** Since both QAOA and GA optimize the exact same function, we can fairly compare their results. If the GA uses a different fitness function, any performance difference could be attributed to either the search strategy OR the objective difference — making it impossible to isolate the quantum advantage.

4. **Single Q matrix = build once, use everywhere.** The Q matrix is computed once from the grid data and shared across both algorithms. No duplication, no inconsistency.

### 6.2 Binary Variables

We define `N = 2^q` binary variables, one for each grid cell:

```
x_i = 1  if a new charger is placed in grid cell i
x_i = 0  otherwise
```

The solution vector `x` has exactly `m` entries equal to 1, where `m` is the number of new chargers to place.

### 6.3 Notation Reference

Before defining the objective terms, here is the notation used throughout:

| Symbol | Meaning |
|--------|---------|
| `N` | Total number of grid cells (= 2^q) |
| `m` | Number of new chargers to place |
| `x_i` | Binary variable: 1 if charger placed in cell i, 0 otherwise |
| `w_c` | Cell weight of cell c (from Section 4.3, continuous, 0 to scale_factor) |
| `nw_c` | Normalized cell weight of cell c (= w_c / scale_factor, range [0, 1]) |
| `d(i, j)` | Chebyshev distance between cells i and j |
| `C_POI` | Set of cells that contain at least one POI |
| `C_existing` | Set of cells that contain an existing charger |
| `g_i` | 1 if cell i contains a gas station, 0 otherwise |

### 6.4 Objective Term H1 — POI Attraction (Minimize → Place Chargers Near Dense Areas)

```
H1 = - Σ_i  x_i × Σ_{c ∈ C_POI}  ( w_c / (1 + d(i, c)) )
```

**What it does:** For each candidate cell `i`, it computes how attractive that cell is based on the weighted proximity to ALL POI cells. Cells closer to high-weight POI clusters get a more negative (better) score.

**Why designed this way:**

- The **negative sign** means minimizing H1 rewards placing chargers where the attraction is highest.

- The weight `w_c` comes from the **POI cell**, not the candidate cell. This is important: an empty cell (weight = 0) adjacent to a dense cell (weight = 4.0) should still be attractive for charger placement. If we used the candidate cell's own weight, empty cells would never attract chargers, even when they're right next to areas that need them.

- The **`1 / (1 + d(i,c))`** decay function ensures closer POIs contribute more. At distance 0 (same cell), contribution is `w_c / 1 = w_c`. At distance 1 (adjacent), it's `w_c / 2`. At distance 5, it's `w_c / 6`. The decay is gradual — distant POIs still matter, just less.

- **Summing over all POI cells** means the algorithm naturally prefers locations that are centrally positioned relative to many POIs, not just close to one.

**QUBO placement:** This is linear in `x_i`, so each term goes on the **diagonal** of the Q matrix:

```
Q_ii += α₁ × ( - Σ_{c ∈ C_POI}  w_c / (1 + d(i, c)) )
```

### 6.5 Objective Term H2 — Gas Station Co-location Bonus (Minimize → Prefer Gas Station Cells)

```
H2 = - Σ_i  x_i × β × g_i
```

**What it does:** Gives a bonus (negative contribution = better score) when a charger is placed in a cell that has a gas station.

**Why designed this way:**

- The **negative sign** rewards gas station co-location.

- `g_i` is binary (1 if gas station present, 0 otherwise), so this term only activates for cells with gas stations.

- `β` is the **bonus weight** — a tunable parameter that controls how important gas station co-location is relative to other objectives. If `β` is too large, the algorithm would force chargers onto gas stations even when better POI-serving locations exist. If too small, the bonus becomes negligible. This is deliberately a soft bonus, not a hard constraint.

- This reflects the real-world advantage: gas stations already have land, power infrastructure, and customer traffic patterns that make them natural candidates for EV charger installation.

**QUBO placement:** Linear in `x_i`, goes on the **diagonal**:

```
Q_ii += α₂ × ( - β × g_i )
```

### 6.6 Objective Term H3 — Existing Charger Penalty (Minimize → Avoid Redundancy in Low-Density Areas)

```
H3 = + Σ_i  x_i × γ × (1 - nw_i) × Σ_{e ∈ C_existing}  ( 1 / (1 + d(i, e)) )
```

**What it does:** Penalizes placing a new charger near an existing charger, BUT the penalty is scaled by how low-density the candidate cell is.

**Why designed this way:**

- The **positive sign** means this adds to the cost — it's a penalty.

- **`(1 - nw_i)` is the key design element.** This uses the normalized weight of the candidate cell `i` (where the new charger would go):
  - High-density cell (nw_i = 0.8): penalty factor = 0.2 → almost no penalty. Rationale: busy areas can support multiple nearby chargers because charging takes time and queues form.
  - Medium-density cell (nw_i = 0.5): penalty factor = 0.5 → moderate penalty.
  - Low-density cell (nw_i = 0.1): penalty factor = 0.9 → strong penalty. Rationale: in sparse areas, spreading coverage is more valuable than clustering.
  - Empty cell (nw_i = 0.0): penalty factor = 1.0 → maximum penalty. No reason to cluster near existing chargers if there's no demand.

- The **`1 / (1 + d(i,e))`** decay means the penalty is strongest when directly on top of an existing charger and fades with distance.

- `γ` is the **penalty factor** — a tunable parameter controlling overall penalty strength.

**QUBO placement:** Linear in `x_i`, goes on the **diagonal**:

```
Q_ii += α₃ × ( γ × (1 - nw_i) × Σ_{e ∈ C_existing}  1 / (1 + d(i, e)) )
```

### 6.7 Objective Term H4 — New Charger Spacing (Minimize → Spread New Chargers in Low-Density Areas)

```
H4 = + Σ_i Σ_{j>i}  x_i × x_j × δ × (1 - max(nw_i, nw_j)) / (1 + d(i, j))
```

**What it does:** Penalizes placing two NEW chargers close to each other, but only when neither cell is high-density.

**Why designed this way:**

- This is the **only quadratic term** (involves pairs `x_i × x_j`), which is why we need QUBO format — linear programming can't express this interaction between two placement decisions.

- **`max(nw_i, nw_j)` instead of average:** If EITHER cell in the pair is high-density, the penalty drops. This is intentional — if two chargers are close but one is in a high-demand area, that clustering is justified. Using average would penalize a high-density cell paired with a low-density neighbor, which doesn't make sense. Using max means: "if there's strong demand at either location, allow the clustering."

- The **`1 / (1 + d(i,j))`** decay means only nearby pairs get penalized. Two chargers on opposite ends of the grid don't interact.

- `δ` is the **spacing penalty factor** — controls how strongly we enforce spread.

- **Summing over `j > i`** (not `j ≠ i`) avoids double-counting each pair.

**QUBO placement:** This fills the **off-diagonal** entries of the Q matrix:

```
Q_ij += α₄ × ( δ × (1 - max(nw_i, nw_j)) / (1 + d(i, j)) )    for i ≠ j
```

### 6.8 Constraint Term H5 — Exact Number of Chargers

```
H5 = λ × ( Σ_i x_i  -  m )²
```

**What it does:** Forces the solution to place exactly `m` new chargers — no more, no less.

**Why designed this way:**

- This is a **hard constraint disguised as a penalty.** By making `λ` sufficiently large (much larger than the other terms), any solution that doesn't have exactly `m` chargers becomes so expensive that the optimizer avoids it.

- **Expanding the square** gives us QUBO-compatible terms:

```
H5 = λ × ( Σ_i Σ_j x_i x_j  -  2m × Σ_i x_i  +  m² )
```

Which breaks down into:
  - Diagonal: `Q_ii += α₅ × λ × (1 - 2m)` — penalizes each selected cell (counterbalanced by the off-diagonal reward when exactly m are selected)
  - Off-diagonal: `Q_ij += α₅ × λ` — penalizes every pair of selected cells
  - Constant: `λ × m²` — doesn't affect optimization, can be ignored

- The **`λ` value** must be chosen carefully. Too small: the optimizer violates the constraint. Too large: it dominates the objective and the optimizer ignores placement quality. A common heuristic is to set `λ` to be ~2-5× the magnitude of the largest other term.

**QUBO placement:** Both **diagonal and off-diagonal**:

```
Q_ii += α₅ × λ × (1 - 2m)
Q_ij += α₅ × λ                  for i ≠ j
```

### 6.9 Complete QUBO Formulation

The full objective to **minimize**:

```
H_final = α₁H₁ + α₂H₂ + α₃H₃ + α₄H₄ + α₅H₅
```

**Assembled Q matrix:**

```
DIAGONAL (Q_ii) — encodes single-cell properties:

  Q_ii = α₁ × ( - Σ_{c ∈ C_POI}  w_c / (1 + d(i,c)) )        ← POI attraction
       + α₂ × ( - β × g_i )                                     ← gas station bonus
       + α₃ × ( γ(1 - nw_i) × Σ_{e ∈ C_existing} 1/(1+d(i,e)) )  ← existing charger penalty
       + α₅ × λ × (1 - 2m)                                      ← charger count constraint


OFF-DIAGONAL (Q_ij, i ≠ j) — encodes pairwise interactions:

  Q_ij = α₄ × ( δ(1 - max(nw_i, nw_j)) / (1+d(i,j)) )         ← new charger spacing
       + α₅ × λ                                                  ← charger count constraint
```

**How each algorithm uses Q:**

| Algorithm | How it uses Q |
|-----------|--------------|
| **QAOA** | Q is converted to a cost Hamiltonian (Ising model). QAOA circuit parameters are optimized to find x that minimizes x^T Q x. Multiple shots yield multiple candidate solutions. |
| **GA** | Given an individual [42, 117, 203], convert to binary vector x, compute fitness = x^T Q x. Lower value = better individual. Standard GA operators (selection, crossover, mutation) evolve the population toward lower fitness. |

### 6.10 Tunable Parameters Summary

| Parameter | Role | Default | Notes |
|-----------|------|---------|-------|
| `α₁` | Weight of POI attraction term | TBD | Primary driver — should be dominant |
| `α₂` | Weight of gas station bonus term | TBD | Should be meaningful but not dominant |
| `α₃` | Weight of existing charger penalty term | TBD | Moderate influence |
| `α₄` | Weight of new charger spacing term | TBD | Moderate influence |
| `α₅` | Weight of charger count constraint | TBD | Must be large enough to enforce constraint |
| `β` | Gas station bonus magnitude | TBD | Controls how much gas stations attract chargers |
| `γ` | Existing charger penalty magnitude | TBD | Controls how much existing chargers repel new ones |
| `δ` | New charger spacing penalty magnitude | TBD | Controls how much new chargers repel each other |
| `λ` | Constraint penalty multiplier | TBD | Must dominate other terms (~2-5× largest term) |
| `scale_factor` | Cell weight scaling (from Section 4.3) | 5 | Controls weight differentiation |
| `min_weight` | Cell weight floor (from Section 4.3) | 0.5 | Floor for lowest-density cells with POIs |

Note: `α` parameters and `β, γ, δ` are technically redundant (e.g., `α₂ × β` could be a single parameter). They are kept separate for clarity — `α` values control the relative importance between objectives, while `β, γ, δ` control magnitudes within each objective. During tuning, some may be collapsed.

### 6.11 Known Limitation — The min() Problem

The ideal scoring metric would use the minimum distance from each POI to its nearest charger: `Σ min_distance(poi, nearest_charger)`. However, the `min()` function cannot be expressed as a quadratic function of binary variables. This is the same limitation identified in the reference paper (Section IV-B).

Our H1 term approximates this by summing weighted inverse distances from ALL POI cells — this creates attraction toward POI-dense regions and correlates well with the true min-distance metric when chargers are reasonably distributed. The GA refinement step after QAOA helps close any remaining gap between the QUBO proxy and the true objective.

If needed, the GA can optionally use a richer non-QUBO fitness function (with actual min-distance) as a secondary evaluation for final ranking, while keeping the QUBO as the primary optimization target for both algorithms.

---

## 7. Output Specification

### 7.1 Number of Output Grid IDs
The algorithm outputs more candidate grids than the number of new stations needed:

```
num_output = n + f(q)
```

Where `n` = number of new stations to place, `q` = number of qubits, and `f(q)` is a function we will calibrate (initial trial: `f(q) = q/3`).

This gives planners a shortlist with flexibility, since not every grid location may be feasible in practice.

### 7.2 Output Format
A ranked list of grid IDs with their fitness scores, showing why each was selected (proximity to which POIs, gas station presence, etc.).

---

## 8. Quantum Component Details

### 8.1 Framework
**QAOA (Quantum Approximate Optimization Algorithm)** implemented via **Qiskit** (or Cirq as alternative).

### 8.2 Role in the Hybrid
The quantum component's job is to produce a high-quality initial population for the GA by:
- Exploring the combinatorial space of possible charger placements
- Leveraging quantum superposition to evaluate many configurations simultaneously
- Returning multiple good candidate solutions (not just one)

### 8.3 Integration Approaches to Explore

| Approach | Description | Complexity |
|----------|-------------|------------|
| **A: Multi-shot seeding** | Run QAOA multiple times → top-k solutions become GA initial population | Low (start here) |
| **B: Warm-start GA** | QAOA best result → generate GA population as mutations around it | Medium |
| **C: Iterative hybrid** | QAOA → GA → feed back to QAOA → repeat | High |

We will start with Approach A and experiment with others if time permits.

---

## 9. Genetic Algorithm Component Details

### 9.1 Individual Encoding
Each individual = a list of grid IDs where new chargers are placed.

Example: `[42, 117, 203]` for placing 3 new chargers.

### 9.2 Operators (To Be Explored)
- **Selection:** Tournament selection or roulette wheel (TBD)
- **Crossover:** Swap subsets of charger placements between parents (TBD — papers to review)
- **Mutation:** Shift a charger to a neighboring grid cell (spatially aware) (TBD — papers to review)

### 9.3 GA Library
Python-based. Options include DEAP (a popular evolutionary computation library) or custom implementation — to be decided during development.

---

## 10. Dataset Strategy

### 10.1 Synthetic Data
We will generate synthetic city-like datasets since real-world data may be hard to obtain. The data will model realistic distributions:

- **POIs:** Clustered to simulate neighborhoods/districts
- **Gas stations:** Distributed along simulated main roads
- **Existing chargers:** Placed unevenly to simulate real-world coverage gaps

### 10.2 Target Scale
Ballpark for a realistic mid-size city:
- 50–200 POIs across all tiers
- 10–30 existing charging stations
- 20–50 gas stations
- 5–20 new stations to place

Exact numbers to be calibrated during development.

### 10.3 Scaling Plan
Start small (paper-comparable datasets) → scale to mid-size city → push to see where the algorithm's limits are.

---

## 11. Evaluation Plan

### 11.1 Methods to Compare
| Method | Description |
|--------|-------------|
| **Only QAOA** | Quantum algorithm alone |
| **Only GA (random seed)** | Classical GA with random initial population |
| **Hybrid (QAOA + GA)** | Proposed method — QAOA seeds the GA |

### 11.2 Metrics
- Primary fitness score (as defined in Section 6)
- Convergence speed (generations to reach good solution)
- Solution stability (variance across multiple runs)
- Paper's scoring metric (sum of min distances from each POI to nearest charger) — for benchmarking against the reference paper

### 11.3 Visualizations
- Grid maps showing POIs, existing chargers, gas stations, and new placements
- Fitness vs. generation convergence plots (GA only vs. Hybrid)
- Bar charts comparing scores across methods and datasets (similar to Figures 4 and 5 in the reference paper)

---

## 12. Development Roadmap

### Step 1: Data Generation Module
- Synthetic city data generator
- POIs with population density values (0 to 1), gas stations, existing chargers
- Configurable parameters (city size, density distribution, etc.)

### Step 2: Grid Discretization Module
- Flexible 2^q gridding with layout options
- Grid data table construction
- Chebyshev distance computation utilities

### Step 3: Quantum Solver Module (QAOA)
- Problem formulation as cost Hamiltonian
- QAOA circuit construction and execution via Qiskit
- Multi-shot solution collection

### Step 4: Genetic Algorithm Module
- Individual encoding, fitness function
- Selection, crossover, mutation operators
- Population management and evolution loop

### Step 5: Hybrid Integration
- QAOA output → GA initial population pipeline
- End-to-end hybrid execution

### Step 6: Evaluation & Benchmarking
- Run all three methods on same datasets
- Compute metrics, generate plots
- Compare with reference paper results

### Step 7: Scaling & Optimization
- Test on larger datasets
- Tune hyperparameters
- Explore alternative integration approaches (B, C)

---

## 13. Open Items (To Be Resolved During Development)

| Item | Status |
|------|--------|
| Exact formula for output candidate count (`n + f(q)`) | Trial: `f(q) = q/3`, to be calibrated |
| Crossover and mutation strategies | Papers to review, then decide |
| Specific QAOA circuit design and parameter optimization | To be designed in Step 3 |
| GA library choice (DEAP vs custom) | To be decided in Step 4 |
| Exact dataset sizes for benchmarking | To be calibrated in Step 1 |
| Paper's scoring metric integration | To be addressed when relevant |
| QUBO α weights (`α₁` through `α₅`) | To be tuned experimentally |
| Gas station bonus magnitude (`β`) | To be tuned experimentally |
| Existing charger penalty magnitude (`γ`) | To be tuned experimentally |
| New charger spacing magnitude (`δ`) | To be tuned experimentally |
| Constraint penalty multiplier (`λ`) | Heuristic: ~2-5× largest other term |
| `scale_factor` for cell weighting | Default: 5, to be tuned experimentally |
| `min_weight` floor for cell weighting | Default: 0.5, to be tuned experimentally |

---

## 14. Key Innovations Over the Reference Paper

1. **Grid-based discretization tied to qubit count** — scalable and practical
2. **Continuous density-based cell weighting** — eliminates arbitrary tier boundaries, adapts automatically to grid resolution changes, and preserves full granularity of the density distribution (see Section 4.2 for detailed rationale vs. discrete tiers)
3. **Unified QUBO formulation for both QAOA and GA** — same Q matrix drives both algorithms, enabling fair comparison and consistent optimization landscape (see Section 6.1)
4. **Smooth charger clustering penalty** — inversely scaled by cell weight, allowing clustering in high-demand areas and penalizing in low-demand areas with no arbitrary cutoffs
5. **Gas station co-location bonus** — leverages existing infrastructure as a soft incentive
6. **QAOA instead of D-Wave annealing** — accessible without specialized hardware
7. **Flexible output (shortlist > n)** — gives planners practical decision support
8. **Chebyshev grid distance** — simpler and more intuitive than Euclidean for zonal planning

---

*Document Version: 1.2 — February 14, 2026*
*Status: Baseline updated — unified QUBO formulation added with full derivation*