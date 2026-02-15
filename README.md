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
f(x) = x^T Q x = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j
```

Where `x` is a vector of binary variables and `Q` is an upper triangular matrix of weights. Since `x_i ∈ {0,1}`, we have `x_i² = x_i`, so diagonal terms are effectively linear.

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
| `s_c` | Service gap factor for POI cell c (how underserved it is by existing chargers) |
| `R` | Service radius for coverage redundancy check (tunable, default ~3-4 cells) |

### 6.4 Objective Term H1 — POI Attraction (Minimize → Place Chargers Near Underserved Dense Areas)

```
H1 = - Σ_i  x_i × Σ_{c ∈ C_POI}  ( w_c × s_c / (1 + d(i, c)) )
```

where the service gap factor `s_c` is a precomputed constant:

```
s_c = 1 / (1 + Σ_{e ∈ C_existing}  1 / (1 + d(c, e)))
```

**What it does:** For each candidate cell `i`, it computes how attractive that cell is based on the weighted proximity to ALL POI cells, discounted by how well those POIs are already served by existing chargers. Cells closer to high-weight, underserved POI clusters get a more negative (better) score.

**Why designed this way:**

- The **negative sign** means minimizing H1 rewards placing chargers where the attraction is highest.

- The weight `w_c` comes from the **POI cell**, not the candidate cell. This is critical: an empty cell (weight = 0) adjacent to a dense cell (weight = 4.0) should still be attractive for charger placement. If we used the candidate cell's own weight, empty cells would never attract chargers, even when they're right next to areas that need them.

- The **service gap factor `s_c`** reduces attraction to POIs that are already well-served by existing chargers. If POI cell `c` already has an existing charger at distance 0, `s_c = 1/(1+1) = 0.5` — attraction is halved. If two existing chargers are nearby, attraction drops further. If no existing chargers are near, `s_c ≈ 1.0` — full attraction. This prevents the algorithm from piling new chargers onto areas that already have good coverage, and instead directs them to underserved areas. Importantly, `s_c` is precomputed from the data (it doesn't depend on `x`), so H1 remains linear in `x_i` and QUBO-compatible.

- The **`1 / (1 + d(i,c))`** decay function ensures closer POIs contribute more. At distance 0 (same cell), contribution is `w_c × s_c`. At distance 1 (adjacent), it's `w_c × s_c / 2`. At distance 5, it's `w_c × s_c / 6`. The decay is gradual — distant POIs still matter, just less.

- **Summing over all POI cells** means the algorithm naturally prefers locations that are centrally positioned relative to many POIs, not just close to one.

**Scenario tests:**

| Scenario | Behavior | Correct? |
|----------|----------|----------|
| Cell right on high-weight POI (w=4.0), no existing charger (s≈1.0) | Strong attraction: -4.0 | ✓ |
| Cell right on high-weight POI (w=4.0), existing charger at d=0 (s=0.5) | Reduced attraction: -2.0 | ✓ |
| Empty cell adjacent to dense POI (d=1) | Attracted: -w×s/2 | ✓ |
| Cell far from all POIs (d>10) | Weak attraction: ~0 | ✓ |

**QUBO placement:** Linear in `x_i` → **diagonal** of Q matrix:

```
Q_ii += α₁ × ( - Σ_{c ∈ C_POI}  w_c × s_c / (1 + d(i, c)) )
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

**Scenario tests:**

| Scenario | Behavior | Correct? |
|----------|----------|----------|
| Cell has gas station | Bonus: -β | ✓ |
| Cell has no gas station | No effect: 0 | ✓ |

**QUBO placement:** Linear in `x_i` → **diagonal**:

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

**Relationship with H1's service gap factor:** H1 (via `s_c`) reduces attraction to already-served POIs. H3 directly repels new chargers from existing charger locations. These are complementary, not redundant: a cell could be near an underserved POI (high `s_c` → H1 attracts) yet still close to an existing charger serving a different area (H3 repels). Both perspectives are needed.

**Scenario tests:**

| Scenario | Behavior | Correct? |
|----------|----------|----------|
| High-density cell (nw=0.9) on existing charger (d=0) | Tiny penalty: 0.1γ | ✓ |
| Empty cell (nw=0) on existing charger (d=0) | Max penalty: γ | ✓ |
| Any cell far from existing chargers (d>10) | Negligible penalty | ✓ |
| Medium cell (nw=0.5) at d=2 from existing | Moderate: 0.167γ | ✓ |

**QUBO placement:** Linear in `x_i` → **diagonal**:

```
Q_ii += α₃ × ( γ × (1 - nw_i) × Σ_{e ∈ C_existing}  1 / (1 + d(i, e)) )
```

### 6.7 Objective Term H4 — New Charger Spacing (Minimize → Spread New Chargers in Low-Density Areas)

```
H4 = + Σ_i Σ_{j>i}  x_i × x_j × δ × (1 - max(nw_i, nw_j)) / (1 + d(i, j))
```

**What it does:** Penalizes placing two NEW chargers close to each other, but only when neither cell is high-density.

**Why designed this way:**

- This is a **quadratic term** (involves pairs `x_i × x_j`), capturing the interaction between two placement decisions that linear terms cannot express.

- **`max(nw_i, nw_j)` instead of average:** If EITHER cell in the pair is high-density, the penalty drops. This is intentional — if two chargers are close but one is in a high-demand area, that clustering is justified. Using average would penalize a high-density cell paired with a low-density neighbor, which doesn't make sense. Using max means: "if there's strong demand at either location, allow the clustering."

- The **`1 / (1 + d(i,j))`** decay means only nearby pairs get penalized. Two chargers on opposite ends of the grid don't interact.

- `δ` is the **spacing penalty factor** — controls how strongly we enforce spread.

- **Summing over `j > i`** (not `j ≠ i`) avoids double-counting each pair.

**Scenario tests:**

| Scenario | Behavior | Correct? |
|----------|----------|----------|
| Two chargers in adjacent high-density cells (nw=0.9, d=1) | Tiny: 0.05δ | ✓ |
| Two chargers in adjacent empty cells (nw=0, d=1) | Strong: 0.5δ | ✓ |
| Two chargers far apart (d=15) | Negligible regardless of density | ✓ |
| One high-density (0.9), one empty (0.0), adjacent | Low: 0.05δ (high density justifies) | ✓ |

**QUBO placement:** Quadratic → **off-diagonal** (upper triangular, i < j):

```
Q_ij += α₄ × ( δ × (1 - max(nw_i, nw_j)) / (1 + d(i, j)) )
```

### 6.8 Constraint Term H5 — Exact Number of Chargers

```
H5 = λ × ( Σ_i x_i  -  m )²
```

**What it does:** Forces the solution to place exactly `m` new chargers — no more, no less.

**Why designed this way:**

- This is a **hard constraint disguised as a penalty.** By making `λ` sufficiently large (much larger than the other terms), any solution that doesn't have exactly `m` chargers becomes so expensive that the optimizer avoids it.

- **Expanding the square** (with careful handling of binary variables where `x_i² = x_i`):

```
(Σ_i x_i - m)² = (Σ_i x_i)² - 2m(Σ_i x_i) + m²

(Σ_i x_i)² = Σ_i x_i² + Σ_{i≠j} x_i x_j
            = Σ_i x_i + 2 Σ_{i<j} x_i x_j

Therefore:
H5 = λ × ( Σ_i x_i + 2 Σ_{i<j} x_i x_j - 2m Σ_i x_i + m² )
   = λ × Σ_i (1 - 2m) x_i + 2λ × Σ_{i<j} x_i x_j + λm²
```

Which gives:
  - Diagonal: `λ(1 - 2m)` per cell — this is negative for m ≥ 1, meaning it rewards selecting cells individually
  - Off-diagonal: `2λ` per pair — this penalizes every pair of selected cells, counterbalancing the diagonal reward
  - Constant: `λm²` — doesn't affect optimization, can be ignored

**Important: The off-diagonal coefficient is `2λ`, not `λ`.** This comes from the fact that `(Σ x_i)²` produces `Σ_{i≠j} x_i x_j`, which when converted to upper triangular form (i < j only) doubles the coefficient. Getting this wrong would make the constraint only half as strong on pairwise terms, potentially allowing the optimizer to select too many or too few chargers.

- The **`λ` value** must be chosen carefully. Too small: the optimizer violates the constraint. Too large: it dominates the objective and the optimizer ignores placement quality. A common heuristic is to set `λ` to be ~2-5× the magnitude of the largest other term.

**Verification:**

| Chargers selected | H5 value | Correct? |
|-------------------|----------|----------|
| Exactly m | λ × 0 = 0 | ✓ No penalty |
| m + 1 | λ × 1 = λ | ✓ Penalty |
| m - 1 | λ × 1 = λ | ✓ Same penalty |
| m + 2 | λ × 4 = 4λ | ✓ Quadratically increasing |

**QUBO placement:** Both **diagonal and off-diagonal**:

```
Q_ii += α₅ × λ × (1 - 2m)
Q_ij += α₅ × 2λ                  for i < j
```

### 6.9 Objective Term H6 — Coverage Redundancy (Minimize → Don't Waste Chargers on Same Cluster)

```
H6 = + Σ_i Σ_{j>i}  x_i × x_j × ε × Σ_{c ∈ C_POI, d(i,c)≤R, d(j,c)≤R}  w_c / ((1 + d(i,c)) × (1 + d(j,c)))
```

**What it does:** Penalizes placing two new chargers such that they both serve the same POI cluster, wasting coverage.

**Why this term is needed (and why H4 alone is not enough):**

H4 penalizes chargers being physically close to each other. H6 penalizes chargers serving the same POIs — these are related but different situations:

| Situation | H4 catches it? | H6 catches it? |
|-----------|----------------|----------------|
| Two chargers adjacent, both near same POI | ✓ (but weakly if high density) | ✓ |
| Two chargers far apart, but both equidistant to same POI cluster (e.g., on opposite sides) | ✗ (they're far from each other) | ✓ |
| Two chargers adjacent, serving different POI clusters | ✓ (unnecessary penalty) | ✗ (correctly no penalty) |

H6 catches the second case that H4 misses entirely, and avoids the false positive in the third case.

**Why designed this way:**

- The **product `1/((1+d(i,c)) × (1+d(j,c)))`** is large only when BOTH chargers are close to the same POI cell `c`. If one is close and the other is far, the product is small. This precisely captures "both serving the same area."

- **`w_c` weighting** means redundancy near high-weight POIs is penalized more (a bigger waste of resources than redundancy near low-weight POIs).

- **Distance cutoff `R`:** The sum only includes POI cells where BOTH chargers are within `R` cells. Without this cutoff, every pair of chargers would accumulate small penalty contributions from all POI cells across the entire grid, creating background noise that acts as a general "don't place any two chargers" penalty — which is not the intent. The cutoff keeps H6 focused on actual service redundancy. Only POI cells that a charger can realistically "serve" (within R cells) are considered. Default R = 3-4 cells; tunable.

- `ε` is the **redundancy penalty factor** — controls how strongly we penalize coverage overlap.

**Scenario tests:**

| Scenario | Behavior | Correct? |
|----------|----------|----------|
| Charger i at POI cell c (d=0), charger j adjacent to c (d=1) | Penalty: ε × w_c / (1×2) = ε×w_c/2 | ✓ High penalty |
| Charger i near cluster A (d=1), charger j near different cluster B (d=1), clusters far apart | Cluster A terms: small (j far from A). Cluster B terms: small (i far from B). Total: ~0 | ✓ No penalty |
| Both chargers at d=3 from same POI, R=3 | ε × w_c / (4×4) = ε×w_c/16 | ✓ Moderate penalty |
| Both chargers at d=3 from same POI, R=2 | Not included (d > R) → 0 | ✓ Cutoff works |

**QUBO placement:** Quadratic → **off-diagonal** (upper triangular, i < j):

```
Q_ij += α₆ × ( ε × Σ_{c ∈ C_POI, d(i,c)≤R, d(j,c)≤R}  w_c / ((1 + d(i,c)) × (1 + d(j,c))) )
```

### 6.10 Complete QUBO Formulation

The full objective to **minimize**:

```
H_final = α₁H₁ + α₂H₂ + α₃H₃ + α₄H₄ + α₅H₅ + α₆H₆
```

**Assembled Q matrix (upper triangular form):**

```
DIAGONAL (Q_ii) — encodes single-cell properties:

  Q_ii = α₁ × ( - Σ_{c ∈ C_POI}  w_c × s_c / (1 + d(i,c)) )          ← H1: POI attraction
       + α₂ × ( - β × g_i )                                             ← H2: gas station bonus
       + α₃ × ( γ × (1-nw_i) × Σ_{e ∈ C_existing} 1/(1+d(i,e)) )      ← H3: existing charger penalty
       + α₅ × λ × (1 - 2m)                                              ← H5: charger count (diagonal)

  where s_c = 1 / (1 + Σ_{e ∈ C_existing} 1/(1+d(c,e)))                 ← precomputed constant


OFF-DIAGONAL (Q_ij, i < j) — encodes pairwise interactions:

  Q_ij = α₄ × ( δ × (1 - max(nw_i, nw_j)) / (1 + d(i,j)) )            ← H4: new charger spacing
       + α₅ × 2λ                                                         ← H5: charger count (off-diagonal)
       + α₆ × ( ε × Σ_{c: d(i,c)≤R, d(j,c)≤R} w_c/((1+d(i,c))×(1+d(j,c))) )  ← H6: coverage redundancy
```

**How each algorithm uses Q:**

| Algorithm | How it uses Q |
|-----------|--------------|
| **QAOA** | Q is converted to a cost Hamiltonian (Ising model). QAOA circuit parameters are optimized to find x that minimizes x^T Q x. Multiple shots yield multiple candidate solutions. |
| **GA** | Given an individual [42, 117, 203], convert to binary vector x, compute fitness = x^T Q x. Lower value = better individual. Standard GA operators (selection, crossover, mutation) evolve the population toward lower fitness. |

### 6.11 Completeness Verification

Every requirement from our design discussions is covered by at least one term:

| Requirement | Primary Term | Supporting Term |
|-------------|-------------|-----------------|
| Place chargers near dense POIs | H1 (attraction) | — |
| Don't over-serve already-covered POIs | H1 (service gap s_c) | H6 (redundancy) |
| Prefer gas station cells (soft bonus) | H2 (bonus) | — |
| Avoid existing chargers in low-density areas | H3 (penalty) | H1 (reduced attraction via s_c) |
| Allow charger clustering in high-density areas | H3 (low penalty when nw high) | H4 (low penalty when nw high) |
| Spread chargers in low-density areas | H3 (high penalty when nw low) | H4 (high penalty when nw low) |
| Don't waste two chargers on same cluster | H6 (coverage redundancy) | H4 (spatial spread) |
| Place exactly m chargers | H5 (hard constraint) | — |
| All terms QUBO-compatible | ✓ All linear or quadratic in binary x | — |
| Works for both QAOA and GA | ✓ Single Q matrix shared | — |

No gaps identified. All terms are mutually complementary with no unintended cancellations.

### 6.12 Implementation Notes

**Q matrix sparsity:** For large grids (N = 1024), the full Q matrix has ~500K off-diagonal entries. However, H4 and H6 contributions decay with distance, so entries where `d(i,j) > R` (or a similar threshold) can be set to zero. Only the H5 off-diagonal term (`2λ`) applies to all pairs. In practice, the Q matrix can be stored as a sparse matrix with the H5 constant added implicitly during evaluation.

**Precomputation:** The following values should be computed once from grid data before building Q:
- `w_c` for all cells (Section 4.3 aggregation + scaling)
- `nw_c = w_c / scale_factor` for all cells
- `s_c` for all POI cells (service gap factor)
- `g_i` for all cells (gas station presence)
- Pairwise Chebyshev distances (or compute on-the-fly with distance cutoff)

### 6.13 Tunable Parameters Summary

| Parameter | Role | Default | Notes |
|-----------|------|---------|-------|
| `α₁` | Weight of H1 (POI attraction) | TBD | Primary driver — should be dominant |
| `α₂` | Weight of H2 (gas station bonus) | TBD | Meaningful but not dominant |
| `α₃` | Weight of H3 (existing charger penalty) | TBD | Moderate influence |
| `α₄` | Weight of H4 (new charger spacing) | TBD | Moderate influence |
| `α₅` | Weight of H5 (charger count constraint) | TBD | Must enforce constraint |
| `α₆` | Weight of H6 (coverage redundancy) | TBD | Moderate influence |
| `β` | Gas station bonus magnitude | TBD | Controls gas station attraction |
| `γ` | Existing charger penalty magnitude | TBD | Controls existing charger repulsion |
| `δ` | New charger spacing magnitude | TBD | Controls new charger mutual repulsion |
| `ε` | Coverage redundancy magnitude | TBD | Controls overlap penalty |
| `λ` | Constraint penalty multiplier | TBD | Heuristic: ~2-5× largest other term |
| `R` | Service radius for H6 cutoff | 3-4 cells | Defines "serving the same area" |
| `scale_factor` | Cell weight scaling (Section 4.3) | 5 | Controls weight differentiation |
| `min_weight` | Cell weight floor (Section 4.3) | 0.5 | Floor for lowest-density cells with POIs |

Note: `α` parameters and `β, γ, δ, ε` are technically redundant (e.g., `α₂ × β` could be a single parameter). They are kept separate for clarity — `α` values control the relative importance between objectives, while `β, γ, δ, ε` control magnitudes within each objective. During tuning, some may be collapsed.

### 6.14 Known Limitation — The min() Problem

The ideal scoring metric would use the minimum distance from each POI to its nearest charger: `Σ min_distance(poi, nearest_charger)`. However, the `min()` function cannot be expressed as a quadratic function of binary variables. This is the same limitation identified in the reference paper (Section IV-B).

Our H1 term approximates this by summing weighted inverse distances from ALL POI cells — this creates attraction toward POI-dense regions and correlates well with the true min-distance metric when chargers are reasonably distributed. The addition of H6 (coverage redundancy) further improves this approximation by discouraging the degenerate case where all chargers cluster at the single most attractive point. The GA refinement step after QAOA helps close any remaining gap between the QUBO proxy and the true objective.

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