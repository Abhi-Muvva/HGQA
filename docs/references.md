# HGQA References

Cross-checked against both `docs/` initial draft and `ALGO_RESEARCH/CITATION_MAP_FOR_PAPER.md`.
Every entry is individually verified (arXiv page fetched or confirmed via search).
Caveats and relevance notes are included — do not remove them.

---

## 1. Core QAOA Framework

### [QAOA] Farhi, Goldstone, Gutmann (2014)
**A Quantum Approximate Optimization Algorithm**
arXiv:1411.4028

_Cite for: the QAOA algorithm — alternating cost/mixer layers, variational parameters γ, β, p-layer circuit structure._

```
E. Farhi, J. Goldstone, S. Gutmann,
"A Quantum Approximate Optimization Algorithm,"
arXiv:1411.4028, 2014.
```

### [VQA] Cerezo et al. (2021)
**Variational Quantum Algorithms**
Nature Reviews Physics 3, 625–644 | DOI: 10.1038/s42254-021-00348-9

_Cite for: situating QAOA within the broader variational/NISQ algorithm family._

```
M. Cerezo et al.,
"Variational Quantum Algorithms,"
Nature Reviews Physics, vol. 3, pp. 625–644, 2021.
DOI: 10.1038/s42254-021-00348-9
```

---

## 2. QUBO Formulation and Ising Mapping

### [QUBO] Glover, Kochenberger, Du (2019)
**Quantum Bridge Analytics I: A Tutorial on Formulating and Using QUBO Models**
4OR 17, 335–371 | arXiv:1811.11538 | DOI: 10.1007/s10288-019-00424-y

_Cite for: the QUBO framework itself and the quadratic penalty encoding of constraints (H5 = λ(Σxᵢ − m)²)._

> **Note:** arXiv title is "A Tutorial on Formulating and Using QUBO Models"; the published journal version is titled "Quantum Bridge Analytics I." Same paper, same arXiv ID.

```
F. Glover, G. Kochenberger, Y. Du,
"Quantum Bridge Analytics I: A Tutorial on Formulating and Using QUBO Models,"
4OR, vol. 17, no. 4, pp. 335–371, 2019.
arXiv:1811.11538
```

### [ISING] Lucas (2014)
**Ising Formulations of Many NP Problems**
Frontiers in Physics 2:5 | arXiv:1302.5843 | DOI: 10.3389/fphy.2014.00005

_Cite for: the QUBO → Ising mapping via xᵢ = (1 − zᵢ)/2, performed explicitly in `qaoa_builder.py` lines 13–30 and 63–155._

```
A. Lucas,
"Ising Formulations of Many NP Problems,"
Frontiers in Physics, vol. 2, article 5, 2014.
DOI: 10.3389/fphy.2014.00005
```

---

## 3. XY-Ring Mixer (Hamming-Weight-Preserving)

### [XY-ANSATZ] Hadfield, Wang, O'Gorman, Rieffel, Venturelli, Biswas (2019)
**From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz**
Algorithms (MDPI) 12, 34 | arXiv:1709.03489 | DOI: 10.3390/a12020034

_Cite for: the foundational theoretical framework — XY-type mixers as feasibility-preserving operators that keep QAOA within the target Hamming-weight subspace. Establishes why H5 is unnecessary when using a structure-preserving mixer._

> **Note:** arXiv preprint from 2017; published in Algorithms (MDPI) in 2019 — cite with journal year 2019.

```
S. Hadfield, Z. Wang, B. O'Gorman, E. G. Rieffel, D. Venturelli, R. Biswas,
"From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz,"
Algorithms, vol. 12, no. 2, article 34, 2019.
arXiv:1709.03489
```

### [XY-ANALYSIS] Wang, Rubin, Dominy, Rieffel (2020)
**XY Mixers: Analytical and Numerical Results for QAOA**
Physical Review A 101, 012320 | arXiv:1904.09314 | DOI: 10.1103/PhysRevA.101.012320

_Cite for: analytical study of XY-mixer QAOA including ring vs. complete-graph topology comparisons and performance vs. the standard X-mixer._

> **Caveat:** This paper's application domain is graph coloring, not cardinality-constrained placement. The XY mixer's Hamming-weight preservation is a general property, but do not imply their numerical results transfer directly to our setting. Also note: their results show the complete-graph XY mixer outperforms the ring topology at finite p — this is relevant to our ring choice and should be acknowledged in the paper.

```
Z. Wang, N. C. Rubin, J. M. Dominy, E. G. Rieffel,
"XY Mixers: Analytical and Numerical Results for QAOA,"
Physical Review A, vol. 101, article 012320, 2020.
arXiv:1904.09314
```

### [CONSTRAINED-MIX] Fuchs, Lye, Nilsen, Stasik, Sartor (2022)
**Constrained Mixers for the Quantum Approximate Optimization Algorithm**
arXiv:2203.06095

_Cite for: formal framework for constructing mixers that preserve arbitrary constraint subspaces, with explicit coverage of XY-ring mixer for Hamming-weight/cardinality constraints and efficient Trotterized gate decompositions. Most directly relevant paper to our actual mixer design._

```
F. G. Fuchs, K. O. Lye, H. M. Nilsen, A. J. Stasik, G. Sartor,
"Constrained Mixers for the Quantum Approximate Optimization Algorithm,"
arXiv:2203.06095, 2022.
```

### [KVERTEX] Cook, Eidenbenz, Bärtschi (2019)
**The Quantum Alternating Operator Ansatz on Maximum k-Vertex Cover**
arXiv:1910.13483

_Cite for: the closest benchmark to our problem — a fixed-cardinality subset selection problem with hard constraints. Directly compares XY ring mixer vs. complete-graph XY mixer, and classical initial states vs. Dicke state initialization._

```
J. Cook, S. Eidenbenz, A. Bärtschi,
"The Quantum Alternating Operator Ansatz on Maximum k-Vertex Cover,"
in Proc. IEEE QCE 2020, pp. 83–92.
arXiv:1910.13483
```

---

## 4. Dicke State Initialization

> **⚠ TBD — current code does NOT prepare the exact Dicke state.**
>
> `_build_qaoa_xy_circuit()` applies X gates to `initial_positions`, producing one computational basis state with exactly m ones — not a uniform superposition over all C(K,m) bitstrings. `run_qaoa_mps()` randomizes those positions per restart.
>
> Add these citations only after implementing and verifying exact Dicke prep (equal amplitudes over all weight-m bitstrings, confirmed by statevector fidelity test on small N).

### [DICKE] Bärtschi, Eidenbenz (2019) — PENDING
**Deterministic Preparation of Dicke States**
FCT 2019, LNCS 11651, pp. 126–139 | arXiv:1904.07358 | DOI: 10.1007/978-3-030-25027-0_9

```
A. Bärtschi, S. Eidenbenz,
"Deterministic Preparation of Dicke States,"
in Proc. FCT 2019, LNCS vol. 11651, Springer, pp. 126–139, 2019.
arXiv:1904.07358
```

### [STATE-ALIGN] He, Shaydulin, Chakrabarti, Herman, Li, Sun, Pistoia (2023) — PENDING
**Alignment between Initial State and Mixer Improves QAOA Performance for Constrained Optimization**
npj Quantum Information 9, 121 (2023) | arXiv:2305.03857 | DOI: 10.1038/s41534-023-00787-5

_Cite when Dicke state is implemented: proves that setting the initial state to the mixer's ground state (Dicke state for complete-XY mixer) significantly improves constrained QAOA across all p depths, validated experimentally on a 32-qubit trapped-ion processor._

```
Z. He, R. Shaydulin, S. Chakrabarti, D. Herman, C. Li, Y. Sun, M. Pistoia,
"Alignment between Initial State and Mixer Improves QAOA Performance for Constrained Optimization,"
npj Quantum Information, vol. 9, article 121, 2023.
arXiv:2305.03857
```

---

## 5. MPS Simulation of QAOA

### [MPS-THEORY] Vidal (2003)
**Efficient Classical Simulation of Slightly Entangled Quantum Computations**
Physical Review Letters 91, 147902 | arXiv:quant-ph/0301063 | DOI: 10.1103/PhysRevLett.91.147902

_Cite for: the theoretical basis that quantum circuits with bounded bipartite entanglement can be classically simulated with MPS at fixed bond dimension — directly justifies our MPS truncation scheme (`truncation_threshold=1e-12`)._

```
G. Vidal,
"Efficient Classical Simulation of Slightly Entangled Quantum Computations,"
Physical Review Letters, vol. 91, article 147902, 2003.
arXiv:quant-ph/0301063
```

### [MPS-REVIEW] Orús (2014)
**A Practical Introduction to Tensor Networks: Matrix Product States and Projected Entangled Pair States**
Annals of Physics 349, 117–158 | arXiv:1306.2164 | DOI: 10.1016/j.aop.2014.06.013

_Cite for: bond dimension, SVD truncation, and computational complexity — directly maps to our `truncation_threshold` and `max_bond_dimension` parameters._

```
R. Orús,
"A Practical Introduction to Tensor Networks: Matrix Product States and Projected Entangled Pair States,"
Annals of Physics, vol. 349, pp. 117–158, 2014.
arXiv:1306.2164
```

### [MPS-QAOA] Feeney, Tate, Golden, Eidenbenz (2025)
**MPS-JuliQAOA: User-friendly, Scalable MPS-based Simulation for Quantum Optimization**
arXiv:2508.05883

_Cite for: explicit MPS-based QAOA simulation, scaling to 512 qubits. Most directly relevant paper to our `run_qaoa_mps()` approach. Demonstrates scalability and accuracy tradeoffs of bond-dimension-limited QAOA simulation._

> **Framing note:** This is a 2025 concurrent/parallel work, not a prior work that inspired HGQA. Frame accordingly in the paper (e.g., "independently, Feeney et al. demonstrate MPS-based QAOA simulation at larger scale").

```
S. Feeney, R. Tate, J. Golden, S. Eidenbenz,
"MPS-JuliQAOA: User-friendly, Scalable MPS-based Simulation for Quantum Optimization,"
arXiv:2508.05883, 2025.
```

> **Qiskit Aer MPS backend:** If using `AerSimulator(method="matrix_product_state")`, also cite the Qiskit Aer documentation directly:
> https://qiskit.github.io/qiskit-aer/tutorials/7_matrix_product_state_method.html

---

## 6. QUBO Variable Reduction (Three-Tier Pruning)

### [PRUNE] Lewis, Glover (2017)
**Quadratic Unconstrained Binary Optimization Problem Preprocessing: Theory and Empirical Analysis**
Networks 70, 79–97 | arXiv:1705.09844 | DOI: 10.1002/net.21751

_Cite for: the established QUBO preprocessing tradition — variable fixing rules that identify variables whose optimal value can be predetermined. Conceptual basis for Tier 1 and Tier 2._

> **Caveat:** Our three-tier design (dead-cell elimination, bound-based elimination using a greedy reference score, spatial BFS deduplication) is HGQA-specific. Lewis & Glover covers the broader preprocessing motivation; do not imply the specific tier rules are from this paper. Suggested wording: "Following the QUBO preprocessing tradition [Lewis & Glover 2017], we reduce the grid before QAOA using a novel three-tier pruning pipeline."

```
M. Lewis, F. Glover,
"Quadratic Unconstrained Binary Optimization Problem Preprocessing: Theory and Empirical Analysis,"
Networks, vol. 70, no. 2, pp. 79–97, 2017.
arXiv:1705.09844
```

### [PSEUDO-BOOL] Boros, Hammer (2002)
**Pseudo-Boolean Optimization**
Discrete Applied Mathematics 123, 155–225 | DOI: 10.1016/S0166-218X(01)00341-9

_Cite for: classical pseudo-Boolean optimization background including relaxation and bounding ideas. Supports the mathematical family that Tier 2 bound elimination belongs to. Secondary / optional._

```
E. Boros, P. L. Hammer,
"Pseudo-Boolean Optimization,"
Discrete Applied Mathematics, vol. 123, pp. 155–225, 2002.
DOI: 10.1016/S0166-218X(01)00341-9
```

---

## 7. EV Charging Station Placement (Domain)

### [EV-FORM] Lam, Leung, Chu (2014)
**Electric Vehicle Charging Station Placement: Formulation, Complexity, and Solutions**
arXiv:1310.6925

_Cite for: formal combinatorial formulation of the EVCSPP and proof of NP-hardness — directly justifies why heuristic/quantum approaches are needed._

```
A. Y. S. Lam, Y.-W. Leung, X. Chu,
"Electric Vehicle Charging Station Placement: Formulation, Complexity, and Solutions,"
arXiv:1310.6925, 2014.
```

### [EV-REVIEW] Kchaou-Boujelben (2021)
**Charging Station Location Problem: A Comprehensive Review on Models and Solution Approaches**
Transportation Research Part C 132, 103376 | DOI: 10.1016/j.trc.2021.103376

_Cite for: domain background — comprehensive survey of CSLP models and solution approaches. Motivates the importance and difficulty of the siting problem._

```
M. Kchaou-Boujelben,
"Charging Station Location Problem: A Comprehensive Review on Models and Solution Approaches,"
Transportation Research Part C, vol. 132, article 103376, 2021.
DOI: 10.1016/j.trc.2021.103376
```

### [EV-GA-1] Akbari, Brenna, Longo (2018)
**Optimal Locating of Electric Vehicle Charging Stations by Application of Genetic Algorithm**
Sustainability (MDPI) 10, 1076 | DOI: 10.3390/su10041076

_Cite for: classical GA precedent for EV charging station siting — predates quantum approaches._

```
M. Akbari, M. Brenna, M. Longo,
"Optimal Locating of Electric Vehicle Charging Stations by Application of Genetic Algorithm,"
Sustainability, vol. 10, no. 4, article 1076, 2018.
DOI: 10.3390/su10041076
```

### [EV-GA-2] Jordán, Palanca Cámara, Del Val Noguera, Julián Inglada, Botti (2021)
**Localization of Charging Stations for Electric Vehicles using Genetic Algorithms**
Neurocomputing 452, 416–423 | DOI: 10.1016/j.neucom.2019.11.122

_Cite for: GA-based EV station siting incorporating urban data (social networks, mobility) — supports the real-world constraint motivation for H1–H6._

> **Note:** DOI prefix (2019) reflects acceptance date; published volume is 2021.

```
J. Jordán, J. Palanca Cámara, E. Del Val Noguera, V. J. Julián Inglada, V. Botti,
"Localization of Charging Stations for Electric Vehicles using Genetic Algorithms,"
Neurocomputing, vol. 452, pp. 416–423, 2021.
DOI: 10.1016/j.neucom.2019.11.122
```

### [EV-QA-GA] Chandra, Lalwani, Jajodia (2022)
**Towards an Optimal Hybrid Algorithm for EV Charging Stations Placement using Quantum Annealing and Genetic Algorithms**
IEEE TQCEBT 2022 | arXiv:2111.01622 | DOI: 10.1109/TQCEBT54229.2022.10041464

_Cite for: closest direct prior work — also solves EV placement with a quantum + GA hybrid, improving over vanilla QA by 42.89% on POI proximity._

> **Key distinction:** Uses D-Wave quantum annealing (not gate-based QAOA), no XY mixer, no Dicke state, no MPS simulation, no variable reduction. Make this distinction explicit in the related work section.

```
A. Chandra, J. Lalwani, B. Jajodia,
"Towards an Optimal Hybrid Algorithm for EV Charging Stations Placement
using Quantum Annealing and Genetic Algorithms,"
in Proc. IEEE TQCEBT, 2022.
arXiv:2111.01622
```

### [EV-BILEVEL] Piedra-de-la-Cuadra, Ortega (2024)
**Bilevel Optimization for the Deployment of Refuelling Stations for Electric Vehicles on Road Networks**
Computers & Operations Research 162, 106460 | DOI: 10.1016/j.cor.2023.106460

_Cite for: motivation behind H2 — treating existing gas-station infrastructure as candidate co-location sites for EV chargers. The bilevel model minimizes number of required refueling points under reinforced coverage service constraints._

> **Caveat:** The paper focuses on road-network coverage, not grid-based QUBO formulation. Cite only for the real-world motivation that gas/refuelling stations are natural EV charger candidates (H2's co-location bonus).

```
R. Piedra-de-la-Cuadra, F. Ortega,
"Bilevel Optimization for the Deployment of Refuelling Stations for Electric Vehicles on Road Networks,"
Computers & Operations Research, vol. 162, article 106460, 2024.
DOI: 10.1016/j.cor.2023.106460
```

---

## 8. Hybrid QAOA → GA

### [EV-QA-GA] Chandra et al. (2022)
_(Already listed above — same paper serves dual purpose: EV domain + quantum-seeded GA methodology.)_

### [GA] Goldberg (1989)
**Genetic Algorithms in Search, Optimization, and Machine Learning**
Addison-Wesley | ISBN: 0-201-15767-5

_Cite for: tournament selection and elitist selection — both are described in this book. **Do not cite for uniform crossover** — that is Syswerda 1989 (see [UNIFORM-XO] below)._

```
D. E. Goldberg,
Genetic Algorithms in Search, Optimization, and Machine Learning,
Addison-Wesley, Reading, MA, 1989.
```

### [UNIFORM-XO] Syswerda (1989)
**Uniform Crossover in Genetic Algorithms**
in Proc. ICGA 1989, pp. 2–9

_Cite for: uniform crossover — the specific crossover operator used in `ga_solver.py` `_crossover()`. Syswerda (1989) introduced and analyzed uniform crossover; our popcount-repair variant builds directly on this design. Goldberg 1989 does not introduce uniform crossover._

```
G. Syswerda,
"Uniform Crossover in Genetic Algorithms,"
in Proc. 3rd Int. Conf. on Genetic Algorithms (ICGA), pp. 2–9, 1989.
```

---

## 9. Shallow QAOA Limitations (Discussion / Framing Negative Results)

These citations are for the discussion section — they explain why shallow/fixed-depth QAOA underperformance is a known, scientifically documented phenomenon and not a flaw in the formulation.

### [REACHABILITY] Akshay, Philathong, Zacharov, Biamonte (2021)
**Reachability Deficits in Quantum Approximate Optimization of Graph Problems**
Quantum 5, 532 (2021) | arXiv:2007.09148 | DOI: 10.22331/q-2021-08-19-532

_Cite for: problem constraint density acts as a performance indicator — higher density degrades fixed-depth QAOA quality. Relevant because our QUBO is dense (H4, H6 create many off-diagonal interactions)._

> **Note:** Published in *Quantum* (vol. 5, p. 532, 2021), not PRL. There is a separate Akshay/Biamonte PRL paper (PRL 124, 090504, 2020, arXiv:1906.11259) with different co-authors — do not conflate.

```
V. Akshay, H. Philathong, I. Zacharov, J. Biamonte,
"Reachability Deficits in Quantum Approximate Optimization of Graph Problems,"
Quantum, vol. 5, p. 532, 2021.
DOI: 10.22331/q-2021-08-19-532
arXiv:2007.09148
```

### [TRAINABILITY] Rajakumar, Golden, Bärtschi, Eidenbenz (2024)
**Trainability Barriers in Low-Depth QAOA Landscapes**
ACM Computing Frontiers 2024 | arXiv:2402.10188

_Cite for: superpolynomial growth of poor local minima in QAOA landscapes — explains why random-initialization COBYLA restarts fail to find good parameters consistently._

```
J. Rajakumar, J. Golden, A. Bärtschi, S. Eidenbenz,
"Trainability Barriers in Low-Depth QAOA Landscapes,"
in Proc. ACM Computing Frontiers, 2024.
arXiv:2402.10188
```

### [SYMMETRY] Bravyi, Kliesch, Koenig, Tang (2020)
**Obstacles to State Preparation and Variational Optimization from Symmetry Protection**
Physical Review Letters 125, 260505 | arXiv:1910.08980 | DOI: 10.1103/PhysRevLett.125.260505

_Cite for: fundamental fixed-depth locality barriers in variational quantum algorithms (secondary citation — Akshay et al. is more directly applicable to our dense-constraint setting)._

```
S. Bravyi, A. Kliesch, R. Koenig, E. Tang,
"Obstacles to State Preparation and Variational Optimization from Symmetry Protection,"
Physical Review Letters, vol. 125, article 260505, 2020.
arXiv:1910.08980
```

---

## 10. Future Work / Parameter Optimization

These are cited only in a "future work" or "limitations" context.

### [PARAM-HEUR] Zhou, Wang, Choi, Pichler, Lukin (2020)
**Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices**
Physical Review X 10, 021067 | arXiv:1812.01041

_Cite for: parameter initialization heuristics (INTERP/FOURIER) that scale QAOA parameters across p levels, reducing the outer-loop optimization burden. Relevant to improving our random-restart COBYLA approach._

```
L. Zhou, S.-T. Wang, S. Choi, H. Pichler, M. D. Lukin,
"Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices,"
Physical Review X, vol. 10, article 021067, 2020.
arXiv:1812.01041
```

### [PARAM-TRANSFER] Brandao, Broughton, Farhi, Gutmann, Neven (2018)
**For Fixed Control Parameters the QAOA Objective Function Value Concentrates for Typical Instances**
arXiv:1812.04170

_Cite for: QAOA objective concentration — fixed parameters generalize across typical instances of the same problem class. Theoretical basis for parameter transfer across datasets._

```
F. G. S. L. Brandao, M. Broughton, E. Farhi, S. Gutmann, H. Neven,
"For Fixed Control Parameters the Quantum Approximate Optimization Algorithm's Objective
Function Value Concentrates for Typical Instances,"
arXiv:1812.04170, 2018.
```

### [WARM-START] Egger, Mareček, Woerner (2021)
**Warm-Starting Quantum Optimization**
Quantum 5, 479 | arXiv:2009.10095 | DOI: 10.22331/q-2021-06-17-479

_Cite ONLY in a "future directions" context: using a classical relaxation solution to initialize QAOA parameters (opposite direction to what we currently do — we warm GA with QAOA, not QAOA with classical)._

> **Do not cite as support for the current QAOA→GA seeding.** The directions are reversed.

```
D. J. Egger, J. Mareček, S. Woerner,
"Warm-Starting Quantum Optimization,"
Quantum, vol. 5, article 479, 2021.
arXiv:2009.10095
```

---

## Dropped Citations and Reasons

| Citation | Reason |
|---|---|
| Farhi et al. arXiv:1412.6062 | Specific to Max E3LIN2 approximation guarantee; not relevant to EV placement |
| Bärtschi & Eidenbenz 2020 (Grover mixers) | We use XY mixer, not Grover mixer — different circuit structure |
| White 1992 (DMRG) | Condensed matter physics; not the MPS quantum circuit simulation we use |
| Vidal 2004 (TEBD) | Time-evolution of MPS; we do static circuit simulation |
| Lykov et al. 2023 arXiv:2309.04841 | **NOT MPS simulation** — uses precomputed diagonal Hamiltonian; keep only as future non-MPS simulator redesign |
| Mugel et al. 2022 (portfolio, PRR) | Portfolio optimization; methodology overlap too weak to justify |
| Rao & Sodhi 2022 (Soft Computing EV) | D-Wave annealing, redundant with Chandra et al. already cited |
| Glover, Lewis, Kochenberger 2018 (EJOR) | Logical implication rules; less aligned with our bound-based approach than Lewis & Glover 2017 |
| Holland 1975 | Goldberg 1989 covers our specific operators and is sufficient |
| Choi 2008 (minor-embedding) | Optional at best; minor-embedding is for annealing hardware, not gate-based QAOA pruning |

---

## Pre-Submission Checklist

- [ ] **Dicke state**: implement exact prep → add [DICKE] and [STATE-ALIGN]
- [x] **Bilevel EV paper**: confirm exact authors and full journal citation
- [ ] **Ring vs. complete-graph XY mixer**: acknowledge in paper that Wang et al. 2020 shows complete-graph outperforms ring — justify ring choice (hardware connectivity, gate efficiency)
- [ ] **MPS backend**: confirm whether using Qiskit Aer MPS or custom implementation — add Qiskit Aer docs citation if former
- [ ] **Lykov 2023**: remove from any MPS-related citations — it is not an MPS paper
- [ ] **Negative QAOA results**: use [REACHABILITY] and [TRAINABILITY] in discussion to frame results scientifically
- [ ] **H1–H6 novelty**: explicitly state in paper that the six-term objective is HGQA-specific; cite [QUBO], [ISING], [EV-FORM], and [EV-REVIEW] only for the surrounding formulation tradition

## Publication-Safe Novelty Claims (from CITATION_MAP analysis)

**Supported by code:**
- QUBO-aware three-tier pruning that reduces EVCP grid variables before QAOA
- Multi-term QUBO with service-gap saturation (H1) and coverage redundancy (H6)
- XY-ring mixer replacing explicit cardinality penalty (H5-free MPS branch)
- QAOA/greedy/random seeding comparison under one shared QUBO objective

**Do not claim until verified:**
- "We prepare the exact Dicke state" — current code uses a random computational basis state with m ones
- "QAOA outperforms GA" — current results suggest the opposite
- "Tier 3 pruning is optimality-preserving" — it is explicitly heuristic
- "MPS demonstrates quantum advantage" — it is a classical simulator
