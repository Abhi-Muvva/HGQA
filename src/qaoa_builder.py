"""
QAOA Builder Module
====================
Baseline reference: Section — Algorithm Pipeline, Phase 2 (Quantum Optimization)

Two-layer architecture:
  LAYER 1 — Pure math (no Qiskit):  Ising conversion, evaluation, buffer
  LAYER 2 — Qiskit-dependent:       Hamiltonian construction, QAOA circuit, sampling

Layer 1 can be imported and tested independently for verification.
Layer 2 imports Qiskit lazily (only when called).

QUBO → Ising mapping
---------------------
  QUBO variable:  x_i ∈ {0, 1}
  Ising variable: z_i ∈ {-1, +1}
  Substitution:   x_i = (1 - z_i) / 2

  Diagonal Q[(i,i)]:
    Q_ii × x_i  =  Q_ii/2            (constant)
                  − Q_ii/2 × Z_i     (single-Z)

  Off-diagonal Q[(i,j)], i < j:
    Q_ij × x_i × x_j  =  Q_ij/4                   (constant)
                         − Q_ij/4 × Z_i             (single-Z)
                         − Q_ij/4 × Z_j             (single-Z)
                         + Q_ij/4 × Z_i Z_j         (two-Z)

  So:  QUBO_energy(x)  =  Ising_energy(z) + offset
  where offset collects all constants discarded during conversion.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Tuple, Optional


# ===========================================================================
# LAYER 1 — Pure math (no Qiskit dependency)
# ===========================================================================

def compute_buffer(m: int, use_buffer: bool = True) -> int:
    """
    Compute output buffer size — how many extra candidates to return
    beyond the m chargers requested.

    Parameters
    ----------
    m          : number of new chargers to place
    use_buffer : if False, returns 0 (exact m solutions only)

    Returns
    -------
    int — buffer size in [0, 5]
    """
    if not use_buffer:
        return 0
    return min(max(1, math.ceil(m / 2)), 5)


def qubo_to_ising_coeffs(
    Q_obj: Dict[Tuple[int, int], float],
    N: int,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """
    Convert sparse Q_obj (H1–H4, H6) to Ising coefficients.

    Parameters
    ----------
    Q_obj : sparse QUBO dict {(i,j): float}, i <= j
    N     : number of qubits / grid cells

    Returns
    -------
    h      : dict {qubit_index: coeff} — single-Z coefficients
    J      : dict {(i,j): coeff}, i < j — two-body ZZ coefficients
    offset : float — constant energy offset (add to Ising energy to recover QUBO energy)
    """
    h = {}
    J = {}
    offset = 0.0

    for (i, j), val in Q_obj.items():
        if val == 0.0:
            continue

        if i == j:
            # Q_ii × x_i = Q_ii/2  −  Q_ii/2 × Z_i
            offset += val / 2.0
            h[i] = h.get(i, 0.0) - val / 2.0
        else:
            # Q_ij × x_i x_j = Q_ij/4  −  Q_ij/4 Z_i  −  Q_ij/4 Z_j  +  Q_ij/4 Z_iZ_j
            # Ensure i < j for J storage
            a, b = (i, j) if i < j else (j, i)
            offset += val / 4.0
            h[a] = h.get(a, 0.0) - val / 4.0
            h[b] = h.get(b, 0.0) - val / 4.0
            J[(a, b)] = J.get((a, b), 0.0) + val / 4.0

    return h, J, offset


def h5_to_ising_coeffs(
    h5_params: Dict,
    N: int,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """
    Convert H5 constraint term to Ising coefficients.

    H5 = α5 × λ × (Σ x_i − m)²

    After QUBO expansion (baseline Section 6, H5 derivation):
      Diagonal per cell i:    α5 × λ × (1 − 2m)
      Off-diagonal per pair:  α5 × 2λ           ← the factor of 2 is critical
      Constant:               α5 × λ × m²

    Parameters
    ----------
    h5_params : dict with keys 'lam', 'alpha5', 'm'
    N         : number of qubits / grid cells

    Returns
    -------
    h, J, offset — same format as qubo_to_ising_coeffs
    """
    lam    = h5_params['lam']
    alpha5 = h5_params['alpha5']
    m      = h5_params['m']

    diag_coeff    = alpha5 * lam * (1 - 2 * m)     # per cell
    offdiag_coeff = alpha5 * 2.0 * lam              # per pair (i<j)
    h5_constant   = alpha5 * lam * m * m            # from m² in expansion

    h = {}
    J = {}
    offset = h5_constant

    # Diagonal contributions: diag_coeff × x_i for each cell
    for i in range(N):
        # diag_coeff × x_i = diag_coeff/2  −  diag_coeff/2 × Z_i
        offset += diag_coeff / 2.0
        h[i] = h.get(i, 0.0) - diag_coeff / 2.0

    # Off-diagonal contributions: offdiag_coeff × x_i × x_j for each pair
    for i in range(N):
        for j in range(i + 1, N):
            # offdiag_coeff × x_i x_j = offdiag_coeff/4 × (1 - Z_i - Z_j + Z_iZ_j)
            offset += offdiag_coeff / 4.0
            h[i] = h.get(i, 0.0) - offdiag_coeff / 4.0
            h[j] = h.get(j, 0.0) - offdiag_coeff / 4.0
            J[(i, j)] = J.get((i, j), 0.0) + offdiag_coeff / 4.0

    return h, J, offset


def merge_ising_coeffs(
    coeffs_list: List[Tuple[Dict, Dict, float]],
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """
    Merge multiple (h, J, offset) tuples into one combined Ising model.

    Parameters
    ----------
    coeffs_list : list of (h, J, offset) tuples

    Returns
    -------
    h_total, J_total, offset_total
    """
    h_total = {}
    J_total = {}
    offset_total = 0.0

    for h, J, offset in coeffs_list:
        offset_total += offset
        for k, v in h.items():
            h_total[k] = h_total.get(k, 0.0) + v
        for k, v in J.items():
            J_total[k] = J_total.get(k, 0.0) + v

    return h_total, J_total, offset_total


def evaluate_ising(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    offset: float,
    bitstring: List[int],
) -> float:
    """
    Evaluate Ising energy for a computational basis state.

    Converts bitstring x ∈ {0,1}^N to z ∈ {-1,+1}^N via z_i = 1 − 2x_i,
    then computes:
      E = offset + Σ_i h_i z_i + Σ_{i<j} J_ij z_i z_j

    The result equals the QUBO energy f(x) = x^T Q x (including H5 if
    H5 Ising coefficients are merged in).

    Parameters
    ----------
    h         : single-Z coefficients
    J         : two-body ZZ coefficients
    offset    : constant from Ising conversion
    bitstring : list of 0/1 values, length N

    Returns
    -------
    float — energy (equals QUBO energy for the corresponding x)
    """
    z = [1 - 2 * x for x in bitstring]

    energy = offset
    for i, coeff in h.items():
        energy += coeff * z[i]
    for (i, j), coeff in J.items():
        energy += coeff * z[i] * z[j]

    return energy


def bitstring_from_solution(solution: List[int], N: int) -> List[int]:
    """Convert a list of selected cell IDs to a binary bitstring of length N."""
    bits = [0] * N
    for cell_id in solution:
        bits[cell_id] = 1
    return bits


def solution_from_integer(integer: int, N: int) -> List[int]:
    """
    Convert a measurement integer to a list of selected cell IDs.

    Qiskit integers are little-endian: bit 0 of the integer = qubit 0 = cell 0.

    Parameters
    ----------
    integer : measurement outcome as int
    N       : number of qubits

    Returns
    -------
    list of cell IDs where the corresponding bit is 1
    """
    return [i for i in range(N) if (integer >> i) & 1]


# ===========================================================================
# LAYER 2 — Qiskit-dependent functions
# ===========================================================================
# These import qiskit lazily so the module can be imported for Layer 1
# testing even without qiskit installed.

def ising_to_sparse_pauli_op(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    offset: float,
    N: int,
):
    """
    Convert Ising coefficients to a Qiskit SparsePauliOp on N qubits.

    Builds Pauli strings:
      - "III...I"  with coeff = offset   (identity, constant term)
      - "III..Z..I" with coeff = h[i]    (Z on qubit i)
      - "II..Z..Z..I" with coeff = J[(i,j)]  (ZZ on qubits i,j)

    Qiskit Pauli string ordering: rightmost character = qubit 0.
    So to place Z on qubit k in an N-qubit string:
      label[N-1-k] = 'Z', rest = 'I'

    Parameters
    ----------
    h, J, offset : from qubo_to_ising_coeffs / h5_to_ising_coeffs / merge
    N            : number of qubits

    Returns
    -------
    SparsePauliOp
    """
    from qiskit.quantum_info import SparsePauliOp

    labels = []
    coeffs = []

    # Constant (identity) term
    if offset != 0.0:
        labels.append('I' * N)
        coeffs.append(offset)

    # Single-Z terms
    for i, coeff in h.items():
        if coeff == 0.0:
            continue
        label = ['I'] * N
        label[N - 1 - i] = 'Z'  # Qiskit: rightmost = qubit 0
        labels.append(''.join(label))
        coeffs.append(coeff)

    # ZZ terms
    for (i, j), coeff in J.items():
        if coeff == 0.0:
            continue
        label = ['I'] * N
        label[N - 1 - i] = 'Z'
        label[N - 1 - j] = 'Z'
        labels.append(''.join(label))
        coeffs.append(coeff)

    return SparsePauliOp.from_list(list(zip(labels, coeffs))).simplify()


def build_cost_hamiltonian(
    Q_obj: Dict,
    h5_params: Dict,
    N: int,
):
    """
    Build the full cost Hamiltonian as a SparsePauliOp.

    Combines:
      - Q_obj (H1–H4, H6) → Ising
      - H5 (constraint)    → Ising
      - Merged → single SparsePauliOp

    Parameters
    ----------
    Q_obj     : sparse QUBO dict from build_qubo()
    h5_params : dict from build_qubo()
    N         : number of qubits

    Returns
    -------
    SparsePauliOp — total cost Hamiltonian (objective + constraint)
    """
    obj_coeffs = qubo_to_ising_coeffs(Q_obj, N)
    h5_coeffs  = h5_to_ising_coeffs(h5_params, N)
    h_total, J_total, offset_total = merge_ising_coeffs([obj_coeffs, h5_coeffs])

    return ising_to_sparse_pauli_op(h_total, J_total, offset_total, N)


def build_total_ising_coeffs(
    Q_obj: Dict,
    h5_params: Dict,
    N: int,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """
    Build merged Ising coefficients for the full cost function.

    Returns the same Hamiltonian as build_cost_hamiltonian(), but in
    coefficient form so shot-based backends can estimate energies from counts.
    """
    obj_coeffs = qubo_to_ising_coeffs(Q_obj, N)
    h5_coeffs = h5_to_ising_coeffs(h5_params, N)
    return merge_ising_coeffs([obj_coeffs, h5_coeffs])


def counts_to_energy(
    counts: Dict[str, int],
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    offset: float,
    N: int,
) -> float:
    """
    Estimate <H> from shot counts using the full Ising energy.
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return float("inf")

    energy = 0.0
    for bitstring, count in counts.items():
        integer = int(bitstring.replace(" ", ""), 2)
        bits = [(integer >> i) & 1 for i in range(N)]
        energy += (count / total_shots) * evaluate_ising(h, J, offset, bits)
    return energy


def run_qaoa(
    Q_obj: Dict,
    h5_params: Dict,
    N: int,
    m: int,
    p: int = 3,
    max_iter: int = 300,
    use_buffer: bool = True,
    n_restarts: int = 3,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[Tuple[float, List[int], float]], np.ndarray]:
    """
    QAOA optimization via exact statevector simulation.

    Finds optimal variational parameters, then extracts the exact statevector
    to read off every basis state's probability.  Returns ranked solutions AND
    the optimal parameters (for optional noisy re-run via run_qaoa_noisy).

    Parameters
    ----------
    Q_obj      : sparse QUBO dict from build_qubo()
    h5_params  : dict from build_qubo()
    N          : number of qubits / grid cells
    m          : number of new chargers to place
    p          : QAOA circuit depth (reps), default 3
    max_iter   : COBYLA max iterations per restart, default 300
    use_buffer : include buffer in output count, default True
    n_restarts : number of COBYLA restarts (best wins), default 3
    seed       : random seed for reproducibility (None = random)
    verbose    : print progress, default True

    Returns
    -------
    (results, optimal_params)
      results       : List of (score, [cell_ids], probability) sorted ascending.
      optimal_params: np.ndarray — QAOA variational parameters.
    """
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from scipy.optimize import minimize as scipy_minimize
    from qubo_builder import evaluate_solution

    top_k = m + compute_buffer(m, use_buffer)

    if verbose:
        print(f"QAOA Pipeline: N={N}, m={m}, p={p}, "
              f"restarts={n_restarts}, max_iter={max_iter}, top_k={top_k}")

    # ── Build cost Hamiltonian ──
    cost_op = build_cost_hamiltonian(Q_obj, h5_params, N)
    if verbose:
        print(f"  Cost Hamiltonian: {len(cost_op)} Pauli terms")

    # ── Strip constant offset for optimizer ──
    identity_label = 'I' * N
    offset = 0.0
    non_identity_terms = []
    for label, coeff in cost_op.to_list():
        if label == identity_label:
            offset += coeff.real
        else:
            non_identity_terms.append((label, coeff))

    if non_identity_terms:
        cost_op_shifted = SparsePauliOp.from_list(non_identity_terms).simplify()
    else:
        cost_op_shifted = SparsePauliOp.from_list([(identity_label, 0.0)])

    if verbose:
        print(f"  Stripped offset: {offset:.4f}  "
              f"(optimizer sees {len(cost_op_shifted)} terms, zero-centered)")

    # ── Build and decompose QAOA circuit ──
    circuit = QAOAAnsatz(cost_op, reps=p).decompose(reps=3)
    n_params = circuit.num_parameters
    if verbose:
        print(f"  QAOA circuit: {n_params} parameters (2×p = 2×{p}), "
              f"{circuit.size()} gates after decomposition")

    # ── Variational optimization with restarts ──
    estimator = StatevectorEstimator()
    rng = np.random.default_rng(seed)

    best_result = None
    best_cost = float('inf')

    for restart in range(n_restarts):
        init_params = rng.uniform(-np.pi, np.pi, n_params)

        eval_count = 0
        def cost_fn(params):
            nonlocal eval_count
            eval_count += 1
            pub = (circuit, cost_op_shifted, params)
            job = estimator.run([pub])
            return float(job.result()[0].data.evs)

        result = scipy_minimize(
            cost_fn,
            init_params,
            method='COBYLA',
            options={'maxiter': max_iter},
        )

        if verbose:
            print(f"  Restart {restart+1}/{n_restarts}: "
                  f"cost={result.fun:.4f} (shifted), "
                  f"raw={result.fun + offset:.4f}, "
                  f"evals={eval_count}")

        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result

    if verbose:
        print(f"  Best shifted cost: {best_cost:.4f}  "
              f"(raw: {best_cost + offset:.4f})")

    # ── Extract exact statevector ──
    final_circuit = circuit.assign_parameters(best_result.x)
    sv = Statevector.from_instruction(final_circuit)
    probs = sv.probabilities()

    if verbose:
        total_prob_feasible = sum(
            probs[i] for i in range(2**N)
            if bin(i).count('1') == m
        )
        print(f"  Probability on feasible states (popcount={m}): "
              f"{total_prob_feasible:.4f} ({total_prob_feasible*100:.1f}%)")

    # ── Score feasible basis states ──
    scored = []
    for integer in range(2**N):
        prob = probs[integer]
        if prob < 1e-12:
            continue
        solution = solution_from_integer(integer, N)
        if len(solution) == m:
            score = evaluate_solution(Q_obj, solution)
            scored.append((score, sorted(solution), prob))

    scored.sort(key=lambda x: x[0])

    seen = set()
    results = []
    for score, solution, prob in scored:
        key = tuple(solution)
        if key not in seen:
            seen.add(key)
            results.append((score, solution, prob))

    results = results[:top_k]

    if verbose:
        print(f"  Unique feasible solutions: {len(scored)}")
        print(f"  Returning top {len(results)}")

    return results, best_result.x



# ===========================================================================
# Dicke state preparation — Bärtschi & Eidenbenz (2019), arXiv:1904.07358
#
# Prepares |D(K,m)⟩ = 1/√C(K,m) Σ_{|x|=m} |x⟩, the uniform superposition
# over all K-qubit strings with exactly m ones.  This is the ground state of
# the XY ring mixer and the provably optimal initial state for XY-QAOA on
# cardinality-constrained problems (Wang et al. 2020, He et al. 2023).
#
# Gate counts:  O(m·K) two-qubit gates,  O(K) depth.
# Max qubit span of any gate: m  (highly MPS-compatible; bond dim ≤ m+1).
# ===========================================================================

def _dicke_gate_ry_i(n: int) -> "QuantumCircuit":
    """Gate (i) from §2.2 of arXiv:1904.07358 — 2-qubit split rotation."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.append(RYGate(2.0 * np.arccos(np.sqrt(1.0 / n))).control(ctrl_state="1"), [1, 0])
    qc.cx(0, 1)
    return qc


def _dicke_gate_ry_ii(l: int, n: int) -> "QuantumCircuit":
    """Gate (ii)_l from §2.2 of arXiv:1904.07358 — 3-qubit split rotation."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    qc.append(
        RYGate(2.0 * np.arccos(np.sqrt(float(l) / n))).control(
            num_ctrl_qubits=2, ctrl_state="11"
        ),
        [2, 1, 0],
    )
    qc.cx(0, 2)
    return qc


def _dicke_scs(n: int, k: int) -> "QuantumCircuit":
    """SCS_{n,k} gate (Definition 3, arXiv:1904.07358) on k+1 qubits."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(k + 1)
    qc.append(_dicke_gate_ry_i(n), [k - 1, k])
    for l in range(2, k + 1):
        qc.append(_dicke_gate_ry_ii(l, n), [k - l, k - l + 1, k])
    return qc


def _dicke_block1(n: int, k: int, l: int) -> "QuantumCircuit":
    """First-product block from Lemma 2 of arXiv:1904.07358."""
    from qiskit import QuantumCircuit, QuantumRegister
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)
    first = l - k - 1
    last  = n - l
    index = list(range(n))
    if first:
        index = index[first:]
    if last:
        index = index[:-last]
    qc.append(_dicke_scs(l, k), index)
    return qc


def _dicke_block2(n: int, k: int, l: int) -> "QuantumCircuit":
    """Second-product block from Lemma 2 of arXiv:1904.07358."""
    from qiskit import QuantumCircuit, QuantumRegister
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)
    last  = n - l
    index = list(range(n))
    if last:
        index = index[:-last]
    qc.append(_dicke_scs(l, l - 1), index)
    return qc


def _dicke_state_circuit(K: int, m: int) -> "QuantumCircuit":
    """
    Prepare Dicke state |D(K,m)⟩ on K qubits.

    Uses the Bärtschi-Eidenbenz algorithm (arXiv:1904.07358, Lemma 2).
    O(m·K) two-qubit gates; max gate span = m (MPS bond dim stays ≤ m+1).

    Parameters
    ----------
    K : number of qubits
    m : Hamming weight (number of ones)

    Returns
    -------
    QuantumCircuit (no measurements, no classical bits)
    """
    from qiskit import QuantumCircuit, QuantumRegister
    qr = QuantumRegister(K)
    qc = QuantumCircuit(qr)
    if m == 0:
        return qc
    if m == K:
        qc.x(qr)
        return qc
    qc.x(qr[-m:])                              # |0^{K-m} 1^m⟩ starting state
    for l in range(m + 1, K + 1)[::-1]:        # first product in Lemma 2
        qc.append(_dicke_block1(K, m, l), range(K))
    for l in range(2, m + 1)[::-1]:            # second product in Lemma 2
        qc.append(_dicke_block2(K, m, l), range(K))
    return qc


def _build_qaoa_xy_circuit(
    K: int,
    m: int,
    p: int,
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
) -> "QuantumCircuit":
    """
    QAOA circuit with XY ring mixer and Dicke state initialisation.

    Initial state : |D(K,m)⟩ — uniform superposition over all weight-m states.
                    Ground state of the XY mixer; provably optimal start for
                    cardinality-constrained XY-QAOA (arXiv:1904.07358, 2305.03857).
    Cost layer    : Rz / Rzz rotations from h and J (no H5)
    Mixer layer   : ring XY = Σ_{i=0}^{K-2} RXX(2β) RYY(2β) on pairs (i, i+1)

    The XY mixer swaps |01⟩↔|10⟩ while leaving |00⟩ and |11⟩ unchanged,
    so Hamming weight is preserved throughout. All sampled bitstrings have
    exactly m ones — H5 penalty is never needed.

    Parameters
    ----------
    K, m, p : qubits, target ones, QAOA depth
    h       : {qubit: Ising-Z coeff}
    J       : {(i, j): Ising-ZZ coeff}, i < j

    Returns
    -------
    QuantumCircuit with 2*p parameters θ[0..p-1] (gamma) and θ[p..2p-1] (beta),
    plus K measurement gates.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    theta = ParameterVector('θ', 2 * p)   # θ[0..p-1]=gamma, θ[p..2p-1]=beta

    qc = QuantumCircuit(K, K)

    # Initial state: Dicke state |D(K,m)⟩ — ground state of the XY mixer.
    # This replaces the previous single-basis-state (random X-gate) initialisation,
    # which broke adiabatic alignment and caused the QAOA-vs-GA quality gap.
    qc.compose(_dicke_state_circuit(K, m), inplace=True)

    for layer in range(p):
        gamma_l = theta[layer]
        beta_l  = theta[p + layer]

        # Cost layer: exp(-i γ H_cost)
        # Single-Z:  exp(-i γ h_i Z_i) = Rz(2 γ h_i, i)
        for i, coeff in h.items():
            if coeff != 0.0:
                qc.rz(2.0 * coeff * gamma_l, i)
        # ZZ pairs:  exp(-i γ J_{ij} Z_i Z_j) = Rzz(2 γ J_{ij}, i, j)
        for (i, j), coeff in J.items():
            if coeff != 0.0:
                qc.rzz(2.0 * coeff * gamma_l, i, j)

        # XY ring mixer: exp(-i β (XX+YY)_{i,i+1}) for each neighbouring pair
        # XX and YY commute → product = joint exponential
        for i in range(K - 1):
            qc.rxx(2.0 * beta_l, i, i + 1)
            qc.ryy(2.0 * beta_l, i, i + 1)

    qc.measure(range(K), range(K))
    return qc


def run_qaoa_mps(
    Q_obj: Dict,
    h5_params: Dict,
    N: int,
    m: int,
    p: int = 3,
    max_iter: int = 300,
    shots: int = 20000,
    use_buffer: bool = True,
    n_restarts: int = 3,
    seed: Optional[int] = None,
    verbose: bool = True,
    max_bond_dimension: Optional[int] = None,
    truncation_threshold: float = 1e-12,
    sampler_measure_algorithm: str = "mps_apply_measure",
    include_h5: bool = False,
) -> Tuple[List[Tuple[float, List[int], float]], np.ndarray]:
    """
    QAOA optimization and sampling via Qiskit Aer MPS simulation.

    IMPORTANT
    ---------
    Unlike exact statevector simulation, this function does NOT enumerate all 2**N
    basis states. That would destroy the benefit of MPS for large N.

    Instead:
      1) optimize QAOA parameters using an MPS-backed estimator
      2) sample bitstrings from the final circuit using MPS
      3) keep unique feasible solutions (popcount == m)
      4) rank them by exact classical QUBO score

    Parameters
    ----------
    Q_obj      : sparse QUBO dict from build_qubo()
    h5_params  : dict from build_qubo()
    N          : number of qubits / grid cells
    m          : number of new chargers to place
    p          : QAOA circuit depth (reps), default 3
    max_iter   : COBYLA max iterations per restart, default 300
    shots      : number of MPS measurement shots for final sampling
    use_buffer : include buffer in output count, default True
    n_restarts : number of COBYLA restarts (best wins), default 3
    seed       : random seed for reproducibility
    verbose    : print progress, default True
    max_bond_dimension : optional MPS bond-dimension cap
    truncation_threshold : MPS truncation threshold
    sampler_measure_algorithm : "mps_apply_measure" or "mps_probabilities"
    include_h5 : if False (default), uses XY-ring-mixer QAOA.
                 The XY mixer swaps |01⟩↔|10⟩, preserving Hamming weight, so all
                 sampled bitstrings have exactly m ones — H5 is never needed.
                 Initial state is |1^m 0^{K-m}⟩. Circuit has 2p parameters and
                 K-1 XY pairs per layer instead of K*(K-1)/2 H5 ZZ pairs.
                 For K=74: removes 2701 ZZ gates, adds 73 XY pairs per layer.
                 If True, uses standard QAOAAnsatz with H5 merged into the
                 Hamiltonian (original behaviour, ~3 min/eval for K≥74).

    Returns
    -------
    (results, optimal_params)
      results       : List of (score, [cell_ids], empirical_probability), sorted ascending
      optimal_params: np.ndarray of optimal QAOA parameters
    """
    from scipy.optimize import minimize as scipy_minimize

    import os
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    from qubo_builder import evaluate_solution

    if include_h5:
        from qiskit.circuit.library import QAOAAnsatz

    # ------------------------------------------------------------------
    # Local helper
    # ------------------------------------------------------------------
    def solution_from_bitstring(bitstring: str, num_qubits: int) -> List[int]:
        """
        Convert a measurement bitstring into selected cell IDs.

        Qiskit returns bitstrings with the highest-index qubit on the left.
        We reverse so index 0 corresponds to qubit 0.
        """
        reversed_bits = bitstring[::-1]
        return [index for index, bit in enumerate(reversed_bits[:num_qubits]) if bit == "1"]

    # ------------------------------------------------------------------
    # Validate args
    # ------------------------------------------------------------------
    _valid_measure_algos = {"mps_apply_measure", "mps_probabilities"}
    if sampler_measure_algorithm not in _valid_measure_algos:
        raise ValueError(
            f"sampler_measure_algorithm must be one of {_valid_measure_algos}, "
            f"got {sampler_measure_algorithm!r}"
        )

    # ------------------------------------------------------------------
    # Build AerSimulator with MPS options.
    #
    # TUTORIAL-CONFIRMED PATTERN (qiskit-aer docs tutorial 7):
    #   simulator = AerSimulator(method='matrix_product_state')
    #   tcirc = transpile(circ, simulator)
    #   result = simulator.run(tcirc).result()
    #
    # We do NOT use EstimatorV2 here. EstimatorV2 wraps the backend in
    # its own execution pipeline and it is NOT confirmed that set_options()
    # values survive that wrapping. Direct backend.run() is what the docs
    # show and what is guaranteed to work.
    #
    # For the optimization loop we estimate <H> from shot counts using
    # counts_to_energy(), matching the run_qaoa_noisy pattern exactly.
    # ------------------------------------------------------------------
    _n_threads = int(os.environ.get("QISKIT_NUM_THREADS", os.cpu_count() or 1))

    simulator = AerSimulator(method="matrix_product_state")
    simulator.set_options(
        matrix_product_state_truncation_threshold=truncation_threshold,
        mps_sample_measure_algorithm=sampler_measure_algorithm,
        max_parallel_threads=_n_threads,
        max_parallel_shots=_n_threads,
        max_parallel_experiments=1,
    )
    if max_bond_dimension is not None:
        simulator.set_options(
            matrix_product_state_max_bond_dimension=max_bond_dimension
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    top_k = m + compute_buffer(m, use_buffer)

    if verbose:
        bd_str = str(max_bond_dimension) if max_bond_dimension is not None else "unlimited"
        print(
            f"QAOA MPS Pipeline: N={N}, m={m}, p={p}, "
            f"restarts={n_restarts}, max_iter={max_iter}, "
            f"shots={shots}, top_k={top_k}"
        )
        print(
            f"  AerSimulator: method=matrix_product_state, "
            f"bond_dim={bd_str}, trunc={truncation_threshold}, "
            f"measure_algo={sampler_measure_algorithm}, threads={_n_threads}"
        )

    # ── Build Ising coefficients ───────────────────────────────────────────────
    if include_h5:
        h_total, J_total, offset_total = build_total_ising_coeffs(Q_obj, h5_params, N)
        cost_ham = build_cost_hamiltonian(Q_obj, h5_params, N)
        if verbose:
            print(f"  H5 INCLUDED: adds {N*(N-1)//2} all-to-all ZZ terms to circuit")
    else:
        h_total, J_total, offset_total = qubo_to_ising_coeffs(Q_obj, N)
        if verbose:
            n_zz = sum(1 for v in J_total.values() if v != 0.0)
            print(
                f"  XY-mixer QAOA: H5 excluded. "
                f"Objective has {len(h_total)} Z terms, {n_zz} ZZ terms. "
                f"Feasibility guaranteed by Hamming-weight-preserving XY ring mixer."
            )

    # Strip constant offset — optimizer sees a zero-centred landscape.
    if include_h5:
        identity_label = "I" * N
        offset = 0.0
        for label, coeff in cost_ham.to_list():
            if label == identity_label:
                offset += float(np.real(coeff))
    else:
        offset = offset_total  # Ising constant from Q_obj = identity coefficient

    # ── Build and transpile circuit ONCE ──────────────────────────────────────
    # IMPORTANT: do NOT pass 'simulator' to transpile().
    # AerSimulator(method='matrix_product_state') inherits an IBM coupling map
    # (~63 qubits) which causes CircuitTooWideForTarget for K>63 even with
    # coupling_map=None. Omit backend; supply basis_gates directly instead.
    if include_h5:
        qaoa_circuit_bare = QAOAAnsatz(
            cost_operator=cost_ham,
            reps=p,
        ).decompose(reps=3)
        n_params = qaoa_circuit_bare.num_parameters
        measured_template = QuantumCircuit(N, N)
        measured_template.compose(qaoa_circuit_bare, inplace=True)
        measured_template.measure(range(N), range(N))
        circuit_to_transpile = measured_template
    else:
        # XY path: all restarts share the same Dicke-state circuit.
        # Build and transpile ONCE here; the restart loop uses .copy().
        n_params = 2 * p
        transpiled_template = transpile(
            _build_qaoa_xy_circuit(N, m, p, h_total, J_total),
            coupling_map=None,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            seed_transpiler=seed,
            optimization_level=1,
        )
        if verbose:
            print(
                f"  Circuit [XY-ring + Dicke |D({N},{m})⟩]: {n_params} params, "
                f"{transpiled_template.size()} gates (post-transpile, shared across restarts)"
            )

    if include_h5:
        transpiled_template = transpile(
            circuit_to_transpile,
            coupling_map=None,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            seed_transpiler=seed,
            optimization_level=1,
        )
        if verbose:
            print(
                f"  Circuit [QAOAAnsatz+H5]: {n_params} params, "
                f"{transpiled_template.size()} gates (post-transpile)"
            )

    # shots_opt: fewer shots during optimization to keep COBYLA evals fast.
    # The tutorial shows counts → energy works at any shot count.
    shots_opt = max(512, shots // 10)

    rng = np.random.default_rng(seed)
    best_cost_shifted = float("inf")
    best_optimal_params: Optional[np.ndarray] = None
    best_template = None  # transpiled circuit of the best restart (for final sampling)

    # ------------------------------------------------------------------
    # Variational optimization — sequential restarts with per-eval tqdm
    #
    # MPS simulation is parallelized internally via BLAS/OpenMP, so each
    # sequential restart uses all _n_threads CPU cores. ProcessPoolExecutor
    # on macOS uses "spawn", requiring fresh Python interpreter startup per
    # worker — that overhead alone exceeded 16 minutes before any simulation
    # began.
    # ------------------------------------------------------------------
    try:
        from tqdm.auto import tqdm as _tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    initial_params_list = [rng.uniform(-np.pi, np.pi, n_params) for _ in range(n_restarts)]

    if verbose:
        print(f"  Sequential restarts: {n_restarts}, up to {max_iter} COBYLA evals each")

    for restart_idx in range(n_restarts):
        init_params = initial_params_list[restart_idx]
        eval_count = [0]
        best_so_far = [float("inf")]

        my_template = transpiled_template.copy()

        bar = None
        if _has_tqdm and verbose:
            bar = _tqdm(
                total=max_iter,
                desc=f"Restart {restart_idx+1}/{n_restarts}  best=?",
                unit="eval",
                dynamic_ncols=True,
                leave=True,
            )

        def objective(
            params: np.ndarray,
            _bar=bar, _ec=eval_count, _bs=best_so_far,
            _tmpl=my_template, _ridx=restart_idx,
        ) -> float:
            _ec[0] += 1
            bound = _tmpl.assign_parameters(params)
            counts = simulator.run(
                bound, shots=shots_opt, seed_simulator=None,
            ).result().get_counts()
            energy = counts_to_energy(counts, h_total, J_total, offset_total, N) - offset
            if energy < _bs[0]:
                _bs[0] = energy
            if _bar is not None:
                _bar.update(1)
                _bar.set_description(
                    f"Restart {_ridx+1}/{n_restarts}  best={_bs[0]+offset:.4f}"
                )
            return energy

        opt = scipy_minimize(objective, init_params, method="COBYLA", options={"maxiter": max_iter})

        if bar is not None:
            bar.set_description(
                f"Restart {restart_idx+1}/{n_restarts}  DONE  best={float(opt.fun)+offset:.4f}"
            )
            bar.close()

        cost = float(opt.fun)
        if verbose:
            print(
                f"  Restart {restart_idx+1}/{n_restarts}: "
                f"cost={cost:.6f} (shifted), raw={cost+offset:.6f}, evals={eval_count[0]}"
            )
        if cost < best_cost_shifted:
            best_cost_shifted = cost
            best_optimal_params = np.asarray(opt.x, dtype=float)
            best_template = my_template

    if best_optimal_params is None:
        raise RuntimeError("QAOA optimization failed to produce any result.")

    if verbose:
        print(
            f"  Best shifted cost: {best_cost_shifted:.6f} "
            f"(raw: {best_cost_shifted + offset:.6f})"
        )

    optimal_params = best_optimal_params

    # ------------------------------------------------------------------
    # Final sampling — bind optimal params from best restart, run with full shot budget
    # ------------------------------------------------------------------
    bound_final = best_template.assign_parameters(optimal_params)

    run_result = simulator.run(
        bound_final,
        shots=shots,
        seed_simulator=seed,
    ).result()

    counts = run_result.get_counts()

    # ------------------------------------------------------------------
    # Process sampled bitstrings
    # ------------------------------------------------------------------
    unique_scored: Dict[Tuple[int, ...], Tuple[float, List[int], float]] = {}
    feasible_shots = 0

    for bitstring, count in counts.items():
        solution = solution_from_bitstring(bitstring, N)

        if len(solution) != m:
            continue

        feasible_shots += count

        solution_key = tuple(sorted(solution))
        empirical_probability = count / shots
        score = float(evaluate_solution(Q_obj, solution))

        # Keep the probability observed for this sampled solution.
        # If a solution somehow appears in merged count formats, accumulate.
        if solution_key in unique_scored:
            previous_score, previous_solution, previous_probability = unique_scored[solution_key]
            unique_scored[solution_key] = (
                previous_score,
                previous_solution,
                previous_probability + empirical_probability,
            )
        else:
            unique_scored[solution_key] = (
                score,
                list(solution_key),
                empirical_probability,
            )

    results = sorted(unique_scored.values(), key=lambda row: row[0])[:top_k]

    if verbose:
        feasible_fraction = feasible_shots / shots if shots > 0 else 0.0
        print(
            f"  Feasible sampled shots (popcount={m}): "
            f"{feasible_shots}/{shots} = {feasible_fraction:.4f}"
        )
        print(f"  Unique feasible sampled solutions: {len(unique_scored)}")
        print(f"  Returning top {len(results)}")

        metadata = run_result.results[0].metadata if run_result.results else {}
        if metadata:
            # These keys may or may not exist depending on Aer version/options.
            maybe_log_keys = [
                "MPS_log_data",
                "matrix_product_state_max_bond_dimension",
                "matrix_product_state_truncation_threshold",
            ]
            present = {key: metadata[key] for key in maybe_log_keys if key in metadata}
            if present:
                print(f"  MPS metadata: {present}")

    return results, optimal_params


def run_qaoa_noisy(
    Q_obj: Dict,
    h5_params: Dict,
    N: int,
    m: int,
    backend,
    p: int = 3,
    max_iter: int = 200,
    use_buffer: bool = True,
    n_restarts: int = 2,
    seed: Optional[int] = None,
    shots_opt: int = 2048,
    shots_final: int = 8192,
    optimization_level: int = 1,
    verbose: bool = True,
) -> Tuple[List[Tuple[float, List[int], float]], np.ndarray]:
    """
    Full QAOA pipeline with optimization AND sampling on a noisy fake backend.

    Uses qiskit_ibm_runtime.SamplerV2 with a user-provided fake backend.
    Every COBYLA evaluation runs shots through the noisy backend.
    The parametrized circuit is transpiled ONCE; each eval binds params and runs.

    Parameters
    ----------
    Q_obj, h5_params, N, m : standard QUBO inputs
    backend         : fake backend instance — must have >= N qubits.
                      Create it yourself:
                        from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
                        backend = FakeGuadalupeV2()
    p               : QAOA depth, default 3
    max_iter        : COBYLA iterations per restart, default 200
    use_buffer      : include buffer in output, default True
    n_restarts      : COBYLA restarts, default 2
    seed            : random seed
    shots_opt       : shots per COBYLA evaluation, default 2048
    shots_final     : shots for final sampling, default 8192
    optimization_level : transpiler level, default 1
    verbose         : print progress

    Returns
    -------
    (results, optimal_params)
      results       : List of (score, [cell_ids], frequency) best first
      optimal_params: np.ndarray
    """
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2
    from scipy.optimize import minimize as scipy_minimize
    from qubo_builder import evaluate_solution

    top_k = m + compute_buffer(m, use_buffer)

    if verbose:
        backend_name = type(backend).__name__
        print(f"QAOA Noisy Pipeline: N={N}, m={m}, p={p}, "
              f"backend={backend_name}")
        print(f"  restarts={n_restarts}, max_iter={max_iter}, "
              f"shots_opt={shots_opt}, shots_final={shots_final}, top_k={top_k}")

    # ── Build Ising coefficients (for energy estimation from counts) ──
    h_total, J_total, offset_total = build_total_ising_coeffs(Q_obj, h5_params, N)
    cost_op = ising_to_sparse_pauli_op(h_total, J_total, offset_total, N)

    # Strip offset for optimizer
    identity_label = 'I' * N
    offset = 0.0
    for label, coeff in cost_op.to_list():
        if label == identity_label:
            offset += coeff.real

    if verbose:
        print(f"  Cost Hamiltonian: {len(cost_op)} terms, offset={offset:.4f}")

    # ── Build and decompose QAOA circuit ──
    circuit = QAOAAnsatz(cost_op, reps=p).decompose(reps=3)
    n_params = circuit.num_parameters

    # Add measurements to parametrized circuit (before transpile)
    measured_circuit = circuit.copy()
    measured_circuit.measure_all()

    if verbose:
        print(f"  Circuit: {n_params} params, {measured_circuit.size()} gates "
              f"(pre-transpile)")

    # ── Transpile ONCE onto the provided backend ──
    if verbose:
        print(f"  Transpiling parametrized circuit...")

    transpiled = transpile(
        measured_circuit, backend,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )

    if verbose:
        print(f"  Transpiled: {transpiled.size()} gates, depth={transpiled.depth()}")

    # ── Create sampler ──
    sampler = SamplerV2(backend)

    # ── Optimization loop ──
    rng = np.random.default_rng(seed)
    best_result = None
    best_cost = float('inf')

    for restart in range(n_restarts):
        init_params = rng.uniform(-np.pi, np.pi, n_params)

        eval_count = 0
        def cost_fn(params):
            nonlocal eval_count
            eval_count += 1

            bound = transpiled.assign_parameters(params)
            job = sampler.run([bound], shots=shots_opt)
            counts = job.result()[0].data.meas.get_counts()
            raw_energy = counts_to_energy(counts, h_total, J_total, offset_total, N)
            return raw_energy - offset

        result = scipy_minimize(
            cost_fn,
            init_params,
            method='COBYLA',
            options={'maxiter': max_iter},
        )

        if verbose:
            print(f"  Restart {restart+1}/{n_restarts}: "
                  f"cost={result.fun:.4f} (shifted), "
                  f"raw={result.fun + offset:.4f}, "
                  f"evals={eval_count}")

        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result

    if verbose:
        print(f"  Best shifted cost: {best_cost:.4f}  "
              f"(raw: {best_cost + offset:.4f})")

    # ── Final sampling with more shots ──
    if verbose:
        print(f"  Final sampling: {shots_final} shots via SamplerV2...")

    bound_final = transpiled.assign_parameters(best_result.x)
    job = sampler.run([bound_final], shots=shots_final)
    counts = job.result()[0].data.meas.get_counts()

    # ── Post-process ──
    total_shots = sum(counts.values())
    total_feasible = 0
    scored = []

    for bitstring, count in counts.items():
        integer = int(bitstring.replace(" ", ""), 2)
        solution = solution_from_integer(integer, N)
        if len(solution) == m:
            total_feasible += count
            score = evaluate_solution(Q_obj, solution)
            scored.append((score, sorted(solution), count / total_shots))

    if verbose:
        print(f"  Feasible: {total_feasible}/{total_shots} "
              f"({total_feasible/total_shots*100:.1f}%)")
        print(f"  Unique feasible solutions: {len(scored)}")

    scored.sort(key=lambda x: x[0])
    seen = set()
    results = []
    for score, solution, freq in scored:
        key = tuple(solution)
        if key not in seen:
            seen.add(key)
            results.append((score, solution, freq))

    results = results[:top_k]
    if verbose:
        print(f"  Returning top {len(results)}")

    return results, best_result.x


def print_qaoa_results(
    results: List[Tuple[float, List[int], float]],
    N: int,
    num_cols: int,
):
    """
    Pretty-print QAOA output with grid positions and probabilities.

    Parameters
    ----------
    results  : list of (score, [cell_ids], probability) from run_qaoa()
    N        : number of grid cells
    num_cols : grid columns (for row/col display)
    """
    print(f"\n{'Rank':<6} {'Score':>10} {'Prob':>10}   "
          f"{'Grid IDs':<25} {'Positions (row,col)'}")
    print("─" * 80)
    for rank, (score, solution, prob) in enumerate(results, 1):
        ids_str = str(solution)
        pos = [(cid // num_cols, cid % num_cols) for cid in solution]
        pos_str = str(pos)
        print(f"  {rank:<4} {score:>10.4f} {prob:>10.6f}   "
              f"{ids_str:<25} {pos_str}")