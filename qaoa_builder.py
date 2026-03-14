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