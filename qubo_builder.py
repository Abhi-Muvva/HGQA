"""
QUBO Construction Module v2
============================
README reference: Section — Fitness Function / Unified QUBO Formulation
Baseline reference: Section 6

Builds the sparse objective Q matrix (H1–H4, H6) and a separate H5 parameter
bundle for EV charging station placement.

ARCHITECTURE — H5 IS NOT IN Q_obj
------------------------------------

Term layout in Q_obj (upper triangular, i <= j):
  DIAGONAL  Q[(i,i)]
    H1: POI attraction       → negative (reward), normalized before α
    H2: Gas station bonus    → negative (reward), normalized before α
    H3: Existing charger pen → positive (penalty), normalized before α

  OFF-DIAGONAL  Q[(i,j)], i < j
    H4: New charger spacing  → positive (penalty), within R4
    H6: Coverage redundancy  → positive (penalty), within R6

PER-TERM NORMALIZATION (H1, H2, H3 diagonals only)
----------------------------------------------------
Before applying α, each term's raw values are divided by their maximum
absolute value across all N cells, scaling them to [-1, 0] or [0, 1].
This makes α weights dataset-independent: α1=2, α3=1 genuinely means
"POI attraction is twice as important as existing charger penalty"
regardless of grid size or POI count.
H4, H6 (quadratic off-diagonal) and H5 (constraint) are NOT normalized.

"""

import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chebyshev(i: int, j: int, num_cols: int) -> int:
    """Chebyshev distance between cells i and j. Baseline Section 3.3."""
    ri, ci = divmod(i, num_cols)
    rj, cj = divmod(j, num_cols)
    return max(abs(ri - rj), abs(ci - cj))


def _norm_scale(arr: np.ndarray) -> float:
    """
    Max absolute value of array, or 1.0 if all zeros.
    Used as the normalization divisor — returns 1.0 (safe no-op) when a
    term contributes nothing (e.g., no gas stations → H2 all zero).
    """
    max_abs = np.max(np.abs(arr))
    return float(max_abs) if max_abs > 0 else 1.0


def _precompute(grid_details: Dict, cell_weights: Dict, num_cols: int, Rs: int) -> Dict:
    """
    Precompute all per-cell quantities needed to build Q_obj.

    Parameters
    ----------
    grid_details : dict — from divide_graph_into_parts()
    cell_weights : dict — from calculate_cell_weights()
    num_cols     : int  — grid columns (for Chebyshev distance)
    Rs           : int  — service gap radius (for s_c computation in H1)

    Returns
    -------
    dict with keys:
      N      — total number of cells
      w      — cell weight array [N]
      nw     — normalized cell weight array [N]  (= w / scale_factor)
      g      — gas station COUNT array [N]
      C_POI  — set of cell IDs with at least one POI
      C_ex   — set of cell IDs with at least one existing charger
      E_all  — flat list of cell IDs, one entry per individual charger
      s      — service gap dict {cell_id: float} for POI cells only
      Rs     — radius used for s_c (stored for reporting)
    """
    N = len(grid_details)
    w  = np.zeros(N)
    nw = np.zeros(N)
    g  = np.zeros(N)

    C_POI = set()
    C_ex  = set()
    E_all = []  

    for gid, info in grid_details.items():
        w[gid]  = cell_weights[gid]['weight']
        nw[gid] = cell_weights[gid]['normalized_weight']

        # Gas station COUNT 
        g[gid] = float(len(info.get('gas_stations', [])))

        if info['num_pois'] > 0:
            C_POI.add(gid)

        # Build E_all: one entry per individual charger in this cell
        chargers_here = info.get('existing_chargers', [])
        if chargers_here:
            C_ex.add(gid)
            for _ in chargers_here:
                E_all.append(gid)  # cell ID repeated once per charger

    # Service gap factor s_c (README H1 definition).
    # s_c = 1 / (1 + Σ_{e ∈ E_all, d(c,e) ≤ Rs} 1/(1 + d(c,e)))
    #
    # Sums over INDIVIDUAL chargers within Rs of POI cell c.
    # A cell with 3 chargers at d=0 contributes 3 × (1/(1+0)) = 3 to the sum,
    # giving s_c = 1/(1+3) = 0.25 — stronger saturation than a single charger.
    # If no chargers within Rs: denom = 1.0 → s_c = 1.0 (full attraction).
    s = {}
    for c in C_POI:
        denom = 1.0
        for e_cell in E_all:
            d_ce = _chebyshev(c, e_cell, num_cols)
            if d_ce <= Rs:
                denom += 1.0 / (1.0 + d_ce)
        s[c] = 1.0 / denom

    return {
        'N': N,
        'num_cols': num_cols,
        'w': w,
        'nw': nw,
        'g': g,
        'C_POI': C_POI,
        'C_ex': C_ex,
        'E_all': E_all,
        's': s,
        'Rs': Rs,
    }


def build_qubo(
    grid_details: Dict,
    cell_weights: Dict,
    plot_deets: Dict,
    m: int,
    # Alpha weights — applied AFTER per-term normalization for H1/H2/H3
    alpha1: float = 1.0,
    alpha2: float = 1.0,
    alpha3: float = 1.0,
    alpha4: float = 1.0,
    alpha5: float = 1.0,   # only appears in h5_params, not Q_obj
    alpha6: float = 1.0,
    # Per-term magnitude parameters
    beta:    float = 1.0,  # gas station bonus magnitude (H2)
    gamma:   float = 1.0,  # existing charger penalty magnitude (H3)
    delta:   float = 1.0,  # new charger spacing magnitude (H4)
    epsilon: float = 1.0,  # coverage redundancy magnitude (H6)
    lam:     float = 10.0, # constraint penalty λ — returned in h5_params only
    # Distance radii — one per term (README recommends R1 >= R4 > R3 >= R6)
    R1: int = 5,   # H1 attraction radius
    Rs: int = 5,   # service gap radius for s_c (recommended to match R1)
    R3: int = 4,   # H3 existing charger penalty radius
    R4: int = 4,   # H4 new charger spacing radius
    R6: int = 3,   # H6 coverage redundancy radius
) -> Tuple[Dict, Dict, Dict]:
    """
    Build the sparse QUBO objective matrix Q_obj (H1–H4, H6) and
    return H5 parameters separately as h5_params.

    README reference: Section — Complete QUBO Formulation

    Parameters
    ----------
    grid_details : dict   — from divide_graph_into_parts()
    cell_weights : dict   — from calculate_cell_weights()
    plot_deets   : dict   — from divide_graph_into_parts() (provides num_cols)
    m            : int    — number of new chargers to place
    alpha1–6     : float  — relative term weights (H1/H2/H3 applied after
                            normalization; H4/H6 applied to raw values;
                            alpha5 goes into h5_params only)
    beta         : float  — gas station bonus magnitude (H2)
    gamma        : float  — existing charger penalty magnitude (H3)
    delta        : float  — new charger spacing magnitude (H4)
    epsilon      : float  — coverage redundancy magnitude (H6)
    lam          : float  — constraint penalty λ (H5 only, not in Q_obj)
    R1, Rs, R3, R4, R6 : int — per-term distance radii

    Returns
    -------
    Q_obj     : dict {(i,j): float}, i <= j — sparse objective Q (H1–H4, H6)
    h5_params : dict — {'lam': float, 'alpha5': float, 'm': int}
    diags     : dict — per-term diagnostics for inspection and debugging

    Evaluation
    ----------
    GA:   evaluate_solution(Q_obj, solution)              # H5 skipped
    QAOA: evaluate_solution(Q_obj, solution, h5_params)   # H5 included
    """
    num_cols = plot_deets['num_cols']
    pre = _precompute(grid_details, cell_weights, num_cols, Rs)
    N, w, nw, g, C_POI, E_all, s = (
        pre['N'], pre['w'], pre['nw'], pre['g'],
        pre['C_POI'], pre['E_all'], pre['s']
    )

    Q_obj = defaultdict(float)

    # -----------------------------------------------------------------------
    # STEP 1: Compute raw diagonal values for H1, H2, H3
    # -----------------------------------------------------------------------
    h1_raw = np.zeros(N)
    h2_raw = np.zeros(N)
    h3_raw = np.zeros(N)

    for i in range(N):
        # H1 — POI Attraction
        # raw_h1[i] = - Σ_{c∈C_POI, d(i,c)≤R1} w_c × s_c / (1 + d(i,c))
        # Weight w_c comes from the POI cell c, not from cell i — empty cells
        # adjacent to dense POIs are still attractive for placement.
        h1_val = 0.0
        for c in C_POI:
            d_ic = _chebyshev(i, c, num_cols)
            if d_ic <= R1:
                h1_val -= w[c] * s[c] / (1.0 + d_ic)
        h1_raw[i] = h1_val

        # H2 — Gas Station Bonus
        # raw_h2[i] = - β × g_i  (g_i = COUNT of gas stations in cell i)
        h2_raw[i] = -beta * g[i]

        # H3 — Existing Charger Penalty
        # raw_h3[i] = γ × (1 - nw_i) × Σ_{e∈E_all, d(i,e)≤R3} 1/(1+d(i,e))
        # Sums over INDIVIDUAL chargers. A cell with 3 chargers at d=0
        # contributes 3 × 1.0 = 3 to the sum — stronger saturation signal.
        h3_val = 0.0
        for e_cell in E_all:
            d_ie = _chebyshev(i, e_cell, num_cols)
            if d_ie <= R3:
                h3_val += 1.0 / (1.0 + d_ie)
        h3_raw[i] = gamma * (1.0 - nw[i]) * h3_val

    # -----------------------------------------------------------------------
    # STEP 2: Normalize H1, H2, H3 raw values, then apply α weights
    # -----------------------------------------------------------------------
    ns_h1 = _norm_scale(h1_raw)  # normalization scale (stored in diags)
    ns_h2 = _norm_scale(h2_raw)
    ns_h3 = _norm_scale(h3_raw)

    h1_diag = alpha1 * (h1_raw / ns_h1)
    h2_diag = alpha2 * (h2_raw / ns_h2)
    h3_diag = alpha3 * (h3_raw / ns_h3)

    # -----------------------------------------------------------------------
    # STEP 3: Write diagonal to Q_obj (H1 + H2 + H3 only — no H5)
    # -----------------------------------------------------------------------
    for i in range(N):
        val = h1_diag[i] + h2_diag[i] + h3_diag[i]
        if val != 0.0:
            Q_obj[(i, i)] = val

    # -----------------------------------------------------------------------
    # STEP 4: Off-diagonal terms H4, H6 (H5 NOT included)
    # -----------------------------------------------------------------------

    # Precompute per-cell POI neighbor sets for H6 efficiency
    poi_in_R6 = {
        i: {c for c in C_POI if _chebyshev(i, c, num_cols) <= R6}
        for i in range(N)
    }

    h4_count = 0
    h6_count = 0
    h4_peak  = 0.0   # actual max H4 entry written to Q_obj (post α×δ)
    h6_peak  = 0.0   # actual max H6 entry written to Q_obj (post α×ε)

    for i in range(N):
        for j in range(i + 1, N):
            d_ij = _chebyshev(i, j, num_cols)
            val = 0.0

            # H4 — New Charger Spacing (within R4)
            # Q_ij += α4 × δ × (1 - max(nw_i, nw_j)) / (1 + d(i,j))
            # max() means: if EITHER cell is high-density, penalty drops —
            # clustering in high-demand areas is justified.
            if d_ij <= R4:
                h4_c = alpha4 * delta * (1.0 - max(nw[i], nw[j])) / (1.0 + d_ij)
                if h4_c != 0.0:
                    val += h4_c
                    h4_count += 1
                    if h4_c > h4_peak:
                        h4_peak = h4_c

            # H6 — Coverage Redundancy (both cells within R6 of shared POI)
            # Q_ij += α6 × ε × Σ_{c: d(i,c)≤R6 AND d(j,c)≤R6}
            #                    w_c / ((1+d(i,c)) × (1+d(j,c)))
            shared = poi_in_R6[i] & poi_in_R6[j]
            if shared:
                h6_raw = sum(
                    w[c] / ((1.0 + _chebyshev(i, c, num_cols))
                             * (1.0 + _chebyshev(j, c, num_cols)))
                    for c in shared
                )
                h6_c = alpha6 * epsilon * h6_raw
                if h6_c != 0.0:
                    val += h6_c
                    h6_count += 1
                    if h6_c > h6_peak:
                        h6_peak = h6_c

            if val != 0.0:
                Q_obj[(i, j)] = val

    # -----------------------------------------------------------------------
    # STEP 5: H5 parameters — returned separately, never stored in Q_obj
    # -----------------------------------------------------------------------
    h5_params = {'lam': lam, 'alpha5': alpha5, 'm': m}

    # -----------------------------------------------------------------------
    # Diagnostics bundle
    # -----------------------------------------------------------------------
    diags = {
        'N': N,
        'm': m,
        'num_cols': num_cols,
        'C_POI': pre['C_POI'],
        'C_ex':  pre['C_ex'],
        'E_all': E_all,
        'precomputed': pre,
        # Normalization scales (raw max-abs before α)
        'norm_scale_h1': ns_h1,
        'norm_scale_h2': ns_h2,
        'norm_scale_h3': ns_h3,
        # Raw ranges (pre-normalization — for sanity checking)
        'h1_raw_range': (float(h1_raw.min()), float(h1_raw.max())),
        'h2_raw_range': (float(h2_raw.min()), float(h2_raw.max())),
        'h3_raw_range': (float(h3_raw.min()), float(h3_raw.max())),
        # Post-normalization ranges (what's actually in Q_obj diagonals)
        'h1_diag_range': (float(h1_diag.min()), float(h1_diag.max())),
        'h2_diag_range': (float(h2_diag.min()), float(h2_diag.max())),
        'h3_diag_range': (float(h3_diag.min()), float(h3_diag.max())),
        # Off-diagonal counts and actual peaks (post α weighting)
        'h4_nonzero_pairs': h4_count,
        'h6_nonzero_pairs': h6_count,
        'h4_actual_peak':   h4_peak,   # max H4 entry in Q_obj — for calibrate_alphas()
        'h6_actual_peak':   h6_peak,   # max H6 entry in Q_obj — for calibrate_alphas()
        'total_Q_obj_entries': len(Q_obj),
        # Radii
        'R1': R1, 'Rs': Rs, 'R3': R3, 'R4': R4, 'R6': R6,
    }

    return dict(Q_obj), h5_params, diags


# ---------------------------------------------------------------------------
# Post-build calibration
# ---------------------------------------------------------------------------

def calibrate_alphas(
    Q_obj: Dict,
    params: Dict,
    diags: Dict,
    target_ratio: float = 0.5,
    verbose: bool = True,
) -> Tuple[Dict, bool]:
    """
    Post-build magnitude check: rescale α4 and α6 so off-diagonal peaks
    don't exceed target_ratio × diagonal peak (absolute value).

    WHY THIS IS NEEDED
    ------------------
    H1/H2/H3 are normalized to [-1, 0] / [0, 1] before α weighting.
    H4 and H6 are NOT normalized — their effective magnitude is α × raw_peak,
    where raw_peak depends on the dataset (POI weights, grid geometry).
    suggest_parameters() estimates this and pre-caps α4/α6, but the estimate
    uses simplified loops and may differ from the actual Q_obj values.
    This function corrects using the actual Q_obj values after build.

    WHAT IT DOES
    ------------
    1. Reads h4_actual_peak and h6_actual_peak from diags (set by build_qubo).
    2. Computes diagonal peak = max |Q[(i,i)]|.
    3. Target: each off-diagonal term peak ≤ target_ratio × diagonal peak.
       Default target_ratio=0.5 means H4 and H6 are tiebreakers at most,
       never the dominant driver.
    4. If H4 or H6 peak exceeds target, rescales the corresponding α.
    5. Returns updated params dict and a bool indicating whether a rebuild
       is needed (True = at least one α was changed).

    USAGE
    -----
        Q_obj, h5_params, diags = build_qubo(..., **params)
        params, needs_rebuild = calibrate_alphas(Q_obj, params, diags)
        if needs_rebuild:
            Q_obj, h5_params, diags = build_qubo(..., **params)

    Parameters
    ----------
    Q_obj        : dict — sparse Q from build_qubo()
    params       : dict — flat param dict passed to build_qubo()
                   (keys: alpha1..alpha6, beta, gamma, delta, epsilon, lam, R1, Rs, R3, R4, R6)
    diags        : dict — diagnostics from build_qubo()
    target_ratio : float — H4/H6 peak allowed as fraction of diagonal peak (default 0.5)
    verbose      : bool — print calibration report

    Returns
    -------
    new_params   : dict — updated params (copy of input, α4/α6 adjusted if needed)
    needs_rebuild: bool — True if any α was changed (rebuild Q_obj before using it)
    """
    if not Q_obj:
        raise ValueError("Q_obj is empty — run build_qubo() first.")

    diag_vals = [abs(v) for (i, j), v in Q_obj.items() if i == j]
    if not diag_vals:
        raise ValueError("Q_obj has no diagonal entries — cannot calibrate.")

    diag_peak = max(diag_vals)
    target    = target_ratio * diag_peak

    h4_peak = diags.get('h4_actual_peak', 0.0)
    h6_peak = diags.get('h6_actual_peak', 0.0)

    new_params   = dict(params)
    needs_rebuild = False
    changes      = []

    if h4_peak > target and h4_peak > 0:
        scale = target / h4_peak
        old   = new_params.get('alpha4', 1.0)
        new_params['alpha4'] = round(old * scale, 4)
        needs_rebuild = True
        changes.append(
            f"  α4: {old:.4f} → {new_params['alpha4']:.4f}  "
            f"(H4 peak {h4_peak:.4f} > target {target:.4f})"
        )

    if h6_peak > target and h6_peak > 0:
        scale = target / h6_peak
        old   = new_params.get('alpha6', 1.0)
        new_params['alpha6'] = round(old * scale, 4)
        needs_rebuild = True
        changes.append(
            f"  α6: {old:.4f} → {new_params['alpha6']:.4f}  "
            f"(H6 peak {h6_peak:.4f} > target {target:.4f})"
        )

    if verbose:
        print("=" * 55)
        print("CALIBRATE_ALPHAS")
        print(f"  Diagonal peak:  {diag_peak:.4f}")
        print(f"  Target (×{target_ratio}): {target:.4f}")
        print(f"  H4 actual peak: {h4_peak:.4f}  {'✗ TOO HIGH' if h4_peak > target else '✓ OK'}")
        print(f"  H6 actual peak: {h6_peak:.4f}  {'✗ TOO HIGH' if h6_peak > target else '✓ OK'}")
        if changes:
            print("  Adjustments:")
            for c in changes:
                print(c)
            print("  → Rebuild Q_obj with updated params.")
        else:
            print("  → No rebuild needed.")
        print("=" * 55)

    return new_params, needs_rebuild

def evaluate_solution(
    Q_obj: Dict,
    solution: List[int],
    h5_params: Dict = None,
) -> float:
    """
    Evaluate f(x) = f_obj(x) [+ f_h5(x)] for a given solution.

      f_obj(x) = Σ_i Q_obj[(i,i)] x_i + Σ_{i<j} Q_obj[(i,j)] x_i x_j
      f_h5(x)  = α5 × λ × (|solution| − m)²

    Parameters
    ----------
    Q_obj     : sparse objective Q from build_qubo() — H1–H4, H6 only
    solution  : list of grid IDs where chargers are placed
    h5_params : dict from build_qubo() — {'lam', 'alpha5', 'm'}
                Pass None to skip H5 (correct for GA; H5=0 by encoding).

    Returns
    -------
    float — objective value (lower is better)
    """
    s_set = set(solution)
    total = 0.0
    for (i, j), val in Q_obj.items():
        if i == j:
            if i in s_set:
                total += val
        else:
            if i in s_set and j in s_set:
                total += val

    if h5_params is not None:
        violation = len(solution) - h5_params['m']
        total += h5_params['alpha5'] * h5_params['lam'] * violation ** 2

    return total