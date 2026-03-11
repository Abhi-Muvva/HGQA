import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Tuple, Dict, Union
from tabulate import tabulate

STANDARD_FIGSIZE = (11, 8)


"""
Grid Discretization Module
===========================
Baseline reference: Sections 3.1–3.4, 4.4 (geometry only — weight calculation is separate)

Grid ID convention: row-major, row 0 = bottom (y_min)
  grid_id = row * num_cols + col
  row = grid_id // num_cols
  col = grid_id % num_cols
"""


def calculate_cell_weights(grid_details: Dict, scale_factor: float = 5.0, min_weight: float = 0.5) -> Dict:
    """
    Calculate continuous cell weights from POI densities.

    Baseline reference: Section 4.3 (Cell Weight Calculation)

    Algorithm:
    1. Aggregate: Sum POI densities in each cell → raw_score
    2. Normalize: raw_score / max(raw_score) → [0, 1]
    3. Scale with floor: weight = max(normalized × scale_factor, min_weight) if POIs exist, else 0

    Parameters
    ----------
    grid_details : dict
        Grid data from divide_graph_into_parts()
    scale_factor : float, default=5.0
        Amplifies differentiation at top end (Section 4.3, default: 5)
    min_weight : float, default=0.5
        Floor for lowest-density cells that have POIs (Section 4.3, default: 0.5)

    Returns
    -------
    dict : {grid_id: {'raw_score': float, 'normalized_score': float, 'weight': float, 'normalized_weight': float}}

    Example
    -------
    Cell with 2 POIs (densities 0.9, 0.4):
      raw_score = 0.9 + 0.4 = 1.3
      If max raw_score across all cells = 2.0:
        normalized = 1.3 / 2.0 = 0.65
        weight = max(0.65 × 5, 0.5) = 3.25
        normalized_weight = 3.25 / 5 = 0.65
    """
    num_cells = len(grid_details)
    weights = {}

    # Step 1: Aggregate POI densities per cell
    for gid, info in grid_details.items():
        raw_score = sum(poi[2] for poi in info['pois'])  # poi[2] is density
        weights[gid] = {'raw_score': raw_score}

    # Step 2: Normalize by max
    max_raw_score = max((w['raw_score'] for w in weights.values()), default=1.0)

    if max_raw_score == 0:
        # Edge case: no POIs at all
        for gid in weights:
            weights[gid]['normalized_score'] = 0.0
            weights[gid]['weight'] = 0.0
            weights[gid]['normalized_weight'] = 0.0
    else:
        for gid in weights:
            normalized = weights[gid]['raw_score'] / max_raw_score
            weights[gid]['normalized_score'] = normalized

            # Step 3: Scale with floor
            if weights[gid]['raw_score'] > 0:
                # Cell has POIs: apply scaling with floor
                weight = max(normalized * scale_factor, min_weight)
            else:
                # Empty cell: weight = 0
                weight = 0.0

            weights[gid]['weight'] = weight
            weights[gid]['normalized_weight'] = weight / scale_factor  # nw_c for use in H3, H4

    return weights



def chebyshev_distance(cell_a: int, cell_b: int, num_cols: int) -> int:
    """Chebyshev distance between two grid cells. d = max(|Δrow|, |Δcol|)."""
    row_a, col_a = divmod(cell_a, num_cols)
    row_b, col_b = divmod(cell_b, num_cols)
    return max(abs(row_a - row_b), abs(col_a - col_b))


def get_grid_dimensions(grid_details: Dict) -> Tuple[int, int]:
    """Extract num_rows and num_cols from grid_details."""
    num_cells = len(grid_details)
    max_row = max(info['row'] for info in grid_details.values())
    max_col = max(info['col'] for info in grid_details.values())
    num_rows = max_row + 1
    num_cols = max_col + 1
    return num_rows, num_cols


def _get_factorizations(n):
    factors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            factors.append((i, n // i))
    factors.sort(key=lambda pair: pair[1] / pair[0])
    return factors


def _assign_to_cell(x, y, x_min, x_max, y_min, y_max, num_rows, num_cols):
    col = min(max(int((x - x_min) / (x_max - x_min) * num_cols), 0), num_cols - 1)
    row = min(max(int((y - y_min) / (y_max - y_min) * num_rows), 0), num_rows - 1)
    return row * num_cols + col


def _build_grid_details(x_min, x_max, y_min, y_max, num_rows, num_cols,
                        points_of_interest, existing_chargers, gas_stations):
    n = num_rows * num_cols
    cell_w = (x_max - x_min) / num_cols
    cell_h = (y_max - y_min) / num_rows

    grid = {}
    for gid in range(n):
        r, c = divmod(gid, num_cols)
        grid[gid] = {
            'row': r, 'col': c,
            'x_start': x_min + c * cell_w, 'x_end': x_min + (c + 1) * cell_w,
            'y_start': y_min + r * cell_h, 'y_end': y_min + (r + 1) * cell_h,
            'pois': [], 'num_pois': 0,
            'existing_chargers': [], 'has_existing_charger': False, 'num_existing_chargers': 0,
            'gas_stations': [], 'has_gas_station': False, 'num_gas_stations': 0,
        }

    for poi in points_of_interest:
        x, y, density = poi
        cid = _assign_to_cell(x, y, x_min, x_max, y_min, y_max, num_rows, num_cols)
        grid[cid]['pois'].append((x, y, density))
        grid[cid]['num_pois'] += 1

    for ch in existing_chargers:
        cid = _assign_to_cell(ch[0], ch[1], x_min, x_max, y_min, y_max, num_rows, num_cols)
        grid[cid]['existing_chargers'].append(ch)
        grid[cid]['has_existing_charger'] = True
        grid[cid]['num_existing_chargers'] += 1

    for gs in gas_stations:
        cid = _assign_to_cell(gs[0], gs[1], x_min, x_max, y_min, y_max, num_rows, num_cols)
        grid[cid]['gas_stations'].append(gs)
        grid[cid]['has_gas_station'] = True
        grid[cid]['num_gas_stations'] += 1

    return grid


def _plot_grid(x_min, x_max, y_min, y_max, num_rows, num_cols,
               points_of_interest, existing_chargers, gas_stations,
               grid_details, title_suffix=""):
    fig, ax = plt.subplots(1, 1, figsize=STANDARD_FIGSIZE)
    cell_w = (x_max - x_min) / num_cols
    cell_h = (y_max - y_min) / num_rows

    for i in range(num_cols + 1):
        ax.axvline(x=x_min + i * cell_w, color='black', linestyle='--', linewidth=0.5)
    for j in range(num_rows + 1):
        ax.axhline(y=y_min + j * cell_h, color='black', linestyle='--', linewidth=0.5)

    for gid, info in grid_details.items():
        cx = (info['x_start'] + info['x_end']) / 2
        cy = (info['y_start'] + info['y_end']) / 2
        ax.annotate(f'{gid}', (cx, cy), ha='center', va='center',
                    fontsize=max(5, min(8, 200 // max(num_rows, num_cols))),
                    color='gray', alpha=0.7)

    if points_of_interest:
        sc = ax.scatter([p[0] for p in points_of_interest], [p[1] for p in points_of_interest],
                        c=[p[2] for p in points_of_interest], cmap='YlOrRd',
                        edgecolors='green', linewidths=1.5, s=60, vmin=0, vmax=1,
                        zorder=5, label='POIs')
        plt.colorbar(sc, ax=ax, label='POI Density', shrink=0.7)
    if existing_chargers:
        ax.scatter(*zip(*existing_chargers), color='blue', marker='s', s=80,
                   zorder=5, edgecolors='darkblue', linewidths=1, label='Existing Chargers')
    if gas_stations:
        ax.scatter(*zip(*gas_stations), color='orange', marker='^', s=80,
                   zorder=5, edgecolors='darkorange', linewidths=1, label='Gas Stations')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'{num_rows}×{num_cols} Grid ({num_rows * num_cols} cells){title_suffix}')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def _print_grid_summary(grid_details, num_rows, num_cols):
    n = num_rows * num_cols
    cells_poi = sum(1 for v in grid_details.values() if v['num_pois'] > 0)
    cells_ch = sum(1 for v in grid_details.values() if v['has_existing_charger'])
    cells_gs = sum(1 for v in grid_details.values() if v['has_gas_station'])
    total_pois = sum(v['num_pois'] for v in grid_details.values())

    # print(f"Grid Layout: {num_rows} rows × {num_cols} cols = {n} cells")
    # print(f"{'='*60}")
    # print(f"  Total POIs assigned:          {total_pois}")
    # print(f"  Cells with POIs:              {cells_poi} / {n}")
    # print(f"  Cells with existing chargers: {cells_ch} / {n}")
    # print(f"  Cells with gas stations:      {cells_gs} / {n}")
    # print(f"{'─'*60}")
    print("  Grid Info:")
    for gid in sorted(grid_details.keys()):
        info = grid_details[gid]
        if info['num_pois'] > 0 or info['has_existing_charger'] or info['has_gas_station']:
            parts = []
            if info['num_pois'] > 0:
                densities = [f"{p[2]:.2f}" for p in info['pois']]
                parts.append(f"{info['num_pois']} POI(s) [densities: {', '.join(densities)}]")
            if info['has_existing_charger']:
                parts.append(f"{len(info['existing_chargers'])} charger(s)")
            if info['has_gas_station']:
                parts.append(f"{len(info['gas_stations'])} gas station(s)")
            print(f"    Grid {gid:>4} (row={info['row']}, col={info['col']}): {' | '.join(parts)}")


def divide_graph_into_parts(
    x_min: float, x_max: float, y_min: float, y_max: float,
    num_qubits: int,
    points_of_interest: List[Tuple[float, float, float]],
    existing_chargers: List[Tuple[float, float]],
    gas_stations: List[Tuple[float, float]],
    grid_division: Union[str, List[int]] = "default",
    grid_details_flag: bool = True
) -> Union[Dict, List[Dict], None]:
    """
    Divide geographic area into N grid cells and assign data points.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float — geographic boundaries
    num_qubits : int — total number of grid cells / available qubits
    points_of_interest : list of (x, y, density) — density in [0, 1]
    existing_chargers : list of (x, y)
    gas_stations : list of (x, y)
    grid_division :
        "default"            → closest square grid
        "show_possibilities" → plot all valid layouts, return list of options
        [rows, cols]         → explicit layout (rows*cols must == total_cells)
    grid_details_flag : bool — if True, return grid dict; always prints regardless

    Returns
    -------
    "default" / [r,c]: (grid_details, plot_deets) if grid_details_flag=True, else (None, plot_deets)
    "show_possibilities": list of {'layout': [r,c], 'grid_details': dict, 'plot_deets': dict}

    plot_deets keys: x_min, x_max, y_min, y_max, num_rows, num_cols,
                     num_qubits, total_cells, cell_w, cell_h
    """
    total_cells = num_qubits

    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(f"Invalid bounds: [{x_min},{x_max}] × [{y_min},{y_max}]")

    def _run_layout(num_rows, num_cols, suffix=""):
        grid = _build_grid_details(x_min, x_max, y_min, y_max, num_rows, num_cols,
                                   points_of_interest, existing_chargers, gas_stations)
        _plot_grid(x_min, x_max, y_min, y_max, num_rows, num_cols,
                   points_of_interest, existing_chargers, gas_stations, grid, suffix)
        _print_grid_summary(grid, num_rows, num_cols)
        deets = {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'num_rows': num_rows, 'num_cols': num_cols,
            'num_qubits': num_qubits, 'total_cells': total_cells,
            'cell_w': (x_max - x_min) / num_cols,
            'cell_h': (y_max - y_min) / num_rows,
        }
        return grid, deets

    # --- MODE: default ---
    if grid_division == "default":
        num_rows, num_cols = _get_factorizations(total_cells)[0]
        grid, plot_deets = _run_layout(num_rows, num_cols, " [default]")
        return (grid if grid_details_flag else None), plot_deets

    # --- MODE: show_possibilities ---
    elif grid_division == "show_possibilities":
        factorizations = _get_factorizations(total_cells)
        print(f"\nAll valid grid layouts for {total_cells} cells:")
        print(f"{'─'*40}")
        for idx, (r, c) in enumerate(factorizations):
            tag = " ← default (squarest)" if idx == 0 else ""
            print(f"  Option {idx + 1}: {r} × {c}{tag}")
        print(f"{'─'*40}\n")

        results = []
        for idx, (nr, nc) in enumerate(factorizations):
            grid, plot_deets = _run_layout(nr, nc, f" [Option {idx + 1}]")
            results.append({'layout': [nr, nc], 'grid_details': grid, 'plot_deets': plot_deets})
        return results

    # --- MODE: explicit [rows, cols] ---
    elif isinstance(grid_division, list):
        if len(grid_division) != 2:
            raise ValueError(f"grid_division must be [rows, cols], got {grid_division}")
        num_rows, num_cols = grid_division
        if num_rows * num_cols != total_cells:
            raise ValueError(
                f"[{num_rows}, {num_cols}] = {num_rows * num_cols} cells, "
                f"need {total_cells} total cells")
        grid, plot_deets = _run_layout(num_rows, num_cols, f" [custom {num_rows}×{num_cols}]")
        return (grid if grid_details_flag else None), plot_deets

    else:
        raise ValueError(f"grid_division must be 'default', 'show_possibilities', or [rows, cols]. Got: {grid_division}")


def plot_cell_weights(plot_deets, grid_details, cell_weights,
                      points_of_interest=None, existing_chargers=None, gas_stations=None,
                      show_data_points=True, show_weight_values=True,
                      show_grid_ids=True):
    """
    Visualize cell weights as a heatmap.

    Baseline reference: Section 4.3 visualization

    Parameters
    ----------
    plot_deets : dict
        Plotting parameters from divide_graph_into_parts() — ensures consistency across all plots.
        Keys: x_min, x_max, y_min, y_max, num_rows, num_cols, cell_w, cell_h
    grid_details : dict
        Grid data from divide_graph_into_parts()
    cell_weights : dict
        Weights from calculate_cell_weights()
    points_of_interest : list, optional
        POI data for overlay
    existing_chargers : list, optional
        Existing charger data for overlay
    gas_stations : list, optional
        Gas station data for overlay
    show_data_points : bool, default=True
        Whether to overlay data points on heatmap
    show_weight_values : bool, default=True
        Whether to annotate cells with weight values (only practical for small grids)
    show_grid_ids : bool, default=True
        Whether to annotate each cell with its grid ID
    """
    x_min     = plot_deets['x_min']
    x_max     = plot_deets['x_max']
    y_min     = plot_deets['y_min']
    y_max     = plot_deets['y_max']
    num_rows  = plot_deets['num_rows']
    num_cols  = plot_deets['num_cols']
    cell_w    = plot_deets['cell_w']
    cell_h    = plot_deets['cell_h']

    fig, ax = plt.subplots(1, 1, figsize=STANDARD_FIGSIZE)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Create weight matrix — mask zeros so they show as white background
    weight_matrix = np.zeros((num_rows, num_cols))
    for gid, info in grid_details.items():
        weight_matrix[info['row'], info['col']] = cell_weights[gid]['weight']
    weight_matrix_masked = np.ma.masked_where(weight_matrix == 0, weight_matrix)

    # Plot heatmap (left colorbar: Cell Weight)
    im = ax.imshow(weight_matrix_masked, cmap='YlOrRd', origin='lower',
                   extent=[x_min, x_max, y_min, y_max],
                   aspect='auto', interpolation='nearest', alpha=0.6,
                   vmin=0, vmax=max(w['weight'] for w in cell_weights.values()))

    divider = make_axes_locatable(ax)
    cax_left = divider.append_axes("left", size="4%", pad=0.25)
    cax_right = divider.append_axes("right", size="4%", pad=0.25)

    cb_left = fig.colorbar(im, cax=cax_left)
    cb_left.set_label('Cell Weight')
    cax_left.yaxis.set_ticks_position('left')
    cax_left.yaxis.set_label_position('left')

    # Grid lines — match _plot_grid style
    for i in range(num_cols + 1):
        ax.axvline(x=x_min + i * cell_w, color='black', linestyle='--', linewidth=0.5)
    for j in range(num_rows + 1):
        ax.axhline(y=y_min + j * cell_h, color='black', linestyle='--', linewidth=0.5)

    # Annotate cells with grid IDs and, on small grids, weight values.
    if show_grid_ids or (show_weight_values and num_rows * num_cols <= 64):
        for gid, info in grid_details.items():
            cx = (info['x_start'] + info['x_end']) / 2
            cy = (info['y_start'] + info['y_end']) / 2
            weight = cell_weights[gid]['weight']
            label_parts = []
            if show_grid_ids:
                label_parts.append(str(gid))
            if show_weight_values and num_rows * num_cols <= 64 and weight > 0:
                label_parts.append(f'{weight:.2f}')

            if label_parts:
                ax.annotate('\n'.join(label_parts), (cx, cy), ha='center', va='center',
                           fontsize=max(5, min(8, 200 // max(num_rows, num_cols))),
                           color='gray', alpha=0.7)

    # Overlay data points — match _plot_grid styles exactly
    sc = None
    if show_data_points:
        if points_of_interest:
            sc = ax.scatter([p[0] for p in points_of_interest], [p[1] for p in points_of_interest],
                           c=[p[2] for p in points_of_interest], cmap='YlOrRd',
                           edgecolors='green', linewidths=1.5, s=60, vmin=0, vmax=1,
                           zorder=5, label='POIs')
        if existing_chargers:
            ax.scatter(*zip(*existing_chargers), color='blue', marker='s', s=80,
                      zorder=5, edgecolors='darkblue', linewidths=1, label='Existing Chargers')
        if gas_stations:
            ax.scatter(*zip(*gas_stations), color='orange', marker='^', s=80,
                      zorder=5, edgecolors='darkorange', linewidths=1, label='Gas Stations')

    # Right colorbar: POI Density (only when POIs are shown)
    if sc is not None:
        cb_right = fig.colorbar(sc, cax=cax_right)
        cb_right.set_label('POI Density')
    else:
        cax_right.remove()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Cell Weights — {num_rows}×{num_cols} Grid ({num_rows * num_cols} cells)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def print_weight_summary(cell_weights, grid_details, top_n=10):
    """
    Print summary statistics and top-weighted cells as the baseline Grid Data Table.

    Baseline reference: Section 4.4 — Grid Data Table fields:
      Grid ID | # POIs | Raw Score | Norm Score | Weight | Gas Count | Charger Count

    Parameters
    ----------
    cell_weights : dict
        Weights from calculate_cell_weights()
    grid_details : dict
        Grid data from divide_graph_into_parts()
    top_n : int, default=10
        Number of top cells to display
    """
    num_cells = len(cell_weights)
    cells_with_pois = sum(1 for w in cell_weights.values() if w['raw_score'] > 0)
    total_density = sum(w['raw_score'] for w in cell_weights.values())
    max_weight = max(w['weight'] for w in cell_weights.values())
    min_nonzero_weight = min((w['weight'] for w in cell_weights.values() if w['weight'] > 0), default=0)
    total_gas = sum(info['num_gas_stations'] for info in grid_details.values())
    total_chargers = sum(info['num_existing_chargers'] for info in grid_details.values())

    summary_data = [
        ["Total cells", num_cells],
        ["Cells with POIs", f"{cells_with_pois} ({100*cells_with_pois/num_cells:.1f}%)"],
        ["Total aggregated density", f"{total_density:.2f}"],
        ["Max weight", f"{max_weight:.2f}"],
        ["Min non-zero weight", f"{min_nonzero_weight:.2f}"],
        ["Total gas stations", total_gas],
        ["Total existing chargers", total_chargers],
    ]
    print("CELL WEIGHT SUMMARY")
    print(tabulate(summary_data, tablefmt="simple"))

    # Top cells by weight — full Grid Data Table columns
    sorted_cells = sorted(cell_weights.items(), key=lambda x: x[1]['weight'], reverse=True)
    top_cells = [(gid, w) for gid, w in sorted_cells[:top_n] if w['weight'] > 0]

    rows = []
    for gid, weights in top_cells:
        info = grid_details[gid]
        rows.append([
            gid, info['row'], info['col'], info['num_pois'],
            f"{weights['raw_score']:.3f}",
            f"{weights['normalized_score']:.3f}",
            f"{weights['weight']:.3f}",
            info['num_gas_stations'],
            info['num_existing_chargers'],
        ])

    print(f"\nTOP {top_n} CELLS BY WEIGHT (Grid Data Table):")
    print(tabulate(rows,
                   headers=["Grid ID", "Row", "Col", "# POIs", "Raw", "Norm", "Weight",
                             "Gas Sta.", "Chargers"],
                   tablefmt="simple"))

# ─────────────────────────────────────────────────────────────────────────────
# FA-002: Automatic Parameter Suggester
# ─────────────────────────────────────────────────────────────────────────────

def suggest_parameters(grid_details: Dict, cell_weights: Dict, plot_deets: Dict, m: int) -> Dict:
    """
    Suggest a complete starting-point parameter set by analyzing the input grid data.

    FA-002 implementation. Runs AFTER grid construction and weight calculation,
    BEFORE Q matrix construction. All suggestions are overridable.

    Parameters
    ----------
    grid_details : dict
        Grid data from divide_graph_into_parts()
    cell_weights : dict
        Weights from calculate_cell_weights() — must include 'weight' and 'normalized_weight'
    plot_deets : dict
        Plot metadata from divide_graph_into_parts() — provides num_rows, num_cols
    m : int
        Number of new chargers to place

    Returns
    -------
    dict with keys:
        'radii'        : {'R1', 'Rs', 'R3', 'R4', 'R6'}
        'alpha'        : {'a1'–'a6'}
        'intra'        : {'beta', 'gamma', 'delta', 'epsilon'}
        'lambda'       : float
        'notes'        : list[str] — explains why each suggestion was made
        'warnings'     : list[str] — flags degenerate input conditions
        '_magnitudes'  : raw per-term max values before alpha weighting (for debugging)
        '_diagnostics' : derived data characteristics used in the suggestions

    Baseline reference: FA-002 spec, Section 5 (Tunable Parameters Summary)
    """
    num_rows = plot_deets['num_rows']
    num_cols = plot_deets['num_cols']
    N = num_rows * num_cols
    grid_side = int(N ** 0.5)

    # ── Step 1: Radii from grid geometry ─────────────────────────────────────
    # Fractions from FA-002: R1=30%, R3=R4=20%, R6=15% of grid_side
    # Baseline ordering requirement: R1 >= R4 > R3 >= R6
    R1 = max(2, int(grid_side * 0.30))
    Rs = R1
    R3 = max(2, int(grid_side * 0.20))
    R4 = max(2, int(grid_side * 0.20))
    R6 = max(2, int(grid_side * 0.15))

    # Enforce baseline ordering: R1 >= R4 > R3 >= R6
    # R3 and R4 are equal by formula so R4 > R3 is not guaranteed — nudge R3 down if needed
    if R3 >= R4:
        R3 = max(1, R4 - 1)
    if R6 > R3:
        R6 = R3

    # ── Build derived data structures ─────────────────────────────────────────

    # E_all: one entry per individual charger (not per cell) — baseline Section 4 notation
    E_all = []
    for gid, info in grid_details.items():
        for _ in info['existing_chargers']:
            E_all.append(gid)

    # C_POI: cells that contain at least one POI
    C_POI = [gid for gid, info in grid_details.items() if info['num_pois'] > 0]

    # Precompute s_c for each POI cell (service gap factor)
    # s_c = 1 / (1 + Σ_{e in E_all, d(c,e)<=Rs} 1/(1+d(c,e)))
    # Depends only on existing chargers, not x — QUBO-safe
    def _compute_sc(poi_cell):
        total = sum(
            1.0 / (1.0 + chebyshev_distance(poi_cell, e_cell, num_cols))
            for e_cell in E_all
            if chebyshev_distance(poi_cell, e_cell, num_cols) <= Rs
        )
        return 1.0 / (1.0 + total)

    sc_values = {c: _compute_sc(c) for c in C_POI}
    avg_service_gap = float(np.mean(list(sc_values.values()))) if sc_values else 1.0

    # ── Step 2: Per-term magnitude estimation (β=γ=δ=ε=1 for estimation) ────
    # These give the raw max contribution of each term before alpha weighting.
    # After normalization (baseline Section 5), alpha weights become true relative
    # importance — so estimating at β=γ=δ=ε=1 is correct here.

    # H1: For each cell i, compute attraction score
    h1_vals = []
    for i in range(N):
        score = sum(
            cell_weights[c]['weight'] * sc_values[c] / (1.0 + chebyshev_distance(i, c, num_cols))
            for c in C_POI
            if chebyshev_distance(i, c, num_cols) <= R1
        )
        h1_vals.append(score)
    h1_max = max(h1_vals) if h1_vals else 0.0
    nonzero_h1 = [v for v in h1_vals if v > 0]
    h1_median = float(np.median(nonzero_h1)) if nonzero_h1 else 0.0

    # H2: Max gas station bonus (β=1)
    max_gas_count = max(info['num_gas_stations'] for info in grid_details.values())
    h2_max = float(max_gas_count)  # β × g_i, β=1

    # H3: For each cell i, compute existing-charger penalty (γ=1)
    h3_vals = []
    for i in range(N):
        nw_i = cell_weights[i]['normalized_weight']
        score = (1.0 - nw_i) * sum(
            1.0 / (1.0 + chebyshev_distance(i, e, num_cols))
            for e in E_all
            if chebyshev_distance(i, e, num_cols) <= R3
        )
        h3_vals.append(score)
    h3_max = max(h3_vals) if h3_vals else 0.0

    # H4: Max spacing penalty across all nearby pairs (δ=1)
    # Only iterate pairs within R4 — no need to check all N² pairs
    h4_max = 0.0
    for i in range(N):
        row_i, col_i = divmod(i, num_cols)
        for dr in range(-R4, R4 + 1):
            for dc in range(-R4, R4 + 1):
                row_j, col_j = row_i + dr, col_i + dc
                if not (0 <= row_j < num_rows and 0 <= col_j < num_cols):
                    continue
                j = row_j * num_cols + col_j
                if j <= i:
                    continue
                d = max(abs(dr), abs(dc))
                if d > R4:
                    continue
                nw_i = cell_weights[i]['normalized_weight']
                nw_j = cell_weights[j]['normalized_weight']
                val = (1.0 - max(nw_i, nw_j)) / (1.0 + d)
                if val > h4_max:
                    h4_max = val

    # H6: Max shared-coverage penalty across nearby pairs (ε=1)
    # Only iterate pairs within max(R4, 2*R6)
    h6_radius = max(R4, 2 * R6)
    h6_max = 0.0
    for i in range(N):
        row_i, col_i = divmod(i, num_cols)
        for dr in range(-h6_radius, h6_radius + 1):
            for dc in range(-h6_radius, h6_radius + 1):
                row_j, col_j = row_i + dr, col_i + dc
                if not (0 <= row_j < num_rows and 0 <= col_j < num_cols):
                    continue
                j = row_j * num_cols + col_j
                if j <= i:
                    continue
                if max(abs(dr), abs(dc)) > h6_radius:
                    continue
                shared = sum(
                    cell_weights[c]['weight']
                    / ((1.0 + chebyshev_distance(i, c, num_cols))
                       * (1.0 + chebyshev_distance(j, c, num_cols)))
                    for c in C_POI
                    if (chebyshev_distance(i, c, num_cols) <= R6
                        and chebyshev_distance(j, c, num_cols) <= R6)
                )
                if shared > h6_max:
                    h6_max = shared

    # ── Step 3: POI cluster detection ─────────────────────────────────────────
    # Simple BFS flood-fill on C_POI with merge threshold = R6
    def _count_poi_clusters():
        if not C_POI:
            return 0
        visited = set()
        clusters = 0
        for seed in C_POI:
            if seed in visited:
                continue
            clusters += 1
            queue = [seed]
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                for neighbor in C_POI:
                    if (neighbor not in visited
                            and chebyshev_distance(curr, neighbor, num_cols) <= R6):
                        queue.append(neighbor)
        return clusters

    n_clusters = _count_poi_clusters()

    # ── Step 4: Suggest α weights ─────────────────────────────────────────────

    alpha_1 = 3.0  # Always dominant — primary objective driver

    # α₂: Stronger if gas stations overlap with high-density POI areas
    gas_poi_overlap = 0.0
    if C_POI and max_gas_count > 0:
        median_w = float(np.median([cell_weights[c]['weight'] for c in C_POI]))
        gas_cells = [gid for gid, info in grid_details.items() if info['num_gas_stations'] > 0]
        gas_high_density = sum(1 for gid in gas_cells if cell_weights[gid]['weight'] > median_w)
        gas_poi_overlap = gas_high_density / len(gas_cells) if gas_cells else 0.0
    alpha_2 = 1.0 if gas_poi_overlap > 0.3 else 0.5

    # α₃: Scale by how sparse existing coverage is — if well-covered, repulsion matters more
    alpha_3 = round(2.0 * avg_service_gap, 2)

    # α₄: Relax spacing when m forces multiple chargers per cluster
    alpha_4 = 1.5 if (n_clusters == 0 or m <= n_clusters) else 0.5

    alpha_5 = 1.0  # Applied to λ which is set separately below

    # α₆: Relax redundancy when m > n_clusters (forced double-ups)
    if n_clusters == 0:
        alpha_6 = 0.5
    elif m <= n_clusters:
        alpha_6 = 1.5
    else:
        alpha_6 = round(0.5 * (n_clusters / m), 3)

    # ── Step 5: λ (constraint penalty) ───────────────────────────────────────
    # Must make violating Σx_i = m more expensive than any objective gain.
    # After normalization, best diagonal gain per charger ≈ α₁ + α₂.
    # λ × 1 (off by one charger) must exceed (α₁ + α₂) × m — use 5× safety margin.
    max_obj_gain_per_charger = alpha_1 + alpha_2
    lambda_val = round(5.0 * m * max_obj_gain_per_charger, 1)

    # ── Step 6: Intra-term magnitudes β, γ, δ, ε ─────────────────────────────
    # Baseline Section 5: normalization divides each term by its max absolute value
    # before applying α weights, so α values ARE the true relative importance.
    # Setting β=γ=δ=ε=1.0 is correct when normalization is applied.
    beta    = 1.0
    gamma   = 1.0
    delta   = 1.0
    epsilon = 1.0

    # ── Step 7: Notes and warnings ────────────────────────────────────────────
    notes = []
    warnings = []

    coverage_label = (
        'sparse' if avg_service_gap > 0.7
        else 'moderate' if avg_service_gap > 0.4
        else 'dense'
    )
    notes.append(
        f"Average service gap s_c = {avg_service_gap:.2f} → {coverage_label} existing coverage"
        f" → α₃ = {alpha_3}"
    )

    if n_clusters > 0:
        rel = "fewer/equal" if m <= n_clusters else "more"
        notes.append(
            f"Detected {n_clusters} POI cluster(s), m={m} → {rel} chargers than clusters"
            f" → α₄ = {alpha_4}, α₆ = {alpha_6}"
        )

    if max_gas_count > 0:
        proximity = "near" if gas_poi_overlap > 0.3 else "mostly away from"
        notes.append(
            f"Gas stations {proximity} high-density POI areas"
            f" (overlap fraction: {gas_poi_overlap:.2f}) → α₂ = {alpha_2}"
        )

    notes.append(
        f"λ = {lambda_val} set at 5× safety margin over max objective gain"
        f" ({max_obj_gain_per_charger:.1f}) × m ({m})"
    )

    # Warnings for degenerate inputs
    if not C_POI:
        warnings.append(
            "No POIs in dataset — H1, H6, and s_c have no effect. α₁ and α₆ are irrelevant."
        )
    if not E_all:
        warnings.append(
            "No existing chargers — H3 and s_c have no effect. α₃ set but will evaluate to 0."
        )
    if max_gas_count == 0:
        warnings.append(
            "No gas stations — H2 has no effect. α₂ set but will evaluate to 0."
        )
    if n_clusters > 0 and m > 3 * n_clusters:
        warnings.append(
            f"m={m} >> n_clusters={n_clusters}. H6 redundancy penalty will fight H1 attraction."
            f" Consider reducing α₆ further or setting it to 0."
        )
    if m > N // 2:
        warnings.append(
            f"m={m} is large relative to N={N}. H5 constraint is very tight."
            f" Verify λ={lambda_val} is sufficient — may need manual increase."
        )
    if m < 1:
        warnings.append("m < 1 — no chargers to place. All terms are irrelevant.")

    return {
        'radii':  {'R1': R1, 'Rs': Rs, 'R3': R3, 'R4': R4, 'R6': R6},
        'alpha':  {'a1': alpha_1, 'a2': alpha_2, 'a3': alpha_3,
                   'a4': alpha_4, 'a5': alpha_5, 'a6': alpha_6},
        'intra':  {'beta': beta, 'gamma': gamma, 'delta': delta, 'epsilon': epsilon},
        'lambda': lambda_val,
        'notes':   notes,
        'warnings': warnings,
        '_magnitudes': {
            'h1_max':            round(h1_max, 4),
            'h1_median_nonzero': round(h1_median, 4),
            'h2_max':            round(h2_max, 4),
            'h3_max':            round(h3_max, 4),
            'h4_max':            round(h4_max, 4),
            'h6_max':            round(h6_max, 4),
        },
        '_diagnostics': {
            'n_clusters':          n_clusters,
            'avg_service_gap':     round(avg_service_gap, 4),
            'gas_poi_overlap':     round(gas_poi_overlap, 4),
            'n_poi_cells':         len(C_POI),
            'n_existing_chargers': len(E_all),
            'grid_side':           grid_side,
            'N':                   N,
        },
    }


def print_parameter_suggestions(params: Dict) -> None:
    """
    Pretty-print the output of suggest_parameters().

    Parameters
    ----------
    params : dict
        Return value of suggest_parameters()
    """
    SEP = "=" * 62

    print(f"\n{SEP}")
    print("SUGGESTED PARAMETERS")
    print(SEP)

    # Radii
    print("\nLocal Radii:")
    radii_labels = {
        'R1': 'R₁  — H1 POI attraction radius',
        'Rs': 'Rₛ  — service gap radius (matches R₁)',
        'R3': 'R₃  — H3 existing charger penalty radius',
        'R4': 'R₄  — H4 new charger spacing radius',
        'R6': 'R₆  — H6 coverage redundancy radius',
    }
    print(tabulate(
        [[radii_labels[k], v] for k, v in params['radii'].items()],
        headers=["Radius", "Value"], tablefmt="simple"
    ))

    # Alpha weights
    print("\nObjective Weights (α):")
    alpha_labels = {
        'a1': 'α₁  — H1 POI attraction         [dominant]',
        'a2': 'α₂  — H2 gas station bonus',
        'a3': 'α₃  — H3 existing charger penalty',
        'a4': 'α₄  — H4 new charger spacing',
        'a5': 'α₅  — H5 constraint  (applied to λ)',
        'a6': 'α₆  — H6 coverage redundancy',
    }
    print(tabulate(
        [[alpha_labels[k], v] for k, v in params['alpha'].items()],
        headers=["Parameter", "Value"], tablefmt="simple"
    ))

    # Intra-term magnitudes
    print("\nIntra-term Magnitudes:")
    intra_labels = {
        'beta':    'β  — gas station bonus magnitude',
        'gamma':   'γ  — existing charger penalty magnitude',
        'delta':   'δ  — new charger spacing magnitude',
        'epsilon': 'ε  — coverage redundancy magnitude',
    }
    print(tabulate(
        [[intra_labels[k], v] for k, v in params['intra'].items()],
        headers=["Parameter", "Value"], tablefmt="simple"
    ))

    print(f"\nConstraint Penalty:  λ = {params['lambda']}")

    # Raw magnitudes
    if '_magnitudes' in params:
        print("\nRaw Term Magnitudes (before α weighting, β=γ=δ=ε=1):")
        print(tabulate(
            [[k, v] for k, v in params['_magnitudes'].items()],
            headers=["Term", "Max Value"], tablefmt="simple"
        ))

    # Diagnostics
    if '_diagnostics' in params:
        print("\nData Diagnostics:")
        print(tabulate(
            [[k, v] for k, v in params['_diagnostics'].items()],
            headers=["Metric", "Value"], tablefmt="simple"
        ))

    # Notes
    if params.get('notes'):
        print("\nNotes:")
        for note in params['notes']:
            print(f"  • {note}")

    # Warnings
    if params.get('warnings'):
        print(f"\n{'─'*62}")
        print("  WARNINGS:")
        for w in params['warnings']:
            print(f"  ⚠  {w}")

    print(SEP)
