import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union
from tabulate import tabulate


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
            'existing_chargers': [], 'has_existing_charger': False,
            'gas_stations': [], 'has_gas_station': False,
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

    for gs in gas_stations:
        cid = _assign_to_cell(gs[0], gs[1], x_min, x_max, y_min, y_max, num_rows, num_cols)
        grid[cid]['gas_stations'].append(gs)
        grid[cid]['has_gas_station'] = True

    return grid


def _plot_grid(x_min, x_max, y_min, y_max, num_rows, num_cols,
               points_of_interest, existing_chargers, gas_stations,
               grid_details, title_suffix=""):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
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
    Divide geographic area into 2^q grid cells and assign data points.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float — geographic boundaries
    num_qubits : int — total cells = 2^num_qubits
    points_of_interest : list of (x, y, density) — density in [0, 1]
    existing_chargers : list of (x, y)
    gas_stations : list of (x, y)
    grid_division :
        "default"            → closest square grid
        "show_possibilities" → plot all valid layouts, return list of options
        [rows, cols]         → explicit layout (rows*cols must == 2^q)
    grid_details_flag : bool — if True, return grid dict; always prints regardless

    Returns
    -------
    "default" / [r,c]: (grid_details, plot_deets) if grid_details_flag=True, else (None, plot_deets)
    "show_possibilities": list of {'layout': [r,c], 'grid_details': dict, 'plot_deets': dict}

    plot_deets keys: x_min, x_max, y_min, y_max, num_rows, num_cols,
                     num_qubits, total_cells, cell_w, cell_h
    """
    total_cells = 2 ** num_qubits

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
        print(f"\nAll valid grid layouts for {total_cells} cells (2^{num_qubits}):")
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
                f"need 2^{num_qubits} = {total_cells}")
        grid, plot_deets = _run_layout(num_rows, num_cols, f" [custom {num_rows}×{num_cols}]")
        return (grid if grid_details_flag else None), plot_deets

    else:
        raise ValueError(f"grid_division must be 'default', 'show_possibilities', or [rows, cols]. Got: {grid_division}")


def plot_cell_weights(plot_deets, grid_details, cell_weights,
                      points_of_interest=None, existing_chargers=None, gas_stations=None,
                      show_data_points=True, show_weight_values=True):
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
    """
    x_min     = plot_deets['x_min']
    x_max     = plot_deets['x_max']
    y_min     = plot_deets['y_min']
    y_max     = plot_deets['y_max']
    num_rows  = plot_deets['num_rows']
    num_cols  = plot_deets['num_cols']
    cell_w    = plot_deets['cell_w']
    cell_h    = plot_deets['cell_h']

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
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

    cb_left = fig.colorbar(im, ax=ax, location='left', label='Cell Weight', shrink=0.7, pad=0.08)

    # Grid lines — match _plot_grid style
    for i in range(num_cols + 1):
        ax.axvline(x=x_min + i * cell_w, color='black', linestyle='--', linewidth=0.5)
    for j in range(num_rows + 1):
        ax.axhline(y=y_min + j * cell_h, color='black', linestyle='--', linewidth=0.5)

    # Annotate cells with weights (only if grid is small enough)
    if show_weight_values and num_rows * num_cols <= 64:
        for gid, info in grid_details.items():
            cx = (info['x_start'] + info['x_end']) / 2
            cy = (info['y_start'] + info['y_end']) / 2
            weight = cell_weights[gid]['weight']
            if weight > 0:
                ax.annotate(f'{weight:.2f}', (cx, cy), ha='center', va='center',
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
        fig.colorbar(sc, ax=ax, location='right', label='POI Density', shrink=0.7, pad=0.08)

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
    Print summary statistics and top-weighted cells.

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

    summary_data = [
        ["Total cells", num_cells],
        ["Cells with POIs", f"{cells_with_pois} ({100*cells_with_pois/num_cells:.1f}%)"],
        ["Total aggregated density", f"{total_density:.2f}"],
        ["Max weight", f"{max_weight:.2f}"],
        ["Min non-zero weight", f"{min_nonzero_weight:.2f}"],
    ]
    print("CELL WEIGHT SUMMARY")
    print(tabulate(summary_data, tablefmt="simple"))

    # Top cells by weight
    sorted_cells = sorted(cell_weights.items(), key=lambda x: x[1]['weight'], reverse=True)
    top_cells = [(gid, w) for gid, w in sorted_cells[:top_n] if w['weight'] > 0]

    rows = []
    for gid, weights in top_cells:
        info = grid_details[gid]
        rows.append([gid, info['row'], info['col'], info['num_pois'],
                     f"{weights['raw_score']:.3f}",
                     f"{weights['normalized_score']:.3f}",
                     f"{weights['weight']:.3f}"])

    print(f"\nTOP {top_n} CELLS BY WEIGHT:")
    print(tabulate(rows, headers=["Grid ID", "Row", "Col", "POIs", "Raw", "Norm", "Weight"], tablefmt="simple"))