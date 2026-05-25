"""
HGQA — Hybrid Quantum-Genetic Algorithm for EV Charging Station Placement.

Public API re-exported for convenience:

    from src.data_loader import load_dataset
    from src.helpers import divide_graph_into_parts, calculate_cell_weights
    from src.qubo_builder import build_qubo, evaluate_solution
    from src.cell_pruner import prune_cells, remap_qubo_for_qaoa
    from src.qaoa_builder import run_qaoa, run_qaoa_mps
    from src.ga_solver import run_ga
    from src.pruning_validation import run_pruning_validation
"""
