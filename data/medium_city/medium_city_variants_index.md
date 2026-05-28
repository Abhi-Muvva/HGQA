# Medium City Dataset Variants

These files intentionally remain directly under `data/` because the active medium-city notebooks and runner read these exact paths.

| File | Scenario |
| --- | --- |
| `dataset_medium_city.xlsx` | Canonical medium-city dataset. |
| `dataset_medium_city_sparse_suburban.xlsx` | Sparse suburban layout for low-density coverage tests. |
| `dataset_medium_city_polycentric.xlsx` | Multiple demand centers across city districts. |
| `dataset_medium_city_corridor.xlsx` | Diagonal commuter/highway corridor pattern. |
| `dataset_medium_city_edge_growth.xlsx` | Growth pushed toward city edges and corners. |
| `dataset_medium_city_underserved_corner.xlsx` | High-demand corner underserved by existing chargers. |
| `dataset_medium_city_dense_core.xlsx` | Dense downtown pressure with secondary districts. |

Generated variants come from `scripts/generate_medium_city_variants.py`.

