from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BOUNDS = {"x_min": 0, "x_max": 28, "y_min": 0, "y_max": 27}


def p(x, y, density):
    return {"x": round(float(x), 2), "y": round(float(y), 2), "density": round(float(density), 2)}


def xy(x, y):
    return {"x": round(float(x), 2), "y": round(float(y), 2)}


DATASETS = [
    {
        "name": "medium_city_polycentric",
        "description": "27 POIs across downtown, hospital/university, retail, and residential districts; mixed dense and moderate clusters, m=5.",
        "m": 5,
        "pois": [
            p(13.8, 14.1, 0.96), p(14.9, 13.2, 0.91), p(12.6, 15.4, 0.88), p(15.7, 15.0, 0.82),
            p(20.9, 18.5, 0.87), p(22.1, 19.3, 0.80), p(21.0, 16.7, 0.74), p(23.0, 17.8, 0.69),
            p(6.8, 20.7, 0.79), p(8.1, 21.9, 0.73), p(7.5, 18.8, 0.66), p(5.6, 19.5, 0.58),
            p(18.2, 7.4, 0.72), p(19.7, 6.2, 0.67), p(16.8, 8.7, 0.61), p(21.4, 8.8, 0.54),
            p(4.2, 6.1, 0.51), p(6.1, 7.9, 0.47), p(2.5, 12.8, 0.40), p(25.7, 23.4, 0.43),
            p(26.0, 10.2, 0.45), p(10.4, 3.2, 0.38), p(12.0, 24.5, 0.46), p(3.0, 24.2, 0.34),
            p(24.6, 4.4, 0.36), p(15.0, 21.8, 0.63), p(11.2, 9.5, 0.57),
        ],
        "chargers": [xy(13.0, 14.0), xy(15.5, 13.8), xy(21.6, 18.4), xy(7.1, 20.2), xy(18.9, 7.2), xy(5.7, 7.2)],
        "gas": [xy(4.0, 13.0), xy(11.0, 4.0), xy(17.0, 10.0), xy(24.0, 8.0), xy(23.5, 23.0)],
    },
    {
        "name": "medium_city_edge_growth",
        "description": "26 POIs with new growth pushed to north/east edges and a weaker old core; tests corner and boundary behavior, m=5.",
        "m": 5,
        "pois": [
            p(24.9, 22.8, 0.94), p(26.1, 23.9, 0.88), p(23.7, 25.1, 0.81), p(26.7, 20.7, 0.74),
            p(25.4, 17.8, 0.78), p(27.1, 15.6, 0.70), p(22.8, 14.4, 0.63),
            p(5.2, 23.8, 0.72), p(3.8, 21.9, 0.66), p(7.1, 25.5, 0.61),
            p(13.7, 13.4, 0.69), p(14.8, 14.7, 0.64), p(12.3, 12.1, 0.55), p(16.0, 11.8, 0.49),
            p(4.8, 5.4, 0.52), p(6.2, 7.0, 0.46), p(8.4, 4.2, 0.40),
            p(19.5, 4.8, 0.50), p(22.2, 5.9, 0.44), p(25.7, 7.2, 0.42),
            p(1.3, 15.8, 0.39), p(2.5, 10.4, 0.33), p(10.9, 20.2, 0.58), p(17.8, 22.0, 0.56),
            p(20.0, 26.1, 0.47), p(27.4, 2.9, 0.31),
        ],
        "chargers": [xy(13.5, 13.5), xy(14.8, 15.1), xy(5.8, 6.5), xy(22.8, 6.2), xy(4.5, 22.5), xy(20.2, 18.0)],
        "gas": [xy(2.0, 10.0), xy(9.0, 4.0), xy(18.0, 5.0), xy(26.0, 8.0), xy(25.0, 24.0), xy(6.0, 25.0)],
    },
    {
        "name": "medium_city_corridor",
        "description": "25 POIs along a diagonal commuter/highway corridor plus two off-corridor neighborhoods; tests linear spatial bias, m=5.",
        "m": 5,
        "pois": [
            p(3.2, 4.4, 0.48), p(5.4, 6.2, 0.55), p(7.6, 8.1, 0.63), p(9.7, 9.8, 0.73),
            p(11.8, 11.4, 0.84), p(13.9, 13.1, 0.92), p(16.2, 14.8, 0.89), p(18.4, 16.8, 0.80),
            p(20.5, 18.7, 0.74), p(22.7, 20.2, 0.66), p(24.9, 22.0, 0.58),
            p(10.9, 14.8, 0.77), p(15.1, 10.5, 0.71), p(19.2, 13.2, 0.60),
            p(6.3, 21.2, 0.68), p(7.9, 23.0, 0.62), p(4.8, 19.4, 0.53),
            p(22.3, 6.4, 0.61), p(24.1, 8.5, 0.57), p(20.6, 4.9, 0.49),
            p(2.1, 14.2, 0.35), p(13.2, 24.3, 0.38), p(26.4, 15.3, 0.41), p(15.8, 3.1, 0.36),
            p(27.1, 25.4, 0.30),
        ],
        "chargers": [xy(8.0, 8.5), xy(12.5, 12.0), xy(16.8, 15.5), xy(21.0, 19.0), xy(6.8, 21.8), xy(23.0, 7.0)],
        "gas": [xy(4.0, 5.0), xy(10.0, 10.0), xy(15.0, 14.0), xy(20.0, 18.0), xy(25.0, 22.0)],
    },
    {
        "name": "medium_city_sparse_suburban",
        "description": "22 POIs, lower-density suburban pattern with a few neighborhood centers and many separated POIs; tests sparse coverage, m=5.",
        "m": 5,
        "pois": [
            p(5.4, 5.8, 0.58), p(7.1, 7.2, 0.51), p(4.2, 8.6, 0.44),
            p(20.8, 6.1, 0.62), p(23.0, 7.4, 0.55), p(18.9, 8.6, 0.49),
            p(8.5, 20.8, 0.60), p(10.2, 22.4, 0.52), p(6.7, 23.1, 0.46),
            p(21.6, 21.0, 0.57), p(24.2, 23.2, 0.50),
            p(14.1, 14.0, 0.66), p(15.8, 12.3, 0.59), p(12.4, 15.9, 0.54),
            p(2.0, 15.5, 0.32), p(13.5, 3.0, 0.35), p(26.3, 13.2, 0.38), p(15.0, 25.2, 0.34),
            p(1.8, 25.3, 0.28), p(26.0, 2.4, 0.30), p(3.2, 2.7, 0.26), p(25.7, 25.1, 0.33),
        ],
        "chargers": [xy(6.0, 6.5), xy(21.5, 7.0), xy(9.0, 21.0), xy(14.5, 13.8), xy(22.5, 22.0)],
        "gas": [xy(3.0, 4.0), xy(13.0, 3.0), xy(24.0, 6.0), xy(25.0, 15.0), xy(8.0, 24.0), xy(20.0, 24.0)],
    },
    {
        "name": "medium_city_dense_core",
        "description": "30 POIs with heavy downtown pressure, secondary districts, and low-density fringe; tests high medium-city POI count, m=5.",
        "m": 5,
        "pois": [
            p(13.1, 13.2, 0.98), p(14.0, 13.7, 0.96), p(14.8, 14.4, 0.94), p(12.4, 14.7, 0.90),
            p(15.6, 12.7, 0.88), p(11.8, 12.4, 0.85), p(13.6, 15.5, 0.83), p(16.3, 15.2, 0.79),
            p(10.6, 15.9, 0.76), p(17.2, 11.5, 0.72),
            p(6.0, 19.7, 0.68), p(7.5, 21.1, 0.61), p(5.0, 22.4, 0.55),
            p(21.5, 17.7, 0.66), p(23.1, 19.0, 0.60), p(22.4, 15.4, 0.53),
            p(18.7, 6.6, 0.58), p(20.6, 7.9, 0.52), p(16.4, 5.1, 0.47),
            p(4.2, 6.2, 0.42), p(2.4, 11.8, 0.36), p(3.1, 25.0, 0.31), p(9.7, 3.7, 0.39),
            p(12.6, 24.1, 0.45), p(17.5, 23.2, 0.49), p(24.8, 24.2, 0.38), p(26.2, 10.8, 0.41),
            p(25.4, 4.4, 0.34), p(1.6, 3.0, 0.25), p(27.0, 26.0, 0.29),
        ],
        "chargers": [xy(12.8, 13.5), xy(14.4, 14.2), xy(15.8, 12.8), xy(6.5, 20.4), xy(22.0, 18.2), xy(19.5, 7.2)],
        "gas": [xy(5.0, 8.0), xy(9.0, 4.0), xy(18.0, 6.0), xy(24.0, 11.0), xy(23.0, 23.0)],
    },
    {
        "name": "medium_city_underserved_corner",
        "description": "24 POIs with high demand in the southwest corner but existing chargers mostly central/east; tests underserved-corner placement, m=5.",
        "m": 5,
        "pois": [
            p(2.4, 3.3, 0.93), p(4.0, 4.6, 0.88), p(5.6, 2.5, 0.81), p(3.1, 6.4, 0.77), p(6.8, 5.7, 0.70),
            p(12.8, 13.9, 0.75), p(14.4, 15.2, 0.70), p(15.9, 12.8, 0.64), p(11.7, 12.1, 0.57),
            p(22.8, 20.5, 0.63), p(24.7, 22.1, 0.56), p(21.2, 23.4, 0.48),
            p(20.4, 6.4, 0.52), p(23.3, 8.0, 0.46), p(25.8, 5.5, 0.39),
            p(6.2, 18.9, 0.55), p(8.0, 21.5, 0.50), p(4.1, 24.0, 0.43),
            p(10.1, 6.8, 0.44), p(16.9, 20.3, 0.41), p(26.3, 15.7, 0.37), p(18.2, 2.6, 0.34),
            p(1.2, 13.8, 0.31), p(14.1, 25.0, 0.33),
        ],
        "chargers": [xy(13.8, 14.0), xy(15.2, 13.5), xy(22.5, 20.8), xy(21.0, 7.0), xy(7.2, 20.0), xy(25.0, 14.8)],
        "gas": [xy(2.0, 4.0), xy(7.0, 7.0), xy(12.0, 6.0), xy(18.0, 3.0), xy(24.0, 8.0), xy(25.0, 21.0)],
    },
]


def validate(dataset):
    coords = []
    for poi in dataset["pois"]:
        assert BOUNDS["x_min"] <= poi["x"] <= BOUNDS["x_max"], (dataset["name"], poi)
        assert BOUNDS["y_min"] <= poi["y"] <= BOUNDS["y_max"], (dataset["name"], poi)
        assert 0 <= poi["density"] <= 1, (dataset["name"], poi)
        coords.append((poi["x"], poi["y"]))
    for key in ("chargers", "gas"):
        for point in dataset[key]:
            assert BOUNDS["x_min"] <= point["x"] <= BOUNDS["x_max"], (dataset["name"], point)
            assert BOUNDS["y_min"] <= point["y"] <= BOUNDS["y_max"], (dataset["name"], point)
            coords.append((point["x"], point["y"]))
    assert len(coords) == len(set(coords)), dataset["name"]


def write_dataset(dataset):
    validate(dataset)
    path = DATA_DIR / f"dataset_{dataset['name']}.xlsx"
    config = [
        ("dataset_name", dataset["name"]),
        ("description", dataset["description"]),
        ("m", dataset["m"]),
        ("x_min", BOUNDS["x_min"]),
        ("x_max", BOUNDS["x_max"]),
        ("y_min", BOUNDS["y_min"]),
        ("y_max", BOUNDS["y_max"]),
        ("n_pois", len(dataset["pois"])),
        ("n_chargers", len(dataset["chargers"])),
        ("n_gas", len(dataset["gas"])),
    ]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(config).to_excel(writer, sheet_name="config", header=False, index=False)
        pd.DataFrame(dataset["pois"], columns=["x", "y", "density"]).to_excel(writer, sheet_name="pois", index=False)
        pd.DataFrame(dataset["chargers"], columns=["x", "y"]).to_excel(writer, sheet_name="existing_chargers", index=False)
        pd.DataFrame(dataset["gas"], columns=["x", "y"]).to_excel(writer, sheet_name="gas_stations", index=False)
    return path


def write_index(paths):
    lines = [
        "# Medium City Dataset Variants",
        "",
        "Generated by `scripts/generate_medium_city_variants.py`.",
        "",
        "These files intentionally remain directly under `data/` because the active medium-city notebooks and runner read these exact paths.",
        "",
        "All variants keep the same coordinate frame as the original medium city dataset: x=[0,28], y=[0,27]. They are intended for medium-scale testing without overwriting `dataset_medium_city.xlsx`.",
        "",
        "| File | POIs | Chargers | Gas | m | Scenario |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for dataset, path in zip(DATASETS, paths):
        lines.append(
            f"| `{path.name}` | {len(dataset['pois'])} | {len(dataset['chargers'])} | "
            f"{len(dataset['gas'])} | {dataset['m']} | {dataset['description']} |"
        )
    lines.append("")
    lines.append("Suggested smoke-test order: sparse suburban, polycentric, corridor, edge growth, underserved corner, dense core.")
    (DATA_DIR / "medium_city_variants_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    paths = [write_dataset(dataset) for dataset in DATASETS]
    write_index(paths)
    for path in paths:
        print(path.relative_to(DATA_DIR.parent))
    print((DATA_DIR / "medium_city_variants_index.md").relative_to(DATA_DIR.parent))


if __name__ == "__main__":
    main()
