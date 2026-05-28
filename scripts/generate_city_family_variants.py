from __future__ import annotations

import math
import random
import shutil
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


FAMILIES = {
    "small_city": {
        "folder": "small_city",
        "canonical": "dataset_small_city.xlsx",
        "bounds": {"x_min": 0, "x_max": 25, "y_min": 0, "y_max": 27},
        "m": 4,
        "target_chargers": 4,
        "target_gas": 4,
        "poi_counts": {
            "polycentric": 17,
            "edge_growth": 16,
            "corridor": 15,
            "sparse_suburban": 13,
            "dense_core": 18,
            "underserved_corner": 14,
        },
    },
    "large_city": {
        "folder": "large_city",
        "canonical": "dataset_large_city.xlsx",
        "bounds": {"x_min": 0, "x_max": 35, "y_min": 0, "y_max": 30},
        "m": 7,
        "target_chargers": 8,
        "target_gas": 8,
        "poi_counts": {
            "polycentric": 44,
            "edge_growth": 42,
            "corridor": 40,
            "sparse_suburban": 36,
            "dense_core": 48,
            "underserved_corner": 38,
        },
    },
    "metro_region": {
        "folder": "metro_region",
        "canonical": "dataset_metro_region.xlsx",
        "bounds": {"x_min": 0, "x_max": 40, "y_min": 0, "y_max": 35},
        "m": 10,
        "target_chargers": 12,
        "target_gas": 10,
        "poi_counts": {
            "polycentric": 66,
            "edge_growth": 63,
            "corridor": 59,
            "sparse_suburban": 54,
            "dense_core": 72,
            "underserved_corner": 58,
        },
    },
}


SCENARIOS = {
    "polycentric": "Multiple job and retail centers with balanced demand across districts.",
    "edge_growth": "New demand pushed toward outer growth corridors and city edges.",
    "corridor": "POIs follow a commuter/highway spine with secondary neighborhoods nearby.",
    "sparse_suburban": "Lower-density separated neighborhoods with several small activity centers.",
    "dense_core": "Heavy central-city pressure with secondary districts and low-density fringe.",
    "underserved_corner": "High demand concentrated in one corner while chargers sit mostly elsewhere.",
}


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def point(x: float, y: float) -> dict[str, float]:
    return {"x": round(float(x), 2), "y": round(float(y), 2)}


def poi(x: float, y: float, density: float) -> dict[str, float]:
    return {**point(x, y), "density": round(clamp(density, 0.22, 0.99), 2)}


def unique_add(items: list[dict[str, float]], item: dict[str, float], used: set[tuple[float, float]]) -> None:
    x, y = item["x"], item["y"]
    step = 0
    while (x, y) in used:
        step += 1
        x = round(item["x"] + 0.07 * step, 2)
        y = round(item["y"] + 0.05 * step, 2)
    item = dict(item)
    item["x"], item["y"] = x, y
    used.add((x, y))
    items.append(item)


def scaled_layout(family: dict, scenario: str) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    b = family["bounds"]
    w = b["x_max"] - b["x_min"]
    h = b["y_max"] - b["y_min"]
    sx = lambda v: b["x_min"] + v * w
    sy = lambda v: b["y_min"] + v * h

    if scenario == "polycentric":
        clusters = [
            (sx(0.50), sy(0.52), 0.055, 0.92),
            (sx(0.72), sy(0.67), 0.060, 0.82),
            (sx(0.27), sy(0.72), 0.060, 0.74),
            (sx(0.66), sy(0.27), 0.060, 0.66),
            (sx(0.22), sy(0.24), 0.055, 0.54),
            (sx(0.88), sy(0.39), 0.050, 0.46),
        ]
        chargers = [(sx(0.48), sy(0.52)), (sx(0.55), sy(0.49)), (sx(0.72), sy(0.66)), (sx(0.28), sy(0.72)), (sx(0.66), sy(0.28))]
    elif scenario == "edge_growth":
        clusters = [
            (sx(0.82), sy(0.78), 0.060, 0.93),
            (sx(0.91), sy(0.58), 0.055, 0.79),
            (sx(0.22), sy(0.81), 0.055, 0.68),
            (sx(0.50), sy(0.47), 0.060, 0.61),
            (sx(0.22), sy(0.24), 0.055, 0.48),
            (sx(0.68), sy(0.20), 0.050, 0.43),
        ]
        chargers = [(sx(0.48), sy(0.48)), (sx(0.55), sy(0.51)), (sx(0.23), sy(0.25)), (sx(0.65), sy(0.22)), (sx(0.24), sy(0.80))]
    elif scenario == "corridor":
        clusters = [
            (sx(0.18), sy(0.18), 0.040, 0.52),
            (sx(0.30), sy(0.30), 0.040, 0.65),
            (sx(0.43), sy(0.43), 0.040, 0.80),
            (sx(0.56), sy(0.55), 0.040, 0.92),
            (sx(0.70), sy(0.68), 0.040, 0.76),
            (sx(0.84), sy(0.79), 0.040, 0.58),
            (sx(0.25), sy(0.78), 0.050, 0.61),
            (sx(0.78), sy(0.24), 0.050, 0.55),
        ]
        chargers = [(sx(0.30), sy(0.31)), (sx(0.43), sy(0.43)), (sx(0.56), sy(0.55)), (sx(0.70), sy(0.68)), (sx(0.25), sy(0.78))]
    elif scenario == "sparse_suburban":
        clusters = [
            (sx(0.22), sy(0.23), 0.050, 0.56),
            (sx(0.78), sy(0.24), 0.050, 0.59),
            (sx(0.27), sy(0.78), 0.050, 0.57),
            (sx(0.78), sy(0.78), 0.050, 0.54),
            (sx(0.52), sy(0.51), 0.055, 0.63),
        ]
        chargers = [(sx(0.23), sy(0.24)), (sx(0.78), sy(0.25)), (sx(0.28), sy(0.77)), (sx(0.52), sy(0.50))]
    elif scenario == "dense_core":
        clusters = [
            (sx(0.48), sy(0.49), 0.040, 0.98),
            (sx(0.54), sy(0.52), 0.040, 0.94),
            (sx(0.43), sy(0.55), 0.040, 0.88),
            (sx(0.25), sy(0.72), 0.050, 0.65),
            (sx(0.73), sy(0.64), 0.050, 0.61),
            (sx(0.67), sy(0.24), 0.050, 0.55),
            (sx(0.18), sy(0.24), 0.050, 0.43),
        ]
        chargers = [(sx(0.47), sy(0.49)), (sx(0.52), sy(0.52)), (sx(0.56), sy(0.47)), (sx(0.25), sy(0.72)), (sx(0.73), sy(0.63))]
    elif scenario == "underserved_corner":
        clusters = [
            (sx(0.12), sy(0.13), 0.050, 0.94),
            (sx(0.22), sy(0.20), 0.050, 0.78),
            (sx(0.51), sy(0.50), 0.055, 0.68),
            (sx(0.78), sy(0.72), 0.055, 0.58),
            (sx(0.78), sy(0.24), 0.050, 0.48),
            (sx(0.25), sy(0.74), 0.050, 0.45),
        ]
        chargers = [(sx(0.52), sy(0.50)), (sx(0.58), sy(0.47)), (sx(0.78), sy(0.71)), (sx(0.78), sy(0.25)), (sx(0.26), sy(0.73))]
    else:
        raise ValueError(scenario)

    charger_backfill = [
        (sx(0.18), sy(0.18)),
        (sx(0.35), sy(0.30)),
        (sx(0.50), sy(0.48)),
        (sx(0.65), sy(0.35)),
        (sx(0.82), sy(0.55)),
        (sx(0.72), sy(0.77)),
        (sx(0.42), sy(0.72)),
        (sx(0.18), sy(0.58)),
        (sx(0.90), sy(0.20)),
        (sx(0.08), sy(0.85)),
        (sx(0.57), sy(0.16)),
        (sx(0.92), sy(0.86)),
    ]
    chargers = chargers + [candidate for candidate in charger_backfill if candidate not in chargers]

    gas = [
        (sx(0.15), sy(0.18)),
        (sx(0.34), sy(0.17)),
        (sx(0.58), sy(0.21)),
        (sx(0.82), sy(0.28)),
        (sx(0.86), sy(0.72)),
        (sx(0.22), sy(0.84)),
        (sx(0.45), sy(0.78)),
        (sx(0.67), sy(0.82)),
        (sx(0.08), sy(0.54)),
        (sx(0.94), sy(0.52)),
    ]
    return clusters, chargers, gas


def make_pois(family_name: str, family: dict, scenario: str) -> list[dict[str, float]]:
    count = family["poi_counts"][scenario]
    clusters, _, _ = scaled_layout(family, scenario)
    b = family["bounds"]
    rng = random.Random(f"{family_name}:{scenario}:2026")
    used: set[tuple[float, float]] = set()
    pois: list[dict[str, float]] = []

    weights = [c[3] for c in clusters]
    total = sum(weights)
    allocations = [max(1, round(count * w / total)) for w in weights]
    while sum(allocations) > count:
        i = max(range(len(allocations)), key=lambda idx: allocations[idx])
        allocations[i] -= 1
    while sum(allocations) < count:
        i = max(range(len(allocations)), key=lambda idx: weights[idx] / allocations[idx])
        allocations[i] += 1

    for cluster_idx, ((cx, cy, spread, density), n_points) in enumerate(zip(clusters, allocations)):
        for j in range(n_points):
            radius = spread * (0.45 + rng.random() * 1.5)
            angle = rng.random() * math.tau + j * 0.73 + cluster_idx * 0.31
            x = clamp(cx + math.cos(angle) * radius * (b["x_max"] - b["x_min"]), b["x_min"] + 0.6, b["x_max"] - 0.6)
            y = clamp(cy + math.sin(angle) * radius * (b["y_max"] - b["y_min"]), b["y_min"] + 0.6, b["y_max"] - 0.6)
            local_density = density - 0.055 * j + rng.uniform(-0.035, 0.035)
            unique_add(pois, poi(x, y, local_density), used)

    return pois


def trim_points(points: list[tuple[float, float]], target: int, used: set[tuple[float, float]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for x, y in points[:target]:
        unique_add(out, point(x, y), used)
    return out


def make_dataset(family_name: str, scenario: str) -> dict:
    family = FAMILIES[family_name]
    _, chargers_raw, gas_raw = scaled_layout(family, scenario)
    used = {(p["x"], p["y"]) for p in make_pois(family_name, family, scenario)}
    pois = make_pois(family_name, family, scenario)
    used = {(p["x"], p["y"]) for p in pois}
    chargers = trim_points(chargers_raw, family["target_chargers"], used)
    gas = trim_points(gas_raw, family["target_gas"], used)
    return {
        "name": f"{family_name}_{scenario}",
        "description": f"{len(pois)} POIs. {SCENARIOS[scenario]} m={family['m']}.",
        "m": family["m"],
        "bounds": family["bounds"],
        "pois": pois,
        "chargers": chargers,
        "gas": gas,
    }


def validate(dataset: dict) -> None:
    b = dataset["bounds"]
    coords = []
    for item in dataset["pois"]:
        assert b["x_min"] <= item["x"] <= b["x_max"], (dataset["name"], item)
        assert b["y_min"] <= item["y"] <= b["y_max"], (dataset["name"], item)
        assert 0 <= item["density"] <= 1, (dataset["name"], item)
        coords.append((item["x"], item["y"]))
    for key in ("chargers", "gas"):
        for item in dataset[key]:
            assert b["x_min"] <= item["x"] <= b["x_max"], (dataset["name"], item)
            assert b["y_min"] <= item["y"] <= b["y_max"], (dataset["name"], item)
            coords.append((item["x"], item["y"]))
    assert len(coords) == len(set(coords)), dataset["name"]


def write_dataset(dataset: dict, out_dir: Path) -> Path:
    validate(dataset)
    b = dataset["bounds"]
    path = out_dir / f"dataset_{dataset['name']}.xlsx"
    config = [
        ("dataset_name", dataset["name"]),
        ("description", dataset["description"]),
        ("m", dataset["m"]),
        ("x_min", b["x_min"]),
        ("x_max", b["x_max"]),
        ("y_min", b["y_min"]),
        ("y_max", b["y_max"]),
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


def write_family_index(family_name: str, datasets: list[dict], paths: list[Path], out_dir: Path) -> None:
    title = family_name.replace("_", " ").title()
    b = FAMILIES[family_name]["bounds"]
    lines = [
        f"# {title} Dataset Variants",
        "",
        "Generated by `scripts/generate_city_family_variants.py`.",
        "",
        f"Coordinate frame: x=[{b['x_min']},{b['x_max']}], y=[{b['y_min']},{b['y_max']}].",
        "",
        "| File | POIs | Chargers | Gas | m | Scenario |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for dataset, path in zip(datasets, paths):
        lines.append(
            f"| `{path.name}` | {len(dataset['pois'])} | {len(dataset['chargers'])} | "
            f"{len(dataset['gas'])} | {dataset['m']} | {dataset['description']} |"
        )
    lines.append("")
    lines.append("Suggested smoke-test order: sparse suburban, polycentric, corridor, edge growth, underserved corner, dense core.")
    (out_dir / f"{family_name}_variants_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def organize_canonical(family_name: str, out_dir: Path) -> Path:
    canonical = FAMILIES[family_name]["canonical"]
    src = DATA_DIR / canonical
    dst = out_dir / canonical
    if dst.exists():
        return dst
    if src.exists():
        shutil.move(str(src), str(dst))
        return dst
    raise FileNotFoundError(f"Missing canonical dataset: {src}")


def write_top_index(family_paths: dict[str, list[Path]]) -> None:
    lines = [
        "# Dataset Families",
        "",
        "Non-medium city datasets are organized into per-family folders. Medium-city datasets remain in `data/` because the active medium-city runner reads those exact paths.",
        "",
        "| Family | Folder | Variant files |",
        "| --- | --- | ---: |",
    ]
    for family_name, paths in family_paths.items():
        folder = FAMILIES[family_name]["folder"]
        lines.append(f"| `{family_name}` | `data/{folder}/` | {len(paths)} |")
    lines.extend(
        [
            "",
            "Medium-city files:",
            "",
            "- `data/dataset_medium_city.xlsx`",
            "- `data/dataset_medium_city_sparse_suburban.xlsx`",
            "- `data/dataset_medium_city_polycentric.xlsx`",
            "- `data/dataset_medium_city_corridor.xlsx`",
            "- `data/dataset_medium_city_edge_growth.xlsx`",
            "- `data/dataset_medium_city_underserved_corner.xlsx`",
            "- `data/dataset_medium_city_dense_core.xlsx`",
        ]
    )
    lines.append("")
    lines.append("Run `scripts/generate_city_family_variants.py` to regenerate the non-medium family folders.")
    (DATA_DIR / "dataset_families_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    family_paths: dict[str, list[Path]] = {}
    for family_name, family in FAMILIES.items():
        out_dir = DATA_DIR / family["folder"]
        out_dir.mkdir(exist_ok=True)
        canonical = organize_canonical(family_name, out_dir)
        datasets = [make_dataset(family_name, scenario) for scenario in SCENARIOS]
        paths = [canonical] + [write_dataset(dataset, out_dir) for dataset in datasets]
        write_family_index(family_name, datasets, paths[1:], out_dir)
        family_paths[family_name] = paths
        for path in paths:
            print(path.relative_to(DATA_DIR.parent))
        print((out_dir / f"{family_name}_variants_index.md").relative_to(DATA_DIR.parent))
    write_top_index(family_paths)
    print((DATA_DIR / "dataset_families_index.md").relative_to(DATA_DIR.parent))


if __name__ == "__main__":
    main()
