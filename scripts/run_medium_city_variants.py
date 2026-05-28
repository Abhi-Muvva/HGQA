#!/usr/bin/env python3
"""Run the medium-city HGQA notebook across medium-city variants.

The script waits for an already-running notebook kernel to calm down before
starting the remaining variants. Each variant is executed from a temporary
Python copy generated from notebooks/07_hgqa_medium_city_runner.ipynb with
only DATASET_FILE changed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "07_hgqa_medium_city_runner.ipynb"
RESULTS = ROOT / "results"
WORK = ROOT / "results" / "_medium_variant_runner"
LOG = WORK / "runner.log"

VARIANTS = [
    "medium_city_sparse_suburban",
    "medium_city_polycentric",
    "medium_city_corridor",
    "medium_city_edge_growth",
    "medium_city_underserved_corner",
    "medium_city_dense_core",
]

DATASET_RE = re.compile(
    r'DATASET_FILE\s*=\s*os\.path\.join\(DATA_DIR,\s*"dataset_[^"]+\.xlsx"\)'
)
PROGRESS_RE = re.compile(
    r"Restart\s+(?P<restart>\d+)/(?P<total_restarts>\d+).*?"
    r"(?P<eval>\d+)/(?P<total_eval>\d+)\s+\["
    r"(?P<elapsed_h>\d+):(?P<elapsed_m>\d+)"
    r".*?,\s*(?P<sec_per_eval>[0-9.]+)s/eval\]"
)


def log(message: str) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(cmd: list[str], *, cwd: Path = ROOT, stdout=None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), text=True, stdout=stdout, stderr=subprocess.STDOUT)


def ps_cpu(pid: int) -> float | None:
    proc = run(["ps", "-o", "%cpu=", "-p", str(pid)], stdout=subprocess.PIPE)
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    if not text:
        return None
    try:
        return float(text.splitlines()[-1].strip())
    except ValueError:
        return None


def latest_notebook_progress() -> dict[str, float | int] | None:
    try:
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    except Exception:
        return None

    matches = []
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if "QAOA on pruned Q" not in source:
            continue
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            matches.extend(PROGRESS_RE.finditer(text))

    if not matches:
        return None
    match = matches[-1]
    return {
        "restart": int(match.group("restart")),
        "total_restarts": int(match.group("total_restarts")),
        "eval": int(match.group("eval")),
        "total_eval": int(match.group("total_eval")),
        "sec_per_eval": float(match.group("sec_per_eval")),
    }


def next_progress_sleep(default_seconds: int) -> int:
    progress = latest_notebook_progress()
    if progress is None:
        return default_seconds

    eval_left = max(0, int(progress["total_eval"]) - int(progress["eval"]))
    seconds_to_restart_tail = eval_left * float(progress["sec_per_eval"])
    if seconds_to_restart_tail <= 0:
        return 300
    return int(max(300, min(1800, seconds_to_restart_tail + 90)))


def wait_for_pid_idle(pid: int | None, idle_cpu: float, idle_checks: int, poll_seconds: int) -> None:
    if pid is None:
        log("No live kernel PID supplied; starting queued variant runs now.")
        return

    log(f"Watching live kernel PID {pid}; waiting for {idle_checks} idle checks below {idle_cpu:.1f}% CPU.")
    quiet = 0
    while True:
        cpu = ps_cpu(pid)
        if cpu is None:
            log(f"PID {pid} is no longer present; continuing.")
            return
        if cpu < idle_cpu:
            quiet += 1
            log(f"PID {pid} CPU {cpu:.1f}% ({quiet}/{idle_checks} idle checks).")
            if quiet >= idle_checks:
                return
        else:
            quiet = 0
            next_sleep = next_progress_sleep(poll_seconds)
            progress = latest_notebook_progress()
            if progress:
                log(
                    "PID "
                    f"{pid} active at {cpu:.1f}% CPU; latest notebook progress "
                    f"restart {progress['restart']}/{progress['total_restarts']} "
                    f"eval {progress['eval']}/{progress['total_eval']} "
                    f"@ {progress['sec_per_eval']:.2f}s/eval; next check in {next_sleep}s."
                )
            else:
                log(f"PID {pid} still active at {cpu:.1f}% CPU; next check in {next_sleep}s.")
        time.sleep(next_sleep if cpu >= idle_cpu else poll_seconds)


def notebook_to_script(dataset_name: str) -> Path:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    parts: list[str] = [
        "# Auto-generated from notebooks/07_hgqa_medium_city_runner.ipynb.\n",
        "# Do not edit by hand; edit the notebook instead.\n",
        "import matplotlib\n",
        "matplotlib.use('Agg')\n\n",
    ]

    replacement = f'DATASET_FILE = os.path.join(DATA_DIR, "dataset_{dataset_name}.xlsx")'
    replaced = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("%")
        )
        if "DATASET_FILE = os.path.join" in source:
            source, count = DATASET_RE.subn(replacement, source, count=1)
            replaced = replaced or count == 1
        if source.strip():
            parts.append("\n# %%\n")
            parts.append(source.rstrip() + "\n")

    if not replaced:
        raise RuntimeError("Could not replace DATASET_FILE in mk07 notebook.")

    WORK.mkdir(parents=True, exist_ok=True)
    script = WORK / f"medium_city_runner_{dataset_name}.py"
    script.write_text("".join(parts), encoding="utf-8")
    return script


def existing_result_dirs(dataset_name: str) -> set[Path]:
    return set(RESULTS.glob(f"{dataset_name}_196q_p10/run_*"))


def newest_plain_medium_dir(before: set[Path]) -> Path | None:
    candidates = set((RESULTS / "medium_city_196q_p10").glob("run_*")) if (RESULTS / "medium_city_196q_p10").exists() else set()
    new_dirs = [path for path in candidates - before if (path / "summary.json").exists()]
    if not new_dirs:
        return None
    return max(new_dirs, key=lambda p: p.stat().st_mtime)


def next_run_dir(base: Path) -> Path:
    idx = 0
    while (base / f"run_{idx}").exists():
        idx += 1
    return base / f"run_{idx}"


def normalize_plain_result(dataset_name: str, before_plain: set[Path]) -> None:
    plain = newest_plain_medium_dir(before_plain)
    if plain is None:
        return
    target_base = RESULTS / f"{dataset_name}_196q_p10"
    target = next_run_dir(target_base)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(plain), str(target))
    log(f"Renamed plain medium-city result {plain} -> {target}.")


def run_variant(dataset_name: str) -> None:
    before_plain = set((RESULTS / "medium_city_196q_p10").glob("run_*")) if (RESULTS / "medium_city_196q_p10").exists() else set()
    before_variant = existing_result_dirs(dataset_name)
    script = notebook_to_script(dataset_name)
    log(f"Starting {dataset_name} via {script.name}.")
    log_path = WORK / f"{dataset_name}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(ROOT / "notebooks"),
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "MPLCONFIGDIR": str(WORK / "mplconfig")},
        )
        rc = proc.wait()
    if rc != 0:
        log(f"{dataset_name} failed with exit code {rc}; see {log_path}. Stopping queue.")
        raise SystemExit(rc)

    normalize_plain_result(dataset_name, before_plain)
    after_variant = existing_result_dirs(dataset_name)
    new_dirs = sorted(after_variant - before_variant, key=lambda p: p.stat().st_mtime)
    if new_dirs:
        log(f"Finished {dataset_name}; saved {new_dirs[-1]}.")
    else:
        log(f"Finished {dataset_name}; no new variant-named result directory detected.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--current-variant")
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--idle-cpu", type=float, default=10.0)
    parser.add_argument("--idle-checks", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=600)
    args = parser.parse_args()

    skip = set(args.skip)
    before_current_plain = (
        set((RESULTS / "medium_city_196q_p10").glob("run_*"))
        if args.current_variant and (RESULTS / "medium_city_196q_p10").exists()
        else set()
    )
    wait_for_pid_idle(args.wait_pid, args.idle_cpu, args.idle_checks, args.poll_seconds)
    if args.current_variant:
        normalize_plain_result(args.current_variant, before_current_plain)

    for dataset_name in VARIANTS:
        if dataset_name in skip:
            log(f"Skipping {dataset_name}.")
            continue
        run_variant(dataset_name)

    log("All queued medium-city variants complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
