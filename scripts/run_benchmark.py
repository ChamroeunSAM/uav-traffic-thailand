"""Orchestrate the full Stage 1 benchmark — N detectors x M seeds.

Usage
-----
    python scripts/run_benchmark.py                       # full 4 detectors x 5 seeds
    python scripts/run_benchmark.py --pilot                # pilot config only
    python scripts/run_benchmark.py --detectors yolo11m yolo12m
    python scripts/run_benchmark.py --seeds 0 1 2
    python scripts/run_benchmark.py --skip-existing        # skip runs with results.json

Behaviour
---------
1. Build a list of (config_path, seed) pairs from the flags.
2. For each pair: train via 01_train_detector.py, then evaluate via
   01_eval_detector.py. Each is a subprocess so a crash in one run cannot
   poison the orchestrator's state.
3. After all runs: aggregate results/stage1/all_runs.csv into
   results/stage1/aggregated.csv (mean / std across seeds).
4. Estimate total GPU time up front and ask for confirmation before
   launching anything heavy (full benchmark only).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.io import load_config, load_json  # noqa: E402
from utils.logging import configure_logging  # noqa: E402

logger = logging.getLogger("run_benchmark")

CONFIG_DIR = PROJECT_ROOT / "configs" / "detectors"
ALL_DETECTORS = ("yolov9m", "yolov10m", "yolo11m", "yolo12m", "yolo26m", "rtdetr_l")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
PILOT_CONFIG = "pilot_yolo11n"

# Rough GPU-hour estimates (per-run on 32GB benchmark hardware). Used only
# for the upfront confirmation; actual durations land in
# results/stage1/aggregated.csv.
ROUGH_HOURS_PER_RUN = {
    "yolov9m": 1.5,
    "yolov10m": 1.5,
    "yolo11m": 2.0,
    "yolo12m": 3.5,    # smaller batch (4) and area-attention overhead
    "yolo26m": 1.5,
    "rtdetr_l": 2.5,
    "pilot_yolo11n": 0.3,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Stage 1 detector benchmark.")
    p.add_argument("--pilot", action="store_true", help="Run only the pilot config.")
    p.add_argument(
        "--detectors", nargs="+", default=None,
        help=f"Subset of detector names. Default: {ALL_DETECTORS}",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help=f"Subset of seeds. Default: {list(DEFAULT_SEEDS)}",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip runs whose results.json already exists.")
    p.add_argument("--no-eval", action="store_true", help="Train only; skip evaluation step.")
    p.add_argument("--yes", action="store_true", help="Skip the GPU-time confirmation prompt.")
    return p.parse_args()


def build_run_list(args) -> list[tuple[str, Path, int]]:
    if args.pilot:
        return [(PILOT_CONFIG, CONFIG_DIR / f"{PILOT_CONFIG}.yaml", 42)]

    detectors = tuple(args.detectors) if args.detectors else ALL_DETECTORS
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    runs = []
    for det in detectors:
        cfg_path = CONFIG_DIR / f"{det}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"No config for detector {det!r} at {cfg_path}")
        for seed in seeds:
            runs.append((det, cfg_path, seed))
    return runs


def estimate_total_hours(runs: list[tuple[str, Path, int]]) -> float:
    return sum(ROUGH_HOURS_PER_RUN.get(det, 5.0) for det, _, _ in runs)


def output_dir_for(cfg_path: Path, seed: int) -> Path:
    cfg = load_config(cfg_path)
    name = cfg["name"].format(seed=seed)
    return PROJECT_ROOT / cfg["project"] / name


def already_done(out_dir: Path) -> bool:
    return (out_dir / "results.json").exists()


def run_subprocess(cmd: list[str]) -> int:
    logger.info("$ %s", " ".join(cmd))
    return subprocess.call(cmd)


def aggregate_runs(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists():
        logger.warning("No runs CSV found at %s; skipping aggregation.", csv_path)
        return
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        logger.warning("Runs CSV is empty.")
        return

    metric_cols = [
        "overall_mAP50", "overall_mAP50_95",
        "100m_mAP50", "100m_mAP50_95",
        "120m_mAP50", "120m_mAP50_95",
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["model_name"]].append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_name", "n_seeds"] + sum(
        ([f"{m}_mean", f"{m}_std"] for m in metric_cols), []
    )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for model, items in grouped.items():
            row: dict = {"model_name": model, "n_seeds": len(items)}
            for m in metric_cols:
                vals = []
                for it in items:
                    try:
                        v = float(it[m])
                        if not math.isnan(v):
                            vals.append(v)
                    except (TypeError, ValueError, KeyError):
                        pass
                row[f"{m}_mean"] = statistics.mean(vals) if vals else float("nan")
                row[f"{m}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            w.writerow(row)
    logger.info("Aggregated %d models -> %s", len(grouped), out_path)


def main() -> int:
    configure_logging()
    args = parse_args()
    runs = build_run_list(args)

    print("=" * 78)
    print(f"Stage 1 benchmark — {len(runs)} runs queued")
    for det, cfg, seed in runs:
        print(f"  - {det:>16s}  seed={seed}  config={cfg.name}")
    total_h = estimate_total_hours(runs)
    print(f"  estimated GPU-hours: ~{total_h:.1f} (rough)")
    print("=" * 78)

    if not args.yes and total_h > 1.0:
        ans = input(f"Proceed with ~{total_h:.1f} GPU-hours of training? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return 1

    train_script = PROJECT_ROOT / "scripts" / "01_train_detector.py"
    eval_script = PROJECT_ROOT / "scripts" / "01_eval_detector.py"

    started = datetime.now(timezone.utc)
    for det, cfg_path, seed in runs:
        out_dir = output_dir_for(cfg_path, seed)
        if args.skip_existing and already_done(out_dir):
            logger.info("Skipping %s (results.json exists)", out_dir)
            continue

        rc = run_subprocess([sys.executable, str(train_script), "--config", str(cfg_path), "--seed", str(seed)])
        if rc != 0:
            logger.error("Training failed for %s seed=%d (exit %d). Continuing.", det, seed, rc)
            continue

        if args.no_eval:
            continue

        results = load_json(out_dir / "results.json")
        weights = PROJECT_ROOT / results["best_weights_path"]
        rc = run_subprocess([sys.executable, str(eval_script), "--weights", str(weights), "--config", str(cfg_path)])
        if rc != 0:
            logger.error("Evaluation failed for %s seed=%d (exit %d).", det, seed, rc)

    aggregate_runs(
        PROJECT_ROOT / "results" / "stage1" / "all_runs.csv",
        PROJECT_ROOT / "results" / "stage1" / "aggregated.csv",
    )
    finished = datetime.now(timezone.utc)
    print(f"\nBenchmark wall-clock: {finished - started}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
