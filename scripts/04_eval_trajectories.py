"""Score Stage 4 world trajectories along the three D4 axes -> results CSV.

Stage 4 Phase C / D. For every clip with a calibration + GT, this scores each
tracker's *linked* trajectory against GT on:

  Axis 1  identity persistence   (completeness, frags/GT, ID switches)
  Axis 2  kinematic accuracy     (ADE/FDE/RMSE m, speed MAE m/s, heading deg)
  Axis 3  physical plausibility  (accel-in-bounds, jerk, curvature, speed) -- GT-free

and runs the **GT trajectories through the same pipeline as a reference
ceiling** (Axis 3 only -- GT has no GT to match against). Every accuracy /
plausibility metric is reported twice, for the raw and the smoothed series, so
smoothing is a studied variable (decision_log D4).

Trajectory ``.npz`` inputs are produced by ``04_extract_trajectories.py``; any
missing one is extracted on demand (``--auto-extract``, on by default) by
shelling out to that script, so a fresh checkout can go GT.txt -> results in one
command. Metrics are class-agnostic and averaged across clips (D4).

Output: ``results/stage4/trajectory_metrics.csv`` -- one row per
(clip, source, variant), plus an aggregate ``ALL`` row per (source, variant).

Example:
    python scripts/04_eval_trajectories.py                 # all clips, all trackers
    python scripts/04_eval_trajectories.py --no-auto-extract   # require pre-extracted npz
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.trajectory_metrics import (  # noqa: E402
    TrajParams, evaluate_against_gt, evaluate_plausibility_only,
)

logger = logging.getLogger(__name__)

TRACKERS = ["bytetrack", "botsort", "ocsort", "deepocsort"]
DEFAULT_GT_DIR = PROJECT_ROOT / "data/video_annotations/ground_truth"
DEFAULT_TRK_DIR = PROJECT_ROOT / "data/video_annotations/tracker_outputs"
DEFAULT_TRAJ_DIR = PROJECT_ROOT / "data/video_annotations/trajectories"
DEFAULT_OUT = PROJECT_ROOT / "results/stage4/trajectory_metrics.csv"
EXTRACT_SCRIPT = PROJECT_ROOT / "scripts/04_extract_trajectories.py"

# Preferred column order; any extra keys are appended after these.
METRIC_COLS = [
    # Axis 1 -- identity persistence
    "a1_n_gt_tracks", "a1_completeness_mean", "a1_frag_per_gt_mean",
    "a1_id_switches", "a1_mostly_tracked_frac",
    # Axis 2 -- kinematic accuracy
    "a2_n_matched", "a2_ade_m", "a2_fde_m", "a2_pos_rmse_m",
    "a2_speed_mae_mps", "a2_heading_mae_deg",
    # Axis 3 -- physical plausibility (GT-free)
    "a3_n_samples", "a3_frac_accel_ok", "a3_frac_speed_ok",
    "a3_accel_p95_mps2", "a3_accel_max_mps2", "a3_jerk_mean_mps3",
    "a3_jerk_p95_mps3", "a3_curv_mean_inv_m",
    "a3_speed_median_kmh", "a3_speed_p95_kmh", "a3_speed_max_kmh",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    p.add_argument("--tracker-dir", type=Path, default=DEFAULT_TRK_DIR)
    p.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    p.add_argument("--trackers", nargs="+", default=TRACKERS, choices=TRACKERS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--auto-extract", dest="auto_extract", action="store_true",
                   default=True, help="Extract any missing trajectory npz (default).")
    p.add_argument("--no-auto-extract", dest="auto_extract", action="store_false",
                   help="Fail on a missing trajectory npz instead of extracting it.")
    # plausibility / matching thresholds (D4 reference bounds)
    p.add_argument("--a-max", type=float, default=TrajParams.a_max)
    p.add_argument("--speed-cap", type=float, default=TrajParams.speed_cap)
    p.add_argument("--match-gate", type=float, default=TrajParams.match_gate)
    return p.parse_args()


def ensure_traj(traj_npz: Path, tracks_txt: Path, clip: str, auto: bool) -> bool:
    """Make sure ``traj_npz`` exists, extracting from ``tracks_txt`` if allowed."""
    if traj_npz.exists():
        return True
    if not auto:
        logger.error("missing trajectory %s (run 04_extract_trajectories.py or "
                     "drop --no-auto-extract)", traj_npz.name)
        return False
    if not tracks_txt.exists():
        logger.warning("cannot extract %s -- source tracks %s absent",
                       traj_npz.name, tracks_txt.name)
        return False
    logger.info("extracting %s from %s ...", traj_npz.name, tracks_txt.name)
    r = subprocess.run(
        [sys.executable, str(EXTRACT_SCRIPT), "--tracks", str(tracks_txt),
         "--clip", clip, "--out", str(traj_npz)],
        cwd=str(PROJECT_ROOT))
    if r.returncode != 0 or not traj_npz.exists():
        logger.error("extraction failed for %s (rc=%s)", traj_npz.name, r.returncode)
        return False
    return True


def aggregate(rows: list[dict], source: str, variant: str) -> dict | None:
    """Mean across clips for one (source, variant); skip non-aggregable keys."""
    sel = [r for r in rows if r["source"] == source and r["variant"] == variant
           and r["clip"] != "ALL"]
    if not sel:
        return None
    agg = {"clip": "ALL", "source": source, "variant": variant}
    keys = [k for k in sel[0] if k not in ("clip", "source", "variant")]
    for k in keys:
        vals = [r[k] for r in sel if isinstance(r.get(k), (int, float))
                and not (isinstance(r[k], float) and np.isnan(r[k]))]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    return agg


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    params = TrajParams(a_max=args.a_max, speed_cap=args.speed_cap,
                        match_gate=args.match_gate)

    gt_files = sorted(args.gt_dir.glob("*.gt.txt"))
    if not gt_files:
        logger.error("No ground-truth files (*.gt.txt) in %s", args.gt_dir)
        return 2
    clips = [f.name[: -len(".gt.txt")] for f in gt_files]
    logger.info("Found %d GT clip(s): %s", len(clips), ", ".join(clips))

    rows: list[dict] = []
    for clip in clips:
        # GT trajectory (ceiling -- Axis 3 only)
        gt_npz = args.traj_dir / f"{clip}.gt.npz"
        gt_txt = args.gt_dir / f"{clip}.gt.txt"
        if not ensure_traj(gt_npz, gt_txt, clip, args.auto_extract):
            logger.warning("skipping clip %s (no GT trajectory)", clip)
            continue
        for variant, mt in evaluate_plausibility_only(gt_npz, params).items():
            rows.append(dict(clip=clip, source="gt", variant=variant, **mt))

        # each tracker's linked trajectory (all three axes vs GT)
        for tracker in args.trackers:
            pr_npz = args.traj_dir / f"{clip}.{tracker}.linked.npz"
            pr_txt = args.tracker_dir / f"{clip}.{tracker}.linked.txt"
            if not ensure_traj(pr_npz, pr_txt, clip, args.auto_extract):
                continue
            res = evaluate_against_gt(pr_npz, gt_npz, params)
            for variant, mt in res.items():
                rows.append(dict(clip=clip, source=tracker, variant=variant, **mt))
            sm = res["smoothed"]
            logger.info("%-22s %-10s  compl=%.2f  ADE=%.2fm  speedMAE=%.2f  "
                        "accel_ok=%.2f",
                        clip, tracker, sm["a1_completeness_mean"], sm["a2_ade_m"],
                        sm["a2_speed_mae_mps"], sm["a3_frac_accel_ok"])

    if not rows:
        logger.error("No trajectories scored.")
        return 1

    # aggregate ALL rows per (source, variant)
    sources = ["gt"] + list(args.trackers)
    for source in sources:
        for variant in ("smoothed", "raw"):
            agg = aggregate(rows, source, variant)
            if agg:
                rows.append(agg)

    # header: identity columns + metric columns present, preferred order first
    extra = [k for k in rows[0] if k not in ("clip", "source", "variant")
             and k not in METRIC_COLS]
    metric_header = [k for k in METRIC_COLS if any(k in r for r in rows)] + extra
    header = ["clip", "source", "variant"] + metric_header

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
    logger.info("Wrote %d rows -> %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
