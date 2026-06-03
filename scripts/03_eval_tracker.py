"""Evaluate tracker outputs against Stage 2 ground truth (HOTA / MOTA / IDF1).

Stage 3 scoring (decision-log D3). For each tracker and each "arm" (raw
tracker output and/or tracker+linker output), this scores every clip that has
a ground-truth file, then combines the clips into one aggregate row using
TrackEval's accumulators. Metrics come from `src/evaluation/tracking_metrics.py`,
which wraps TrackEval -- the MOTChallenge reference implementation.

File conventions:
    GT:           <gt-dir>/<clip>.gt.txt                  (from 02e_unpack_cvat_mot.py)
    raw arm:      <tracker-dir>/<clip>.<tracker>.txt       (from 03_run_tracker.py)
    linked arm:   <tracker-dir>/<clip>.<tracker>.linked.txt(from 02b_link_tracks.py)

Output: a tidy CSV (one row per clip x tracker x arm x scope) plus an
aggregate "ALL" row per tracker x arm x scope. `scope` is "overall"
(class-agnostic) or a vehicle class name when --per-class is given.

Example:
    python scripts/03_eval_tracker.py --arms raw linked --per-class
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.tracking_metrics import (  # noqa: E402
    CLASS_NAMES, combine, evaluate_files, summarize,
)

logger = logging.getLogger(__name__)

TRACKERS = ["bytetrack", "botsort", "ocsort", "deepocsort"]
DEFAULT_GT_DIR = PROJECT_ROOT / "data/video_annotations/ground_truth"
DEFAULT_TRK_DIR = PROJECT_ROOT / "data/video_annotations/tracker_outputs"
DEFAULT_OUT = PROJECT_ROOT / "results/stage3/tracker_metrics.csv"

# Preferred CSV column order; any extra summary keys are appended after.
METRIC_COLS = ["HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1", "IDR",
               "IDP", "IDSW", "Frag", "MT", "PT", "ML", "CLR_TP", "CLR_FP",
               "CLR_FN"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    p.add_argument("--tracker-dir", type=Path, default=DEFAULT_TRK_DIR)
    p.add_argument("--trackers", nargs="+", default=TRACKERS, choices=TRACKERS)
    p.add_argument("--arms", nargs="+", default=["raw"], choices=["raw", "linked"],
                   help="Which output arms to score. Default: raw.")
    p.add_argument("--per-class", action="store_true",
                   help="Also report per-class (LV/MV/SV) in addition to overall.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output CSV. Default: {DEFAULT_OUT}")
    return p.parse_args()


def tracker_file(trk_dir: Path, clip: str, tracker: str, arm: str) -> Path:
    suffix = "txt" if arm == "raw" else "linked.txt"
    return trk_dir / f"{clip}.{tracker}.{suffix}"


def scopes(per_class: bool) -> list[tuple[str, int | None]]:
    out: list[tuple[str, int | None]] = [("overall", None)]
    if per_class:
        out += [(name, cid) for cid, name in enumerate(CLASS_NAMES)]
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()

    gt_files = sorted(args.gt_dir.glob("*.gt.txt"))
    if not gt_files:
        logger.error("No ground-truth files (*.gt.txt) in %s", args.gt_dir)
        logger.error("Stage 2 annotation must be exported first "
                     "(see 02e_unpack_cvat_mot.py).")
        return 2
    clips = [f.name[: -len(".gt.txt")] for f in gt_files]
    logger.info("Found %d GT clip(s): %s", len(clips), ", ".join(clips))

    rows: list[dict] = []
    for tracker in args.trackers:
        for arm in args.arms:
            for scope_name, cls in scopes(args.per_class):
                per_clip_raw: list[dict] = []
                for clip in clips:
                    tf = tracker_file(args.tracker_dir, clip, tracker, arm)
                    if not tf.exists():
                        logger.warning("missing %s -- skipping", tf.name)
                        continue
                    gt = args.gt_dir / f"{clip}.gt.txt"
                    raw = evaluate_files(gt, tf, cls=cls)
                    per_clip_raw.append(raw)
                    s = summarize(raw)
                    rows.append(dict(clip=clip, tracker=tracker, arm=arm,
                                     scope=scope_name, **s))
                    logger.info("%-22s %-10s %-6s %-7s  HOTA=%.3f MOTA=%.3f IDF1=%.3f",
                                clip, tracker, arm, scope_name,
                                s["HOTA"], s["MOTA"], s["IDF1"])
                if per_clip_raw:
                    agg = summarize(combine(per_clip_raw))
                    rows.append(dict(clip="ALL", tracker=tracker, arm=arm,
                                     scope=scope_name, **agg))

    if not rows:
        logger.error("No (clip, tracker) pairs evaluated -- check --tracker-dir.")
        return 1

    # union of metric keys, preferred order first
    extra = [k for k in rows[0] if k not in ("clip", "tracker", "arm", "scope")
             and k not in METRIC_COLS]
    metric_header = [k for k in METRIC_COLS if any(k in r for r in rows)] + extra
    header = ["clip", "tracker", "arm", "scope"] + metric_header

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
