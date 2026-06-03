"""Stage 4 calibration: fit a per-clip image->world homography from GCPs.

Two modes (see docs/calibration_protocol.md):

  1. --dump-frame : extract one full-resolution reference frame from a clip
     so you can read off ground-control-point pixel locations in any image
     viewer (most show pixel coords on hover). Writes a PNG under
     data/video_annotations/calibration/frames/<clip>.png.

  2. (fit)        : read a GCP spec (configs/calibration/<clip>.gcp.yaml),
     fit the homography, report the reprojection RMSE (the error budget)
     and -- if a validation segment is present -- an independent scale
     check, then save the calibration JSON under
     data/video_annotations/calibration/<clip>.json.

The GCP world coordinates are a LOCAL metric ground frame (metres) anchored
to known road-marking geometry; no global georeferencing is needed because
kinematics depend only on relative distance (decision_log D4).

Examples:
    # 1) dump a reference frame to pick GCPs from
    python scripts/00_calibrate.py --video data/video_clips/120_m_DJI_0006_1min.mp4 --dump-frame

    #    (markings occluded by vehicles? choose a clearer frame)
    python scripts/00_calibrate.py --video data/video_clips/120_m_DJI_0006_1min.mp4 --dump-frame --frame 900

    # 2) fit from the filled-in GCP spec
    python scripts/00_calibrate.py --gcp configs/calibration/120_m_DJI_0006_1min.gcp.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trajectory.calibration import (  # noqa: E402
    Calibration, fit_homography, pixel_to_world, save_calibration,
)

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = PROJECT_ROOT / "data/video_annotations/calibration"
DEFAULT_FRAME_DIR = DEFAULT_OUT_DIR / "frames"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, default=None,
                   help="Clip mp4 (for --dump-frame).")
    p.add_argument("--dump-frame", action="store_true",
                   help="Extract a reference frame PNG and exit.")
    p.add_argument("--frame", type=int, default=0,
                   help="Frame index to dump (default 0). Pick a frame where "
                        "road markings are not occluded by vehicles.")
    p.add_argument("--gcp", type=Path, default=None,
                   help="GCP spec YAML to fit a homography from.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def dump_frame(video: Path, frame_idx: int, out_dir: Path) -> int:
    import cv2

    if not video.exists():
        logger.error("Video not found: %s", video)
        return 2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video)
        return 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        logger.error("Could not read frame %d from %s", frame_idx, video.name)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video.stem}.png"
    cv2.imwrite(str(out_path), img)
    h, w = img.shape[:2]
    logger.info("Wrote reference frame %d (%dx%d) -> %s", frame_idx, w, h, out_path)
    logger.info("Open it in an image viewer, read off the GCP pixel coords, "
                "and fill in configs/calibration/%s.gcp.yaml", video.stem)
    return 0


def fit_from_gcp(gcp_path: Path, out_dir: Path) -> int:
    if not gcp_path.exists():
        logger.error("GCP spec not found: %s", gcp_path)
        return 2
    spec = yaml.safe_load(gcp_path.read_text(encoding="utf-8")) or {}
    clip = spec.get("clip") or gcp_path.stem.replace(".gcp", "")
    points = spec.get("points") or []

    pixel, world, names = [], [], []
    for pt in points:
        if pt.get("pixel") is None or pt.get("world") is None:
            logger.warning("skipping incomplete point: %s", pt)
            continue
        pixel.append(pt["pixel"]); world.append(pt["world"])
        names.append(pt.get("name", f"p{len(names)+1}"))

    if len(pixel) < 4:
        logger.error("Need >=4 fully-specified GCPs in %s (found %d). "
                     "Fill in pixel:[u,v] and world:[X,Y] for each point.",
                     gcp_path.name, len(pixel))
        return 1

    H, rmse = fit_homography(pixel, world)
    logger.info("Fit homography for %s from %d GCPs", clip, len(pixel))
    logger.info("  reprojection RMSE = %.4f m", rmse)

    # per-point residuals, so a bad point is easy to spot
    proj = pixel_to_world(pixel, H)
    world_arr = np.asarray(world, dtype=float)
    for nm, p in zip(names, np.linalg.norm(proj - world_arr, axis=1)):
        logger.info("    %-8s residual %.4f m", nm, p)

    # optional independent scale check (catches a globally-wrong reference)
    validation = {}
    val = spec.get("validation")
    if val and val.get("pixel_a") and val.get("pixel_b") and val.get("known_distance_m"):
        a, b = pixel_to_world([val["pixel_a"], val["pixel_b"]], H)
        measured = float(np.linalg.norm(a - b))
        known = float(val["known_distance_m"])
        err = measured - known
        validation = {"known_distance_m": known, "measured_distance_m": round(measured, 4),
                      "error_m": round(err, 4)}
        logger.info("  scale check: known %.3f m -> measured %.3f m (err %+.3f m, %+.1f%%)",
                    known, measured, err, 100.0 * err / known if known else 0.0)

    if rmse > 1.0:
        logger.warning("  reprojection RMSE %.3f m is large (>1 m) -- re-check "
                       "the GCP pixel/world entries before trusting kinematics.", rmse)

    calib = Calibration(clip=clip, H=H.tolist(), rmse_m=round(rmse, 6),
                        n_gcp=len(pixel), units="m",
                        note=str(spec.get("reference", "")), validation=validation)
    out_path = out_dir / f"{clip}.json"
    save_calibration(calib, out_path)
    logger.info("Wrote calibration -> %s", out_path)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()

    if args.dump_frame:
        if args.video is None:
            logger.error("--dump-frame requires --video")
            return 2
        return dump_frame(args.video, args.frame, DEFAULT_FRAME_DIR)

    if args.gcp is not None:
        return fit_from_gcp(args.gcp, args.out_dir)

    logger.error("Nothing to do: pass --dump-frame (with --video) or --gcp <spec>.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
