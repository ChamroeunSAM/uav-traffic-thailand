"""Extract world-coordinate vehicle trajectories with kinematics (Stage 4, Phase B).

Reads a MOT-format track file (a Stage 3 tracker output or the Stage 2 GT),
projects each track's per-frame box centre to metric world coordinates via the
clip's calibration homography, then per track:
  - splits into segments wherever a frame gap exceeds --max-gap-fill (so motion
    is never fabricated across a long occlusion the linker stitched over),
  - linearly interpolates SHORT gaps onto a uniform per-frame grid,
  - Savitzky-Golay-smooths the world position (decision_log D4), and
  - derives kinematics (speed, acceleration, heading, jerk) at dt = 1/fps.

Box centre = (x + w/2, y + h/2). Output is a long-format table (one row per
frame per track segment) saved as an .npz under
data/video_annotations/trajectories/.

Run on the GT first and eyeball the speed summary before trusting any tracker.

Examples:
    python scripts/04_extract_trajectories.py \\
        --tracks data/video_annotations/ground_truth/120_m_DJI_0006_1min.gt.txt \\
        --clip 120_m_DJI_0006_1min

    python scripts/04_extract_trajectories.py \\
        --tracks data/video_annotations/tracker_outputs/120_m_DJI_0006_1min.botsort.linked.txt \\
        --clip 120_m_DJI_0006_1min
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trajectory.calibration import load_calibration, pixel_to_world  # noqa: E402
from trajectory.kinematics import compute_kinematics  # noqa: E402
from trajectory.smoothing import savgol_smooth_xy  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CALIB_DIR = PROJECT_ROOT / "data/video_annotations/calibration"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/video_annotations/trajectories"
FPS = 29.97

COLUMNS = ["frame", "track_id", "seg", "cls", "x_m", "y_m", "xs_m", "ys_m",
           "vx", "vy", "speed_mps", "ax", "ay", "accel_mps2", "heading_deg",
           "jerk_mps3", "speed_raw_mps", "gap_filled"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracks", type=Path, required=True,
                   help="MOT-format track file (tracker output or GT).")
    p.add_argument("--clip", default=None,
                   help="Clip stem for calibration lookup (default: inferred "
                        "from the tracks filename).")
    p.add_argument("--calibration", type=Path, default=None,
                   help="Calibration JSON. Default: <calib-dir>/<clip>.json")
    p.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    p.add_argument("--fps", type=float, default=FPS)
    p.add_argument("--smooth-window", type=int, default=15,
                   help="Savitzky-Golay window in frames (odd; ~0.5s at 30fps).")
    p.add_argument("--smooth-polyorder", type=int, default=2)
    p.add_argument("--max-gap-fill", type=int, default=15,
                   help="Frame gaps <= this are interpolated; larger gaps split "
                        "the track into separate segments. Default 15 (~0.5s).")
    p.add_argument("--max-speed", type=float, default=42.0,
                   help="Implied speed (m/s) above which a jump between "
                        "consecutive detections is treated as an identity error "
                        "and splits the track. Default 42 m/s (~151 km/h).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--out", type=Path, default=None,
                   help="Output .npz. Default: <out-dir>/<tracks-stem>.npz")
    return p.parse_args()


KNOWN_STEMS = ["100_m_DJI_0001_1min", "100_m_DJI_0054_1min",
               "120_m_DJI_0006_1min", "120_m_DJI_0047_1min"]


def infer_clip(tracks: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = tracks.name
    for stem in KNOWN_STEMS:
        if stem in name:
            return stem
    raise SystemExit(f"Could not infer clip from '{tracks.name}'; pass --clip.")


def load_mot(path: Path) -> dict[int, list[tuple]]:
    """track_id -> list of (frame, cx, cy, cls) sorted by frame."""
    tracks: dict[int, list[tuple]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        a = line.split(",")
        frame, tid = int(a[0]), int(a[1])
        x, y, w, h = (float(a[i]) for i in range(2, 6))
        cls = int(float(a[7])) if len(a) > 7 else -1
        tracks[tid].append((frame, x + w / 2.0, y + h / 2.0, cls))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return tracks


def split_segments(frames, world, dt, max_gap_fill, max_speed):
    """Split a track into contiguous [start, end) index runs.

    A break is inserted between consecutive detections when EITHER the frame
    gap exceeds ``max_gap_fill`` (a long occlusion we won't interpolate over)
    OR the implied speed between them exceeds ``max_speed`` -- a
    physically-impossible jump, i.e. an identity error, treated like a gap so
    it neither gets interpolated into a sustained false speed nor
    differentiated into a spike. Returns (segments, n_teleport_breaks).
    """
    n = len(frames)
    if n < 2:
        return [(0, n)], 0
    df = np.diff(frames).astype(float)
    disp = np.linalg.norm(np.diff(world, axis=0), axis=1)
    implied = disp / (df * dt)
    gap_break = df > max_gap_fill
    tele_break = implied > max_speed
    breaks = np.where(gap_break | tele_break)[0] + 1
    n_tele = int((tele_break & ~gap_break).sum())
    bounds = [0, *breaks.tolist(), n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)], n_tele


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()

    if not args.tracks.exists():
        logger.error("Tracks file not found: %s", args.tracks)
        return 2
    clip = infer_clip(args.tracks, args.clip)
    calib_path = args.calibration or (args.calib_dir / f"{clip}.json")
    if not calib_path.exists():
        logger.error("Calibration not found: %s (run 00_calibrate.py first)", calib_path)
        return 2

    calib = load_calibration(calib_path)
    H = calib.matrix()
    dt = 1.0 / args.fps
    logger.info("Clip %s | calib rmse=%.4f m | dt=%.5f s (%.2f fps)",
                clip, calib.rmse_m, dt, args.fps)

    tracks = load_mot(args.tracks)
    logger.info("Loaded %d tracks from %s", len(tracks), args.tracks.name)

    rows: list[list[float]] = []
    n_seg = 0
    teleport_breaks = 0
    for tid, recs in tracks.items():
        frames = np.array([r[0] for r in recs], dtype=int)
        px = np.array([[r[1], r[2]] for r in recs], dtype=float)
        cls_arr = np.array([r[3] for r in recs], dtype=int)
        world = pixel_to_world(px, H)            # (N,2) metres

        # dedupe duplicate frames (keep first) so the grid is strictly increasing
        _, uniq = np.unique(frames, return_index=True)
        frames, world, cls_arr = frames[uniq], world[uniq], cls_arr[uniq]

        segs, n_tele = split_segments(frames, world, dt, args.max_gap_fill, args.max_speed)
        teleport_breaks += n_tele
        for (s, e) in segs:
            sf, sw, sc = frames[s:e], world[s:e], cls_arr[s:e]
            grid = np.arange(sf[0], sf[-1] + 1)
            xs = np.interp(grid, sf, sw[:, 0])
            ys = np.interp(grid, sf, sw[:, 1])
            cls_g = np.interp(grid, sf, sc).round().astype(int)
            gap = ~np.isin(grid, sf)
            raw_xy = np.column_stack([xs, ys])

            sm = savgol_smooth_xy(raw_xy, args.smooth_window, args.smooth_polyorder)
            kin = compute_kinematics(sm, dt)
            speed_raw = compute_kinematics(raw_xy, dt)["speed"]

            for i, fr in enumerate(grid):
                rows.append([
                    int(fr), tid, n_seg, int(cls_g[i]),
                    raw_xy[i, 0], raw_xy[i, 1], sm[i, 0], sm[i, 1],
                    kin["velocity"][i, 0], kin["velocity"][i, 1], kin["speed"][i],
                    kin["accel"][i, 0], kin["accel"][i, 1], kin["accel_mag"][i],
                    kin["heading_deg"][i], kin["jerk_mag"][i], speed_raw[i],
                    float(gap[i]),
                ])
            n_seg += 1

    if not rows:
        logger.error("No trajectory rows produced.")
        return 1
    data = np.asarray(rows, dtype=np.float64)

    # --- sanity summary (eyeball plausibility) ---
    speed_kmh = data[:, COLUMNS.index("speed_mps")] * 3.6
    accel = data[:, COLUMNS.index("accel_mps2")]
    gapfrac = data[:, COLUMNS.index("gap_filled")].mean()
    logger.info("Trajectories: %d tracks, %d segments, %d rows", len(tracks), n_seg, len(data))
    logger.info("  speed km/h: median=%.1f  85th=%.1f  95th=%.1f  max=%.1f",
                np.median(speed_kmh), np.percentile(speed_kmh, 85),
                np.percentile(speed_kmh, 95), speed_kmh.max())
    logger.info("  |accel| m/s^2: median=%.2f  95th=%.2f  max=%.2f",
                np.median(accel), np.percentile(accel, 95), accel.max())
    logger.info("  gap-filled rows: %.1f%%  | teleport splits (impossible jumps): %d",
                100.0 * gapfrac, teleport_breaks)

    out = args.out or (args.out_dir / f"{args.tracks.stem}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(clip=clip, tracks_file=str(args.tracks), calib_rmse_m=calib.rmse_m,
                fps=args.fps, smooth_window=args.smooth_window,
                smooth_polyorder=args.smooth_polyorder,
                max_gap_fill=args.max_gap_fill, max_speed=args.max_speed,
                n_tracks=len(tracks), n_segments=n_seg, n_rows=len(data),
                teleport_breaks=teleport_breaks)
    np.savez_compressed(out, data=data, columns=np.array(COLUMNS),
                        meta=json.dumps(meta))
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
