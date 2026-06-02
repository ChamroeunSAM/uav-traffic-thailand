"""Render a MOT-format track file over its source video as an MP4 overlay.

Used to visually validate pseudo-tracks or linked tracks. Each track ID gets
a stable color so the same vehicle keeps the same color across frames; the
label is "<id> <class>" (LV / MV / SV). Watch for "teleporting" tracks --
the visual signature of a false merge from the linker.

Example:
    python scripts/02c_render_mot.py \\
        --video data/video_clips/100_m_DJI_0054_1min.mp4 \\
        --mot data/video_annotations/pseudo_tracks/100_m_DJI_0054_1min.linked.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES = ["LV", "MV", "SV"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", type=Path, required=True,
                   help="Source video (matches what produced the MOT file).")
    p.add_argument("--mot", type=Path, required=True,
                   help="MOT-format txt to render.")
    p.add_argument("--output", type=Path, default=None,
                   help="Default: <mot-dir>/<mot-stem>.vis.mp4")
    p.add_argument("--thickness", type=int, default=3,
                   help="Box edge thickness. Default 3.")
    p.add_argument("--font-scale", type=float, default=1.0,
                   help="Label font scale. Default 1.0.")
    return p.parse_args()


def id_color(tid: int) -> tuple[int, int, int]:
    """Stable BGR color per track ID via golden-angle hue selection."""
    h = (int(tid) * 137) % 180
    hsv = np.array([[[h, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def load_mot_by_frame(path: Path) -> dict[int, list[tuple]]:
    by_frame: dict[int, list[tuple]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            cls = int(parts[7])
            by_frame[frame].append((tid, x, y, w, h, conf, cls))
    return dict(by_frame)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.video.exists():
        logger.error("Video not found: %s", args.video)
        return 2
    if not args.mot.exists():
        logger.error("MOT file not found: %s", args.mot)
        return 2

    out_path = args.output or args.mot.with_suffix(".vis.mp4")

    logger.info("Loading MOT: %s", args.mot)
    by_frame = load_mot_by_frame(args.mot)
    n_unique = len({r[0] for rows in by_frame.values() for r in rows})
    logger.info("  frames with detections: %d   unique track IDs: %d",
                len(by_frame), n_unique)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", args.video)
        return 2

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # mp4v works without external codecs on Windows OpenCV builds. avc1/H.264
    # would be smaller but requires openh264 which isn't always present.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        logger.error("Cannot open VideoWriter for %s", out_path)
        cap.release()
        return 2

    logger.info("Rendering -> %s  (%dx%d @ %.2f fps, %d frames)",
                out_path, w, h, fps, n_frames)

    t0 = time.monotonic()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        for tid, x, y, bw, bh, _conf, cls in by_frame.get(frame_idx, []):
            color = id_color(tid)
            x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, args.thickness)
            cls_name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
            label = f"{tid} {cls_name}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, 2,
            )
            label_y = max(y1, th + 4)
            cv2.rectangle(
                frame,
                (x1, label_y - th - 4),
                (x1 + tw + 4, label_y),
                color, -1,
            )
            cv2.putText(
                frame, label, (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, args.font_scale,
                (255, 255, 255), 2,
            )

        writer.write(frame)
        if frame_idx % 200 == 0:
            elapsed = time.monotonic() - t0
            logger.info("  frame %d/%d  (%.1f fps render)",
                        frame_idx, n_frames, frame_idx / elapsed)

    cap.release()
    writer.release()
    elapsed = time.monotonic() - t0
    logger.info("Done: %d frames in %.1fs (%.1f fps render)",
                frame_idx, elapsed, frame_idx / elapsed)
    logger.info("Output: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
