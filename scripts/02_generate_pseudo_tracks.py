"""Run a detector + ByteTrack on a video clip and emit MOT-format pseudo-tracks.

Stage 2 of the project: seeds tracker-evaluation annotations that a human
corrects in CVAT to produce Stage 3 ground truth.

Output (MOT Challenge txt, one row per detection per frame, 1-indexed):
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, -1, -1

Columns 9 and 10 (visibility, z) are emitted as -1. The class column carries
the dataset class ID (0=LV, 1=MV, 2=SV).

Example:
    python scripts/02_generate_pseudo_tracks.py \\
        --video data/video_clips/120_m_DJI_0006_1min.mp4 \\
        --weights runs/stage1_detectors/rtdetr_l_seed0/weights/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", type=Path, required=True,
                   help="Input video clip (mp4/mov).")
    p.add_argument("--weights", type=Path, required=True,
                   help="Detector weights (e.g. RT-DETR-L best.pt from Stage 1).")
    p.add_argument("--tracker-config", type=Path,
                   default=Path("configs/trackers/bytetrack.yaml"),
                   help="Ultralytics-compatible tracker YAML.")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Detector confidence threshold (default 0.25 -- "
                        "over-detect for annotator correction).")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Inference image size (matches Stage 1 training).")
    p.add_argument("--iou", type=float, default=0.6,
                   help="Detector NMS IoU threshold.")
    p.add_argument("--device", default="0",
                   help="CUDA device index or 'cpu'.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/video_annotations/pseudo_tracks"),
                   help="Directory for MOT-format output.")
    p.add_argument("--save-vis", action="store_true",
                   help="Also save a copy of the video with overlaid boxes "
                        "and track IDs for human spot-check.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    for label, path in [("video", args.video),
                        ("weights", args.weights),
                        ("tracker-config", args.tracker_config)]:
        if not path.exists():
            logger.error("%s not found: %s", label, path)
            return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.video.stem}.txt"

    # Late import: keeps --help fast and avoids importing torch on argparse errors.
    from ultralytics import YOLO

    logger.info("Detector: %s", args.weights)
    model = YOLO(str(args.weights))

    logger.info("Tracking %s  (conf=%.2f imgsz=%d tracker=%s)",
                args.video.name, args.conf, args.imgsz,
                args.tracker_config.name)

    t0 = time.monotonic()
    results_iter = model.track(
        source=str(args.video),
        tracker=str(args.tracker_config),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        persist=False,
        verbose=False,
        save=args.save_vis,
        # Ultralytics nests output under runs/{task}/ when `project` is
        # relative; resolve to absolute so the vis lands beside the MOT txt.
        project=str(args.output_dir.resolve()) if args.save_vis else None,
        name=f"{args.video.stem}_vis" if args.save_vis else None,
    )

    n_frames = 0
    n_detections = 0
    track_ids: set[int] = set()

    with out_path.open("w", encoding="utf-8") as f:
        for frame_idx, res in enumerate(results_iter, start=1):
            n_frames = frame_idx
            if res.boxes is None or len(res.boxes) == 0:
                continue
            # ByteTrack may yield boxes without IDs (rejected during
            # association); skip them since MOT format requires an ID.
            if res.boxes.id is None:
                continue

            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            ids = res.boxes.id.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c, k, tid in zip(xyxy, conf, cls, ids):
                w = x2 - x1
                h = y2 - y1
                f.write(
                    f"{frame_idx},{tid},{x1:.2f},{y1:.2f},"
                    f"{w:.2f},{h:.2f},{c:.4f},{k},-1,-1\n"
                )
                n_detections += 1
                track_ids.add(int(tid))

            if frame_idx % 100 == 0:
                logger.info("  frame %4d   tracks=%d   dets=%d",
                            frame_idx, len(track_ids), n_detections)

    elapsed = time.monotonic() - t0
    fps = n_frames / elapsed if elapsed > 0 else 0.0

    logger.info("Wrote %s", out_path)
    logger.info("  frames=%d  unique-tracks=%d  total-detections=%d",
                n_frames, len(track_ids), n_detections)
    logger.info("  elapsed=%.1fs  throughput=%.1f fps", elapsed, fps)

    return 0


if __name__ == "__main__":
    sys.exit(main())
