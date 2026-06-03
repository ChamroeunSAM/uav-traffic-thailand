"""Run a BoxMOT tracker over RT-DETR-L detections and emit MOT-format tracks.

Stage 3 of the project. Each of the four trackers (ByteTrack, BoT-SORT,
OC-SORT, Deep OC-SORT) is coupled to the Stage 1 winner detector (RT-DETR-L)
and scored against the Stage 2 ground truth (decision-log D3).

DETECT ONCE, TRACK FOUR TIMES. RT-DETR-L is run over the clip a single time
and the per-frame detections are cached to `<clip>.detections.npy`. Every
subsequent tracker run on the same clip reuses that cache, so the *only*
variable across the four trackers is the association algorithm -- never the
detector. The cache is an (M,7) float32 array of rows:

    [frame (1-indexed), x1, y1, x2, y2, conf, cls]

Output (MOT-Challenge txt, 1-indexed frames, 0-based class to match the GT
produced by 02e_unpack_cvat_mot.py):

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, -1, -1

REID CAVEAT (BoT-SORT, Deep OC-SORT): BoxMOT's default ReID weights are
PERSON re-identification models (osnet / MSMT17). Top-down vehicles are not
what those encoders were trained on. A vehicle-appropriate ReID model is the
correct Stage 3 choice and remains an open decision; `--reid-weights` lets
you swap it. The default is recorded so results are at least reproducible.

Examples:
    # ByteTrack on the roundabout clip (creates the detection cache)
    python scripts/03_run_tracker.py \\
        --video data/video_clips/120_m_DJI_0006_1min.mp4 --tracker bytetrack

    # ...then OC-SORT reuses the cache (no re-detection)
    python scripts/03_run_tracker.py \\
        --video data/video_clips/120_m_DJI_0006_1min.mp4 --tracker ocsort
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# tracker key -> (wrapper module, wrapper class)
TRACKERS = {
    "bytetrack": ("trackers.bytetrack_wrapper", "ByteTrackWrapper"),
    "botsort": ("trackers.botsort_wrapper", "BotSortWrapper"),
    "ocsort": ("trackers.ocsort_wrapper", "OcSortWrapper"),
    "deepocsort": ("trackers.deep_ocsort_wrapper", "DeepOcSortWrapper"),
}

DEFAULT_WEIGHTS = PROJECT_ROOT / "runs/stage1_detectors/rtdetr_l_seed0/weights/best.pt"
DEFAULT_CFG_DIR = PROJECT_ROOT / "configs/trackers/boxmot"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/video_annotations/tracker_outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True, help="Input video clip.")
    p.add_argument("--tracker", required=True, choices=sorted(TRACKERS),
                   help="Which BoxMOT tracker to run.")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help="RT-DETR-L detector weights (Stage 1 winner).")
    p.add_argument("--tracker-config", type=Path, default=None,
                   help="Tracker YAML. Default: configs/trackers/boxmot/<tracker>.yaml")
    p.add_argument("--reid-weights", default="osnet_x0_25_msmt17.pt",
                   help="ReID weights for botsort/deepocsort (BoxMOT auto-"
                        "downloads known names). See REID CAVEAT in the header.")
    p.add_argument("--detections", type=Path, default=None,
                   help="Detection cache path. Default: <out-dir>/<clip>.detections.npy")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--conf", type=float, default=0.1,
                   help="Detector confidence floor (low: trackers do their own "
                        "thresholding). Default 0.1.")
    p.add_argument("--iou", type=float, default=0.6, help="Detector NMS IoU.")
    p.add_argument("--imgsz", type=int, default=1280, help="Detector image size.")
    p.add_argument("--device", default="0", help="CUDA index or 'cpu'.")
    p.add_argument("--half", action="store_true", help="Half precision for ReID.")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Process only the first N frames (validation). 0 = all. "
                        "When >0 the detection cache is NOT written (would be partial).")
    p.add_argument("--force-detect", action="store_true",
                   help="Re-run the detector even if a cache exists.")
    return p.parse_args()


def compute_detections(weights: Path, video: Path, conf: float, iou: float,
                       imgsz: int, device: str, max_frames: int) -> np.ndarray:
    """Run RT-DETR-L over the clip; return (M,7) [frame,x1,y1,x2,y2,conf,cls]."""
    from ultralytics import YOLO

    logger.info("Detecting with %s (conf=%.2f imgsz=%d)", weights.name, conf, imgsz)
    model = YOLO(str(weights))
    rows: list[tuple] = []
    t0 = time.monotonic()
    for idx, res in enumerate(
        model.predict(source=str(video), conf=conf, iou=iou, imgsz=imgsz,
                      device=device, stream=True, verbose=False), start=1):
        if max_frames and idx > max_frames:
            break
        b = res.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        cf = b.conf.cpu().numpy()
        cl = b.cls.cpu().numpy()
        for (x1, y1, x2, y2), c, k in zip(xyxy, cf, cl):
            rows.append((idx, x1, y1, x2, y2, c, k))
        if idx % 200 == 0:
            logger.info("  detect frame %d  (cum dets=%d)", idx, len(rows))
    arr = np.array(rows, dtype=np.float32).reshape(-1, 7)
    logger.info("  detected %d boxes over %d frames in %.1fs",
                len(arr), idx, time.monotonic() - t0)
    return arr


def group_by_frame(dets: np.ndarray) -> dict[int, np.ndarray]:
    """Map frame -> (N,6) [x1,y1,x2,y2,conf,cls]."""
    by_frame: dict[int, np.ndarray] = {}
    if len(dets):
        fr = dets[:, 0].astype(int)
        for u in np.unique(fr):
            by_frame[int(u)] = np.ascontiguousarray(dets[fr == u][:, 1:7])
    return by_frame


def build_tracker(name: str, config: Path, reid_weights: str, device: str, half: bool):
    mod_name, cls_name = TRACKERS[name]
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(config_path=config, reid_weights=reid_weights, device=device, half=half)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()

    import cv2

    if not args.video.exists():
        logger.error("Video not found: %s", args.video)
        return 2
    if not args.weights.exists():
        logger.error("Detector weights not found: %s", args.weights)
        return 2

    config = args.tracker_config or (DEFAULT_CFG_DIR / f"{args.tracker}.yaml")
    if not config.exists():
        logger.error("Tracker config not found: %s", config)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.detections or (args.output_dir / f"{args.video.stem}.detections.npy")

    # --- detections (compute once, reuse) ---
    if cache.exists() and not args.force_detect:
        dets = np.load(cache)
        logger.info("Loaded cached detections: %s  (%d boxes)", cache.name, len(dets))
    else:
        dets = compute_detections(args.weights, args.video, args.conf, args.iou,
                                  args.imgsz, args.device, args.max_frames)
        if args.max_frames:
            logger.info("max-frames set -> NOT caching partial detections")
        else:
            np.save(cache, dets)
            logger.info("Cached detections -> %s", cache)

    by_frame = group_by_frame(dets)

    # --- tracker replay ---
    logger.info("Building tracker: %s  (config=%s)", args.tracker, config.name)
    tracker = build_tracker(args.tracker, config, args.reid_weights,
                            args.device, args.half)

    out_path = args.output_dir / f"{args.video.stem}.{args.tracker}.txt"
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", args.video)
        return 2

    t0 = time.monotonic()
    frame_idx = 0
    n_rows = 0
    track_ids: set[int] = set()
    empty6 = np.empty((0, 6), dtype=np.float32)

    with out_path.open("w", encoding="utf-8") as f:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break
            tracks = tracker.update(by_frame.get(frame_idx, empty6), img)
            for row in tracks:
                x1, y1, x2, y2, tid, cf, cl = row[:7]
                w, h = x2 - x1, y2 - y1
                f.write(f"{frame_idx},{int(tid)},{x1:.2f},{y1:.2f},"
                        f"{w:.2f},{h:.2f},{cf:.4f},{int(cl)},-1,-1\n")
                n_rows += 1
                track_ids.add(int(tid))
            if frame_idx % 200 == 0:
                logger.info("  track frame %d  tracks=%d  rows=%d",
                            frame_idx, len(track_ids), n_rows)

    cap.release()
    elapsed = time.monotonic() - t0
    logger.info("Wrote %s", out_path)
    logger.info("  frames=%d  unique-tracks=%d  rows=%d  (%.1f fps)",
                frame_idx, len(track_ids), n_rows,
                frame_idx / elapsed if elapsed else 0.0)
    logger.info("Next: link with 02b_link_tracks.py for the post-linker eval arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
