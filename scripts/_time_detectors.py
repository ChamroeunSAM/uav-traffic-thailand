"""Measure single-image inference latency / FPS for ONE Stage-1 detector.

Times one detector per process so the GPU is fully released on exit (Ultralytics
retains model references that del/empty_cache do not free, so looping all six in
one process piles VRAM up and thrashes). A shell loop calls this once per
detector; each call APPENDS a row to results/stage1/inference_speed.csv.

Run (per detector):
    python scripts/_time_detectors.py --detector yolo11m
Driver: see scripts/_time_detectors.sh
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs/stage1_detectors"
TEST_IMAGES = PROJECT_ROOT / "data/annotations/merged/test/images"
OUT = PROJECT_ROOT / "results/stage1/inference_speed.csv"

DETECTORS = ["yolov9m", "yolov10m", "yolo11m", "yolo12m", "yolo26m", "rtdetr_l"]
IMGSZ = 1280
DEVICE = 0
N_WARMUP = 15
N_TIME = 80
FIELDS = ["detector", "infer_ms_mean", "infer_ms_std", "fps", "imgsz",
          "n_images", "device", "precision"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=DETECTORS)
    args = ap.parse_args()
    det = args.detector

    import cv2
    from ultralytics import YOLO

    w = RUNS / f"{det}_seed0/weights/best.pt"
    if not w.exists():
        print(f"SKIP {det}: no weights at {w}")
        return 1

    # one in-memory image, reused every call -> flat memory, no dataset build-up
    imgs = sorted(TEST_IMAGES.glob("*.jpg")) + sorted(TEST_IMAGES.glob("*.png"))
    img = cv2.imread(str(imgs[0]))

    model = YOLO(str(w))
    for _ in range(N_WARMUP):
        model.predict(img, imgsz=IMGSZ, device=DEVICE, verbose=False)
    infer_ms = []
    t0 = time.monotonic()
    for _ in range(N_TIME):
        res = model.predict(img, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
        infer_ms.append(res.speed["inference"])
    wall = time.monotonic() - t0
    infer = np.array(infer_ms, dtype=float)
    row = {
        "detector": det,
        "infer_ms_mean": round(float(infer.mean()), 3),
        "infer_ms_std": round(float(infer.std()), 3),
        "fps": round(1000.0 / float(infer.mean()), 1),
        "imgsz": IMGSZ, "n_images": N_TIME, "device": "cuda:0", "precision": "fp32",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            wr.writeheader()
        wr.writerow(row)
    print(f"{det:10s}  {row['infer_ms_mean']:6.2f} ms  {row['fps']:6.1f} FPS  "
          f"(wall {wall:.1f}s)  -> appended", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
