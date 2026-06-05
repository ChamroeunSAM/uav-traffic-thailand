"""Native-validator precision/recall pass (P/R source for Table T2).

For each Stage-1 detector this runs Ultralytics' *native* validator on the
stratified test split (across all available seeds) to obtain precision/recall --
which the supervision-based canonical evaluator does not report. T2 appends these
P/R columns to its canonical (supervision) mAP. mAP@.5 / mAP@.5:.95, parameter
count and GFLOPs (at the 1280 eval resolution) are also recorded for reference;
inference FPS is pulled from the existing timing harness output.

Note: native-val mAP differs from the canonical supervision mAP in
results/stage1/aggregated.csv by -0.013..+0.013 depending on detector
(implementation difference, decision_log D5) -- T2 uses supervision mAP and only
borrows P/R from here.

Output: results/stage1/efficiency.csv  (one row per detector, seed-averaged).

Run:  python scripts/eval_efficiency.py
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs/stage1_detectors"
DATA_YAML = PROJECT_ROOT / "data/splits/dataset.yaml"
SPEED_CSV = PROJECT_ROOT / "results/stage1/inference_speed.csv"
OUT_CSV = PROJECT_ROOT / "results/stage1/efficiency.csv"
IMGSZ = 1280
VAL_BATCH = 4  # val is gradient-free; 4 is safe for YOLO12m's attention at 1280

# detector base -> loader kind (rtdetr routes a different Ultralytics validator)
DETECTORS = [
    ("yolov9m", "yolo"), ("yolov10m", "yolo"), ("yolo11m", "yolo"),
    ("yolo12m", "yolo"), ("yolo26m", "yolo"), ("rtdetr_l", "rtdetr"),
]


def _fps_map() -> dict:
    out = {}
    if SPEED_CSV.exists():
        for r in csv.DictReader(SPEED_CSV.open(encoding="utf-8")):
            out[r["detector"]] = float(r["fps"])
    return out


def main() -> int:
    from ultralytics import RTDETR, YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    if not DATA_YAML.exists():
        print(f"!! {DATA_YAML} missing -- regenerate splits first")
        return 1

    fps = _fps_map()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["detector", "n_seeds",
              "precision_mean", "precision_std", "recall_mean", "recall_std",
              "mAP50_mean", "mAP50_std", "mAP50_95_mean", "mAP50_95_std",
              "params_M", "gflops", "fps"]
    rows = []

    for base, kind in DETECTORS:
        seed_wts = sorted(RUNS_DIR.glob(f"{base}_seed*/weights/best.pt"))
        if not seed_wts:
            print(f"  -- {base}: no weights, skipping")
            continue
        ctor = RTDETR if kind == "rtdetr" else YOLO
        P, R, m50, m95 = [], [], [], []
        for wts in seed_wts:
            model = ctor(str(wts))
            res = model.val(data=str(DATA_YAML), split="test", imgsz=IMGSZ,
                            batch=VAL_BATCH, verbose=False, plots=False,
                            save_json=False)
            P.append(float(res.box.mp)); R.append(float(res.box.mr))
            m50.append(float(res.box.map50)); m95.append(float(res.box.map))
            print(f"  {wts.parent.parent.name:18s} P={P[-1]:.4f} R={R[-1]:.4f} "
                  f"mAP50={m50[-1]:.4f} mAP50-95={m95[-1]:.4f}")

        # params / GFLOPs are architecture constants -> take from the first seed
        net = ctor(str(seed_wts[0])).model
        params_m = get_num_params(net) / 1e6
        gflops = get_flops(net, imgsz=IMGSZ)

        rows.append({
            "detector": base, "n_seeds": len(seed_wts),
            "precision_mean": np.mean(P), "precision_std": np.std(P),
            "recall_mean": np.mean(R), "recall_std": np.std(R),
            "mAP50_mean": np.mean(m50), "mAP50_std": np.std(m50),
            "mAP50_95_mean": np.mean(m95), "mAP50_95_std": np.std(m95),
            "params_M": params_m, "gflops": gflops,
            "fps": fps.get(base, float("nan")),
        })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nwrote {OUT_CSV.relative_to(PROJECT_ROOT)}  ({len(rows)} detectors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
