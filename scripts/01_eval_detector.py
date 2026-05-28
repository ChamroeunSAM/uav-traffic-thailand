"""Evaluate ONE trained detector with overall / 100m / 120m stratified mAP.

Usage
-----
    python scripts/01_eval_detector.py --weights runs/stage1_detectors/yolo11m_seed0/weights/best.pt --config configs/detectors/yolo11m.yaml

Approach
--------
We do test-set inference once and compute three mAP breakdowns from the same
prediction set, sliced by image filename prefix using altitude_index.json:

  * overall  — all test images
  * 100m      — only images listed under data/splits/altitude_index.json[test][100m]
  * 120m     — only images listed under data/splits/altitude_index.json[test][120m]

Per-class metrics are reported alongside the aggregate. We use
`supervision.metrics.MeanAveragePrecision` rather than ultralytics' YOLO.val()
because (a) it works uniformly for both YOLO and RF-DETR predictions and
(b) it accepts arbitrary subsets of (predictions, ground-truth) pairs, which
makes the altitude split trivial.

Outputs
-------
  * runs/stage1_detectors/<run_name>/eval_results.json — full nested metrics::

        {
          "overall": {"mAP50": ..., "mAP50_95": ..., "precision": NaN, "recall": NaN},
          "100m":     {"mAP50": ..., "mAP50_95": ..., "precision": NaN, "recall": NaN},
          "120m":    {"mAP50": ..., "mAP50_95": ..., "precision": NaN, "recall": NaN},
          "per_class": {
            "<class_name>": {
              "overall": {"AP50": ..., "AP50_95": ...},
              "100m":     {"AP50": ..., "AP50_95": ...},
              "120m":    {"AP50": ..., "AP50_95": ...}
            },
            ...
          }
        }

    Aggregate precision/recall are NaN — supervision's MAP doesn't expose
    them on the aggregate result. A follow-up task will compute them from
    the per-class AP confusion matrix.
  * results/stage1/all_runs.csv — one row per (detector, seed) for aggregation.
  * Console summary table.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.io import load_config, load_json, save_json  # noqa: E402
from utils.logging import configure_logging  # noqa: E402

logger = logging.getLogger("eval_detector")

ALTITUDES = ("100m", "120m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified evaluation for one trained detector.")
    p.add_argument("--weights", required=True, type=Path, help="Path to trained weights (best.pt / .pth).")
    p.add_argument("--config", required=True, type=Path, help="Detector config YAML used for training.")
    p.add_argument(
        "--altitude-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "altitude_index.json",
        help="Path to altitude_index.json produced by 00_prepare_splits.py.",
    )
    p.add_argument(
        "--all-runs-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "stage1" / "all_runs.csv",
        help="CSV to append a one-row summary to for cross-run aggregation.",
    )
    return p.parse_args()


def build_wrapper(detector_class: str, config: dict):
    detector_class = detector_class.lower()
    if detector_class == "ultralytics":
        from detectors.ultralytics_wrapper import UltralyticsWrapper

        return UltralyticsWrapper(config)
    if detector_class == "rfdetr":
        from detectors.rfdetr_wrapper import RFDETRWrapper

        return RFDETRWrapper(config)
    raise ValueError(f"Unknown detector_class: {detector_class!r}")


def load_test_groundtruth(data_yaml_path: Path):
    """Load full test-set GT as supervision DetectionDataset."""
    import supervision as sv

    data_yaml = load_config(data_yaml_path)
    merged_root = Path(data_yaml["path"])
    test_images = merged_root / data_yaml["test"]
    test_labels = merged_root / data_yaml["test"].replace("images", "labels")
    return sv.DetectionDataset.from_yolo(
        images_directory_path=str(test_images),
        annotations_directory_path=str(test_labels),
        data_yaml_path=str(data_yaml_path),
    )


def filter_by_altitude(image_paths: list[Path], altitude_filenames: set[str]) -> list[int]:
    """Return indices of `image_paths` whose basename is in `altitude_filenames`."""
    return [i for i, p in enumerate(image_paths) if p.name in altitude_filenames]


def compute_map(predictions, targets, class_names: list[str]) -> dict:
    """Compute aggregate + per-class mAP via supervision.metrics.

    Returns:
        {
          "aggregate": {"mAP50", "mAP50_95", "precision", "recall"},
          "per_class": {<class_name>: {"AP50", "AP50_95"}}
        }

    Per-class entries that didn't appear in either predictions or targets
    on this split get NaN. Aggregate precision/recall are NaN — supervision's
    MAP result doesn't expose them and we'll add a per-class-AP-derived
    calculation in a follow-up task (post-pilot).
    """
    import numpy as np
    from supervision.metrics import MeanAveragePrecision

    nan = float("nan")
    aggregate_nan = {"mAP50": nan, "mAP50_95": nan, "precision": nan, "recall": nan}
    per_class_nan = {name: {"AP50": nan, "AP50_95": nan} for name in class_names}

    if not predictions or not targets:
        return {"aggregate": aggregate_nan, "per_class": per_class_nan}

    metric = MeanAveragePrecision().update(predictions=predictions, targets=targets)
    result = metric.compute()

    aggregate = {
        "mAP50": float(getattr(result, "map50", nan)),
        "mAP50_95": float(getattr(result, "map50_95", nan)),
        "precision": nan,
        "recall": nan,
    }

    # ap_per_class shape is (n_matched_classes, 10); cols are IoU thresholds
    # 0.50, 0.55, ..., 0.95 (per supervision.metrics.MeanAveragePrecision).
    matched = [int(c) for c in getattr(result, "matched_classes", [])]
    ap_matrix = np.asarray(getattr(result, "ap_per_class", []), dtype=float)

    per_class = dict(per_class_nan)  # start NaN, fill rows we have
    for row_idx, cls_id in enumerate(matched):
        if cls_id < 0 or cls_id >= len(class_names):
            continue
        name = class_names[cls_id]
        row = ap_matrix[row_idx]
        per_class[name] = {
            "AP50": float(row[0]) if row.size > 0 else nan,
            "AP50_95": float(row.mean()) if row.size > 0 else nan,
        }

    return {"aggregate": aggregate, "per_class": per_class}


def main() -> int:
    configure_logging()
    args = parse_args()

    config = load_config(args.config)
    weights = args.weights.resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    altitude_index = load_json(args.altitude_index)
    test_index = altitude_index.get("test", {})
    altitude_files: dict[str, set[str]] = {alt: set(test_index.get(alt, [])) for alt in ALTITUDES}

    print("=" * 78)
    print("Stage 1 stratified evaluation")
    print(f"  config       : {args.config}")
    print(f"  weights      : {weights}")
    print(f"  altitude idx : {args.altitude_index}")
    for alt, fs in altitude_files.items():
        print(f"  {alt:>5} images : {len(fs)}")
    print("=" * 78)

    data_yaml = PROJECT_ROOT / config["data_yaml"]
    data_yaml_dict = load_config(data_yaml)
    class_names: list[str] = list(data_yaml_dict.get("names", []))
    if not class_names:
        raise ValueError(f"No class names found in {data_yaml}")
    logger.info("Class names: %s", class_names)

    gt_dataset = load_test_groundtruth(data_yaml)
    image_paths: list[Path] = []
    targets = []
    for img_path, _img, gt in gt_dataset:
        image_paths.append(Path(img_path))
        targets.append(gt)
    logger.info("Loaded %d test images with ground truth.", len(image_paths))

    wrapper = build_wrapper(config["detector_class"], config)
    wrapper.load_weights(weights)

    pred_lists = wrapper.predict(image_paths)
    predictions = [_dets_to_sv(d) for d in pred_lists]

    splits: dict[str, dict] = {"overall": compute_map(predictions, targets, class_names)}
    for alt in ALTITUDES:
        idxs = filter_by_altitude(image_paths, altitude_files[alt])
        sub_pred = [predictions[i] for i in idxs]
        sub_tgt = [targets[i] for i in idxs]
        splits[alt] = compute_map(sub_pred, sub_tgt, class_names)

    # Build the user-facing JSON shape:
    # {overall, 100m, 120m, per_class: {<class>: {overall, 100m, 120m}}}
    report: dict[str, Any] = {
        "overall": splits["overall"]["aggregate"],
        "100m": splits["100m"]["aggregate"],
        "120m": splits["120m"]["aggregate"],
        "per_class": {
            name: {
                "overall": splits["overall"]["per_class"][name],
                "100m": splits["100m"]["per_class"][name],
                "120m": splits["120m"]["per_class"][name],
            }
            for name in class_names
        },
    }

    eval_record = {
        "config_path": str(args.config),
        "weights_path": str(weights),
        "n_test_images": len(image_paths),
        "n_per_altitude": {alt: len(altitude_files[alt]) for alt in ALTITUDES},
        "class_names": class_names,
        "metrics": report,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    eval_path = weights.parent.parent / "eval_results.json"
    save_json(eval_record, eval_path)

    _append_csv_row(args.all_runs_csv, config, report, weights)

    print()
    print(f"{'split':<10}{'mAP50':>10}{'mAP50_95':>12}")
    print("-" * 32)
    for split in ("overall", "100m", "120m"):
        m = report[split]
        print(f"{split:<10}{m['mAP50']:>10.4f}{m['mAP50_95']:>12.4f}")
    print()
    header = f"{'class':<14}" + "".join(f"{s:>10}" for s in ("overall", "100m", "120m"))
    print(header + "   (AP50)")
    print("-" * len(header))
    for name in class_names:
        row = report["per_class"][name]
        cells = "".join(f"{row[s]['AP50']:>10.4f}" for s in ("overall", "100m", "120m"))
        print(f"{name:<14}" + cells)
    print()
    print(f"  eval_results.json : {eval_path}")
    print(f"  CSV row appended  : {args.all_runs_csv}")
    return 0


def _dets_to_sv(dets):
    """Convert list[Detection] to supervision.Detections."""
    import numpy as np
    import supervision as sv

    if not dets:
        return sv.Detections.empty()
    xyxy = np.array([d.bbox for d in dets], dtype=np.float32)
    conf = np.array([d.confidence for d in dets], dtype=np.float32)
    cls = np.array([d.class_id for d in dets], dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)


def _append_csv_row(csv_path: Path, config: dict, metrics: dict, weights: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # The config only holds the unsubstituted name template (e.g.
    # "yolov9m_seed{seed}") and its default seed. The training script already
    # baked the real seed into the run directory, so derive both from the
    # weights path (.../yolov9m_seed3/weights/best.pt) — correct for both
    # orchestrated and manual eval invocations.
    run_name = weights.parent.parent.name
    seed_match = re.search(r"seed(\d+)", run_name)
    seed = int(seed_match.group(1)) if seed_match else config.get("seed", 0)
    fieldnames = [
        "run_name", "detector_class", "model_name", "seed", "imgsz", "epochs",
        "weights_path",
        "overall_mAP50", "overall_mAP50_95",
        "100m_mAP50", "100m_mAP50_95",
        "120m_mAP50", "120m_mAP50_95",
        "evaluated_at",
    ]
    row = {
        "run_name": run_name,
        "detector_class": config["detector_class"],
        "model_name": config["model_name"],
        "seed": seed,
        "imgsz": config["imgsz"],
        "epochs": config["epochs"],
        "weights_path": str(weights),
        "overall_mAP50": metrics["overall"]["mAP50"],
        "overall_mAP50_95": metrics["overall"]["mAP50_95"],
        "100m_mAP50": metrics["100m"]["mAP50"],
        "100m_mAP50_95": metrics["100m"]["mAP50_95"],
        "120m_mAP50": metrics["120m"]["mAP50"],
        "120m_mAP50_95": metrics["120m"]["mAP50_95"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    sys.exit(main())
