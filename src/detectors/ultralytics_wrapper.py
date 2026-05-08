"""Ultralytics-family wrapper (covers YOLO11, YOLO12, YOLO26).

The Ultralytics `YOLO` class auto-detects the architecture from the model
name (yolo11m.pt, yolo12m.pt, yolo26m.pt, yolo11n.pt, etc.), so a single
wrapper handles all three Stage 1 Ultralytics detectors. RF-DETR has a
separate wrapper because its training API is incompatible.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .base import Detection, DetectorWrapper

logger = logging.getLogger(__name__)


class UltralyticsWrapper(DetectorWrapper):
    """Wraps `ultralytics.YOLO` for train / predict / load_weights."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        from ultralytics import YOLO

        self._YOLO = YOLO
        model_name = config["model_name"]
        self.model = YOLO(model_name)

    def train(self, output_dir: Path) -> dict[str, Any]:
        cfg = self.config

        train_kwargs = {
            "data": cfg["data_yaml"],
            "epochs": cfg["epochs"],
            "imgsz": cfg["imgsz"],
            "batch": cfg["batch"],
            "patience": cfg.get("patience", 0),
            "seed": cfg["seed"],
            "workers": cfg.get("workers", 4),
            "device": cfg.get("device", 0),
            "project": str(output_dir.parent),
            "name": output_dir.name,
            "exist_ok": True,
            "cache": cfg.get("cache", False),
            "verbose": True,
            "cos_lr": cfg.get("cos_lr", True),
        }

        logger.info("Ultralytics training kwargs: %s", train_kwargs)
        t0 = time.monotonic()
        results = self.model.train(**train_kwargs)
        elapsed = time.monotonic() - t0

        run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else output_dir
        best_weights = run_dir / "weights" / "best.pt"

        metrics = self._extract_metrics(results)

        return {
            "best_weights_path": best_weights,
            "training_time_seconds": elapsed,
            "final_metrics": metrics,
            "run_dir": run_dir,
        }

    def predict(self, image_paths: list[Path]) -> list[list[Detection]]:
        if not image_paths:
            return []

        sources = [str(p) for p in image_paths]
        results = self.model.predict(
            source=sources,
            verbose=False,
            device=self.config.get("device", 0),
            imgsz=self.config.get("imgsz", 1280),
            conf=self.config.get("conf_threshold", 0.001),
            iou=self.config.get("iou_threshold", 0.6),
            stream=False,
        )

        out: list[list[Detection]] = []
        for img_path, res in zip(image_paths, results):
            dets: list[Detection] = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                    dets.append(
                        Detection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=float(c),
                            class_id=int(k),
                            image_path=img_path,
                        )
                    )
            out.append(dets)
        return out

    def load_weights(self, weights_path: Path) -> None:
        self.model = self._YOLO(str(weights_path))

    @staticmethod
    def _extract_metrics(results: Any) -> dict[str, float]:
        """Pull the four headline val metrics out of an Ultralytics results obj.

        Ultralytics' `results.results_dict` keys vary slightly by version;
        we try several common spellings and fall back to NaN if absent.
        """
        rd = getattr(results, "results_dict", {}) or {}

        def first(*keys: str, default: float = float("nan")) -> float:
            for k in keys:
                if k in rd:
                    try:
                        return float(rd[k])
                    except (TypeError, ValueError):
                        continue
            return default

        return {
            "val_mAP50": first("metrics/mAP50(B)", "metrics/mAP50", "mAP50"),
            "val_mAP50_95": first("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"),
            "val_precision": first("metrics/precision(B)", "metrics/precision", "precision"),
            "val_recall": first("metrics/recall(B)", "metrics/recall", "recall"),
        }
