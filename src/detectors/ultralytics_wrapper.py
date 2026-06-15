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
        from ultralytics import YOLO, settings as ul_settings

        # Ultralytics' MLflow callback crashes on Windows when its tracking URI
        # is a bare path like `C:\...\runs\mlflow` (mlflow parses `C` as an
        # unknown URI scheme). We don't use MLflow in this project, so disable
        # the callback via the settings API. .update() persists to the user
        # settings file too, so the fix is idempotent across machines.
        if ul_settings.get("mlflow", False):
            ul_settings.update({"mlflow": False})

        self._YOLO = YOLO
        model_name = config["model_name"]
        self.model = YOLO(model_name)

        # Custom-architecture path (Nadir-YOLO, decision_log D6): when
        # `model_name` is an architecture YAML, YOLO() builds it with random
        # init; `pretrained` transfers matching layers from a checkpoint
        # (Ultralytics skips shape-mismatched layers, so edited modules keep
        # their random init while the untouched backbone/neck load weights).
        pretrained = config.get("pretrained")
        if pretrained:
            if not str(model_name).endswith((".yaml", ".yml")):
                raise ValueError(
                    "`pretrained` is only valid when model_name is an "
                    f"architecture YAML, got model_name={model_name!r}"
                )
            self.model.load(pretrained)

    def train(self, output_dir: Path) -> dict[str, Any]:
        cfg = self.config

        # Optional VRAM guard (Nadir-YOLO runs): Windows WDDM silently spills
        # past-physical VRAM into system RAM (~6x slowdown observed on the
        # GDP P2 variant at batch=8, 45GB "GPU mem"). Capping the allocator
        # converts that silent slow mode into an immediate OOM so the batch
        # size gets fixed instead. No effect when the key is absent.
        mem_frac = cfg.get("cuda_mem_fraction")
        if mem_frac:
            import torch

            dev = cfg.get("device", 0)
            if torch.cuda.is_available() and str(dev).isdigit():
                torch.cuda.set_per_process_memory_fraction(float(mem_frac), int(dev))
                logger.info("CUDA memory capped at %.0f%% of device %s",
                            float(mem_frac) * 100, dev)

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
            "fraction": cfg.get("fraction", 1.0),
        }

        # MSP route (ii-A): swap in the GSD-input-channel trainer (4ch first
        # conv + per-image GSD plane). Runtime subclass only; stock runs are
        # untouched when the key is absent. See src/detectors/msp_patches.py.
        if cfg.get("msp_mode") == "ii_a":
            from .msp_patches import MSPTrainer

            train_kwargs["trainer"] = MSPTrainer
            logger.info("MSP route ii-A enabled: GSD input channel (4ch first conv)")

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

    # Chunk size for inference. At imgsz=1280, Ultralytics accumulates GPU
    # tensors across a predict() call; passing all test images at once
    # (e.g. 451) leaks VRAM until it OOMs. Processing in small chunks and
    # forcing empty_cache() between them keeps the working set bounded.
    PREDICT_CHUNK = 16

    def predict(self, image_paths: list[Path]) -> list[list[Detection]]:
        if not image_paths:
            return []

        import torch

        out: list[list[Detection]] = []
        for start in range(0, len(image_paths), self.PREDICT_CHUNK):
            chunk = image_paths[start:start + self.PREDICT_CHUNK]
            results = self.model.predict(
                source=[str(p) for p in chunk],
                verbose=False,
                device=self.config.get("device", 0),
                imgsz=self.config.get("imgsz", 1280),
                conf=self.config.get("conf_threshold", 0.001),
                iou=self.config.get("iou_threshold", 0.6),
                stream=False,
            )
            for img_path, res in zip(chunk, results):
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
            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
