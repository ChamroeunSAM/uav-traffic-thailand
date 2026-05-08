"""RF-DETR wrapper.

The rfdetr package (Roboflow) ships variant classes — RFDETRNano /
RFDETRMedium / RFDETRBase / RFDETRLarge — each of which exposes a uniform
``train(**kwargs)`` / ``predict(images)`` API. The kwargs are forwarded to
:class:`rfdetr.config.TrainConfig`, which is materially different from
Ultralytics' shape, so this wrapper does the translation.

Key adapter responsibilities
----------------------------
1. Dataset-format adapter
   With ``dataset_file="yolo"`` rfdetr accepts a Roboflow-style YOLO layout
   under a single ``dataset_dir``::

       dataset_dir/
         data.yaml
         train/{images,labels}/
         valid/{images,labels}/    <-- note: 'valid', not 'val'
         test/{images,labels}/

   Our merged dataset uses ``val/`` (per scripts/00_prepare_splits.py).
   We hard-copy ``val/`` to ``valid/`` if the destination does not exist.
   Hard copy (rather than symlink) avoids Windows symlink-permission pain
   and works identically on every platform. The copy is idempotent: if
   ``valid/`` already exists it is left untouched.

2. Result conversion
   ``RFDETRMedium.predict`` returns ``supervision.Detections`` objects. We
   unpack those into the project-wide :class:`Detection` dataclass.

3. Pretrained weight handling
   Instantiating any RFDETR* class auto-downloads the pretrained checkpoint
   (~386 MB for medium) on first use. Trained checkpoints land in
   ``output_dir/`` (the wrapper picks ``checkpoint_best_total.pth`` if
   present, otherwise the last checkpoint).

Open questions (pending pilot run, see scaffolding notes):
- Whether resolution=1280 satisfies the ``patch_size * num_windows``
  divisibility constraint for RFDETRMedium specifically.
- The exact filename produced for "best" weights varies across rfdetr
  releases (``checkpoint_best_total.pth``, ``checkpoint_best_ema.pth``).
  The wrapper resolves whichever exists.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from .base import Detection, DetectorWrapper

logger = logging.getLogger(__name__)


_RFDETR_VARIANTS = {
    "rfdetr_n": "RFDETRNano",
    "rfdetr_s": "RFDETRSmall",
    "rfdetr_m": "RFDETRMedium",
    "rfdetr_b": "RFDETRBase",
    "rfdetr_l": "RFDETRLarge",
}


class RFDETRWrapper(DetectorWrapper):
    """Wraps an rfdetr.RFDETR* variant for train / predict / load_weights."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model_name = config["model_name"]
        self._variant_cls = self._resolve_variant_cls(self.model_name)
        self._init_kwargs: dict[str, Any] = {}
        if "pretrain_weights" in config:
            self._init_kwargs["pretrain_weights"] = config["pretrain_weights"]
        self.model = self._variant_cls(**self._init_kwargs)

    def train(self, output_dir: Path) -> dict[str, Any]:
        cfg = self.config
        dataset_dir = self._prepare_dataset_dir(Path(cfg["data_yaml"]))

        train_kwargs = {
            "dataset_dir": str(dataset_dir),
            "dataset_file": "yolo",
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch"],
            "grad_accum_steps": cfg.get("grad_accum_steps", 4),
            "resolution": cfg["imgsz"],
            "output_dir": str(output_dir),
            "seed": cfg["seed"],
            "device": cfg.get("device", "cuda:0"),
            "num_workers": cfg.get("workers", 4),
            "early_stopping": cfg.get("patience", 0) > 0,
            "early_stopping_patience": cfg.get("patience", 10),
            "lr_scheduler": cfg.get("lr_scheduler", "cosine"),
            "tensorboard": cfg.get("tensorboard", True),
        }

        class_names = cfg.get("class_names")
        if class_names:
            train_kwargs["class_names"] = list(class_names)

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("RF-DETR training kwargs: %s", train_kwargs)
        t0 = time.monotonic()
        results = self.model.train(**train_kwargs)
        elapsed = time.monotonic() - t0

        best_weights = self._resolve_best_weights(output_dir)
        metrics = self._extract_metrics(results, output_dir)

        return {
            "best_weights_path": best_weights,
            "training_time_seconds": elapsed,
            "final_metrics": metrics,
            "run_dir": output_dir,
        }

    def predict(self, image_paths: list[Path]) -> list[list[Detection]]:
        if not image_paths:
            return []

        threshold = self.config.get("conf_threshold", 0.001)
        out: list[list[Detection]] = []
        for img_path in image_paths:
            sv_dets = self.model.predict(str(img_path), threshold=threshold)
            out.append(self._sv_to_detections(sv_dets, img_path))
        return out

    def load_weights(self, weights_path: Path) -> None:
        kwargs = dict(self._init_kwargs)
        kwargs["pretrain_weights"] = str(weights_path)
        self.model = self._variant_cls(**kwargs)

    @staticmethod
    def _resolve_variant_cls(model_name: str):
        import rfdetr

        key = model_name.lower()
        if key in _RFDETR_VARIANTS:
            cls_name = _RFDETR_VARIANTS[key]
        else:
            cls_name = model_name
        try:
            return getattr(rfdetr, cls_name)
        except AttributeError as e:
            raise ValueError(
                f"Unknown RF-DETR variant '{model_name}'. "
                f"Expected one of {list(_RFDETR_VARIANTS)} or a class name "
                f"present in rfdetr (e.g. RFDETRMedium)."
            ) from e

    @staticmethod
    def _prepare_dataset_dir(data_yaml_path: Path) -> Path:
        """Resolve the merged dataset root from dataset.yaml and prepare the
        layout rfdetr expects.

        Our project's dataset.yaml (e.g. ``data/splits/dataset.yaml``) is
        an index file that points at the actual merged dataset via its
        ``path:`` field — typically ``data/annotations/merged/``. We read
        that to find the real root.

        rfdetr's YOLO loader requires two things in the dataset root that
        our merged layout does not produce by default:

          * a ``valid/`` directory mirroring ``val/`` (rfdetr hardcodes
            that folder name),
          * a ``data.yaml`` describing the dataset (for class names).

        Both are materialised as idempotent hard copies the first time we
        need them. Hard copy (rather than symlink) avoids Windows
        symlink-permission pain.
        """
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            ds_cfg = yaml.safe_load(f)
        required_keys = {"path", "nc", "names"}
        missing = required_keys - set(ds_cfg.keys() if ds_cfg else ())
        if missing:
            raise ValueError(
                f"dataset.yaml at {data_yaml_path} is missing required keys: "
                f"{sorted(missing)}. RF-DETR requires path, nc, and names fields."
            )
        merged_root = Path(ds_cfg["path"]).resolve()

        val = merged_root / "val"
        if not val.exists():
            raise FileNotFoundError(f"Expected validation split at {val}")

        valid = merged_root / "valid"
        if valid.exists():
            logger.info("RF-DETR 'valid/' already present at %s — skipping copy.", valid)
        else:
            logger.info("Copying %s -> %s for RF-DETR (one-time hard copy)...", val, valid)
            shutil.copytree(val, valid)
            logger.info("Copy complete.")

        rfdetr_yaml = merged_root / "data.yaml"
        if not rfdetr_yaml.exists():
            logger.info("Copying %s -> %s for RF-DETR's YOLO loader.", data_yaml_path, rfdetr_yaml)
            shutil.copy2(data_yaml_path, rfdetr_yaml)

        return merged_root

    @staticmethod
    def _resolve_best_weights(output_dir: Path) -> Path:
        candidates = [
            "checkpoint_best_total.pth",
            "checkpoint_best_ema.pth",
            "checkpoint_best.pth",
            "checkpoint.pth",
            "best.pth",
        ]
        for name in candidates:
            p = output_dir / name
            if p.exists():
                return p
        # fall back to the most recent .pth in output_dir
        pths = sorted(output_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pths:
            return pths[0]
        raise FileNotFoundError(f"No RF-DETR checkpoint found under {output_dir}")

    @staticmethod
    def _extract_metrics(results: Any, output_dir: Path) -> dict[str, float]:
        """Best-effort metric extraction.

        rfdetr's train() return value shape is not stable across releases; the
        canonical source of final-epoch metrics is the `results.json` /
        `metrics.json` file written by PTL callbacks under output_dir. We
        defer detailed parsing to the eval script and return NaNs here when
        we cannot read them inline.
        """
        nan = float("nan")
        defaults = {
            "val_mAP50": nan,
            "val_mAP50_95": nan,
            "val_precision": nan,
            "val_recall": nan,
        }
        if isinstance(results, dict):
            for k_out, k_in_options in {
                "val_mAP50": ("val_mAP50", "metrics/mAP50", "mAP50"),
                "val_mAP50_95": ("val_mAP50_95", "metrics/mAP50_95", "mAP50_95"),
                "val_precision": ("val_precision", "precision"),
                "val_recall": ("val_recall", "recall"),
            }.items():
                for k in k_in_options:
                    if k in results:
                        try:
                            defaults[k_out] = float(results[k])
                            break
                        except (TypeError, ValueError):
                            pass
        return defaults

    @staticmethod
    def _sv_to_detections(sv_dets: Any, img_path: Path) -> list[Detection]:
        """Unpack a `supervision.Detections` into our Detection dataclass."""
        if sv_dets is None or len(sv_dets) == 0:
            return []
        out: list[Detection] = []
        xyxy = sv_dets.xyxy
        conf = sv_dets.confidence if sv_dets.confidence is not None else [1.0] * len(sv_dets)
        cls = sv_dets.class_id if sv_dets.class_id is not None else [0] * len(sv_dets)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            out.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(c),
                    class_id=int(k),
                    image_path=img_path,
                )
            )
        return out
