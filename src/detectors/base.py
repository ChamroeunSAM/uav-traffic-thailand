"""Abstract base class for detector wrappers.

Each concrete wrapper hides one library's quirks behind a uniform interface so
that the Stage 1 training and evaluation scripts can stay detector-agnostic.
The contract is intentionally narrow: build from a parsed config dict, train
and return summary metrics, run inference on a list of images, and load
trained weights for evaluation. Anything else (data conversion, callbacks,
loss curves) is the wrapper's private concern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Detection:
    """A single predicted bounding box on one image.

    Coordinates are absolute pixels in (x1, y1, x2, y2) order, matching the
    convention used by both Ultralytics and supervision's `sv.Detections`.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    image_path: Path


class DetectorWrapper(ABC):
    """Abstract base class — each detector library has one concrete subclass.

    Subclasses MUST:
      * Accept a parsed config dict in __init__ (from configs/detectors/*.yaml).
      * Implement train(output_dir) returning a dict with at minimum the keys
        listed in the train() docstring below.
      * Implement predict(image_paths) returning a list-of-lists of Detection.
      * Implement load_weights(path) so the eval script can re-instantiate
        from a checkpoint without retraining.

    The wrapper is responsible for: device placement, seed plumbing, data
    format adapters, and turning whatever the underlying library returns into
    the structured outputs declared here.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def train(self, output_dir: Path) -> dict[str, Any]:
        """Train the model end-to-end.

        Args:
            output_dir: Directory under which the wrapper should write its
                native run artifacts (checkpoints, logs). For Ultralytics this
                becomes `project=<output_dir.parent>, name=<output_dir.name>`.

        Returns:
            A dict containing AT LEAST the following keys::

                {
                    "best_weights_path": Path,        # absolute path to best.pt or .pth
                    "training_time_seconds": float,
                    "final_metrics": {
                        "val_mAP50": float,
                        "val_mAP50_95": float,
                        "val_precision": float,
                        "val_recall": float,
                    },
                }

            Wrappers MAY include extra keys (e.g. epochs_completed) but the
            four shown are the minimum the training script will record.
        """

    @abstractmethod
    def predict(self, image_paths: list[Path]) -> list[list[Detection]]:
        """Run inference on a batch of images.

        Returns one Detection list per input image, in the same order as
        `image_paths`. Empty lists are valid (image with no detections).
        """

    @abstractmethod
    def load_weights(self, weights_path: Path) -> None:
        """Load trained weights into the wrapper, in place.

        After this returns, predict() must work using the loaded weights.
        """
