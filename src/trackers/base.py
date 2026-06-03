"""Uniform wrapper around BoxMOT trackers for the Stage 3 benchmark.

All four Stage 3 trackers (ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT) are
driven through one interface so the *only* variable in the benchmark is the
association algorithm. Detections are produced once by the Stage 1 winner
(RT-DETR-L) and replayed into each tracker via `update()`.

BoxMOT's low-level tracker classes share the contract:

    update(dets: (N,6) [x1,y1,x2,y2,conf,cls], img: HxWx3 BGR) -> (M,8)
        returned columns: [x1, y1, x2, y2, track_id, conf, cls, det_idx]

Subclasses set `tracker_type` (BoxMOT key) and `needs_reid`. Construction
goes through BoxMOT's `create_tracker`, which fills any unspecified
constructor args from each tracker class's own defaults -- so a config file
only needs to list the params we deliberately override.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _as_torch_device(device: str | int):
    import torch
    d = str(device)
    if d == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{d}" if d.isdigit() else d)


class BaseTrackerWrapper:
    """Common interface over a BoxMOT tracker instance."""

    tracker_type: str = ""      # BoxMOT key, e.g. "bytetrack" (set by subclass)
    needs_reid: bool = False    # appearance-based trackers need ReID weights

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        reid_weights: str | Path = "osnet_x0_25_msmt17.pt",
        device: str | int = "0",
        half: bool = False,
    ) -> None:
        if not self.tracker_type:
            raise ValueError(f"{type(self).__name__} must set `tracker_type`")

        # Late import: keeps argparse/--help fast and avoids importing torch
        # on callers that only need the class symbol.
        from boxmot.trackers.tracker_zoo import create_tracker

        params: dict = {}
        if config_path is not None:
            loaded = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            params = loaded or {}

        kwargs = dict(tracker_type=self.tracker_type, evolve_param_dict=params)
        if self.needs_reid:
            kwargs.update(
                reid_weights=Path(reid_weights),
                device=_as_torch_device(device),
                half=half,
            )

        self.tracker = create_tracker(**kwargs)
        logger.info(
            "Initialized %-10s (reid=%s)  overrides=%s",
            self.tracker_type, self.needs_reid, params or "{} (all defaults)",
        )

    def update(self, dets: np.ndarray | None, img: np.ndarray) -> np.ndarray:
        """Advance the tracker by one frame; returns (M,8) track rows.

        `dets` is (N,6) [x1,y1,x2,y2,conf,cls]. Empty/None is allowed (the
        tracker still propagates existing tracks for this frame).
        """
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        dets = np.ascontiguousarray(dets, dtype=np.float32)
        return self.tracker.update(dets, img)
