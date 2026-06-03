"""Deep OC-SORT wrapper (BoxMOT). OC-SORT motion + appearance (ReID)."""

from __future__ import annotations

from trackers.base import BaseTrackerWrapper


class DeepOcSortWrapper(BaseTrackerWrapper):
    tracker_type = "deepocsort"
    needs_reid = True
