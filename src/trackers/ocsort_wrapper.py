"""OC-SORT wrapper (BoxMOT). Observation-centric motion association, no ReID."""

from __future__ import annotations

from trackers.base import BaseTrackerWrapper


class OcSortWrapper(BaseTrackerWrapper):
    tracker_type = "ocsort"
    needs_reid = False
