"""BoT-SORT wrapper (BoxMOT). Motion + IoU + appearance (ReID) association."""

from __future__ import annotations

from trackers.base import BaseTrackerWrapper


class BotSortWrapper(BaseTrackerWrapper):
    tracker_type = "botsort"
    needs_reid = True
