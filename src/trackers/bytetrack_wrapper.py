"""ByteTrack wrapper (BoxMOT). Motion + IoU association, no appearance/ReID."""

from __future__ import annotations

from trackers.base import BaseTrackerWrapper


class ByteTrackWrapper(BaseTrackerWrapper):
    tracker_type = "bytetrack"
    needs_reid = False
