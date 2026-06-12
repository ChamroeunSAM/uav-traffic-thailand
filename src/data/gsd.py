"""Known-GSD lookup from altitude-prefixed names (Nadir-YOLO, decision_log D6).

The metric-scale-prior module needs one scalar per image: the ground-sample
distance in metres/pixel. In this project that scalar is *known*, not
estimated — every dataset image carries an altitude prefix (``100m_...`` /
``120m_...``, from 01_rename_with_altitude_prefix.py) and every video clip
stem carries the ``120_m_...`` form, while the per-altitude GSD values come
from the Stage 4 GCP homography calibration, which cross-validated scale by
altitude (docs/calibration_protocol.md; configs/calibration/*.gcp.yaml).
"""

from __future__ import annotations

import re
from pathlib import Path

# Metres/pixel at 4K from the Stage 4 GCP calibration (per-clip homography
# scales cross-validated by altitude: 100 m ~= 0.030, 120 m ~= 0.040 m/px).
# Dataset frames share the sensor/optics/altitudes of the calibrated clips.
# Per-clip fitted values (data/video_annotations/calibration/<clip>.json)
# override these defaults where exactness matters (e.g. world trajectories);
# the detector prior only needs the per-altitude scalar below.
GSD_M_PER_PX: dict[int, float] = {
    100: 0.030,
    120: 0.040,
}

# Matches both naming forms: image files "100m_<...>.jpg" / clip stems
# "120_m_DJI_0006_1min". Anchored at the start — altitude tokens elsewhere in
# the name (e.g. the raw-source "..._100m_DJI0049_..." mid-string) are NOT
# trusted, only the deliberate provenance prefix is.
_ALTITUDE_PREFIX = re.compile(r"^(\d{2,3})_?m_")


def altitude_m(name: str | Path) -> int | None:
    """Altitude in metres from an altitude-prefixed file/clip name, else None."""
    m = _ALTITUDE_PREFIX.match(Path(name).name)
    return int(m.group(1)) if m else None


def gsd_m_per_px(name: str | Path, default: float | None = None) -> float | None:
    """Known GSD (m/px) for an altitude-prefixed name.

    Returns ``default`` when the name has no altitude prefix or the altitude
    has no calibrated GSD — callers decide whether that means "disable the
    prior" (e.g. VisDrone images) or an error (our own dataset).
    """
    alt = altitude_m(name)
    if alt is None:
        return default
    return GSD_M_PER_PX.get(alt, default)
