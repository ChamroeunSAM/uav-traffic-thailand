"""Image-to-world calibration via a ground-plane homography (Stage 4).

Stage 4 converts pixel trajectories to metric world coordinates so that
kinematics (speed m/s, acceleration m/s^2, heading) are physically
meaningful. For near-nadir UAV footage over (locally) flat road, a single
planar homography ``H`` maps homogeneous image pixels (u, v, 1) to a local
metric ground frame (X, Y) in metres:

    [X', Y', w]^T = H @ [u, v, 1]^T,    (X, Y) = (X'/w, Y'/w)

``H`` is fit per clip from >=4 ground control points (GCPs) whose pixel
locations are read off a reference frame and whose world coordinates are
derived from known road-marking geometry (lane width, dashed centre-line
segment / gap lengths). See ``docs/calibration_protocol.md`` and
``configs/calibration/<clip>.gcp.yaml``.

Fit quality is summarised by the **reprojection RMSE in metres**: the mean
distance between each GCP's surveyed world position and the homography's
projection of its pixel position. This is the calibration error budget. A
gross point-entry or local-fit mistake produces a large RMSE -- but note
RMSE *cannot* detect a globally-wrong reference distance (that scales every
projected coordinate uniformly), which is why the protocol also records a
held-out validation segment as an independent scale check (see D4).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Calibration:
    """A fitted pixel->world homography and its provenance."""

    clip: str
    H: list                       # 3x3 row-major homography, pixel -> world (metres)
    rmse_m: float                 # reprojection RMSE over the GCPs, in metres
    n_gcp: int
    units: str = "m"
    note: str = ""
    validation: dict = field(default_factory=dict)  # optional held-out scale check

    def matrix(self) -> np.ndarray:
        return np.asarray(self.H, dtype=float).reshape(3, 3)


def fit_homography(pixel_pts, world_pts) -> tuple[np.ndarray, float]:
    """Fit a pixel->world homography. Returns (H 3x3 float64, rmse_m float).

    ``pixel_pts``: (N, 2) image coords (u, v). ``world_pts``: (N, 2) metres.
    N >= 4. Exact solve for N == 4 (``getPerspectiveTransform``); for N > 4
    an all-points least-squares fit (``findHomography`` method=0 -- no
    RANSAC, since hand-picked GCPs are all trusted inliers).
    """
    import cv2

    src = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(world_pts, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] != dst.shape[0]:
        raise ValueError(f"pixel/world count mismatch: {src.shape[0]} vs {dst.shape[0]}")
    if src.shape[0] < 4:
        raise ValueError(f"need >=4 GCPs, got {src.shape[0]}")

    if src.shape[0] == 4:
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    else:
        H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("homography fit failed (degenerate GCP configuration?)")
    H = np.asarray(H, dtype=np.float64)
    return H, reprojection_rmse(H, src, dst)


def pixel_to_world(pts, H) -> np.ndarray:
    """Project (N, 2) pixel coords to (N, 2) world metres through ``H``."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))])      # (N, 3)
    proj = (np.asarray(H, dtype=np.float64) @ hom.T).T      # (N, 3)
    return proj[:, :2] / proj[:, 2:3]


def reprojection_rmse(H, pixel_pts, world_pts) -> float:
    """RMSE (metres) between surveyed world GCPs and reprojected pixels."""
    proj = pixel_to_world(pixel_pts, H)
    world = np.asarray(world_pts, dtype=np.float64).reshape(-1, 2)
    d = np.linalg.norm(proj - world, axis=1)
    return float(np.sqrt(np.mean(d ** 2)))


def save_calibration(calib: Calibration, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calib), indent=2), encoding="utf-8")


def load_calibration(path: str | Path) -> Calibration:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return Calibration(**d)
