"""Trajectory smoothing for Stage 4 (Savitzky-Golay primary; decision_log D4).

Raw per-frame box-centre positions are noisy, and differentiating them directly
for velocity / acceleration amplifies that noise. A Savitzky-Golay filter fits a
low-order polynomial over a sliding window, denoising position while preserving
genuine manoeuvres. It is the D4 primary smoother; Kalman / spline smoothers can
be added behind the same call signature later.

Smoothing is reported pre/post (the caller keeps the unsmoothed series too), so
its effect is a *studied variable*, not a hidden step.
"""

from __future__ import annotations

import numpy as np


def adapt_window(window: int, n: int, polyorder: int) -> int | None:
    """Largest valid odd window <= min(window, n) and > polyorder; None if the
    series is too short to smooth at this polyorder."""
    w = min(int(window), int(n))
    if w % 2 == 0:
        w -= 1
    if w <= polyorder:
        return None
    return w


def savgol_smooth_xy(xy, window: int = 15, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay-smooth an (N, 2) position series along axis 0.

    The window is adapted down to the series length (kept odd and > polyorder).
    If the series is too short to smooth, the input is returned unchanged (a
    copy), so the call is always safe on tiny tracks.
    """
    from scipy.signal import savgol_filter

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"expected (N,2) positions, got shape {xy.shape}")
    w = adapt_window(window, xy.shape[0], polyorder)
    if w is None:
        return xy.copy()
    out = np.empty_like(xy)
    out[:, 0] = savgol_filter(xy[:, 0], w, polyorder)
    out[:, 1] = savgol_filter(xy[:, 1], w, polyorder)
    return out
