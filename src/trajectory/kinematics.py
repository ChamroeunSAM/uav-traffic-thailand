"""Kinematic features from smoothed world trajectories (Stage 4; decision_log D4).

Given a position series in metres sampled at uniform ``dt`` seconds, derive
velocity, speed, acceleration, heading, and jerk by central finite differences
(``np.gradient``: central in the interior, one-sided at the ends). Heading is
the trajectory tangent direction -- per decision_log D1, heading on top-down
imagery is recovered from motion, not from the (axis-aligned) box.

All inputs/outputs are SI: metres, seconds, m/s, m/s^2, m/s^3, degrees.
"""

from __future__ import annotations

import numpy as np


def compute_kinematics(xy, dt: float) -> dict:
    """Kinematics of an (N, 2) metric position series at uniform step ``dt`` s.

    Returns a dict of arrays (all length N):
      velocity (N,2), speed (N,), accel (N,2), accel_mag (N,),
      jerk (N,2), jerk_mag (N,), heading_deg (N,).
    Series shorter than 2 samples yield zeros (no derivative is defined).
    """
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    if n < 2:
        z2 = np.zeros((n, 2)); z1 = np.zeros(n)
        return dict(velocity=z2, speed=z1.copy(), accel=z2.copy(),
                    accel_mag=z1.copy(), jerk=z2.copy(), jerk_mag=z1.copy(),
                    heading_deg=z1.copy())

    vel = np.gradient(xy, dt, axis=0)            # (N,2) m/s
    speed = np.linalg.norm(vel, axis=1)          # (N,)  m/s
    acc = np.gradient(vel, dt, axis=0)           # (N,2) m/s^2
    acc_mag = np.linalg.norm(acc, axis=1)
    jerk = np.gradient(acc, dt, axis=0)          # (N,2) m/s^3
    jerk_mag = np.linalg.norm(jerk, axis=1)
    heading_deg = np.degrees(np.arctan2(vel[:, 1], vel[:, 0]))
    return dict(velocity=vel, speed=speed, accel=acc, accel_mag=acc_mag,
                jerk=jerk, jerk_mag=jerk_mag, heading_deg=heading_deg)
