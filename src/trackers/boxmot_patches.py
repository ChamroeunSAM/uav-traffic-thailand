"""Runtime numerical-stability patches for BoxMOT (Stage 3).

BoxMOT's Kalman-filter update
(`boxmot.motion.kalman_filters.base.BaseKalmanFilter.update_state`)
factorizes the innovation covariance with `scipy.linalg.cho_factor`, which
raises `LinAlgError` ("N-th leading minor of the array is not positive
definite") when that covariance drifts numerically non-positive-definite.

This fires in **Deep OC-SORT**'s freeze/unfreeze observation re-update on
dense top-down traffic scenes: the busiest Stage 2 clip crashes within ~20
frames, sparser clips survive to ~1700, and every Deep OC-SORT run is
truncated. The other three trackers (ByteTrack, BoT-SORT, OC-SORT) reach
the degenerate branch rarely enough that they complete.

Fix policy (benchmark-fair):
- The normal path is left **bit-identical to upstream** -- `update_state`
  is only wrapped, not reimplemented. An update that factorizes cleanly is
  untouched.
- ONLY a would-be-crash triggers a minimal repair: symmetrize the state
  covariance ``P`` and add an escalating diagonal jitter, then retry the
  *original* update. No tracker hyperparameter is changed, so cross-tracker
  comparison is preserved.
- Applied at runtime (monkeypatch), so the locked environment and
  ``site-packages`` are untouched and the fix is reproducible from the repo.

Call `apply_boxmot_patches()` before constructing any BoxMOT tracker;
`BaseTrackerWrapper.__init__` does this. It is idempotent.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_APPLIED = False
_N_REPAIRS = 0


def repair_count() -> int:
    """Number of KF covariance repairs performed since the patch was applied."""
    return _N_REPAIRS


def apply_boxmot_patches() -> None:
    """Install the robust-Cholesky guard on BoxMOT's Kalman filter (once)."""
    global _APPLIED
    if _APPLIED:
        return

    from boxmot.motion.kalman_filters import base as kf_base

    kf_cls = kf_base.BaseKalmanFilter
    orig_update_state = kf_cls.update_state

    def robust_update_state(self, z, R=None, H=None):
        try:
            return orig_update_state(self, z, R=R, H=H)
        except np.linalg.LinAlgError as err:
            # The innovation covariance H P H^T + R failed Cholesky because
            # the state covariance P has drifted *genuinely indefinite*
            # (observed eigenvalues ~ -8 on dense scenes) during Deep
            # OC-SORT's freeze/unfreeze observation re-update -- not merely
            # near-singular, so a small diagonal jitter cannot rescue it.
            #
            # Project P onto the nearest positive-semidefinite matrix by
            # flooring its eigenvalues, then retry the untouched original
            # update. With P PSD and R positive-definite, H P H^T + R is PD.
            # Floors escalate only if needed; the first is microscopic, so a
            # healthy update never reaches here and is bit-identical upstream.
            global _N_REPAIRS
            P = np.nan_to_num(np.asarray(self.P, dtype=float),
                              nan=0.0, posinf=1e9, neginf=-1e9)
            P = 0.5 * (P + P.T)
            eigvals, eigvecs = np.linalg.eigh(P)
            scale = float(np.max(np.abs(eigvals))) or 1.0
            for floor in (1e-6 * scale, 1e-3 * scale, 1e-1 * scale, scale):
                clipped = np.clip(eigvals, floor, None)
                self.P = (eigvecs * clipped) @ eigvecs.T
                try:
                    result = orig_update_state(self, z, R=R, H=H)
                    _N_REPAIRS += 1
                    if _N_REPAIRS == 1:
                        logger.warning(
                            "BoxMOT KF covariance went indefinite; repaired "
                            "via PSD projection (eigenvalue floor). Further "
                            "repairs are silent; call repair_count() for the "
                            "running total.")
                    return result
                except np.linalg.LinAlgError:
                    continue
            # Even full-rank loading failed -- surface it rather than
            # silently corrupting tracker state.
            raise err

    kf_cls.update_state = robust_update_state
    _APPLIED = True
    logger.info(
        "Applied BoxMOT robust-Cholesky KF patch "
        "(BaseKalmanFilter.update_state: regularize-and-retry on non-PD)."
    )
