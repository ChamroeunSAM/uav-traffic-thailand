"""Trajectory-quality metrics in metric world coordinates (Stage 4, Phase C).

Scores the world-coordinate trajectories produced by ``04_extract_trajectories.py``
(stored as ``.npz`` long-format tables under
``data/video_annotations/trajectories/``) along the three axes pre-registered in
decision_log **D4**:

  Axis 1 -- identity persistence (vs GT): completeness (fraction of each GT
            track's lifetime that a prediction covers), fragments per GT track,
            ID switches, mostly-tracked fraction. Largely redundant with the
            Stage 3 box metrics -- reported to confirm.
  Axis 2 -- positional / kinematic accuracy (vs matched GT, world units):
            ADE / FDE / position-RMSE (m), speed MAE (m/s), heading error (deg).
            Invisible to Stage 3: a box can stay within IoU tolerance every
            frame yet jitter, and that jitter dominates differentiated speed.
  Axis 3 -- intrinsic physical plausibility (GT-free): fraction of acceleration
            samples within a physical bound, jerk distribution, path curvature,
            and speed sanity. The novel, reusable core -- it scores trajectories
            where no GT exists (the Paper 2 / SSM use case).

GT<->prediction correspondence is established by **per-frame greedy nearest
matching in world metres** under a distance gate (``match_gate_m``): Stage 3
matches boxes by pixel IoU; Axis 2 needs metric correspondence instead, because
the whole point is to measure world-space position/speed error that IoU cannot
see. Matching always uses the *smoothed* world position (the most stable signal);
accuracy/plausibility are then reported for both the raw and the smoothed series
so smoothing is a studied variable (D4), not a hidden step.

Inputs are the ``.npz`` files written by ``04_extract_trajectories.py``; the
column layout is taken from each file's stored ``columns`` array, not assumed.
All outputs are SI (metres, m/s, m/s^2, m/s^3, degrees). Metrics are class-
agnostic (D4 author preference).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- physical-plausibility reference bounds (Axis 3) -------------------------
# Comfortable car braking is ~0.3-0.5 g; ~0.8 g (8 m/s^2) is about the tyre-grip
# ceiling on dry road, so |a| above it is treated as non-physical (tracker
# jitter, not vehicle motion). The speed cap rejects gross identity-swap
# teleports that survived extraction. Both are reference bounds for *scoring*
# plausibility, not filters applied to the data.
A_MAX_MPS2 = 8.0          # accel magnitude above this is implausible
SPEED_CAP_MPS = 41.7      # ~150 km/h; above this is implausible on these roads
MOVE_SPEED_MPS = 1.0      # below this, heading / curvature are undefined (at rest)
MATCH_GATE_M = 5.0        # max GT<->pred world distance to call a frame a match


@dataclass
class TrajParams:
    """Tunable thresholds; defaults are the D4 reference bounds above."""

    a_max: float = A_MAX_MPS2
    speed_cap: float = SPEED_CAP_MPS
    move_speed: float = MOVE_SPEED_MPS
    match_gate: float = MATCH_GATE_M


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_traj(npz_path: str | Path) -> tuple[np.ndarray, dict[str, int], dict]:
    """Load a trajectory ``.npz`` -> (data (R,C) float64, col->index, meta dict)."""
    d = np.load(Path(npz_path), allow_pickle=True)
    cols = [str(c) for c in d["columns"]]
    idx = {c: i for i, c in enumerate(cols)}
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    return np.asarray(d["data"], dtype=np.float64), idx, meta


def _by_frame(data: np.ndarray, idx: dict[str, int]) -> dict[int, np.ndarray]:
    """Group row indices by integer frame number."""
    frames = data[:, idx["frame"]].astype(int)
    out: dict[int, list[int]] = {}
    for i, fr in enumerate(frames):
        out.setdefault(int(fr), []).append(i)
    return {fr: np.asarray(rows, dtype=int) for fr, rows in out.items()}


# ---------------------------------------------------------------------------
# Raw-kinematics recompute (the .npz stores smoothed accel/jerk but only raw
# *speed*; Axis-2/3 "pre-smoothing" need raw accel/jerk, so recompute them per
# segment from the raw world positions on the same uniform per-frame grid).
# ---------------------------------------------------------------------------
def raw_kinematics(data: np.ndarray, idx: dict[str, int], dt: float) -> dict[str, np.ndarray]:
    """Per-row raw velocity/accel/jerk/heading recomputed from raw (x_m, y_m).

    Each ``seg`` is a contiguous, gap-filled, strictly per-frame run (built by
    04_extract_trajectories.py), so differentiating within a segment is valid.
    Returns arrays aligned to ``data`` rows: vx, vy, ax, ay, accel_mag,
    jerk_mag, speed, heading_deg.
    """
    from trajectory.kinematics import compute_kinematics

    n = data.shape[0]
    out = {k: np.zeros(n) for k in
           ("vx", "vy", "ax", "ay", "accel_mag", "jerk_mag", "speed", "heading_deg")}
    seg = data[:, idx["seg"]].astype(int)
    fr = data[:, idx["frame"]].astype(int)
    xy = data[:, [idx["x_m"], idx["y_m"]]]
    for s in np.unique(seg):
        rows = np.where(seg == s)[0]
        rows = rows[np.argsort(fr[rows])]      # frame order (already sorted, be safe)
        kin = compute_kinematics(xy[rows], dt)
        out["vx"][rows] = kin["velocity"][:, 0]
        out["vy"][rows] = kin["velocity"][:, 1]
        out["ax"][rows] = kin["accel"][:, 0]
        out["ay"][rows] = kin["accel"][:, 1]
        out["accel_mag"][rows] = kin["accel_mag"]
        out["jerk_mag"][rows] = kin["jerk_mag"]
        out["speed"][rows] = kin["speed"]
        out["heading_deg"][rows] = kin["heading_deg"]
    return out


# ---------------------------------------------------------------------------
# Axis 3 -- intrinsic physical plausibility (GT-free)
# ---------------------------------------------------------------------------
def axis3_plausibility(data: np.ndarray, idx: dict[str, int], meta: dict,
                       variant: str, p: TrajParams) -> dict:
    """Plausibility of one trajectory set, GT-free.

    ``variant`` is "smoothed" (uses the stored smoothed kinematics) or "raw"
    (recomputes kinematics from the raw world positions). Curvature kappa =
    |vx*ay - vy*ax| / speed^3 is evaluated only on moving samples.
    """
    dt = 1.0 / float(meta.get("fps", 29.97))
    if variant == "smoothed":
        speed = data[:, idx["speed_mps"]]
        accel = data[:, idx["accel_mps2"]]
        jerk = data[:, idx["jerk_mps3"]]
        vx, vy = data[:, idx["vx"]], data[:, idx["vy"]]
        ax, ay = data[:, idx["ax"]], data[:, idx["ay"]]
    elif variant == "raw":
        k = raw_kinematics(data, idx, dt)
        speed, accel, jerk = k["speed"], k["accel_mag"], k["jerk_mag"]
        vx, vy, ax, ay = k["vx"], k["vy"], k["ax"], k["ay"]
    else:
        raise ValueError(f"variant must be 'smoothed' or 'raw', got {variant!r}")

    moving = speed > p.move_speed
    cross = np.abs(vx * ay - vy * ax)
    curv = np.full(speed.shape, np.nan)
    sp3 = speed[moving] ** 3
    curv[moving] = np.where(sp3 > 1e-9, cross[moving] / sp3, np.nan)
    curv_moving = curv[moving & np.isfinite(curv)]

    n = speed.shape[0]
    return {
        "n_samples": int(n),
        "frac_accel_ok": float(np.mean(accel <= p.a_max)) if n else float("nan"),
        "frac_speed_ok": float(np.mean(speed <= p.speed_cap)) if n else float("nan"),
        "accel_p95_mps2": float(np.percentile(accel, 95)) if n else float("nan"),
        "accel_max_mps2": float(accel.max()) if n else float("nan"),
        "jerk_mean_mps3": float(np.mean(jerk)) if n else float("nan"),
        "jerk_p95_mps3": float(np.percentile(jerk, 95)) if n else float("nan"),
        "curv_mean_inv_m": float(np.mean(curv_moving)) if curv_moving.size else float("nan"),
        "speed_median_kmh": float(np.median(speed) * 3.6) if n else float("nan"),
        "speed_p95_kmh": float(np.percentile(speed, 95) * 3.6) if n else float("nan"),
        "speed_max_kmh": float(speed.max() * 3.6) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# GT <-> prediction matching (per-frame greedy nearest, gated, world metres)
# ---------------------------------------------------------------------------
@dataclass
class Matches:
    """Per-frame matched GT<->pred point pairs (parallel arrays)."""

    frame: np.ndarray       # (M,) int
    gt_id: np.ndarray       # (M,) int  GT track_id
    pr_id: np.ndarray       # (M,) int  pred track_id
    gt_row: np.ndarray      # (M,) int  row index into gt_data
    pr_row: np.ndarray      # (M,) int  row index into pr_data
    dist: np.ndarray        # (M,) float  smoothed-position distance (m)


def match_frames(gt: np.ndarray, gi: dict[str, int],
                 pr: np.ndarray, pj: dict[str, int], gate: float) -> Matches:
    """Greedy nearest GT<->pred matching per frame on smoothed world position.

    Within each shared frame, candidate pairs under ``gate`` metres are matched
    shortest-distance first, each GT and pred point used at most once.
    """
    gt_frames = _by_frame(gt, gi)
    pr_frames = _by_frame(pr, pj)
    gxy = gt[:, [gi["xs_m"], gi["ys_m"]]]
    pxy = pr[:, [pj["xs_m"], pj["ys_m"]]]
    gid = gt[:, gi["track_id"]].astype(int)
    pid = pr[:, pj["track_id"]].astype(int)

    fr_out, gid_out, pid_out, gr_out, pr_out, d_out = [], [], [], [], [], []
    for fr in sorted(set(gt_frames) & set(pr_frames)):
        grows, prows = gt_frames[fr], pr_frames[fr]
        d = np.linalg.norm(gxy[grows][:, None, :] - pxy[prows][None, :, :], axis=2)
        cand = np.argwhere(d <= gate)
        if cand.size == 0:
            continue
        order = np.argsort(d[cand[:, 0], cand[:, 1]])
        g_used, p_used = set(), set()
        for k in order:
            a, b = cand[k]
            if a in g_used or b in p_used:
                continue
            g_used.add(a); p_used.add(b)
            fr_out.append(fr)
            gid_out.append(int(gid[grows[a]])); pid_out.append(int(pid[prows[b]]))
            gr_out.append(int(grows[a])); pr_out.append(int(prows[b]))
            d_out.append(float(d[a, b]))
    arr = lambda L, t: np.asarray(L, dtype=t)  # noqa: E731
    return Matches(arr(fr_out, int), arr(gid_out, int), arr(pid_out, int),
                   arr(gr_out, int), arr(pr_out, int), arr(d_out, float))


# ---------------------------------------------------------------------------
# Axis 1 -- identity persistence (vs GT)
# ---------------------------------------------------------------------------
def axis1_identity(gt: np.ndarray, gi: dict[str, int], m: Matches) -> dict:
    """Completeness, fragments-per-GT, ID switches, mostly-tracked fraction.

    Completeness of a GT track = matched frames / total GT frames for that
    track. Fragments = number of distinct predicted track_ids that cover it
    (ideal 1). ID switches = transitions between predicted ids along the GT
    track in frame order.
    """
    gt_frames = gt[:, gi["frame"]].astype(int)
    gt_ids = gt[:, gi["track_id"]].astype(int)
    total = {tid: len(np.unique(gt_frames[gt_ids == tid]))
             for tid in np.unique(gt_ids)}

    completeness, frags, switches = [], [], []
    for tid in total:
        sel = m.gt_id == tid
        matched_frames = np.unique(m.frame[sel])
        completeness.append(len(matched_frames) / total[tid])
        pr_ids = m.pr_id[sel]
        frags.append(len(np.unique(pr_ids)) if pr_ids.size else 0)
        if pr_ids.size > 1:
            order = np.argsort(m.frame[sel])
            seq = pr_ids[order]
            switches.append(int(np.sum(seq[1:] != seq[:-1])))
        else:
            switches.append(0)

    completeness = np.asarray(completeness, dtype=float)
    return {
        "n_gt_tracks": len(total),
        "completeness_mean": float(completeness.mean()) if completeness.size else float("nan"),
        "frag_per_gt_mean": float(np.mean(frags)) if frags else float("nan"),
        "id_switches": int(np.sum(switches)),
        "mostly_tracked_frac": float(np.mean(completeness >= 0.8)) if completeness.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Axis 2 -- positional / kinematic accuracy (vs matched GT)
# ---------------------------------------------------------------------------
def _wrap_deg(a: np.ndarray) -> np.ndarray:
    """Smallest absolute angular difference, degrees in [0, 180]."""
    return np.abs((a + 180.0) % 360.0 - 180.0)


def axis2_accuracy(gt: np.ndarray, gi: dict[str, int],
                   pr: np.ndarray, pj: dict[str, int],
                   m: Matches, variant: str, p: TrajParams) -> dict:
    """Position/speed/heading error over matched GT<->pred frames.

    ``variant`` selects raw vs smoothed series for the position, speed, and
    heading channels (matching itself is always on the smoothed position).
    FDE is the position error at each GT track's last matched frame, averaged.
    """
    if m.frame.size == 0:
        return {"n_matched": 0, "ade_m": float("nan"), "fde_m": float("nan"),
                "pos_rmse_m": float("nan"), "speed_mae_mps": float("nan"),
                "heading_mae_deg": float("nan")}

    if variant == "smoothed":
        gp = gt[:, [gi["xs_m"], gi["ys_m"]]]
        pp = pr[:, [pj["xs_m"], pj["ys_m"]]]
        gs, ps = gt[:, gi["speed_mps"]], pr[:, pj["speed_mps"]]
        gh, ph = gt[:, gi["heading_deg"]], pr[:, pj["heading_deg"]]
    else:  # raw
        gp = gt[:, [gi["x_m"], gi["y_m"]]]
        pp = pr[:, [pj["x_m"], pj["y_m"]]]
        gs, ps = gt[:, gi["speed_raw_mps"]], pr[:, pj["speed_raw_mps"]]
        gh = raw_kinematics(gt, gi, 1.0 / 29.97)["heading_deg"]  # dt cancels in heading
        ph = raw_kinematics(pr, pj, 1.0 / 29.97)["heading_deg"]

    gr, prr = m.gt_row, m.pr_row
    pos_err = np.linalg.norm(gp[gr] - pp[prr], axis=1)
    ade = float(pos_err.mean())
    pos_rmse = float(np.sqrt(np.mean(pos_err ** 2)))
    speed_mae = float(np.mean(np.abs(gs[gr] - ps[prr])))

    # heading error only where both are genuinely moving
    move = (np.abs(gs[gr]) > p.move_speed) & (np.abs(ps[prr]) > p.move_speed)
    heading_mae = (float(np.mean(_wrap_deg(gh[gr][move] - ph[prr][move])))
                   if move.any() else float("nan"))

    # FDE: error at each GT track's last matched frame
    fde_vals = []
    for tid in np.unique(m.gt_id):
        sel = m.gt_id == tid
        last = m.frame[sel].argmax()
        fde_vals.append(pos_err[np.where(sel)[0][last]])
    fde = float(np.mean(fde_vals)) if fde_vals else float("nan")

    return {"n_matched": int(m.frame.size), "ade_m": ade, "fde_m": fde,
            "pos_rmse_m": pos_rmse, "speed_mae_mps": speed_mae,
            "heading_mae_deg": heading_mae}


# ---------------------------------------------------------------------------
# Top-level: evaluate one prediction file (against GT) for both variants
# ---------------------------------------------------------------------------
def evaluate_against_gt(pred_npz: str | Path, gt_npz: str | Path,
                        p: TrajParams | None = None) -> dict[str, dict]:
    """All three axes for a prediction trajectory vs its GT, per smoothing variant.

    Returns ``{"smoothed": {...}, "raw": {...}}`` where each inner dict carries
    the Axis-1/2/3 metrics flattened with ``a1_``/``a2_``/``a3_`` prefixes.
    Axis 1 is matching-only (variant-independent) so it repeats across variants.
    """
    p = p or TrajParams()
    pr, pj, pmeta = load_traj(pred_npz)
    gt, gi, _ = load_traj(gt_npz)
    m = match_frames(gt, gi, pr, pj, p.match_gate)
    a1 = axis1_identity(gt, gi, m)
    out = {}
    for variant in ("smoothed", "raw"):
        a2 = axis2_accuracy(gt, gi, pr, pj, m, variant, p)
        a3 = axis3_plausibility(pr, pj, pmeta, variant, p)
        out[variant] = {
            **{f"a1_{k}": v for k, v in a1.items()},
            **{f"a2_{k}": v for k, v in a2.items()},
            **{f"a3_{k}": v for k, v in a3.items()},
        }
    return out


def evaluate_plausibility_only(npz: str | Path,
                               p: TrajParams | None = None) -> dict[str, dict]:
    """Axis-3 only (GT-free), per variant -- used for the GT reference ceiling."""
    p = p or TrajParams()
    data, idx, meta = load_traj(npz)
    return {variant: {f"a3_{k}": v
                      for k, v in axis3_plausibility(data, idx, meta, variant, p).items()}
            for variant in ("smoothed", "raw")}
