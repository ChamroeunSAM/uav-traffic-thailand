"""HOTA / MOTA / IDF1 via TrackEval -- the MOTChallenge reference implementation.

We do NOT reimplement the tracking metrics. TrackEval's HOTA / CLEAR / Identity
classes do the math (they are the implementations MOTChallenge and the HOTA
paper use). This module only:

  * loads MOT-Challenge txt (ground truth and tracker output),
  * builds the per-sequence `data` dict that TrackEval's `eval_sequence()`
    expects -- dense 0-indexed ids plus per-timestep IoU similarity matrices,
  * runs the three metric classes and flattens the results to scalars,
  * combines sequences with TrackEval's own accumulators (NOT by averaging).

Class handling: tracking metrics are evaluated **class-agnostically** by
default (identity preservation, the MOTChallenge convention). Pass `cls` to
restrict to a single vehicle class id (0=LV, 1=MV, 2=SV) for a per-class
breakdown.

Input format (both GT and tracker), 1-indexed frames, 0-based class:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, -1, -1
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from trackeval.metrics import CLEAR, HOTA, Identity

logger = logging.getLogger(__name__)

CLASS_NAMES = ["LV", "MV", "SV"]


def _make_metrics():
    """Instantiate the three metric objects (config print suppressed)."""
    def safe(cls):
        try:
            return cls({"PRINT_CONFIG": False})
        except Exception:  # noqa: BLE001 -- some metrics lack PRINT_CONFIG
            return cls()
    return safe(HOTA), safe(CLEAR), safe(Identity)


def load_mot(path: str | Path, cls: int | None = None) -> dict[int, list[tuple]]:
    """Load a MOT txt as {frame: [(track_id, x, y, w, h, class), ...]}.

    If `cls` is given, keep only rows with that class id.
    """
    by_frame: dict[int, list[tuple]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frame = int(p[0])
            tid = int(p[1])
            x, y, w, h = (float(v) for v in p[2:6])
            c = int(float(p[7])) if len(p) > 7 and p[7] not in ("", "-1") else -1
            if cls is not None and c != cls:
                continue
            by_frame[frame].append((tid, x, y, w, h, c))
    return dict(by_frame)


def _iou_matrix(g: np.ndarray, t: np.ndarray) -> np.ndarray:
    """IoU between gt boxes g (G,4) and tracker boxes t (T,4), both xywh."""
    if len(g) == 0 or len(t) == 0:
        return np.zeros((len(g), len(t)), dtype=float)
    gx1, gy1, gx2, gy2 = g[:, 0], g[:, 1], g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    tx1, ty1, tx2, ty2 = t[:, 0], t[:, 1], t[:, 0] + t[:, 2], t[:, 1] + t[:, 3]
    ix1 = np.maximum(gx1[:, None], tx1[None, :])
    iy1 = np.maximum(gy1[:, None], ty1[None, :])
    ix2 = np.minimum(gx2[:, None], tx2[None, :])
    iy2 = np.minimum(gy2[:, None], ty2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ga = (g[:, 2] * g[:, 3])[:, None]
    ta = (t[:, 2] * t[:, 3])[None, :]
    union = ga + ta - inter
    return np.where(union > 0, inter / union, 0.0)


def build_seq_data(gt_bf: dict[int, list[tuple]],
                   tr_bf: dict[int, list[tuple]]) -> dict:
    """Build the per-sequence `data` dict TrackEval's eval_sequence expects.

    Track ids are remapped to dense 0-based indices (TrackEval uses them as
    array indices). Timesteps run 1..max_frame in order (CLEAR's ID-switch
    logic depends on temporal continuity).
    """
    gt_ids_all = sorted({r[0] for rows in gt_bf.values() for r in rows})
    tr_ids_all = sorted({r[0] for rows in tr_bf.values() for r in rows})
    gmap = {rid: i for i, rid in enumerate(gt_ids_all)}
    tmap = {rid: i for i, rid in enumerate(tr_ids_all)}

    max_frame = max([*gt_bf.keys(), *tr_bf.keys()], default=0)

    gt_ids: list[np.ndarray] = []
    tracker_ids: list[np.ndarray] = []
    sims: list[np.ndarray] = []
    n_gt = n_tr = 0
    for fr in range(1, max_frame + 1):
        g = gt_bf.get(fr, [])
        t = tr_bf.get(fr, [])
        gt_ids.append(np.array([gmap[r[0]] for r in g], dtype=int))
        tracker_ids.append(np.array([tmap[r[0]] for r in t], dtype=int))
        gb = np.array([r[1:5] for r in g], dtype=float).reshape(-1, 4)
        tb = np.array([r[1:5] for r in t], dtype=float).reshape(-1, 4)
        sims.append(_iou_matrix(gb, tb))
        n_gt += len(g)
        n_tr += len(t)

    return {
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": sims,
        "num_gt_dets": n_gt,
        "num_tracker_dets": n_tr,
        "num_gt_ids": len(gt_ids_all),
        "num_tracker_ids": len(tr_ids_all),
        "num_timesteps": max_frame,
    }


def eval_data(data: dict) -> dict:
    """Run the three metric classes on a prepared data dict -> raw res dict."""
    hota, clear, ident = _make_metrics()
    return {
        "HOTA": hota.eval_sequence(data),
        "CLEAR": clear.eval_sequence(data),
        "Identity": ident.eval_sequence(data),
    }


def evaluate_files(gt_file: str | Path, tr_file: str | Path,
                   cls: int | None = None) -> dict:
    """Evaluate one tracker file against one GT file -> raw res dict."""
    data = build_seq_data(load_mot(gt_file, cls), load_mot(tr_file, cls))
    return eval_data(data)


def combine(raw_list: list[dict]) -> dict:
    """Combine per-sequence raw results into one, via TrackEval accumulators."""
    hota, clear, ident = _make_metrics()
    return {
        "HOTA": hota.combine_sequences({i: r["HOTA"] for i, r in enumerate(raw_list)}),
        "CLEAR": clear.combine_sequences({i: r["CLEAR"] for i, r in enumerate(raw_list)}),
        "Identity": ident.combine_sequences({i: r["Identity"] for i, r in enumerate(raw_list)}),
    }


def summarize(raw: dict) -> dict:
    """Flatten a raw res dict to the headline scalars + key sub-metrics."""
    h, c, i = raw["HOTA"], raw["CLEAR"], raw["Identity"]
    mean = lambda x: float(np.mean(x))  # noqa: E731
    out = {
        "HOTA": mean(h["HOTA"]),
        "DetA": mean(h["DetA"]),
        "AssA": mean(h["AssA"]),
        "LocA": mean(h["LocA"]),
        "MOTA": float(c["MOTA"]),
        "MOTP": float(c["MOTP"]),
        "IDF1": float(i["IDF1"]),
        "IDR": float(i["IDR"]),
        "IDP": float(i["IDP"]),
    }
    for k in ("IDSW", "Frag", "MT", "PT", "ML", "CLR_TP", "CLR_FP", "CLR_FN"):
        if k in c:
            out[k] = int(c[k])
    return out
