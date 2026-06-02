"""Link fragmented tracks in MOT pseudo-track output via offline post-processing.

Stage 2 helper. ByteTrack splits a single vehicle's trajectory into multiple
IDs for several reasons: marking-crossing detector misses, brief partial
occlusion, and detector double-detection that NMS at iou=0.6 can't suppress.
This script applies three passes to consolidate the fragments.

Passes (run in this order):

  1. Overlap-merge -- when two tracks are alive in the same frames with mean
     IoU >= --overlap-min-iou over >= --overlap-min-shared shared frames,
     they're the same vehicle being tracked as two parallel IDs (caused by
     persistent detector double-boxes that survive NMS, including cross-class
     LV/MV duplicates on the same vehicle). Keep the longer-lived track.

  2. Gap-link -- match the END of one track to the START of another when all
     of: same mode class, median box areas within --size-tol, frame gap <=
     --max-gap, and (if --use-position) start position within --dist-tol box
     diagonals of A's constant-velocity extrapolation.

  3. Per-frame dedup -- transient single-frame double detections that escaped
     pass 1 because the spurious track was <5 frames long: at any frame with
     two boxes whose IoU >= --dedup-min-iou, drop the row from the shorter
     track (tie-break: lower confidence). Empty tracks are dropped.

Each pass iterates to convergence. Output is a new MOT-format txt; original
input is not modified.

Defaults are tuned for top-down UAV traffic at 100-120 m altitude based on
empirical validation -- in particular --use-position defaults to OFF because
constant-velocity Kalman extrapolation is mismatched to turning vehicles in
intersection / roundabout scenes.

Example:
    python scripts/02b_link_tracks.py \\
        --input data/video_annotations/pseudo_tracks/120_m_DJI_0047_1min.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True,
                   help="MOT-format pseudo-track txt from 02_generate_pseudo_tracks.py.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output path. Default: <input>.linked.txt alongside input.")
    p.add_argument("--max-gap", type=int, default=180,
                   help="Max gap in frames between A's end and B's start. "
                        "Default 180 (~6 s at 29.97 fps) -- tuned for top-down "
                        "UAV traffic where vehicles can be briefly hidden by "
                        "trees / overpasses / signal posts.")
    p.add_argument("--size-tol", type=float, default=0.5,
                   help="Allowed fractional area difference between A's tail "
                        "median size and B's head median size. Default 0.5 -- "
                        "permissive because partial-entry/exit detections on "
                        "the frame edge distort the median.")
    p.add_argument("--dist-tol", type=float, default=20.0,
                   help="Allowed position error as multiple of box diagonal. "
                        "Default 20.0 -- only relevant when --use-position "
                        "is passed.")
    p.add_argument("--use-position", action="store_true",
                   help="Enable position-extrapolation check (off by default). "
                        "Constant-velocity Kalman extrapolation is mismatched "
                        "to turning / decelerating UAV traffic and rejected "
                        "more valid merges than it caught in our scenes. "
                        "Pass this flag for sparse-traffic clips where false "
                        "merges are expensive.")
    p.add_argument("--edge-frames", type=int, default=5,
                   help="Number of frames at each track edge to use for "
                        "median size and velocity estimation. Default 5.")
    p.add_argument("--no-overlap", action="store_true",
                   help="Skip the overlap-merge pass (parallel-track dedup).")
    p.add_argument("--overlap-min-shared", type=int, default=5,
                   help="Min shared frames for overlap-merge consideration. "
                        "Default 5.")
    p.add_argument("--overlap-min-iou", type=float, default=0.5,
                   help="Min mean IoU on shared frames to merge as parallel "
                        "duplicates. Default 0.5.")
    p.add_argument("--no-dedup", action="store_true",
                   help="Skip the per-frame dedup pass (transient doubles).")
    p.add_argument("--dedup-min-iou", type=float, default=0.6,
                   help="Min IoU between two boxes in the same frame to treat "
                        "as a duplicate detection. Default 0.6.")
    return p.parse_args()


# ---------- IO ----------

def load_tracks(path: Path) -> dict[int, list[tuple]]:
    """Load MOT txt as {track_id: [(frame, x, y, w, h, conf, cls), ...]}."""
    tracks: dict[int, list[tuple]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            cls = int(parts[7])
            tracks[tid].append((frame, x, y, w, h, conf, cls))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return dict(tracks)


def write_tracks(tracks: dict[int, list[tuple]], path: Path) -> None:
    rows = []
    for tid, rs in tracks.items():
        for r in rs:
            rows.append((r[0], tid, r[1], r[2], r[3], r[4], r[5], r[6]))
    rows.sort(key=lambda r: (r[0], r[1]))
    with path.open("w", encoding="utf-8") as f:
        for frame, tid, x, y, w, h, conf, cls in rows:
            f.write(
                f"{frame},{tid},{x:.2f},{y:.2f},"
                f"{w:.2f},{h:.2f},{conf:.4f},{cls},-1,-1\n"
            )


# ---------- per-track features ----------

def summarize(rows: list[tuple], edge: int) -> dict:
    """Derive merging features from a single track's rows (already frame-sorted)."""
    n = len(rows)
    n_edge = min(edge, n)

    frames = np.array([r[0] for r in rows])
    centers = np.array([(r[1] + r[3] / 2.0, r[2] + r[4] / 2.0) for r in rows])
    areas = np.array([r[3] * r[4] for r in rows])

    mode_cls = Counter(r[6] for r in rows).most_common(1)[0][0]

    head_area = float(np.median(areas[:n_edge]))
    tail_area = float(np.median(areas[-n_edge:]))

    head_center = centers[0]
    tail_center = centers[-1]

    if n_edge >= 2:
        dframes = max(int(frames[-1] - frames[-n_edge]), 1)
        tail_vel = (centers[-1] - centers[-n_edge]) / dframes
    else:
        tail_vel = np.zeros(2)

    tail_w, tail_h = rows[-1][3], rows[-1][4]
    head_w, head_h = rows[0][3], rows[0][4]
    tail_diag = float(np.hypot(tail_w, tail_h))
    head_diag = float(np.hypot(head_w, head_h))

    return {
        "start_frame": int(frames[0]),
        "end_frame": int(frames[-1]),
        "mode_cls": int(mode_cls),
        "head_area": head_area,
        "tail_area": tail_area,
        "head_center": head_center,
        "tail_center": tail_center,
        "tail_vel": tail_vel,
        "head_diag": head_diag,
        "tail_diag": tail_diag,
        "n_rows": n,
    }


# ---------- merge scoring ----------

def merge_score(a: dict, b: dict, max_gap: int, size_tol: float,
                dist_tol: float, use_position: bool) -> float | None:
    """Return a score for merging tail A -> head B, or None if the pair fails any gate.

    Lower score is better. Returns None when any required condition fails.
    """
    gap = b["start_frame"] - a["end_frame"]
    if gap < 1 or gap > max_gap:
        return None
    if a["mode_cls"] != b["mode_cls"]:
        return None

    ratio = b["head_area"] / max(a["tail_area"], 1e-6)
    if ratio < (1.0 - size_tol) or ratio > (1.0 + size_tol):
        return None
    size_err = abs(ratio - 1.0)

    pos_err_norm = 0.0
    if use_position:
        predicted = a["tail_center"] + a["tail_vel"] * gap
        err = float(np.linalg.norm(b["head_center"] - predicted))
        diag = max(a["tail_diag"], b["head_diag"], 1.0)
        if err > dist_tol * diag:
            return None
        pos_err_norm = err / diag

    # Lower is better. Weight gap mildly so tighter gaps win ties.
    return pos_err_norm + size_err + 0.01 * gap


# ---------- single-pass merging ----------

def find_one_pass_merges(summaries: dict[int, dict], max_gap: int,
                         size_tol: float, dist_tol: float,
                         use_position: bool) -> list[tuple[int, int]]:
    """Greedy assignment: each tail picks its best head; each head accepted once."""
    candidates: list[tuple[float, int, int]] = []
    ids = list(summaries.keys())
    for tail in ids:
        a = summaries[tail]
        for head in ids:
            if tail == head:
                continue
            b = summaries[head]
            score = merge_score(a, b, max_gap, size_tol, dist_tol, use_position)
            if score is not None:
                candidates.append((score, tail, head))
    candidates.sort(key=lambda x: x[0])

    used_tails: set[int] = set()
    used_heads: set[int] = set()
    merges: list[tuple[int, int]] = []
    for _score, tail, head in candidates:
        if tail in used_tails or head in used_heads:
            continue
        merges.append((tail, head))
        used_tails.add(tail)
        used_heads.add(head)
    return merges


def apply_merges(tracks: dict[int, list[tuple]],
                 merges: list[tuple[int, int]]) -> dict[int, list[tuple]]:
    """Relabel each second-id's rows with its first-id's ID. Resolves transitive
    chains. For frames where the merged tracks both had a row (overlap-merge
    case), keeps the higher-confidence row.
    """
    # merges contains (kept_id, removed_id) pairs; removed gets relabeled to kept.
    # For gap-merge: kept = tail, removed = head (head's start is after tail's end).
    # For overlap-merge: kept = longer-lived track, removed = shorter parallel.
    head_to_tail = {head: tail for tail, head in merges}

    def canonical(tid: int) -> int:
        seen = set()
        while tid in head_to_tail:
            if tid in seen:
                break
            seen.add(tid)
            tid = head_to_tail[tid]
        return tid

    new_tracks: dict[int, list[tuple]] = defaultdict(list)
    for tid, rs in tracks.items():
        new_tracks[canonical(tid)].extend(rs)

    # Frame-level dedup: when overlap-merging produces two rows for the same
    # frame (the parallel-duplicate case), keep the higher-confidence row.
    # Idempotent for gap-merges since their merged tracks share no frames.
    for tid in new_tracks:
        by_frame: dict[int, tuple] = {}
        for r in new_tracks[tid]:
            f = r[0]
            if f not in by_frame or r[5] > by_frame[f][5]:
                by_frame[f] = r
        new_tracks[tid] = sorted(by_frame.values(), key=lambda r: r[0])
    return dict(new_tracks)


# ---------- overlap-merge (parallel-duplicate pass) ----------

def _box_iou(a: tuple, b: tuple) -> float:
    """IoU between two boxes given as (x, y, w, h)."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    xa, ya = max(ax1, bx1), max(ay1, by1)
    xb, yb = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def find_one_pass_overlap_merges(tracks: dict[int, list[tuple]],
                                 min_shared: int,
                                 min_iou: float) -> list[tuple[int, int]]:
    """Find pairs of tracks that overlap consistently -- the same vehicle being
    tracked as two parallel IDs (caused by detector double-boxes that survive NMS).

    Returns [(keep_id, remove_id), ...]. Each track participates in at most one
    merge per pass (greedy by mean IoU); iterate to convergence.
    """
    # Build per-track frame index for fast intersection.
    frame_idx: dict[int, dict[int, tuple]] = {
        tid: {r[0]: r for r in rs} for tid, rs in tracks.items()
    }
    ids = list(frame_idx.keys())

    candidates: list[tuple[float, int, int, int]] = []  # (-iou, shared, keep, remove)
    for i, a in enumerate(ids):
        fa = frame_idx[a]
        for b in ids[i + 1:]:
            fb = frame_idx[b]
            # Skip without computing IoU when there's no temporal overlap.
            shared_frames = fa.keys() & fb.keys()
            if len(shared_frames) < min_shared:
                continue
            ious = []
            for f in shared_frames:
                ra = fa[f]
                rb = fb[f]
                ious.append(_box_iou(
                    (ra[1], ra[2], ra[3], ra[4]),
                    (rb[1], rb[2], rb[3], rb[4]),
                ))
            mean_iou = sum(ious) / len(ious)
            if mean_iou < min_iou:
                continue
            # Keep the longer-lived track (more total detections).
            if len(fa) >= len(fb):
                keep, remove = a, b
            else:
                keep, remove = b, a
            candidates.append((-mean_iou, len(shared_frames), keep, remove))

    # Best (highest IoU) first.
    candidates.sort()

    used: set[int] = set()
    merges: list[tuple[int, int]] = []
    for _miou, _shared, keep, remove in candidates:
        if keep in used or remove in used:
            continue
        merges.append((keep, remove))
        used.add(keep)
        used.add(remove)
    return merges


def overlap_until_convergence(tracks: dict[int, list[tuple]],
                              min_shared: int,
                              min_iou: float) -> tuple[dict, int, int]:
    total_merges = 0
    iteration = 0
    while True:
        iteration += 1
        merges = find_one_pass_overlap_merges(tracks, min_shared, min_iou)
        if not merges:
            break
        tracks = apply_merges(tracks, merges)
        total_merges += len(merges)
        logger.info("  overlap iter %d: %d merges -> %d tracks remain",
                    iteration, len(merges), len(tracks))
    return tracks, total_merges, iteration


# ---------- per-frame dedup (transient double-detection cleanup) ----------

def dedup_per_frame(tracks: dict[int, list[tuple]],
                    min_iou: float) -> tuple[dict, int, int]:
    """For each frame, remove the box from the shorter track when two boxes
    overlap with IoU >= min_iou. Tie-break by lower confidence.

    Handles transient single-frame doubles that overlap-merge missed because
    the spurious track was too short (<5 frames) to meet its min_shared gate.

    Returns: (cleaned_tracks, n_rows_removed, n_empty_tracks_dropped).
    """
    # Track length at entry -- used for selection. Recomputing per-frame after
    # removals would let earlier deletions shrink a track and change later
    # tie-breaks, which is unstable; freezing length keeps decisions consistent.
    track_len = {tid: len(rs) for tid, rs in tracks.items()}

    # Build per-frame view: {frame: [(tid, row), ...]}
    by_frame: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for tid, rs in tracks.items():
        for r in rs:
            by_frame[r[0]].append((tid, r))

    # Mark (tid, frame) tuples to remove. Using a set keeps lookup O(1).
    drop: set[tuple[int, int]] = set()

    for frame, rows in by_frame.items():
        if len(rows) < 2:
            continue
        n = len(rows)
        # Pairwise check. n is typically small (few-to-tens), so O(n^2) is fine.
        for i in range(n):
            if (rows[i][0], frame) in drop:
                continue
            tid_a, ra = rows[i]
            for j in range(i + 1, n):
                if (rows[j][0], frame) in drop:
                    continue
                tid_b, rb = rows[j]
                box_a = (ra[1], ra[2], ra[3], ra[4])
                box_b = (rb[1], rb[2], rb[3], rb[4])
                if _box_iou(box_a, box_b) < min_iou:
                    continue
                # Decide which to drop: shorter track loses; ties go to lower conf.
                la, lb = track_len[tid_a], track_len[tid_b]
                if la > lb:
                    drop.add((tid_b, frame))
                elif lb > la:
                    drop.add((tid_a, frame))
                    break  # ra is gone for this frame
                elif ra[5] >= rb[5]:
                    drop.add((tid_b, frame))
                else:
                    drop.add((tid_a, frame))
                    break

    if not drop:
        return tracks, 0, 0

    # Filter rows and drop tracks that ended up empty.
    cleaned: dict[int, list[tuple]] = {}
    n_removed = 0
    for tid, rs in tracks.items():
        kept = [r for r in rs if (tid, r[0]) not in drop]
        n_removed += len(rs) - len(kept)
        if kept:
            cleaned[tid] = kept
    n_empty_dropped = len(tracks) - len(cleaned)
    return cleaned, n_removed, n_empty_dropped


# ---------- driver ----------

def link_until_convergence(tracks: dict[int, list[tuple]], max_gap: int,
                           size_tol: float, dist_tol: float,
                           use_position: bool, edge: int) -> tuple[dict, int, int]:
    """Repeatedly link tracks until no more merges happen.

    Returns: (final_tracks, total_merges, n_iterations).
    """
    total_merges = 0
    iteration = 0
    while True:
        iteration += 1
        summaries = {tid: summarize(rs, edge) for tid, rs in tracks.items()}
        merges = find_one_pass_merges(summaries, max_gap, size_tol,
                                      dist_tol, use_position)
        if not merges:
            break
        tracks = apply_merges(tracks, merges)
        total_merges += len(merges)
        logger.info("  iter %d: %d merges -> %d tracks remain",
                    iteration, len(merges), len(tracks))
    return tracks, total_merges, iteration


def class_breakdown(tracks: dict[int, list[tuple]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for rs in tracks.values():
        mode = Counter(r[6] for r in rs).most_common(1)[0][0]
        counts[int(mode)] += 1
    return dict(counts)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logger.error("Input not found: %s", args.input)
        return 2

    out_path = args.output or args.input.with_suffix(".linked.txt")

    logger.info("Loading %s", args.input)
    tracks = load_tracks(args.input)
    n_before = len(tracks)
    cls_before = class_breakdown(tracks)
    logger.info("Loaded %d tracks (LV=%d, MV=%d, SV=%d)",
                n_before,
                cls_before.get(0, 0),
                cls_before.get(1, 0),
                cls_before.get(2, 0))
    logger.info("Config: max_gap=%d  size_tol=%.2f  dist_tol=%.1f  position=%s "
                "overlap=%s (min_shared=%d, min_iou=%.2f) "
                "dedup=%s (min_iou=%.2f)",
                args.max_gap, args.size_tol, args.dist_tol,
                "on" if args.use_position else "off",
                "off" if args.no_overlap else "on",
                args.overlap_min_shared, args.overlap_min_iou,
                "off" if args.no_dedup else "on",
                args.dedup_min_iou)

    overlap_merges = 0
    overlap_iters = 0
    if not args.no_overlap:
        logger.info("Overlap-merge pass (parallel-duplicate dedup):")
        tracks, overlap_merges, overlap_iters = overlap_until_convergence(
            tracks,
            min_shared=args.overlap_min_shared,
            min_iou=args.overlap_min_iou,
        )
        if overlap_merges == 0:
            logger.info("  no parallel duplicates detected")

    logger.info("Gap-link pass (fragment stitching):")
    tracks, gap_merges, gap_iters = link_until_convergence(
        tracks,
        max_gap=args.max_gap,
        size_tol=args.size_tol,
        dist_tol=args.dist_tol,
        use_position=args.use_position,
        edge=args.edge_frames,
    )

    dedup_removed = 0
    dedup_emptied = 0
    if not args.no_dedup:
        logger.info("Per-frame dedup pass (transient doubles):")
        tracks, dedup_removed, dedup_emptied = dedup_per_frame(
            tracks, min_iou=args.dedup_min_iou,
        )
        logger.info("  removed %d duplicate rows; dropped %d empty tracks",
                    dedup_removed, dedup_emptied)

    n_after = len(tracks)
    cls_after = class_breakdown(tracks)
    total_merges = overlap_merges + gap_merges
    logger.info("Done: %d -> %d tracks  "
                "(overlap=%d/%d iter, gap=%d/%d iter, dedup=%d rows / %d empty)",
                n_before, n_after,
                overlap_merges, overlap_iters,
                gap_merges, gap_iters,
                dedup_removed, dedup_emptied)
    logger.info("  per-class: LV %d->%d   MV %d->%d   SV %d->%d",
                cls_before.get(0, 0), cls_after.get(0, 0),
                cls_before.get(1, 0), cls_after.get(1, 0),
                cls_before.get(2, 0), cls_after.get(2, 0))

    write_tracks(tracks, out_path)
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
