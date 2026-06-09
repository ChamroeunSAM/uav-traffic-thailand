"""Ablation: detector domain gradient (COCO -> VisDrone -> target) -> trajectory.

Motivation (for the SSM / Paper-2 reader). Stage 1-4 answer "which detector /
tracker is best *given* fine-tuning". They never quantify the cost of NOT
fine-tuning -- i.e. grabbing an off-the-shelf detector. This ablation runs the
SAME end-to-end pipeline (detect -> ByteTrack -> link -> world trajectory ->
3-axis quality) across a DOMAIN GRADIENT of detectors and reports how the
detector domain-gap PROPAGATES into the trajectory layer that SSM consumes:

  COCO-pretrained (general)   rtdetr-l.pt, yolo26m.pt        -- wrong modality
  VisDrone-pretrained (aerial) visdrone_yolov8x.pt           -- right modality,
                               (off-the-shelf, larger backbone) wrong geography
                                                               / altitude / taxonomy
  fine-tuned (target)         Stage-1 best.pt                -- this dataset

The VisDrone point is the intermediate control that answers the obvious
reviewer objection -- "wouldn't any aerial-pretrained model already suffice?".
It does not: VisDrone lands between COCO and fine-tuned on every axis, yet still
trails the fine-tuned *medium* models on completeness despite its larger
architecture -> it is target-domain fine-tuning, not merely aerial pretraining,
that recovers trajectories.

Matrix: 5 cells (see CELLS) x 4 GT clips. ByteTrack only (motion-only -> the
detector is the only variable; no ReID confound).

Honest-metric note. The Stage 2 GT was semi-automatically SEEDED from the
fine-tuned RT-DETR, so Axis-2 positional ADE-vs-GT is biased in the fine-tuned
detector's favour (ADE ~ 3 mm by construction) and is NOT a fair headline here.
We therefore lead with provenance-free metrics:
  * per-class LOCALIZATION recall  (a GT box is "found" if any pred box overlaps
    it at IoU >= --iou-recall; class-agnostic, so a fair test of "did the
    detector see this vehicle at all" -- the SSM-relevant question);
  * Axis-1 trajectory completeness + frags/GT  (missed/weak dets -> incomplete,
    fragmented tracks -> fewer usable trajectories for SSM);
  * Axis-3 GT-free plausibility (jerk)  (off-domain boxes are noisier).
Axis-2 is still written to the per-cell CSVs but flagged, not headlined.

Class remap to our {LV, MV, SV} taxonomy (by NAME, robust to id drift; see
NAME_TO_OURS). COCO: car->MV, motorcycle/bicycle->SV, bus/truck->LV, rest
dropped. VisDrone: car/van->MV, truck/bus->LV, motor/bicycle/tricycle/
awning-tricycle->SV, pedestrian/people dropped. Fine-tuned models emit
{LV,MV,SV} natively (no remap). Pickups/vans/SUVs are MV in our GT but COCO
often calls them "truck"; that mislabel hurts class-correct recall (reported)
but not localization recall (headline) -- exactly the domain-specific boundary
fine-tuning fixes.

Outputs (under results/stage4/ablation_pretrained/ ; isolated -- canonical
Stage 3/4 artifacts are never touched):
    recall_by_class.csv          per (detector, weights, clip) class recall
    trajectory_<cell>.csv        04_eval_trajectories output per cell
Detection caches + tracks + trajectories live under
data/video_annotations/ablation/<cell>/ (gitignored with the rest of data/).

Resumable: any step whose output already exists is skipped unless --force.

Example:
    python scripts/ablation_finetune_vs_pretrained.py            # full matrix
    python scripts/ablation_finetune_vs_pretrained.py --clips 120_m_DJI_0006_1min
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

VISDRONE_WEIGHTS = Path(
    r"C:\Users\User\Desktop\UAV data collection Khonkaen\models\visdrone_yolov8x.pt")

# --- cells: each a detector checkpoint run through the whole pipeline -------- #
# regime = COCO-general / VisDrone-aerial(off-the-shelf) / target-fine-tuned.
# taxonomy selects how the model's native class names are remapped to LV/MV/SV.
# cell key (== output dir name) is kept stable so earlier runs stay cached.
CELLS = [
    {"key": "rtdetr_l_pretrained", "detector": "rtdetr_l", "regime": "pretrained",
     "weights": PROJECT_ROOT / "rtdetr-l.pt", "taxonomy": "coco"},
    {"key": "rtdetr_l_finetuned", "detector": "rtdetr_l", "regime": "finetuned",
     "weights": PROJECT_ROOT / "runs/stage1_detectors/rtdetr_l_seed0/weights/best.pt",
     "taxonomy": "native"},
    {"key": "yolo26m_pretrained", "detector": "yolo26m", "regime": "pretrained",
     "weights": PROJECT_ROOT / "yolo26m.pt", "taxonomy": "coco"},
    {"key": "yolo26m_finetuned", "detector": "yolo26m", "regime": "finetuned",
     "weights": PROJECT_ROOT / "runs/stage1_detectors/yolo26m_seed0/weights/best.pt",
     "taxonomy": "native"},
    # off-the-shelf AERIAL model (different/larger architecture, VisDrone-trained):
    # the intermediate domain point between COCO and target fine-tuning.
    {"key": "visdrone8x_aerial", "detector": "visdrone8x", "regime": "visdrone",
     "weights": VISDRONE_WEIGHTS, "taxonomy": "visdrone"},
]

CLIP_DIR = PROJECT_ROOT / "data/video_clips"
GT_DIR = PROJECT_ROOT / "data/video_annotations/ground_truth"
ABLATION_DIR = PROJECT_ROOT / "data/video_annotations/ablation"
RESULTS_DIR = PROJECT_ROOT / "results/stage4/ablation_pretrained"

CLASS_NAMES = {0: "LV", 1: "MV", 2: "SV"}
# class-name -> our class id, per source taxonomy; resolved to a per-model id map
# at runtime from model.names (robust to id ordering). taxonomy "native" => the
# model already emits {0:LV,1:MV,2:SV} and no remap is applied.
NAME_TO_OURS = {
    "coco": {"car": 1, "motorcycle": 2, "bicycle": 2, "bus": 0, "truck": 0},
    "visdrone": {"car": 1, "van": 1, "truck": 0, "bus": 0,
                 "motor": 2, "bicycle": 2, "tricycle": 2, "awning-tricycle": 2},
}

RUN_TRACKER = PROJECT_ROOT / "scripts/03_run_tracker.py"
LINK_TRACKS = PROJECT_ROOT / "scripts/02b_link_tracks.py"
EVAL_TRAJ = PROJECT_ROOT / "scripts/04_eval_trajectories.py"


# --------------------------------------------------------------------------- #
# detection (+ COCO remap) and per-class recall
# --------------------------------------------------------------------------- #
def load_gt_by_frame(gt_txt: Path) -> dict[int, list[tuple[float, float, float, float, int]]]:
    """frame -> list of (x1,y1,x2,y2,cls) GT boxes (MOT ltwh -> xyxy)."""
    by_frame: dict[int, list] = {}
    for line in gt_txt.read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        fr = int(p[0]); l, t, w, h = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
        cls = int(float(p[7]))
        by_frame.setdefault(fr, []).append((l, t, l + w, t + h, cls))
    return by_frame


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4 xyxy) and b (M,4 xyxy) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def greedy_recall(gt: np.ndarray, gt_cls: np.ndarray, pred: np.ndarray,
                  pred_cls: np.ndarray, iou_thr: float,
                  loc_hit: dict, cls_hit: dict, total: dict) -> None:
    """Accumulate per-class localization + class-correct recall for one frame.

    Greedy one-to-one GT<-pred matching by descending IoU. ``loc_hit`` counts a
    GT box matched by ANY pred class; ``cls_hit`` requires the pred class to
    equal the GT class. ``total`` counts GT boxes per class.
    """
    for c in gt_cls:
        total[int(c)] = total.get(int(c), 0) + 1
    if len(pred) == 0 or len(gt) == 0:
        return
    iou = iou_matrix(gt, pred)
    pairs = [(iou[i, j], i, j) for i in range(len(gt)) for j in range(len(pred))
             if iou[i, j] >= iou_thr]
    pairs.sort(reverse=True)
    used_gt, used_pred = set(), set()
    for v, i, j in pairs:
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i); used_pred.add(j)
        gc = int(gt_cls[i])
        loc_hit[gc] = loc_hit.get(gc, 0) + 1
        if int(pred_cls[j]) == gc:
            cls_hit[gc] = cls_hit.get(gc, 0) + 1


def detect_and_recall(weights: Path, video: Path, gt_by_frame: dict, taxonomy: str,
                      conf: float, iou: float, imgsz: int, device: str,
                      iou_recall: float) -> tuple[np.ndarray, dict]:
    """Detect over the clip; return (cache (M,7), recall-stats dict).

    ``taxonomy`` "native" -> model already emits {0:LV,1:MV,2:SV}, no remap.
    Otherwise the named map in ``NAME_TO_OURS`` folds the model's vehicle classes
    into {LV,MV,SV} (by class NAME, robust to id order) and drops the rest.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    name_to_ours = None
    if taxonomy != "native":
        name_map = NAME_TO_OURS[taxonomy]
        name_to_ours = {cid: name_map[nm] for cid, nm in model.names.items()
                        if nm in name_map}

    rows: list[tuple] = []
    loc_hit, cls_hit, total = {}, {}, {}
    idx = 0
    for idx, res in enumerate(
        model.predict(source=str(video), conf=conf, iou=iou, imgsz=imgsz,
                      device=device, stream=True, verbose=False), start=1):
        b = res.boxes
        if b is None or len(b) == 0:
            xyxy = np.zeros((0, 4), np.float32); cf = np.zeros(0, np.float32)
            cl = np.zeros(0, np.float32)
        else:
            xyxy = b.xyxy.cpu().numpy()
            cf = b.conf.cpu().numpy()
            cl = b.cls.cpu().numpy().astype(int)
        if name_to_ours is not None:
            keep = np.array([c in name_to_ours for c in cl], dtype=bool)
            xyxy, cf = xyxy[keep], cf[keep]
            cl = np.array([name_to_ours[c] for c in cl[keep]], dtype=int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, cf, cl):
            rows.append((idx, x1, y1, x2, y2, c, int(k)))
        # per-class recall vs GT for this frame
        gt_boxes = gt_by_frame.get(idx, [])
        if gt_boxes:
            g = np.array([b[:4] for b in gt_boxes], np.float32)
            gc = np.array([b[4] for b in gt_boxes], int)
            greedy_recall(g, gc, xyxy.astype(np.float32), cl, iou_recall,
                          loc_hit, cls_hit, total)
        if idx % 300 == 0:
            logger.info("    frame %d  (cum dets=%d)", idx, len(rows))
    cache = np.array(rows, dtype=np.float32).reshape(-1, 7)
    stats = {"loc_hit": loc_hit, "cls_hit": cls_hit, "total": total,
             "n_frames": idx, "n_dets": len(cache)}
    return cache, stats


# --------------------------------------------------------------------------- #
# pipeline drivers (reuse existing Stage 3/4 scripts unchanged)
# --------------------------------------------------------------------------- #
def run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"command failed (rc={r.returncode}): {' '.join(map(str, cmd))}")


def track_and_link(cell_dir: Path, clip: str, cache: Path, force: bool) -> Path:
    raw = cell_dir / f"{clip}.bytetrack.txt"
    linked = cell_dir / f"{clip}.bytetrack.linked.txt"
    if force or not raw.exists():
        run([sys.executable, str(RUN_TRACKER),
             "--video", str(CLIP_DIR / f"{clip}.mp4"), "--tracker", "bytetrack",
             "--detections", str(cache), "--output-dir", str(cell_dir)])
    if force or not linked.exists():
        run([sys.executable, str(LINK_TRACKS), "--input", str(raw),
             "--output", str(linked)])
    return linked


# --------------------------------------------------------------------------- #
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", nargs="+", default=[c["key"] for c in CELLS],
                   choices=[c["key"] for c in CELLS],
                   help="Which detector cells to run (default: all).")
    p.add_argument("--clips", nargs="+", default=None,
                   help="Clip stems (default: all 4 GT clips).")
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--iou-recall", type=float, default=0.5,
                   help="IoU threshold for the per-class recall match.")
    p.add_argument("--force", action="store_true", help="Recompute existing outputs.")
    p.add_argument("--skip-eval", action="store_true",
                   help="Stop after track+link (don't run 04_eval_trajectories).")
    args = p.parse_args()

    clips = args.clips or sorted(f.name[:-len(".gt.txt")]
                                 for f in GT_DIR.glob("*.gt.txt"))
    logger.info("clips: %s", ", ".join(clips))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    recall_rows: list[dict] = []

    for spec in CELLS:
        if spec["key"] not in args.cells:
            continue
        det, kind, cell = spec["detector"], spec["regime"], spec["key"]
        weights, taxonomy = spec["weights"], spec["taxonomy"]
        if not weights.exists():
            logger.error("missing weights %s -- skipping cell %s", weights, cell)
            continue
        cell_dir = ABLATION_DIR / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=== CELL %s  (weights=%s, taxonomy=%s) ===",
                    cell, weights.name, taxonomy)

        for clip in clips:
            gt_txt = GT_DIR / f"{clip}.gt.txt"
            if not gt_txt.exists():
                logger.warning("no GT for %s -- skipping", clip); continue
            cache_path = cell_dir / f"{clip}.detections.npy"
            stats_path = cell_dir / f"{clip}.recall.npz"

            if not args.force and cache_path.exists() and stats_path.exists():
                logger.info("  [%s] cache+recall exist -> skip detect", clip)
                s = np.load(stats_path, allow_pickle=True)
                stats = {k: s[k].item() for k in ("loc_hit", "cls_hit", "total")}
                stats["n_frames"] = int(s["n_frames"]); stats["n_dets"] = int(s["n_dets"])
            else:
                logger.info("  [%s] detecting ...", clip)
                gt_by_frame = load_gt_by_frame(gt_txt)
                cache, stats = detect_and_recall(
                    weights, CLIP_DIR / f"{clip}.mp4", gt_by_frame,
                    taxonomy=taxonomy, conf=args.conf, iou=args.iou,
                    imgsz=args.imgsz, device=args.device, iou_recall=args.iou_recall)
                np.save(cache_path, cache)
                np.savez(stats_path, loc_hit=stats["loc_hit"], cls_hit=stats["cls_hit"],
                         total=stats["total"], n_frames=stats["n_frames"],
                         n_dets=stats["n_dets"])

            row = {"detector": det, "weights": kind, "cell": cell, "clip": clip,
                   "n_frames": stats["n_frames"], "n_dets": stats["n_dets"]}
            for c, nm in CLASS_NAMES.items():
                tot = stats["total"].get(c, 0)
                row[f"recall_loc_{nm}"] = stats["loc_hit"].get(c, 0) / tot if tot else float("nan")
                row[f"recall_cls_{nm}"] = stats["cls_hit"].get(c, 0) / tot if tot else float("nan")
                row[f"n_gt_{nm}"] = tot
            recall_rows.append(row)
            logger.info("  [%s] recall(loc) LV=%.2f MV=%.2f SV=%.2f  dets=%d",
                        clip, row["recall_loc_LV"], row["recall_loc_MV"],
                        row["recall_loc_SV"], stats["n_dets"])

            track_and_link(cell_dir, clip, cache_path, args.force)

        # per-cell trajectory eval over all clips of this cell
        if not args.skip_eval:
            cell_csv = RESULTS_DIR / f"trajectory_{cell}.csv"
            traj_dir = cell_dir / "trajectories"
            run([sys.executable, str(EVAL_TRAJ),
                 "--tracker-dir", str(cell_dir), "--traj-dir", str(traj_dir),
                 "--trackers", "bytetrack", "--out", str(cell_csv)])

    # write the recall table
    recall_csv = RESULTS_DIR / "recall_by_class.csv"
    if recall_rows:
        cols = list(recall_rows[0].keys())
        with recall_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(recall_rows)
        logger.info("wrote %s (%d rows)", recall_csv, len(recall_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
