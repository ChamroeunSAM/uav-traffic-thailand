"""Generate publication-ready LaTeX tables for the Paper-1 manuscript.

Reads the committed result CSVs (and the GT / calibration artifacts under
gitignored data/) and emits booktabs-style .tex files into paper/tables/. Each
table is self-contained (table environment + caption + label) and uses only
core LaTeX + booktabs -- no siunitx/array dependencies -- so it drops into an
Elsevier (TR-C) template directly.

Tables emitted when their inputs exist:
  T1  dataset / study design        <- GT files + clip metadata
  T2  detector benchmark            <- results/stage1/aggregated.csv (mAP) +
                                       results/stage1/efficiency.csv (P/R, if present)
  T4  tracker benchmark             <- results/stage3/tracker_metrics.csv
  T5  trajectory quality (3-axis)   <- results/stage4/trajectory_metrics.csv
  T5b axis-3 smoothing effect       <- results/stage4/trajectory_metrics.csv
  T6  GT-free generalisation        <- results/stage4/trajectory_metrics_gtfree.csv (if present)
  T7  calibration summary           <- data/video_annotations/calibration/*.json
  ST1 complexity / efficiency       <- results/stage1/efficiency.csv
                                       (Params/GFLOPs/FPS; run eval_efficiency.py)

Blocked (need a generation step, not emitted here):
  T3  per-class detection AP -- aggregated.csv has no per-class columns; needs a
      per-class re-eval pass (open TODO).

Run:  python scripts/make_paper_tables.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "paper/tables"
GT_DIR = PROJECT_ROOT / "data/video_annotations/ground_truth"
CALIB_DIR = PROJECT_ROOT / "data/video_annotations/calibration"
STAGE1 = PROJECT_ROOT / "results/stage1/aggregated.csv"
EFFICIENCY = PROJECT_ROOT / "results/stage1/efficiency.csv"  # P/R source for T2
STAGE3 = PROJECT_ROOT / "results/stage3/tracker_metrics.csv"
STAGE4 = PROJECT_ROOT / "results/stage4/trajectory_metrics.csv"
STAGE4_GTFREE = PROJECT_ROOT / "results/stage4/trajectory_metrics_gtfree.csv"

logger = logging.getLogger(__name__)

DETECTOR_NAME = {
    "yolov9m.pt": "YOLOv9-m", "yolov10m.pt": "YOLOv10-m", "yolo11m.pt": "YOLO11-m",
    "yolo12m.pt": "YOLO12-m", "yolo26m.pt": "YOLO26-m", "rtdetr-l.pt": "RT-DETR-L",
    "rtdetr_l.pt": "RT-DETR-L",
}
TRACKER_NAME = {"bytetrack": "ByteTrack", "botsort": "BoT-SORT",
                "ocsort": "OC-SORT", "deepocsort": "Deep OC-SORT"}
TRACKER_ORDER = ["bytetrack", "botsort", "ocsort", "deepocsort"]
CLASS_NAMES = ["LV", "MV", "SV"]
FPS = 29.97


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def f(x, dec=3, dash="--"):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return dash
    if np.isnan(x):
        return dash
    return f"{x:.{dec}f}"


def i(x, dash="--"):
    try:
        return str(int(round(float(x))))
    except (TypeError, ValueError):
        return dash


def bold(s):
    return f"\\textbf{{{s}}}"


def wrap_table(body: str, caption: str, label: str, colspec: str,
               header: str, note: str = "") -> str:
    note_block = f"\n  \\par\\smallskip\n  \\footnotesize {note}\n" if note else "\n"
    return (
        "\\begin{table}[ht]\n  \\centering\n"
        f"  \\caption{{{caption}}}\n  \\label{{{label}}}\n"
        f"  \\begin{{tabular}}{{{colspec}}}\n    \\toprule\n"
        f"{header}\n    \\midrule\n{body}\n    \\bottomrule\n"
        "  \\end{tabular}"
        f"{note_block}"
        "\\end{table}\n"
    )


def write(name: str, content: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(content, encoding="utf-8")
    logger.info("wrote paper/tables/%s", name)


# --------------------------------------------------------------------------- #
# T1 -- dataset / study design (computed from GT files)
# --------------------------------------------------------------------------- #
SCENE = {  # author-confirmable scene labels
    "100_m_DJI_0001_1min": "Intersection",
    "100_m_DJI_0054_1min": "Intersection (dense)",
    "120_m_DJI_0006_1min": "Roundabout",
    "120_m_DJI_0047_1min": "Suburban intersection",
    "120_m_DJI_0001_5min": "Urban intersection",
    "120_m_DJI_0041_5min": "Arterial intersection",
}


def gt_stats(path: Path) -> dict:
    frames, tids, classes = [], set(), []
    nrows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        a = line.split(",")
        frames.append(int(a[0])); tids.add(int(a[1]))
        classes.append(int(float(a[7])) if len(a) > 7 else -1)
        nrows += 1
    frames = np.asarray(frames)
    classes = np.asarray(classes)
    nfr = int(frames.max()) if frames.size else 0
    mix = {c: int((classes == ci).sum()) for ci, c in enumerate(CLASS_NAMES)}
    tot = max(sum(mix.values()), 1)
    return {
        "n_tracks": len(tids), "n_frames": nfr, "n_rows": nrows,
        "density": nrows / nfr if nfr else 0.0,
        "mix": {c: 100.0 * v / tot for c, v in mix.items()},
    }


def table_dataset():
    gt_clips = sorted(GT_DIR.glob("*.gt.txt")) if GT_DIR.exists() else []
    if not gt_clips:
        logger.warning("T1: no GT files -- skipping")
        return
    rows = []
    for p in gt_clips:
        clip = p.name[:-len(".gt.txt")]
        alt = "100" if clip.startswith("100") else "120"
        s = gt_stats(p)
        rows.append(
            f"    {SCENE.get(clip, '--')} & {alt} & "
            f"{s['n_frames']/FPS:.0f} & {s['n_frames']} & {s['n_tracks']} & "
            f"{s['density']:.1f} & "
            f"{s['mix']['LV']:.0f}/{s['mix']['MV']:.0f}/{s['mix']['SV']:.0f} & "
            f"Tracker GT \\\\")
    # GT-free clips (no GT; metadata only)
    for clip in ("120_m_DJI_0001_5min", "120_m_DJI_0041_5min"):
        if (CALIB_DIR / f"{clip}.json").exists():
            rows.append(
                f"    {SCENE.get(clip, '--')} & 120 & 300 & 9021 & -- & -- & -- "
                f"& GT-free \\\\")
    header = ("    Scene & Alt.\\ (m) & Dur.\\ (s) & Frames & "
              "Tracks & Veh./frame & LV/MV/SV (\\%) & Role \\\\")
    body = "\n".join(rows)
    note = ("Vehicle classes: LV = large (truck/bus), MV = medium (car/van/pickup), "
            "SV = small (motorcycle). Density = mean annotated vehicles per frame "
            "(ground truth). GT-free clips carry no ground truth and are scored by "
            "intrinsic plausibility only.")
    write("T1_dataset.tex", wrap_table(
        body, "UAV traffic clips used in the study.", "tab:dataset",
        "llrrrrcl", header, note))


# --------------------------------------------------------------------------- #
# T2 -- detector benchmark
# --------------------------------------------------------------------------- #
def table_detectors():
    if not STAGE1.exists():
        logger.warning("T2: %s missing -- skipping", STAGE1)
        return
    rows = list(csv.DictReader(STAGE1.open(encoding="utf-8")))
    rows.sort(key=lambda r: float(r["overall_mAP50_95_mean"]), reverse=True)
    best = rows[0]["model_name"]

    # P/R are not produced by the supervision mAP evaluator -> pull them from the
    # native-validator pass (same stratified test set, 5-seed mean/std), keyed by
    # detector base name (e.g. "rtdetr-l.pt" -> "rtdetr_l").
    pr = {}
    if EFFICIENCY.exists():
        pr = {r["detector"]: r for r in csv.DictReader(EFFICIENCY.open(encoding="utf-8"))}

    def base_of(model_name):
        return model_name.replace(".pt", "").replace("-", "_")

    def cell(mean, std):
        return f"{f(mean)}\\,({f(std, 3)})"

    def pr_cell(model_name, kind):
        e = pr.get(base_of(model_name))
        if not e:
            return "--"
        return cell(e[f"{kind}_mean"], e[f"{kind}_std"])

    body_lines = []
    for r in rows:
        name = DETECTOR_NAME.get(r["model_name"], r["model_name"])
        disp = bold(name) if r["model_name"] == best else name
        body_lines.append(
            f"    {disp} & "
            f"{pr_cell(r['model_name'], 'precision')} & "
            f"{pr_cell(r['model_name'], 'recall')} & "
            f"{cell(r['overall_mAP50_mean'], r['overall_mAP50_std'])} & "
            f"{cell(r['overall_mAP50_95_mean'], r['overall_mAP50_95_std'])} & "
            f"{cell(r['100m_mAP50_mean'], r['100m_mAP50_std'])} & "
            f"{cell(r['120m_mAP50_mean'], r['120m_mAP50_std'])} \\\\")
    header = ("    Detector & P & R & mAP@.5 & mAP@.5:.95 & mAP@.5 (100\\,m) "
              "& mAP@.5 (120\\,m) \\\\")
    note = (f"Mean (std) over {rows[0]['n_seeds']} seeds on the stratified test "
            "set; top detector (by mAP@.5:.95) in bold. mAP is the canonical "
            "supervision metric; precision and recall are from Ultralytics' native "
            "validator at its operating point (the supervision evaluator does not "
            "expose P/R) and are reported for completeness. 100\\,m exceeds "
            "120\\,m throughout. All detectors trained at 1280$\\times$1280.")
    write("T2_detectors.tex", wrap_table(
        "\n".join(body_lines),
        "Detector benchmark: six detectors $\\times$ five seeds.",
        "tab:detectors", "lcccccc", header, note))


# --------------------------------------------------------------------------- #
# ST1 -- model complexity / efficiency (Params, GFLOPs, FPS)
# --------------------------------------------------------------------------- #
def table_efficiency():
    if not EFFICIENCY.exists():
        logger.info("ST1: %s not present -- run scripts/eval_efficiency.py first",
                    EFFICIENCY.name)
        return
    eff = list(csv.DictReader(EFFICIENCY.open(encoding="utf-8")))

    # order rows to match T2 (by supervision mAP@.5:.95, descending)
    rank = {}
    if STAGE1.exists():
        for r in csv.DictReader(STAGE1.open(encoding="utf-8")):
            base = r["model_name"].replace(".pt", "").replace("-", "_")
            rank[base] = float(r["overall_mAP50_95_mean"])
    eff.sort(key=lambda r: rank.get(r["detector"], 0.0), reverse=True)

    # best efficiency per column: fewest params, fewest GFLOPs, highest FPS
    best_params = min(eff, key=lambda r: float(r["params_M"]))
    best_gflops = min(eff, key=lambda r: float(r["gflops"]))
    best_fps = max(eff, key=lambda r: float(r["fps"]))

    body = []
    for r in eff:
        name = DETECTOR_NAME.get(r["detector"] + ".pt", r["detector"])
        p = f(r["params_M"], 1); g = f(r["gflops"], 1); fp = f(r["fps"], 1)
        if r is best_params:
            p = bold(p)
        if r is best_gflops:
            g = bold(g)
        if r is best_fps:
            fp = bold(fp)
        body.append(f"    {name} & {p} & {g} & {fp} \\\\")
    header = "    Detector & Param\\,(M) & GFLOPs & FPS \\\\"
    note = ("Model complexity and throughput. Parameter count and GFLOPs are "
            "architecture constants computed at the 1280$\\times$1280 evaluation "
            "resolution; FPS is single-image fp32 inference throughput on one "
            "RTX\\,5090. Best per column in bold (fewest Param/GFLOPs, highest "
            "FPS). Detectors ordered as in Table~\\ref{tab:detectors}. RT-DETR-L's "
            "accuracy lead (Table~\\ref{tab:detectors}) comes at $\\sim$2$\\times$ "
            "the parameters and $\\sim$3$\\times$ the latency of the leading YOLOs.")
    write("ST1_efficiency.tex", wrap_table(
        "\n".join(body),
        "Detector complexity and inference efficiency.",
        "tab:efficiency", "lccc", header, note))


# --------------------------------------------------------------------------- #
# T4 -- tracker benchmark (raw vs linked)
# --------------------------------------------------------------------------- #
def table_trackers():
    if not STAGE3.exists():
        logger.warning("T4: %s missing -- skipping", STAGE3)
        return
    allrows = [r for r in csv.DictReader(STAGE3.open(encoding="utf-8"))
               if r["clip"] == "ALL" and r.get("scope", "overall") == "overall"]
    by = {(r["tracker"], r["arm"]): r for r in allrows}

    linked = {t: by.get((t, "linked")) for t in TRACKER_ORDER}
    best_hota = max((float(v["HOTA"]) for v in linked.values() if v), default=None)
    best_idf1 = max((float(v["IDF1"]) for v in linked.values() if v), default=None)
    best_idsw = min((float(v["IDSW"]) for v in linked.values() if v), default=None)

    body_lines = []
    for t in TRACKER_ORDER:
        for arm in ("raw", "linked"):
            r = by.get((t, arm))
            if not r:
                continue
            name = TRACKER_NAME[t] if arm == "raw" else ""
            hota, idf1, idsw = f(r["HOTA"]), f(r["IDF1"]), i(r["IDSW"])
            if arm == "linked":
                if best_hota is not None and abs(float(r["HOTA"]) - best_hota) < 1e-9:
                    hota = bold(hota)
                if best_idf1 is not None and abs(float(r["IDF1"]) - best_idf1) < 1e-9:
                    idf1 = bold(idf1)
                if best_idsw is not None and abs(float(r["IDSW"]) - best_idsw) < 1e-9:
                    idsw = bold(idsw)
            body_lines.append(
                f"    {name} & {arm} & {f(r['HOTA'])} & {f(r['DetA'])} & "
                f"{f(r['AssA'])} & {f(r['MOTA'])} & {idf1 if arm=='linked' else f(r['IDF1'])} & "
                f"{idsw} & {i(r['Frag'])} \\\\".replace(
                    f"& {f(r['HOTA'])} &", f"& {hota} &", 1))
        body_lines.append("    \\addlinespace")
    if body_lines and body_lines[-1].strip() == "\\addlinespace":
        body_lines.pop()
    header = ("    Tracker & Arm & HOTA & DetA & AssA & MOTA & IDF1 & IDSW & Frag \\\\")
    note = ("Clip-averaged over the four annotated sequences, class-agnostic. "
            "``raw'' = tracker output; ``linked'' = after the offline fragment "
            "linker. Best linked HOTA/IDF1/IDSW in bold. Raw DetA $\\approx$ "
            "0.82--0.84 across trackers (shared RT-DETR detections), so the raw "
            "HOTA spread is pure association; the linker's dedup pass lifts DetA.")
    write("T4_trackers.tex", wrap_table(
        "\n".join(body_lines),
        "Tracker benchmark (HOTA/MOTA/IDF1), raw vs.\\ post-linker.",
        "tab:trackers", "llccccccc", header, note))


# --------------------------------------------------------------------------- #
# T5 / T5b -- trajectory quality
# --------------------------------------------------------------------------- #
def _stage4_all(variant):
    rows = [r for r in csv.DictReader(STAGE4.open(encoding="utf-8"))
            if r["clip"] == "ALL" and r["variant"] == variant]
    return {r["source"]: r for r in rows}


def table_trajectory():
    if not STAGE4.exists():
        logger.warning("T5: %s missing -- skipping", STAGE4)
        return
    sm = _stage4_all("smoothed")
    order = ["gt"] + TRACKER_ORDER
    label = {"gt": "GT (ceiling)", **TRACKER_NAME}

    # best among trackers (not GT) per column; lower-is-better for frag/idsw/
    # ade/fde/speed/heading/jerk/curv, higher for completeness/accel_ok
    trk = {t: sm[t] for t in TRACKER_ORDER if t in sm}

    def best(col, lower=True):
        vals = [(t, float(r[col])) for t, r in trk.items()
                if r.get(col) not in (None, "") and not np.isnan(float(r[col]))]
        if not vals:
            return None
        return (min if lower else max)(vals, key=lambda kv: kv[1])[0]

    bcols = {
        "a1_completeness_mean": best("a1_completeness_mean", lower=False),
        "a1_frag_per_gt_mean": best("a1_frag_per_gt_mean"),
        "a1_id_switches": best("a1_id_switches"),
        "a2_ade_m": best("a2_ade_m"), "a2_fde_m": best("a2_fde_m"),
        "a2_speed_mae_mps": best("a2_speed_mae_mps"),
        "a2_heading_mae_deg": best("a2_heading_mae_deg"),
        "a3_frac_accel_ok": best("a3_frac_accel_ok", lower=False),
        "a3_jerk_mean_mps3": best("a3_jerk_mean_mps3"),
    }

    def cell(r, col, dec=3, isint=False):
        if r.get(col) in (None, ""):
            return "--"
        val = i(r[col]) if isint else f(r[col], dec)
        if r["source"] == bcols.get(col):
            val = bold(val)
        return val

    body = []
    for src in order:
        if src not in sm:
            continue
        r = sm[src]
        body.append(
            f"    {label[src]} & "
            f"{cell(r,'a1_completeness_mean')} & {cell(r,'a1_frag_per_gt_mean',2)} & "
            f"{cell(r,'a1_id_switches',isint=True)} & "
            f"{cell(r,'a2_ade_m')} & {cell(r,'a2_fde_m')} & "
            f"{cell(r,'a2_speed_mae_mps')} & {cell(r,'a2_heading_mae_deg',2)} & "
            f"{cell(r,'a3_frac_accel_ok')} & {cell(r,'a3_jerk_mean_mps3',1)} \\\\")
    header = ("    & \\multicolumn{3}{c}{Axis 1: identity} & "
              "\\multicolumn{4}{c}{Axis 2: accuracy vs.\\ GT} & "
              "\\multicolumn{2}{c}{Axis 3: plausibility} \\\\\n"
              "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-10}\n"
              "    Source & Compl. & Frag/GT & IDSW & ADE\\,(m) & FDE\\,(m) & "
              "Sp.\\,MAE & Head\\,($^\\circ$) & Acc.\\,ok & Jerk \\\\")
    note = ("Smoothed (Savitzky--Golay) trajectories, clip-averaged, class-"
            "agnostic. GT run through the same pipeline as a reference ceiling "
            "(Axis 1/2 undefined for GT). Best tracker per column in bold. "
            "Speed MAE in m/s; Acc.\\,ok = fraction of $|a|\\le 8$\\,m/s$^2$; "
            "Jerk = mean $|j|$ (m/s$^3$). ADE is near-zero because GT shares the "
            "RT-DETR detections; the discriminating signal is in speed/heading.")
    write("T5_trajectory_quality.tex", wrap_table(
        "\n".join(body),
        "Trajectory-quality evaluation along three axes.",
        "tab:trajectory", "lccccccccc", header, note))

    # T5b -- Axis-3 raw vs smoothed (smoothing effect)
    rw = _stage4_all("raw")
    body2 = []
    for src in order:
        if src not in sm:
            continue
        s, r = sm[src], rw.get(src, {})
        body2.append(
            f"    {label[src]} & {f(r.get('a3_frac_accel_ok'))} & "
            f"{f(s['a3_frac_accel_ok'])} & {f(r.get('a3_jerk_mean_mps3'),1)} & "
            f"{f(s['a3_jerk_mean_mps3'],1)} \\\\")
    header2 = ("    & \\multicolumn{2}{c}{Frac.\\ $|a|\\le8$\\,m/s$^2$} & "
               "\\multicolumn{2}{c}{Mean $|$jerk$|$ (m/s$^3$)} \\\\\n"
               "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
               "    Source & raw & smoothed & raw & smoothed \\\\")
    note2 = ("Effect of Savitzky--Golay smoothing on the Axis-3 plausibility "
             "metrics (smoothing reported as a studied variable). Smoothing lifts "
             "every source toward the GT ceiling; the tracker ordering is "
             "preserved in both columns.")
    write("T5b_smoothing_effect.tex", wrap_table(
        "\n".join(body2),
        "Axis-3 plausibility: raw vs.\\ smoothed.",
        "tab:smoothing", "lcccc", header2, note2))


# --------------------------------------------------------------------------- #
# T6 -- GT-free generalisation (only if the CSV exists)
# --------------------------------------------------------------------------- #
def table_gtfree():
    if not STAGE4_GTFREE.exists():
        logger.info("T6: %s not present yet -- skipping (run --gt-free-clips first)",
                    STAGE4_GTFREE.name)
        return
    allrows = list(csv.DictReader(STAGE4_GTFREE.open(encoding="utf-8")))
    sm = {(r["clip"], r["source"]): r for r in allrows if r["variant"] == "smoothed"}
    rw = {(r["clip"], r["source"]): r for r in allrows if r["variant"] == "raw"}
    clips = sorted({r["clip"] for r in allrows if r["clip"] != "ALL"})

    body = []
    for clip in clips + ["ALL"]:
        clabel = clip.replace("_", "\\_")
        body.append(f"    \\multicolumn{{7}}{{l}}{{\\textit{{{clabel}}}}} \\\\")
        for t in TRACKER_ORDER:
            r = sm.get((clip, t))
            if not r:
                continue
            # accel-in-bounds from the RAW series (smoothing washes out the
            # difference; raw is the discriminating signal). Jerk smoothed.
            accel_raw = rw.get((clip, t), {}).get("a3_frac_accel_ok")
            body.append(
                f"    \\quad {TRACKER_NAME[t]} & {i(r['sc_n_tracks'])} & "
                f"{f(r['sc_seg_per_track'],2)} & {i(r['sc_teleport_breaks'])} & "
                f"{f(r['sc_mean_track_dur_s'],1)} & {f(accel_raw)} & "
                f"{f(r['a3_jerk_mean_mps3'],1)} \\\\")
        body.append("    \\addlinespace")
    if body and body[-1].strip() == "\\addlinespace":
        body.pop()
    header = ("    Tracker & Tracks & Seg/trk & Teleports & Mean dur.\\,(s) "
              "& Acc.\\,ok (raw) & Jerk \\\\")
    note = ("Two unlabelled 120\\,m clips scored without ground truth: "
            "self-consistency proxies (segments per track, impossible-jump "
            "splits, mean track duration) and Axis-3 plausibility. Acc.\\,ok = "
            "fraction $|a|\\le8$\\,m/s$^2$ on the \\emph{raw} series (smoothing "
            "equalises it); Jerk = mean $|j|$ smoothed (m/s$^3$). Lower Seg/trk, "
            "Teleports, Jerk and higher Mean dur., Acc.\\,ok are better. "
            "ByteTrack/BoT-SORT lead on every axis, replicating the labelled "
            "benchmark with no ground truth.")
    write("T6_gtfree.tex", wrap_table(
        "\n".join(body),
        "GT-free generalisation on two new 120\\,m clips.",
        "tab:gtfree", "lcccccc", header, note))


# --------------------------------------------------------------------------- #
# T7 -- calibration summary
# --------------------------------------------------------------------------- #
def _scale_m_per_px(H: np.ndarray) -> float:
    """Median local metres-per-pixel over a grid (Jacobian of the homography)."""
    us = np.linspace(800, 3000, 5)
    vs = np.linspace(600, 1800, 5)
    scales = []
    for u in us:
        for v in vs:
            p = np.array([u, v, 1.0])
            w = H @ p
            wx = (H @ np.array([u + 1, v, 1.0]))
            wy = (H @ np.array([u, v + 1, 1.0]))
            o = w[:2] / w[2]
            sx = np.linalg.norm(wx[:2] / wx[2] - o)
            sy = np.linalg.norm(wy[:2] / wy[2] - o)
            scales.append(0.5 * (sx + sy))
    return float(np.median(scales))


def table_calibration():
    if not CALIB_DIR.exists():
        logger.warning("T7: %s missing -- skipping", CALIB_DIR)
        return
    jsons = sorted(CALIB_DIR.glob("*.json"))
    if not jsons:
        logger.warning("T7: no calibration JSONs -- skipping")
        return
    body = []
    for jp in jsons:
        d = json.loads(jp.read_text(encoding="utf-8"))
        H = np.asarray(d["H"], dtype=float).reshape(3, 3)
        alt = "100" if d["clip"].startswith("100") else "120"
        scale = _scale_m_per_px(H)
        body.append(
            f"    {d['clip'].replace('_', '\\_')} & {alt} & {d['n_gcp']} & "
            f"{d['rmse_m']:.3f} & {scale:.4f} \\\\")
    header = "    Clip & Alt.\\,(m) & GCPs & Reproj.\\ RMSE\\,(m) & Scale\\,(m/px) \\\\"
    note = ("Per-clip ground-plane homography from road-marking GCPs. With four "
            "points the fit is exact (RMSE $\\approx$ 0); calibration quality is "
            "instead guarded by the scale-consistency spread and the agreement of "
            "Scale with the expected altitude GSD ($\\sim$0.03\\,m/px at 100\\,m, "
            "$\\sim$0.04\\,m/px at 120\\,m).")
    write("T7_calibration.tex", wrap_table(
        "\n".join(body), "Pixel-to-world calibration summary.",
        "tab:calibration", "llccc", header, note))


# --------------------------------------------------------------------------- #
# T3 -- per-class detection difficulty (from per-seed eval_results.json)
# --------------------------------------------------------------------------- #
RUNS_DIR = PROJECT_ROOT / "runs/stage1_detectors"
DET_BASE_NAME = {
    "yolov9m": "YOLOv9-m", "yolov10m": "YOLOv10-m", "yolo11m": "YOLO11-m",
    "yolo12m": "YOLO12-m", "yolo26m": "YOLO26-m", "rtdetr_l": "RT-DETR-L",
}
DET_ORDER = ["rtdetr_l", "yolo26m", "yolov10m", "yolo11m", "yolo12m", "yolov9m"]


def table_per_class():
    jsons = sorted(RUNS_DIR.glob("*/eval_results.json"))
    if not jsons:
        logger.warning("T3: no eval_results.json under %s -- skipping", RUNS_DIR)
        return
    # (detector_base, class) -> {"AP50": [...], "AP50_95": [...]}
    agg: dict[tuple, dict[str, list]] = {}
    for jp in jsons:
        run = jp.parent.name
        base = run.rsplit("_seed", 1)[0]
        d = json.loads(jp.read_text(encoding="utf-8"))
        pc = d.get("metrics", {}).get("per_class", {})
        for cls, splits in pc.items():
            ov = splits.get("overall", {})
            a = agg.setdefault((base, cls), {"AP50": [], "AP50_95": []})
            for k in ("AP50", "AP50_95"):
                v = ov.get(k)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    a[k].append(float(v))

    def cell(base, cls, key):
        vals = agg.get((base, cls), {}).get(key, [])
        if not vals:
            return "--", None
        return f"{np.mean(vals):.3f}", float(np.mean(vals))

    # best detector per (class, metric) for bolding
    best = {}
    for cls in CLASS_NAMES:
        for key in ("AP50", "AP50_95"):
            ranked = [(b, np.mean(agg[(b, cls)][key]))
                      for b in DET_ORDER if (b, cls) in agg and agg[(b, cls)][key]]
            if ranked:
                best[(cls, key)] = max(ranked, key=lambda kv: kv[1])[0]

    body = []
    for base in DET_ORDER:
        if not any((base, c) in agg for c in CLASS_NAMES):
            continue
        cells = []
        for cls in CLASS_NAMES:
            for key in ("AP50", "AP50_95"):
                txt, _ = cell(base, cls, key)
                if best.get((cls, key)) == base and txt != "--":
                    txt = bold(txt)
                cells.append(txt)
        body.append(f"    {DET_BASE_NAME.get(base, base)} & " + " & ".join(cells) + " \\\\")
    header = ("    & \\multicolumn{2}{c}{LV (large)} & "
              "\\multicolumn{2}{c}{MV (medium)} & \\multicolumn{2}{c}{SV (small)} \\\\\n"
              "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
              "    Detector & AP@.5 & AP@.5:.95 & AP@.5 & AP@.5:.95 "
              "& AP@.5 & AP@.5:.95 \\\\")
    note = ("Mean over 5 seeds, overall test set; best per column in bold. "
            "Classes: LV = large (truck/bus), MV = medium (car/van/pickup), "
            "SV = small (motorcycle). SV is hardest at the strict AP@.5:.95 "
            "(small objects, tight-localisation penalty), despite high AP@.5. "
            "Per-seed std is small ($\\leq$0.01) and omitted for width.")
    write("T3_per_class.tex", wrap_table(
        "\n".join(body), "Per-class detection accuracy (AP) by detector.",
        "tab:perclass", "lcccccc", header, note))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    table_dataset()
    table_detectors()
    table_per_class()
    table_efficiency()
    table_trackers()
    table_trajectory()
    table_gtfree()
    table_calibration()
    logger.info("done -> %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
