"""Generate paper figures from the Stage-4 trajectory artifacts.

One function per figure; select with a positional arg. Saves both a PNG (for
on-screen inspection) and a PDF (for the manuscript) into paper/figures/.

Usage:
    python scripts/make_paper_figures.py accel       # F6a accel plausibility
    python scripts/make_paper_figures.py smoothing   # F5  smoothing effect
    python scripts/make_paper_figures.py ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import csv  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation.trajectory_metrics import load_traj, raw_kinematics  # noqa: E402

TRAJ = PROJECT_ROOT / "data/video_annotations/trajectories"
FIG_DIR = PROJECT_ROOT / "paper/figures"

BENCH_CLIPS = ["100_m_DJI_0001_1min", "100_m_DJI_0054_1min",
               "120_m_DJI_0006_1min", "120_m_DJI_0047_1min"]
TRACKERS = ["bytetrack", "botsort", "ocsort", "deepocsort"]
LABEL = {"gt": "GT (ceiling)", "bytetrack": "ByteTrack", "botsort": "BoT-SORT",
         "ocsort": "OC-SORT", "deepocsort": "Deep OC-SORT"}
COLOR = {"gt": "#000000", "bytetrack": "#0072B2", "botsort": "#009E73",
         "ocsort": "#D55E00", "deepocsort": "#CC79A7"}
STYLE = {"gt": (0, (4, 2)), "bytetrack": "-", "botsort": "-",
         "ocsort": "-", "deepocsort": (0, (3, 1, 1, 1))}  # OC pair overlap; dashdot DeepOC
A_MAX = 8.0  # physical accel bound (m/s^2)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 120, "savefig.bbox": "tight",
})


def _save(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png, pdf = FIG_DIR / f"{stem}.png", FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"wrote {png.relative_to(PROJECT_ROOT)}  and  {pdf.name}")


def _gather_raw(source: str, field: str) -> np.ndarray:
    """Concatenate a raw-kinematics field across the 4 benchmark clips."""
    vals = []
    for clip in BENCH_CLIPS:
        npz = (TRAJ / f"{clip}.gt.npz") if source == "gt" else \
              (TRAJ / f"{clip}.{source}.linked.npz")
        if not npz.exists():
            continue
        data, idx, meta = load_traj(npz)
        k = raw_kinematics(data, idx, 1.0 / float(meta.get("fps", 29.97)))
        vals.append(k[field])
    return np.concatenate(vals) if vals else np.array([])


def _gather_col(source: str, col: str) -> np.ndarray:
    """Concatenate a stored .npz column across the 4 benchmark clips."""
    vals = []
    for clip in BENCH_CLIPS:
        npz = (TRAJ / f"{clip}.gt.npz") if source == "gt" else \
              (TRAJ / f"{clip}.{source}.linked.npz")
        if not npz.exists():
            continue
        data, idx, _ = load_traj(npz)
        vals.append(data[:, idx[col]])
    return np.concatenate(vals) if vals else np.array([])


def _ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys


# --------------------------------------------------------------------------- #
# F6a -- acceleration plausibility (ECDF, raw kinematics)
# --------------------------------------------------------------------------- #
def fig_accel():
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for src in ["gt"] + TRACKERS:
        a = _gather_raw(src, "accel_mag")
        if a.size == 0:
            continue
        xs, ys = _ecdf(a)
        frac_ok = float(np.mean(a <= A_MAX))
        ax.plot(xs, ys, color=COLOR[src], linestyle=STYLE[src], linewidth=1.9,
                label=f"{LABEL[src]}  ({100*frac_ok:.1f}% ≤ bound)")
    ax.axvline(A_MAX, color="0.4", linestyle=":", linewidth=1.3)
    ax.text(A_MAX + 0.15, 0.06, f"physical bound\n{A_MAX:.0f} m/s²",
            color="0.3", fontsize=9)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 1.005)
    ax.set_xlabel("Acceleration |a| (m/s², raw)")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.25)
    _save(fig, "F6a_accel_plausibility")


# --------------------------------------------------------------------------- #
# F6b -- jerk distribution (ECDF, smoothed kinematics = relative smoothness)
# --------------------------------------------------------------------------- #
def fig_jerk():
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    xmax = 60.0
    for src in ["gt"] + TRACKERS:
        j = _gather_col(src, "jerk_mps3")
        if j.size == 0:
            continue
        xs, ys = _ecdf(j)
        med = float(np.median(j))
        ax.plot(xs, ys, color=COLOR[src], linestyle=STYLE[src], linewidth=1.9,
                label=f"{LABEL[src]}  (median {med:.0f} m/s³)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.005)
    ax.set_xlabel("Jerk |j| (m/s³, smoothed)")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.25)
    _save(fig, "F6b_jerk_smoothness")


# --------------------------------------------------------------------------- #
# F5 -- smoothing effect on one example track (speed + acceleration)
# --------------------------------------------------------------------------- #
def fig_smoothing(clip="120_m_DJI_0006_1min", seg=26):
    data, idx, meta = load_traj(TRAJ / f"{clip}.gt.npz")
    fps = float(meta.get("fps", 29.97))
    k = raw_kinematics(data, idx, 1.0 / fps)
    m = data[:, idx["seg"]].astype(int) == seg
    fr = data[m, idx["frame"]]
    t = (fr - fr.min()) / fps
    sp_raw = data[m, idx["speed_raw_mps"]] * 3.6
    sp_sm = data[m, idx["speed_mps"]] * 3.6
    ac_raw = k["accel_mag"][m]
    ac_sm = data[m, idx["accel_mps2"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True)
    raw_c, sm_c = "#BBBBBB", "#0072B2"

    ax1.plot(t, sp_raw, color=raw_c, lw=1.0, label="raw (finite-difference)")
    ax1.plot(t, sp_sm, color=sm_c, lw=2.0, label="smoothed (Savitzky–Golay)")
    ax1.set_ylabel("Speed (km/h)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.25)

    ax2.plot(t, ac_raw, color=raw_c, lw=1.0, label="raw |a|")
    ax2.plot(t, ac_sm, color=sm_c, lw=2.0, label="smoothed |a|")
    ax2.axhline(A_MAX, color="#D55E00", ls=":", lw=1.3, label=f"{A_MAX:.0f} m/s² bound")
    ax2.set_ylim(0, min(float(np.nanmax(ac_raw)) * 1.05, 60))
    ax2.set_ylabel("Acceleration |a| (m/s²)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(0, t[-1])
    _save(fig, "F5_smoothing_effect")


# --------------------------------------------------------------------------- #
# F7 -- world-coordinate trajectory spaghetti, coloured by speed
# --------------------------------------------------------------------------- #
def fig_spaghetti(clip="120_m_DJI_0006_1min", source="gt", figsize=(8.6, 4.2)):
    npz = (TRAJ / f"{clip}.gt.npz") if source == "gt" else \
          (TRAJ / f"{clip}.{source}.linked.npz")
    data, idx, meta = load_traj(npz)
    x = data[:, idx["xs_m"]]
    y = data[:, idx["ys_m"]]
    spd = data[:, idx["speed_mps"]] * 3.6  # km/h

    if figsize is None:  # auto-size to the clip's world-extent aspect (equal axes)
        dx, dy = float(np.ptp(x)), float(np.ptp(y))
        aspect = dx / dy if dy else 1.5
        h = 5.0
        figsize = (min(max(4.0 * aspect + 1.6, 5.0), 12.0), h)
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(x, y, c=spd, s=2.0, cmap="turbo", vmin=0,
                    vmax=float(np.percentile(spd, 98)), alpha=0.55, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=30, pad=0.02)
    cb.set_label("Speed (km/h)")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # match overhead image orientation
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    n_tracks = int(meta.get("n_tracks", 0))
    src_lbl = "ground truth" if source == "gt" else LABEL.get(source, source)
    ax.grid(True, alpha=0.2)
    _save(fig, f"F7_trajectories_{clip}_{source}")


# --------------------------------------------------------------------------- #
# F4 -- box-tracking quality vs trajectory plausibility (the thesis figure)
# --------------------------------------------------------------------------- #
STAGE3 = PROJECT_ROOT / "results/stage3/tracker_metrics.csv"
STAGE4 = PROJECT_ROOT / "results/stage4/trajectory_metrics.csv"
MARKER = {"bytetrack": "o", "botsort": "s", "ocsort": "^", "deepocsort": "D"}


def fig_boxvstraj():
    # per-clip linked HOTA (Stage 3) and raw accel-ok (Stage 4)
    hota = {(r["clip"], r["tracker"]): float(r["HOTA"])
            for r in csv.DictReader(STAGE3.open(encoding="utf-8"))
            if r["arm"] == "linked" and r.get("scope", "overall") == "overall"}
    accel = {(r["clip"], r["source"]): float(r["a3_frac_accel_ok"])
             for r in csv.DictReader(STAGE4.open(encoding="utf-8"))
             if r["variant"] == "raw" and r["a3_frac_accel_ok"] not in ("", None)}
    gt_ceiling = accel.get(("ALL", "gt"))

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    if gt_ceiling:
        ax.axhline(gt_ceiling, color="0.3", ls="--", lw=1.3)
        ax.text(0.836, gt_ceiling + 0.004, "GT plausibility ceiling",
                color="0.3", fontsize=9)
    for t in TRACKERS:
        # per-clip points (small, translucent)
        xs = [hota[(c, t)] for c in BENCH_CLIPS if (c, t) in hota and (c, t) in accel]
        ys = [accel[(c, t)] for c in BENCH_CLIPS if (c, t) in hota and (c, t) in accel]
        ax.scatter(xs, ys, c=COLOR[t], marker=MARKER[t], s=14, alpha=0.35,
                   linewidths=0)
        # aggregate point (large, labelled)
        ax_, ay_ = hota.get(("ALL", t)), accel.get(("ALL", t))
        if ax_ and ay_:
            ax.scatter([ax_], [ay_], c=COLOR[t], marker=MARKER[t], s=70,
                       edgecolors="black", linewidths=0.8, zorder=5,
                       label=LABEL[t])
    ax.set_xlabel("HOTA (post-linker)")
    ax.set_ylabel("Fraction |a| ≤ 8 m/s²")
    ax.legend(loc="upper left", frameon=True, fontsize=8.5, title_fontsize=8.5,
              title="aggregate (small = per clip)")
    ax.grid(True, alpha=0.25)
    _save(fig, "F4_box_vs_trajectory")


# --------------------------------------------------------------------------- #
# F9 -- GT-free tracker ranking on the two new unlabelled 120 m clips
# --------------------------------------------------------------------------- #
STAGE4_GTFREE = PROJECT_ROOT / "results/stage4/trajectory_metrics_gtfree.csv"


def fig_gtfree():
    rows = list(csv.DictReader(STAGE4_GTFREE.open(encoding="utf-8")))
    sm = {r["source"]: r for r in rows if r["clip"] == "ALL" and r["variant"] == "smoothed"}
    rw = {r["source"]: r for r in rows if r["clip"] == "ALL" and r["variant"] == "raw"}

    panels = [
        ("sc_seg_per_track", sm, "segments per track", "↓ better", 2),
        ("sc_teleport_breaks", sm, "impossible-jump splits", "↓ better", 0),
        ("a3_frac_accel_ok", rw, "frac. |a| ≤ 8 m/s² (raw)", "↑ better", 3),
        ("a3_jerk_mean_mps3", sm, "mean |jerk| (m/s³, smoothed)", "↓ better", 1),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    names = [LABEL[t] for t in TRACKERS]
    # one style per subplot (trackers are identified by the x-axis, not colour);
    # distinct hatch per panel so the panels read in greyscale too
    PANEL = [("#A1C9F4", "///"), ("#FFB482", "xxx"),
             ("#8DE5A1", "..."), ("#D0BBFF", "\\\\\\")]
    for k, (ax, (col, src, ylab, better, dec)) in enumerate(zip(axes.flat, panels)):
        fc, hatch = PANEL[k]
        vals = [float(src[t][col]) for t in TRACKERS]
        bars = ax.bar(names, vals, color=fc, edgecolor="black", linewidth=0.7,
                      hatch=hatch)
        ax.set_title(f"{ylab}   ({better})", fontsize=10.5)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.{dec}f}",
                    ha="center", va="bottom", fontsize=8.5)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    _save(fig, "F9_gtfree_ranking")


# --------------------------------------------------------------------------- #
# F8 -- calibration overlay: GCP quad + reprojected metric grid on a frame
# --------------------------------------------------------------------------- #
CALIB_DIR = PROJECT_ROOT / "data/video_annotations/calibration"
FRAME_DIR = CALIB_DIR / "frames"
GCP_DIR = PROJECT_ROOT / "configs/calibration"


def _world_to_pixel(world_xy, Hinv):
    pts = np.asarray(world_xy, dtype=float).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    proj = (Hinv @ hom.T).T
    return proj[:, :2] / proj[:, 2:3]


def fig_calibration(clip="120_m_DJI_0041_5min", step=10.0):
    import json
    import yaml

    img = plt.imread(FRAME_DIR / f"{clip}.png")
    H_img, W_img = img.shape[:2]
    H = np.asarray(json.loads((CALIB_DIR / f"{clip}.json").read_text())["H"],
                   dtype=float).reshape(3, 3)
    Hinv = np.linalg.inv(H)
    spec = yaml.safe_load((GCP_DIR / f"{clip}.gcp.yaml").read_text())
    pts = [(p["name"], p["pixel"]) for p in spec["points"]]

    # world extent from the image corners (near-nadir -> whole frame ~ ground)
    corners = np.array([[0, 0], [W_img, 0], [W_img, H_img], [0, H_img]], dtype=float)
    hom = np.hstack([corners, np.ones((4, 1))])
    cw = (H @ hom.T).T
    cw = cw[:, :2] / cw[:, 2:3]
    x0, y0 = np.floor(cw[:, 0].min() / step) * step, np.floor(cw[:, 1].min() / step) * step
    x1, y1 = np.ceil(cw[:, 0].max() / step) * step, np.ceil(cw[:, 1].max() / step) * step

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.imshow(img)

    # reprojected metric grid (constant-X and constant-Y world lines)
    gx = np.arange(x0, x1 + step, step)
    gy = np.arange(y0, y1 + step, step)
    tline = np.linspace(y0, y1, 120)
    for X in gx:
        px = _world_to_pixel(np.column_stack([np.full_like(tline, X), tline]), Hinv)
        ax.plot(px[:, 0], px[:, 1], color="#00E5FF", lw=0.5, alpha=0.55)
    tline = np.linspace(x0, x1, 120)
    for Y in gy:
        px = _world_to_pixel(np.column_stack([tline, np.full_like(tline, Y)]), Hinv)
        ax.plot(px[:, 0], px[:, 1], color="#00E5FF", lw=0.5, alpha=0.55)

    import matplotlib.patheffects as pe
    stroke = [pe.withStroke(linewidth=2.4, foreground="black")]

    # GCP quad + points
    quad = np.array([p[1] for p in pts] + [pts[0][1]], dtype=float)
    ax.plot(quad[:, 0], quad[:, 1], color="#FFD400", lw=2.0, zorder=4)
    for name, (u, v) in pts:
        ax.scatter([u], [v], s=70, facecolors="#FF1744", edgecolors="white",
                   linewidths=1.2, zorder=5)
        ax.annotate(name, (u, v), textcoords="offset points", xytext=(7, 6),
                    color="white", fontsize=10, fontweight="bold",
                    zorder=6).set_path_effects(stroke)

    # annotate the two reference edges with their known world length
    world = [np.asarray(p["world"], dtype=float) for p in spec["points"]]
    pix = [np.asarray(p[1], dtype=float) for p in pts]
    for a, b in ((0, 1), (0, 3)):
        L = float(np.linalg.norm(world[a] - world[b]))
        mid = (pix[a] + pix[b]) / 2
        ax.annotate(f"{L:.0f} m", mid, textcoords="offset points", xytext=(0, -4),
                    color="#FFD400", fontsize=11, fontweight="bold", ha="center",
                    zorder=6).set_path_effects(stroke)

    ax.set_xlim(0, W_img)
    ax.set_ylim(H_img, 0)
    ax.set_xticks([]); ax.set_yticks([])
    _save(fig, f"F8_calibration_{clip}")


# --------------------------------------------------------------------------- #
# F2 -- scene gallery: six clips at a populated frame with box overlays
# --------------------------------------------------------------------------- #
VIDEO_DIR = PROJECT_ROOT / "data/video_clips"
GT_DIR = PROJECT_ROOT / "data/video_annotations/ground_truth"
TRK_DIR = PROJECT_ROOT / "data/video_annotations/tracker_outputs"
CLS_COLOR = {0: "#E41A1C", 1: "#4DAF4A", 2: "#377EB8"}  # LV / MV / SV
CLS_NAME = {0: "LV", 1: "MV", 2: "SV"}

# (clip, scene label, box source file, source tag, frame index)
SCENE_SPECS = [
    ("100_m_DJI_0001_1min", "Intersection (100 m)",
     GT_DIR / "100_m_DJI_0001_1min.gt.txt", "GT", 1114),
    ("100_m_DJI_0054_1min", "Dense intersection (100 m)",
     GT_DIR / "100_m_DJI_0054_1min.gt.txt", "GT", 540),
    ("120_m_DJI_0006_1min", "Roundabout (120 m)",
     GT_DIR / "120_m_DJI_0006_1min.gt.txt", "GT", 907),
    ("120_m_DJI_0047_1min", "Suburban intersection (120 m)",
     GT_DIR / "120_m_DJI_0047_1min.gt.txt", "GT", 675),
    ("120_m_DJI_0001_5min", "Urban intersection (120 m, GT-free)",
     TRK_DIR / "120_m_DJI_0001_5min.bytetrack.linked.txt", "ByteTrack", 2899),
    ("120_m_DJI_0041_5min", "Arterial intersection (120 m, GT-free)",
     TRK_DIR / "120_m_DJI_0041_5min.bytetrack.linked.txt", "ByteTrack", 2891),
]


def _boxes_at(path: Path, frame: int):
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        a = ln.split(",")
        if int(a[0]) != frame:
            continue
        x, y, w, h = (float(a[i]) for i in range(2, 6))
        cls = int(float(a[7])) if len(a) > 7 else 1
        out.append((x, y, w, h, cls))
    return out


def fig_scenes():
    import cv2
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 5.0))
    for ax, (clip, label, src, tag, frame) in zip(axes.flat, SCENE_SPECS):
        cap = cv2.VideoCapture(str(VIDEO_DIR / f"{clip}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            ax.set_visible(False)
            continue
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = _boxes_at(src, frame)
        for x, y, w, h, cls in boxes:
            ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                   edgecolor=CLS_COLOR.get(cls, "#FFFFFF"),
                                   linewidth=0.6))
        ax.set_title(f"{label}\n{tag}, {len(boxes)} vehicles", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])

    handles = [plt.Line2D([0], [0], color=CLS_COLOR[c], lw=2, label=CLS_NAME[c])
               for c in (0, 1, 2)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    _save(fig, "F2_scene_gallery")


# --------------------------------------------------------------------------- #
# F3 -- detector accuracy vs inference cost (Pareto)
# --------------------------------------------------------------------------- #
STAGE1_AGG = PROJECT_ROOT / "results/stage1/aggregated.csv"
STAGE1_SPEED = PROJECT_ROOT / "results/stage1/inference_speed.csv"
DET_DISPLAY = {"yolov9m": "YOLOv9-m", "yolov10m": "YOLOv10-m", "yolo11m": "YOLO11-m",
               "yolo12m": "YOLO12-m", "yolo26m": "YOLO26-m", "rtdetr_l": "RT-DETR-L"}


def fig_detectors():
    acc = {}
    for r in csv.DictReader(STAGE1_AGG.open(encoding="utf-8")):
        base = r["model_name"].replace(".pt", "").replace("-", "_")
        acc[base] = (float(r["overall_mAP50_95_mean"]), float(r["overall_mAP50_mean"]))
    speed = {r["detector"]: float(r["fps"])
             for r in csv.DictReader(STAGE1_SPEED.open(encoding="utf-8"))}

    pts = [(d, speed[d], acc[d][0]) for d in DET_DISPLAY if d in speed and d in acc]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    # Pareto frontier (upper-right is best): scanning from fastest, keep points
    # whose accuracy exceeds everything faster than them.
    front, best_acc = [], -1.0
    for d, fps, a in sorted(pts, key=lambda p: -p[1]):  # high FPS first
        if a > best_acc:
            front.append((fps, a)); best_acc = a
    front.sort()
    ax.plot([f for f, _ in front], [a for _, a in front],
            color="0.6", ls="--", lw=1.0, zorder=1, label="Pareto frontier")

    # per-detector label offsets (dx, dy in pts, horizontal alignment) so labels
    # stay inside the box and clear of each other / the right edge
    LBL = {
        "rtdetr_l": (10, 2, "left"), "yolo12m": (0, 11, "center"),
        "yolov9m": (0, -16, "center"), "yolo26m": (6, 8, "left"),
        "yolo11m": (0, -16, "center"), "yolov10m": (6, 8, "left"),
    }
    for d, fps, a in pts:
        is_rt = d == "rtdetr_l"
        ax.scatter([fps], [a], s=150 if is_rt else 110,
                   c="#D55E00" if is_rt else "#0072B2",
                   marker="*" if is_rt else "o",
                   edgecolors="black", linewidths=0.8, zorder=4)
        dx, dy, ha = LBL.get(d, (8, 6, "left"))
        ax.annotate(DET_DISPLAY[d], (fps, a), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=9.5)
    ax.set_xlim(40, 180)
    ax.set_ylim(0.75, 0.9)
    ax.set_xlabel("Inference speed")
    ax.set_ylabel("mAP50:95 (overall)")
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.25)
    _save(fig, "F3_detector_accuracy_speed")


# --------------------------------------------------------------------------- #
# F7-overlay -- trajectories projected back onto a real video frame
# --------------------------------------------------------------------------- #
def fig_spaghetti_overlay(clip="120_m_DJI_0006_1min", source="gt"):
    import json
    img = plt.imread(FRAME_DIR / f"{clip}.png")
    Hh, Ww = img.shape[:2]
    H = np.asarray(json.loads((CALIB_DIR / f"{clip}.json").read_text())["H"],
                   dtype=float).reshape(3, 3)
    Hinv = np.linalg.inv(H)
    npz = (TRAJ / f"{clip}.gt.npz") if source == "gt" else \
          (TRAJ / f"{clip}.{source}.linked.npz")
    data, idx, meta = load_traj(npz)
    world = data[:, [idx["xs_m"], idx["ys_m"]]]
    spd = data[:, idx["speed_mps"]] * 3.6
    px = _world_to_pixel(world, Hinv)

    fig, ax = plt.subplots(figsize=(9.0, 9.0 * Hh / Ww))
    ax.imshow(img)
    sc = ax.scatter(px[:, 0], px[:, 1], c=spd, s=1.4, cmap="turbo", vmin=0,
                    vmax=float(np.percentile(spd, 98)), alpha=0.5, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=30, pad=0.015)
    cb.set_label("Speed (km/h)")
    ax.set_xlim(0, Ww); ax.set_ylim(Hh, 0)
    ax.set_xticks([]); ax.set_yticks([])
    _save(fig, f"F7_trajectories_{clip}_{source}")


def fig_track_compare(clip, trackers=("botsort", "ocsort"),
                      labels=("BoT-SORT (clean)", "OC-SORT (noisy)")):
    """Two stacked panels: each tracker's linked tracks drawn as connected
    pixel-space polylines over a frame. Identity errors (impossible jumps) show
    as long lines crossing the scene -- the noisy tracker has many more."""
    img = plt.imread(FRAME_DIR / f"{clip}.png")
    Hh, Ww = img.shape[:2]
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 2 * 7.6 * Hh / Ww + 0.3))
    for ax, trk, lab in zip(axes, trackers, labels):
        ax.imshow(img)
        tracks: dict[int, list] = {}
        for ln in (TRK_DIR / f"{clip}.{trk}.linked.txt").read_text().splitlines():
            if not ln.strip():
                continue
            a = ln.split(",")
            fr, tid = int(a[0]), int(a[1])
            x, y, w, h = (float(a[i]) for i in range(2, 6))
            tracks.setdefault(tid, []).append((fr, x + w / 2, y + h / 2))
        for tid, pts in tracks.items():
            pts.sort()
            ax.plot([p[1] for p in pts], [p[2] for p in pts],
                    color="#0050B0", lw=0.4, alpha=0.35, solid_capstyle="round")
        ax.set_title(f"{lab}  ({len(tracks)} tracks)", fontsize=11)
        ax.set_xlim(0, Ww); ax.set_ylim(Hh, 0)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    _save(fig, f"F7compare_{clip}")


FIGURES = {
    "accel": fig_accel,
    "compare_0041_5min": lambda: fig_track_compare("120_m_DJI_0041_5min"),
    "compare_0001_5min": lambda: fig_track_compare("120_m_DJI_0001_5min"),
    "jerk": fig_jerk,
    "smoothing": fig_smoothing,
    # F7 = post-linker trajectories overlaid (speed-coloured) on a real frame
    "f7_0001_1min": lambda: fig_spaghetti_overlay("100_m_DJI_0001_1min", "gt"),
    "f7_0054_1min": lambda: fig_spaghetti_overlay("100_m_DJI_0054_1min", "gt"),
    "f7_0006_1min": lambda: fig_spaghetti_overlay("120_m_DJI_0006_1min", "gt"),
    "f7_0047_1min": lambda: fig_spaghetti_overlay("120_m_DJI_0047_1min", "gt"),
    "f7_0001_5min": lambda: fig_spaghetti_overlay("120_m_DJI_0001_5min", "botsort"),
    "f7_0041_5min": lambda: fig_spaghetti_overlay("120_m_DJI_0041_5min", "botsort"),
    "boxvstraj": fig_boxvstraj,
    "gtfree": fig_gtfree,
    "calibration": fig_calibration,
    "scenes": fig_scenes,
    "detectors": fig_detectors,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in FIGURES:
        print("usage: make_paper_figures.py <" + "|".join(FIGURES) + ">")
        return 2
    FIGURES[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
