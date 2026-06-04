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
    ax.set_xlabel("acceleration magnitude  |a|  (m/s², raw)")
    ax.set_ylabel("cumulative fraction of samples")
    ax.set_title("Acceleration plausibility (Axis 3, GT-free)")
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
    ax.set_xlabel("jerk magnitude  |j|  (m/s³, smoothed)")
    ax.set_ylabel("cumulative fraction of samples")
    ax.set_title("Trajectory smoothness (Axis 3, GT-free)")
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
    ax1.set_ylabel("speed (km/h)")
    ax1.set_title(f"Effect of smoothing on differentiated kinematics\n"
                  f"(one vehicle, {clip.replace('_', ' ')}, {t[-1]:.0f}s)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.25)

    ax2.plot(t, ac_raw, color=raw_c, lw=1.0, label="raw |a|")
    ax2.plot(t, ac_sm, color=sm_c, lw=2.0, label="smoothed |a|")
    ax2.axhline(A_MAX, color="#D55E00", ls=":", lw=1.3, label=f"{A_MAX:.0f} m/s² bound")
    ax2.set_ylim(0, min(float(np.nanmax(ac_raw)) * 1.05, 60))
    ax2.set_ylabel("acceleration |a| (m/s²)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(0, t[-1])
    _save(fig, "F5_smoothing_effect")


# --------------------------------------------------------------------------- #
# F7 -- world-coordinate trajectory spaghetti, coloured by speed
# --------------------------------------------------------------------------- #
def fig_spaghetti(clip="120_m_DJI_0006_1min", source="gt"):
    npz = (TRAJ / f"{clip}.gt.npz") if source == "gt" else \
          (TRAJ / f"{clip}.{source}.linked.npz")
    data, idx, meta = load_traj(npz)
    x = data[:, idx["xs_m"]]
    y = data[:, idx["ys_m"]]
    spd = data[:, idx["speed_mps"]] * 3.6  # km/h

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    sc = ax.scatter(x, y, c=spd, s=2.0, cmap="turbo", vmin=0,
                    vmax=float(np.percentile(spd, 98)), alpha=0.55, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("speed (km/h)")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # match overhead image orientation
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    n_tracks = int(meta.get("n_tracks", 0))
    src_lbl = "ground truth" if source == "gt" else LABEL.get(source, source)
    ax.set_title(f"World-coordinate trajectories, coloured by speed\n"
                 f"({clip.replace('_', ' ')}, {src_lbl}, {n_tracks} tracks)")
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
        ax.scatter(xs, ys, c=COLOR[t], marker=MARKER[t], s=28, alpha=0.35,
                   linewidths=0)
        # aggregate point (large, labelled)
        ax_, ay_ = hota.get(("ALL", t)), accel.get(("ALL", t))
        if ax_ and ay_:
            ax.scatter([ax_], [ay_], c=COLOR[t], marker=MARKER[t], s=130,
                       edgecolors="black", linewidths=1.0, zorder=5,
                       label=LABEL[t])
    ax.set_xlabel("HOTA  (box tracking, post-linker)  →  better")
    ax.set_ylabel("frac. |a| ≤ 8 m/s²  (trajectory plausibility)  →  better")
    ax.set_title("Box-tracking quality vs. trajectory plausibility")
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
    colors = [COLOR[t] for t in TRACKERS]
    for ax, (col, src, ylab, better, dec) in zip(axes.flat, panels):
        vals = [float(src[t][col]) for t in TRACKERS]
        bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(f"{ylab}   ({better})", fontsize=10.5)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.{dec}f}",
                    ha="center", va="bottom", fontsize=8.5)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle("GT-free tracker ranking — two unlabelled 120 m clips "
                 "(ByteTrack/BoT-SORT lead every axis)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "F9_gtfree_ranking")


FIGURES = {
    "accel": fig_accel,
    "jerk": fig_jerk,
    "smoothing": fig_smoothing,
    "spaghetti": fig_spaghetti,
    "boxvstraj": fig_boxvstraj,
    "gtfree": fig_gtfree,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in FIGURES:
        print("usage: make_paper_figures.py <" + "|".join(FIGURES) + ">")
        return 2
    FIGURES[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
