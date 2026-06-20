"""Representative scene gallery: 4 frames from 100 m + 4 frames from 120 m,
one per site type (suburban, curve, roundabout, urban intersection) at each
altitude. Ground-truth boxes are overlaid coloured by class (LV/MV/SV), as
in F2.

Per the request, subtitles carry only the capture elevation -- the site
type is not labelled. Frames are picked from data/annotations/<alt>m/ and
resolved by filename prefix (Roboflow appends a content hash to each name).
Layout: 4 rows x 2 columns, each row pairing a site type across altitudes.

Output: paper/figures/F2b_scene_representatives.{png,pdf}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNO = PROJECT_ROOT / "data/annotations"
OUT = PROJECT_ROOT / "paper/figures"

CLS_COLOR = {0: "#E41A1C", 1: "#4DAF4A", 2: "#377EB8"}  # LV / MV / SV
CLS_NAME = {0: "LV", 1: "MV", 2: "SV"}

# (altitude, filename prefix) in display order (row-major over a 4x2 grid, so
# each row pairs the same site type across altitudes). Site type is encoded in
# the prefix but deliberately not shown in the figure.
PICKS = [
    ("100 m", "100m_01suburban_evening_100m_DJI0054_t010"),        # suburban
    ("120 m", "120m_01suburban_evening_120m_DJI0063_t110"),
    ("100 m", "100m_02curve_evening_100m_DJI0007_t140"),           # curve
    ("120 m", "120m_02curve_evening_120m_DJI0015_t070"),
    ("100 m", "100m_03roundabout_eveningpeak_100m_DJI0005_t070"),  # roundabout
    ("120 m", "120m_03roundabout_eveningpeak_120m_DJI0003_t240"),
    ("100 m", "100m_04urban_eveningpeak_100m_DJI0001_t240"),       # urban intersection
    ("120 m", "120m_04urban_eveningpeak_120m_DJI0007_t290"),
]


def _resolve(prefix: str) -> Path:
    alt = prefix.split("m_")[0] + "m"  # "100m" / "120m"
    for split in ("train", "valid", "test"):
        hits = list((ANNO / alt / split / "images").glob(prefix + "*.jpg"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no image found for prefix {prefix!r}")


def _label_path(img: Path) -> Path:
    return img.parent.parent / "labels" / (img.stem + ".txt")


def _draw_boxes(ax, img: Path, w: int, h: int) -> int:
    """Overlay YOLO-format GT boxes (class cx cy bw bh, normalised)."""
    lbl = _label_path(img)
    if not lbl.exists():
        return 0
    n = 0
    for ln in lbl.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        ax.add_patch(Rectangle((x1, y1), bw * w, bh * h, fill=False,
                               edgecolor=CLS_COLOR.get(cls, "#FFFFFF"),
                               linewidth=0.6))
        n += 1
    return n


def main():
    fig, axes = plt.subplots(4, 2, figsize=(7.4, 9.6))
    for ax, (elev, prefix) in zip(axes.flat, PICKS):
        path = _resolve(prefix)
        im = mpimg.imread(path)
        h, w = im.shape[:2]
        ax.imshow(im)
        n = _draw_boxes(ax, path, w, h)
        ax.set_title(elev, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"  {elev:6s} ({n:3d} boxes) <- {path.name}")

    handles = [plt.Line2D([0], [0], color=CLS_COLOR[c], lw=2, label=CLS_NAME[c])
               for c in (0, 1, 2)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.subplots_adjust(hspace=0.22, wspace=0.02)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT / f"F2b_scene_representatives.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
