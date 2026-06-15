"""Build a GSD-normalized copy of the merged dataset (Nadir-YOLO MSP route i).

Metric Scale Prior, mechanism (b) realized as preprocessing (design doc §8c):
make the same physical object the same pixel size regardless of flight
altitude, so a stock detector trains on a scale-invariant input.

At a fixed network input (1280), equalizing ground-sample-distance (GSD) means
either cropping the coarser-altitude frames or downscaling the finer ones. We
equalize to the FINEST native GSD (100 m = 0.030 m/px) to preserve small-vehicle
detail, so:

  * 100 m frames (0.030 m/px): plain resize 3840x2160 -> OUT (no crop).
  * 120 m frames (0.040 m/px): centre-crop to fraction 0.030/0.040 = 0.75
    (2880x1620), then resize to OUT.

Both land at the same OUT pixel size with identical px/m (~11.11 px/m), so the
downstream stock training/letterbox path treats them identically. The trade-off
(documented for the ablation): cropping drops the outer ~12.5% margin of each
120 m frame, so edge objects in 120 m frames are lost from training. If route i
shows signal, design doc §8c escalates to route ii (metric DFL regression),
which needs no cropping.

Labels are YOLO-normalized (xc,yc,w,h); a uniform final resize leaves them
unchanged, so only the 120 m centre-crop transforms them (shift -> clip -> drop
empties -> renormalize).

Dry-run by default; pass --apply to write. Mirrors the repo's split layout into
``data/annotations/merged_gsdnorm/`` + a ``configs/data/dataset_gsdnorm.yaml``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.gsd import GSD_M_PER_PX, altitude_m  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "annotations" / "merged"
DST = ROOT / "data" / "annotations" / "merged_gsdnorm"
YAML_OUT = ROOT / "configs" / "data" / "dataset_gsdnorm.yaml"
# Matched control: same 1280x720 source pipeline, but NO GSD-equalizing crop
# (both altitudes plain-resized). Isolates the scale-invariance variable from
# the input-resolution change, so the MSP A/B is clean (design doc §8c).
DST_CTRL = ROOT / "data" / "annotations" / "merged_plain1280"
YAML_CTRL = ROOT / "configs" / "data" / "dataset_plain1280.yaml"
SPLITS = ("train", "val", "test")
OUT_W, OUT_H = 1280, 720          # 16:9, matching native 3840x2160 aspect
NATIVE_W, NATIVE_H = 3840, 2160
MIN_NORM_SIDE = 1.0 / OUT_W       # drop boxes smaller than ~1 px after crop

TARGET_GSD = min(GSD_M_PER_PX.values())   # finest = 0.030 (100 m)


def crop_fraction(alt: int) -> float:
    """Centred keep-fraction per axis to bring altitude `alt` to TARGET_GSD."""
    return TARGET_GSD / GSD_M_PER_PX[alt]


def transform_labels(lines: list[str], f: float) -> list[str]:
    """Remap YOLO labels for a centred crop of keep-fraction `f` (per axis).

    f == 1.0 is the identity. Boxes fully outside the crop are dropped; partial
    boxes are clipped to the crop and renormalized.
    """
    if abs(f - 1.0) < 1e-9:
        return lines
    lo = (1.0 - f) / 2.0          # crop spans [lo, 1-lo] in original norm coords
    out: list[str] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts[0], *map(float, parts[1:])
        # original normalized edges
        x1, x2 = xc - w / 2, xc + w / 2
        y1, y2 = yc - h / 2, yc + h / 2
        # into crop frame, renormalized by f
        x1, x2 = (x1 - lo) / f, (x2 - lo) / f
        y1, y2 = (y1 - lo) / f, (y2 - lo) / f
        # clip to [0,1]
        x1, x2 = max(0.0, x1), min(1.0, x2)
        y1, y2 = max(0.0, y1), min(1.0, y2)
        nw, nh = x2 - x1, y2 - y1
        if nw < MIN_NORM_SIDE or nh < MIN_NORM_SIDE:
            continue
        nxc, nyc = (x1 + x2) / 2, (y1 + y2) / 2
        out.append(f"{cls} {nxc:.6f} {nyc:.6f} {nw:.6f} {nh:.6f}")
    return out


def process_image(src_img: Path, dst_img: Path, f: float, apply: bool) -> None:
    if not apply:
        return
    im = Image.open(src_img).convert("RGB")
    if abs(f - 1.0) >= 1e-9:
        cw, ch = round(NATIVE_W * f), round(NATIVE_H * f)
        left, top = (im.width - cw) // 2, (im.height - ch) // 2
        im = im.crop((left, top, left + cw, top + ch))
    im = im.resize((OUT_W, OUT_H), Image.BILINEAR)
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst_img, quality=95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write files (default dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="cap images per split (debug)")
    ap.add_argument("--control", action="store_true",
                    help="matched control: plain-resize both altitudes (no GSD crop)")
    args = ap.parse_args()

    equalize = not args.control
    dst = DST if equalize else DST_CTRL
    yaml_out = YAML_OUT if equalize else YAML_CTRL
    frac = (lambda a: crop_fraction(a)) if equalize else (lambda a: 1.0)

    mode = "GSD-EQUALIZED" if equalize else "CONTROL (plain-resize, no crop)"
    print(f"{mode} dataset build  (TARGET_GSD={TARGET_GSD} m/px, OUT={OUT_W}x{OUT_H})")
    print(f"  fractions: " + ", ".join(f"{a}m->{frac(a):.3f}" for a in sorted(GSD_M_PER_PX)))
    print(f"  src={SRC}\n  dst={dst}\n  apply={args.apply}\n")

    if args.apply and dst.exists():
        shutil.rmtree(dst)

    totals = {"images": 0, "no_alt": 0, "boxes_in": 0, "boxes_out": 0, "dropped_imgs": 0}
    per_alt = {}
    for split in SPLITS:
        img_dir = SRC / split / "images"
        lbl_dir = SRC / split / "labels"
        if not img_dir.exists():
            print(f"  [skip] {split}: no images dir")
            continue
        imgs = sorted(img_dir.iterdir())
        if args.limit:
            imgs = imgs[: args.limit]
        n_split = 0
        for img in imgs:
            alt = altitude_m(img.name)
            if alt is None or alt not in GSD_M_PER_PX:
                totals["no_alt"] += 1
                continue
            f = frac(alt)
            lbl = lbl_dir / (img.stem + ".txt")
            lines = lbl.read_text().splitlines() if lbl.exists() else []
            new_lines = transform_labels(lines, f)
            totals["boxes_in"] += len(lines)
            totals["boxes_out"] += len(new_lines)
            per_alt.setdefault(alt, {"imgs": 0, "in": 0, "out": 0})
            per_alt[alt]["imgs"] += 1
            per_alt[alt]["in"] += len(lines)
            per_alt[alt]["out"] += len(new_lines)
            # an image that loses ALL its boxes to cropping is still a valid
            # (background) frame; keep it, but count it.
            if lines and not new_lines:
                totals["dropped_imgs"] += 1
            process_image(img, dst / split / "images" / img.name, f, args.apply)
            if args.apply:
                out_lbl = dst / split / "labels" / (img.stem + ".txt")
                out_lbl.parent.mkdir(parents=True, exist_ok=True)
                out_lbl.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
            n_split += 1
            totals["images"] += 1
        print(f"  {split}: {n_split} images")

    print("\nper-altitude:")
    for a, d in sorted(per_alt.items()):
        kept = 100.0 * d["out"] / d["in"] if d["in"] else 100.0
        print(f"  {a}m: {d['imgs']} imgs, boxes {d['in']}->{d['out']} ({kept:.1f}% kept)")
    print(f"\ntotals: {totals}")

    if args.apply:
        yaml_out.write_text(
            f"path: {dst}\n"
            "train: train/images\n"
            "val: val/images\n"
            "test: test/images\n"
            "nc: 3\n"
            "names:\n- LV\n- MV\n- SV\n"
        )
        print(f"\nwrote {yaml_out}")
    else:
        print("\n(dry-run; re-run with --apply to write images/labels/yaml)")


if __name__ == "__main__":
    main()
