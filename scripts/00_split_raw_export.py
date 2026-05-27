"""Partition per-altitude raw Roboflow exports into train/valid/test (70/15/15).

Source layout (Roboflow YOLOv8 export — all images currently in a single
`train/` folder, no pre-existing split):

    data/annotations/data_raw/UAV_Thai_100m.v1i.yolov8/train/{images,labels}/
    data/annotations/data_raw/UAV_Thai_120m.v1i.yolov8/train/{images,labels}/

Destination layout (matches the pre-split convention that
`01_rename_with_altitude_prefix.py` and `00_prepare_splits.py` consume):

    data/annotations/100m/{train,valid,test}/{images,labels}/
    data/annotations/100m/data.yaml
    data/annotations/120m/{train,valid,test}/{images,labels}/
    data/annotations/120m/data.yaml

Behaviour
- Deterministic given --seed (default 42).
- Source files in data_raw/ are NEVER modified — files are copied.
- Existing contents of data/annotations/{100m,120m}/ are removed before
  writing (the previous placeholder dataset is being replaced).
- Stale derived artifacts that were built from placeholder data are also
  cleared: data/annotations/merged/ and data/splits/.

Dry-run by default; pass --apply to actually copy and delete files.

Next steps after applying:
    python scripts/01_rename_with_altitude_prefix.py --apply
    python scripts/00_prepare_splits.py --force
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOT_ROOT = PROJECT_ROOT / "data" / "annotations"
RAW_ROOT = ANNOT_ROOT / "data_raw"

ALTITUDES = ("100m", "120m")
RAW_DIR_FOR = {
    "100m": "UAV_Thai_100m.v1i.yolov8",
    "120m": "UAV_Thai_120m.v1i.yolov8",
}

DST_SPLITS = ("train", "valid", "test")
SPLIT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}

# Canonical class-name mapping. Roboflow exports name the classes "0", "1",
# "2"; we rewrite them here so downstream evaluation reports use human-
# readable labels. Position in the list = class ID in the .txt labels.
#   0 -> LV  (large vehicle: trucks, buses, trailers)
#   1 -> MV  (medium vehicle: cars, vans, pickups, SUVs)
#   2 -> SV  (small vehicle: motorcycles)
CLASS_NAMES = ["LV", "MV", "SV"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT = ".txt"

STALE_DERIVED_DIRS = (
    ANNOT_ROOT / "merged",
    PROJECT_ROOT / "data" / "splits",
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def discover_pairs(src_train_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = src_train_dir / "images"
    labels_dir = src_train_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(
            f"missing images/ or labels/ under {src_train_dir.relative_to(PROJECT_ROOT)}"
        )

    pairs: list[tuple[Path, Path]] = []
    orphan_images: list[str] = []
    for img in sorted(images_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = labels_dir / f"{img.stem}{LABEL_EXT}"
        if not lbl.is_file():
            orphan_images.append(img.name)
            continue
        pairs.append((img, lbl))

    if orphan_images:
        log.warning(
            "  Skipping %d image(s) without a matching .txt label (first 5): %s",
            len(orphan_images), orphan_images[:5],
        )

    orphan_labels = [
        p.name for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == LABEL_EXT
        and not any(p.stem == img.stem for img, _ in pairs)
    ]
    if orphan_labels:
        log.warning(
            "  Found %d .txt label(s) without a matching image (first 5): %s",
            len(orphan_labels), orphan_labels[:5],
        )

    return pairs


def split_indices(n: int, seed: int) -> dict[str, list[int]]:
    """Shuffle [0..n) and slice into train/valid/test by SPLIT_RATIOS.

    Test gets the trailing slice (= n - n_train - n_valid) so rounding error
    lands there; train and valid get exact rounded counts.
    """
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = round(n * SPLIT_RATIOS["train"])
    n_valid = round(n * SPLIT_RATIOS["valid"])
    return {
        "train": idx[:n_train],
        "valid": idx[n_train:n_train + n_valid],
        "test": idx[n_train + n_valid:],
    }


def write_data_yaml(dst_alt_dir: Path, class_names: list[str]) -> None:
    text = (
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    (dst_alt_dir / "data.yaml").write_text(text, encoding="utf-8")


def process_altitude(altitude: str, seed: int, apply: bool) -> dict[str, int]:
    raw_alt_dir = RAW_ROOT / RAW_DIR_FOR[altitude]
    src_train_dir = raw_alt_dir / "train"
    dst_alt_dir = ANNOT_ROOT / altitude

    log.info("=== %s ===", altitude)
    log.info("  Source: %s", src_train_dir.relative_to(PROJECT_ROOT))
    log.info("  Dest:   %s", dst_alt_dir.relative_to(PROJECT_ROOT))

    pairs = discover_pairs(src_train_dir)
    if not pairs:
        raise SystemExit(f"{altitude}: no image/label pairs found")

    splits = split_indices(len(pairs), seed)
    counts = {s: len(splits[s]) for s in DST_SPLITS}

    log.info(
        "  Pairs: %d  ->  train=%d  valid=%d  test=%d  (class_names=%s)",
        len(pairs), counts["train"], counts["valid"], counts["test"], CLASS_NAMES,
    )

    if not apply:
        return counts

    if dst_alt_dir.exists():
        log.info("  Clearing existing %s/", dst_alt_dir.relative_to(PROJECT_ROOT))
        shutil.rmtree(dst_alt_dir)

    for split in DST_SPLITS:
        (dst_alt_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_alt_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in splits[split]:
            img_src, lbl_src = pairs[i]
            shutil.copy2(img_src, dst_alt_dir / split / "images" / img_src.name)
            shutil.copy2(lbl_src, dst_alt_dir / split / "labels" / lbl_src.name)

    write_data_yaml(dst_alt_dir, CLASS_NAMES)
    log.info("  Wrote data.yaml")
    return counts


def clear_stale_derived(apply: bool) -> None:
    log.info("=== Stale derived artifacts (built from placeholder data) ===")
    for d in STALE_DERIVED_DIRS:
        rel = d.relative_to(PROJECT_ROOT)
        if d.exists():
            log.info("  %s %s/", "Removing" if apply else "Would remove", rel)
            if apply:
                shutil.rmtree(d)
        else:
            log.info("  Already absent: %s/", rel)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually copy and delete files (default: dry-run)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the shuffle (default: 42)",
    )
    args = parser.parse_args()

    if not RAW_ROOT.is_dir():
        raise SystemExit(f"raw export root not found: {RAW_ROOT}")

    total = sum(SPLIT_RATIOS.values())
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(f"SPLIT_RATIOS must sum to 1.0; got {total}")

    log.info(
        "=== %s | seed=%d | ratios train=%.2f valid=%.2f test=%.2f ===",
        "APPLY" if args.apply else "DRY-RUN",
        args.seed,
        SPLIT_RATIOS["train"], SPLIT_RATIOS["valid"], SPLIT_RATIOS["test"],
    )

    summary: dict[str, dict[str, int]] = {}
    for altitude in ALTITUDES:
        summary[altitude] = process_altitude(altitude, args.seed, args.apply)

    clear_stale_derived(args.apply)

    log.info("=== Summary ===")
    for altitude, counts in summary.items():
        total = sum(counts.values())
        log.info(
            "  %s  total=%d  train=%d  valid=%d  test=%d",
            altitude, total, counts["train"], counts["valid"], counts["test"],
        )

    if not args.apply:
        log.info("(dry-run) Re-run with --apply to perform the operations.")
    else:
        log.info("")
        log.info("Done. Next steps:")
        log.info("  python scripts/01_rename_with_altitude_prefix.py --apply")
        log.info("  python scripts/00_prepare_splits.py --force")

    return 0


if __name__ == "__main__":
    sys.exit(main())
