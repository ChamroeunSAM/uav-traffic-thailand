"""Merge 90m and 120m YOLOv8 exports into a unified train/val/test dataset.

Source layout (already prefixed by 01_rename_with_altitude_prefix.py):
    data/annotations/{90m,120m}/{train,valid,test}/{images,labels}
    data/annotations/{90m,120m}/data.yaml

Output:
    data/annotations/merged/{train,val,test}/{images,labels}   (val, not valid)
    data/splits/dataset.yaml
    data/splits/altitude_index.json
    data/splits/split_metadata.json

Re-runs need --force to overwrite an existing merged/.
"""

import argparse
import json
import logging
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOT_ROOT = PROJECT_ROOT / "data" / "annotations"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MERGED_ROOT = ANNOT_ROOT / "merged"

SRC_SPLITS = ("train", "valid", "test")
DST_SPLIT_FOR = {"train": "train", "valid": "val", "test": "test"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOW_TRAIN_THRESHOLD = 50

NOTES = (
    "Combined-altitude training (Decision D2). Frames extracted from continuous "
    "UAV recordings at 90 m and 120 m above ground. Class IDs are reconciled to "
    "a unified namespace; per-altitude provenance preserved in altitude_index.json "
    "for stratified evaluation."
)


def read_data_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def index_images(images_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not images_dir.is_dir():
        return out
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out[p.stem] = p
    return out


def index_labels(labels_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not labels_dir.is_dir():
        return out
    for p in labels_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".txt":
            out[p.stem] = p
    return out


def verify_source(altitude: str, src_root: Path) -> dict:
    """Verify the source layout, prefix correctness, and image-label pairing."""
    if not src_root.is_dir():
        raise FileNotFoundError(f"{altitude} source not found: {src_root}")

    yaml_path = src_root / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"{altitude} missing data.yaml: {yaml_path}")
    cfg = read_data_yaml(yaml_path)
    if "names" not in cfg:
        raise ValueError(f"{altitude} data.yaml missing 'names': {yaml_path}")
    class_names = list(cfg["names"])

    prefix = f"{altitude}_"
    pairs_per_split: dict[str, list[tuple[Path, Path]]] = {}
    for split in SRC_SPLITS:
        split_dir = src_root / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"{altitude}/{split} missing images/ or labels/: {split_dir}"
            )
        images = index_images(images_dir)
        labels = index_labels(labels_dir)
        matched = sorted(set(images) & set(labels))
        orphan_imgs = sorted(set(images) - set(labels))
        orphan_lbls = sorted(set(labels) - set(images))
        if orphan_imgs:
            logging.warning(
                "%s/%s: %d orphan image(s) without label, will be skipped. "
                "First few: %s",
                altitude, split, len(orphan_imgs), orphan_imgs[:5],
            )
        if orphan_lbls:
            logging.warning(
                "%s/%s: %d orphan label(s) without image, will be skipped. "
                "First few: %s",
                altitude, split, len(orphan_lbls), orphan_lbls[:5],
            )
        bad_prefix = [s for s in matched if not s.startswith(prefix)]
        if bad_prefix:
            raise ValueError(
                f"{altitude}/{split}: {len(bad_prefix)} file(s) missing the "
                f"'{prefix}' prefix. Run 01_rename_with_altitude_prefix.py first. "
                f"First few: {bad_prefix[:5]}"
            )
        pairs_per_split[split] = [(images[s], labels[s]) for s in matched]
        logging.info(
            "%s/%s: verified %d matched pair(s).", altitude, split, len(matched),
        )

    return {"class_names": class_names, "pairs_per_split": pairs_per_split}


def reconcile_classes(
    names_90m: list[str], names_120m: list[str]
) -> tuple[list[str], dict[int, int], dict[int, int]]:
    """Return (unified_names, remap_90m, remap_120m). Identity remap if equal."""
    if names_90m == names_120m:
        identity = {i: i for i in range(len(names_90m))}
        logging.info(
            "Class lists identical across altitudes (%d classes): %s",
            len(names_90m), names_90m,
        )
        return list(names_90m), identity, identity

    unified = list(names_90m)
    for n in names_120m:
        if n not in unified:
            unified.append(n)
    remap_90m = {i: unified.index(n) for i, n in enumerate(names_90m)}
    remap_120m = {i: unified.index(n) for i, n in enumerate(names_120m)}
    logging.warning(
        "Class lists differ — building unified namespace.\n"
        "  90m:     %s\n"
        "  120m:    %s\n"
        "  unified: %s\n"
        "  remap_90m:  %s\n"
        "  remap_120m: %s",
        names_90m, names_120m, unified, remap_90m, remap_120m,
    )
    return unified, remap_90m, remap_120m


def copy_label_with_remap(
    src: Path, dst: Path, remap: dict[int, int], num_classes: int,
) -> Counter:
    """Copy a YOLO label, rewriting class IDs through `remap`. Returns class Counter."""
    counts: Counter = Counter()
    out_lines: list[str] = []
    with src.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            try:
                old_id = int(parts[0])
            except (IndexError, ValueError):
                logging.warning("Malformed label line in %s, copied unchanged: %r", src, line)
                out_lines.append(line)
                continue
            if old_id not in remap:
                logging.error(
                    "Label %s has class id %d not in remap (skipping line).",
                    src, old_id,
                )
                continue
            new_id = remap[old_id]
            if not (0 <= new_id < num_classes):
                logging.error(
                    "Remapped id %d out of range [0,%d) in %s (skipping line).",
                    new_id, num_classes, src,
                )
                continue
            parts[0] = str(new_id)
            out_lines.append(" ".join(parts))
            counts[new_id] += 1
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    shutil.copystat(src, dst)
    return counts


def merge_split(
    altitude: str,
    src_split: str,
    dst_split: str,
    src_root: Path,
    pairs: list[tuple[Path, Path]],
    remap: dict[int, int],
    num_classes: int,
) -> tuple[list[str], Counter]:
    """Copy a split's pairs into MERGED_ROOT/dst_split. Returns (filenames, counts)."""
    dst_images = MERGED_ROOT / dst_split / "images"
    dst_labels = MERGED_ROOT / dst_split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    filenames: list[str] = []
    counts: Counter = Counter()
    for img_src, lbl_src in pairs:
        img_dst = dst_images / img_src.name
        lbl_dst = dst_labels / lbl_src.name
        if img_dst.exists() or lbl_dst.exists():
            logging.error(
                "Collision while merging %s/%s -> %s: %s already present. "
                "Skipping pair.", altitude, src_split, dst_split, img_src.name,
            )
            continue
        shutil.copy2(img_src, img_dst)
        counts.update(copy_label_with_remap(lbl_src, lbl_dst, remap, num_classes))
        filenames.append(img_src.name)
    logging.info(
        "Merged %s/%s -> %s/: %d pair(s), %d object(s).",
        altitude, src_split, dst_split, len(filenames), sum(counts.values()),
    )
    return filenames, counts


def write_dataset_yaml(class_names: list[str]) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(MERGED_ROOT.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": list(class_names),
    }
    out = SPLITS_DIR / "dataset.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return out


def write_altitude_index(filenames_by_split_alt: dict) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out = SPLITS_DIR / "altitude_index.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(filenames_by_split_alt, f, indent=2)
    return out


def write_split_metadata(
    src_90m: Path,
    src_120m: Path,
    class_names: list[str],
    remap_90m: dict[int, int],
    remap_120m: dict[int, int],
    counts_by_split_alt: dict[tuple[str, str], Counter],
    image_counts_by_split_alt: dict[tuple[str, str], int],
) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    is_identity_90m = all(k == v for k, v in remap_90m.items())
    is_identity_120m = all(k == v for k, v in remap_120m.items())

    split_counts: dict[str, dict] = {}
    for dst_split in ("train", "val", "test"):
        per_alt = {}
        for altitude in ("90m", "120m"):
            counts = counts_by_split_alt.get((dst_split, altitude), Counter())
            per_alt[altitude] = {
                "images": image_counts_by_split_alt.get((dst_split, altitude), 0),
                "objects_total": int(sum(counts.values())),
                "objects_per_class": {
                    class_names[cid]: int(counts.get(cid, 0))
                    for cid in range(len(class_names))
                },
            }
        split_counts[dst_split] = per_alt

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT.resolve()),
        "sources": {
            "90m": str(src_90m.resolve()),
            "120m": str(src_120m.resolve()),
        },
        "merged_root": str(MERGED_ROOT.resolve()),
        "class_names": list(class_names),
        "class_remap_90m": (
            "identity" if is_identity_90m else {str(k): v for k, v in remap_90m.items()}
        ),
        "class_remap_120m": (
            "identity" if is_identity_120m else {str(k): v for k, v in remap_120m.items()}
        ),
        "split_counts": split_counts,
        "notes": NOTES,
    }
    out = SPLITS_DIR / "split_metadata.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def print_summary(
    class_names: list[str],
    image_counts_by_split_alt: dict[tuple[str, str], int],
    counts_by_split_alt: dict[tuple[str, str], Counter],
) -> None:
    print()
    print("=" * 72)
    print("MERGE SUMMARY")
    print("=" * 72)

    print("Per-split image counts:")
    print(f"  {'Split':<8}{'90m':>8}{'120m':>8}{'Total':>8}")
    grand_imgs = 0
    for split in ("train", "val", "test"):
        a = image_counts_by_split_alt.get((split, "90m"), 0)
        b = image_counts_by_split_alt.get((split, "120m"), 0)
        grand_imgs += a + b
        print(f"  {split:<8}{a:>8}{b:>8}{a + b:>8}")
    print(f"  {'TOTAL':<8}{'':>8}{'':>8}{grand_imgs:>8}")

    print()
    print("Per-split object counts:")
    print(f"  {'Split':<8}{'90m':>8}{'120m':>8}{'Total':>8}")
    grand_objs = 0
    for split in ("train", "val", "test"):
        a = sum(counts_by_split_alt.get((split, "90m"), Counter()).values())
        b = sum(counts_by_split_alt.get((split, "120m"), Counter()).values())
        grand_objs += a + b
        print(f"  {split:<8}{a:>8}{b:>8}{a + b:>8}")
    print(f"  {'TOTAL':<8}{'':>8}{'':>8}{grand_objs:>8}")

    print()
    print("Per-class object distribution:")
    header = f"  {'Class':<14}{'train':>8}{'val':>8}{'test':>8}{'total':>8}  status"
    print(header)
    warnings = []
    for cid, name in enumerate(class_names):
        per_split = {}
        for split in ("train", "val", "test"):
            n = sum(
                counts_by_split_alt.get((split, alt), Counter()).get(cid, 0)
                for alt in ("90m", "120m")
            )
            per_split[split] = n
        status_tags = []
        if per_split["train"] < LOW_TRAIN_THRESHOLD:
            status_tags.append("[LOW]")
            warnings.append(f"{name}: only {per_split['train']} training instances")
        if per_split["val"] == 0:
            status_tags.append("[MISSING IN VAL]")
            warnings.append(f"{name}: zero val instances")
        if per_split["test"] == 0:
            status_tags.append("[MISSING IN TEST]")
            warnings.append(f"{name}: zero test instances")
        status = " ".join(status_tags) if status_tags else "OK"
        total = sum(per_split.values())
        print(
            f"  {name:<14}{per_split['train']:>8}{per_split['val']:>8}"
            f"{per_split['test']:>8}{total:>8}  {status}"
        )

    print()
    if warnings:
        print("Sanity warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("All classes adequately represented across splits ✓")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export-90m", default="90m",
                        help="Folder name under data/annotations/ for 90m export.")
    parser.add_argument("--export-120m", default="120m",
                        help="Folder name under data/annotations/ for 120m export.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing merged/ directory.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    src_90m = ANNOT_ROOT / args.export_90m
    src_120m = ANNOT_ROOT / args.export_120m
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Source 90m:  %s", src_90m)
    logging.info("Source 120m: %s", src_120m)
    logging.info("Merged out:  %s", MERGED_ROOT)

    if MERGED_ROOT.exists():
        if not args.force:
            logging.error(
                "Merged output already exists: %s. Re-run with --force to overwrite.",
                MERGED_ROOT,
            )
            return 1
        logging.warning("Removing existing merged/ (--force).")
        shutil.rmtree(MERGED_ROOT)

    info_90m = verify_source("90m", src_90m)
    info_120m = verify_source("120m", src_120m)

    class_names, remap_90m, remap_120m = reconcile_classes(
        info_90m["class_names"], info_120m["class_names"],
    )
    num_classes = len(class_names)

    filenames_by_split_alt: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"90m": [], "120m": []}
    )
    counts_by_split_alt: dict[tuple[str, str], Counter] = {}
    image_counts_by_split_alt: dict[tuple[str, str], int] = {}

    for altitude, info, remap in (
        ("90m", info_90m, remap_90m),
        ("120m", info_120m, remap_120m),
    ):
        src_root = src_90m if altitude == "90m" else src_120m
        for src_split in SRC_SPLITS:
            dst_split = DST_SPLIT_FOR[src_split]
            pairs = info["pairs_per_split"][src_split]
            filenames, counts = merge_split(
                altitude, src_split, dst_split, src_root, pairs, remap, num_classes,
            )
            filenames_by_split_alt[dst_split][altitude] = filenames
            counts_by_split_alt[(dst_split, altitude)] = counts
            image_counts_by_split_alt[(dst_split, altitude)] = len(filenames)

    yaml_path = write_dataset_yaml(class_names)
    idx_path = write_altitude_index(dict(filenames_by_split_alt))
    meta_path = write_split_metadata(
        src_90m, src_120m, class_names, remap_90m, remap_120m,
        counts_by_split_alt, image_counts_by_split_alt,
    )
    logging.info("Wrote %s", yaml_path)
    logging.info("Wrote %s", idx_path)
    logging.info("Wrote %s", meta_path)

    print_summary(class_names, image_counts_by_split_alt, counts_by_split_alt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
