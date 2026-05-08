"""Prefix image and label files in data/annotations/{90m,120m} with altitude tags.

Roboflow exports drop matching basenames into per-altitude folders. To merge them
into a combined dataset later, every filename needs to carry its altitude so we
can disambiguate collisions and trace any sample back to its source.

Dry-run by default; pass --apply to actually rename.
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ALTITUDES = ("90m", "120m")
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT = ".txt"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOT_ROOT = PROJECT_ROOT / "data" / "annotations"


def index_images(images_dir: Path) -> dict[str, Path]:
    """Map stem -> image Path. Warn on duplicate stems with different extensions."""
    out: dict[str, Path] = {}
    seen: dict[str, list[Path]] = defaultdict(list)
    if not images_dir.is_dir():
        return out
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        seen[p.stem].append(p)
    for stem, paths in seen.items():
        if len(paths) > 1:
            logging.warning(
                "Multiple image extensions for stem %r in %s: %s — using %s",
                stem,
                images_dir,
                [p.name for p in paths],
                paths[0].name,
            )
        out[stem] = paths[0]
    return out


def index_labels(labels_dir: Path) -> dict[str, Path]:
    """Map stem -> label Path."""
    out: dict[str, Path] = {}
    if not labels_dir.is_dir():
        return out
    for p in labels_dir.iterdir():
        if p.is_file() and p.suffix.lower() == LABEL_EXT:
            out[p.stem] = p
    return out


def plan_split(altitude: str, split: str) -> dict:
    """Inspect one split and produce a rename plan."""
    images_dir = ANNOT_ROOT / altitude / split / "images"
    labels_dir = ANNOT_ROOT / altitude / split / "labels"
    prefix = f"{altitude}_"

    images = index_images(images_dir)
    labels = index_labels(labels_dir)

    image_stems = set(images)
    label_stems = set(labels)
    matched_stems = image_stems & label_stems
    orphan_images = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)

    already_prefixed = []
    to_rename = []
    collisions = []

    for stem in sorted(matched_stems):
        if stem.startswith(prefix):
            already_prefixed.append(stem)
            continue
        img_src = images[stem]
        lbl_src = labels[stem]
        img_dst = img_src.with_name(prefix + img_src.name)
        lbl_dst = lbl_src.with_name(prefix + lbl_src.name)
        if img_dst.exists() or lbl_dst.exists():
            collisions.append((stem, img_dst.exists(), lbl_dst.exists()))
            continue
        to_rename.append((img_src, img_dst, lbl_src, lbl_dst))

    return {
        "altitude": altitude,
        "split": split,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "matched_pairs": len(matched_stems),
        "already_prefixed": already_prefixed,
        "to_rename": to_rename,
        "collisions": collisions,
        "orphan_images": orphan_images,
        "orphan_labels": orphan_labels,
    }


def print_summary(plans: list[dict], header: str) -> None:
    print()
    print("=" * 60)
    print(header)
    print("=" * 60)
    print(f"{'Altitude':<8} {'Split':<6} {'Pairs':>6} {'Already':>8} "
          f"{'To rename':>10} {'Orphans':>8}")
    total_already = total_rename = total_orphan = 0
    for p in plans:
        orphans = len(p["orphan_images"]) + len(p["orphan_labels"])
        total_already += len(p["already_prefixed"])
        total_rename += len(p["to_rename"])
        total_orphan += orphans
        print(
            f"{p['altitude']:<8} {p['split']:<6} "
            f"{p['matched_pairs']:>6} {len(p['already_prefixed']):>8} "
            f"{len(p['to_rename']):>10} {orphans:>8}"
        )
    print("-" * 60)
    print(f"{'TOTAL':<15} {'':>6} {total_already:>8} {total_rename:>10} "
          f"{total_orphan:>8}")
    print("=" * 60)


def report_issues(plans: list[dict], dry_run: bool) -> None:
    for p in plans:
        tag = f"{p['altitude']}/{p['split']}"
        if p["orphan_images"]:
            logging.warning(
                "%s: %d image(s) with no matching label (will not be renamed). "
                "First few: %s",
                tag, len(p["orphan_images"]), p["orphan_images"][:5],
            )
        if p["orphan_labels"]:
            logging.warning(
                "%s: %d label(s) with no matching image (will not be renamed). "
                "First few: %s",
                tag, len(p["orphan_labels"]), p["orphan_labels"][:5],
            )
        if p["collisions"]:
            logging.error(
                "%s: %d collision(s) where target name already exists — these "
                "pairs will be skipped. First few: %s",
                tag, len(p["collisions"]), p["collisions"][:5],
            )
    if dry_run:
        for p in plans:
            for img_src, img_dst, lbl_src, _ in p["to_rename"][:3]:
                logging.info(
                    "[dry-run] %s/%s: %s -> %s (and matching label)",
                    p["altitude"], p["split"], img_src.name, img_dst.name,
                )


def apply_renames(plans: list[dict]) -> tuple[int, int]:
    """Returns (successful_pair_count, failed_pair_count)."""
    ok = 0
    fail = 0
    for p in plans:
        for img_src, img_dst, lbl_src, lbl_dst in p["to_rename"]:
            try:
                img_src.rename(img_dst)
            except OSError as e:
                logging.error("Failed to rename image %s: %s", img_src, e)
                fail += 1
                continue
            try:
                lbl_src.rename(lbl_dst)
            except OSError as e:
                logging.error(
                    "Renamed image but FAILED to rename label %s: %s. "
                    "Reverting image rename to keep pair in lockstep.",
                    lbl_src, e,
                )
                try:
                    img_dst.rename(img_src)
                except OSError as revert_err:
                    logging.error(
                        "Could not revert %s back to %s: %s. Manual fix required.",
                        img_dst, img_src, revert_err,
                    )
                fail += 1
                continue
            ok += 1
    return ok, fail


def verify_post_apply() -> bool:
    """Re-scan and confirm every matched pair carries the correct prefix."""
    print()
    print("=" * 60)
    print("POST-APPLY VERIFICATION")
    print("=" * 60)
    all_good = True
    for altitude in ALTITUDES:
        prefix = f"{altitude}_"
        for split in SPLITS:
            images = index_images(ANNOT_ROOT / altitude / split / "images")
            labels = index_labels(ANNOT_ROOT / altitude / split / "labels")
            matched = set(images) & set(labels)
            unprefixed = [s for s in matched if not s.startswith(prefix)]
            orphans = (set(images) ^ set(labels))
            status = "OK" if not unprefixed and not orphans else "WARN"
            print(
                f"{altitude:<8} {split:<6} pairs={len(matched):>4}  "
                f"unprefixed={len(unprefixed):>3}  orphans={len(orphans):>3}  "
                f"[{status}]"
            )
            if unprefixed:
                all_good = False
                logging.warning(
                    "%s/%s: %d pair(s) still missing prefix. First few: %s",
                    altitude, split, len(unprefixed), unprefixed[:5],
                )
            if orphans:
                all_good = False
    print("=" * 60)
    return all_good


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually rename. Without this flag the script only previews.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not ANNOT_ROOT.exists():
        logging.error("Annotations root not found: %s", ANNOT_ROOT)
        return 1

    plans = []
    for altitude in ALTITUDES:
        for split in SPLITS:
            split_root = ANNOT_ROOT / altitude / split
            if not split_root.is_dir():
                logging.warning("Missing split dir, skipping: %s", split_root)
                continue
            plans.append(plan_split(altitude, split))

    if not plans:
        logging.error("No splits found under %s", ANNOT_ROOT)
        return 1

    header = "RENAME SUMMARY (apply)" if args.apply else "RENAME SUMMARY (dry-run)"
    print_summary(plans, header)
    report_issues(plans, dry_run=not args.apply)

    total_to_rename = sum(len(p["to_rename"]) for p in plans)
    total_collisions = sum(len(p["collisions"]) for p in plans)

    if not args.apply:
        if total_to_rename == 0:
            logging.info("Nothing to do — all matched pairs are already prefixed.")
        else:
            logging.info(
                "Dry-run complete. %d pair(s) would be renamed. "
                "Re-run with --apply to perform the rename.",
                total_to_rename,
            )
        if total_collisions:
            logging.error(
                "%d collision(s) detected; these pairs would be skipped. "
                "Resolve before --apply.", total_collisions,
            )
        return 0

    if total_to_rename == 0:
        logging.info("Nothing to rename — all matched pairs already prefixed.")
        verify_post_apply()
        return 0

    ok, fail = apply_renames(plans)
    logging.info("Renamed %d pair(s) successfully, %d failure(s).", ok, fail)
    all_good = verify_post_apply()
    return 0 if all_good and fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
