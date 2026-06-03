"""Pack MOT pseudo-track txt files into CVAT-importable MOT 1.1 zip archives.

CVAT's "MOT 1.1" annotation importer does NOT accept a bare .txt file -- it
recognizes the format by archive *structure*, and expects a zip laid out as:

    <clip>.zip
    \\- gt/
       |- gt.txt           (the track rows)
       \\- labels.txt       (class names, one per line; mandatory for our
                            non-official LV/MV/SV labels)

Both files live INSIDE gt/. (CVAT/Datumaro ignores a labels.txt placed at
the archive root when gt/gt.txt is present, then falls back to the built-in
MOT label enum -- which is why a root-level labels.txt yields the error
"Label 'pedestrian' is not registered for this task".)

This script wraps each Stage 2 seed file into that structure so it can be
uploaded via CVAT's "Actions -> Upload annotations -> MOT 1.1".

It also normalizes each row to canonical 9-column MOT gt form:

    frame, id, x, y, w, h, conf, class, visibility

Two transforms applied per row:

  * The class column is rewritten 0-based -> 1-based (LV 0->1, MV 1->2,
    SV 2->3). Datumaro's MOT reader does `label_id = class_id - 1` to index
    labels.txt, so the on-disk class_id MUST be 1-based or every box lands
    on the wrong label (silently -- no error, just shuffled classes).
  * The linker emits a 10-column tail (..., conf, class, -1, -1); the
    trailing -1 visibility can be read by CVAT as "ignored/invisible", so
    the visibility column is forced to 1.

Frame and id are passed through unchanged (1-indexed, per MOT spec -- CVAT
converts to its 0-indexed internal frames on import).

The labels.txt order MUST match the dataset class-id order (0=LV, 1=MV,
2=SV) and the CVAT project label order (see stage2_annotation_protocol.md
sec.5).

Example:
    # pack every *.linked_v3.txt seed under the default pseudo_tracks dir
    python scripts/02d_pack_cvat_mot.py

    # pack a single file to a chosen output
    python scripts/02d_pack_cvat_mot.py \\
        --input data/video_annotations/pseudo_tracks/120_m_DJI_0006_1min.linked_v3.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Index = class id. Order is the dataset's large->small order (0=LV,1=MV,2=SV)
# and must match the CVAT project label-creation order.
CLASS_NAMES = ["LV", "MV", "SV"]

DEFAULT_DIR = Path("data/video_annotations/pseudo_tracks")
DEFAULT_GLOB = "*.linked_v3.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", type=Path, default=None,
        help="A single MOT txt to pack. If omitted, all files matching "
             f"'{DEFAULT_GLOB}' under --dir are packed.",
    )
    p.add_argument(
        "--dir", type=Path, default=DEFAULT_DIR,
        help=f"Directory to scan when --input is omitted. Default: {DEFAULT_DIR}",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write the .zip archives. Default: alongside each input.",
    )
    return p.parse_args()


def normalize_rows(src: Path) -> tuple[list[str], int, int]:
    """Read a MOT txt and return canonical 9-column gt.txt lines.

    Returns (lines, n_rows, n_tracks). Lines have no trailing newline.
    """
    out: list[str] = []
    track_ids: set[int] = set()
    with src.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                logger.warning("  skipping malformed row (<8 cols): %s", line)
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = (float(parts[i]) for i in range(2, 6))
            conf = float(parts[6])
            cls = int(parts[7])
            if not 0 <= cls < len(CLASS_NAMES):
                logger.warning("  class id %d out of range in row: %s", cls, line)
            track_ids.add(tid)
            # canonical 9-col MOT gt: class shifted 0-based -> 1-based for
            # Datumaro's `label_id = class_id - 1` lookup; visibility -> 1.
            out.append(
                f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},{cls + 1},1"
            )
    return out, len(out), len(track_ids)


def pack_one(src: Path, out_dir: Path | None) -> Path:
    lines, n_rows, n_tracks = normalize_rows(src)
    if not lines:
        raise ValueError(f"No valid MOT rows found in {src}")

    # strip the compound .linked_v3.txt suffix down to the clip stem
    stem = src.name
    for suffix in (".linked_v3.txt", ".txt"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    dest_dir = out_dir or src.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{stem}.cvat_mot.zip"

    gt_blob = "\n".join(lines) + "\n"
    labels_blob = "\n".join(CLASS_NAMES) + "\n"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("gt/gt.txt", gt_blob)
        zf.writestr("gt/labels.txt", labels_blob)

    logger.info(
        "%-40s -> %s   (%d rows, %d tracks)",
        src.name, zip_path.name, n_rows, n_tracks,
    )
    return zip_path


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if args.input is not None:
        sources = [args.input]
    else:
        sources = sorted(args.dir.glob(DEFAULT_GLOB))
        if not sources:
            logger.error("No files matching '%s' under %s", DEFAULT_GLOB, args.dir)
            return 2

    logger.info("Packing %d file(s) into CVAT MOT 1.1 zip(s)", len(sources))
    failures = 0
    for src in sources:
        if not src.exists():
            logger.error("Input not found: %s", src)
            failures += 1
            continue
        try:
            pack_one(src, args.out_dir)
        except Exception as exc:  # noqa: BLE001 -- report and continue
            logger.error("Failed packing %s: %s", src, exc)
            failures += 1

    if failures:
        logger.error("%d file(s) failed", failures)
        return 1
    logger.info("Done. Upload each .cvat_mot.zip via CVAT -> Upload annotations -> MOT 1.1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
