"""Unpack a CVAT MOT 1.1 export into a repo-convention Stage 2 ground-truth txt.

This is the inverse of `02d_pack_cvat_mot.py`. After a clip's annotation is
finished in CVAT and exported via "Actions -> Export task dataset -> MOT 1.1",
you get a zip laid out as:

    <export>.zip
    \\- gt/
       |- gt.txt        # frame,id,x,y,w,h,not_ignored,class_id,visibility
       \\- labels.txt

CVAT/Datumaro writes the class column **1-based** (it reads back with
`label_id = class_id - 1`), so an exported MOT file uses 1=LV, 2=MV, 3=SV.
The rest of this repo -- detector, dataset.yaml, linker, renderer -- uses the
**0-based** convention 0=LV, 1=MV, 2=SV. This script shifts the class column
1-based -> 0-based so the ground truth matches everything downstream, and
writes a canonical MOT gt file to data/video_annotations/ground_truth/.

It also runs the protocol sec.8 sanity check (track / row / frame counts).

Output row format (0-based class, MOT-Challenge style to match the linker):

    frame, id, x, y, w, h, conf, class, -1, -1

Examples:
    # from the exported zip, clip stem inferred from the known Stage 2 stems
    python scripts/02e_unpack_cvat_mot.py \\
        --input "C:/Downloads/task_uav-thai-0006-2026_06_03-mot 1.1.zip"

    # be explicit about the output clip stem
    python scripts/02e_unpack_cvat_mot.py \\
        --input export.zip --clip 120_m_DJI_0006_1min

    # input may also be an already-extracted gt.txt
    python scripts/02e_unpack_cvat_mot.py --input gt/gt.txt --clip 120_m_DJI_0006_1min
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Index = repo (0-based) class id. Must match 02d_pack_cvat_mot.py / dataset.
CLASS_NAMES = ["LV", "MV", "SV"]

KNOWN_STEMS = [
    "100_m_DJI_0001_1min",
    "100_m_DJI_0054_1min",
    "120_m_DJI_0006_1min",
    "120_m_DJI_0047_1min",
]

DEFAULT_OUT_DIR = Path("data/video_annotations/ground_truth")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True,
                   help="CVAT MOT 1.1 export zip, or an extracted gt.txt.")
    p.add_argument("--clip", default=None,
                   help="Output clip stem (e.g. 120_m_DJI_0006_1min). If "
                        "omitted, inferred from the input filename against the "
                        "four known Stage 2 stems.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output directory. Default: {DEFAULT_OUT_DIR}")
    return p.parse_args()


def read_gt_text(path: Path) -> str:
    """Return the raw gt.txt contents from a zip export or a plain txt file."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            candidates = [n for n in zf.namelist() if n.endswith("gt.txt")]
            if not candidates:
                raise ValueError(f"No gt.txt found inside {path}")
            # prefer gt/gt.txt if multiple
            name = next((c for c in candidates if c.endswith("gt/gt.txt")),
                        candidates[0])
            logger.info("Reading %s from %s", name, path.name)
            return zf.read(name).decode("utf-8")
    return path.read_text(encoding="utf-8")


def infer_clip(input_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = input_path.stem
    for stem in KNOWN_STEMS:
        if stem in name:
            return stem
    raise SystemExit(
        f"Could not infer clip stem from '{input_path.name}'. "
        f"Pass --clip explicitly (one of: {', '.join(KNOWN_STEMS)})."
    )


def convert(raw_text: str) -> tuple[list[str], dict]:
    """Convert CVAT MOT rows (1-based class) to repo rows (0-based class)."""
    out: list[str] = []
    tracks: dict[int, list[int]] = defaultdict(list)
    bad_class = 0
    for line in raw_text.splitlines():
        line = line.strip()
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
        cls_export = int(parts[7])          # CVAT 1-based
        cls = cls_export - 1                 # -> repo 0-based
        if not 0 <= cls < len(CLASS_NAMES):
            bad_class += 1
            logger.warning("  class id %d (->%d) out of range in row: %s",
                           cls_export, cls, line)
        tracks[tid].append(frame)
        out.append(
            f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},{cls},-1,-1"
        )
    max_frame = max((max(fs) for fs in tracks.values()), default=0)
    stats = {
        "rows": len(out),
        "tracks": len(tracks),
        "max_frame": max_frame,
        "bad_class": bad_class,
    }
    return out, stats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()

    if not args.input.exists():
        logger.error("Input not found: %s", args.input)
        return 2

    clip = infer_clip(args.input, args.clip)
    raw_text = read_gt_text(args.input)
    rows, stats = convert(raw_text)
    if not rows:
        logger.error("No valid MOT rows parsed from %s", args.input)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{clip}.gt.txt"
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    logger.info("Wrote %s", out_path)
    logger.info("  tracks=%d  rows=%d  max-frame=%d",
                stats["tracks"], stats["rows"], stats["max_frame"])
    if stats["bad_class"]:
        logger.warning("  %d row(s) had out-of-range class after the 1->0 "
                       "shift -- check the CVAT project label order (LV,MV,SV).",
                       stats["bad_class"])
    if stats["max_frame"] != 1799:
        logger.warning("  max-frame is %d, expected 1799 (full clip). Verify "
                       "the export covered all frames.", stats["max_frame"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
