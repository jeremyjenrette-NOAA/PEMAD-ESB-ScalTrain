#!/usr/bin/env python3
"""
Step 2: Stage image files for scallop2224.

Copies (or symlinks) the annotated scallop images into
    <dest-root>/viame/images/
and background (empty) images into
    <dest-root>/empties/
based on the filtered CSVs produced by 01_filter_annotations.py.

Usage:
    python 02_stage_images.py \
        --groundtruth-csv scallop2224/annotations/groundtruth2224.csv \
        --empties-csv scallop2224/annotations/empties2224.csv \
        --source-root /projects/sharkpulse/.../images \
        --dest-root scallop2224 \
        --mode symlink
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def index_images_recursive(root: Path):
    index = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            index.setdefault(p.name, []).append(p.resolve())
    return index


def place_file(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def stage(imagenames, index, dest_dir: Path, mode: str, label: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    unique_names = sorted(set(imagenames))
    missing, ambiguous, placed = [], [], 0
    for name in unique_names:
        matches = index.get(name, [])
        if not matches:
            missing.append(name)
            continue
        if len(matches) > 1:
            ambiguous.append((name, matches))
        src = matches[0]
        place_file(src, dest_dir / name, mode)
        placed += 1

    print(f"[{label}] placed {placed}/{len(unique_names)} images into {dest_dir}")
    if ambiguous:
        print(f"[{label}] WARNING: {len(ambiguous)} basenames matched multiple source files "
              f"(used first match). First few: {[a[0] for a in ambiguous[:5]]}")
    if missing:
        print(f"[{label}] WARNING: {len(missing)} images not found under source root. "
              f"First few: {missing[:10]}")
    return missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groundtruth-csv", required=True, type=Path)
    ap.add_argument("--empties-csv", required=True, type=Path)
    ap.add_argument("--source-root", required=True, type=Path,
                    help="Root directory to search recursively for original images")
    ap.add_argument("--dest-root", required=True, type=Path,
                    help="scallop2224 root; images go to <dest-root>/viame/images and <dest-root>/empties")
    ap.add_argument("--mode", choices=["copy", "symlink"], default="symlink")
    args = ap.parse_args()

    gt = pd.read_csv(args.groundtruth_csv, low_memory=False)
    em = pd.read_csv(args.empties_csv, low_memory=False)

    print(f"Indexing source images under {args.source_root} (this can take a while)...")
    index = index_images_recursive(args.source_root)
    print(f"Indexed {sum(len(v) for v in index.values())} image files "
          f"({len(index)} unique basenames)")

    missing_annot = stage(gt["imagename"], index, args.dest_root / "viame" / "images",
                           args.mode, "annotated")
    missing_empty = stage(em["imagename"], index, args.dest_root / "empties",
                           args.mode, "empties")

    report = args.dest_root / "annotations" / "staging_missing_images.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w") as f:
        f.write("# annotated images not found\n")
        f.write("\n".join(missing_annot))
        f.write("\n\n# empty images not found\n")
        f.write("\n".join(missing_empty))
    print(f"\nMissing-image report written to {report}")


if __name__ == "__main__":
    main()
