#!/usr/bin/env python3
"""
Generalized Multi-Class YOLO Label Generator & Manifest Builder

Processes groundtruth bounding box annotations and dataset split manifests for any taxa,
generates normalized YOLO label text files, outputs data.yaml configurations,
and updates manifest CSVs with dynamic per-class counts.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ==========================================
# Coordinate Helpers
# ==========================================
def clamp(v: float, lo: float, hi: float) -> float:
    """Clip coordinates tightly to image bounds."""
    return max(lo, min(hi, v))


def bbox_to_yolo(x1: float, y1: float, x2: float, y2: float, w: float, h: float):
    """Converts absolute bounding box coordinates to normalized YOLO format."""
    x1, y1, x2, y2 = abs(float(x1)), abs(float(y1)), abs(float(x2)), abs(float(y2))

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    x_min = clamp(x_min, 0.0, w - 1.0)
    x_max = clamp(x_max, 0.0, w - 1.0)
    y_min = clamp(y_min, 0.0, h - 1.0)
    y_max = clamp(y_max, 0.0, h - 1.0)

    bw = x_max - x_min
    bh = y_max - y_min

    if bw <= 1 or bh <= 1:
        return None

    xc = x_min + bw / 2.0
    yc = y_min + bh / 2.0
    return (xc / w, yc / h, bw / w, bh / h)


def sanitize_class_name(name: str) -> str:
    """Cleans a class label into a standardized slug (e.g., 'Astropecten sp.' -> 'astropecten_sp')."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean if clean else "unclassified"


# ==========================================
# File Link Deployment Helper
# ==========================================
def deploy_image_file(src: Path, dst: Path, mode: str):
    """Safely deploys target images into YOLO split directory structure."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)


# ==========================================
# Summary Evaluation Printer
# ==========================================
def print_dataset_summary(df: pd.DataFrame, class_cols: list[str]):
    """Prints a structured summary table evaluating multi-class splits."""
    print("\n" + "=" * 60)
    print(" GENERATED MULTI-CLASS YOLO DATASET SUMMARY")
    print("=" * 60)

    print("\n--- 1. Image Split Composition ---")
    comp_col = df["is_empty"].map({True: "Empty Background", False: "Positive Target"})
    composition = pd.crosstab(df["dataset"], comp_col)
    print(composition.to_string())

    print("\n--- 2. Extracted Class Totals across Splits ---")
    avail_class_cols = [c for c in class_cols if c in df.columns]
    if avail_class_cols:
        totals = df.groupby("dataset")[avail_class_cols].sum()
        print(totals.to_string())
    else:
        print("No class columns found in manifest.")
    print("=" * 60 + "\n")


# ==========================================
# CLI Argument Parser
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generalized Multi-Class YOLO Dataset Builder for HabCam Imagery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory of dataset repo (e.g., ~/PEMAD-ESB-ScalTrain/train_arc/star24)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to input dataset split manifest CSV",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Path to groundtruth annotations CSV",
    )
    parser.add_argument(
        "--pos-dir",
        type=Path,
        default=None,
        help="Path to annotated images directory",
    )
    parser.add_argument(
        "--emp-dir",
        type=Path,
        default=None,
        help="Path to empty images directory",
    )
    parser.add_argument(
        "--yolo-dir",
        type=Path,
        default=None,
        help="Output directory for generated YOLO hierarchy",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Path to output updated dataset split CSV",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "copy", "hardlink"],
        default="symlink",
        help="File placement strategy into yolo/images/",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional explicit list of target class names to include (case-insensitive)",
    )

    args = parser.parse_args()

    # Resolve default paths relative to base_dir if provided
    if args.base_dir:
        base = args.base_dir.resolve()
        if args.manifest is None:
            matches = list((base / "annotations").glob("dataset_split*.csv"))
            args.manifest = matches[0] if matches else base / "annotations" / "dataset_split.csv"

        if args.gt is None:
            matches = list((base / "annotations").glob("groundtruth*.csv"))
            args.gt = matches[0] if matches else base / "annotations" / "groundtruth.csv"

        if args.pos_dir is None:
            args.pos_dir = base / "viame" / "images" if (base / "viame").exists() else base / "annotated"

        if args.emp_dir is None:
            args.emp_dir = base / "empty" if (base / "empty").exists() else base / "empties"

        if args.yolo_dir is None:
            args.yolo_dir = base / "yolo"

        if args.output_manifest is None:
            args.output_manifest = args.manifest

    # Fallback assertions
    if not args.manifest or not args.gt:
        parser.error("You must specify either --base-dir or both --manifest and --gt paths.")

    return args


# ==========================================
# Main Execution Pipeline
# ==========================================
def main():
    args = parse_args()

    print("= = = Launching Generalized Multi-Class YOLO Dataset Pipeline = = =\n")
    print(f"Manifest Path:      {args.manifest}")
    print(f"Groundtruth Path:   {args.gt}")
    print(f"Positives Directory:{args.pos_dir}")
    print(f"Empties Directory:  {args.emp_dir}")
    print(f"Output YOLO Dir:    {args.yolo_dir}")
    print(f"Link Strategy:      {args.link_mode}\n")

    # Reset YOLO directory layout
    if args.yolo_dir.exists():
        print(f"Purging legacy YOLO directory structure at: {args.yolo_dir}")
        shutil.rmtree(args.yolo_dir)

    for split in ["train", "val"]:
        (args.yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Load Data Structures
    manifest_df = pd.read_csv(args.manifest)
    gt_df = pd.read_csv(args.gt)

    # Standardize groundtruth column names to lowercase
    gt_df.columns = [c.strip().lower() for c in gt_df.columns]

    # Resolve class column name
    class_col = "classname" if "classname" in gt_df.columns else "label" if "label" in gt_df.columns else None
    if not class_col:
        raise KeyError("Could not find class column ('classname' or 'label') in groundtruth CSV.")

    # 2. Extract & Map Classes
    gt_df["class_slug"] = gt_df[class_col].apply(sanitize_class_name)

    if args.classes:
        filter_classes = set(sanitize_class_name(c) for c in args.classes)
        gt_df = gt_df[gt_df["class_slug"].isin(filter_classes)].copy()

    unique_slugs = sorted(gt_df["class_slug"].unique().tolist())
    if not unique_slugs:
        raise ValueError("No valid class labels found in groundtruth annotations!")

    class_to_id = {slug: idx for idx, slug in enumerate(unique_slugs)}
    class_to_col = {slug: f"n_{slug}" for slug in unique_slugs}

    print(f"Discovered {len(unique_slugs)} distinct class(es):")
    for slug, idx in class_to_id.items():
        print(f"  [{idx}] -> {slug} (Column: {class_to_col[slug]})")
    print()

    # 3. Process Groundtruth Coordinates
    gt_map = {}
    image_species_counts = {}

    grouped_gt = gt_df.groupby("imagename")

    for img_name, sub in grouped_gt:
        src_path = args.pos_dir / img_name
        if not src_path.exists():
            continue

        try:
            with Image.open(src_path) as im:
                w, h = im.size
        except Exception as e:
            print(f"Warning: Unable to open image {src_path}: {e}")
            continue

        rows = []
        counts = {idx: 0 for idx in class_to_id.values()}

        for r in sub.itertuples():
            slug = getattr(r, "class_slug")
            if slug not in class_to_id:
                continue

            class_idx = class_to_id[slug]
            yolo_box = bbox_to_yolo(r.tlx, r.tly, r.brx, r.bry, w, h)

            if yolo_box:
                rows.append(
                    f"{class_idx} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}"
                )
                counts[class_idx] += 1

        if rows:
            gt_map[img_name] = rows
            image_species_counts[img_name] = counts

    # 4. Deploy Files & Update Manifest
    print(f"Deploying image files and labels using [{args.link_mode}] mode...")
    deployed_records = []

    for _, row in manifest_df.iterrows():
        img_name = row["imagename"]
        is_empty = bool(row["is_empty"])

        # Check split assignment
        is_train = bool(row.get("is_train", row.get("dataset") == "train"))
        split = "train" if is_train else "val"

        src_img_path = args.emp_dir / img_name if is_empty else args.pos_dir / img_name
        dst_img_path = args.yolo_dir / "images" / split / img_name

        if not src_img_path.exists():
            continue

        deploy_image_file(src_img_path, dst_img_path, args.link_mode)

        dst_txt_path = args.yolo_dir / "labels" / split / f"{Path(img_name).stem}.txt"
        box_rows = gt_map.get(img_name, [])
        dst_txt_path.write_text("\n".join(box_rows) + ("\n" if box_rows else ""))

        # Build class count map
        img_counts = image_species_counts.get(img_name, {idx: 0 for idx in class_to_id.values()})

        updated_row = row.to_dict()
        for slug, class_idx in class_to_id.items():
            updated_row[class_to_col[slug]] = img_counts[class_idx]

        deployed_records.append(updated_row)

    # 5. Generate data.yaml Training Configuration
    yaml_lines = [
        f"path: {args.yolo_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for slug, class_idx in class_to_id.items():
        yaml_lines.append(f"  {class_idx}: {slug}")

    yaml_path = args.yolo_dir / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n")
    print(f"Generated training configuration at: {yaml_path}")

    # 6. Export Updated Manifest
    final_df = pd.DataFrame(deployed_records)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_manifest, index=False)
    print(f"Exported updated dataset manifest to: {args.output_manifest}")

    # 7. Print Summary
    class_cols = [class_to_col[s] for s in unique_slugs]
    print_dataset_summary(final_df, class_cols)


if __name__ == "__main__":
    main()