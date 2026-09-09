# ==============================================================================
# File: config/extract_crops.py
# Purpose: Extract ground-truth crops from the ORIGINAL multi-class YOLO
#          dataset for Stage 2 classifier training. Fully dynamic to any
#          taxonomy configuration — pass --taxonomy_json, --yolo_root,
#          --csv_split_path, --output_dir for whichever taxon you're running.
#
#          Validates the taxonomy JSON against yolo_root/data.yaml before
#          touching any images: if the two disagree about which class id
#          means which species, this exits with a clear error instead of
#          silently saving crops into the wrong species folders.
# ==============================================================================

import argparse
import cv2
import json
import pandas as pd
from pathlib import Path

from taxonomy_utils import load_taxonomy, validate_taxonomy_against_yolo


def extract_crops_with_full_lineage(
    csv_split_path: str,
    yolo_root: str,
    output_dir: str,
    taxonomy_json: str,
    skip_validation: bool = False,
):
    yolo_root = Path(yolo_root)
    output_dir = Path(output_dir)
    meta_df = pd.read_csv(csv_split_path).set_index("imagename")

    taxonomy = load_taxonomy(taxonomy_json)

    data_yaml = yolo_root / "data.yaml"
    if not skip_validation:
        validate_taxonomy_against_yolo(taxonomy, str(data_yaml), taxonomy_json_path=taxonomy_json)
        print(f"Taxonomy validated OK against {data_yaml}")

    manifest = []

    for split in ["train", "val"]:
        img_dir = yolo_root / "images" / split
        lbl_dir = yolo_root / "labels" / split

        for img_path in img_dir.glob("*.*"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w, _ = image.shape

            # Retrieve original CSV metadata row
            csv_row = meta_df.loc[img_path.name] if img_path.name in meta_df.index else None

            with open(lbl_path, "r") as f:
                lines = [line.strip().split() for line in f if line.strip()]

            for gt_idx, line in enumerate(lines):
                cls_id = str(line[0])
                if cls_id not in taxonomy["classes"]:
                    continue

                cls_info = taxonomy["classes"][cls_id]
                xc, yc, bw, bh = map(float, line[1:5])

                # Pixel coordinates
                xmin = max(0, int((xc - bw / 2) * w))
                ymin = max(0, int((yc - bh / 2) * h))
                xmax = min(w, int((xc + bw / 2) * w))
                ymax = min(h, int((yc + bh / 2) * h))

                crop = image[ymin:ymax, xmin:xmax]
                if crop.size == 0:
                    continue

                crop_filename = f"{img_path.stem}_gt{gt_idx}_{cls_info['name']}.png"
                save_path = output_dir / split / cls_info["name"] / crop_filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), crop)

                manifest.append({
                    "crop_filename": crop_filename,
                    "crop_path": str(save_path),
                    "split": split,
                    "imagename": img_path.name,
                    "gt_box_index": gt_idx,
                    "TLx": xmin,
                    "TLy": ymin,
                    "BRx": xmax,
                    "BRy": ymax,
                    "class_id": cls_id,
                    "species_label": cls_info["name"],
                    "latitude": csv_row["latitude"] if csv_row is not None else None,
                    "longitude": csv_row["longitude"] if csv_row is not None else None,
                    "bottom_depth": csv_row["bottom_depth"] if csv_row is not None else None
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(output_dir / "crop_manifest.csv", index=False)
    print(f"Extraction complete. {len(manifest_df)} crops saved with full metadata tracking.")


def main():
    parser = argparse.ArgumentParser(description="Extract Stage 2 training crops from a multi-class YOLO dataset.")
    parser.add_argument("--csv_split_path", required=True, help="e.g. crabdata2426/annotations/dataset_split_crab_multiclass.csv")
    parser.add_argument("--yolo_root", required=True, help="ORIGINAL multi-class yolo root, e.g. crabdata2426/yolo (NOT yolo_broad)")
    parser.add_argument("--output_dir", required=True, help="e.g. crabdata2426/crops")
    parser.add_argument("--taxonomy_json", required=True, help="e.g. config/crabdata_taxonomy.json")
    parser.add_argument("--skip_validation", action="store_true",
                         help="Skip the taxonomy/data.yaml cross-check. Not recommended.")
    args = parser.parse_args()

    extract_crops_with_full_lineage(
        csv_split_path=args.csv_split_path,
        yolo_root=args.yolo_root,
        output_dir=args.output_dir,
        taxonomy_json=args.taxonomy_json,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    main()
