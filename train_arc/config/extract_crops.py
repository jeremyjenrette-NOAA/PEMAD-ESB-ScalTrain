import cv2
import json
import pandas as pd
from pathlib import Path


def extract_crops_with_full_lineage(
    csv_split_path: str,
    yolo_root: str,
    output_dir: str,
    taxonomy_json: str
):
    yolo_root = Path(yolo_root)
    output_dir = Path(output_dir)
    meta_df = pd.read_csv(csv_split_path).set_index("imagename")

    with open(taxonomy_json, "r") as f:
        taxonomy = json.load(f)

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

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(output_dir / "crop_manifest.csv", index=False)
    print(f"Extraction complete. {len(manifest_df)} crops saved with full metadata tracking.")


if __name__ == "__main__":
    extract_crops_with_full_lineage(
        csv_split_path="star24/annotations/dataset_split_star.csv",
        yolo_root="star24/yolo",
        output_dir="star24/crops",
        taxonomy_json="config/asteroidea_taxonomy.json"
    )