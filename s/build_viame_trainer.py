#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def find_image_for_stem(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def index_images_recursive(root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            index.setdefault(p.stem, []).append(p.resolve())
    return index


def resolve_original_image(
    yolo_img_path: Path,
    viame_index: Optional[Dict[str, List[Path]]] = None
) -> Path:
    """
    Prefer the original resolved path if the YOLO image is a symlink.
    Otherwise, if a viame image index is provided, try to map by basename stem.
    Fallback to the YOLO image path itself.
    """
    try:
        real = yolo_img_path.resolve()
        if real.exists():
            return real
    except Exception:
        pass

    if viame_index is not None:
        matches = viame_index.get(yolo_img_path.stem, [])
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous basename match for stem '{yolo_img_path.stem}'. "
                f"Matches:\n" + "\n".join(str(x) for x in matches)
            )

    return yolo_img_path.resolve()


def yolo_to_coco_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int
) -> Tuple[float, float, float, float]:
    w = width * img_w
    h = height * img_h
    x = (x_center * img_w) - (w / 2.0)
    y = (y_center * img_h) - (h / 2.0)

    # clamp lightly to image bounds
    x = max(0.0, min(x, float(img_w)))
    y = max(0.0, min(y, float(img_h)))
    w = max(0.0, min(w, float(img_w) - x))
    h = max(0.0, min(h, float(img_h) - y))

    return x, y, w, h


def parse_yolo_label_file(
    label_path: Path,
    img_w: int,
    img_h: int,
    category_id_lookup: Dict[int, int]
) -> List[Dict]:
    anns = []
    # Gracefully handle missing label files (treating them as empty images)
    if not label_path.exists():
        return anns

    text = label_path.read_text().strip()
    if not text:
        return anns

    for line_num, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Malformed YOLO label in {label_path} line {line_num}: {line}"
            )

        cls_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        if cls_id not in category_id_lookup:
            raise KeyError(
                f"YOLO class id {cls_id} not present in category lookup "
                f"for file {label_path}"
            )

        x, y, w, h = yolo_to_coco_bbox(
            x_center, y_center, width, height, img_w, img_h
        )

        anns.append({
            "category_id": category_id_lookup[cls_id],
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
        })

    return anns


def build_coco_split(
    yolo_root: Path,
    split: str,
    categories: List[Dict],
    viame_index: Optional[Dict[str, List[Path]]] = None
) -> Dict:
    labels_dir = yolo_root / "labels" / split
    images_dir = yolo_root / "images" / split

    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    # FLIP LOGIC: Read physical images as the source of truth, not labels
    image_files = []
    for ext in IMG_EXTS:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
    image_files = sorted(list(set(image_files)))

    if not image_files:
        raise RuntimeError(f"No images found in {images_dir}")

    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    yolo_to_coco_cat = {i: cat["id"] for i, cat in enumerate(categories)}

    image_id = 1
    ann_id = 1

    for yolo_img_path in image_files:
        stem = yolo_img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        
        actual_img_path = resolve_original_image(yolo_img_path, viame_index)

        try:
            with Image.open(actual_img_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            print(f"Warning: Skipping {actual_img_path} due to read error: {e}")
            continue

        coco["images"].append({
            "id": image_id,
            "file_name": str(actual_img_path),
            "width": img_w,
            "height": img_h,
        })

        anns = parse_yolo_label_file(
            label_path=label_path,
            img_w=img_w,
            img_h=img_h,
            category_id_lookup=yolo_to_coco_cat
        )

        for ann in anns:
            ann["id"] = ann_id
            ann["image_id"] = image_id
            coco["annotations"].append(ann)
            ann_id += 1

        image_id += 1

    return coco


def main():
    parser = argparse.ArgumentParser(
        description="Build fixed VIAME/NetHarn train/val JSONs from a YOLO split."
    )
    parser.add_argument(
        "--yolo-root",
        required=True,
        help="Root of YOLO dataset, e.g. .../data2022/yolo_test"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where training_truth.json and validation_truth.json are written"
    )
    parser.add_argument(
        "--viame-image-root",
        default=None,
        help=(
            "Optional root of original VIAME images to map basenames back to. "
            "If omitted, the script uses resolved YOLO image paths."
        )
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["scallop"],
        help="Class names in YOLO order, e.g. scallop"
    )

    args = parser.parse_args()

    yolo_root = Path(args.yolo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    viame_index = None
    if args.viame_image_root:
        viame_root = Path(args.viame_image_root).resolve()
        print(f"Indexing original VIAME images under: {viame_root}")
        viame_index = index_images_recursive(viame_root)
        print(f"Indexed {sum(len(v) for v in viame_index.values())} images")

    categories = [
        {"id": i + 1, "name": name}
        for i, name in enumerate(args.class_names)
    ]

    print("Building training_truth.json")
    train_coco = build_coco_split(
        yolo_root=yolo_root,
        split="train",
        categories=categories,
        viame_index=viame_index
    )

    print("Building validation_truth.json")
    val_coco = build_coco_split(
        yolo_root=yolo_root,
        split="val",
        categories=categories,
        viame_index=viame_index
    )

    train_out = output_dir / "training_truth.json"
    val_out = output_dir / "validation_truth.json"

    with open(train_out, "w") as f:
        json.dump(train_coco, f, indent=2)

    with open(val_out, "w") as f:
        json.dump(val_coco, f, indent=2)

    print(f"\n--- VIAME JSON BUILD SUMMARY ---")
    print(f"Wrote: {train_out}")
    print(f"Wrote: {val_out}")
    print(f"Train images: {len(train_coco['images'])}")
    print(f"Train annotations: {len(train_coco['annotations'])}")
    print(f"Val images: {len(val_coco['images'])}")
    print(f"Val annotations: {len(val_coco['annotations'])}")


if __name__ == "__main__":
    main()