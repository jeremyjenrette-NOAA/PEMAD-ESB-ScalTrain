#!/usr/bin/env python3
"""
Step 5: Validate the staged YOLO dataset (+ optionally the VIAME truth JSONs)
for scallop2224 before kicking off training.

Checks:
  - every image opens and every label file matches an image (and vice versa)
  - every YOLO label line has 5 fields, class id in range, coords in [0,1]
  - per-split box/image counts (sanity numbers to eyeball)
  - VIAME truth JSON category ids line up with data.yaml class order
  - VIAME truth JSON boxes have positive area and fall within image bounds

For a random visual spot-check, use the existing vis_yolo.py against
<yolo-root>/labels/<split> and <yolo-root>/images/<split>.

Usage:
    python 05_validate_dataset.py \
        --yolo-root scallop2224/yolo \
        --viame-truth-dir scallop2224/viame_truth
"""
import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def validate_yolo_split(yolo_root: Path, split: str, n_classes: int, errors: list):
    img_dir = yolo_root / "images" / split
    lbl_dir = yolo_root / "labels" / split
    if not img_dir.exists() or not lbl_dir.exists():
        errors.append(f"[{split}] missing images/ or labels/ directory")
        return

    img_stems = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}

    only_imgs = set(img_stems) - lbl_stems
    only_lbls = lbl_stems - set(img_stems)
    if only_imgs:
        errors.append(f"[{split}] {len(only_imgs)} images have no label file, e.g. {sorted(only_imgs)[:5]}")
    if only_lbls:
        errors.append(f"[{split}] {len(only_lbls)} label files have no matching image, e.g. {sorted(only_lbls)[:5]}")

    n_boxes = 0
    n_empty_labels = 0
    for stem, img_path in img_stems.items():
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            errors.append(f"[{split}] unreadable image {img_path}: {e}")
            continue

        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            continue
        text = lbl_path.read_text().strip()
        if not text:
            n_empty_labels += 1
            continue

        for i, line in enumerate(text.splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"[{split}] {lbl_path.name}:{i} does not have 5 fields: {line!r}")
                continue
            try:
                cid = int(parts[0])
                xc, yc, w, h = (float(x) for x in parts[1:])
            except ValueError:
                errors.append(f"[{split}] {lbl_path.name}:{i} non-numeric field: {line!r}")
                continue
            if not (0 <= cid < n_classes):
                errors.append(f"[{split}] {lbl_path.name}:{i} class id {cid} out of range [0,{n_classes})")
            for name, v in [("xc", xc), ("yc", yc), ("w", w), ("h", h)]:
                if not (0.0 <= v <= 1.0):
                    errors.append(f"[{split}] {lbl_path.name}:{i} {name}={v} outside [0,1]")
            n_boxes += 1

    print(f"[{split}] {len(img_stems)} images, {n_boxes} boxes, {n_empty_labels} background/empty labels")


def validate_viame_truth(viame_truth_dir: Path, class_names: list, errors: list):
    for split_file, split_name in [("training_truth.json", "train"), ("validation_truth.json", "val")]:
        path = viame_truth_dir / split_file
        if not path.exists():
            errors.append(f"missing VIAME truth file: {path}")
            continue
        data = json.loads(path.read_text())

        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        expected = {i + 1: name for i, name in enumerate(class_names)}
        if cats != expected:
            errors.append(
                f"[{split_name}] VIAME category ids/names don't match data.yaml order:\n"
                f"  expected {expected}\n  got      {cats}"
            )

        images_by_id = {im["id"]: im for im in data.get("images", [])}
        n_bad_boxes = 0
        for ann in data.get("annotations", []):
            im = images_by_id.get(ann["image_id"])
            if im is None:
                errors.append(f"[{split_name}] annotation {ann['id']} references unknown image_id {ann['image_id']}")
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                n_bad_boxes += 1
                continue
            if x < 0 or y < 0 or x + w > im["width"] + 1 or y + h > im["height"] + 1:
                n_bad_boxes += 1
        if n_bad_boxes:
            errors.append(f"[{split_name}] {n_bad_boxes} annotations have non-positive area or fall outside image bounds")

        print(f"[{split_name}] VIAME truth: {len(data.get('images', []))} images, "
              f"{len(data.get('annotations', []))} annotations")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-root", required=True, type=Path)
    ap.add_argument("--viame-truth-dir", type=Path, default=None,
                    help="Optional: also validate training_truth.json / validation_truth.json")
    args = ap.parse_args()

    data_yaml = args.yolo_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    names = yaml.safe_load(data_yaml.read_text())["names"]
    class_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    n_classes = len(class_names)

    errors = []
    for split in ["train", "val"]:
        validate_yolo_split(args.yolo_root, split, n_classes, errors)

    if args.viame_truth_dir:
        validate_viame_truth(args.viame_truth_dir, class_names, errors)

    print("\n--- VALIDATION SUMMARY ---")
    if errors:
        print(f"{len(errors)} ERROR(S) FOUND:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
