import argparse
from pathlib import Path
import random
import shutil
from PIL import Image
import pandas as pd
import yaml


def convert_viame_to_yolo_bbox(
    tl_x: float, tl_y: float, br_x: float, br_y: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Converts absolute pixel coordinates (tl_x, tl_y, br_x, br_y) to normalized

    YOLO format (x_center, y_center, width, height).
    """
    w = (br_x - tl_x) / img_w
    h = (br_y - tl_y) / img_h
    x_center = (tl_x + br_x) / (2.0 * img_w)
    y_center = (tl_y + br_y) / (2.0 * img_h)

    # Clamp values to strictly stay within [0.0, 1.0]
    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def build_yolo_dataset(
    csv_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    use_symlinks: bool = False,
    include_background: bool = True,
):
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    yolo_dir = Path(output_dir) / "yolo"

    # Validate split ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-5:
        raise ValueError(f"Ratios must sum to 1.0 (current sum: {total_ratio})")

    if not csv_path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {csv_path}")

    # Read sanitized CSV
    df = pd.read_csv(csv_path)

    # Map dynamic class labels to contiguous integer IDs
    unique_classes = sorted(df["class_label"].unique())
    class_to_id = {cls_name: i for i, cls_name in enumerate(unique_classes)}
    id_to_class = {i: cls_name for i, cls_name in enumerate(unique_classes)}

    print(f"Class Mapping: {class_to_id}")

    # Discover all candidate images
    annotated_images = set(df["image_filename"].unique())
    all_disk_images = {p.name for p in images_dir.glob("*.jpg")} | {
        p.name for p in images_dir.glob("*.png")
    }

    if include_background:
        target_images = sorted(list(all_disk_images))
    else:
        target_images = sorted(list(annotated_images & all_disk_images))

    # Shuffle and split by unique image filename
    random.seed(seed)
    random.shuffle(target_images)

    n_total = len(target_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_imgs = target_images[:n_train]
    val_imgs = target_images[n_train : n_train + n_val]
    test_imgs = target_images[n_train + n_val :]

    splits = {"train": train_imgs, "val": val_imgs}
    if test_ratio > 0:
        splits["test"] = test_imgs

    # Prepare directory paths
    for split in splits.keys():
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process files per split
    for split, img_list in splits.items():
        print(f"Processing split: '{split}' ({len(img_list)} images)...")

        for img_name in img_list:
            src_img_path = images_dir / img_name
            dst_img_path = yolo_dir / "images" / split / img_name
            label_txt_path = (
                yolo_dir / "labels" / split / f"{Path(img_name).stem}.txt"
            )

            if not src_img_path.exists():
                print(f"  [Warning] Missing image file: {src_img_path}")
                continue

            # Get dimensions using PIL without loading heavy array buffers
            with Image.open(src_img_path) as img:
                img_w, img_h = img.size

            # Extract image annotations
            img_anns = df[df["image_filename"] == img_name]
            yolo_lines = []

            for _, row in img_anns.iterrows():
                class_id = class_to_id[str(row["class_label"]).strip()]
                x_c, y_c, w, h = convert_viame_to_yolo_bbox(
                    float(row["tl_x"]),
                    float(row["tl_y"]),
                    float(row["br_x"]),
                    float(row["br_y"]),
                    img_w,
                    img_h,
                )
                yolo_lines.append(
                    f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                )

            # Write label file (empty text file for background images)
            with open(label_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            # Transfer image file
            if dst_img_path.exists() or dst_img_path.is_symlink():
                dst_img_path.unlink()

            if use_symlinks:
                dst_img_path.symlink_to(src_img_path.resolve())
            else:
                shutil.copy2(src_img_path, dst_img_path)

    # Create data.yaml config for Ultralytics
    yaml_data = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": id_to_class,
    }
    if test_ratio > 0:
        yaml_data["test"] = "images/test"

    yaml_file_path = yolo_dir / "data.yaml"
    with open(yaml_file_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print("\nDataset Build Complete!")
    print(f"Saved YOLO dataset configuration to: {yaml_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build YOLO training dataset from VIAME annotations CSV."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="sealdata26",
        help="Root path to target dataset directory (e.g. sealdata26)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying them to save space",
    )
    parser.add_argument(
        "--exclude-background",
        action="store_true",
        help="Exclude images without annotations",
    )

    args = parser.parse_args()

    root = Path(args.dataset_dir)
    csv_p = root / "annotations" / "groundtruth.csv"
    img_p = root / "viame" / "images"

    build_yolo_dataset(
        csv_path=csv_p,
        images_dir=img_p,
        output_dir=root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        use_symlinks=args.symlink,
        include_background=not args.exclude_background,
    )