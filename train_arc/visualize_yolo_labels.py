import argparse
from pathlib import Path
import cv2
import yaml


def visualize_yolo_dataset(
    yolo_dir: str | Path,
    split: str = "train",
    num_samples: int = 10,
    output_dir: str | Path = None,
):
    yolo_path = Path(yolo_dir)
    yaml_path = yolo_path / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Could not find data.yaml at: {yaml_path}")

    # Load class mapping from data.yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    class_names = data_config.get("names", {})
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    img_dir = yolo_path / "images" / split
    lbl_dir = yolo_path / "labels" / split

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {img_dir}")

    out_dir = (
        Path(output_dir)
        if output_dir
        else yolo_path / "visualizations" / split
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        [
            p
            for p in img_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )

    if not img_files:
        print(f"No images found in {img_dir}")
        return

    sample_imgs = img_files[:num_samples] if num_samples > 0 else img_files
    print(
        f"Visualizing {len(sample_imgs)} sample(s) from split '{split}' into '{out_dir}'..."
    )

    for img_file in sample_imgs:
        lbl_file = lbl_dir / f"{img_file.stem}.txt"

        image = cv2.imread(str(img_file))
        if image is None:
            print(f"  [Warning] Could not read image: {img_file.name}")
            continue

        img_h, img_w = image.shape[:2]

        if not lbl_file.exists():
            print(f"  [Info] No label file found for {img_file.name}")
            continue

        with open(lbl_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        box_count = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            # Denormalize YOLO coordinates back to absolute image pixels
            tl_x = int((x_center - w / 2.0) * img_w)
            tl_y = int((y_center - h / 2.0) * img_h)
            br_x = int((x_center + w / 2.0) * img_w)
            br_y = int((y_center + h / 2.0) * img_h)

            class_name = class_names.get(class_id, f"class_{class_id}")

            # Draw bounding box (Green)
            color = (0, 255, 0)
            cv2.rectangle(image, (tl_x, tl_y), (br_x, br_y), color, thickness=3)

            # Draw label banner
            text = f"{class_name} ({class_id})"
            label_y = max(20, tl_y - 10)
            cv2.putText(
                image,
                text,
                (tl_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            box_count += 1

        out_file = out_dir / f"check_{img_file.name}"
        cv2.imwrite(str(out_file), image)
        print(f"  Saved: {out_file.name} ({box_count} boxes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize converted YOLO labels on dataset images."
    )
    parser.add_argument(
        "--yolo-dir",
        type=str,
        default="sealdata26/yolo",
        help="Path to yolo/ root directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to check",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sample images to render (-1 for all)",
    )

    args = parser.parse_args()

    visualize_yolo_dataset(
        yolo_dir=args.yolo_dir, split=args.split, num_samples=args.num_samples
    )