import argparse
from pathlib import Path

def convert_labels_to_broad(src_lbl_dir: Path, dst_lbl_dir: Path, target_class_id: int = 0):
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for lbl_file in src_lbl_dir.glob("*.txt"):
        with open(lbl_file, "r") as f:
            lines = f.readlines()

        broad_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            # Remap class ID (parts[0]) to single broad target class (0)
            parts[0] = str(target_class_id)
            broad_lines.append(" ".join(parts) + "\n")

        with open(dst_lbl_dir / lbl_file.name, "w") as f:
            f.writelines(broad_lines)

def main():
    parser = argparse.ArgumentParser(description="Remap multi-class YOLO annotations to broad single-class.")
    parser.add_argument("--src_yolo", type=str, default="yolo", help="Source multi-class YOLO directory")
    parser.add_argument("--dst_yolo", type=str, default="yolo_broad", help="Destination broad YOLO directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    src_root = Path(args.src_yolo)
    dst_root = Path(args.dst_yolo)

    for split in args.splits:
        print(f"Converting split '{split}' to broad target class...")
        src_img = src_root / "images" / split
        dst_img = dst_root / "images" / split
        dst_img.mkdir(parents=True, exist_ok=True)

        # Symlink image files to avoid duplicating disk usage
        for img in src_img.glob("*.*"):
            link_path = dst_img / img.name
            if not link_path.exists():
                link_path.symlink_to(img.resolve())

        # Remap text label files
        src_lbl = src_root / "labels" / split
        dst_lbl = dst_root / "labels" / split
        convert_labels_to_broad(src_lbl, dst_lbl, target_class_id=0)

    # Create data_broad.yaml for YOLO training
    yaml_content = f"""path: {dst_root.resolve()}
train: images/train
val: images/val

names:
  0: cancer_crab
"""
    with open(dst_root / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Broad YOLO dataset successfully created at: {dst_root}")

if __name__ == "__main__":
    main()