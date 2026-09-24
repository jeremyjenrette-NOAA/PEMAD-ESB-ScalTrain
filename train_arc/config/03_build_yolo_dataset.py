#!/usr/bin/env python3
"""
Step 3: Build a YOLO-format dataset for scallop2224 from the staged images
and filtered groundtruth CSV. Produces an unstratified 90/10 train/val split
at the image level (matching gcp_yolo_job.sh / the SLURM job, which trains on
images/train and evaluates on images/val -- there is no separate test split).

Background (no-scallop) images from the empties directory are included as
negatives with empty label files unless --exclude-background is passed.

By default all scallop sub-categories in `class_name` (live, swimming,
width-line, clapper, etc.) are collapsed into a single YOLO class, "scallop"
(--class-mode single, the default). Pass --class-mode multiclass to instead
keep each distinct class_name value as its own class.

Usage:
python config/03_build_yolo_dataset.py \
    --groundtruth-csv scallop2224/annotations/groundtruth2224.csv \
    --annotated-images-dir scallop2224/viame/images \
    --empty-images-dir scallop2224/empties \
    --out-root scallop2224 \
    --train-ratio 0.995 --val-ratio 0.005 \
    --seed 42 --symlink
"""
import argparse
import random
import shutil
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def convert_bbox(tlx, tly, brx, bry, img_w, img_h):
    w = (brx - tlx) / img_w
    h = (bry - tly) / img_h
    xc = (tlx + brx) / (2.0 * img_w)
    yc = (tly + bry) / (2.0 * img_h)
    return (
        max(0.0, min(1.0, xc)),
        max(0.0, min(1.0, yc)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def clean_yolo_output(yolo_dir: Path):
    """
    Remove images/ and labels/ from a prior build entirely (not just the
    files this run happens to touch). Without this, images/labels left over
    from a previous run with a different split ratio, seed, or class-mode
    silently persist and get swept into whatever reads yolo/ next (Ultralytics
    training, build_viame_trainer_multiclass.py, etc.), inflating counts and
    mixing splits across runs.
    """
    for sub in ["images", "labels"]:
        d = yolo_dir / sub
        if d.exists():
            n_files = sum(1 for _ in d.rglob("*") if _.is_file() or _.is_symlink())
            print(f"[clean] removing existing {d} ({n_files} files from a previous build)")
            shutil.rmtree(d)


def build(
    groundtruth_csv: Path,
    annotated_images_dir: Path,
    empty_images_dir: Path,
    out_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    use_symlinks: bool,
    include_background: bool,
    class_mode: str,
    single_class_name: str,
    clean: bool = True,
):
    if abs((train_ratio + val_ratio) - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + val_ratio must sum to 1.0 (no test split; val doubles "
            "as the held-out evaluation set, matching the existing job scripts)"
        )

    yolo_dir = out_root / "yolo"
    if clean:
        clean_yolo_output(yolo_dir)

    df = pd.read_csv(groundtruth_csv, low_memory=False)

    if class_mode == "single":
        source_classes = sorted(df["class_name"].unique())
        class_to_id = {c: 0 for c in source_classes}
        id_to_class = {0: single_class_name}
        print(f"Collapsing {len(source_classes)} scallop sub-classes into a single "
              f"class '{single_class_name}':")
        for c in source_classes:
            print(f"  {c!r} -> 0 ({single_class_name})")
    else:
        unique_classes = sorted(df["class_name"].unique())
        class_to_id = {c: i for i, c in enumerate(unique_classes)}
        id_to_class = {i: c for i, c in enumerate(unique_classes)}
        print("Class mapping:")
        for i, c in id_to_class.items():
            print(f"  {i}: {c}")

    annotated_images = {p.name for p in annotated_images_dir.glob("*") if p.suffix.lower() in IMG_EXTS}
    background_images = set()
    if include_background and empty_images_dir.exists():
        background_images = {p.name for p in empty_images_dir.glob("*") if p.suffix.lower() in IMG_EXTS}

    annotated_in_csv = set(df["imagename"].unique())
    missing_from_disk = annotated_in_csv - annotated_images
    if missing_from_disk:
        print(f"[WARN] {len(missing_from_disk)} annotated images in CSV not found on disk "
              f"under {annotated_images_dir}; they will be skipped.")

    all_images = sorted((annotated_images & annotated_in_csv) | background_images)
    if not all_images:
        raise RuntimeError("No images resolved to build the dataset from.")

    rng = random.Random(seed)
    rng.shuffle(all_images)
    n_total = len(all_images)
    n_train = round(n_total * train_ratio)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:],
    }

    for split in splits:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    ann_by_image = {name: g for name, g in df.groupby("imagename")}

    counts = {"train": {"images": 0, "boxes": 0}, "val": {"images": 0, "boxes": 0}}
    for split, img_list in splits.items():
        for img_name in img_list:
            src_dir = annotated_images_dir if img_name in annotated_images else empty_images_dir
            src_path = src_dir / img_name
            if not src_path.exists():
                print(f"[WARN] expected image missing: {src_path}")
                continue

            dst_img = yolo_dir / "images" / split / img_name
            label_path = yolo_dir / "labels" / split / f"{Path(img_name).stem}.txt"

            with Image.open(src_path) as im:
                img_w, img_h = im.size

            lines = []
            for _, row in ann_by_image.get(img_name, pd.DataFrame()).iterrows():
                cid = class_to_id[row["class_name"]]
                xc, yc, w, h = convert_bbox(
                    float(row["tlx"]), float(row["tly"]),
                    float(row["brx"]), float(row["bry"]),
                    img_w, img_h,
                )
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            label_path.write_text("\n".join(lines))

            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            if use_symlinks:
                dst_img.symlink_to(src_path.resolve())
            else:
                shutil.copy2(src_path, dst_img)

            counts[split]["images"] += 1
            counts[split]["boxes"] += len(lines)

    yaml_data = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": id_to_class,
    }
    yaml_path = yolo_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print("\n--- YOLO BUILD SUMMARY ---")
    for split, c in counts.items():
        print(f"{split}: {c['images']} images, {c['boxes']} boxes")
    print(f"Wrote: {yaml_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groundtruth-csv", required=True, type=Path)
    ap.add_argument("--annotated-images-dir", required=True, type=Path)
    ap.add_argument("--empty-images-dir", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--exclude-background", action="store_true")
    ap.add_argument("--class-mode", choices=["single", "multiclass"], default="single",
                     help="'single' (default) collapses all scallop sub-classes into one "
                          "class; 'multiclass' keeps each class_name as its own class")
    ap.add_argument("--single-class-name", default="scallop",
                     help="Class name to use when --class-mode single (default: scallop)")
    ap.add_argument("--no-clean", action="store_true",
                     help="Do NOT wipe existing yolo/images and yolo/labels before building. "
                          "Off by default -- leaving this off is what causes stale images/labels "
                          "from a previous run (different ratio/seed/class-mode) to linger and "
                          "get counted alongside the new split.")
    args = ap.parse_args()

    build(
        groundtruth_csv=args.groundtruth_csv,
        annotated_images_dir=args.annotated_images_dir,
        empty_images_dir=args.empty_images_dir,
        out_root=args.out_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_symlinks=args.symlink,
        include_background=not args.exclude_background,
        class_mode=args.class_mode,
        single_class_name=args.single_class_name,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
