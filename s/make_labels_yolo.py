import os
import argparse
from pathlib import Path
import random
import shutil

import pandas as pd
from PIL import Image


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def bbox_to_yolo(x1, y1, x2, y2, w, h, min_wh_px=4.0):
    # clip to image bounds
    x1 = clamp(float(x1), 0, w - 1)
    y1 = clamp(float(y1), 0, h - 1)
    x2 = clamp(float(x2), 0, w - 1)
    y2 = clamp(float(y2), 0, h - 1)

    # order corners
    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= min_wh_px or bh <= min_wh_px:
        return None

    xc = x_min + bw / 2.0
    yc = y_min + bh / 2.0

    # normalize to [0,1]
    return (xc / w, yc / h, bw / w, bh / h)


def stratified_split_by_object_count(counts_df, seed=7, val_frac=0.1):
    """
    counts_df must have columns: image, n (objects per image)
    """
    counts = counts_df.copy()
    q = min(5, max(2, counts["n"].nunique()))
    counts["bin"] = pd.qcut(counts["n"], q=q, duplicates="drop")

    rng = random.Random(seed)
    val_images = set()
    for _, sub in counts.groupby("bin"):
        imgs = sub["image"].tolist()
        rng.shuffle(imgs)
        k = max(1, int(round(len(imgs) * val_frac)))
        val_images.update(imgs[:k])

    train_images = set(counts["image"]) - val_images
    return train_images, val_images


def split_list(items, seed=7, val_frac=0.1):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    k = int(round(len(items) * val_frac))
    return set(items[k:]), set(items[:k])  # train, val


def link_image(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def cap_negs(neg_set, pos_n, ratio, seed):
    if ratio < 0:
        return set(neg_set)  # all
    cap = int(round(pos_n * ratio))
    neg_list = list(neg_set)
    rng = random.Random(seed)
    rng.shuffle(neg_list)
    return set(neg_list[:min(len(neg_list), cap)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--label", default="scallop")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--link_mode", choices=["symlink", "hardlink", "copy"], default="symlink")

    # negatives control
    ap.add_argument("--neg_ratio", type=float, default=2.0,
                    help="Max negatives per positive image. Use -1 to include ALL negatives.")
    ap.add_argument("--neg_val_frac", type=float, default=None,
                    help="Override val_frac for negatives only.")

    # tiny-box filtering (for point->box artifacts)
    ap.add_argument("--min_wh_px", type=float, default=4.0,
                    help="Minimum bbox width/height in pixels. Increase to filter tiny point-derived boxes.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    img_dir = Path(args.img_dir)
    out_root = Path(args.out_root)

    if out_root.exists():
        if not args.force:
            raise FileExistsError(f"{out_root} exists. Re-run with --force to overwrite.")
        shutil.rmtree(out_root)

    # Output structure
    img_train = out_root / "images" / "train"
    img_val   = out_root / "images" / "val"
    lab_train = out_root / "labels" / "train"
    lab_val   = out_root / "labels" / "val"
    for p in [img_train, img_val, lab_train, lab_val]:
        p.mkdir(parents=True, exist_ok=True)

    # ---- Load + standardize CSV ----
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Imagename": "image", "ClassName": "label"})

    for col in ["image", "label", "TLx", "TLy", "BRx", "BRy"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'. Available: {list(df.columns)}")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["image"] = df["image"].astype(str)

    # Keep only target label rows
    df = df[df["label"] == args.label].copy()
    if df.empty:
        raise ValueError(f"No rows with label == {args.label}")

    # Parse bbox numeric + drop missing
    for c in ["TLx", "TLy", "BRx", "BRy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["TLx", "TLy", "BRx", "BRy"]).copy()
    if df.empty:
        raise ValueError("All rows dropped due to missing bbox coordinates.")

    # List ALL images available in img_dir
    all_images = sorted([p.name for p in img_dir.glob("*") if p.is_file()])
    all_set = set(all_images)
    if not all_images:
        raise ValueError(f"No image files found in img_dir: {img_dir}")

    # Keep only bbox rows whose image exists on disk
    df = df[df["image"].isin(all_set)].copy()
    if df.empty:
        raise ValueError("No remaining scallop rows after filtering to existing images.")

    grouped = df.groupby("image")

    # ---- Precompute YOLO rows per image AFTER small-box filtering ----
    rows_by_image = {}
    valid_counts = []
    tiny_only_images = 0

    for fname, sub in grouped:
        src_img = img_dir / fname
        try:
            with Image.open(src_img) as im:
                w, h = im.size
        except Exception as e:
            print(f"WARNING: could not open image {src_img}: {e}")
            continue

        rows = []
        for r in sub.itertuples(index=False):
            y = bbox_to_yolo(r.TLx, r.TLy, r.BRx, r.BRy, w, h, min_wh_px=args.min_wh_px)
            if y is None:
                continue
            xc, yc, bw, bh = y
            rows.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if rows:
            rows_by_image[fname] = rows
            valid_counts.append((fname, len(rows)))
        else:
            tiny_only_images += 1  # had scallop rows, but all filtered out as tiny/invalid

    counts_df = pd.DataFrame(valid_counts, columns=["image", "n"])
    if counts_df.empty:
        raise ValueError("No valid boxes remain after min_wh_px filtering. Try lowering --min_wh_px.")

    # Final positives are ONLY those with >=1 valid box
    pos_set = set(rows_by_image.keys())

    # Final negatives are images on disk that are not in pos_set
    excluded_images = set(grouped.groups.keys()) - set(rows_by_image.keys())
    neg_images_all = sorted(list(all_set - pos_set - excluded_images))

    # ---- Split positives (stratified by valid scallop count) ----
    train_pos, val_pos = stratified_split_by_object_count(counts_df, seed=args.seed, val_frac=args.val_frac)

    # ---- Split negatives (random) then cap by ratio using FINAL pos counts ----
    neg_val_frac = args.val_frac if args.neg_val_frac is None else args.neg_val_frac
    train_neg_all, val_neg_all = split_list(neg_images_all, seed=args.seed, val_frac=neg_val_frac)

    train_neg = cap_negs(train_neg_all, pos_n=len(train_pos), ratio=args.neg_ratio, seed=args.seed + 1337)
    val_neg   = cap_negs(val_neg_all,   pos_n=len(val_pos),   ratio=args.neg_ratio, seed=args.seed + 7331)

    # ---- Writers ----
    def write_one(split, fname, rows):
        src_img = img_dir / fname
        dst_img = (img_train if split == "train" else img_val) / fname
        dst_lab = (lab_train if split == "train" else lab_val) / (Path(fname).stem + ".txt")

        # labels always written (empty file = true negative)
        if rows:
            dst_lab.write_text("\n".join(rows) + "\n")
        else:
            dst_lab.write_text("")

        link_image(src_img, dst_img, mode=args.link_mode)

    n_pos_written = 0
    n_neg_written = 0

    # positives
    for fname in sorted(list(train_pos | val_pos)):
        split = "train" if fname in train_pos else "val"
        write_one(split, fname, rows_by_image[fname])
        n_pos_written += 1

    # negatives
    for fname in sorted(list(train_neg)):
        write_one("train", fname, [])
        n_neg_written += 1
    for fname in sorted(list(val_neg)):
        write_one("val", fname, [])
        n_neg_written += 1

    # YAML
    yaml_text = f"""# scallop.yaml
path: {out_root.resolve()}
train: images/train
val: images/val

names:
  0: {args.label}
"""
    (out_root / "scallop.yaml").write_text(yaml_text)

    # ---- Audit invariants ----
    def stems_in_files(dirpath: Path):
        return sorted({p.stem for p in dirpath.glob("*") if p.is_file()})

    train_img_stems = stems_in_files(img_train)
    val_img_stems   = stems_in_files(img_val)
    train_lab_stems = sorted({p.stem for p in lab_train.glob("*.txt")})
    val_lab_stems   = sorted({p.stem for p in lab_val.glob("*.txt")})

    def diff(a, b):
        return sorted(set(a) - set(b))

    orphan_train_imgs = diff(train_img_stems, train_lab_stems)
    orphan_val_imgs   = diff(val_img_stems, val_lab_stems)
    orphan_train_labs = diff(train_lab_stems, train_img_stems)
    orphan_val_labs   = diff(val_lab_stems, val_img_stems)

    print("\n=== YOLO dataset build summary ===")
    print("img_dir:", img_dir.resolve())
    print("all images in img_dir:", len(all_images))
    print("positives (>=1 valid box after filter):", len(pos_set))
    print("negatives available:", len(neg_images_all))
    print("scallop-annotated images that became tiny-only after filter (not positives):", tiny_only_images)
    print("neg_ratio (cap):", args.neg_ratio)
    print("val_frac:", args.val_frac, "neg_val_frac:", neg_val_frac)
    print("min_wh_px:", args.min_wh_px)

    print("\nSelected for TRAIN:")
    print("  positives:", len(train_pos), "negatives:", len(train_neg), "total:", len(train_pos) + len(train_neg))
    print("Selected for VAL:")
    print("  positives:", len(val_pos), "negatives:", len(val_neg), "total:", len(val_pos) + len(val_neg))

    print("\nWritten:")
    print("  positives:", n_pos_written)
    print("  negatives:", n_neg_written)

    if orphan_train_imgs or orphan_val_imgs or orphan_train_labs or orphan_val_labs:
        print("\nWARNING: Found mismatches (should be empty lists).")
        print("orphan_train_imgs:", orphan_train_imgs[:10])
        print("orphan_val_imgs:", orphan_val_imgs[:10])
        print("orphan_train_labs:", orphan_train_labs[:10])
        print("orphan_val_labs:", orphan_val_labs[:10])
    else:
        print("\nAudit OK: 1:1 image↔label mapping in train and val.")

    print("\nWrote YAML:", out_root / "scallop.yaml")


if __name__ == "__main__":
    main()