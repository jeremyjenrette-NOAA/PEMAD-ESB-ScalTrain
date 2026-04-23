import os
import argparse
from pathlib import Path
import random
import shutil

import pandas as pd
from PIL import Image


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def bbox_to_yolo(x1, y1, x2, y2, w, h):
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
    if bw <= 1 or bh <= 1:
        return None

    xc = x_min + bw / 2.0
    yc = y_min + bh / 2.0

    # normalize to [0,1]
    return (xc / w, yc / h, bw / w, bh / h)

def parse_bool_series(x: pd.Series) -> pd.Series:
    """
    Robust boolean parser for values like:
    True, False, true, false, 1, 0, yes, no
    """
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False)

    return (
        x.astype(str)
         .str.strip()
         .str.lower()
         .map({
             "true": True,
             "false": False,
             "1": True,
             "0": False,
             "yes": True,
             "no": False,
             "y": True,
             "n": False
         })
         .fillna(False)
    )


def load_split_manifest(split_csv: Path, train_col: str, test_col: str) -> pd.DataFrame:
    sdf = pd.read_csv(split_csv)

    cols_lower = {c.lower(): c for c in sdf.columns}
    if "imagename" not in cols_lower:
        raise ValueError(
            f"{split_csv} missing 'imagename' column. Found: {list(sdf.columns)}"
        )

    img_col = cols_lower["imagename"]

    if train_col not in sdf.columns:
        raise ValueError(
            f"{split_csv} missing train column '{train_col}'. Found: {list(sdf.columns)}"
        )
    if test_col not in sdf.columns:
        raise ValueError(
            f"{split_csv} missing test column '{test_col}'. Found: {list(sdf.columns)}"
        )

    sdf = sdf[[img_col, train_col, test_col]].copy()
    sdf = sdf.rename(columns={img_col: "image"})
    sdf["image"] = sdf["image"].astype(str).str.strip()
    sdf[train_col] = parse_bool_series(sdf[train_col])
    sdf[test_col] = parse_bool_series(sdf[test_col])

    # keep only rows assigned to one of the two YOLO groups
    sdf = sdf[(sdf[train_col]) | (sdf[test_col])].copy()

    # sanity check: image cannot be both train and test
    bad = sdf[sdf[train_col] & sdf[test_col]]
    if not bad.empty:
        raise ValueError(
            f"{split_csv} contains {len(bad)} images marked TRUE for both "
            f"'{train_col}' and '{test_col}'."
        )

    sdf = sdf.drop_duplicates(subset=["image"])

    return sdf

def stratified_split_by_object_count(counts_df, seed=7, val_frac=0.1):
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


def load_point_exclusions_from_csv(
    ann_csv: Path,
    scope: str,
    scallop_class_ids: list[int],
) -> set[str]:
    """
    ann_csv must contain:
      - imagename (or Imagename)
      - geom_type  (values like point/line/bbox...)
    optionally:
      - class_id (if scope == 'scallop')
    """
    adf = pd.read_csv(ann_csv)

    # normalize column names
    cols = {c.lower(): c for c in adf.columns}
    if "imagename" not in cols:
        raise ValueError(f"{ann_csv} missing 'imagename' column. Found: {list(adf.columns)}")
    if "geom_type" not in cols:
        raise ValueError(f"{ann_csv} missing 'geom_type' column. Found: {list(adf.columns)}")

    img_col = cols["imagename"]
    geom_col = cols["geom_type"]

    adf[geom_col] = adf[geom_col].astype(str).str.strip().str.lower()
    adf[img_col] = adf[img_col].astype(str).str.strip()

    pts = adf[adf[geom_col] == "point"].copy()

    if scope == "any":
        return set(pts[img_col].dropna().unique().tolist())

    if scope == "scallop":
        # require class_id column
        if "class_id" not in cols:
            raise ValueError(
                f"--point_scope scallop requires 'class_id' column in {ann_csv}. "
                f"Found: {list(adf.columns)}"
            )
        cid_col = cols["class_id"]
        pts[cid_col] = pd.to_numeric(pts[cid_col], errors="coerce")
        pts = pts[pts[cid_col].isin(scallop_class_ids)].copy()
        return set(pts[img_col].dropna().unique().tolist())

    raise ValueError("--point_scope must be 'any' or 'scallop'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="groundtruth CSV with bboxes")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--label", default="scallop")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--link_mode", choices=["symlink", "hardlink", "copy"], default="symlink")

    ap.add_argument("--use_split_file", action="store_true",
                    help="If set, use an external CSV manifest to define train/test assignment.")
    ap.add_argument("--split_file_path", default=None,
                    help="Path to dataset split CSV containing imagename + boolean split columns.")
    ap.add_argument("--train_col", default="is_train",
                    help="Column in split CSV indicating YOLO training images.")
    ap.add_argument("--test_col", default="is_test",
                    help="Column in split CSV indicating YOLO test/validation images.")

    # negatives
    ap.add_argument("--neg_ratio", type=float, default=2.0,
                    help="Max negatives per positive image. Use -1 to include ALL negatives.")
    ap.add_argument("--neg_val_frac", type=float, default=None)

    # point exclusion using processed annotations CSV
    ap.add_argument("--ann_csv", help="processed annotations CSV with imagename + geom_type")
    ap.add_argument("--point_scope", choices=["any", "scallop"], default="any",
                    help="Exclude images with point annotations for any class, or only scallop class_ids (requires class_id column).")
    ap.add_argument(
        "--scallop_class_ids",
        default="185,515,197,207,920,213,912,916,525,919,215,915",
        help="Comma-separated class_id values treated as scallop in the processed annotation table."
    )
    args = ap.parse_args()

    if args.use_split_file:
        print("Note: --val_frac is ignored because split assignment comes from split_file_path")

    csv_path = Path(args.csv)
    img_dir = Path(args.img_dir)
    out_root = Path(args.out_root)
    ann_csv = Path(args.ann_csv)

    scallop_ids = [int(x.strip()) for x in args.scallop_class_ids.split(",") if x.strip()]

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

    # List ALL images available in img_dir
    all_images = sorted([p.name for p in img_dir.glob("*") if p.is_file()])
    all_set = set(all_images)
    if not all_images:
        raise ValueError(f"No image files found in img_dir: {img_dir}")

    # --- Load point exclusions from processed CSV ---
    point_excluded = load_point_exclusions_from_csv(
        ann_csv=ann_csv,
        scope=args.point_scope,
        scallop_class_ids=scallop_ids,
    )

    # Only exclude those that exist on disk
    point_excluded = set([x for x in point_excluded if x in all_set])

    # ---- Load bbox CSV (groundtruth) ----
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Imagename": "image", "ClassName": "label"})

    for col in ["image", "label", "TLx", "TLy", "BRx", "BRy"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'. Available: {list(df.columns)}")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["image"] = df["image"].astype(str).str.strip()
    print("Unique scallop images BEFORE ANY filtering:", df["image"].nunique())

    # Keep only target label rows
    df = df[df["label"] == args.label].copy()
    if df.empty:
        raise ValueError(f"No rows with label == {args.label}")

    print("Unique scallop images AFTER label filtering:", df["image"].nunique())
    # Parse bbox numeric + drop missing
    for c in ["TLx", "TLy", "BRx", "BRy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["TLx", "TLy", "BRx", "BRy"]).copy()
    if df.empty:
        raise ValueError("All rows dropped due to missing bbox coordinates.")

    print("Unique scallop images AFTER bbox filtering:", df["image"].nunique())
    # Keep only rows whose image exists AND is not point-excluded
    df = df[df["image"].isin(all_set)].copy()
    df = df[~df["image"].isin(point_excluded)].copy()
    if df.empty:
        raise ValueError("No remaining scallop bbox rows after disk + point exclusion filtering.")

    print("Unique scallop images AFTER exist and point-excl filtering:", df["image"].nunique())
    grouped = df.groupby("image")

    # ---- Build YOLO rows per image (no small-box filtering) ----
    rows_by_image = {}
    valid_counts = []
    n_bad_img_open = 0

    for fname, sub in grouped:
        src_img = img_dir / fname
        try:
            with Image.open(src_img) as im:
                w, h = im.size
        except Exception as e:
            print(f"WARNING: could not open image {src_img}: {e}")
            n_bad_img_open += 1
            continue

        rows = []
        for r in sub.itertuples(index=False):
            y = bbox_to_yolo(r.TLx, r.TLy, r.BRx, r.BRy, w, h)
            if y is None:
                continue
            xc, yc, bw, bh = y
            rows.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if rows:
            rows_by_image[fname] = rows
            valid_counts.append((fname, len(rows)))
            
        # if not rows:
        #     print(f"\nDROPPED IMAGE: {fname}")
        #     print(sub[["TLx", "TLy", "BRx", "BRy"]].head())
        #     print("Image size:", w, h)
        #     break

    total_images = len(grouped)
    kept_images = len(rows_by_image)
    dropped_images = total_images - kept_images

    print("Images entering bbox conversion:", total_images)
    print("Images with ≥1 valid box:", kept_images)
    print("Images dropped (no valid boxes):", dropped_images)

    counts_df = pd.DataFrame(valid_counts, columns=["image", "n"])
    if counts_df.empty:
        raise ValueError("No valid boxes remain after converting bboxes to YOLO format.")

    pos_set = set(rows_by_image.keys())

    # Exclude point images from *everything*
    usable_images = all_set - point_excluded

    # Negatives are images on disk, usable, and not positives
    neg_images_all = sorted(list(usable_images - pos_set))
    neg_val_frac = args.val_frac if args.neg_val_frac is None else args.neg_val_frac
    # ---- Split assignment ----
    if args.use_split_file:
        if args.split_file_path is None:
            raise ValueError("--use_split_file requires --split_file_path")

        split_csv = Path(args.split_file_path)
        split_df = load_split_manifest(
            split_csv=split_csv,
            train_col=args.train_col,
            test_col=args.test_col,
        )

        split_map = split_df.set_index("image")

        manifest_images = set(split_df["image"])
        manifest_images = manifest_images & usable_images

        train_manifest = set(
            split_df.loc[split_df[args.train_col], "image"].tolist()
        ) & usable_images

        val_manifest = set(
            split_df.loc[split_df[args.test_col], "image"].tolist()
        ) & usable_images

        # positives obey manifest directly
        train_pos = pos_set & train_manifest
        val_pos = pos_set & val_manifest

        # negatives also obey manifest directly
        train_neg_all = set(neg_images_all) & train_manifest
        val_neg_all = set(neg_images_all) & val_manifest

        train_neg = cap_negs(
            train_neg_all,
            pos_n=len(train_pos),
            ratio=args.neg_ratio,
            seed=args.seed + 1337
        )
        val_neg = cap_negs(
            val_neg_all,
            pos_n=len(val_pos),
            ratio=args.neg_ratio,
            seed=args.seed + 7331
        )

        print("\n=== Using external split manifest ===")
        print("split file:", split_csv.resolve())
        print("manifest images on disk and usable:", len(manifest_images))
        print("train manifest images:", len(train_manifest))
        print("val/test manifest images:", len(val_manifest))

        missing_from_manifest = sorted(list(usable_images - manifest_images))
        (out_root / "images_not_in_split_manifest.txt").write_text(
            "\n".join(missing_from_manifest) + ("\n" if missing_from_manifest else "")
        )

    else:
        # ---- Original internal splitting behavior ----
        train_pos, val_pos = stratified_split_by_object_count(
            counts_df, seed=args.seed, val_frac=args.val_frac
        )

        train_neg_all, val_neg_all = split_list(
            neg_images_all, seed=args.seed, val_frac=neg_val_frac
        )

        train_neg = cap_negs(
            train_neg_all, pos_n=len(train_pos), ratio=args.neg_ratio, seed=args.seed + 1337
        )
        val_neg = cap_negs(
            val_neg_all, pos_n=len(val_pos), ratio=args.neg_ratio, seed=args.seed + 7331
        )

    overlap = (train_pos | train_neg) & (val_pos | val_neg)
    if overlap:
        raise ValueError(f"Found {len(overlap)} images assigned to both train and val")

    print("\n=== Final split counts ===")
    print("train_pos:", len(train_pos))
    print("val_pos:", len(val_pos))
    print("train_neg:", len(train_neg))
    print("val_neg:", len(val_neg))
    print("train total:", len(train_pos) + len(train_neg))
    print("val total:", len(val_pos) + len(val_neg))

    # ---- Writers ----
    def write_one(split, fname, rows):
        src_img = img_dir / fname
        dst_img = (img_train if split == "train" else img_val) / fname
        dst_lab = (lab_train if split == "train" else lab_val) / (Path(fname).stem + ".txt")

        if rows:
            dst_lab.write_text("\n".join(rows) + "\n")
        else:
            dst_lab.write_text("")  # true negative

        link_image(src_img, dst_img, mode=args.link_mode)

    n_pos_written = 0
    n_neg_written = 0

    for fname in sorted(list(train_pos | val_pos)):
        split = "train" if fname in train_pos else "val"
        write_one(split, fname, rows_by_image[fname])
        n_pos_written += 1

    for fname in sorted(list(train_neg)):
        write_one("train", fname, [])
        n_neg_written += 1
    for fname in sorted(list(val_neg)):
        write_one("val", fname, [])
        n_neg_written += 1

    # Save excluded list for auditing
    (out_root / "excluded_point_images.txt").write_text("\n".join(sorted(point_excluded)) + "\n")

    # YAML
    yaml_text = f"""# label.yaml
path: {out_root.resolve()}
train: images/train
val: images/val

names:
  0: {args.label}
"""
    (out_root / "label.yaml").write_text(yaml_text)

    print("\n=== YOLO dataset build summary ===")
    print("img_dir:", img_dir.resolve())
    print("all images in img_dir:", len(all_images))
    print("point-excluded images (exist on disk):", len(point_excluded))
    print("positives (>=1 scallop box, not point-excluded):", len(pos_set))
    print("negatives available (usable, not positive, not point-excluded):", len(neg_images_all))
    print("neg_ratio (cap):", args.neg_ratio)
    print("val_frac:", args.val_frac, "neg_val_frac:", neg_val_frac)
    print("bad images that failed to open:", n_bad_img_open)
    print("Wrote excluded list:", out_root / "excluded_point_images.txt")
    print("Wrote YAML:", out_root / "label.yaml")


if __name__ == "__main__":
    main()