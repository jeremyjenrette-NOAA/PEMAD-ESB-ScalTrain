import os
import json
import random
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def build_coco(df, img_dir: Path, categories, image_ids):
    """
    df must contain columns: image, TLx, TLy, BRx, BRy, label
    image_ids: dict {filename -> int}
    """
    # COCO requires:
    # images: [{id, file_name, width, height}]
    # annotations: [{id, image_id, category_id, bbox[x,y,w,h], area, iscrowd}]
    # categories: [{id, name}]
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cid, "name": name} for name, cid in categories.items()],
    }

    img_meta_done = set()
    ann_id = 1

    for row in df.itertuples(index=False):
        fname = getattr(row, "image")
        label = getattr(row, "label")

        # only keep known labels
        if label not in categories:
            continue

        path = img_dir / fname
        if not path.exists():
            # skip missing images
            continue

        img_id = image_ids[fname]

        if fname not in img_meta_done:
            with Image.open(path) as im:
                w, h = im.size
            coco["images"].append({
                "id": img_id,
                "file_name": fname,
                "width": w,
                "height": h
            })
            img_meta_done.add(fname)
        else:
            # we still need w/h for clipping; load from existing entry (small)
            # but easiest: keep a dict cache in real pipelines
            # here, re-open is okay for a first run
            with Image.open(path) as im:
                w, h = im.size

        # read coords
        x1 = float(getattr(row, "TLx"))
        y1 = float(getattr(row, "TLy"))
        x2 = float(getattr(row, "BRx"))
        y2 = float(getattr(row, "BRy"))

        # clip to bounds (0..w-1, 0..h-1)
        x1 = clamp(x1, 0, w - 1)
        y1 = clamp(y1, 0, h - 1)
        x2 = clamp(x2, 0, w - 1)
        y2 = clamp(y2, 0, h - 1)

        # enforce ordering
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

        bw = x_max - x_min
        bh = y_max - y_min

        # drop invalid boxes
        if bw <= 1 or bh <= 1:
            continue

        coco_bbox = [x_min, y_min, bw, bh]  # COCO: x,y,width,height
        area = bw * bh

        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": categories[label],
            "bbox": [round(v, 3) for v in coco_bbox],
            "area": round(area, 3),
            "iscrowd": 0
        })
        ann_id += 1

    return coco

def stratified_split_by_object_count(df, seed=7, val_frac=0.1):
    """
    Cluster-friendly stratification:
    - compute scallop count per image
    - bin into quantiles (so busy images don't all land in train or val)
    """
    counts = df.groupby("image").size().reset_index(name="n")
    # quantile bins; handle small datasets gracefully
    q = min(5, max(2, counts["n"].nunique()))
    counts["bin"] = pd.qcut(counts["n"], q=q, duplicates="drop")

    rng = random.Random(seed)
    val_images = set()

    for b, sub in counts.groupby("bin"):
        imgs = sub["image"].tolist()
        rng.shuffle(imgs)
        k = max(1, int(round(len(imgs) * val_frac)))
        val_images.update(imgs[:k])

    train_images = set(counts["image"]) - val_images
    return train_images, val_images

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022, help="Select survey year")
    args = ap.parse_args()
    # ---- EDIT THESE ----
    csv_path = Path("../data/raw/parsedann2224.csv")
    img_dir  = Path("/Volumes/PortableSSD/saltnoaa/images/2022tr_split")
    out_dir  = Path("../data/processed")
    seed     = 7
    val_frac = 0.10
    target_label = "scallop"   # <--- enforce lowercase target
    # -------------------

    df = pd.read_csv(csv_path)

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[(df["anntype"] != "point") & (df["year"] == args.year)].copy()

    # Normalize columns
    rename_map = {
        "image": "image",
        "TLx": "TLx",
        "TLy": "TLy",
        "BRx": "BRx",
        "BRy": "BRy",
        "label": "label",
    }
    df = df.rename(columns=rename_map)

    keep = ["image", "TLx", "TLy", "BRx", "BRy", "label"]
    
    df = df[keep].copy()

    df["image"] = df["image"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # ------------------------------------------------------
    # 🔹 FILTER FOR SCALLOP ONLY
    # ------------------------------------------------------
    df = df[df["label"] == target_label].copy()

    # Remove rows whose image does not exist on disk
    df["exists"] = df["image"].apply(lambda x: (img_dir / x).exists())
    missing = df[~df["exists"]]["image"].unique()

    if len(missing) > 0:
        print(f"WARNING: {len(missing)} images referenced in CSV not found in img_dir")
        print("Example:", missing[:5])

    df = df[df["exists"]].drop(columns="exists")

    if df.empty:
        raise ValueError("No rows found for label == 'scallop'. Check label casing.")

    print(f"Remaining rows after filtering: {len(df)}")

    # Since we filtered, categories are now single-class
    categories = {"scallop": 1}
    print("Categories:", categories)

    # Stratified split by scallop count per image
    train_imgs, val_imgs = stratified_split_by_object_count(
        df, seed=seed, val_frac=val_frac
    )
    print(f"Train images: {len(train_imgs)} | Val images: {len(val_imgs)}")

    # Stable image IDs
    all_imgs = sorted(set(df["image"]))
    image_ids = {fname: i+1 for i, fname in enumerate(all_imgs)}

    train_df = df[df["image"].isin(train_imgs)].copy()
    val_df   = df[df["image"].isin(val_imgs)].copy()

    train_coco = build_coco(train_df, img_dir, categories, image_ids)
    val_coco   = build_coco(val_df, img_dir, categories, image_ids)

    with open(out_dir / "train.json", "w") as f:
        json.dump(train_coco, f)

    with open(out_dir / "val.json", "w") as f:
        json.dump(val_coco, f)

    print("Wrote:")
    print(" -", out_dir / "train.json")
    print(" -", out_dir / "val.json")

if __name__ == "__main__":
    main()
