#!/usr/bin/env python3

import argparse
import re
import math
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

def parse_geometry_to_square(geom_text):
    """
    Parses Habcam GEOMETRY_TEXT.
    Converts the annotated line segment into a square bounding box 
    using the line as the diameter.
    """
    if pd.isna(geom_text):
        return None, None, None, None
    
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(geom_text))
    if len(nums) >= 4:
        x1, y1, x2, y2 = map(float, nums[:4])
        
        diameter = math.hypot(x2 - x1, y2 - y1)
        radius = diameter / 2.0
        
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        
        return cx - radius, cy - radius, cx + radius, cy + radius
    
    return None, None, None, None

def load_path_mapping(txt_file):
    """Reads the source text manifest into a dictionary of {filename: absolute_path}"""
    print(f"📄 Loading source paths from {txt_file}...")
    mapping = {}
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                filename = Path(line).name
                mapping[filename] = line
    return mapping

def crop_and_save(src_path, out_path, side):
    """
    Crops the image down the middle based on the designated side.
    Returns the midpoint and offset for bounding box calculations.
    Returns (None, None, None, None) if the image is corrupted or unreadable.
    """
    try:
        with Image.open(src_path) as im:
            w, h = im.size
            mid = w / 2
            
            if side == "left":
                crop_box = (0, 0, mid, h)
                offset = 0
                new_w = mid
            else: # right
                crop_box = (mid, 0, w, h)
                offset = mid
                new_w = w - mid
            
            cropped_im = im.crop(crop_box)
            cropped_im.save(out_path)
            
            return mid, offset, new_w, h
            
    except Exception as e:
        # Catch PIL.UnidentifiedImageError, OSError, or any FUSE network drops
        tqdm.write(f"\n[WARN] Skipping unreadable image {Path(src_path).name}: {e}")
        return None, None, None, None

def process_annotations(csv_path, src_txt, out_img_dir, out_csv, side):
    """Processes the annotated images, applies bbox math, splits, and remaps."""
    out_img_dir = Path(out_img_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    path_mapping = load_path_mapping(src_txt)

    print(f"📦 Loading annotations from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Normalize columns
    if "IMAGE_NAME" in df.columns:
        df = df.rename(columns={"IMAGE_NAME": "image", "CLASS_NAME": "label"})
    
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["image"] = df["image"].astype(str).str.strip()

    # 🔥 EXCLUDE POINT ANNOTATIONS
    print("🧹 Filtering out 'point' annotations...")
    initial_count = len(df)
    df = df[~df["GEOMETRY_TEXT"].astype(str).str.lower().str.contains("point", na=False)].copy()
    print(f"   Dropped {initial_count - len(df)} point annotations.")

    print("🧠 Converting line annotations to square bounding boxes...")
    bounds = df["GEOMETRY_TEXT"].apply(parse_geometry_to_square)
    df["TLx"] = [b[0] for b in bounds]
    df["TLy"] = [b[1] for b in bounds]
    df["BRx"] = [b[2] for b in bounds]
    df["BRy"] = [b[3] for b in bounds]
    
    df = df.dropna(subset=["TLx", "TLy", "BRx", "BRy", "image"])
    
    # Filter for unique images that are present in the manifest and exist on disk
    unique_images = [img for img in df["image"].unique() if img in path_mapping and Path(path_mapping[img]).exists()]

    new_rows = []

    print(f"\n✂️ Splitting {len(unique_images)} ANNOTATED images ({side.upper()} side)...")
    for fname in tqdm(unique_images):
        src = path_mapping[fname]
        out_path = out_img_dir / fname
        
        # Capture the new width (new_w) and height (h)
        crop_result = crop_and_save(src, out_path, side)
        
        # If the image was unreadable, skip it and move to the next one
        if crop_result[0] is None:
            continue
            
        mid, offset, new_w, h = crop_result

        sub = df[df["image"] == fname].copy()
        
        # Shift X coordinates
        sub["TLx"] -= offset
        sub["BRx"] -= offset
        
        # Filter out boxes that are entirely on the wrong side
        sub = sub[sub["BRx"] > 0].copy()
        sub = sub[sub["TLx"] < new_w].copy() # FIXED: strictly use the new image width
        
        # 🔥 APPLY STRICT 4-WAY CLIPPING
        sub["TLx"] = sub["TLx"].clip(lower=0)
        sub["BRx"] = sub["BRx"].clip(upper=new_w)
        sub["TLy"] = sub["TLy"].clip(lower=0)
        sub["BRy"] = sub["BRy"].clip(upper=h)
        
        # Drop invalid/sliver boxes (must be > 1x1 pixels)
        sub["bw"] = sub["BRx"] - sub["TLx"]
        sub["bh"] = sub["BRy"] - sub["TLy"]
        sub = sub[(sub["bw"] > 1) & (sub["bh"] > 1)]

        # 🔥 MISSING LOGIC RESTORED: Append surviving rows to our list
        for _, r in sub.iterrows():
            new_rows.append({
                "image": fname,
                "TLx": round(r["TLx"], 2),
                "TLy": round(r["TLy"], 2),
                "BRx": round(r["BRx"], 2),
                "BRy": round(r["BRy"], 2),
                "label": r["label"]
            })

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(out_csv, index=False)
    print(f"💾 Saved {len(new_df)} mapped annotations to → {out_csv}")

def process_zeros(csv_path, src_txt, out_img_dir, out_csv, side):
    """Processes the background images separately."""
    out_img_dir = Path(out_img_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    path_mapping = load_path_mapping(src_txt)

    print(f"\n📦 Loading zero-annotations from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    img_col = "imagename" if "imagename" in df.columns else "IMAGE_NAME"
    zero_images = df[img_col].dropna().unique()
    
    valid_zeros = [img for img in zero_images if img in path_mapping and Path(path_mapping[img]).exists()]

    new_rows = []

    print(f"\n✂️ Splitting {len(valid_zeros)} ZERO images ({side.upper()} side)...")
    for fname in tqdm(valid_zeros):
        src = path_mapping[fname]
        out_path = out_img_dir / fname
        
        crop_result = crop_and_save(src, out_path, side)
        
        # If the image was unreadable, skip it
        if crop_result[0] is None:
            continue

        new_rows.append({
            "image": fname,
            "TLx": None, "TLy": None, "BRx": None, "BRy": None,
            "label": "background"
        })

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(out_csv, index=False)
    print(f"💾 Saved {len(new_df)} background entries to → {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Split Habcam images and apply Square BBox math directly from manifest.")
    
    # Annotation Arguments
    parser.add_argument("--ann_csv", required=True)
    parser.add_argument("--ann_src_txt", required=True)
    parser.add_argument("--out_ann_img_dir", required=True)
    parser.add_argument("--out_ann_csv", required=True)
    
    # Zero Image Arguments
    parser.add_argument("--zero_csv", required=True)
    parser.add_argument("--zero_src_txt", required=True)
    parser.add_argument("--out_zero_img_dir", required=True)
    parser.add_argument("--out_zero_csv", required=True)
    
    # Global Argument
    parser.add_argument("--side", choices=["left", "right"], required=True)

    args = parser.parse_args()

    process_annotations(args.ann_csv, args.ann_src_txt, args.out_ann_img_dir, args.out_ann_csv, args.side)
    process_zeros(args.zero_csv, args.zero_src_txt, args.out_zero_img_dir, args.out_zero_csv, args.side)
    
    print("\n✅ All processing complete!")

if __name__ == "__main__":
    main()