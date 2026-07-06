#!/usr/bin/env python3

"""
Master Label Maker & Stratifier for HabCam Imagery

This script:
1. Loads groundtruth (positive) and empty (negative) annotations.
2. Calculates YOLO [0,1] coordinates in memory for positives.
3. Groups the dataset by image, calculating total annotations (density).
4. Stratifies the data spatially and by density.
5. Builds nested train/test and GAM splits using transect blocks.
6. Writes the images and label files to yolo/images and yolo/labels.
7. Outputs a comprehensive dataset_split.csv manifest.
"""

from __future__ import annotations

import math
import random
import shutil
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# =========================
# Configuration
# =========================
BASE_DIR = Path("/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/data2226")
GT_CSV = BASE_DIR / "annotations" / "groundtruth2226.csv"
EMP_CSV = BASE_DIR / "annotations" / "empties2226.csv"

IMG_DIR_POS = BASE_DIR / "viame" / "images"
IMG_DIR_EMP = BASE_DIR / "empties"

YOLO_DIR = BASE_DIR / "yolo"
OUTPUT_CSV = BASE_DIR / "annotations" / "dataset_split_2226.csv"

# Label settings
TARGET_LABEL = "scallop"
LINK_MODE = "symlink" # options: symlink, copy, hardlink
RANDOM_SEED = 42

# Transect/block split settings
BLOCK_SIZE = 10
TRAIN_PER_BLOCK = 7
TEST_PER_BLOCK = 3
MIN_BLOCK_SIZE_TO_SPLIT = 3
MAX_TIME_GAP_SEC = 60
MAX_DISTANCE_KM = 0.25
MAX_FRAME_GAP = 500

# Nested GAM split chunk size
GAM_BLOCK_SIZE = 20
GAM_TRAIN_FRAC_WITHIN_TEST = 0.65
GAM_TEST_FRAC_WITHIN_TEST = 0.35

# Main proportions (Fallback)
TRAIN_FRAC = 0.70
TEST_FRAC = 0.30

# Stratification settings
N_LAT_BINS = 4
N_LON_BINS = 4
N_DENSITY_BINS = 4
MIN_STRATUM_SIZE = 5

# =========================
# Math & YOLO Helpers
# =========================
def clamp(v, lo, hi):
    """Clip coordinates to image bounds."""
    return max(lo, min(hi, v))

def bbox_to_yolo(x1, y1, x2, y2, w, h):
    """Convert absolute coordinates to normalized YOLO format."""
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    x1 = clamp(x1, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    x2 = clamp(x2, 0, w - 1)
    y2 = clamp(y2, 0, h - 1)

    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 1 or bh <= 1:
        return None

    xc = x_min + bw / 2.0
    yc = y_min + bh / 2.0
    return (xc / w, yc / h, bw / w, bh / h)

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between points."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

def parse_image_tokens(imagename: str) -> pd.Series:
    """Extract temporal tokens from standard Habcam filenames."""
    parts = Path(str(imagename)).stem.split(".")
    out = {"name_cruise_token": np.nan, "name_date_token": np.nan, "name_time_token": np.nan, "name_frame_number": np.nan}
    if len(parts) >= 4:
        out["name_cruise_token"] = parts[0]
        out["name_date_token"] = parts[1]
        out["name_time_token"] = parts[2]
        try:
            out["name_frame_number"] = int(parts[3])
        except ValueError:
            pass
    return pd.Series(out)

def summarize_splits(df: pd.DataFrame):
    """
    Prints a comprehensive summary of the dataset splits, annotation densities,
    empty image incorporation, and spatial stratification.
    """
    print("\n" + "="*50)
    print(" DATASET SUMMARY STATISTICS")
    print("="*50)

    # 1. Empties vs Positives Incorporated
    print("\n--- 1. Image Composition ---")
    composition = df['is_empty'].map({True: 'Empty (Negative)', False: 'Scallop (Positive)'}).value_counts()
    print(composition.to_string())
    print(f"Total Images: {len(df)}")
    empty_pct = (composition.get('Empty (Negative)', 0) / len(df)) * 100
    print(f"Empty Image Proportion: {empty_pct:.2f}%")

    # 2. Split Distribution
    print("\n--- 2. Train/Test/GAM Split Breakdown ---")
    split_counts = df["dataset"].value_counts(dropna=False)
    split_pcts = df["dataset"].value_counts(normalize=True, dropna=False) * 100
    split_summary = pd.DataFrame({'Count': split_counts, 'Percentage': split_pcts.round(2)})
    print(split_summary.to_string())

    # 3. Annotation Statistics
    print("\n--- 3. Annotation Density (Min/Max/Mean) ---")
    # Overall
    pos_only = df[~df['is_empty']]
    print(f"Overall Positives - Min: {pos_only['n_annotations'].min()}, Max: {pos_only['n_annotations'].max()}, Mean: {pos_only['n_annotations'].mean():.2f}")
    
    # By Dataset Split
    print("\nAnnotation Stats by Split (Includes Empties):")
    ann_stats = df.groupby("dataset")["n_annotations"].agg(["count", "mean", "min", "max"])
    print(ann_stats.round(2).to_string())

    # 4. Spatial Stratification Check
    print("\n--- 4. Spatial Stratification Check ---")
    print("Images per Dataset Split within Coarse Geographic Strata:")
    # Using the collapsed stratum column to keep the table readable
    spatial_crosstab = pd.crosstab(df["stratum"], df["dataset"])
    print(spatial_crosstab.to_string())
    print("="*50 + "\n")
# =========================
# File IO Helper
# =========================
def link_image(src: Path, dst: Path, mode: str):
    """Safely move/link images to YOLO directory."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)

# =========================
# Pipeline Execution
# =========================
def build_memory_labels() -> tuple[pd.DataFrame, dict]:
    """Processes positive CSV to build YOLO coordinates in memory."""
    print(f"Loading positives from {GT_CSV}...")
    df = pd.read_csv(GT_CSV)
    df.columns = [c.lower() for c in df.columns]
    
    # Filter to target label
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df = df[df['label'] == TARGET_LABEL].dropna(subset=["tlx", "tly", "brx", "bry"]).copy()
    
    yolo_labels_by_image = {}
    valid_image_meta = []
    
    # Process bounding boxes
    grouped = df.groupby("imagename")
    for img_name, sub in grouped:
        src_img = IMG_DIR_POS / img_name
        if not src_img.exists():
            continue
            
        try:
            with Image.open(src_img) as im:
                w, h = im.size
        except Exception as e:
            print(f"Warning: Cannot open {src_img}: {e}")
            continue

        rows = []
        for r in sub.itertuples(index=False):
            y = bbox_to_yolo(r.tlx, r.tly, r.brx, r.bry, w, h)
            if y:
                rows.append(f"0 {y[0]:.6f} {y[1]:.6f} {y[2]:.6f} {y[3]:.6f}")
        
        if rows:
            yolo_labels_by_image[img_name] = rows
            # Keep first row's metadata for the master table
            meta = sub.iloc[0].to_dict()
            meta['n_annotations'] = len(rows)
            meta['is_empty'] = False
            meta['src_dir'] = IMG_DIR_POS
            valid_image_meta.append(meta)
            
    return pd.DataFrame(valid_image_meta), yolo_labels_by_image

def load_empties() -> pd.DataFrame:
    """Processes empty CSV."""
    print(f"Loading empties from {EMP_CSV}...")
    df = pd.read_csv(EMP_CSV)
    df.columns = [c.lower() for c in df.columns]
    
    df = df.drop_duplicates(subset=["imagename"]).copy()
    df['n_annotations'] = 0
    df['is_empty'] = True
    df['src_dir'] = IMG_DIR_EMP
    
    # Verify existence
    df['exists'] = df['imagename'].apply(lambda x: (IMG_DIR_EMP / str(x)).exists())
    df = df[df['exists']].drop(columns=['exists']).copy()
    
    return df

def apply_stratification(df: pd.DataFrame) -> pd.DataFrame:
    """Applies density and geographic chunking."""
    def safe_qcut(series, q, prefix):
        if series.nunique() <= 1:
            return pd.Series([f"{prefix}_0"] * len(series), index=series.index)
        try:
            return pd.qcut(series, q=q, duplicates="drop").astype(str)
        except ValueError:
            return pd.Series([f"{prefix}_0"] * len(series), index=series.index)

    df["lat_bin"] = safe_qcut(df["latitude"], N_LAT_BINS, "lat")
    df["lon_bin"] = safe_qcut(df["longitude"], N_LON_BINS, "lon")
    df["density_bin"] = safe_qcut(df["n_annotations"], N_DENSITY_BINS, "dens")
    
    df["stratum_fine"] = df["lat_bin"] + "__" + df["lon_bin"] + "__" + df["density_bin"]
    
    counts = df["stratum_fine"].value_counts()
    rare = set(counts[counts < MIN_STRATUM_SIZE].index)
    df["stratum"] = np.where(df["stratum_fine"].isin(rare), 
                             df["lat_bin"] + "__" + df["lon_bin"], 
                             df["stratum_fine"])
    return df

def assign_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Executes local block and nested GAM splitting."""
    rng = random.Random(seed)
    df = df.copy()
    df["dataset"] = None
    
    # Level 1: Block Splitting (Train/Test)
    for block_id, sub in df.groupby("transect_block_id", sort=False):
        idx = list(sub.index)
        n = len(idx)
        if n == 0: continue
        
        n_train = max(1, int(round(n * TRAIN_PER_BLOCK / BLOCK_SIZE))) if n >= MIN_BLOCK_SIZE_TO_SPLIT else int(round(n * TRAIN_FRAC))
        n_train = min(n_train, n)
        n_test = n - n_train
        if n_test == 0 and n > 1:
            n_test, n_train = 1, n - 1
            
        # Density spread for test assignments
        idx_sorted = sorted(idx, key=lambda i: df.loc[i, "n_annotations"])
        test_idx = []
        if n_test > 0:
            positions = [n//2] if n_test == 1 else np.linspace(0, n-1, n_test).round().astype(int).tolist()
            test_idx = list(dict.fromkeys([idx_sorted[p] for p in positions]))
            
            rem = [i for i in idx if i not in test_idx]
            rng.shuffle(rem)
            while len(test_idx) < n_test and rem: test_idx.append(rem.pop())
            
        train_idx = [i for i in idx if i not in test_idx]
        df.loc[train_idx, "dataset_level1"] = "train"
        df.loc[test_idx, "dataset_level1"] = "test"
        df.loc[train_idx, "dataset"] = "train"
        
    # Level 2: GAM Splitting (within Test)
    test_df = df[df["dataset_level1"] == "test"].sort_values(["transect_group", "transect_position", "imagename"])
    for group, sub in test_df.groupby("transect_group", sort=False):
        idx = list(sub.index)
        for start in range(0, len(idx), GAM_BLOCK_SIZE):
            chunk = idx[start:start + GAM_BLOCK_SIZE]
            n = len(chunk)
            if n == 0: continue
            
            n_gam_train = min(max(int(round(n * GAM_TRAIN_FRAC_WITHIN_TEST)), 0), n)
            n_gam_test = n - n_gam_train
            
            chunk_sorted = sorted(chunk, key=lambda i: df.loc[i, "n_annotations"])
            gam_test_chunk = []
            if n_gam_test > 0:
                pos = np.linspace(0, n-1, n_gam_test).round().astype(int)
                gam_test_chunk = list(dict.fromkeys([chunk_sorted[p] for p in pos]))
                rem = [i for i in chunk if i not in gam_test_chunk]
                rng.shuffle(rem)
                while len(gam_test_chunk) < n_gam_test and rem: gam_test_chunk.append(rem.pop())
                
            gam_train_chunk = [i for i in chunk if i not in gam_test_chunk]
            df.loc[gam_train_chunk, "dataset"] = "test_GAM_train"
            df.loc[gam_test_chunk, "dataset"] = "test_GAM_test"
            
    df["is_train"] = df["dataset"] == "train"
    df["is_test"] = df["dataset"].isin(["test_GAM_train", "test_GAM_test"])
    return df

def write_yolo_files(df: pd.DataFrame, yolo_labels: dict):
    """Dispatches physical files to YOLO architecture."""
    print("\nWriting physical YOLO dataset...")
    import os
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
        
    for split in ["train", "val"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True)

    for _, row in df.iterrows():
        img_name = row['imagename']
        split = "train" if row['is_train'] else "val"
        
        # Link Image
        src_img = row['src_dir'] / img_name
        dst_img = YOLO_DIR / "images" / split / img_name
        link_image(src_img, dst_img, LINK_MODE)
        
        # Write Label
        dst_lab = YOLO_DIR / "labels" / split / f"{Path(img_name).stem}.txt"
        rows = yolo_labels.get(img_name, [])
        dst_lab.write_text("\n".join(rows) + ("\n" if rows else ""))

    # YAML
    yaml_text = f"path: {YOLO_DIR.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: {TARGET_LABEL}\n"
    (YOLO_DIR / "label.yaml").write_text(yaml_text)

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    pos_df, yolo_labels = build_memory_labels()
    emp_df = load_empties()
    
    # Merge and build blocks
    df = pd.concat([pos_df, emp_df], ignore_index=True)
    
    # ADD THIS LINE: Drop duplicates, keeping the positive annotation (which comes first)
    df = df.drop_duplicates(subset=["imagename"], keep="first").reset_index(drop=True)
    
    df = pd.concat([df, df["imagename"].apply(parse_image_tokens)], axis=1)
    
    df["image_timestamp_parsed"] = pd.to_datetime(df["image_timestamp"], errors="coerce")
    cruise_col = "cruise_id" if "cruise_id" in df.columns else "name_cruise_token"
    df["transect_group"] = df[cruise_col].astype(str) + "_" + df["name_date_token"].astype(str)
    
    df = df.sort_values(["transect_group", "image_timestamp_parsed", "name_time_token", "name_frame_number", "imagename"]).reset_index(drop=True)
    
    df["transect_position"] = df.groupby("transect_group").cumcount()
    df["transect_block"] = df["transect_position"] // BLOCK_SIZE
    df["transect_block_id"] = df["transect_group"].astype(str) + "_block_" + df["transect_block"].astype(str)
    
    print("\nApplying stratification and splitting...")
    df = apply_stratification(df)
    df = assign_splits(df, RANDOM_SEED)
    
    write_yolo_files(df, yolo_labels)
    
    # ---> CALL THE SUMMARY FUNCTION HERE <---
    summarize_splits(df)
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Define the core columns we want at the front
    clean_cols = [
        'imagename', 'n_annotations', 'is_empty', 'dataset', 'dataset_level1', 
        'is_train', 'is_test', 'latitude', 'longitude', 'stratum', 'transect_block_id'
    ]
    
    # 2. Define the columns you explicitly want to drop (plus our temporary src_dir)
    drop_cols = [
        'tlx', 'tly', 'brx', 'bry', 'label', 'data_identifier', 
        'assignment_id', 'annotation_timestamp', 'annotator_user_id', 
        'geometry_text', 'class_id', 'src_dir'
    ]
    
    # 3. Build the final column list: core columns first, then any remaining metadata, excluding drops
    final_cols = clean_cols + [c for c in df.columns if c not in clean_cols and c not in drop_cols]
    
    # 4. Export
    df[final_cols].to_csv(OUTPUT_CSV, index=False)
    
    print(f"Success. Final dataset generated at {OUTPUT_CSV.resolve()}")