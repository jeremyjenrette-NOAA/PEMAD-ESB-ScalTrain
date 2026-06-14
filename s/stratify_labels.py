#!/usr/bin/env python3

"""
Build nested train/test/GAM splits for HabCam scallop imagery.

This script:
1. Reads image paths from yolo/images/train and yolo/images/val
2. Matches each image to its YOLO label file in yolo/labels/train or yolo/labels/val
3. Counts YOLO annotations per image as a proxy for scallop density
4. Joins metadata from annotations/meta_all2224.csv
5. Builds stratification bins using:
      - latitude
      - longitude
      - annotation count
6. Creates nested splits:
      - train (70%)
      - test (30%)
          - test_GAM_train (65% of test)
          - test_GAM_test (35% of test)
7. Writes a final CSV describing all images and their assignments
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# =========================
# Transect/block split settings
# =========================

SPLIT_MODE = "transect_block"  # options: "transect_block", "coarse_strata"

BLOCK_SIZE = 10
TRAIN_PER_BLOCK = 7
TEST_PER_BLOCK = 3

# Nested GAM split chunk size among test images
GAM_BLOCK_SIZE = 20
GAM_TRAIN_FRAC_WITHIN_TEST = 0.65
GAM_TEST_FRAC_WITHIN_TEST = 0.35

# Transect break thresholds
MAX_TIME_GAP_SEC = 60          # break transect if images are >60 sec apart
MAX_DISTANCE_KM = 0.25         # break transect if images jump >250 m
MAX_FRAME_GAP = 500            # break transect if frame token jumps too far

# Images in very small inferred groups are still split, but less perfectly
MIN_BLOCK_SIZE_TO_SPLIT = 3

# =========================
# User settings
# =========================
BASE_DIR = Path("train_arc/data2224") 
META_CSV = BASE_DIR / "annotations" / "meta_all2224.csv"
MISSING_CSV = BASE_DIR / "annotations" / "missing_metadata_images.csv"

YOLO_IMG_DIR = BASE_DIR / "yolo" / "images"
YOLO_LABEL_DIR = BASE_DIR / "yolo" / "labels"

OUTPUT_CSV = BASE_DIR / "annotations" / "dataset_split_2224.csv"

RANDOM_SEED = 42

# Main split proportions
TRAIN_FRAC = 0.70
TEST_FRAC = 0.30

# Nested split within test
GAM_TRAIN_FRAC_WITHIN_TEST = 0.65
GAM_TEST_FRAC_WITHIN_TEST = 0.35

# Stratification settings
N_LAT_BINS = 4
N_LON_BINS = 4
N_DENSITY_BINS = 4

# If a stratum has fewer than this many images, collapse to coarser grouping
MIN_STRATUM_SIZE = 5


# =========================
# Helpers
# =========================
def clean_bool(x) -> bool:
    """Robust bool parser for strings/logicals."""
    return str(x).strip().lower() in ["true", "1", "yes", "y"]


def parse_image_tokens(imagename: str) -> pd.Series:
    """
    Parse HabCam filename like:
    202203.20220608.004026617.8503.png

    Expected structure:
      cruise_token.date_token.time_token.frame_token.ext
    """
    stem = Path(str(imagename)).stem
    parts = stem.split(".")

    out = {
        "name_cruise_token": np.nan,
        "name_date_token": np.nan,
        "name_time_token": np.nan,
        "name_frame_token": np.nan,
        "name_frame_number": np.nan,
    }

    if len(parts) >= 4:
        out["name_cruise_token"] = parts[0]
        out["name_date_token"] = parts[1]
        out["name_time_token"] = parts[2]
        out["name_frame_token"] = parts[3]

        try:
            out["name_frame_number"] = int(parts[3])
        except ValueError:
            out["name_frame_number"] = np.nan

    return pd.Series(out)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between sequential lat/lon points."""
    r = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * r * np.arcsin(np.sqrt(a))

def assign_transect_block_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Assign nested dataset splits using local transect-position blocks.

    Main split:
      - Within each ordered transect block, assign ~70% to YOLO train
      - Assign ~30% to test

    Nested GAM split:
      - Among only the test images, assign ~65% to test_GAM_train
      - Assign ~35% to test_GAM_test

    Important design principle:
      - The 70/30 split is made locally within consecutive image blocks.
      - The GAM split is made afterward among test images using
        assign_gam_splits_from_test(), which avoids excessive rounding problems
        when individual transect blocks contain only a few test images.
    """

    rng = random.Random(seed)
    df = df.copy()

    # ---- initialize split columns ----
    df["dataset"] = None
    df["dataset_level1"] = None
    df["dataset_level2"] = None

    # ---- basic required-column checks ----
    required_cols = [
        "transect_block_id",
        "n_annotations"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise RuntimeError(
            "assign_transect_block_splits() is missing required columns: "
            + ", ".join(missing_cols)
            + ". Make sure add_transect_structure() and make_strata() were run first."
        )

    # =========================================================
    # 1. Main local train/test split
    # =========================================================
    for block_id, sub in df.groupby("transect_block_id", sort=False):

        idx = list(sub.index)
        n = len(idx)

        if n == 0:
            continue

        # -----------------------------------------------------
        # Decide how many images go to train/test
        # -----------------------------------------------------
        if n < MIN_BLOCK_SIZE_TO_SPLIT:
            # Very small orphan blocks cannot support a clean 7/3 split.
            # Use the global requested ratio as fallback.
            n_train, n_test = choose_split_counts(n, TRAIN_FRAC, TEST_FRAC)

        else:
            # Approximate 7/3 split for full or partial local blocks.
            n_train = int(round(n * TRAIN_PER_BLOCK / BLOCK_SIZE))
            n_train = min(max(n_train, 1), n)
            n_test = n - n_train

            # Ensure usable blocks contribute at least one test image.
            if n_test == 0:
                n_test = 1
                n_train = n - 1

        # -----------------------------------------------------
        # Choose test images within the block
        # -----------------------------------------------------
        # Density-aware option:
        # Sort by annotation count and spread test picks across the local
        # density distribution. This avoids accidentally putting all dense
        # or all empty images from a local block into the same split.
        idx_sorted = sorted(idx, key=lambda i: df.loc[i, "n_annotations"])

        test_idx = []

        if n_test > 0:

            if n_test == 1:
                # Pick the median-density image
                candidate_positions = [n // 2]
            else:
                # Spread selected test images across density ranks
                candidate_positions = (
                    np.linspace(0, n - 1, n_test)
                    .round()
                    .astype(int)
                    .tolist()
                )

            for pos in candidate_positions:
                test_idx.append(idx_sorted[pos])

            # Remove possible duplicate positions caused by rounding
            test_idx = list(dict.fromkeys(test_idx))

            # Fill any shortfall randomly from remaining images
            remaining = [i for i in idx if i not in test_idx]
            rng.shuffle(remaining)

            while len(test_idx) < n_test and remaining:
                test_idx.append(remaining.pop())

        train_idx = [i for i in idx if i not in test_idx]

        # -----------------------------------------------------
        # Assign level-1 split
        # -----------------------------------------------------
        df.loc[train_idx, "dataset_level1"] = "train"
        df.loc[test_idx, "dataset_level1"] = "test"

    # =========================================================
    # 2. Assign train rows
    # =========================================================
    train_mask = df["dataset_level1"] == "train"

    df.loc[train_mask, "dataset"] = "train"
    df.loc[train_mask, "dataset_level2"] = None

    # =========================================================
    # 3. Nested GAM split among test rows
    # =========================================================
    # This function should split only rows where dataset_level1 == "test"
    # into test_GAM_train and test_GAM_test.
    df = assign_gam_splits_from_test(df, seed=seed)

    # =========================================================
    # 4. Safety fallback
    # =========================================================
    unassigned = df["dataset"].isna()

    if unassigned.any():
        print(f"Warning: assigning {unassigned.sum()} unassigned rows to train")

        df.loc[unassigned, "dataset"] = "train"
        df.loc[unassigned, "dataset_level1"] = "train"
        df.loc[unassigned, "dataset_level2"] = None

    # =========================================================
    # 5. Convenience flags
    # =========================================================
    df["is_train"] = df["dataset"] == "train"

    df["is_test"] = df["dataset"].isin([
        "test_GAM_train",
        "test_GAM_test"
    ])

    df["is_test_gam_train"] = df["dataset"] == "test_GAM_train"
    df["is_test_gam_test"] = df["dataset"] == "test_GAM_test"

    return df

def add_transect_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build local ordered blocks from image order.

    This version uses cruise/date/image time to preserve transect order,
    but does not over-fragment the data using strict time/spatial gaps.
    Gaps are retained as diagnostics.
    """
    df = df.copy()

    token_df = df["imagename"].apply(parse_image_tokens)
    df = pd.concat([df, token_df], axis=1)

    df["image_timestamp_parsed"] = pd.to_datetime(
        df["image_timestamp"],
        errors="coerce"
    )

    # Broad grouping unit: cruise + image date
    # This is less fragile than trying to infer every micro-transect.
    cruise_col = "CRUISE_ID" if "CRUISE_ID" in df.columns else "name_cruise_token"

    df["transect_group"] = (
        df[cruise_col].astype(str) + "_" +
        df["name_date_token"].astype(str)
    )

    sort_cols = [
        "transect_group",
        "image_timestamp_parsed",
        "name_time_token",
        "name_frame_number",
        "imagename"
    ]

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Diagnostics only
    df["time_gap_sec"] = (
        df["image_timestamp_parsed"] -
        df.groupby("transect_group")["image_timestamp_parsed"].shift(1)
    ).dt.total_seconds()

    df["frame_gap"] = (
        df["name_frame_number"] -
        df.groupby("transect_group")["name_frame_number"].shift(1)
    )

    df["spatial_gap_km"] = haversine_km(
        df.groupby("transect_group")["latitude"].shift(1),
        df.groupby("transect_group")["longitude"].shift(1),
        df["latitude"],
        df["longitude"]
    )

    # Flag large gaps but do not split on them
    df["large_gap_flag"] = (
        (df["time_gap_sec"].abs() > MAX_TIME_GAP_SEC) |
        (df["spatial_gap_km"] > MAX_DISTANCE_KM) |
        (df["frame_gap"].abs() > MAX_FRAME_GAP)
    )

    # Local position within broad ordered group
    df["transect_position"] = df.groupby("transect_group").cumcount()

    # Consecutive local blocks of 10
    df["transect_block"] = df["transect_position"] // BLOCK_SIZE

    df["transect_block_id"] = (
        df["transect_group"].astype(str) +
        "_block_" +
        df["transect_block"].astype(str)
    )

    # Keep a numeric transect id for compatibility
    df["transect_id"] = pd.factorize(df["transect_group"])[0] + 1

    return df

def count_label_rows(label_path: Path) -> int:
    """Count non-empty YOLO rows in a label file."""
    if not label_path.exists():
        return 0

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines()]

    return sum(1 for x in lines if x != "")


def safe_qcut(series: pd.Series, q: int, prefix: str) -> pd.Series:
    """
    Quantile binning that gracefully handles duplicate edges / low variability.
    Returns string labels.
    """
    non_na = series.dropna()

    if non_na.nunique() <= 1:
        return pd.Series([f"{prefix}_0"] * len(series), index=series.index)

    try:
        bins = pd.qcut(series, q=q, duplicates="drop")
        return bins.astype(str).fillna(f"{prefix}_missing")
    except ValueError:
        # fallback to a single bin if quantiles fail
        return pd.Series([f"{prefix}_0"] * len(series), index=series.index)


def choose_split_counts(n: int, frac_a: float, frac_b: float) -> tuple[int, int]:
    """
    Split n into two counts whose proportions approximate frac_a and frac_b.
    """
    if n == 0:
        return 0, 0
    n_a = int(round(n * frac_a))
    n_a = min(max(n_a, 0), n)
    n_b = n - n_a
    return n_a, n_b


def build_inventory() -> pd.DataFrame:
    """
    Build one row per image found in yolo/images/train and yolo/images/val.
    """
    rows = []

    for split_name in ["train", "val"]:
        img_dir = YOLO_IMG_DIR / split_name
        label_dir = YOLO_LABEL_DIR / split_name

        if not img_dir.exists():
            print(f"Warning: image dir not found: {img_dir}")
            continue

        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                continue

            imagename = img_path.name
            label_path = label_dir / f"{img_path.stem}.txt"
            n_annotations = count_label_rows(label_path)

            rows.append({
                "imagename": imagename,
                "img_path": str(img_path),
                "label_path": str(label_path) if label_path.exists() else None,
                "yolo_source_split": split_name,
                "n_annotations": n_annotations,
                "has_label_file": label_path.exists(),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No images found in yolo/images/train or yolo/images/val")

    # Drop duplicates just in case
    df = df.drop_duplicates(subset=["imagename"]).reset_index(drop=True)
    return df


def load_metadata() -> pd.DataFrame:
    """
    Load metadata and standardize key columns.
    """
    meta = pd.read_csv(META_CSV)

    # Standardize name column
    if "IMAGE_NAME" not in meta.columns:
        raise RuntimeError("Expected IMAGE_NAME column in meta_all2224.csv")

    meta = meta.rename(columns={
        "IMAGE_NAME": "imagename",
        "SHIP_LATITUDE": "latitude",
        "SHIP_LONGITUDE": "longitude",
        "FIELD_OF_VIEW_SQ_METER": "field_of_view_sq_meter",
        "MILLIMETER_PER_PIXEL": "millimeter_per_pixel",
        "IMAGE_TIMESTAMP": "image_timestamp",
    })

    return meta


def make_strata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined stratification labels from space and density.
    """
    df = df.copy()

    df["lat_bin"] = safe_qcut(df["latitude"], N_LAT_BINS, "lat")
    df["lon_bin"] = safe_qcut(df["longitude"], N_LON_BINS, "lon")
    df["density_bin"] = safe_qcut(df["n_annotations"], N_DENSITY_BINS, "dens")

    df["stratum_fine"] = (
        df["lat_bin"].astype(str) + "__" +
        df["lon_bin"].astype(str) + "__" +
        df["density_bin"].astype(str)
    )

    # Collapse rare strata
    stratum_counts = df["stratum_fine"].value_counts()
    rare = set(stratum_counts[stratum_counts < MIN_STRATUM_SIZE].index)

    df["stratum"] = np.where(
        df["stratum_fine"].isin(rare),
        # coarser fallback: just spatial bins
        df["lat_bin"].astype(str) + "__" + df["lon_bin"].astype(str),
        df["stratum_fine"]
    )

    return df

def assign_gam_splits_from_test(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Split only test images into GAM train/test.

    Uses ordered test images within transect_group to avoid tiny-block rounding
    problems.
    """
    rng = random.Random(seed)
    df = df.copy()

    test_df = df[df["dataset_level1"] == "test"].copy()

    # Order test images locally
    test_df = test_df.sort_values([
        "transect_group",
        "transect_position",
        "imagename"
    ])

    gam_train_indices = []
    gam_test_indices = []

    for group, sub in test_df.groupby("transect_group", sort=False):
        idx = list(sub.index)

        # Split in chunks of test images, not original images
        # This avoids every tiny block producing bad rounding.
        for start in range(0, len(idx), GAM_BLOCK_SIZE):
            chunk = idx[start:start + GAM_BLOCK_SIZE]
            n = len(chunk)

            if n == 0:
                continue

            n_gam_train = int(round(n * GAM_TRAIN_FRAC_WITHIN_TEST))
            n_gam_train = min(max(n_gam_train, 0), n)

            # Density-aware ordering inside chunk
            chunk_sorted = sorted(chunk, key=lambda i: df.loc[i, "n_annotations"])

            n_gam_test = n - n_gam_train

            gam_test_chunk = []
            if n_gam_test > 0:
                positions = np.linspace(0, n - 1, n_gam_test).round().astype(int)
                gam_test_chunk = [chunk_sorted[p] for p in positions]
                gam_test_chunk = list(dict.fromkeys(gam_test_chunk))

                remaining = [i for i in chunk if i not in gam_test_chunk]
                rng.shuffle(remaining)

                while len(gam_test_chunk) < n_gam_test and remaining:
                    gam_test_chunk.append(remaining.pop())

            gam_train_chunk = [i for i in chunk if i not in gam_test_chunk]

            gam_train_indices.extend(gam_train_chunk)
            gam_test_indices.extend(gam_test_chunk)

    df.loc[gam_train_indices, "dataset"] = "test_GAM_train"
    df.loc[gam_train_indices, "dataset_level2"] = "test_GAM_train"

    df.loc[gam_test_indices, "dataset"] = "test_GAM_test"
    df.loc[gam_test_indices, "dataset_level2"] = "test_GAM_test"

    return df




def summarize(df: pd.DataFrame) -> None:
    """
    Print quick sanity checks.
    """
    print("\nOverall counts:")
    print(df["dataset"].value_counts(dropna=False))

    print("\nOverall proportions:")
    print((df["dataset"].value_counts(normalize=True, dropna=False) * 100).round(2))

    print("\nBroad train/test proportions:")
    print(df["test_dataset"].value_counts(normalize=True).mul(100).round(2))

    print("\nMean annotations by dataset:")
    print(df.groupby("dataset")["n_annotations"].agg(["count", "mean", "median", "min", "max"]))

    print("\nZero annotations by dataset:")
    print(df.assign(is_zero=df["n_annotations"] == 0)
            .groupby("dataset")["is_zero"]
            .agg(["sum", "mean", "count"]))

    print("\nLatitude/longitude summaries by dataset:")
    print(df.groupby("dataset")[["latitude", "longitude"]].agg(["mean", "std", "min", "max"]))

    print("\nTransect block sizes:")
    print(df.groupby("transect_block_id").size().describe())

    print("\nImages per dataset within coarse stratum:")
    print(pd.crosstab(df["stratum"], df["dataset"]))

    print("\nImages missing metadata:")
    missing_meta = df["latitude"].isna() | df["longitude"].isna()
    print(int(missing_meta.sum()))

    print("\nImages with zero annotations:")
    print(int((df["n_annotations"] == 0).sum()))

    print("\nNumber of inferred transects:")
    print(df["transect_id"].nunique())

    print("\nNumber of inferred transect blocks:")
    print(df["transect_block_id"].nunique())


# =========================
# Main
# =========================
def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    inventory = build_inventory()
    meta = load_metadata()

    df = inventory.merge(meta, on="imagename", how="left")

    # Keep only images with metadata
    missing_meta = df["latitude"].isna() | df["longitude"].isna()

    missing_df = df.loc[missing_meta]

    missing_df.to_csv(MISSING_CSV, index=False)
    
    if missing_meta.any():
        print(f"Warning: dropping {missing_meta.sum()} images missing metadata")
        df = df.loc[~missing_meta].copy()

    print(f"Wrote missing metadata list to: {MISSING_CSV}")

    # Make strata
    # Keep coarse strata as diagnostic columns only
    df = make_strata(df)

    # Add transect and local block structure
    df = add_transect_structure(df)

    # Assign nested splits using local transect blocks
    df = assign_transect_block_splits(df, seed=RANDOM_SEED)

    # Optional: define the broad test set explicitly
    df["test_dataset"] = np.where(
        df["dataset"].isin(["test_GAM_train", "test_GAM_test"]),
        "test",
        "train"
    )

    # Reorder columns for readability
    preferred_cols = [
        "imagename",
        "img_path",
        "label_path",
        "has_label_file",
        "yolo_source_split",
        "n_annotations",
        "dataset",
        "test_dataset",
        "dataset_level1",
        "dataset_level2",
        "is_train",
        "is_test",
        "is_test_gam_train",
        "is_test_gam_test",
        "latitude",
        "longitude",
        "field_of_view_sq_meter",
        "millimeter_per_pixel",
        "image_timestamp",
        "transect_id",
        "transect_position",
        "transect_block",
        "transect_block_id",
        "time_gap_sec",
        "frame_gap",
        "spatial_gap_km",
        "large_gap_flag",
        "lat_bin",
        "lon_bin",
        "density_bin",
        "stratum",
    ]

    other_cols = [c for c in df.columns if c not in preferred_cols]
    df = df[preferred_cols + other_cols]

    summarize(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()