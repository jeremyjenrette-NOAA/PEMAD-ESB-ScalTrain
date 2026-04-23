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


def assign_nested_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Assign train/test then nested GAM splits within test, stratified within each stratum.
    """
    rng = random.Random(seed)
    df = df.copy()

    df["dataset"] = None
    df["dataset_level1"] = None
    df["dataset_level2"] = None

    for stratum, idx in df.groupby("stratum").groups.items():
        idx = list(idx)
        rng.shuffle(idx)

        n = len(idx)
        n_train, n_test = choose_split_counts(n, TRAIN_FRAC, TEST_FRAC)

        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        df.loc[train_idx, "dataset_level1"] = "train"
        df.loc[test_idx, "dataset_level1"] = "test"

        # Within test, split into GAM train / GAM test
        test_idx = list(test_idx)
        rng.shuffle(test_idx)

        n_gam_train, n_gam_test = choose_split_counts(
            len(test_idx),
            GAM_TRAIN_FRAC_WITHIN_TEST,
            GAM_TEST_FRAC_WITHIN_TEST
        )

        gam_train_idx = test_idx[:n_gam_train]
        gam_test_idx = test_idx[n_gam_train:]

        df.loc[train_idx, "dataset"] = "train"
        df.loc[train_idx, "dataset_level2"] = None

        df.loc[gam_train_idx, "dataset"] = "test_GAM_train"
        df.loc[gam_train_idx, "dataset_level2"] = "test_GAM_train"

        df.loc[gam_test_idx, "dataset"] = "test_GAM_test"
        df.loc[gam_test_idx, "dataset_level2"] = "test_GAM_test"

    # Convenience flags
    df["is_train"] = df["dataset"] == "train"
    df["is_test"] = df["dataset"].isin(["test_GAM_train", "test_GAM_test"])
    df["is_test_gam_train"] = df["dataset"] == "test_GAM_train"
    df["is_test_gam_test"] = df["dataset"] == "test_GAM_test"

    return df


def summarize(df: pd.DataFrame) -> None:
    """
    Print quick sanity checks.
    """
    print("\nOverall counts:")
    print(df["dataset"].value_counts(dropna=False))

    print("\nOverall proportions:")
    print((df["dataset"].value_counts(normalize=True, dropna=False) * 100).round(2))

    print("\nMean annotations by dataset:")
    print(df.groupby("dataset")["n_annotations"].agg(["count", "mean", "median", "min", "max"]))

    print("\nLatitude/longitude summaries by dataset:")
    print(df.groupby("dataset")[["latitude", "longitude"]].agg(["mean", "std", "min", "max"]))

    print("\nImages missing metadata:")
    missing_meta = df["latitude"].isna() | df["longitude"].isna()
    print(int(missing_meta.sum()))

    print("\nImages with zero annotations:")
    print(int((df["n_annotations"] == 0).sum()))


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
    df = make_strata(df)

    # Assign nested splits
    df = assign_nested_splits(df, seed=RANDOM_SEED)

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