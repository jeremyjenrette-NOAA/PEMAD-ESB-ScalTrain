#!/usr/bin/env python3
"""
Step 1: Filter the master groundtruth/empties CSVs down to the 2022-2024
scallop training subset (scallop2224). 2026 (and any other year) is excluded
entirely.

Reads:
  - groundtruth2226.csv          (required)
  - groundtruth2226_supplemental.csv (optional, merged in if given)
  - empties2226.csv               (required)

Writes:
  - <out-dir>/groundtruth2224.csv
  - <out-dir>/empties2224.csv

Usage:
    python 01_filter_annotations.py \
        --groundtruth /data2226/annotations/groundtruth2226.csv \
        --supplemental /data2226/annotations/groundtruth2226_supplemental.csv \
        --empties /data2226/annotations/empties2226.csv \
        --out-dir scallop2224/annotations \
        --years 2022 2023 2024
"""
import argparse
from pathlib import Path

import pandas as pd

# Columns used to de-duplicate annotation rows that may appear in both the
# main groundtruth file and the supplemental file.
DEDUP_KEYS = ["imagename", "tlx", "tly", "brx", "bry", "annotator_user_id", "annotation_timestamp"]


def load_and_concat(paths):
    frames = []
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            print(f"[WARN] skipping missing file: {p}")
            continue
        df = pd.read_csv(p, low_memory=False)
        df["_source_file"] = p.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No input files could be loaded.")
    return pd.concat(frames, ignore_index=True, sort=False)


def filter_groundtruth(df: pd.DataFrame, years: set) -> pd.DataFrame:
    required = {"year", "class_category_phylum", "class_name", "imagename"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"groundtruth file(s) missing expected columns: {missing}")

    before = len(df)
    df = df[df["year"].isin(years)]
    df = df[df["class_category_phylum"].astype(str).str.strip().str.lower() == "scallop"]

    dedup_cols = [c for c in DEDUP_KEYS if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols, keep="first")

    print(f"[groundtruth] {before} rows -> {len(df)} rows after year+phylum filter+dedup")
    return df.reset_index(drop=True)


def filter_empties(df: pd.DataFrame, years: set) -> pd.DataFrame:
    required = {"year", "imagename"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"empties file missing expected columns: {missing}")

    before = len(df)
    df = df[df["year"].isin(years)]
    df = df.drop_duplicates(subset=["imagename"], keep="first")
    print(f"[empties]     {before} rows -> {len(df)} rows after year filter+dedup")
    return df.reset_index(drop=True)


def summarize(df: pd.DataFrame, label_col: str, name: str):
    print(f"\n--- {name}: counts by year ---")
    print(df["year"].value_counts().sort_index().to_string())
    if label_col in df.columns:
        print(f"\n--- {name}: counts by {label_col} ---")
        print(df[label_col].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groundtruth", required=True, type=Path)
    ap.add_argument("--supplemental", type=Path, default=None,
                    help="Optional groundtruth2226_supplemental.csv to merge in")
    ap.add_argument("--empties", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--tag", default="2224", help="Suffix used in output filenames")
    args = ap.parse_args()

    years = set(args.years)
    if 2026 in years:
        raise ValueError("2026 must be excluded per project scope; remove it from --years")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_raw = load_and_concat([args.groundtruth, args.supplemental])
    gt_filtered = filter_groundtruth(gt_raw, years)
    summarize(gt_filtered, "class_name", "groundtruth")

    empties_raw = load_and_concat([args.empties])
    empties_filtered = filter_empties(empties_raw, years)
    summarize(empties_filtered, "class_name", "empties")

    gt_out = args.out_dir / f"groundtruth{args.tag}.csv"
    empties_out = args.out_dir / f"empties{args.tag}.csv"
    gt_filtered.to_csv(gt_out, index=False)
    empties_filtered.to_csv(empties_out, index=False)

    print(f"\nWrote: {gt_out}  ({gt_filtered['imagename'].nunique()} unique annotated images)")
    print(f"Wrote: {empties_out}  ({empties_filtered['imagename'].nunique()} unique background images)")


if __name__ == "__main__":
    main()
