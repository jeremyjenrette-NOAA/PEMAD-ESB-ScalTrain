#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime


def extract_date(fname):
    """
    Extract YYYYMMDD from filename
    Example: 202404.20240627.233956150.18438.png
    """
    try:
        return datetime.strptime(fname.split(".")[1], "%Y%m%d")
    except:
        return None


def split_and_remap(
    csv_path,
    img_dir,
    out_dir,
    label="scallop",
    shift_px=55,
    cutoff_date="2024-05-11"
):
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")

    print("\n📦 Loading annotations...")
    df = pd.read_csv(csv_path)

    # normalize
    df = df.rename(columns={"Imagename": "image", "ClassName": "label"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["image"] = df["image"].astype(str).str.strip()

    # filter scallops
    df = df[df["label"] == label].copy()
    print(f"Initial scallop images: {df['image'].nunique()}")

    # numeric bbox
    for c in ["TLx", "TLy", "BRx", "BRy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["TLx", "TLy", "BRx", "BRy"]).copy()

    new_rows = []

    print("\n🧠 Remapping annotations (no image write)...")

    for fname in tqdm(df["image"].unique()):
        src = img_dir / fname
        if not src.exists():
            continue

        # extract date
        img_date = extract_date(fname)
        if img_date is None:
            continue

        # determine shift
        apply_shift = img_date > cutoff

        # get image width (needed for splitting)
        with Image.open(src) as im:
            w, h = im.size

        # subset annotations
        sub = df[df["image"] == fname].copy()

        # remap x coords (to right image)
        offset = w / 2
        sub["TLx"] = sub["TLx"] - offset
        sub["BRx"] = sub["BRx"] - offset

        # keep only right-image boxes
        sub = sub[sub["BRx"] > 0].copy()

        # clip to image bounds
        right_w = w / 2
        sub["TLx"] = sub["TLx"].clip(lower=0)
        sub["BRx"] = sub["BRx"].clip(upper=right_w)

        # 🔥 APPLY CONDITIONAL SHIFT
        if apply_shift:
            sub["TLx"] += shift_px
            sub["BRx"] += shift_px

            # clip again after shift
            sub["TLx"] = sub["TLx"].clip(lower=0)
            sub["BRx"] = sub["BRx"].clip(upper=right_w)

        # drop invalid boxes
        sub["bw"] = sub["BRx"] - sub["TLx"]
        sub["bh"] = sub["BRy"] - sub["TLy"]
        sub = sub[(sub["bw"] > 1) & (sub["bh"] > 1)].copy()

        if sub.empty:
            continue

        # store updated rows
        for r in sub.itertuples(index=False):
            new_rows.append({
                "image": fname,
                "TLx": r.TLx,
                "TLy": r.TLy,
                "BRx": r.BRx,
                "BRy": r.BRy,
                "label": r.label,
                "date": img_date.strftime("%Y-%m-%d"),
                "shift_applied": apply_shift
            })

    print("\n✅ Done processing")

    new_df = pd.DataFrame(new_rows)

    out_csv = out_dir / "groundtruth24.csv"
    new_df.to_csv(out_csv, index=False)

    print(f"\n💾 Saved annotations → {out_csv}")

    # quick diagnostics
    print("\n📊 Shift summary:")
    print(new_df["shift_applied"].value_counts())

    return new_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label", default="scallop")

    args = parser.parse_args()

    split_and_remap(
        csv_path=args.csv,
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        label=args.label
    )


if __name__ == "__main__":
    main()