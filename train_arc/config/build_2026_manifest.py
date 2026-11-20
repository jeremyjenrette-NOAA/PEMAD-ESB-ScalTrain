#!/usr/bin/env python3
"""
Build the image manifest for full-survey 2026 inference (Tier B weights).

Reads the per-drive inventories of full image paths (one path per line, e.g. the
V: and W: mounts on the GPU server) and writes ONE ROW PER FILE, so every copy of an
image is kept and flagged, plus the chunk each file is processed in.

Columns of manifest_2026.csv.gz
-------------------------------
image_uid           <drive>:<imagename>  (unique key; imagename alone is not, see duplicates)
imagename, drive, src_path (path as listed in the inventory; converted to a native
                    Windows path at read time)
cruise, date, month, time, datetime, frame   parsed from <cruise>.<YYYYMMDD>.<HHMMSSmmm>.<frame>.png
group               habcam_primary (primary month, default 202605), auv (later months),
                    pre_survey (earlier months)
n_copies            number of drives holding this imagename
is_duplicate        n_copies > 1
other_drives        the drive(s) holding the other copy
color_corrected_assumed  True for files on --corrected_drives (default V). This is an
                    ASSUMPTION from the drive, not a measurement: infer_2026_full.py --mode probe
                    and the per-image colour statistics written during inference test it.
is_preferred_copy   one True per imagename (copy on --prefer drive when duplicated); use it
                    for one-row-per-image analyses
annotated_2026      imagename is in the --annotated_csv list (e.g. tierb_manifest.csv)
exclude_reason      non-empty rows are never processed (unparseable name, "(Copy)" files, ...)
frame_keep          imagename kept by the frame thinning (--frame_step, default every 5th frame
                    across V and W; annotated images always kept; preferred-drive frames chosen first)
selected            row is processed (not excluded, frame_keep, and --scope)
chunk_id            processing chunk (1..N) for selected rows, ordered habcam_primary first,
                    then auv, then pre_survey; within a group by date, time, drive, so both
                    copies of a duplicate land in the same chunk

Usage (Git Bash on the GPU server; see job/infer_2026_full.sh)
    python config/build_2026_manifest.py \
        --inv V /path/img_inventory_2026_fullpaths_v.txt \
        --inv W /path/img_inventory_2026_fullpaths_w.txt \
        --out_dir D:/scallop2026_tierb --chunk_size 10000 --scope all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAME_RE = re.compile(r"^(?P<cruise>\d+)\.(?P<date>\d{8})\.(?P<time>\d{9})\.(?P<frame>\d+)\.(?P<ext>png|jpg|jpeg|tif|tiff)$",
                     re.I)
GROUP_ORDER = {"habcam_primary": 0, "auv": 1, "pre_survey": 2}


def to_native(p: str) -> str:
    """'/v/habcam/x.png' -> 'V:/habcam/x.png' on Windows (Git Bash/MSYS style paths)."""
    if os.name == "nt":
        m = re.match(r"^/([A-Za-z])(/.*)?$", p)
        if m:
            return f"{m.group(1).upper()}:{m.group(2) or '/'}"
    return p


def thin_frames(u: pd.DataFrame, step: int) -> pd.DataFrame:
    """Keep about every `step`-th camera frame, using the union of all drives.

    u: one row per unique imagename with date, time, frame and tier (0 = annotated, always kept;
    1 = has a copy on the preferred drive; 2 = other drives only). Frames are grouped into
    sequences (per date; a new sequence starts whenever the frame counter goes backwards).
    Tiers are filled in order: a frame is kept only if it is >= `step` frames from every frame
    already kept in its sequence, so preferred-drive frames are chosen first and other-drive
    frames only fill the gaps. Where the images on disk are already >= `step` frames apart,
    all are kept.
    """
    u = u.sort_values(["date", "time", "frame"]).copy()
    f = u["frame"].to_numpy(dtype=np.int64)
    new = np.r_[True, (u["date"].to_numpy()[1:] != u["date"].to_numpy()[:-1]) | (f[1:] < f[:-1])]
    u["seq"] = np.cumsum(new)
    keep = np.zeros(len(u), dtype=bool)
    pos = {i: k for k, i in enumerate(u.index)}
    for _, g in u.groupby("seq", sort=False):
        kept = np.empty(0, dtype=np.int64)
        for tier in (0, 1, 2):
            c = g[g["tier"].eq(tier)]
            if c.empty:
                continue
            cf = c["frame"].to_numpy(dtype=np.int64)
            acc = np.zeros(len(cf), dtype=bool)
            if tier == 0:
                acc[:] = True
            else:
                lo, hi = np.iinfo(np.int64).min // 2, np.iinfo(np.int64).max // 2
                kp = np.r_[lo, kept, hi]                         # kept frames from earlier tiers, padded
                j = np.searchsorted(kp, cf)
                prev_k, next_k = kp[j - 1], kp[j]
                last = lo
                for i in range(len(cf)):
                    prev = max(prev_k[i], last)
                    if cf[i] - prev >= step and next_k[i] - cf[i] >= step:
                        acc[i] = True
                        last = cf[i]
            for idx in c.index[acc]:
                keep[pos[idx]] = True
            kept = np.sort(np.r_[kept, cf[acc]])
    u["thin_keep"] = keep
    return u


def read_inventory(path: str, drive: str) -> pd.DataFrame:
    with open(to_native(path), encoding="utf-8", errors="replace") as f:
        paths = [ln.rstrip("\r\n") for ln in f]
    paths = [p for p in paths if p.strip()]
    df = pd.DataFrame({"src_path": paths})
    df["drive"] = drive.upper()
    df["imagename"] = df["src_path"].str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    print(f"[{drive}] {len(df):,} paths from {path}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inv", nargs=2, action="append", metavar=("DRIVE", "LIST"), required=True,
                    help="Drive label and its inventory file; repeat per drive (e.g. --inv V v.txt --inv W w.txt)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--chunk_size", type=int, default=10000)
    ap.add_argument("--scope", choices=["all", "preferred"], default="all",
                    help="all = process every copy (duplicates twice, so corrected vs uncorrected can be "
                         "compared); preferred = one copy per imagename")
    ap.add_argument("--prefer", default="V", help="Drive whose copy is the preferred one when duplicated")
    ap.add_argument("--corrected_drives", default="V",
                    help="Comma list of drives assumed to hold colour-corrected images")
    ap.add_argument("--primary_month", default="202605", help="YYYYMM of the primary HabCam survey")
    ap.add_argument("--annotated_csv", default="",
                    help="Optional CSV with an 'imagename' column (e.g. tierb_manifest.csv) to flag annotated images")
    ap.add_argument("--frame_step", type=int, default=5,
                    help="Keep about every Nth camera frame across all drives (>= N frames apart); 0/1 = keep all")
    ap.add_argument("--keep_annotated", type=int, default=1,
                    help="1 = annotated images (--annotated_csv) are always kept by the frame thinning")
    ap.add_argument("--exclude_regex", default=r"\(copy\)|\s",
                    help="Imagenames matching this (case-insensitive) are excluded")
    a = ap.parse_args()

    out = Path(to_native(a.out_dir))
    out.mkdir(parents=True, exist_ok=True)

    df = pd.concat([read_inventory(p, d) for d, p in a.inv], ignore_index=True)
    n_in = len(df)
    df = df.drop_duplicates(["drive", "src_path"])
    if len(df) < n_in:
        print(f"Dropped {n_in - len(df):,} repeated lines")

    # --- parse names
    parts = df["imagename"].str.extract(NAME_RE)
    df = pd.concat([df, parts], axis=1)
    df["exclude_reason"] = ""
    bad = parts["date"].isna()
    df.loc[bad, "exclude_reason"] = "unparseable imagename"
    rx = re.compile(a.exclude_regex, re.I)
    hit = df["imagename"].map(lambda s: bool(rx.search(s)))
    df.loc[hit & ~bad, "exclude_reason"] = "matches exclude_regex"
    within = df.duplicated(["drive", "imagename"], keep="first")
    df.loc[within & df["exclude_reason"].eq(""), "exclude_reason"] = "same imagename twice on one drive"

    df["month"] = df["date"].str[:6]
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    df["datetime"] = pd.to_datetime(df["date"] + df["time"].str[:6], format="%Y%m%d%H%M%S", errors="coerce") \
        + pd.to_timedelta(pd.to_numeric(df["time"].str[6:], errors="coerce").fillna(0), unit="ms")
    df["group"] = np.select([df["month"].eq(a.primary_month), df["month"].gt(a.primary_month),
                             df["month"].lt(a.primary_month)],
                            ["habcam_primary", "auv", "pre_survey"], default="unknown")

    # --- duplicates across drives (only rows that are not excluded)
    ok = df["exclude_reason"].eq("")
    drives_per = df[ok].groupby("imagename")["drive"].agg(lambda s: ",".join(sorted(set(s))))
    df["n_copies"] = df["imagename"].map(drives_per.str.count(",") + 1).fillna(1).astype(int)
    df["is_duplicate"] = df["n_copies"] > 1
    df["other_drives"] = [",".join(d for d in drives_per.get(n, dr).split(",") if d != dr) if dup else ""
                          for n, dr, dup in zip(df["imagename"], df["drive"], df["is_duplicate"])]
    corrected = {d.strip().upper() for d in a.corrected_drives.split(",") if d.strip()}
    df["color_corrected_assumed"] = df["drive"].isin(corrected)

    pref = a.prefer.upper()
    df["_prefrank"] = np.where(df["drive"].eq(pref), 0, 1)
    df = df.sort_values(["imagename", "_prefrank", "drive"])
    df["is_preferred_copy"] = False
    okdf = df[df["exclude_reason"].eq("")]
    df.loc[okdf.index[~okdf.duplicated("imagename").to_numpy()], "is_preferred_copy"] = True

    if a.annotated_csv:
        ann = set(pd.read_csv(to_native(a.annotated_csv), usecols=["imagename"])["imagename"].astype(str))
        df["annotated_2026"] = df["imagename"].isin(ann)
        print(f"Annotated list: {len(ann):,} names, {df.loc[df['annotated_2026'], 'imagename'].nunique():,} found in inventories")
    else:
        df["annotated_2026"] = False

    # --- frame thinning on the union of drives
    okdf = df[df["exclude_reason"].eq("")]
    if a.frame_step and a.frame_step > 1:
        u = okdf.sort_values("_prefrank").drop_duplicates("imagename")[["imagename", "date", "time", "frame", "drive"]].copy()
        on_pref = set(okdf.loc[okdf["drive"].eq(pref), "imagename"])
        ann_names = set(okdf.loc[okdf["annotated_2026"], "imagename"]) if a.keep_annotated else set()
        u["tier"] = np.where(u["imagename"].isin(ann_names), 0, np.where(u["imagename"].isin(on_pref), 1, 2))
        u = thin_frames(u, a.frame_step)
        keep_names = set(u.loc[u["thin_keep"], "imagename"])
        print(f"Frame thinning (every {a.frame_step}th frame across drives): keep {len(keep_names):,} of "
              f"{len(u):,} unique images ({u['seq'].nunique():,} frame sequences; "
              f"{int((u['thin_keep'] & u['tier'].eq(0)).sum()):,} annotated kept, "
              f"{int((u['thin_keep'] & u['tier'].eq(1)).sum()):,} from {pref}, "
              f"{int((u['thin_keep'] & u['tier'].eq(2)).sum()):,} from other drives only)")
        df["frame_keep"] = df["imagename"].isin(keep_names)
    else:
        df["frame_keep"] = True

    # --- selection and chunks
    df["selected"] = (df["exclude_reason"].eq("") & df["frame_keep"]
                      & (df["is_preferred_copy"] if a.scope == "preferred" else True))
    df["_g"] = df["group"].map(GROUP_ORDER).fillna(9)
    df = df.sort_values(["_g", "date", "time", "frame", "drive"], na_position="last").reset_index(drop=True)
    # chunks never span two groups, so running one group (--groups) never half-finishes a chunk
    df["chunk_id"] = pd.array([pd.NA] * len(df), dtype="Int64")
    offset = 0
    for g in sorted(df["group"].unique(), key=lambda x: GROUP_ORDER.get(x, 9)):
        idx = df.index[df["selected"] & df["group"].eq(g)]
        if len(idx):
            k = np.arange(len(idx)) // a.chunk_size
            df.loc[idx, "chunk_id"] = offset + k + 1
            offset += int(k.max()) + 1
    df["image_uid"] = df["drive"] + ":" + df["imagename"]

    cols = ["image_uid", "imagename", "drive", "src_path", "cruise", "date", "month", "time", "datetime",
            "frame", "group", "n_copies", "is_duplicate", "other_drives", "color_corrected_assumed",
            "is_preferred_copy", "annotated_2026", "exclude_reason", "frame_keep", "selected", "chunk_id"]
    df = df[cols]
    man_path = out / "manifest_2026.csv.gz"
    df.to_csv(man_path, index=False, compression="gzip")

    # --- summary
    s = df[df["exclude_reason"].eq("")]
    by = (s.groupby(["group", "drive"])
            .agg(files=("imagename", "size"), duplicates=("is_duplicate", "sum"),
                 preferred=("is_preferred_copy", "sum"), frame_keep=("frame_keep", "sum"),
                 selected=("selected", "sum"),
                 first_date=("date", "min"), last_date=("date", "max"))
            .reset_index())
    print("\n=== Files by group and drive ===")
    print(by.to_string(index=False))
    print(f"\nUnique imagenames: {s['imagename'].nunique():,};  files: {len(s):,};  "
          f"imagenames on >1 drive: {s.loc[s['is_duplicate'], 'imagename'].nunique():,}")
    print("Excluded:", df.loc[df["exclude_reason"].ne(""), "exclude_reason"].value_counts().to_dict())
    for _, r in df[df["exclude_reason"].ne("")].head(10).iterrows():
        print(f"   {r['exclude_reason']}: {r['src_path']}")
    n_chunks = int(df["chunk_id"].max()) if df["selected"].any() else 0
    cg = df[df["selected"]].groupby("group")["chunk_id"].agg(["min", "max"])
    print(f"\nScope '{a.scope}': {int(df['selected'].sum()):,} files in {n_chunks} chunks of <= {a.chunk_size:,}")
    print(cg.rename(columns={"min": "first_chunk", "max": "last_chunk"}).to_string())

    by.to_csv(out / "manifest_2026_summary.csv", index=False)
    (out / "manifest_2026_config.json").write_text(json.dumps(
        {**vars(a), "n_files": int(len(df)), "n_selected": int(df["selected"].sum()), "n_chunks": n_chunks},
        indent=2) + "\n")
    print(f"\nWrote {man_path}")


if __name__ == "__main__":
    sys.exit(main())
