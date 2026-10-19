#!/usr/bin/env python3
"""
Build YOLO-format evaluation sets for the Tier B (leave-one-year-out) test.

Tier B asks how the 2022-2024 YOLOv12n weights perform on a survey year they never
saw (2026), with the GAM stages fitted only on 2022-2024 images those weights never
trained on. This script builds both evaluation sets in the layout that
config/eval_yolo_detections_hung.py already reads (<out>/yolo/images/val + labels/val),
so the evaluator itself is unchanged.

Subcommands
-----------
annotations : 2026 annotation export -> YOLO val set (every annotated image; images with
              no live-scallop annotation become empties with an empty label file).
list        : a list of existing training-pipeline images (e.g. the 2022-2024 calibration
              pool) -> YOLO val set, re-using their existing YOLO label files.

Conventions copied from training (build_strat_labels_v2.py / manuscript Sec. 2.1.1):
  * line segment -> square box centred on the segment midpoint, side = segment length
  * boundingBox geometry used as-is
  * point annotations cannot make a box; by default the whole image is excluded
    (training used point_annotations_used = False)
  * deprecated annotations dropped
  * near-identical boxes within an image (IoU >= 0.95) collapsed (data/training/dedup_boxes.R)
  * positives = Scallop-category classes naming a live or swimming animal, excluding
    dead, clapper and "possible" classes (resolved table is printed and saved)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
POS_INCLUDE = re.compile(r"\b(?:live|swimming)\b", re.I)
POS_EXCLUDE = re.compile(r"dead|clapper|possible", re.I)


# ----------------------------------------------------------------------------- helpers
def platform_of(name: str) -> str:
    n = name.lower()
    if "stella" in n:
        return "stella"
    if "polaris" in n:
        return "polaris"
    return "primary"


def parse_geom(text) -> tuple[str, list[float]]:
    """Return (kind, numbers) from geometry_text like '"line": [[x1, y1], [x2, y2]]'."""
    if not isinstance(text, str) or not text.strip():
        return "none", []
    m = re.match(r'\s*"?(\w+)"?\s*:', text)
    kind = m.group(1) if m else "unknown"
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)]
    return kind, nums


def geom_to_box(kind: str, nums: list[float]):
    if kind == "line" and len(nums) >= 4:
        x1, y1, x2, y2 = nums[:4]
        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
        r = 0.5 * float(np.hypot(x2 - x1, y2 - y1))
        if r <= 0:
            return None
        return [xc - r, yc - r, xc + r, yc + r]
    if kind == "boundingBox" and len(nums) >= 4:
        x1, y1, x2, y2 = nums[:4]
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return None


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedup(boxes: list[list[float]], thr: float) -> list[list[float]]:
    keep: list[list[float]] = []
    for b in boxes:
        if all(iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def to_yolo(b, w, h):
    x1, y1, x2, y2 = max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def index_images(root: Path) -> dict[str, Path]:
    """basename -> full path for every image under root (follows the nested 10-min folders)."""
    idx: dict[str, Path] = {}
    dupes = 0
    for dirpath, _, files in os.walk(root, followlinks=True):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXT:
                if f in idx:
                    dupes += 1
                    continue
                idx[f] = Path(dirpath) / f
    print(f"Indexed {len(idx):,} images under {root} ({dupes} duplicate basenames ignored)")
    return idx


def link(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        import shutil
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def image_size(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.size


def fresh_dirs(out_root: Path):
    img_dir = out_root / "yolo" / "images" / "val"
    lab_dir = out_root / "yolo" / "labels" / "val"
    for d in (img_dir, lab_dir):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()
    return img_dir, lab_dir


# ----------------------------------------------------------------------------- annotations mode
def run_annotations(a):
    ann = pd.read_csv(a.ann_csv, low_memory=False)
    sp = pd.read_csv(a.species_csv)
    sp.columns = [c.strip().upper() for c in sp.columns]
    sp["CLASS_NAME"] = sp["CLASS_NAME"].astype(str).str.strip()
    sc = sp[sp["CLASS_CATEGORY_PHYLUM"].astype(str).str.strip().eq("Scallop")]
    pos = sc[sc["CLASS_NAME"].str.contains(POS_INCLUDE) & ~sc["CLASS_NAME"].str.contains(POS_EXCLUDE)]
    pos_ids = set(pos["CLASS_ID"].astype(int))

    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    pos[["CLASS_ID", "CLASS_NAME"]].sort_values("CLASS_ID").to_csv(out_root / "positive_classes.csv", index=False)
    print(f"Positive classes ({len(pos_ids)}):")
    print(pos[["CLASS_ID", "CLASS_NAME"]].sort_values("CLASS_NAME").to_string(index=False))

    n0 = len(ann)
    if "deprecated" in ann:
        ann = ann[~ann["deprecated"].astype(str).str.lower().eq("true")]
    print(f"Annotations: {n0:,} rows, {n0 - len(ann):,} deprecated dropped")

    ann["platform"] = ann["imagename"].map(platform_of)
    if a.platforms:
        keep = {p.strip().lower() for p in a.platforms.split(",")}
        ann = ann[ann["platform"].isin(keep)]

    ann["is_pos"] = ann["class_id"].isin(pos_ids)
    geo = ann.loc[ann["is_pos"], "geometry_text"].map(parse_geom)
    ann.loc[ann["is_pos"], "gkind"] = geo.map(lambda g: g[0])
    ann.loc[ann["is_pos"], "gnums"] = geo.map(lambda g: g[1])

    idx = {} if a.dry_run else index_images(Path(a.image_root))
    img_dir, lab_dir = (None, None) if a.dry_run else fresh_dirs(out_root)

    manifest, gt_rows = [], []
    for name, sub in ann.groupby("imagename", sort=True):
        p = sub[sub["is_pos"]]
        row = {"imagename": name, "platform": sub["platform"].iat[0],
               "n_raw_pos": len(p), "n_points": int((p["gkind"] == "point").sum()) if len(p) else 0,
               "img_path": "", "img_w": np.nan, "img_h": np.nan,
               "n_annotations": 0, "is_empty": True, "excluded": "",
               "n_dropped_right": 0, "eval_w": np.nan, "cropped": False}
        if row["n_points"] and a.point_policy == "drop_image":
            row["excluded"] = "point annotation"
        src = None
        if not a.dry_run:
            src = idx.get(name)
            if src is None:
                row["excluded"] = row["excluded"] or "missing from disk"
        if row["excluded"]:
            manifest.append(row)
            continue

        boxes = [geom_to_box(k, n) for k, n in zip(p["gkind"], p["gnums"])]
        boxes = dedup([b for b in boxes if b is not None], a.dedup_iou)
        if a.crop_w:
            # 2026 PNGs are side-by-side stereo pairs (3872 x 1216); annotations and the
            # 2022-2024 training frames are the LEFT camera (1936 px). Keep boxes whose
            # centre lies in the left frame.
            n_before = len(boxes)
            boxes = [b for b in boxes if (b[0] + b[2]) / 2 < a.crop_w]
            row["n_dropped_right"] = n_before - len(boxes)
        row["n_annotations"] = len(boxes)
        row["is_empty"] = len(boxes) == 0
        for (k, n, cid) in zip(p["gkind"], p["gnums"], p["class_id"]):
            if k == "line" and len(n) >= 4:
                gt_rows.append({"imagename": name, "class_id": int(cid),
                                "x1": n[0], "y1": n[1], "x2": n[2], "y2": n[3],
                                "length_px": float(np.hypot(n[2] - n[0], n[3] - n[1]))})

        if not a.dry_run:
            try:
                w, h = image_size(src)
            except Exception as e:
                row["excluded"] = f"unreadable: {e}"
                manifest.append(row)
                continue
            row.update(img_path=str(src), img_w=w, img_h=h)
            cropped = bool(a.crop_w) and w > a.crop_w
            if cropped:
                w = a.crop_w
            row.update(eval_w=w, cropped=cropped)
            lines = []
            for b in boxes:
                y = to_yolo(b, w, h)
                if y:
                    lines.append(f"0 {y[0]:.6f} {y[1]:.6f} {y[2]:.6f} {y[3]:.6f}")
            # Full image is linked (no 90k extra PNGs on disk); labels are normalised to the
            # left frame, and eval_yolo_detections_hung.py --crop_w crops at inference time.
            link(src, img_dir / name, a.link_mode)
            (lab_dir / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        manifest.append(row)

    man = pd.DataFrame(manifest)
    man.to_csv(out_root / "tierb_manifest.csv", index=False)
    pd.DataFrame(gt_rows).to_csv(out_root / "tierb_gt_lines.csv", index=False)
    used = man[man["excluded"].eq("")]
    print("\n=== Tier B evaluation set ===")
    print(used.groupby("platform").agg(images=("imagename", "size"),
                                      scallop_images=("is_empty", lambda s: int((~s).sum())),
                                      empty_images=("is_empty", "sum"),
                                      scallops=("n_annotations", "sum")).to_string())
    print("\nExcluded images:", man.loc[man["excluded"].ne(""), "excluded"].value_counts().to_dict())
    print(f"Raw positive annotations: {int(man['n_raw_pos'].sum()):,}; after point exclusion + dedup: "
          f"{int(used['n_annotations'].sum()):,}")
    print(f"Wrote {out_root/'tierb_manifest.csv'} and {out_root/'tierb_gt_lines.csv'}")


# ----------------------------------------------------------------------------- list mode
def run_list(a):
    names = pd.read_csv(a.image_list)["imagename"].astype(str).tolist()
    src_root = Path(a.src_yolo_root)
    out_root = Path(a.out_root)
    img_dir, lab_dir = fresh_dirs(out_root)
    rows, missing = [], 0
    for n in names:
        src = next((src_root / "images" / s / n for s in ("train", "val")
                    if (src_root / "images" / s / n).exists()), None)
        if src is None:
            missing += 1
            rows.append({"imagename": n, "excluded": "missing"})
            continue
        split = src.parent.name
        lab = src_root / "labels" / split / f"{Path(n).stem}.txt"
        link(src, img_dir / n, a.link_mode)
        text = lab.read_text() if lab.exists() else ""
        (lab_dir / f"{Path(n).stem}.txt").write_text(text)
        rows.append({"imagename": n, "excluded": "", "src_split": split,
                     "n_annotations": sum(1 for l in text.splitlines() if l.strip())})
    pd.DataFrame(rows).to_csv(out_root / "list_manifest.csv", index=False)
    print(f"Linked {len(names) - missing:,} of {len(names):,} images ({missing} missing) into {img_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    s1 = sub.add_parser("annotations")
    s1.add_argument("--ann_csv", required=True)
    s1.add_argument("--species_csv", required=True)
    s1.add_argument("--image_root", required=False, default="")
    s1.add_argument("--out_root", required=True)
    s1.add_argument("--platforms", default="primary,polaris,stella")
    s1.add_argument("--dedup_iou", type=float, default=0.95)
    s1.add_argument("--point_policy", choices=["drop_image", "ignore"], default="drop_image")
    s1.add_argument("--link_mode", choices=["symlink", "copy", "hardlink"], default="symlink")
    s1.add_argument("--dry_run", action="store_true", help="Manifests only; no image index, links or labels")
    s1.add_argument("--crop_w", type=int, default=1936,
                    help="Images wider than this are stereo pairs: labels are normalised to the left frame [0, crop_w). "
                         "Run the evaluator with the same --crop_w. 0 = full image")

    s2 = sub.add_parser("list")
    s2.add_argument("--image_list", required=True, help="CSV with an 'imagename' column")
    s2.add_argument("--src_yolo_root", required=True, help="Existing YOLO root with images/{train,val}, labels/{train,val}")
    s2.add_argument("--out_root", required=True)
    s2.add_argument("--link_mode", choices=["symlink", "copy", "hardlink"], default="symlink")

    a = ap.parse_args()
    if a.mode == "annotations":
        if not a.dry_run and not a.image_root:
            sys.exit("--image_root is required unless --dry_run")
        run_annotations(a)
    else:
        run_list(a)


if __name__ == "__main__":
    main()
