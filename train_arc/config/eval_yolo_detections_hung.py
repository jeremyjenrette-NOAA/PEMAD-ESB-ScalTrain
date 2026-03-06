from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment

import numpy as np
import pandas as pd
from ultralytics import YOLO
import json
import yaml

def hungarian_match(
    pred_xyxy: np.ndarray,
    gt_xyxy: np.ndarray,
    iou_thr: float,
    big_cost: float = 1e6
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    Hungarian 1:1 matching maximizing IoU (minimizing 1-IoU),
    with a hard gate: IoU must be >= iou_thr to be considered a valid match.

    Returns:
      pred_to_gt: pred_idx -> gt_idx
      pred_to_iou: pred_idx -> iou
    """
    P = pred_xyxy.shape[0]
    G = gt_xyxy.shape[0]
    if P == 0 or G == 0:
        return {}, {}

    ious = iou_xyxy(pred_xyxy, gt_xyxy)  # (P, G)

    # Cost is 1 - IoU (lower is better). Forbid low-IoU matches.
    cost = 1.0 - ious
    cost = cost.astype(np.float32)

    # Gate: anything below threshold becomes "effectively impossible"
    cost[ious < iou_thr] = big_cost

    # Solve assignment (works for rectangular matrices too)
    row_ind, col_ind = linear_sum_assignment(cost)

    pred_to_gt: Dict[int, int] = {}
    pred_to_iou: Dict[int, float] = {}

    for p, g in zip(row_ind, col_ind):
        p = int(p); g = int(g)
        if p < P and g < G:
            iou = float(ious[p, g])
            # Keep only if it passed the gate (some assignments can still be big_cost)
            if iou >= iou_thr and cost[p, g] < big_cost:
                pred_to_gt[p] = g
                pred_to_iou[p] = iou

    return pred_to_gt, pred_to_iou

def count_pos_neg(img_dir: Path, lab_dir: Path) -> tuple[int, int, int]:
    """Return (n_images, n_pos, n_neg) where pos has a non-empty label txt."""
    imgs = [p for p in img_dir.glob("*") if p.is_file()]
    n_images = len(imgs)
    n_pos = 0
    n_neg = 0
    for im in imgs:
        lab = lab_dir / f"{im.stem}.txt"
        if lab.exists() and lab.read_text().strip():
            n_pos += 1
        else:
            n_neg += 1
    return n_images, n_pos, n_neg

def try_read_ultralytics_args(run_dir: Path) -> dict:
    """Read run_dir/args.yaml if it exists."""
    p = run_dir / "args.yaml"
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between a (N,4) and b (M,4) in xyxy."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1 = a[:, 0:1]
    ay1 = a[:, 1:2]
    ax2 = a[:, 2:3]
    ay2 = a[:, 3:4]

    bx1 = b[:, 0]
    by1 = b[:, 1]
    bx2 = b[:, 2]
    by2 = b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a + area_b - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def yolo_txt_to_xyxy(label_txt: Path, img_w: int, img_h: int) -> np.ndarray:
    """
    YOLO txt format per line: cls xc yc w h (normalized)
    Returns xyxy in pixel coords.
    """
    if not label_txt.exists():
        return np.zeros((0, 4), dtype=np.float32)

    text = label_txt.read_text().strip()
    if not text:
        return np.zeros((0, 4), dtype=np.float32)

    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = map(float, parts[:5])
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h
        rows.append([x1, y1, x2, y2])

    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def greedy_match(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray, iou_thr: float) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    Greedy 1:1 matching between predictions and GT at IoU >= iou_thr.
    Returns:
      pred_to_gt: pred_idx -> gt_idx
      pred_to_iou: pred_idx -> iou
    """
    P = pred_xyxy.shape[0]
    G = gt_xyxy.shape[0]
    if P == 0 or G == 0:
        return {}, {}

    ious = iou_xyxy(pred_xyxy, gt_xyxy)
    pairs = np.argwhere(ious >= iou_thr)
    if pairs.size == 0:
        return {}, {}

    pair_ious = ious[pairs[:, 0], pairs[:, 1]]
    order = np.argsort(-pair_ious)
    pairs = pairs[order]
    pair_ious = pair_ious[order]

    used_p = set()
    used_g = set()
    pred_to_gt: Dict[int, int] = {}
    pred_to_iou: Dict[int, float] = {}

    for (p, g), v in zip(pairs, pair_ious):
        p = int(p); g = int(g)
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        pred_to_gt[p] = g
        pred_to_iou[p] = float(v)

    return pred_to_gt, pred_to_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to best.pt")
    ap.add_argument("--data_root", required=True, help="YOLO dataset root containing images/val and labels/val")
    ap.add_argument("--out_csv", required=True, help="Output CSV (detections table)")
    ap.add_argument("--out_fn_csv", required=False, default=None, help="Optional output CSV for missed GT boxes (FNs)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001, help="Prediction conf threshold (set low for calibration)")
    ap.add_argument("--nms_iou", type=float, default=0.7, help="NMS IoU for prediction")
    ap.add_argument("--match_iou", type=float, default=0.5, help="IoU threshold for TP matching vs GT")
    ap.add_argument("--max_det", type=int, default=600)
    ap.add_argument("--device", default="0")
    ap.add_argument("--spname", default="scallop")
    ap.add_argument("--debug_n", type=int, default=10, help="Print debug for first N images with GT")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--model_name", default="", help="e.g. yolov10n or check/yolov10n.pt (for metadata only)")
    ap.add_argument("--epochs", type=int, default=None, help="If not provided, try reading from args.yaml")
    ap.add_argument("--batch", type=int, default=None, help="If not provided, try reading from args.yaml")
    ap.add_argument("--point_annotations_used", action="store_true",
                    help="Set TRUE only if training included point-derived annotations (default FALSE).")
    ap.add_argument("--run_dir", default=None,
                    help="Ultralytics run directory that contains args.yaml (optional but recommended).")
    ap.add_argument("--gt_out_csv", default=None,
                help="If set, writes a GT-only CSV (one row per GT box in val set).")
    args = ap.parse_args()

    weights = Path(args.weights)
    data_root = Path(args.data_root)
    img_dir = data_root / "images" / "val"
    lab_dir = data_root / "labels" / "val"

    gt_rows = []
    gt_detectid = 0

    images = sorted([p for p in img_dir.glob("*") if p.is_file()])
    if not images:
        raise ValueError(f"No images found in {img_dir}")
    if not lab_dir.exists():
        raise ValueError(f"Label dir not found: {lab_dir}")

    model = YOLO(str(weights))

    # Run prediction in streaming mode, but map results back to the *known* image list.
    # This prevents Imagename becoming image0.jpg etc.
    results_iter = model.predict(
        source=[str(p) for p in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        device=args.device,
        stream=True,
        verbose=False,
        batch=1,
        half=True
    )

    det_rows: List[dict] = []
    fn_rows: List[dict] = []

    detectid = 0
    debug_printed = 0

    for img_path, res in zip(images, results_iter):
        fname = img_path.name

        # Use PIL for trustworthy dims + avoids any ambiguity about res.orig_shape
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            print(f"WARNING: cannot open {img_path}: {e}")
            continue

        # GT boxes from yolo label file (same stem)
        gt_txt = lab_dir / f"{img_path.stem}.txt"
        gt_xyxy = yolo_txt_to_xyxy(gt_txt, w, h)

        if args.gt_out_csv:
            for g in range(gt_xyxy.shape[0]):
                gx1, gy1, gx2, gy2 = gt_xyxy[g].tolist()
                gboxsize = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)

                gt_detectid += 1
                gt_rows.append({
                    "Detectid": gt_detectid,
                    "Imagename": fname,
                    "FrameID": 0,
                    "TLx": gx1,
                    "TLy": gy1,
                    "BRx": gx2,
                    "BRy": gy2,
                    "Conf": 1.0,
                    "Len": 0,
                    "Spname": args.spname,
                    # "ConfPairs": 1.0,
                    "boxsize": gboxsize,
                    # "truedetect": True,
                    # "iu": 1.0,
                    "index": g + 1,   # 1-based like your auto table
                    # "man": "1",
                    # "bin": 100,
                    "gt_id": g,       # extra (optional)
                })

        # Predictions (xyxy in original pixel coords)
        if res.boxes is None or len(res.boxes) == 0:
            pred_xyxy = np.zeros((0, 4), dtype=np.float32)
            pred_conf = np.zeros((0,), dtype=np.float32)
        else:
            pred_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_conf = res.boxes.conf.cpu().numpy().astype(np.float32)

        # Hungarian algorithm matching
        pred_to_gt, pred_to_iou = hungarian_match(pred_xyxy, gt_xyxy, iou_thr=args.match_iou)
        matched_gt = set(pred_to_gt.values())

        # Debug: show first few GT-bearing images to confirm GT is being read
        if debug_printed < args.debug_n and gt_xyxy.shape[0] > 0:
            max_iou_any = float(iou_xyxy(pred_xyxy, gt_xyxy).max()) if (pred_xyxy.shape[0] and gt_xyxy.shape[0]) else 0.0
            print(f"[DEBUG] {fname}  n_gt={gt_xyxy.shape[0]}  n_pred={pred_xyxy.shape[0]}  max_iou_any={max_iou_any:.3f}  gt_exists={gt_txt.exists()}")
            debug_printed += 1

        # Detection-level rows: match your header style
        # Detectid,Imagename,FrameID,TLx,TLy,BRx,BRy,Conf,Len,Spname,ConfPairs,boxsize,truedetect,iu,index,man,bin
        for i in range(pred_xyxy.shape[0]):
            x1, y1, x2, y2 = pred_xyxy[i].tolist()
            conf = float(pred_conf[i]) if pred_conf.size else np.nan
            boxsize = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            truedetect = i in pred_to_gt
            iu = float(pred_to_iou.get(i, 0.0)) if truedetect else 0.0

            detectid += 1
            det_rows.append({
                "Detectid": detectid,
                "Imagename": fname,
                "FrameID": 0,
                "TLx": x1,
                "TLy": y1,
                "BRx": x2,
                "BRy": y2,
                "Conf": conf,
                "Len": 0,
                "Spname": args.spname,
                "ConfPairs": conf,
                "boxsize": boxsize,
                "truedetect": bool(truedetect),
                "iu": iu,
                "index": i + 1,     # 1-based like your example
                "man": "",
                "bin": int(round(conf * 100)) if conf == conf else ""
            })

        # FN table (optional): GT boxes not matched
        if args.out_fn_csv:
            for g in range(gt_xyxy.shape[0]):
                if g in matched_gt:
                    continue
                gx1, gy1, gx2, gy2 = gt_xyxy[g].tolist()
                gboxsize = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
                fn_rows.append({
                    "Imagename": fname,
                    "FrameID": 0,
                    "TLx": gx1, "TLy": gy1, "BRx": gx2, "BRy": gy2,
                    "Spname": args.spname,
                    "boxsize": gboxsize,
                    "missed": True,
                })

    det_df = pd.DataFrame(det_rows)

    # Enforce column order exactly
    col_order = [
        "Detectid","Imagename","FrameID","TLx","TLy","BRx","BRy","Conf","Len","Spname",
        "ConfPairs","boxsize","truedetect","iu","index","man","bin"
    ]
    for c in col_order:
        if c not in det_df.columns:
            det_df[c] = ""
    det_df = det_df[col_order]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    det_df.to_csv(out_csv, index=False)
    print(f"Wrote detections CSV: {out_csv}  (rows={len(det_df)})")

    # ---------------- Run summary ----------------
    train_img_dir = data_root / "images" / "train"
    train_lab_dir = data_root / "labels" / "train"
    val_img_dir   = data_root / "images" / "val"
    val_lab_dir   = data_root / "labels" / "val"

    n_train_images, n_train_pos, n_train_neg = count_pos_neg(train_img_dir, train_lab_dir)
    n_val_images, n_val_pos, n_val_neg = count_pos_neg(val_img_dir, val_lab_dir)

    ultra_args = {}
    if args.run_dir:
        ultra_args = try_read_ultralytics_args(Path(args.run_dir))

    # Prefer CLI-provided epochs/batch, else fallback to args.yaml if present
    epochs = args.epochs if args.epochs is not None else ultra_args.get("epochs", None)
    batch  = args.batch  if args.batch  is not None else ultra_args.get("batch", None)

    summary = {
        "year": args.year,
        "model": args.model_name or ultra_args.get("model", ""),
        "imgsz": args.imgsz,
        "epochs": epochs,
        "batch": batch,

        "data_root": str(data_root),
        "weights": str(weights),
        "run_dir": str(args.run_dir) if args.run_dir else "",

        "n_train_images": n_train_images,
        "n_train_pos": n_train_pos,
        "n_train_neg": n_train_neg,

        "n_val_images": n_val_images,
        "n_val_pos": n_val_pos,
        "n_val_neg": n_val_neg,

        "neg_ratio_observed_train": (n_train_neg / max(n_train_pos, 1)),
        "neg_ratio_observed_val": (n_val_neg / max(n_val_pos, 1)),

        "eval_conf": args.conf,
        "nms_iou": args.nms_iou,
        "match_iou": args.match_iou,
        "max_det": args.max_det,

        "point_annotations_used": bool(args.point_annotations_used),
    }

    # Include a few extra ultralytics args if available (nice-to-have)
    for k in ["optimizer", "lr0", "lrf", "weight_decay", "close_mosaic", "seed", "patience"]:
        if k in ultra_args and k not in summary:
            summary[k] = ultra_args[k]

    # Write JSON + one-row CSV next to out_csv
    out_csv_path = Path(args.out_csv)
    summary_json = out_csv_path.parent / "run_summary.json"
    summary_csv  = out_csv_path.parent / "run_summary.csv"

    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    if args.gt_out_csv:
        gt_df = pd.DataFrame(gt_rows)

        # Keep same core order + optional extras at end
        base_cols = [
            "Detectid","Imagename","FrameID","TLx","TLy","BRx","BRy","Conf","Len","Spname",
            "ConfPairs","boxsize","truedetect","iu","index","man","bin"
        ]
        for c in base_cols:
            if c not in gt_df.columns:
                gt_df[c] = ""

        extra_cols = [c for c in gt_df.columns if c not in base_cols]
        gt_df = gt_df[base_cols + extra_cols]

        out_gt = Path(args.gt_out_csv)
        out_gt.parent.mkdir(parents=True, exist_ok=True)
        gt_df.to_csv(out_gt, index=False)
        print(f"Wrote GT CSV: {out_gt} (rows={len(gt_df)})")

    print(f"Wrote run summary: {summary_json}")
    print(f"Wrote run summary: {summary_csv}")

    if args.out_fn_csv:
        fn_df = pd.DataFrame(fn_rows)
        out_fn = Path(args.out_fn_csv)
        out_fn.parent.mkdir(parents=True, exist_ok=True)
        fn_df.to_csv(out_fn, index=False)
        print(f"Wrote FN CSV: {out_fn}  (rows={len(fn_df)})")

    # Quick sanity summary
    if len(det_df):
        print("TP count:", int(det_df["truedetect"].sum()))
        print("FP count:", int((~det_df["truedetect"]).sum()))
        if det_df["iu"].max() == 0:
            print("WARNING: iu max is 0.0. This almost always means GT labels were not loaded or stems don't match.")


if __name__ == "__main__":
    main()