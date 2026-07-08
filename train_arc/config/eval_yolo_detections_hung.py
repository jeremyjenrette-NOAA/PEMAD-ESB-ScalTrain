from __future__ import annotations

import argparse
import csv
import gc
import torch
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
    P = pred_xyxy.shape[0]
    G = gt_xyxy.shape[0]
    if P == 0 or G == 0:
        return {}, {}

    ious = iou_xyxy(pred_xyxy, gt_xyxy)
    cost = 1.0 - ious
    cost = cost.astype(np.float32)
    cost[ious < iou_thr] = big_cost

    row_ind, col_ind = linear_sum_assignment(cost)

    pred_to_gt: Dict[int, int] = {}
    pred_to_iou: Dict[int, float] = {}

    for p, g in zip(row_ind, col_ind):
        p = int(p); g = int(g)
        if p < P and g < G:
            iou = float(ious[p, g])
            if iou >= iou_thr and cost[p, g] < big_cost:
                pred_to_gt[p] = g
                pred_to_iou[p] = iou

    return pred_to_gt, pred_to_iou

def count_pos_neg(img_dir: Path, lab_dir: Path) -> tuple[int, int, int]:
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
    p = run_dir / "args.yaml"
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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

    images = sorted([p for p in img_dir.glob("*") if p.is_file()])
    if not images:
        raise ValueError(f"No images found in {img_dir}")
    if not lab_dir.exists():
        raise ValueError(f"Label dir not found: {lab_dir}")

    model = YOLO(str(weights))

    # ---------------- Setup Streaming CSV Writers ----------------
    det_cols = [
        "Detectid","Imagename","FrameID","TLx","TLy","BRx","BRy","Conf","Len","Spname",
        "ConfPairs","boxsize","truedetect","iu","index","man","bin"
    ]
    gt_cols = det_cols + ["gt_id"]
    fn_cols = ["Imagename","FrameID","TLx","TLy","BRx","BRy","Spname","boxsize","missed"]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f_det = open(out_csv, mode='w', newline='')
    det_writer = csv.DictWriter(f_det, fieldnames=det_cols)
    det_writer.writeheader()

    f_gt, gt_writer = None, None
    if args.gt_out_csv:
        gt_csv = Path(args.gt_out_csv)
        gt_csv.parent.mkdir(parents=True, exist_ok=True)
        f_gt = open(gt_csv, mode='w', newline='')
        gt_writer = csv.DictWriter(f_gt, fieldnames=gt_cols)
        gt_writer.writeheader()

    f_fn, fn_writer = None, None
    if args.out_fn_csv:
        fn_csv = Path(args.out_fn_csv)
        fn_csv.parent.mkdir(parents=True, exist_ok=True)
        f_fn = open(fn_csv, mode='w', newline='')
        fn_writer = csv.DictWriter(f_fn, fieldnames=fn_cols)
        fn_writer.writeheader()

    # Trackers for the final summary printout
    detectid = 0
    gt_detectid = 0
    debug_printed = 0
    tp_count = 0
    fp_count = 0
    max_iu_found = 0.0

    # ---------------- Process Image Stream in Chunks ----------------
    # Feed the model EXACTLY the batch size. This prevents Ultralytics from 
    # pre-allocating massive VRAM tensors under the hood.
    chunk_size = args.batch if args.batch and args.batch > 0 else 4
    
    for chunk_start in range(0, len(images), chunk_size):
        chunk_images = images[chunk_start : chunk_start + chunk_size]
        
        results_iter = model.predict(
            source=[str(p) for p in chunk_images],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            device=args.device,
            stream=True,
            verbose=False,
            batch=args.batch,
            quantize=True
        )

        for img_path, res in zip(chunk_images, results_iter):
            fname = img_path.name

            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as e:
                print(f"WARNING: cannot open {img_path}: {e}")
                continue

            gt_txt = lab_dir / f"{img_path.stem}.txt"
            gt_xyxy = yolo_txt_to_xyxy(gt_txt, w, h)

            # Write Ground Truths incrementally
            if gt_writer:
                for g in range(gt_xyxy.shape[0]):
                    gx1, gy1, gx2, gy2 = gt_xyxy[g].tolist()
                    gboxsize = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
                    gt_detectid += 1
                    
                    gt_row = {col: "" for col in gt_cols}
                    gt_row.update({
                        "Detectid": gt_detectid, "Imagename": fname, "FrameID": 0,
                        "TLx": gx1, "TLy": gy1, "BRx": gx2, "BRy": gy2,
                        "Conf": 1.0, "Len": 0, "Spname": args.spname,
                        "boxsize": gboxsize, "index": g + 1, "gt_id": g
                    })
                    gt_writer.writerow(gt_row)

            if res.boxes is None or len(res.boxes) == 0:
                pred_xyxy = np.zeros((0, 4), dtype=np.float32)
                pred_conf = np.zeros((0,), dtype=np.float32)
            else:
                pred_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                pred_conf = res.boxes.conf.cpu().numpy().astype(np.float32)

            pred_to_gt, pred_to_iou = hungarian_match(pred_xyxy, gt_xyxy, iou_thr=args.match_iou)
            matched_gt = set(pred_to_gt.values())

            if debug_printed < args.debug_n and gt_xyxy.shape[0] > 0:
                max_iou_any = float(iou_xyxy(pred_xyxy, gt_xyxy).max()) if (pred_xyxy.shape[0] and gt_xyxy.shape[0]) else 0.0
                print(f"[DEBUG] {fname}  n_gt={gt_xyxy.shape[0]}  n_pred={pred_xyxy.shape[0]}  max_iou_any={max_iou_any:.3f}  gt_exists={gt_txt.exists()}")
                debug_printed += 1

            # Write Detections incrementally
            for i in range(pred_xyxy.shape[0]):
                x1, y1, x2, y2 = pred_xyxy[i].tolist()
                conf = float(pred_conf[i]) if pred_conf.size else np.nan
                boxsize = max(0.0, x2 - x1) * max(0.0, y2 - y1)

                truedetect = i in pred_to_gt
                iu = float(pred_to_iou.get(i, 0.0)) if truedetect else 0.0

                if truedetect:
                    tp_count += 1
                else:
                    fp_count += 1
                max_iu_found = max(max_iu_found, iu)

                detectid += 1
                
                det_row = {
                    "Detectid": detectid, "Imagename": fname, "FrameID": 0,
                    "TLx": x1, "TLy": y1, "BRx": x2, "BRy": y2,
                    "Conf": conf, "Len": 0, "Spname": args.spname,
                    "ConfPairs": conf, "boxsize": boxsize,
                    "truedetect": bool(truedetect), "iu": iu,
                    "index": i + 1, "man": "",
                    "bin": int(round(conf * 100)) if conf == conf else ""
                }
                det_writer.writerow(det_row)

            # Write False Negatives incrementally
            if fn_writer:
                for g in range(gt_xyxy.shape[0]):
                    if g in matched_gt:
                        continue
                    gx1, gy1, gx2, gy2 = gt_xyxy[g].tolist()
                    gboxsize = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
                    
                    fn_writer.writerow({
                        "Imagename": fname, "FrameID": 0,
                        "TLx": gx1, "TLy": gy1, "BRx": gx2, "BRy": gy2,
                        "Spname": args.spname, "boxsize": gboxsize, "missed": True
                    })

        # --- EXPLICIT GARBAGE COLLECTION ---
        # This prevents PyTorch and Ultralytics from endlessly hoarding memory across chunks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Close file handles to finalize saving
    f_det.close()
    if f_gt:
        f_gt.close()
    if f_fn:
        f_fn.close()

    print(f"Wrote detections CSV: {out_csv}  (rows={detectid})")
    if args.gt_out_csv:
        print(f"Wrote GT CSV: {args.gt_out_csv} (rows={gt_detectid})")
    if args.out_fn_csv:
        print(f"Wrote FN CSV: {args.out_fn_csv}")

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

    for k in ["optimizer", "lr0", "lrf", "weight_decay", "close_mosaic", "seed", "patience"]:
        if k in ultra_args and k not in summary:
            summary[k] = ultra_args[k]

    summary_json = out_csv.parent / "run_summary.json"
    summary_csv  = out_csv.parent / "run_summary.csv"

    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print(f"Wrote run summary: {summary_json}")
    print(f"Wrote run summary: {summary_csv}")

    if detectid > 0:
        print("TP count:", tp_count)
        print("FP count:", fp_count)
        if max_iu_found == 0.0:
            print("WARNING: iu max is 0.0. This almost always means GT labels were not loaded or stems don't match.")

if __name__ == "__main__":
    main()