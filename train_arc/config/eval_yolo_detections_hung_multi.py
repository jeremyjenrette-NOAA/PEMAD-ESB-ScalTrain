#!/usr/bin/env python3
# ==============================================================================
# File: config/eval_yolo_detections_hung_multi.py
# Purpose: Stage 1 Hungarian matching evaluation for broad object detector.
#          Generates autotest.csv, mantest.csv, fn.csv, links manual 'man' IDs,
#          and preserves original multi-class species in 'gt_label'.
#
#          Validates the taxonomy JSON against gt_data_root/data.yaml before
#          computing gt_label: if the taxonomy's class-id -> name mapping
#          doesn't match what's actually baked into the original multi-class
#          YOLO dataset, every gt_label in autotest.csv would otherwise be
#          silently wrong.
# ==============================================================================
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from taxonomy_utils import load_taxonomy, validate_taxonomy_against_yolo


def parse_gt_class_names(taxonomy_json_path: str, gt_data_root: Path, skip_validation: bool = False) -> list[str]:
    """
    Dynamically parses multi-class ground truth names ordered by integer class ID,
    validating first that the taxonomy JSON agrees with gt_data_root/data.yaml
    (the ORIGINAL multi-class YOLO dataset, not the broad single-class one).
    """
    taxonomy = load_taxonomy(taxonomy_json_path)

    data_yaml = gt_data_root / "data.yaml"
    if not skip_validation:
        validate_taxonomy_against_yolo(taxonomy, str(data_yaml), taxonomy_json_path=taxonomy_json_path)

    classes = taxonomy.get("classes", {})
    max_idx = max([int(k) for k in classes.keys()]) if classes else -1
    gt_names = []
    for idx in range(max_idx + 1):
        str_idx = str(idx)
        if str_idx in classes:
            gt_names.append(classes[str_idx]["name"])
        else:
            gt_names.append(f"unknown_{idx}")
    return gt_names


def clean_string_label(raw_label: str) -> str:
    return str(raw_label).replace("[", "").replace("]", "").replace(",", "").replace("'", "").replace('"', "").strip()


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1, inter_y1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    inter_x2, inter_y2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)

    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a + area_b - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def hungarian_match(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray, iou_thr: float, big_cost: float = 1e6) -> Tuple[Dict[int, int], Dict[int, float]]:
    P, G = pred_xyxy.shape[0], gt_xyxy.shape[0]
    if P == 0 or G == 0:
        return {}, {}

    ious = iou_xyxy(pred_xyxy, gt_xyxy)
    cost = (1.0 - ious).astype(np.float32)
    cost[ious < iou_thr] = big_cost

    row_ind, col_ind = linear_sum_assignment(cost)
    pred_to_gt, pred_to_iou = {}, {}

    for p, g in zip(row_ind, col_ind):
        p, g = int(p), int(g)
        if p < P and g < G:
            iou = float(ious[p, g])
            if iou >= iou_thr and cost[p, g] < big_cost:
                pred_to_gt[p] = g
                pred_to_iou[p] = iou

    return pred_to_gt, pred_to_iou


def yolo_txt_to_xyxy_and_cls(label_txt: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    if not label_txt.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    text = label_txt.read_text().strip()
    if not text:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    rows, classes = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])

        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h

        rows.append([x1, y1, x2, y2])
        classes.append(cls_id)

    if not rows:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.array(rows, dtype=np.float32), np.array(classes, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(description="Stage 1 Broad Detector Evaluation")
    ap.add_argument("--weights", required=True, help="Path to broad model best.pt")
    ap.add_argument("--data_root", required=True, help="Path to yolo_broad root")
    ap.add_argument("--gt_data_root", default=None, help="Path to original multi-class YOLO root")
    ap.add_argument("--taxonomy_json", default=None, help="Path to taxonomy JSON configuration")
    ap.add_argument("--out_csv", required=True, help="Output autotest.csv path")
    ap.add_argument("--out_fn_csv", default=None, help="Output fn.csv path for missed GTs")
    ap.add_argument("--gt_out_csv", default=None, help="Output mantest.csv path for ground truths")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--nms_iou", type=float, default=0.65)
    ap.add_argument("--match_iou", type=float, default=0.1)
    ap.add_argument("--max_det", type=int, default=600)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=4, help="Batch size for model prediction streaming")
    ap.add_argument("--spname", nargs="+", default=None, help="Broad class name override (e.g. cancer_crab)")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--skip_taxonomy_validation", action="store_true",
                     help="Skip the taxonomy/data.yaml cross-check. Not recommended — "
                          "this is what catches silent gt_label mislabeling.")
    args = ap.parse_args()

    weights = Path(args.weights)
    data_root = Path(args.data_root)
    gt_data_root = Path(args.gt_data_root) if args.gt_data_root else data_root

    if args.gt_data_root is None:
        print("WARNING: --gt_data_root not provided, falling back to --data_root (yolo_broad). "
              "That directory's labels are single-class and carry no species info — "
              "gt_label will be meaningless. Pass --gt_data_root explicitly.")

    img_dir = data_root / "images" / "val"
    gt_lab_dir = gt_data_root / "labels" / "val"

    images = sorted([p for p in img_dir.glob("*") if p.is_file()])
    if not images:
        raise ValueError(f"No images found in {img_dir}")

    model = YOLO(str(weights))

    # 1. Broad Detector Prediction Class Name (e.g. cancer_crab)
    if args.spname:
        pred_class_names = [clean_string_label(name) for name in args.spname]
    elif hasattr(model, "names") and isinstance(model.names, (dict, list)):
        names_val = model.names.values() if isinstance(model.names, dict) else model.names
        pred_class_names = [clean_string_label(n) for n in names_val]
    else:
        pred_class_names = ["broad_object"]

    # 2. Multi-Class Ground Truth Species Names — validated against gt_data_root/data.yaml
    if args.taxonomy_json and Path(args.taxonomy_json).exists():
        gt_class_names = parse_gt_class_names(
            args.taxonomy_json, gt_data_root, skip_validation=args.skip_taxonomy_validation
        )
    else:
        gt_class_names = pred_class_names

    base_cols = [
        "Detectid", "Imagename", "FrameID", "TLx", "TLy", "BRx", "BRy", "Conf", "Len", "Spname",
        "ConfPairs", "boxsize", "truedetect", "iu", "index", "man", "bin"
    ]
    broad_conf_cols = [f"conf_{cls}" for cls in pred_class_names]
    det_cols = base_cols + broad_conf_cols + ["gt_label"]
    gt_cols = base_cols + [f"conf_{cls}" for cls in gt_class_names] + ["gt_label", "gt_id"]
    fn_cols = ["Imagename", "FrameID", "TLx", "TLy", "BRx", "BRy", "Spname", "boxsize", "missed"]

    # Open CSV Stream Writers
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f_det = open(out_csv, mode="w", newline="")
    det_writer = csv.DictWriter(f_det, fieldnames=det_cols)
    det_writer.writeheader()

    f_gt, gt_writer = None, None
    if args.gt_out_csv:
        gt_csv = Path(args.gt_out_csv)
        gt_csv.parent.mkdir(parents=True, exist_ok=True)
        f_gt = open(gt_csv, mode="w", newline="")
        gt_writer = csv.DictWriter(f_gt, fieldnames=gt_cols)
        gt_writer.writeheader()

    f_fn, fn_writer = None, None
    if args.out_fn_csv:
        fn_csv = Path(args.out_fn_csv)
        fn_csv.parent.mkdir(parents=True, exist_ok=True)
        f_fn = open(fn_csv, mode="w", newline="")
        fn_writer = csv.DictWriter(f_fn, fieldnames=fn_cols)
        fn_writer.writeheader()

    detectid = 0
    gt_detectid = 0
    tp_count, fp_count = 0, 0

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
            half=True
        )

        for img_path, res in zip(chunk_images, results_iter):
            fname = img_path.name
            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue

            # Read Ground Truths from original multi-class directory
            gt_txt = gt_lab_dir / f"{img_path.stem}.txt"
            gt_xyxy, gt_cls = yolo_txt_to_xyxy_and_cls(gt_txt, w, h)

            # Write mantest.csv rows for all Ground Truths
            if gt_writer:
                for g in range(gt_xyxy.shape[0]):
                    gx1, gy1, gx2, gy2 = gt_xyxy[g].tolist()
                    gboxsize = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
                    gt_detectid += 1

                    g_cls = int(gt_cls[g])
                    g_name = gt_class_names[g_cls] if g_cls < len(gt_class_names) else f"unknown_{g_cls}"

                    gt_row = {col: "" for col in gt_cols}
                    gt_row.update({
                        "Detectid": gt_detectid, "Imagename": fname, "FrameID": 0,
                        "TLx": gx1, "TLy": gy1, "BRx": gx2, "BRy": gy2,
                        "Conf": 1.0, "Len": 0, "Spname": g_name,
                        "boxsize": gboxsize, "index": g + 1, "man": g + 1,
                        "gt_label": g_name, "gt_id": g
                    })
                    gt_writer.writerow(gt_row)

            if res.boxes is None or len(res.boxes) == 0:
                pred_xyxy = np.zeros((0, 4), dtype=np.float32)
                pred_conf = np.zeros((0,), dtype=np.float32)
                pred_cls = np.zeros((0,), dtype=np.int32)
            else:
                pred_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                pred_conf = res.boxes.conf.cpu().numpy().astype(np.float32)
                pred_cls = res.boxes.cls.cpu().numpy().astype(np.int32)

            pred_to_gt, pred_to_iou = hungarian_match(pred_xyxy, gt_xyxy, iou_thr=args.match_iou)
            matched_gt = set(pred_to_gt.values())

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

                detectid += 1
                p_cls = int(pred_cls[i])
                p_name = pred_class_names[p_cls] if p_cls < len(pred_class_names) else f"unknown_{p_cls}"

                # Match original multi-class GT species label and manual GT annotation ID
                gt_label = ""
                man_id = ""
                if truedetect:
                    g_idx = pred_to_gt[i]
                    g_cls = int(gt_cls[g_idx])
                    gt_label = gt_class_names[g_cls] if g_cls < len(gt_class_names) else f"unknown_{g_cls}"
                    man_id = g_idx + 1  # 1-based GT annotation ID

                det_row = {
                    "Detectid": detectid, "Imagename": fname, "FrameID": 0,
                    "TLx": x1, "TLy": y1, "BRx": x2, "BRy": y2,
                    "Conf": conf, "Len": 0, "Spname": p_name,  # Broad category (e.g. cancer_crab)
                    "ConfPairs": conf, "boxsize": boxsize,
                    "truedetect": bool(truedetect), "iu": iu,
                    "index": i + 1, "man": man_id,            # Manual GT annotation ID
                    "bin": int(round(conf * 100)) if conf == conf else "",
                    "gt_label": gt_label                     # Original species (e.g. jonah_crab)
                }

                for cls_name in pred_class_names:
                    det_row[f"conf_{cls_name}"] = conf if p_name == cls_name else 0.0

                det_writer.writerow(det_row)

            # Write fn.csv rows for False Negatives (unmatched GTs)
            if fn_writer:
                for g in range(gt_xyxy.shape[0]):
                    if g in matched_gt:
                        continue
                    gx1, gy1, gx2, gy2 = gt_xyxy[g].tolist()
                    gboxsize = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)

                    fn_cls = int(gt_cls[g])
                    fn_name = gt_class_names[fn_cls] if fn_cls < len(gt_class_names) else f"unknown_{fn_cls}"

                    fn_writer.writerow({
                        "Imagename": fname, "FrameID": 0,
                        "TLx": gx1, "TLy": gy1, "BRx": gx2, "BRy": gy2,
                        "Spname": fn_name, "boxsize": gboxsize, "missed": True
                    })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    f_det.close()
    if f_gt:
        f_gt.close()
    if f_fn:
        f_fn.close()

    print(f"Stage 1 Evaluation Complete:\n - {out_csv}\n - {args.gt_out_csv}\n - {args.out_fn_csv}")


if __name__ == "__main__":
    main()
