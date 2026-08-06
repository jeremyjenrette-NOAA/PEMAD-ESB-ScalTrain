#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment

import numpy as np
import pandas as pd


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
    cost = (1.0 - ious).astype(np.float32)
    cost[ious < iou_thr] = big_cost

    row_ind, col_ind = linear_sum_assignment(cost)

    pred_to_gt: Dict[int, int] = {}
    pred_to_iou: Dict[int, float] = {}

    for p, g in zip(row_ind, col_ind):
        p = int(p)
        g = int(g)
        iou = float(ious[p, g])
        if iou >= iou_thr and cost[p, g] < big_cost:
            pred_to_gt[p] = g
            pred_to_iou[p] = iou

    return pred_to_gt, pred_to_iou


def coco_bbox_to_xyxy(bbox: List[float]) -> List[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def load_coco_gt(gt_json: Path):
    d = json.loads(gt_json.read_text())

    images_by_id = {im["id"]: im for im in d.get("images", [])}
    anns_by_image: Dict[int, List[dict]] = {}
    for ann in d.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    return images_by_id, anns_by_image, d


def load_pred_files(pred_dir: Path) -> Dict[str, dict]:
    pred_map = {}
    for fp in sorted(pred_dir.glob("dets_gid_*_v2.mscoco.json")):
        d = json.loads(fp.read_text())
        imgs = d.get("images", [])
        if len(imgs) != 1:
            raise ValueError(f"Expected exactly 1 image in {fp}, found {len(imgs)}")
        fname = Path(imgs[0]["file_name"]).name
        pred_map[fname] = d
    return pred_map


def count_pos_neg_from_gt(images_by_id: Dict[int, dict], anns_by_image: Dict[int, List[dict]]) -> tuple[int, int, int]:
    n_images = len(images_by_id)
    n_pos = 0
    n_neg = 0
    for image_id in images_by_id:
        n_gt = len(anns_by_image.get(image_id, []))
        if n_gt > 0:
            n_pos += 1
        else:
            n_neg += 1
    return n_images, n_pos, n_neg


def clean_string_label(raw_label: str) -> str:
    """Removes erroneous brackets, commas, quotes, and whitespace from class names."""
    return str(raw_label).replace("[", "").replace("]", "").replace(",", "").replace("'", "").replace('"', "").strip()


def main():
    ap = argparse.ArgumentParser(description="Generalized VIAME/COCO Hungarian Matching Evaluation")
    ap.add_argument("--pred_dir", required=True, help="Directory with dets_gid_*_v2.mscoco.json files")
    ap.add_argument("--gt_json", required=True, help="validation_truth.json")
    ap.add_argument("--train_json", required=False, default=None, help="training_truth.json for run summary")
    ap.add_argument("--out_csv", required=True, help="Output detections CSV")
    ap.add_argument("--gt_out_csv", required=False, default=None, help="Optional GT CSV")
    ap.add_argument("--out_fn_csv", required=False, default=None, help="Optional FN CSV")
    ap.add_argument("--match_iou", type=float, default=0.01)
    ap.add_argument("--conf", type=float, default=0.01, help="Metadata only")
    ap.add_argument("--nms_iou", type=float, default=0.75, help="Metadata only")
    ap.add_argument("--max_det", type=int, default=600, help="Metadata only")
    ap.add_argument(
        "--spname", 
        nargs="+", 
        default=None,
        help="Fallback class names mapped in explicit COCO numerical index order if not in GT JSON"
    )
    ap.add_argument("--debug_n", type=int, default=10)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--model_name", default="viame")
    ap.add_argument("--run_dir", default=None)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_json = Path(args.gt_json)
    train_json = Path(args.train_json) if args.train_json else None

    images_by_id, anns_by_image, gt_full = load_coco_gt(gt_json)
    pred_map = load_pred_files(pred_dir)

    # Build dynamic category mapping
    cat_map = {}
    for cat in gt_full.get("categories", []):
        cat_map[cat["id"]] = clean_string_label(cat["name"])
        
    if not cat_map and args.spname:
        cat_map = {i + 1: clean_string_label(name) for i, name in enumerate(args.spname)}

    # Extract unique class list for dynamic confidence columns
    active_classes = [cat_map[k] for k in sorted(cat_map.keys())] if cat_map else []

    det_rows: List[dict] = []
    gt_rows: List[dict] = []
    fn_rows: List[dict] = []

    detectid = 0
    gt_detectid = 0
    debug_printed = 0

    for image_id, im in sorted(images_by_id.items()):
        file_name = im["file_name"]
        fname = Path(file_name).name

        gt_anns = anns_by_image.get(image_id, [])
        gt_xyxy = np.array([coco_bbox_to_xyxy(a["bbox"]) for a in gt_anns], dtype=np.float32) \
            if gt_anns else np.zeros((0, 4), dtype=np.float32)

        if args.gt_out_csv:
            for g, ann in enumerate(gt_anns):
                gx1, gy1, gx2, gy2 = coco_bbox_to_xyxy(ann["bbox"])
                gboxsize = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
                gt_detectid += 1
                
                g_cat_id = ann.get("category_id")
                g_name = cat_map.get(g_cat_id, f"unknown_{g_cat_id}")

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
                    "Spname": g_name,
                    "boxsize": gboxsize,
                    "index": g + 1,
                    "gt_id": g,
                })

        pred_doc = pred_map.get(fname, None)
        pred_anns = []
        if pred_doc is not None:
            pred_anns = pred_doc.get("annotations", [])
            
        pred_xyxy = np.array([coco_bbox_to_xyxy(a["bbox"]) for a in pred_anns], dtype=np.float32) \
            if pred_anns else np.zeros((0, 4), dtype=np.float32)
        pred_conf = np.array([float(a.get("score", np.nan)) for a in pred_anns], dtype=np.float32) \
            if pred_anns else np.zeros((0,), dtype=np.float32)

        pred_to_gt, pred_to_iou = hungarian_match(pred_xyxy, gt_xyxy, iou_thr=args.match_iou)
        matched_gt = set(pred_to_gt.values())

        if debug_printed < args.debug_n and gt_xyxy.shape[0] > 0:
            max_iou_any = float(iou_xyxy(pred_xyxy, gt_xyxy).max()) if (pred_xyxy.shape[0] and gt_xyxy.shape[0]) else 0.0
            print(f"[DEBUG] {fname}  n_gt={gt_xyxy.shape[0]}  n_pred={pred_xyxy.shape[0]}  max_iou_any={max_iou_any:.3f}")
            debug_printed += 1

        for i in range(pred_xyxy.shape[0]):
            x1, y1, x2, y2 = pred_xyxy[i].tolist()
            conf = float(pred_conf[i]) if pred_conf.size else np.nan
            boxsize = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            truedetect = i in pred_to_gt
            iu = float(pred_to_iou.get(i, 0.0)) if truedetect else 0.0

            p_cat_id = pred_anns[i].get("category_id")
            p_name = cat_map.get(p_cat_id, f"unknown_{p_cat_id}")

            # Extract Ground Truth species label dynamically if True Positive
            gt_label = ""
            if truedetect:
                g_idx = pred_to_gt[i]
                g_ann = gt_anns[g_idx]
                g_cat_id = g_ann.get("category_id")
                gt_label = cat_map.get(g_cat_id, f"unknown_{g_cat_id}")

            detectid += 1
            row_dict = {
                "Detectid": detectid,
                "Imagename": fname,
                "FrameID": 0,
                "TLx": x1,
                "TLy": y1,
                "BRx": x2,
                "BRy": y2,
                "Conf": conf,
                "Len": 0,
                "Spname": p_name,
                "ConfPairs": conf,
                "boxsize": boxsize,
                "truedetect": bool(truedetect),
                "iu": iu,
                "index": i + 1,
                "man": "",
                "bin": int(round(conf * 100)) if conf == conf else "",
                "gt_label": gt_label
            }

            # Map dynamic one-hot confidence columns per species
            for cls_name in active_classes:
                row_dict[f"conf_{cls_name}"] = conf if p_name == cls_name else 0.0

            det_rows.append(row_dict)

        if args.out_fn_csv:
            for g in range(gt_xyxy.shape[0]):
                if g in matched_gt:
                    continue
                gx1, gy1, gx2, gy2 = gt_xyxy[g].tolist()
                gboxsize = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
                
                fn_cat_id = gt_anns[g].get("category_id")
                fn_name = cat_map.get(fn_cat_id, f"unknown_{fn_cat_id}")

                fn_rows.append({
                    "Imagename": fname,
                    "FrameID": 0,
                    "TLx": gx1, "TLy": gy1, "BRx": gx2, "BRy": gy2,
                    "Spname": fn_name,
                    "boxsize": gboxsize,
                    "missed": True,
                })

    det_df = pd.DataFrame(det_rows)
    
    # Construct dynamic column ordering
    base_cols = [
        "Detectid","Imagename","FrameID","TLx","TLy","BRx","BRy","Conf","Len","Spname",
        "ConfPairs","boxsize","truedetect","iu","index","man","bin"
    ]
    dynamic_conf_cols = [f"conf_{cls}" for cls in active_classes]
    col_order = base_cols + dynamic_conf_cols + ["gt_label"]

    for c in col_order:
        if c not in det_df.columns:
            det_df[c] = 0.0 if c.startswith("conf_") else ""
    det_df = det_df[col_order]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    det_df.to_csv(out_csv, index=False)
    print(f"Wrote detections CSV: {out_csv} (rows={len(det_df)})")

    if args.gt_out_csv:
        gt_df = pd.DataFrame(gt_rows)
        for c in base_cols:
            if c not in gt_df.columns:
                gt_df[c] = ""
        extra_cols = [c for c in gt_df.columns if c not in base_cols]
        gt_df = gt_df[base_cols + extra_cols]

        out_gt = Path(args.gt_out_csv)
        out_gt.parent.mkdir(parents=True, exist_ok=True)
        gt_df.to_csv(out_gt, index=False)
        print(f"Wrote GT CSV: {out_gt} (rows={len(gt_df)})")

    if args.out_fn_csv:
        fn_df = pd.DataFrame(fn_rows)
        out_fn = Path(args.out_fn_csv)
        out_fn.parent.mkdir(parents=True, exist_ok=True)
        fn_df.to_csv(out_fn, index=False)
        print(f"Wrote FN CSV: {out_fn} (rows={len(fn_df)})")

    n_val_images, n_val_pos, n_val_neg = count_pos_neg_from_gt(images_by_id, anns_by_image)

    if train_json and train_json.exists():
        tr_images_by_id, tr_anns_by_image, _ = load_coco_gt(train_json)
        n_train_images, n_train_pos, n_train_neg = count_pos_neg_from_gt(tr_images_by_id, tr_anns_by_image)
    else:
        n_train_images = n_train_pos = n_train_neg = None

    summary = {
        "year": args.year,
        "model": args.model_name,
        "run_dir": str(args.run_dir) if args.run_dir else "",
        "pred_dir": str(pred_dir),
        "gt_json": str(gt_json),
        "train_json": str(train_json) if train_json else "",
        "n_train_images": n_train_images,
        "n_train_pos": n_train_pos,
        "n_train_neg": n_train_neg,
        "n_val_images": n_val_images,
        "n_val_pos": n_val_pos,
        "n_val_neg": n_val_neg,
        "neg_ratio_observed_train": None if n_train_pos in (None, 0) else (n_train_neg / max(n_train_pos, 1)),
        "neg_ratio_observed_val": n_val_neg / max(n_val_pos, 1),
        "eval_conf": args.conf,
        "nms_iou": args.nms_iou,
        "match_iou": args.match_iou,
        "max_det": args.max_det,
    }

    summary_json = out_csv.parent / "run_summary.json"
    summary_csv = out_csv.parent / "run_summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print(f"Wrote run summary: {summary_json}")
    print(f"Wrote run summary: {summary_csv}")

    if len(det_df):
        print("TP count:", int(det_df["truedetect"].sum()))
        print("FP count:", int((~det_df["truedetect"]).sum()))


if __name__ == "__main__":
    main()