# ==============================================================================
# File: config/eval_two_stage_predictions.py
# Purpose: Crop predicted boxes from autotest.csv on-the-fly and run Stage 2.
#          Fully dynamic to any taxonomy configuration.
# ==============================================================================

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms

from train_classifier import HierarchicalTaxonomicClassifier, parse_taxonomy_config


def classify_detector_boxes(
    autotest_csv: str,
    val_img_dir: str,
    stage2_weights: str,
    taxonomy_json: str,
    out_csv: str,
    batch_size: int = 64,
    device: str = "cuda:0"
):
    df = pd.read_csv(autotest_csv)
    val_img_dir = Path(val_img_dir)

    # 1. Path Integrity Verification
    if not val_img_dir.exists():
        raise FileNotFoundError(f"Validation image directory not found at: {val_img_dir.resolve()}")
    
    if len(df) > 0:
        sample_img = val_img_dir / df.iloc[0]["Imagename"]
        if not sample_img.exists():
            raise FileNotFoundError(f"Sample image missing at: {sample_img}. Check --val_img_dir path.")

    # 2. Extract Species & Genus Maps from Taxonomy Configuration
    with open(taxonomy_json, "r") as f:
        taxonomy_config = json.load(f)

    classes = taxonomy_config.get("classes", {})
    genus_id_map, _ = parse_taxonomy_config(taxonomy_config)
    num_genera = len(genus_id_map)

    # Map species_id to fine-grained class name
    species_id_to_classname = {}
    for cls_info in classes.values():
        sp_id = cls_info.get("species_id", -100)
        if sp_id != -100:
            species_id_to_classname[sp_id] = cls_info.get("name")

    num_species = len(species_id_to_classname)
    species_names = [species_id_to_classname[i] for i in sorted(species_id_to_classname.keys())]

    # Map class name to genus for genus accuracy calculation
    cls_name_to_genus = {info["name"]: info["genus"] for info in classes.values()}

    # Determine fallback name for masked/indeterminate instances
    indeterminate_fallback = "indeterminate_sp"
    for cls_info in classes.values():
        if cls_info.get("species_id") == -100:
            indeterminate_fallback = cls_info.get("name", "indeterminate_sp")
            break

    # 3. Strip pre-existing broad conf_* columns (keep broad detection score Conf)
    cols_to_drop = [c for c in df.columns if c.startswith("conf_") and c != "Conf"]
    df.drop(columns=cols_to_drop, inplace=True)

    # 4. Load Model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = HierarchicalTaxonomicClassifier(
        backbone_name="convnext_tiny",
        num_genera=num_genera,
        num_species=num_species
    )
    model.load_state_dict(torch.load(stage2_weights, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    stage2_species = []
    stage2_confs = []
    species_probs_matrix = np.zeros((len(df), num_species))

    print(f"Running Stage 2 inference on {len(df)} predicted bounding boxes...")

    # 5. Batch Inference Loop
    for start_idx in range(0, len(df), batch_size):
        batch_df = df.iloc[start_idx : start_idx + batch_size]
        batch_tensors = []
        valid_indices = []

        for offset, (_, row) in enumerate(batch_df.iterrows()):
            img_path = val_img_dir / row["Imagename"]
            if not img_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                w, h = image.size
                tlx, tly, brx, bry = row["TLx"], row["TLy"], row["BRx"], row["BRy"]
                crop = image.crop((max(0, int(tlx)), max(0, int(tly)), min(w, int(brx)), min(h, int(bry))))
                
                if crop.width > 0 and crop.height > 0:
                    batch_tensors.append(transform(crop))
                    valid_indices.append(offset)
            except Exception:
                continue

        batch_preds = ["unknown"] * len(batch_df)
        batch_top_confs = [0.0] * len(batch_df)

        if batch_tensors:
            input_tensor = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                _, species_logits = model(input_tensor)
                probs = torch.softmax(species_logits, dim=1).cpu().numpy()
                top_confs = probs.max(axis=1)
                pred_ids = probs.argmax(axis=1)

            for i, valid_offset in enumerate(valid_indices):
                pid = pred_ids[i]
                batch_preds[valid_offset] = species_id_to_classname.get(pid, indeterminate_fallback)
                batch_top_confs[valid_offset] = top_confs[i]
                global_idx = start_idx + valid_offset
                species_probs_matrix[global_idx] = probs[i]

        stage2_species.extend(batch_preds)
        stage2_confs.extend(batch_top_confs)

    # 6. Append Stage 2 Results
    df["stage2_species"] = stage2_species
    df["stage2_conf"] = stage2_confs

    for sp_id, sp_name in enumerate(species_names):
        df[f"conf_{sp_name}"] = species_probs_matrix[:, sp_id]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # 7. Evaluate Performance on True Positives
    tp_df = df[df["truedetect"] == True].copy()
    tp_df["gt_genus"] = tp_df["gt_label"].map(cls_name_to_genus)
    tp_df["pred_genus"] = tp_df["stage2_species"].map(cls_name_to_genus)

    # Genus Accuracy
    eval_genus_tp = tp_df[tp_df["gt_genus"].notna()]
    correct_g = (eval_genus_tp["pred_genus"] == eval_genus_tp["gt_genus"]).sum()
    total_g = len(eval_genus_tp)
    acc_genus = (correct_g / total_g * 100) if total_g > 0 else 0.0

    # Fine Species Accuracy
    eval_sp_tp = tp_df[tp_df["gt_label"].isin(species_names)]
    correct_sp = (eval_sp_tp["stage2_species"] == eval_sp_tp["gt_label"]).sum()
    total_sp = len(eval_sp_tp)
    acc_species = (correct_sp / total_sp * 100) if total_sp > 0 else 0.0

    print("\n================ SYSTEM EVALUATION SUMMARY ================")
    print(f"Total Broad Detections Evaluated : {len(df)}")
    print(f"Stage 1 True Positives (TP)     : {len(tp_df)}")
    print(f"Stage 1 False Positives (FP)    : {(df['truedetect'] == False).sum()}")
    print(f"Stage 2 Genus Accuracy on TPs   : {acc_genus:.2f}% ({correct_g}/{total_g})")
    print(f"Stage 2 Species Accuracy on TPs : {acc_species:.2f}% ({correct_sp}/{total_sp})")
    print("===========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Two-Stage Cascade Pipeline")
    parser.add_argument("--autotest_csv", required=True)
    parser.add_argument("--val_img_dir", required=True)
    parser.add_argument("--stage2_weights", required=True)
    parser.add_argument("--taxonomy_json", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    classify_detector_boxes(
        autotest_csv=args.autotest_csv,
        val_img_dir=args.val_img_dir,
        stage2_weights=args.stage2_weights,
        taxonomy_json=args.taxonomy_json,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()