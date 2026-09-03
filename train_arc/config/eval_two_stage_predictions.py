# ==============================================================================
# File: config/eval_two_stage_predictions.py
# Purpose: Crop predicted boxes from autotest.csv on-the-fly and run Stage 2
#          Fully dynamic to any taxonomy configuration.
# ==============================================================================

import argparse
import json
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms

from config.train_classifier import HierarchicalTaxonomicClassifier, parse_taxonomy_config


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
            raise FileNotFoundError(
                f"Validation image directory mismatch! Sample frame from CSV not found at: {sample_img}\n"
                f"Verify --val_img_dir path matches your dataset directory."
            )

    # 2. Dynamic Taxonomy Extraction
    with open(taxonomy_json, "r") as f:
        taxonomy_config = json.load(f)

    genus_id_map, species_id_map = parse_taxonomy_config(taxonomy_config)
    num_genera = len(genus_id_map)
    num_species = len(species_id_map)

    # Find indeterminate fallback label name (e.g., class with species_id == -100)
    sp_fallback = "indeterminate_sp"
    for cls_info in taxonomy_config.get("classes", {}).values():
        if cls_info.get("species_id") == -100:
            sp_fallback = cls_info.get("name", cls_info.get("species", "indeterminate_sp"))
            break

    # 3. Load Trained Model with Dynamic Output Heads
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

    print(f"Running Stage 2 inference on {len(df)} predicted bounding boxes...")

    # 4. Batch Inference Loop
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

                # Bounding box extraction
                tlx, tly, brx, bry = row["TLx"], row["TLy"], row["BRx"], row["BRy"]
                xmin = max(0, int(tlx))
                ymin = max(0, int(tly))
                xmax = min(w, int(brx))
                ymax = min(h, int(bry))

                crop = image.crop((xmin, ymin, xmax, ymax))
                if crop.width == 0 or crop.height == 0:
                    continue

                batch_tensors.append(transform(crop))
                valid_indices.append(offset)
            except Exception:
                continue

        batch_preds = ["unknown"] * len(batch_df)
        batch_confs = [0.0] * len(batch_df)

        if batch_tensors:
            input_tensor = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                _, species_logits = model(input_tensor)
                probs = torch.softmax(species_logits, dim=1)
                conf_values, pred_ids = probs.max(dim=1)

            for i, valid_offset in enumerate(valid_indices):
                pid = pred_ids[i].item()
                batch_preds[valid_offset] = species_id_map.get(pid, sp_fallback)
                batch_confs[valid_offset] = conf_values[i].item()

        stage2_species.extend(batch_preds)
        stage2_confs.extend(batch_confs)

    # 5. Append Results & Export
    df["stage2_species"] = stage2_species
    df["stage2_conf"] = stage2_confs

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Inference complete! Results saved to: {out_csv}")

    # 6. Combined System Metrics
    tp_mask = df["truedetect"] == True
    tp_df = df[tp_mask]

    # Evaluate accuracy against fine-grained species ground truths
    valid_species_names = list(species_id_map.values())
    evaluable_tp = tp_df[tp_df["gt_label"].isin(valid_species_names)]
    
    correct = (evaluable_tp["stage2_species"] == evaluable_tp["gt_label"]).sum()
    total = len(evaluable_tp)
    accuracy = (correct / total * 100) if total > 0 else 0.0

    print("\n================ SYSTEM EVALUATION SUMMARY ================")
    print(f"Total Broad Detections Evaluated : {len(df)}")
    print(f"Stage 1 True Positives (TP)     : {len(tp_df)}")
    print(f"Stage 1 False Positives (FP)    : {(df['truedetect'] == False).sum()}")
    print(f"Evaluable TPs with GT Species   : {total}")
    print(f"Stage 2 Accuracy on Detector TPs: {accuracy:.2f}% ({correct}/{total})")
    print("===========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Two-Stage Cascade Pipeline")
    parser.add_argument("--autotest_csv", required=True, help="Path to autotest.csv from broad detector run")
    parser.add_argument("--val_img_dir", required=True, help="Path to validation images")
    parser.add_argument("--stage2_weights", required=True, help="Path to trained Stage 2 weights")
    parser.add_argument("--taxonomy_json", required=True, help="Path to taxonomy config JSON")
    parser.add_argument("--out_csv", required=True, help="Path for merged output CSV")
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