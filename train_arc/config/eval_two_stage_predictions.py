# ==============================================================================
# File: config/eval_two_stage_predictions.py
# Purpose: Crop predicted boxes from autotest.csv on-the-fly and run Stage 2.
#          Fully dynamic to any taxonomy configuration.
#
#          Genus-conditioned decoding: the final species decision is picked
#          from the genus head's top genus, restricted to the species-head
#          candidates that actually belong to that genus, so the two heads
#          can never disagree in the reported output. Genus confidence is
#          computed from genus_logits (previously discarded entirely) and
#          written to the output CSV explicitly, along with which
#          predictions are genus-only ("is_resolved_species": false in the
#          taxonomy) so genus-level-only classes (e.g. Henricia,
#          Sclerasterias) are surfaced instead of silently forced into one
#          of the fully-resolved species.
# ==============================================================================

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms

from train_classifier import HierarchicalTaxonomicClassifier
from taxonomy_utils import parse_taxonomy_config


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

    # Map species_id to fine-grained class name (now includes genus-only
    # placeholder classes, since they carry a real species_id in the
    # taxonomy instead of the old -100 exclusion sentinel).
    species_id_to_classname = {}
    species_id_is_resolved = {}
    genus_id_to_species_ids = {gid: [] for gid in genus_id_map}

    for cls_info in classes.values():
        sp_id = cls_info.get("species_id", -100)
        if sp_id == -100:
            # Reserved for boxes with NO taxonomic info at all (not even
            # genus). None of the currently defined classes should hit this;
            # if one does, it has no species-head slot and is excluded here
            # exactly as before.
            continue
        species_id_to_classname[sp_id] = cls_info.get("name")
        species_id_is_resolved[sp_id] = cls_info.get("is_resolved_species", True)
        gid = cls_info.get("genus_id")
        if gid is not None:
            genus_id_to_species_ids.setdefault(gid, []).append(sp_id)

    num_species = len(species_id_to_classname)
    species_names = [species_id_to_classname[i] for i in sorted(species_id_to_classname.keys())]
    genus_names = [genus_id_map[i] for i in sorted(genus_id_map.keys())]

    genera_with_no_species_slot = [genus_id_map[g] for g, sids in genus_id_to_species_ids.items() if not sids]
    if genera_with_no_species_slot:
        print(f"WARNING: these genera have zero species-head slots and can never be the final "
              f"prediction: {genera_with_no_species_slot}. Give each a species_id (with "
              f"is_resolved_species=false if genus-only) in {taxonomy_json}.")

    # Map class name to genus for genus accuracy calculation / reporting
    cls_name_to_genus = {info["name"]: info["genus"] for info in classes.values()}

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
    pred_genus_names = []
    genus_confs = []
    is_resolved_flags = []
    species_probs_matrix = np.zeros((len(df), num_species))
    genus_probs_matrix = np.zeros((len(df), num_genera))

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

        batch_species = ["unknown"] * len(batch_df)
        batch_species_conf = [0.0] * len(batch_df)
        batch_genus = ["unknown"] * len(batch_df)
        batch_genus_conf = [0.0] * len(batch_df)
        batch_resolved = [None] * len(batch_df)

        if batch_tensors:
            input_tensor = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                genus_logits, species_logits = model(input_tensor)
                genus_probs = torch.softmax(genus_logits, dim=1).cpu().numpy()
                species_probs = torch.softmax(species_logits, dim=1).cpu().numpy()

            for i, valid_offset in enumerate(valid_indices):
                global_idx = start_idx + valid_offset
                genus_probs_matrix[global_idx] = genus_probs[i]
                species_probs_matrix[global_idx] = species_probs[i]

                # --- Genus-conditioned decoding ---
                # Pick the genus head's top genus, then choose the best
                # species-head candidate restricted to that genus's slots.
                # This guarantees the reported species is always consistent
                # with the reported genus, and naturally surfaces
                # genus-only placeholder classes when that genus has no
                # resolved-species candidates competing for probability mass.
                pred_gid = int(genus_probs[i].argmax())
                g_conf = float(genus_probs[i][pred_gid])
                candidate_sp_ids = genus_id_to_species_ids.get(pred_gid, [])

                if candidate_sp_ids:
                    candidate_probs = species_probs[i][candidate_sp_ids]
                    best_local = int(candidate_probs.argmax())
                    final_sp_id = candidate_sp_ids[best_local]
                    final_sp_conf = float(candidate_probs[best_local])
                    sp_name = species_id_to_classname.get(final_sp_id, "indeterminate_sp")
                    resolved = species_id_is_resolved.get(final_sp_id, True)
                else:
                    # Genus has no species-head slot at all (taxonomy not yet
                    # updated for it) — fall back to reporting genus only.
                    sp_name = f"{genus_id_map.get(pred_gid, 'unknown')}_indeterminate"
                    final_sp_conf = g_conf
                    resolved = False

                batch_species[valid_offset] = sp_name
                batch_species_conf[valid_offset] = final_sp_conf
                batch_genus[valid_offset] = genus_id_map.get(pred_gid, "unknown")
                batch_genus_conf[valid_offset] = g_conf
                batch_resolved[valid_offset] = resolved

        stage2_species.extend(batch_species)
        stage2_confs.extend(batch_species_conf)
        pred_genus_names.extend(batch_genus)
        genus_confs.extend(batch_genus_conf)
        is_resolved_flags.extend(batch_resolved)

    # 6. Append Stage 2 Results
    df["pred_genus"] = pred_genus_names
    df["genus_conf"] = genus_confs
    df["stage2_species"] = stage2_species
    df["stage2_conf"] = stage2_confs
    df["species_is_resolved"] = is_resolved_flags  # False => genus-only call (e.g. "henricia_sp")

    for sp_id, sp_name in enumerate(species_names):
        df[f"conf_{sp_name}"] = species_probs_matrix[:, sp_id]
    for g_id, g_name in enumerate(genus_names):
        df[f"conf_genus_{g_name}"] = genus_probs_matrix[:, g_id]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # 7. Evaluate Performance on True Positives
    tp_df = df[df["truedetect"] == True].copy()
    tp_df["gt_genus"] = tp_df["gt_label"].map(cls_name_to_genus)

    # Genus Accuracy — now compared directly against the genus HEAD's own
    # prediction, not re-derived from the final species name. This is the
    # metric that was previously impossible to get right for genus-only
    # ground truth, since the species head had nowhere to put them.
    eval_genus_tp = tp_df[tp_df["gt_genus"].notna()]
    correct_g = (eval_genus_tp["pred_genus"] == eval_genus_tp["gt_genus"]).sum()
    total_g = len(eval_genus_tp)
    acc_genus = (correct_g / total_g * 100) if total_g > 0 else 0.0

    # Fine Species Accuracy (includes genus-only classes as exact-match targets now)
    eval_sp_tp = tp_df[tp_df["gt_label"].isin(species_names)]
    correct_sp = (eval_sp_tp["stage2_species"] == eval_sp_tp["gt_label"]).sum()
    total_sp = len(eval_sp_tp)
    acc_species = (correct_sp / total_sp * 100) if total_sp > 0 else 0.0

    # Accuracy specifically on genus-only ground truth (e.g. Henricia, Sclerasterias)
    genus_only_names = [n for sid, n in species_id_to_classname.items() if not species_id_is_resolved.get(sid, True)]
    eval_gonly_tp = tp_df[tp_df["gt_label"].isin(genus_only_names)]
    correct_gonly = (eval_gonly_tp["stage2_species"] == eval_gonly_tp["gt_label"]).sum()
    total_gonly = len(eval_gonly_tp)
    acc_gonly = (correct_gonly / total_gonly * 100) if total_gonly > 0 else float("nan")

    print("\n================ SYSTEM EVALUATION SUMMARY ================")
    print(f"Total Broad Detections Evaluated : {len(df)}")
    print(f"Stage 1 True Positives (TP)     : {len(tp_df)}")
    print(f"Stage 1 False Positives (FP)    : {(df['truedetect'] == False).sum()}")
    print(f"Stage 2 Genus Accuracy on TPs   : {acc_genus:.2f}% ({correct_g}/{total_g})")
    print(f"Stage 2 Species Accuracy on TPs : {acc_species:.2f}% ({correct_sp}/{total_sp})")
    if total_gonly > 0:
        print(f"Stage 2 Genus-Only Accuracy     : {acc_gonly:.2f}% ({correct_gonly}/{total_gonly})  [{genus_only_names}]")
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
