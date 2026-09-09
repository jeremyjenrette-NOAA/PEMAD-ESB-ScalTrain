# ==============================================================================
# File: config/taxonomy_utils.py
# Purpose: Single source of truth for loading taxonomy JSON configs, and for
#          validating that a taxonomy JSON's class-id -> name mapping agrees
#          with the class-id -> name mapping baked into a YOLO data.yaml.
#
#          This check is what prevents "label cross-contamination": taxonomy
#          JSON and YOLO label files are two independently-maintained sources
#          of the same class-id -> species mapping. If they ever disagree,
#          every script that reads YOLO label ints and looks them up in the
#          taxonomy JSON (crop extraction, Stage 1 eval) will silently
#          mislabel data with no error. This module makes that check explicit
#          and mandatory instead of implicit and skipped.
# ==============================================================================

import json
from pathlib import Path
from typing import Dict

import yaml


class TaxonomyMismatchError(ValueError):
    """Raised when a taxonomy JSON and a YOLO data.yaml disagree about class ids."""


def load_taxonomy(taxonomy_json: str) -> dict:
    with open(taxonomy_json, "r") as f:
        taxonomy = json.load(f)
    if "classes" not in taxonomy:
        raise ValueError(f"Taxonomy config {taxonomy_json} is missing a 'classes' key.")
    return taxonomy


def load_yolo_class_names(data_yaml: str) -> Dict[int, str]:
    """Returns {int class_id: name} as defined by a YOLO dataset's data.yaml."""
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def validate_taxonomy_against_yolo(taxonomy: dict, data_yaml: str, taxonomy_json_path: str = "<taxonomy>") -> None:
    """
    Cross-checks that every class id in the taxonomy JSON maps to the same
    species name as the same class id in the given YOLO data.yaml, and that
    neither side has ids the other doesn't know about.

    Call this BEFORE extracting crops or computing gt_label in eval — i.e.
    against the ORIGINAL multi-class yolo/data.yaml, not the single-class
    yolo_broad/data.yaml (which has no per-species info to check against).

    Raises TaxonomyMismatchError with every mismatch listed if anything is
    inconsistent. Fails loud instead of contaminating labels silently.
    """
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Cannot validate taxonomy: {data_yaml} does not exist.")

    yolo_names = load_yolo_class_names(data_yaml)
    classes = taxonomy.get("classes", {})
    taxonomy_ids = {int(k) for k in classes}

    mismatches = []
    for str_id, info in classes.items():
        cid = int(str_id)
        expected_name = info.get("name")
        actual_name = yolo_names.get(cid)
        if actual_name is None:
            mismatches.append(
                f"  class id {cid} ('{expected_name}') exists in {taxonomy_json_path} "
                f"but is absent from {data_yaml}"
            )
        elif actual_name != expected_name:
            mismatches.append(
                f"  class id {cid}: {taxonomy_json_path} calls this '{expected_name}', "
                f"{data_yaml} calls it '{actual_name}'"
            )

    extra_yolo_ids = set(yolo_names) - taxonomy_ids
    for cid in sorted(extra_yolo_ids):
        mismatches.append(
            f"  class id {cid} ('{yolo_names[cid]}') exists in {data_yaml} "
            f"but is absent from {taxonomy_json_path}"
        )

    if mismatches:
        raise TaxonomyMismatchError(
            "Taxonomy JSON and YOLO data.yaml disagree on class ids — this is exactly "
            "the kind of drift that silently mislabels crops and eval ground truth:\n"
            + "\n".join(mismatches)
            + "\n\nFix the taxonomy JSON (or data.yaml) so ids line up 1:1, then re-run."
        )


def parse_taxonomy_config(taxonomy_config: dict):
    """
    Dynamically parses species and genus id->name mappings from a taxonomy
    config. Moved here (from train_classifier.py) so every script that needs
    this logic imports one copy instead of re-deriving it.
    """
    classes = taxonomy_config.get("classes", {})

    if "genus_id_to_name" in taxonomy_config:
        genus_id_map = {int(k): v for k, v in taxonomy_config["genus_id_to_name"].items()}
    else:
        genus_id_map = {}
        for cls_info in classes.values():
            if "genus_id" in cls_info:
                genus_id_map[cls_info["genus_id"]] = cls_info.get("genus", f"genus_{cls_info['genus_id']}")

    if "species_id_to_name" in taxonomy_config:
        species_id_map = {int(k): v for k, v in taxonomy_config["species_id_to_name"].items()}
    else:
        species_id_map = {}
        for cls_info in classes.values():
            sp_id = cls_info.get("species_id", -100)
            if sp_id != -100:
                species_id_map[sp_id] = cls_info.get("species", cls_info.get("name", f"species_{sp_id}"))

    return genus_id_map, species_id_map
