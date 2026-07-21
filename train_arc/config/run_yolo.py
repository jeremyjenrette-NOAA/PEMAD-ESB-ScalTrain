from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train YOLO on scallop datasets by year.")
    ap.add_argument("--year", type=int, required=True, help="Dataset year, e.g. 2022")
    ap.add_argument(
        "--model",
        default="check/yolov8n.pt",
        help=(
            "YOLO model/weights. Examples: "
            "'check/yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', "
            "'yolov9c.pt' (if supported by your ultralytics version), "
            "or a path to a custom .pt"
        ),
    )

    # training knobs
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="CUDA device id(s), e.g. '0' or '0,1' or 'cpu'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label", default="scallop")

    # paths
    ap.add_argument(
        "--data_root",
        default="data{year}/yolo",
        help="Template for dataset root relative to train_arc. Use {year} placeholder.",
    )
    # Change this line inside parse_args():
    ap.add_argument(
        "--yaml_name",
        default="data.yaml",  # Updated from label.yaml to match dataset builder output
        help="YAML filename under the year-specific yolo folder.",
    )
    ap.add_argument(
        "--project_dir",
        default="/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/output",
        help="Ultralytics project output directory.",
    )
    ap.add_argument("--run_tag", default="", help="Optional extra tag appended to the run name.")
    ap.add_argument("--exist_ok", action="store_true", help="Allow overwriting an existing run folder.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # build data path from year template
    data_root = args.data_root.format(year=args.year)
    data_yaml = str(Path(data_root) / args.yaml_name)

    # slurm-aware naming
    job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    project_dir = Path(args.project_dir)

    # derive a short model tag for filenames
    model_tag = Path(args.model).stem.replace(".", "_").replace("-", "_")
    tag = f"_{args.run_tag}" if args.run_tag else ""
    run_name = f"{args.year}{args.label}_{model_tag}_{tag}{job_id}"

    print("=== Training configuration ===")
    print("data_yaml:", data_yaml)
    print("model:", args.model)
    print("run_name:", run_name)
    print("project_dir:", project_dir)

    model = YOLO(args.model)

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_dir),
        name=run_name,
        exist_ok=args.exist_ok,
    )

    print("Training complete.")
    print("Results object:", results)


if __name__ == "__main__":
    main()