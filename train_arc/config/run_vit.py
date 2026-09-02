# config/run_vit.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from ultralytics import RTDETR  # Native ViT Object Detector

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train RT-DETR ViT on scallop datasets.")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument(
        "--model",
        default="rtdetr-l.pt",  # Options: rtdetr-l.pt, rtdetr-x.pt
        help="RT-DETR ViT model weights."
    )
    ap.add_argument("--epochs", type=int, default=130)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)  # ViTs use more VRAM; reduced default batch
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label", default="scallop")
    ap.add_argument("--data_root", default="data{year}/yolo")
    ap.add_argument("--yaml_name", default="data.yaml")
    ap.add_argument("--project_dir", default="./output")
    ap.add_argument("--run_tag", default="")
    ap.add_argument("--exist_ok", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    data_root = args.data_root.format(year=args.year)
    data_yaml = str(Path(data_root) / args.yaml_name)

    job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    project_dir = Path(args.project_dir)

    model_tag = Path(args.model).stem.replace(".", "_").replace("-", "_")
    tag = f"_{args.run_tag}" if args.run_tag else ""
    run_name = f"{args.year}{args.label}_{model_tag}_{tag}{job_id}"

    print("=== Training RT-DETR ViT configuration ===")
    print("data_yaml:", data_yaml)
    print("model:", args.model)
    print("run_name:", run_name)
    print("project_dir:", project_dir)

    # Initialize RT-DETR Transformer architecture
    model = RTDETR(args.model)

    # config/run_vit.py
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
        amp=False,  # <-- ADD THIS: Prevents FP16 loss explosion/NaNs in RT-DETR
    )

    print("Transformer Training complete.")


if __name__ == "__main__":
    main()