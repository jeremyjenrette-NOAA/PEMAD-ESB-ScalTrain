# PEMAD-ESB-ScalTrain

## Maintainer

Name: Jeremy J

Email: jeremy.jenrette@noaa.gov

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

## Description

PEMAD-ESB-ScalTrain builds the training and evaluation sets and trains and scores the three sea scallop (*Placopecten magellanicus*) detectors used in Jenrette et al., *Automatically counting and measuring Atlantic sea scallops (Placopecten magellanicus) from underwater imagery with object detection and bias-correction* (Ecological Informatics, in revision): YOLOv12n and RT-DETR-L (Ultralytics) and Cascade R-CNN (VIAME). It also stages the Zenodo data record.

The calibration, abundance and size analysis is in the companion repository [PEMAD-ESB-ScalPred](https://github.com/jeremyjenrette-NOAA/PEMAD-ESB-ScalPred). **You do not need this repository to reproduce the paper's results:** the trained weights and every detector output it produced are in the data record ([10.5281/zenodo.23482337](https://doi.org/10.5281/zenodo.23482337), published with the paper). Use it to retrain or re-score the detectors, or to train them on new imagery.

The jobs were written for a Slurm GPU cluster (Virginia Tech ARC) and a GCP workstation; paths under `/projects/...` and the `#SBATCH` account and partition lines are site-specific and need editing before use elsewhere.

## Workflow (paper results)

All scripts are run from `train_arc/`.

| Step | Script | What it does |
|---|---|---|
| 1. Split and labels | `build_strat_labels_v2.py` | Converts annotated line segments to square boxes (midpoint centre, side = segment length), collapses all live-scallop classes to one class, adds empty images, and splits along transects: within each 10-image block, 6 images to the Detector-training set and 4 to the Calibration set; the Calibration set is then split 60/40 within 20-image chunks into Calibration-fitting and Holdout test. Writes the YOLO dataset and `dataset_split_2226.csv`. |
| 2. Check | `config/05_validate_dataset.py` | Checks images, labels and VIAME truth files before training. |
| 3a. YOLOv12n | `job/yolo_job.sh` (`config/run_yolo.py`) | 150 epochs, 640 px, batch 16; then scores the Calibration set. |
| 3b. RT-DETR-L | `job/gcp_vit_job.sh` (`config/run_vit.py`) | 130 epochs, 640 px, batch 4 on a Tesla T4; then scores the Calibration set. |
| 3c. Cascade R-CNN | `job/viame_custom_job.sh` | VIAME netharn Cascade R-CNN, up to 40 epochs with early-stopping patience 7, 640 × 640 windows; then scores the Calibration set. |
| 4. Evaluation | `config/eval_yolo_detections_hung_multi.py`, `config/eval_viame_detections_hung_multi.py` (`job/eval.sh`) | Detections at confidence ≥ 0.1, NMS IoU 0.65, at most 600 per image, matched one-to-one to annotations by the Hungarian algorithm at IoU ≥ 0.1. Writes `autotest*` (detections with match), `mantest*` (annotation boxes), `fn*` (missed annotations) and `run_summary.csv`, the files ScalPred reads. |
| 5. Inference timing | `config/bench_inference.py` (`job/benchmark_job.sh`) | Times YOLOv12n and RT-DETR-L on 500 Calibration-set images after a warm-up, on GPU, CPU threads or Apple MPS. |

## Independent 2026 survey test

YOLOv12n was retrained without 2026 and scored on every annotated 2026 image:

1. `config/01_filter_annotations.py` → `02_stage_images.py` → `03_build_yolo_dataset.py` → `05_validate_dataset.py` build the 2022–2024 training set (`scallop2224`); train with `job/yolo_job.sh` pointed at it.
2. `config/build_tierb_eval.py` builds the 2026 evaluation set from the 2026 annotation export (left camera frame of each stereo pair; images without live scallops become empties).
3. `job/eval_tierb_2026.sh` (or `eval_tierb_2026_cpu.sh`) scores it with the same settings as above. The calibration is then refitted in ScalPred (`s/tierb/`).

## Zenodo data record

`job/stage_zenodo.sh` collects the weights and evaluation outputs, and `config/package_zenodo_images.py` zips the Calibration-set images by year and region (with a manifest and checksums). `config/zenodo/` holds the record metadata and upload helper. The tabular files are built in ScalPred (`s/build_zenodo_tables.py`, `s/stage_pipeline_inputs.py`).

## Repository layout

```
train_arc/
  build_strat_labels_v2.py   transect-block split and YOLO labels (paper split)
  config/                    training, evaluation, timing, 2026-test and Zenodo scripts
  job/                       Slurm and GCP job scripts that call config/
s/, data/training/           earlier data-preparation and visualization scripts (2022-2024 work; not needed for the paper)
```

## Environment

Ultralytics (YOLOv12n, RT-DETR-L; the laptop timing used Ultralytics 8.4.50 and PyTorch 2.12), VIAME with its netharn Cascade R-CNN trainer, and Python 3 with pandas, numpy, scipy and Pillow.

## Citation

Jenrette, J.F., Hart, D.R., Chang, J.-H., Fairclough, C., McManus, M.C. PEMAD-ESB-ScalTrain: training and evaluation of sea scallop object detectors on HabCam imagery. Zenodo. https://doi.org/10.5281/zenodo.23482337

## License

Apache License 2.0 (see `LICENSE`).

## Contact

Jeremy F. Jenrette, NOAA Fisheries, Northeast Fisheries Science Center (jeremy.jenrette@noaa.gov)
