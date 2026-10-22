#!/bin/bash
#SBATCH -J bench_viame
#SBATCH --account=sharkpulse
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/bench_viame_%j.out
#SBATCH --error=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/bench_viame_%j.err
# ==============================================================================
# N8: inference timing for the VIAME Cascade R-CNN, matched to config/bench_inference.py
#   - same 520 images (same seeded draw from data2226/yolo/images/val), left frame only
#   - first 20 images = warm-up; the timed set is the other 500
#   - VIAME times a whole pipeline run, so the per-image cost is taken as a difference:
#       per_image = (T[520 images] - T[20 images]) / 500
#     which cancels VIAME start-up and model loading. It includes reading each PNG
#     (VIAME's own reader), so compare it with bench_inference.json "end_to_end".
#
# GPU:  sbatch job/benchmark_viame_job.sh
# CPU:  DEVICE=cpu sbatch --partition=normal_q --gres=gpu:0 job/benchmark_viame_job.sh
# ==============================================================================
set -e
BASE_DIR="/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc"
MODEL_DIR="${BASE_DIR}/output/2226data_viame_slurm_598695"
DEPLOY="${MODEL_DIR}/deploy.zip"
IMAGES="${BASE_DIR}/data2226/yolo/images/val"
N=500; WARMUP=20; CROP_W=1936
DEVICE="${DEVICE:-gpu}"
WORK="${BASE_DIR}/output/bench_viame_${SLURM_JOB_ID:-local}"
mkdir -p "${WORK}/img" "${BASE_DIR}/log"
cd "${WORK}"
[ -f "${DEPLOY}" ] || { echo "ERROR: ${DEPLOY} not found"; exit 1; }

# ─── 1. Same image draw as bench_inference.py; crop stereo pairs to the left frame ──
module reset 2>/dev/null || true
module load Miniconda3/24.7.1-0 2>/dev/null || true
source ~/.bashrc
conda activate scallopdet
python - "${IMAGES}" "${N}" "${WARMUP}" "${CROP_W}" <<'PY'
import sys, os
from pathlib import Path
import numpy as np, cv2
src, n, warm, crop_w = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
rng = np.random.default_rng(1)                                  # identical to bench_inference.py
imgs = [imgs[i] for i in rng.choice(len(imgs), size=min(n + warm, len(imgs)), replace=False)]
out, n_crop = [], 0
for p in imgs:
    q = Path("img") / p.name
    x = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if x is not None and x.shape[1] > crop_w:                    # stereo pair -> left frame
        cv2.imwrite(str(q), x[:, :crop_w]); n_crop += 1
    elif not q.exists():
        os.symlink(p, q)
    out.append(str(q.resolve()))
open("list_warmup.txt", "w").write("\n".join(out[:warm]) + "\n")
open("list_all.txt", "w").write("\n".join(out) + "\n")
print(f"{len(out)} images ({n_crop} cropped to the left frame)")
PY
conda deactivate

# ─── 2. VIAME environment ───────────────────────────────────────────────────
export VIAME_INSTALL="/projects/sharkpulse/archived/viame"
source ${VIAME_INSTALL}/setup_viame.sh
export KWIVER_DEFAULT_LOG_LEVEL=warn
if [ "${DEVICE}" = "cpu" ]; then export CUDA_VISIBLE_DEVICES=""; else export CUDA_VISIBLE_DEVICES=0; fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)

# Detector pipe = config/embedded_default.pipe (the training template) with the netharn
# detector filled in, plus an image-list reader and a CSV writer.
write_pipe () {   # $1 = image list, $2 = output csv, $3 = pipe file
cat > "$3" <<EOF
config _pipeline:_edge
  :capacity                                    10

process input
  :: video_input
  :video_filename                              $1
  :frame_time                                  1
  :exit_on_invalid                             false
  :video_reader:type                           image_list
  block video_reader:image_list
    :image_reader:type                         vxl
    block image_reader:vxl
      :force_byte                              true
    endblock
  endblock

process detector_input
  :: image_filter
  :filter:type                                 vxl_convert_image
  block filter:vxl_convert_image
    :format                                    byte
    :force_three_channel                       true
  endblock

connect from input.image
        to   detector_input.image

process detector1
  :: image_object_detector
  :detector:type                               netharn
  block detector:netharn
    :deployed                                  ${DEPLOY}
  endblock

connect from detector_input.image
        to   detector1.image

process detector_output
  :: refine_detections
  :refiner:type                                nms
  block refiner:nms
    :max_overlap                               0.50
    :nms_scale_factor                          1.0
    :output_scale_factor                       1.0
  endblock

connect from detector1.detected_object_set
        to   detector_output.detected_object_set

process writer
  :: detected_object_output
  :file_name                                   $2
  :writer:type                                 viame_csv

connect from detector_output.detected_object_set
        to   writer.detected_object_set
connect from input.file_name
        to   writer.image_file_name
EOF
}

run_pipe () {     # prints wall seconds
  if command -v kwiver >/dev/null 2>&1; then RUNNER="kwiver runner"; else RUNNER="pipeline_runner -p"; fi
  local t0=$(date +%s.%N)
  ${RUNNER} "$1" > "$1.log" 2>&1 || { echo "ERROR: pipeline failed, see ${WORK}/$1.log" >&2; tail -30 "$1.log" >&2; exit 1; }
  local t1=$(date +%s.%N)
  echo "$t1 - $t0" | bc -l
}

write_pipe "${WORK}/list_warmup.txt" "${WORK}/dets_warmup.csv" bench_warmup.pipe
write_pipe "${WORK}/list_all.txt"    "${WORK}/dets_all.csv"    bench_all.pipe

echo "Run 1: ${WARMUP} images (start-up + model load + warm-up)"
T_W=$(run_pipe bench_warmup.pipe)
echo "Run 2: $((N + WARMUP)) images"
T_A=$(run_pipe bench_all.pipe)
N_DET=$(grep -vc '^#' dets_all.csv || true)

# ─── 3. Summary JSON, comparable with bench_inference.json ──────────────────
python3 - "$T_W" "$T_A" "$N" "$WARMUP" "${GPU_NAME}" "${DEVICE}" "${N_DET}" <<'PY'
import json, sys, platform, os
tw, ta, n, warm = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
per = (ta - tw) / n
res = {"hardware": {"machine": platform.platform(), "device": sys.argv[6], "gpu": sys.argv[5] or None,
                    "cpu_threads": int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or None},
       "settings": {"model": "VIAME Cascade R-CNN (netharn), 2226data_viame_slurm_598695/deploy.zip",
                    "n": n, "warmup": warm, "nms": 0.5, "frame": "left (1936 px)",
                    "method": "per_image = (T[n+warmup] - T[warmup]) / n; includes image reading"},
       "models": {"cascade_rcnn_viame": {"n_images": n, "wall_s_warmup_run": round(tw, 2),
                  "wall_s_full_run": round(ta, 2), "end_to_end_ms_per_image": round(1000 * per, 1),
                  "end_to_end_images_per_s": round(1 / per, 2), "detection_rows": int(sys.argv[7] or 0)}}}
json.dump(res, open("bench_viame.json", "w"), indent=1)
print(json.dumps(res, indent=1))
PY
cp bench_viame.json "${BASE_DIR}/bench_viame_${DEVICE}.json"
rm -rf "${WORK}/img"      # cropped copies and links only; the source images are untouched
echo "=== Done: ${BASE_DIR}/bench_viame_${DEVICE}.json ==="
