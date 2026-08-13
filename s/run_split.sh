#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "❌ Usage: $0 <YEAR> <SIDE>"
    exit 1
fi

YEAR=$1
SIDE=$2

RAW_DIR="../data/raw"
# 1. Write everything locally first (no permissions issues here!)
LOCAL_OUT_DIR="../data/processed/${YEAR}"
# 2. Define your final Cloud bucket destination
CLOUD_DEST="gs://nmfs-dev-uc1-landing-bucket/NEFSC/HabCam Survey/habcam/proc/Scall_Anno/${YEAR}"

echo "============================================================"
echo "🚀 Starting Pipeline for Year: $YEAR | Splitting: $SIDE"
echo "============================================================"

# Create local directories
mkdir -p "${LOCAL_OUT_DIR}/${YEAR}tr_split"
mkdir -p "${LOCAL_OUT_DIR}/${YEAR}_zero_split"

# Run Python targeting the LOCAL directories
# Run Python targeting the LOCAL directories
python split_script.py \
    --ann_csv "${RAW_DIR}/${YEAR}_annotations.csv" \
    --ann_src_txt "${RAW_DIR}/sources_${YEAR}tr.txt" \
    --out_ann_img_dir "${LOCAL_OUT_DIR}/${YEAR}tr_split" \
    --out_ann_csv "${LOCAL_OUT_DIR}/${YEAR}_annotations.csv" \
    --zero_csv "${RAW_DIR}/${YEAR}_zero.csv" \
    --zero_src_txt "${RAW_DIR}/sources_${YEAR}_zero.txt" \
    --out_zero_img_dir "${LOCAL_OUT_DIR}/${YEAR}_zero_split" \
    --out_zero_csv "${LOCAL_OUT_DIR}/${YEAR}_zero.csv" \
    --side "${SIDE}"

echo "☁️ Uploading results to Google Cloud Storage..."
# Bulk upload everything in parallel (-m) to the bucket
gsutil -m cp -r "${LOCAL_OUT_DIR}/*" "${CLOUD_DEST}/"

echo "✅ Finished Year: $YEAR"