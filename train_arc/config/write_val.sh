#!/bin/bash
# ─── Helper: write validation image manifest ─────────────
write_val_image_csv () {
    DATA_ROOT="$1"
    OUT_CSV="$2"

    mkdir -p "$(dirname "$OUT_CSV")"

    echo "imagename,img_path" > "$OUT_CSV"

    find "${DATA_ROOT}/images/val" \
        -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
        | sort \
        | awk -F/ '{print $NF "," $0}' >> "$OUT_CSV"

    echo "Validation image CSV written to: $OUT_CSV"
}