#!/bin/bash

# 1. Check if a year argument was provided
if [ -z "$1" ]; then
  echo "Error: Please provide a year as an argument."
  echo "Usage: ./copy_images.sh <YYYY>"
  exit 1
fi

YEAR=$1
BASE_DEST="/home/jeremy_jenrette/habcam_bucket/NEFSC/HabCam Survey/habcam/proc/Scall_Anno"

# 2. Define manifest and destination paths dynamically
MANIFEST_TR="../data/raw/sources_${YEAR}tr.txt"
DEST_TR="${BASE_DEST}/${YEAR}tr"

MANIFEST_ZERO="../data/raw/sources_${YEAR}_zero.txt"
DEST_ZERO="${BASE_DEST}/${YEAR}_zero"

# 3. Create a helper function to handle the copying logic
copy_images() {
  local manifest=$1
  local dest=$2
  local label=$3

  if [ -f "$manifest" ]; then
    echo "Starting copy for $YEAR $label images..."
    mkdir -p "$dest"
    
    # Read manifest and copy in parallel (-P 2 runs 2 concurrent processes)
    cat "$manifest" | xargs -n 1 -P 2 -I{} cp -n "{}" "$dest/"
    
    echo "Finished $label images."
  else
    echo "Skipping $label: Manifest file not found ($manifest)"
  fi
}

# 4. Execute the function for both 'tr' and 'zero'
copy_images "$MANIFEST_TR" "$DEST_TR" "Training (tr)"
copy_images "$MANIFEST_ZERO" "$DEST_ZERO" "Zero"