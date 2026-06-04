#!/bin/bash
# Step 1: Baseline language sensitivity diagnosis (OTB-Lang dataset)
#
# Each mode has its own YAML config + checkpoint symlink.
# Standard README-style commands, no env-var tricks.
#
# A0: normal    — dataset text as-is (baseline)
# A1: shuffle   — word-shuffled text
# A2: wrong     — semantically wrong cross-class text
# A3: generic   — "a moving object in the scene"
# A4: no_update — initial dataset text, never updated
#
# Usage: bash run_step1_lang_sensitivity.sh

set -e

TRACKER="dutrack"
DATASET="otb_lang"
THREADS=4
NUM_GPUS=1

MODES=("normal" "shuffle" "wrong" "generic" "no_update")

for mode in "${MODES[@]}"; do
    CONFIG="dutrack_384_full_lang_${mode}"
    echo "============================================"
    echo "  $(date)  LANG_MODE = $mode"
    echo "  Config: $CONFIG"
    echo "============================================"

    python tracking/test.py "$TRACKER" "$CONFIG" \
        --dataset_name "$DATASET" \
        --threads "$THREADS" \
        --num_gpus "$NUM_GPUS"

    echo "Done: $mode"
done

echo ""
echo "All 5 experiments complete."
echo ""
echo "Results:"
for mode in "${MODES[@]}"; do
    echo "  output/test/tracking_results/${TRACKER}/dutrack_384_full_lang_${mode}/"
done
