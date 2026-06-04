#!/bin/bash
# Step 1: Baseline language sensitivity diagnosis (HOOT dataset, ALL 130 sequences)
#
# Standard README-style commands.
#
# A0: normal    — dataset/BLIP text as-is (baseline)
# A1: shuffle   — word-shuffled text
# A2: wrong     — semantically wrong cross-class text
# A3: generic   — "a moving object in the scene"
# A4: no_update — initial text, never updated

set -e

TRACKER="dutrack"
DATASET="hoot_all"
THREADS=4
NUM_GPUS=1

MODES=("normal" "shuffle" "wrong" "generic" "no_update")

for mode in "${MODES[@]}"; do
    CONFIG="dutrack_384_full_hoot_all_lang_${mode}"
    echo "============================================"
    echo "  $(date)  HOOT ALL (130 seqs)  LANG_MODE = $mode"
    echo "  Config: $CONFIG"
    echo "============================================"

    python tracking/test.py "$TRACKER" "$CONFIG" \
        --dataset_name "$DATASET" \
        --threads "$THREADS" \
        --num_gpus "$NUM_GPUS"

    echo "Done: $mode"
done

echo ""
echo "All 5 HOOT experiments complete (130 sequences each)."
echo ""
echo "Results:"
for mode in "${MODES[@]}"; do
    echo "  output/test/tracking_results/${TRACKER}/dutrack_384_full_hoot_all_lang_${mode}/"
done
