#!/bin/bash
# =============================================================================
# slurm_zone2.sh — Zone 2 extraction + eval (Zone 1 chunks must exist)
#
# Usage:
#   sbatch --export=ALL,DATA_DIR=data/Emory_Spring2026 scripts/slurm_zone2.sh
#   sbatch --export=ALL,MODEL=gemma4:31b,DATA_DIR=data/Emory_Spring2026 scripts/slurm_zone2.sh
# =============================================================================

#SBATCH --job-name=zone2
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

if [ ! -f "$CHUNKS_FILE" ]; then
    echo "ERROR: $CHUNKS_FILE not found. Run slurm_pipeline.sh first."
    exit 1
fi

_start_ollama

echo ""
echo "===== ZONE 2 — Extraction ($MODEL) ====="
rm -f "$RESULTS_DIR/zone2_vocab.json"
ZONE2_START=$(date +%s)
python3 zone2/pipeline.py \
    --model "$MODEL" --passes "$PASSES" \
    --chunks "$CHUNKS_FILE" --results-dir "$RESULTS_DIR"
echo "Zone 2 complete: $(($(date +%s) - ZONE2_START))s"

echo ""
echo "===== ZONE 2 EVAL — Extraction Quality ====="
JUDGE_FLAG=""
[ "$JUDGE_MODEL" != "$MODEL" ] && JUDGE_FLAG="--judge-model $JUDGE_MODEL"
python3 evaluation/extraction_quality.py \
    --suffix "$SUFFIX" --model "$MODEL" --sample-size 100 \
    --chunks "$CHUNKS_FILE" --results-dir "${RESULTS_DIR}/eval" $JUDGE_FLAG

_stop_ollama

echo ""
echo "===== ZONE 2 COMPLETE — $(date) ====="
echo "  Results: $RESULTS_DIR/"
