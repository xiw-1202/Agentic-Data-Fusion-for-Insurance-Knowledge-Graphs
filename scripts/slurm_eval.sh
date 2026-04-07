#!/bin/bash
# =============================================================================
# slurm_eval.sh — Eval only (no pipeline re-run)
#
# Requires: Zone 2 graph in Neo4j, Zone 3 ontology in Neo4j
#
# Usage:
#   sbatch --export=ALL,DATA_DIR=data/Emory_Spring2026 scripts/slurm_eval.sh
#   sbatch --export=ALL,DATA_DIR=data/Emory_Spring2026,JUDGE_MODEL=gemma4:31b scripts/slurm_eval.sh
# =============================================================================

#SBATCH --job-name=eval
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/local/scratch/%u/logs/eval_%j.out
#SBATCH --error=/local/scratch/%u/logs/eval_%j.err

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

_start_ollama

# ===== Extraction Quality Eval =====
echo ""
echo "===== EXTRACTION QUALITY EVAL ====="
JUDGE_FLAG=""
[ "$JUDGE_MODEL" != "$MODEL" ] && JUDGE_FLAG="--judge-model $JUDGE_MODEL"
python3 evaluation/extraction_quality.py \
    --suffix "$SUFFIX" --model "$MODEL" --sample-size 100 \
    --chunks "$CHUNKS_FILE" --results-dir "${RESULTS_DIR}/eval" $JUDGE_FLAG

# ===== Riskine Ontology Alignment =====
echo ""
echo "===== RISKINE EVAL (26 classes) ====="
python3 baseline/eval.py \
    --suffix "$SUFFIX" --riskine --all-classes --model "$MODEL"

_stop_ollama

echo ""
echo "===== EVAL COMPLETE — $(date) ====="
echo "  Results: ${RESULTS_DIR}/eval/"
