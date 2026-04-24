#!/bin/bash
# =============================================================================
# slurm_eval.sh — Eval only (no pipeline re-run)
#
# Requires: Zone 2 graph in Neo4j, Zone 3 ontology in Neo4j
#
# Usage:
#   # Default (LLM-as-judge = MODEL, enables self-evaluation bias):
#   sbatch --export=ALL,DATA_DIR=data/Emory_Spring2026 scripts/slurm_eval.sh
#
#   # Recommended: use a different judge model for triple-precision + Riskine scoring:
#   sbatch --export=ALL,DATA_DIR=data/Emory_Spring2026,JUDGE_MODEL=llama3.3:70b scripts/slurm_eval.sh
# =============================================================================

#SBATCH --job-name=eval
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/local/scratch/%u/logs/eval_%j.out
#SBATCH --error=/local/scratch/%u/logs/eval_%j.err

SCRATCH=/local/scratch/$USER
source "$SCRATCH/project/scripts/_env.sh"

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
# Uses JUDGE_MODEL so the LLM judging class alignment is NOT the same model
# that produced the extraction — prevents self-evaluation bias.
echo ""
echo "===== RISKINE EVAL (26 classes, judge=$JUDGE_MODEL) ====="
python3 baseline/eval.py \
    --results-dir "${RESULTS_DIR}/eval" \
    --suffix "$SUFFIX" --riskine --all-classes --model "$JUDGE_MODEL"

_stop_ollama

echo ""
echo "===== EVAL COMPLETE — $(date) ====="
echo "  Results: ${RESULTS_DIR}/eval/"
