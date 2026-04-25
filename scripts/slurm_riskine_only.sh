#!/bin/bash
# =============================================================================
# slurm_riskine_only.sh — Skip extraction_quality, run only baseline/eval.py
# (Riskine alignment + 40-task query benchmark). Useful when Aura has the
# right KG but extraction_quality numbers are already final.
#
# Usage:
#   sbatch --export=ALL,DATA_DIR=data/flood,JUDGE_MODEL=gemma4:31b \
#       scripts/slurm_riskine_only.sh
# =============================================================================

#SBATCH --job-name=riskine
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/local/scratch/%u/logs/riskine_%j.out
#SBATCH --error=/local/scratch/%u/logs/riskine_%j.err

SCRATCH=/local/scratch/$USER
source "$SCRATCH/project/scripts/_env.sh"

_start_ollama

echo ""
echo "===== RISKINE EVAL (judge=$JUDGE_MODEL) ====="
python3 baseline/eval.py \
    --results-dir "${RESULTS_DIR}/eval" \
    --suffix "$SUFFIX" --riskine --all-classes --model "$JUDGE_MODEL"

_stop_ollama

echo ""
echo "===== RISKINE EVAL COMPLETE — $(date) ====="
echo "  Results: ${RESULTS_DIR}/eval/"
