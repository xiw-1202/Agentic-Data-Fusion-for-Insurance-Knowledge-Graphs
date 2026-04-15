#!/bin/bash
# =============================================================================
# slurm_zone3_recursive.sh — Zone 3 Recursive Divisive Ontology Induction + Eval
#
# Requires: Zone 2 already ran (zone2_run_summary.json + graph cache exist)
#
# Usage:
#   sbatch --export=ALL,DATA_DIR=data/Emory_Spring2026 scripts/slurm_zone3_recursive.sh
#   sbatch --export=ALL,MODEL=qwen2.5:72b,DATA_DIR=data/Emory_Spring2026 scripts/slurm_zone3_recursive.sh
# =============================================================================

#SBATCH --job-name=z3-recursive
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

SCRATCH=/local/scratch/$USER
source "$SCRATCH/project/scripts/_env.sh"

_start_ollama

echo ""
echo "===== ZONE 3 — Recursive Divisive Ontology Induction ($MODEL) ====="
ZONE3_START=$(date +%s)
python3 zone3/recursive_induction.py \
    --model "$MODEL" \
    --suffix "recursive_${SUFFIX}" \
    --results-dir "$RESULTS_DIR" \
    --data-dir "$DATA_DIR"
echo "Zone 3 (recursive) complete: $(($(date +%s) - ZONE3_START))s"

echo ""
echo "===== ZONE 3 EVAL — Riskine Ontology Alignment ====="
python3 baseline/eval.py \
    --results-dir "${RESULTS_DIR}/eval" \
    --suffix "recursive_${SUFFIX}" --riskine --all-classes --model "$MODEL"

_stop_ollama

echo ""
echo "===== ZONE 3 RECURSIVE COMPLETE — $(date) ====="
echo "  Results: $RESULTS_DIR/"
