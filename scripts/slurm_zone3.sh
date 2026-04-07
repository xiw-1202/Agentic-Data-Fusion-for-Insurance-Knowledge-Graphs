#!/bin/bash
# =============================================================================
# slurm_zone3.sh — Zone 3 SV-LOI ontology induction + Riskine eval
#
# Requires: Zone 2 already ran (zone2_run_summary.json exists)
#
# Usage:
#   sbatch --export=ALL,DATA_DIR=data/Emory_Spring2026 scripts/slurm_zone3.sh
#   sbatch --export=ALL,MODEL=gemma4:31b,DATA_DIR=data/Emory_Spring2026 scripts/slurm_zone3.sh
# =============================================================================

#SBATCH --job-name=zone3
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

SCRATCH=/local/scratch/$USER
source "$SCRATCH/project/scripts/_env.sh"

_start_ollama

echo ""
echo "===== ZONE 3 — SV-LOI Ontology Induction ($MODEL) ====="
ZONE3_START=$(date +%s)
python3 zone3/sv_loi.py \
    --model "$MODEL" --suffix "$SUFFIX" --results-dir "$RESULTS_DIR"
echo "Zone 3 complete: $(($(date +%s) - ZONE3_START))s"

echo ""
echo "===== ZONE 3 EVAL — Riskine Ontology Alignment (26 classes) ====="
python3 baseline/eval.py \
    --suffix "$SUFFIX" --riskine --all-classes --model "$MODEL"

_stop_ollama

echo ""
echo "===== ZONE 3 COMPLETE — $(date) ====="
echo "  Results: $RESULTS_DIR/"
