#!/bin/bash
# =============================================================================
# slurm_pipeline.sh — Full pipeline: Zone 1 → Zone 2 → Zone 3 (SV-LOI) → Eval
#
# Usage:
#   sbatch scripts/slurm_pipeline.sh                                    # flood, default
#   sbatch --export=ALL,DATA_DIR=data/Emory_Spring2026 scripts/slurm_pipeline.sh
#   sbatch --export=ALL,MODEL=gemma4:31b,DATA_DIR=data/Emory_Spring2026 scripts/slurm_pipeline.sh
# =============================================================================

#SBATCH --job-name=pipeline
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

_start_ollama

# ===== Zone 1: Ingestion (only for non-flood data) =====
if [ "$DATA_DIR" != "data/flood" ]; then
    echo ""
    echo "===== ZONE 1 — Ingestion ($DATA_DIR) ====="
    ZONE1_START=$(date +%s)
    python3 zone1/ingestion.py \
        --data-dir "$DATA_DIR" \
        --output "$CHUNKS_FILE" \
        --model "$MODEL"
    echo "Zone 1 complete: $(($(date +%s) - ZONE1_START))s"
fi

# ===== Zone 2: Extraction =====
echo ""
echo "===== ZONE 2 — Extraction ($MODEL) ====="
rm -f "$RESULTS_DIR/zone2_vocab.json"
ZONE2_START=$(date +%s)
python3 zone2/pipeline.py \
    --model "$MODEL" --passes "$PASSES" \
    --chunks "$CHUNKS_FILE" --results-dir "$RESULTS_DIR"
echo "Zone 2 complete: $(($(date +%s) - ZONE2_START))s"

# ===== Zone 2 Eval: Extraction Quality =====
echo ""
echo "===== ZONE 2 EVAL — Extraction Quality ====="
JUDGE_FLAG=""
[ "$JUDGE_MODEL" != "$MODEL" ] && JUDGE_FLAG="--judge-model $JUDGE_MODEL"
python3 evaluation/extraction_quality.py \
    --suffix "$SUFFIX" --model "$MODEL" --sample-size 100 \
    --chunks "$CHUNKS_FILE" --results-dir "${RESULTS_DIR}/eval" $JUDGE_FLAG

# ===== Zone 3: Ontology Induction (SV-LOI) =====
echo ""
echo "===== ZONE 3 — SV-LOI Ontology Induction ====="
ZONE3_START=$(date +%s)
python3 zone3/sv_loi.py \
    --model "$MODEL" --suffix "$SUFFIX" --results-dir "$RESULTS_DIR"
echo "Zone 3 complete: $(($(date +%s) - ZONE3_START))s"

# ===== Zone 3 Eval: Riskine Alignment =====
echo ""
echo "===== ZONE 3 EVAL — Riskine Ontology Alignment ====="
python3 baseline/eval.py \
    --suffix "$SUFFIX" --riskine --all-classes --model "$MODEL"

_stop_ollama

echo ""
echo "===== PIPELINE COMPLETE ====="
echo "  Results: $RESULTS_DIR/"
echo "  Finished: $(date)"
