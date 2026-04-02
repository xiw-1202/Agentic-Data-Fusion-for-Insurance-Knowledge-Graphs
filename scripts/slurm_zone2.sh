#!/usr/bin/env bash
# =============================================================================
# slurm_zone2.sh — Zone 2 extraction pipeline + extraction quality eval
#
# Runs ONLY Zone 2 (Open IE extraction) and its own eval metrics:
#   Triple Precision, Fact Recall, Source Grounding
#
# Does NOT run Zone 3 or Riskine eval — use slurm_sv_loi.sh etc. for that.
#
# Submit:  sbatch --export=ALL,SUFFIX=zone2_seaf_v4 scripts/slurm_zone2.sh
# Override: sbatch --export=ALL,MODEL=llama3.3:70b,SUFFIX=zone2_v4_llama scripts/slurm_zone2.sh
# =============================================================================

#SBATCH --job-name=cs584_zone2
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

set -euo pipefail

SCRATCH="/local/scratch/${USER:?USER is not set}"
PROJECT="$SCRATCH/project"

MODEL="${MODEL:-qwen2.5:72b}"
SUFFIX="${SUFFIX:-zone2_seaf}"
PASSES="${PASSES:-1}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export PATH="$SCRATCH/bin:$PATH"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$SCRATCH/.cache/huggingface"
export OLLAMA_MODELS="$SCRATCH/models"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export LD_LIBRARY_PATH="$SCRATCH/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_CONTEXT_LENGTH=4096
export CUDA_VISIBLE_DEVICES=0,1
export OLLAMA_NUM_GPU=999

source "$SCRATCH/venv/bin/activate"
cd "$PROJECT"

mkdir -p "$SCRATCH/logs"
pip install -r "$PROJECT/requirements.txt" --quiet

echo "============================================================"
echo "CS584 — Zone 2 Extraction Pipeline"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Host:       $(hostname)"
echo "  Model:      $MODEL"
echo "  Suffix:     $SUFFIX"
echo "  Passes:     $PASSES"
echo "  Start:      $(date)"
echo "============================================================"
echo ""

# GPU info
echo "=== GPU Configuration ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | paste -sd+ | bc)
echo ""
echo "Total VRAM: ${TOTAL_VRAM} MB"
echo ""

# ---------------------------------------------------------------------------
# Start Ollama
# ---------------------------------------------------------------------------
echo "[setup] Starting Ollama server..."
ollama serve &>"$SCRATCH/logs/ollama_$SLURM_JOB_ID.log" &
OLLAMA_PID=$!

OLLAMA_READY=0
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "[setup] Ollama ready (${i}s)"
        OLLAMA_READY=1
        break
    fi
    sleep 1
done
if [ "$OLLAMA_READY" -eq 0 ]; then
    echo "ERROR: Ollama did not become ready within 30s." >&2
    tail -20 "$SCRATCH/logs/ollama_$SLURM_JOB_ID.log" 2>/dev/null || true
    exit 1
fi

echo "[setup] Loading $MODEL into GPU..."
timeout 300 ollama run "$MODEL" "Hello, respond with one word." --nowordwrap 2>&1 | head -5 || true
echo "[setup] Model loaded:"
grep -i "layer" "$SCRATCH/logs/ollama_$SLURM_JOB_ID.log" 2>/dev/null | tail -3 || true
echo ""

# ---------------------------------------------------------------------------
# Zone 2: Extraction
# ---------------------------------------------------------------------------
echo "================================================================"
echo "ZONE 2 — Open IE extraction with $MODEL"
echo "  Started: $(date)"
echo "================================================================"

rm -f "$PROJECT/data/results/zone2_vocab.json"

ZONE2_START=$(date +%s)
python3 zone2/pipeline.py --model "$MODEL" --passes "$PASSES"
ZONE2_END=$(date +%s)
ZONE2_TIME=$((ZONE2_END - ZONE2_START))

echo ""
echo "Zone 2 extraction complete: ${ZONE2_TIME}s ($(date))"
echo ""

# ---------------------------------------------------------------------------
# Zone 2 Evaluation: Triple Precision, Fact Recall, Source Grounding
# ---------------------------------------------------------------------------
echo "================================================================"
echo "ZONE 2 EVAL — Extraction Quality (suffix=$SUFFIX)"
echo "  Started: $(date)"
echo "================================================================"

EVAL_START=$(date +%s)
python3 evaluation/extraction_quality.py --suffix "$SUFFIX" --model "$MODEL" --sample-size 100
EVAL_END=$(date +%s)
EVAL_TIME=$((EVAL_END - EVAL_START))

echo ""
echo "Zone 2 eval complete: ${EVAL_TIME}s ($(date))"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
kill $OLLAMA_PID 2>/dev/null || true

SUFFIX="$SUFFIX" python3 -c "
import json, os
suffix = os.environ['SUFFIX']
try:
    with open(f'data/results/extraction_quality_{suffix}.json') as f:
        d = json.load(f)
    tp = d.get('triple_precision', {})
    fr = d.get('fact_recall', {})
    sg = d.get('source_grounding', {})
    gs = d.get('graph_statistics', {})
    print(f'=== Zone 2 Results ({suffix}) ===')
    print(f'  Triple Precision:   {tp.get(\"precision\", 0):.1%} ({tp.get(\"correct\", 0)}/{tp.get(\"correct\", 0)+tp.get(\"incorrect\", 0)} correct)')
    print(f'  Fact Recall:        {fr.get(\"fact_recall\", 0):.1%} ({fr.get(\"found_facts\", 0)}/{fr.get(\"total_facts\", 0)} facts)')
    print(f'  Source Grounding:   {sg.get(\"grounding_rate\", 0):.1%} ({sg.get(\"supported\", 0)}/{sg.get(\"total_checked\", 0)} grounded)')
    print(f'  Graph:              {gs.get(\"node_count\", \"?\")} nodes, {gs.get(\"edge_count\", \"?\")} edges, {gs.get(\"relation_types\", \"?\")} rel types')
except Exception as e: print(f'  (eval not found: {e})')
"

echo ""
echo "============================================================"
echo "Job complete: $(date)"
echo ""
echo "Timing:"
echo "  Extraction: ${ZONE2_TIME}s"
echo "  Evaluation: ${EVAL_TIME}s"
echo "  Total:      $((ZONE2_TIME + EVAL_TIME))s"
echo ""
echo "Results:"
echo "  data/results/zone2_run_summary.json"
echo "  data/results/extraction_quality_${SUFFIX}.json"
echo ""
echo "NEXT STEPS:"
echo "  1. Fetch:    bash scripts/sync_to_cluster.sh $USER --fetch"
echo "  2. Zone 3:   sbatch --export=ALL,MODEL=$MODEL,SUFFIX=zone3_svloi scripts/slurm_sv_loi.sh"
echo "============================================================"
