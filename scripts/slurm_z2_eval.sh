#!/usr/bin/env bash
# =============================================================================
# slurm_z2_eval.sh — Zone 2 extraction quality evaluation only
#
# Runs extraction_quality.py (Triple Precision, Fact Recall, Source Grounding)
# against the EXISTING Neo4j graph. Use after Zone 2 pipeline has run.
#
# Submit:  sbatch --export=ALL,SUFFIX=zone2_seaf_v2 scripts/slurm_z2_eval.sh
# Override: sbatch --export=ALL,MODEL=llama3.3:70b,SUFFIX=zone2_seaf_v2 scripts/slurm_z2_eval.sh
# =============================================================================

#SBATCH --job-name=cs584_z2eval
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

set -euo pipefail

SCRATCH="/local/scratch/${USER:?USER is not set}"
PROJECT="$SCRATCH/project"

MODEL="${MODEL:-qwen2.5:72b}"
SUFFIX="${SUFFIX:-zone2_seaf_v2}"

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

echo "============================================================"
echo "CS584 — Zone 2 Extraction Quality Evaluation"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Host:       $(hostname)"
echo "  Model:      $MODEL"
echo "  Suffix:     $SUFFIX"
echo "  Start:      $(date)"
echo "============================================================"
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
echo ""

# ---------------------------------------------------------------------------
# Zone 2 Extraction Quality Evaluation
# ---------------------------------------------------------------------------
echo "================================================================"
echo "ZONE 2 EVAL — Triple Precision, Fact Recall, Source Grounding"
echo "  suffix=$SUFFIX, model=$MODEL, sample-size=100"
echo "  Started: $(date)"
echo "================================================================"

EVAL_START=$(date +%s)
python3 evaluation/extraction_quality.py --suffix "$SUFFIX" --model "$MODEL" --sample-size 100
EVAL_END=$(date +%s)
EVAL_TIME=$((EVAL_END - EVAL_START))

echo ""
echo "Evaluation complete: ${EVAL_TIME}s ($(date))"
echo ""

# Print key metrics
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
    vc = d.get('vocabulary_coverage', {})
    print(f'=== Zone 2 Extraction Quality ({suffix}) ===')
    print(f'  Triple Precision:   {tp.get(\"precision\", 0):.1%} ({tp.get(\"correct\", 0)}/{tp.get(\"correct\", 0)+tp.get(\"incorrect\", 0)} correct)')
    print(f'  Fact Recall:        {fr.get(\"fact_recall\", 0):.1%} ({fr.get(\"found_facts\", 0)}/{fr.get(\"total_facts\", 0)} facts)')
    print(f'  Source Grounding:   {sg.get(\"grounding_rate\", 0):.1%} ({sg.get(\"supported\", 0)}/{sg.get(\"total_checked\", 0)} grounded)')
    print(f'  Vocab Coverage:     {vc.get(\"token_coverage\", 0):.1%} ({vc.get(\"covered_terms\", 0)}/{vc.get(\"source_vocab_size\", 0)} terms)')
    print(f'  Graph:              {gs.get(\"node_count\", \"?\")} nodes, {gs.get(\"edge_count\", \"?\")} edges, {gs.get(\"relation_types\", \"?\")} rel types')
except Exception as e: print(f'  (extraction quality eval error: {e})')
"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
kill $OLLAMA_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo "Job complete: $(date)"
echo "  Eval time: ${EVAL_TIME}s"
echo "  Results:   data/results/extraction_quality_${SUFFIX}.json"
echo "  Fetch:     bash scripts/sync_to_cluster.sh xwa2284 --fetch"
echo "============================================================"
