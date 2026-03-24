#!/usr/bin/env bash
# =============================================================================
# slurm_sv_loi.sh — Slurm job: SV-LOI ontology induction (Zone 3 only)
#
# Runs SV-LOI on an EXISTING Zone 2 graph (no re-extraction).
# Zone 2 must have already populated Neo4j with :Entity nodes.
#
# Submit:  sbatch scripts/slurm_sv_loi.sh
# Monitor: squeue -u $USER
# Logs:    tail -f /local/scratch/$USER/logs/<jobid>.out
#
# Override: sbatch --export=ALL,MODEL=llama3.3:70b,SUFFIX=zone3_svloi_llama scripts/slurm_sv_loi.sh
# =============================================================================

#SBATCH --job-name=cs584_sv_loi
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

set -euo pipefail

SCRATCH="/local/scratch/${USER:?USER is not set}"
PROJECT="$SCRATCH/project"

MODEL="${MODEL:-qwen2.5:72b}"
SUFFIX="${SUFFIX:-zone3_svloi}"

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
echo "CS584 — SV-LOI Ontology Induction (Zone 3 Only)"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Host:       $(hostname)"
echo "  Model:      $MODEL"
echo "  Suffix:     $SUFFIX"
echo "  GPUs:       $SLURM_GPUS_ON_NODE"
echo "  Start:      $(date)"
echo "============================================================"
echo ""

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
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
# SV-LOI (Zone 3)
# ---------------------------------------------------------------------------
echo "================================================================"
echo "ZONE 3 — SV-LOI with $MODEL"
echo "  Started: $(date)"
echo "================================================================"

ZONE3_START=$(date +%s)
python3 zone3/sv_loi.py \
    --model "$MODEL" \
    --suffix "$SUFFIX"
ZONE3_END=$(date +%s)
ZONE3_TIME=$((ZONE3_END - ZONE3_START))

echo ""
echo "SV-LOI complete: ${ZONE3_TIME}s ($(date))"
echo ""

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
echo "================================================================"
echo "EVALUATION — suffix=$SUFFIX"
echo "  Started: $(date)"
echo "================================================================"

EVAL_START=$(date +%s)
python3 baseline/eval.py --suffix "$SUFFIX" --riskine --model "$MODEL"
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
    with open(f'data/results/baseline_eval_results_{suffix}.json') as f:
        d = json.load(f)
    bm = d.get('baseline_metrics', {})
    print(f'=== Baseline Eval ({suffix}) ===')
    print(f'  Query accuracy:     {bm.get(\"query_accuracy\", 0):.1%}')
    print(f'  Type inconsistency: {bm.get(\"type_inconsistency_rate\", 0):.1%}')
    print(f'  Duplication:        {bm.get(\"duplication_rate\", 0):.1%}')
except Exception as e: print(f'  (baseline eval not found: {e})')

try:
    with open(f'data/results/riskine_eval_{suffix}.json') as f:
        d = json.load(f)
    print(f'=== Riskine Eval ({suffix}) ===')
    print(f'  Entity Assign F1:   {d.get(\"entity_assignment_f1\", 0):.3f}')
    print(f'  Entity Assign P:    {d.get(\"entity_assignment_precision\", 0):.3f}')
    print(f'  Entity Assign R:    {d.get(\"entity_assignment_recall\", 0):.3f}')
    print(f'  Classes covered:    {d.get(\"entity_assignment_riskine_covered\", [])}')
    sm = d.get('standard_metrics', {})
    if sm:
        print(f'  BERTScore F1:       {sm.get(\"bertscore_f1\", 0):.3f}')
        print(f'  Graph F1:           {sm.get(\"graph_f1\", 0):.3f}')
        print(f'  Continuous F1:      {sm.get(\"continuous_f1\", 0):.3f}')
        print(f'  Wu-Palmer:          {sm.get(\"avg_wu_palmer\", 0):.3f}')
except Exception as e: print(f'  (riskine eval not found: {e})')
"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
kill $OLLAMA_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo "Job complete: $(date)"
echo ""
echo "Timing:"
echo "  Zone 3 (SV-LOI):  ${ZONE3_TIME}s"
echo "  Evaluation:        ${EVAL_TIME}s"
echo "  Total:             $((ZONE3_TIME + EVAL_TIME))s"
echo ""
echo "Results:"
echo "  zone3_svloi_summary_${SUFFIX}.json"
echo "  baseline_eval_results_${SUFFIX}.json"
echo "  riskine_eval_${SUFFIX}.json"
echo ""
echo "Fetch:  bash scripts/sync_to_cluster.sh xwa2284 --fetch"
echo "============================================================"
