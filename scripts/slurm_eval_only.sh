#!/usr/bin/env bash
# =============================================================================
# slurm_eval_only.sh — Slurm job: Evaluation only (no Zone 2/3 re-run)
#
# Runs eval against the EXISTING Neo4j graph. Use after Zone 3 has already run.
#
# Submit:  sbatch --export=ALL,SUFFIX=zone3_svloi_seaf scripts/slurm_eval_only.sh
# Override: sbatch --export=ALL,MODEL=llama3.3:70b,SUFFIX=zone3_rsi_seaf scripts/slurm_eval_only.sh
# =============================================================================

#SBATCH --job-name=cs584_eval
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

set -euo pipefail

SCRATCH="/local/scratch/${USER:?USER is not set}"
PROJECT="$SCRATCH/project"

MODEL="${MODEL:-qwen2.5:72b}"
SUFFIX="${SUFFIX:-zone3_svloi_seaf}"

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
echo "CS584 — Evaluation Only"
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
# Evaluation
# ---------------------------------------------------------------------------
echo "================================================================"
echo "EVALUATION — suffix=$SUFFIX, model=$MODEL, all-classes=yes"
echo "  Started: $(date)"
echo "================================================================"

EVAL_START=$(date +%s)
python3 baseline/eval.py --suffix "$SUFFIX" --riskine --all-classes --model "$MODEL"
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
    with open(f'data/results/riskine_eval_{suffix}.json') as f:
        d = json.load(f)
    print(f'=== Riskine Eval ({suffix}) ===')
    print(f'  Entity Assign F1 (full):    {d.get(\"entity_assignment_f1\", 0):.3f}')
    print(f'  Entity Assign F1 (present): {d.get(\"entity_assignment_f1_present\", 0):.3f}')
    print(f'  Classes covered:            {d.get(\"entity_assignment_riskine_covered\", [])}')
    sm = d.get('standard_metrics', {})
    if sm:
        print(f'  BERTScore F1:               {sm.get(\"bertscore_f1\", 0):.3f}')
        print(f'  Graph F1:                   {sm.get(\"graph_f1\", 0):.3f}')
        print(f'  Continuous F1:              {sm.get(\"continuous_f1\", 0):.3f}')
        print(f'  Wu-Palmer:                  {sm.get(\"avg_wu_palmer\", 0):.3f}')
        print(f'  AUC-ROC:                    {sm.get(\"auc_roc\", 0):.3f}')
        print(f'  PR-AUC:                     {sm.get(\"pr_auc\", 0):.3f}')
        print(f'  MRR:                        {sm.get(\"mrr\", 0):.3f}')
        print(f'  Recall@1:                   {sm.get(\"recall_at_1\", 0):.3f}')
        print(f'  Recall@3:                   {sm.get(\"recall_at_3\", 0):.3f}')
except Exception as e: print(f'  (riskine eval error: {e})')
"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
kill $OLLAMA_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo "Job complete: $(date)"
echo "  Eval time: ${EVAL_TIME}s"
echo "  Results:   data/results/riskine_eval_${SUFFIX}.json"
echo "  Fetch:     bash scripts/sync_to_cluster.sh xwa2284 --fetch"
echo "============================================================"
