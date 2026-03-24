#!/usr/bin/env bash
# =============================================================================
# slurm_pipeline_v2.sh — Slurm job: Full pipeline with qwen2.5:72b on 2× GPUs
#
# Submit:  sbatch scripts/slurm_pipeline_v2.sh
# Monitor: squeue -u $USER
# Logs:    tail -f /local/scratch/$USER/logs/<jobid>.out
#
# MODEL: qwen2.5:72b (Q4_K_M, ~43 GB)
#   Why qwen2.5:72b over llama3.1:70b:
#     1. Superior structured JSON output — critical for triple extraction
#     2. Better instruction following — extracts more triples per chunk
#     3. Stronger multilingual + domain vocabulary understanding
#     4. llama3.1:70b produced compound names (F-07) → lower F1
#     5. llama3.3:70b is backup if qwen underperforms
#
# RESOURCES:
#   2 GPUs with ≥48 GB each — Ollama auto-splits across both GPUs
#   96 GB RAM               — embeddings + Neo4j client + Python overhead
#   6 hours                 — Zone 2 (~120 min) + Zone 3 (~30 min) + eval (~30 min)
#                             + buffer for iterative extraction
#
# Override model: sbatch --export=ALL,MODEL=llama3.3:70b scripts/slurm_pipeline_v2.sh
# =============================================================================

#SBATCH --job-name=cs584_v2_qwen72b
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

set -euo pipefail

SCRATCH=/local/scratch/$USER
PROJECT=$SCRATCH/project

# Default model — override via: sbatch --export=ALL,MODEL=llama3.3:70b ...
MODEL=${MODEL:-qwen2.5:72b}
SUFFIX=${SUFFIX:-zone3_v2_qwen72b}
PASSES=${PASSES:-1}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export PATH=$SCRATCH/bin:$PATH
export XDG_CACHE_HOME=$SCRATCH/.cache
export HF_HOME=$SCRATCH/.cache/huggingface
export OLLAMA_MODELS=$SCRATCH/models
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
export LD_LIBRARY_PATH=$SCRATCH/lib/ollama:${LD_LIBRARY_PATH:-}

# Ollama context window: 4096 tokens is sufficient for our prompts.
# Default (auto) = 131072 → massive KV cache → fewer layers on GPU → slower.
# At 4096: KV cache ~1.3 GB → model gets full GPU memory → all layers on GPU.
export OLLAMA_CONTEXT_LENGTH=4096

# Force Ollama to use both GPUs (it auto-detects, but be explicit)
export CUDA_VISIBLE_DEVICES=0,1

# Tell Ollama to load ALL layers onto GPU (999 = "as many as possible")
# With 2×48GB = 96GB, qwen2.5:72b (43GB) fits entirely in GPU memory
export OLLAMA_NUM_GPU=999

source $SCRATCH/venv/bin/activate
cd $PROJECT

mkdir -p $SCRATCH/logs

# Install/verify Python dependencies
pip install -r $PROJECT/requirements.txt --quiet

echo "============================================================"
echo "CS584 Full Pipeline v2"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Host:       $(hostname)"
echo "  Model:      $MODEL"
echo "  Suffix:     $SUFFIX"
echo "  Passes:     $PASSES"
echo "  GPUs:       $SLURM_GPUS_ON_NODE"
echo "  Start:      $(date)"
echo "============================================================"
echo ""

# GPU info
echo "=== GPU Configuration ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | paste -sd+ | bc)
echo "Total VRAM: ${TOTAL_VRAM} MB"
if [ "$TOTAL_VRAM" -lt 80000 ]; then
    echo "WARNING: Total VRAM is only ${TOTAL_VRAM} MB. qwen2.5:72b needs ~43 GB."
    echo "  Falling back to single-GPU mode (model fits on 1× 48GB GPU)."
fi
echo ""

# ---------------------------------------------------------------------------
# Start Ollama server
# ---------------------------------------------------------------------------
echo "[setup] Starting Ollama server (multi-GPU)..."
ollama serve &>$SCRATCH/logs/ollama_$SLURM_JOB_ID.log &
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
    echo "ERROR: Ollama did not become ready within 30s. Check ollama log:" >&2
    tail -20 "$SCRATCH/logs/ollama_$SLURM_JOB_ID.log" 2>/dev/null || true
    exit 1
fi

# Warm up — load model into GPU VRAM
echo "[setup] Loading $MODEL into GPU..."
timeout 300 ollama run "$MODEL" "Hello, respond with one word." --nowordwrap 2>&1 | head -5 || true
echo "[setup] Model loaded — check Ollama log for GPU layer allocation:"
grep -i "layer" "$SCRATCH/logs/ollama_$SLURM_JOB_ID.log" 2>/dev/null | tail -3 || true
echo ""

# ---------------------------------------------------------------------------
# Zone 2: Domain-agnostic Open IE
# ---------------------------------------------------------------------------
echo "================================================================"
echo "ZONE 2 — Open IE extraction with $MODEL"
echo "  Started: $(date)"
echo "================================================================"

# Delete cached vocab to regenerate with new model
rm -f $PROJECT/data/results/zone2_vocab.json

ZONE2_START=$(date +%s)
python3 zone2/pipeline.py --model $MODEL --passes $PASSES
ZONE2_END=$(date +%s)
ZONE2_TIME=$((ZONE2_END - ZONE2_START))

echo ""
echo "Zone 2 complete: ${ZONE2_TIME}s ($(date))"
echo "  Summary: $(python3 -c "
import json
try:
    with open('data/results/zone2_run_summary.json') as f:
        d = json.load(f)
    print(f\"triples={d.get('total_triples', '?')}, entities={d.get('unique_entities', '?')}, relations={d.get('unique_relations', '?')}\")
except: print('(summary not found)')
")"
echo ""

# ---------------------------------------------------------------------------
# Zone 3: Leiden ontology induction
# ---------------------------------------------------------------------------
echo "================================================================"
echo "ZONE 3 — Leiden ontology induction with $MODEL"
echo "  Started: $(date)"
echo "================================================================"

ZONE3_START=$(date +%s)
python3 zone3/pipeline.py --model $MODEL
ZONE3_END=$(date +%s)
ZONE3_TIME=$((ZONE3_END - ZONE3_START))

echo ""
echo "Zone 3 complete: ${ZONE3_TIME}s ($(date))"
echo "  Summary: $(python3 -c "
import json
try:
    with open('data/results/zone3_run_summary.json') as f:
        d = json.load(f)
    print(f\"clusters={d.get('num_clusters', '?')}, hierarchy={d.get('num_subclass_edges', '?')} edges\")
except: print('(summary not found)')
")"
echo ""

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
echo "================================================================"
echo "EVALUATION — suffix=$SUFFIX"
echo "  Started: $(date)"
echo "================================================================"

EVAL_START=$(date +%s)
python3 baseline/eval.py --suffix $SUFFIX --riskine --model $MODEL
EVAL_END=$(date +%s)
EVAL_TIME=$((EVAL_END - EVAL_START))

echo ""
echo "Evaluation complete: ${EVAL_TIME}s ($(date))"
echo ""

# Print key metrics
python3 -c "
import json, os
# Baseline eval
try:
    with open(f'data/results/baseline_eval_results_${SUFFIX}.json') as f:
        d = json.load(f)
    print(f'=== Baseline Eval (${SUFFIX}) ===')
    print(f'  Query accuracy:     {d.get(\"accuracy\", \"?\"):.1%}')
    print(f'  Type inconsistency: {d.get(\"type_inconsistency\", \"?\"):.1%}')
    print(f'  Duplication:        {d.get(\"duplication_ratio\", \"?\"):.1%}')
except Exception as e: print(f'  (baseline eval not found: {e})')

# Riskine eval
try:
    with open(f'data/results/riskine_eval_${SUFFIX}.json') as f:
        d = json.load(f)
    mb = d.get('member_based', {})
    print(f'=== Riskine Eval (${SUFFIX}) ===')
    print(f'  Member F1:          {mb.get(\"f1\", \"?\"):.3f}')
    print(f'  Member Precision:   {mb.get(\"precision\", \"?\"):.3f}')
    print(f'  Member Recall:      {mb.get(\"recall\", \"?\"):.3f}')
    print(f'  Classes covered:    {mb.get(\"riskine_covered\", \"?\")} / 10')
    print(f'  Induced classes:    {mb.get(\"induced_count\", \"?\")}')
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
echo "  Zone 2:     ${ZONE2_TIME}s"
echo "  Zone 3:     ${ZONE3_TIME}s"
echo "  Evaluation: ${EVAL_TIME}s"
echo "  Total:      $((ZONE2_TIME + ZONE3_TIME + EVAL_TIME))s"
echo ""
echo "Results saved to: $PROJECT/data/results/"
echo "  zone2_run_summary.json"
echo "  zone3_run_summary.json"
echo "  baseline_eval_results_${SUFFIX}.json"
echo "  riskine_eval_${SUFFIX}.json"
echo ""
echo "NEXT STEPS:"
echo "  1. Fetch results:  bash scripts/sync_to_cluster.sh $USER --fetch"
echo "  2. Compare:        python3 evaluation/compare_results.py"
echo "  3. If qwen underperforms, try llama3.3:"
echo "     sbatch --export=ALL,MODEL=llama3.3:70b,SUFFIX=zone3_v2_llama33 scripts/slurm_pipeline_v2.sh"
echo "============================================================"
