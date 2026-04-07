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
# Data directory — override for cross-domain experiments:
#   sbatch --export=ALL,DATA_DIR=data/Emory_Spring2026,RESULTS_DIR=data/results_emory ...
DATA_DIR=${DATA_DIR:-data/flood}
RESULTS_DIR=${RESULTS_DIR:-data/results}

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
# Zone 1: Re-run ingestion (if DATA_DIR != default flood data)
# ---------------------------------------------------------------------------
if [ "$DATA_DIR" != "data/flood" ]; then
    echo "================================================================"
    echo "ZONE 1 — Re-running ingestion on ${DATA_DIR}"
    echo "  Started: $(date)"
    echo "================================================================"
    ZONE1_START=$(date +%s)
    python3 zone1/ingestion.py \
        --data-dir ${DATA_DIR} \
        --output ${DATA_DIR}/processed/zone1_chunks.json \
        --model $MODEL
    ZONE1_END=$(date +%s)
    echo "Zone 1 complete: $((ZONE1_END - ZONE1_START))s"
fi

# ---------------------------------------------------------------------------
# Zone 2: Domain-agnostic Open IE
# ---------------------------------------------------------------------------
echo "================================================================"
echo "ZONE 2 — Open IE extraction with $MODEL"
echo "  Started: $(date)"
echo "================================================================"

# Delete cached vocab to regenerate with new model
rm -f $PROJECT/$RESULTS_DIR/zone2_vocab.json

ZONE2_START=$(date +%s)
python3 zone2/pipeline.py --model $MODEL --passes $PASSES \
    --chunks ${DATA_DIR}/processed/zone1_chunks.json --results-dir $RESULTS_DIR
ZONE2_END=$(date +%s)
ZONE2_TIME=$((ZONE2_END - ZONE2_START))

echo ""
echo "Zone 2 complete: ${ZONE2_TIME}s ($(date))"
echo "  Summary: $(python3 -c "
import json
try:
    with open('${RESULTS_DIR}/zone2_run_summary.json') as f:
        d = json.load(f)
    print(f\"triples={d.get('total_triples', '?')}, entities={d.get('unique_entities', '?')}, relations={d.get('unique_relations', '?')}\")
except: print('(summary not found)')
")"
echo ""

# ---------------------------------------------------------------------------
# Zone 2 Evaluation — extraction quality metrics only
# ---------------------------------------------------------------------------
echo "================================================================"
echo "ZONE 2 EVALUATION — Extraction Quality (suffix=$SUFFIX)"
echo "  Started: $(date)"
echo "================================================================"

Z2_EVAL_START=$(date +%s)
python3 evaluation/extraction_quality.py --suffix $SUFFIX --model $MODEL --sample-size 100
Z2_EVAL_END=$(date +%s)
Z2_EVAL_TIME=$((Z2_EVAL_END - Z2_EVAL_START))

echo ""
echo "Zone 2 eval complete: ${Z2_EVAL_TIME}s ($(date))"

# Print Zone 2 metrics
python3 -c "
import json
try:
    with open(f'${RESULTS_DIR}/extraction_quality_${SUFFIX}.json') as f:
        d = json.load(f)
    tp = d.get('triple_precision', {})
    fr = d.get('fact_recall', {})
    sg = d.get('source_grounding', {})
    gs = d.get('graph_statistics', {})
    print(f'=== Zone 2 Extraction Quality (${SUFFIX}) ===')
    print(f'  Triple Precision:   {tp.get(\"precision\", 0):.1%} ({tp.get(\"correct\", 0)}/{tp.get(\"correct\", 0)+tp.get(\"incorrect\", 0)} correct)')
    print(f'  Fact Recall:        {fr.get(\"fact_recall\", 0):.1%} ({fr.get(\"found_facts\", 0)}/{fr.get(\"total_facts\", 0)} facts)')
    print(f'  Source Grounding:   {sg.get(\"grounding_rate\", 0):.1%} ({sg.get(\"supported\", 0)}/{sg.get(\"total_checked\", 0)} grounded)')
    print(f'  Graph:              {gs.get(\"node_count\", \"?\")} nodes, {gs.get(\"edge_count\", \"?\")} edges')
except Exception as e: print(f'  (extraction quality eval not found: {e})')
"
echo ""

# ---------------------------------------------------------------------------
# Zone 3: Leiden ontology induction
# ---------------------------------------------------------------------------
echo "================================================================"
echo "ZONE 3 — Leiden ontology induction with $MODEL"
echo "  Started: $(date)"
echo "================================================================"

ZONE3_START=$(date +%s)
python3 zone3/pipeline.py --model $MODEL --results-dir $RESULTS_DIR
ZONE3_END=$(date +%s)
ZONE3_TIME=$((ZONE3_END - ZONE3_START))

echo ""
echo "Zone 3 complete: ${ZONE3_TIME}s ($(date))"
echo "  Summary: $(python3 -c "
import json
try:
    with open('${RESULTS_DIR}/zone3_run_summary.json') as f:
        d = json.load(f)
    print(f\"clusters={d.get('num_clusters', '?')}, hierarchy={d.get('num_subclass_edges', '?')} edges\")
except: print('(summary not found)')
")"
echo ""

# ---------------------------------------------------------------------------
# Zone 3 Evaluation — Riskine ontology metrics (Zone 3 only)
# ---------------------------------------------------------------------------
echo "================================================================"
echo "ZONE 3 EVALUATION — Riskine Ontology (suffix=$SUFFIX)"
echo "  Started: $(date)"
echo "================================================================"

EVAL_START=$(date +%s)
python3 baseline/eval.py --suffix $SUFFIX --riskine --model $MODEL
EVAL_END=$(date +%s)
EVAL_TIME=$((EVAL_END - EVAL_START))

echo ""
echo "Zone 3 eval complete: ${EVAL_TIME}s ($(date))"
echo ""

# Print Zone 3 metrics
python3 -c "
import json, os
# Query accuracy + type consistency
try:
    with open(f'${RESULTS_DIR}/baseline_eval_results_${SUFFIX}.json') as f:
        d = json.load(f)
    bm = d.get('baseline_metrics', d)
    print(f'=== Zone 3 Query Eval (${SUFFIX}) ===')
    print(f'  Query accuracy:     {bm.get(\"query_accuracy\", bm.get(\"accuracy\", \"?\")):.1%}')
    print(f'  Type inconsistency: {bm.get(\"type_inconsistency_rate\", bm.get(\"type_inconsistency\", \"?\")):.1%}')
    print(f'  Duplication:        {bm.get(\"duplication_rate\", bm.get(\"duplication_ratio\", \"?\")):.1%}')
except Exception as e: print(f'  (query eval not found: {e})')

# Riskine ontology alignment
try:
    with open(f'${RESULTS_DIR}/riskine_eval_${SUFFIX}.json') as f:
        d = json.load(f)
    sm = d.get('standard_metrics', {})
    print(f'=== Zone 3 Riskine Eval (${SUFFIX}) ===')
    print(f'  Name F1:            {d.get(\"f1\", \"?\"):.3f}')
    print(f'  BERTScore F1:       {sm.get(\"bertscore_f1\", \"?\"):.3f}')
    print(f'  Graph F1:           {sm.get(\"graph_f1\", \"?\"):.3f}')
    print(f'  Wu-Palmer:          {sm.get(\"avg_wu_palmer\", \"?\"):.3f}')
    print(f'  Entity Assign F1:   {d.get(\"entity_assignment_f1\", \"?\"):.3f}')
    print(f'  Induced classes:    {d.get(\"induced_label_count\", \"?\")}')
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
echo "  Zone 2 pipeline:  ${ZONE2_TIME}s"
echo "  Zone 2 eval:      ${Z2_EVAL_TIME}s"
echo "  Zone 3 pipeline:  ${ZONE3_TIME}s"
echo "  Zone 3 eval:      ${EVAL_TIME}s"
echo "  Total:            $((ZONE2_TIME + Z2_EVAL_TIME + ZONE3_TIME + EVAL_TIME))s"
echo ""
echo "Results saved to: $PROJECT/$RESULTS_DIR/"
echo "  zone2_run_summary.json"
echo "  extraction_quality_${SUFFIX}.json       ← Zone 2 eval"
echo "  zone3_run_summary.json"
echo "  baseline_eval_results_${SUFFIX}.json    ← Zone 3 eval"
echo "  riskine_eval_${SUFFIX}.json             ← Zone 3 eval"
echo ""
echo "NEXT STEPS:"
echo "  1. Fetch results:  bash scripts/sync_to_cluster.sh $USER --fetch"
echo "  2. Compare:        python3 evaluation/compare_results.py"
echo "  3. Run SV-LOI:     sbatch --export=ALL,MODEL=$MODEL,SUFFIX=zone3_svloi scripts/slurm_sv_loi.sh"
echo "============================================================"
