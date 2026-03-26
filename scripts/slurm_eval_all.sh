#!/usr/bin/env bash
# =============================================================================
# slurm_eval_all.sh — Re-run all Zone 3 methods + evaluation with updated metrics
#
# Runs each method sequentially (each overwrites the Neo4j ontology layer),
# evaluates immediately after, then moves to the next method.
#
# Order: Leiden (baseline) → RSI-LCR → SV-LOI
#
# Submit:  sbatch scripts/slurm_eval_all.sh
# Monitor: tail -f /local/scratch/$USER/logs/<jobid>.out
# =============================================================================

#SBATCH --job-name=cs584_eval_all
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

set -euo pipefail

SCRATCH="/local/scratch/${USER:?USER is not set}"
PROJECT="$SCRATCH/project"

MODEL="${MODEL:-qwen2.5:72b}"

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
echo "CS584 — Full Evaluation: All Zone 3 Methods"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Host:       $(hostname)"
echo "  Model:      $MODEL"
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
    exit 1
fi

echo "[setup] Loading $MODEL into GPU..."
timeout 300 ollama run "$MODEL" "Hello, respond with one word." --nowordwrap 2>&1 | head -5 || true
echo ""

# ---------------------------------------------------------------------------
# METHOD 1: Leiden (Variant A — baseline)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "METHOD 1/3: Leiden (Variant A — baseline)"
echo "  Started: $(date)"
echo "================================================================"

M1_START=$(date +%s)
python3 zone3/pipeline.py --model "$MODEL" --suffix zone3_leiden || {
    echo "WARNING: Leiden pipeline failed — continuing with eval"
}
python3 baseline/eval.py --suffix zone3_leiden --riskine --model "$MODEL" || {
    echo "WARNING: Leiden eval failed"
}
M1_END=$(date +%s)
M1_TIME=$((M1_END - M1_START))
echo "Leiden complete: ${M1_TIME}s"

# ---------------------------------------------------------------------------
# METHOD 2: RSI-LCR (Variant B)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "METHOD 2/3: RSI-LCR (Variant B)"
echo "  Started: $(date)"
echo "================================================================"

M2_START=$(date +%s)
python3 zone3/rsi_lcr.py --model "$MODEL" --suffix zone3_rsi || {
    echo "WARNING: RSI-LCR pipeline failed — continuing with eval"
}
python3 baseline/eval.py --suffix zone3_rsi --riskine --model "$MODEL" || {
    echo "WARNING: RSI-LCR eval failed"
}
M2_END=$(date +%s)
M2_TIME=$((M2_END - M2_START))
echo "RSI-LCR complete: ${M2_TIME}s"

# ---------------------------------------------------------------------------
# METHOD 3: SV-LOI (Variant D — our contribution)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "METHOD 3/3: SV-LOI (Variant D — our contribution)"
echo "  Started: $(date)"
echo "================================================================"

M3_START=$(date +%s)
python3 zone3/sv_loi.py --model "$MODEL" --suffix zone3_svloi || {
    echo "WARNING: SV-LOI pipeline failed — continuing with eval"
}
python3 baseline/eval.py --suffix zone3_svloi --riskine --model "$MODEL" || {
    echo "WARNING: SV-LOI eval failed"
}
M3_END=$(date +%s)
M3_TIME=$((M3_END - M3_START))
echo "SV-LOI complete: ${M3_TIME}s"

# ---------------------------------------------------------------------------
# Extraction Quality Evaluation (runs on the final graph state = SV-LOI)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "EXTRACTION QUALITY EVALUATION"
echo "  Started: $(date)"
echo "================================================================"

python3 evaluation/extraction_quality.py --suffix zone3_svloi --model "$MODEL" --sample-size 50 || {
    echo "WARNING: Extraction quality eval failed"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "ALL METHODS COMPLETE"
echo "================================================================"
echo ""

# Print comparison table
python3 -c "
import json, os

methods = [
    ('Leiden (A)',  'zone3_leiden'),
    ('RSI-LCR (B)', 'zone3_rsi'),
    ('SV-LOI (D)',  'zone3_svloi'),
]

print(f'{'Method':<16} {'Name F1':>8} {'EA F1':>8} {'EA F1p':>8} {'BERT F1':>8} {'Graph F1':>9} {'WuPalm':>8} {'QAcc':>6}')
print('-' * 85)

for label, suffix in methods:
    rpath = f'data/results/riskine_eval_{suffix}.json'
    bpath = f'data/results/baseline_eval_results_{suffix}.json'

    nf1 = ef1 = efp = bf1 = gf1 = wp = qa = '—'

    if os.path.exists(rpath):
        with open(rpath) as f:
            r = json.load(f)
        nf1 = f'{r.get(\"f1\", 0):.3f}'
        ef1 = f'{r.get(\"entity_assignment_f1\", 0):.3f}'
        efp = f'{r.get(\"entity_assignment_f1_present\", 0):.3f}'
        sm = r.get('standard_metrics', {})
        bf1 = f'{sm.get(\"bertscore_f1\", 0):.3f}'
        gf1 = f'{sm.get(\"graph_f1\", 0):.3f}'
        wp  = f'{sm.get(\"avg_wu_palmer\", 0):.3f}'

    if os.path.exists(bpath):
        with open(bpath) as f:
            b = json.load(f)
        bm = b.get('baseline_metrics', {})
        qa = f'{bm.get(\"query_accuracy\", 0):.0%}'

    print(f'{label:<16} {nf1:>8} {ef1:>8} {efp:>8} {bf1:>8} {gf1:>9} {wp:>8} {qa:>6}')

print()
print('EA F1  = Entity Assignment F1 (full 10-class)')
print('EA F1p = Entity Assignment F1 (present-class only)')
print('BERT   = BERTScore F1')
print('Graph  = Graph F1 (structure-aware)')
print('WuPalm = Wu-Palmer similarity')
print('QAcc   = Query accuracy (20 Cypher tasks)')
"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
kill $OLLAMA_PID 2>/dev/null || true

TOTAL=$((M1_TIME + M2_TIME + M3_TIME))
echo ""
echo "============================================================"
echo "Job complete: $(date)"
echo ""
echo "Timing:"
echo "  Leiden:    ${M1_TIME}s"
echo "  RSI-LCR:  ${M2_TIME}s"
echo "  SV-LOI:   ${M3_TIME}s"
echo "  Total:     ${TOTAL}s"
echo ""
echo "Result files:"
echo "  riskine_eval_zone3_leiden.json"
echo "  riskine_eval_zone3_rsi.json"
echo "  riskine_eval_zone3_svloi.json"
echo "  baseline_eval_results_zone3_leiden.json"
echo "  baseline_eval_results_zone3_rsi.json"
echo "  baseline_eval_results_zone3_svloi.json"
echo ""
echo "Fetch:  bash scripts/sync_to_cluster.sh xwa2284 --fetch"
echo "============================================================"
