#!/usr/bin/env bash
# =============================================================================
# slurm_zone2_zone3.sh — Slurm job: Zone 2 + Zone 3 with llama3.1:70b
#
# Submit:  sbatch scripts/slurm_zone2_zone3.sh
# Monitor: squeue -u $USER
# Logs:    tail -f /local/scratch/$USER/logs/<jobid>.out
#
# Resources:
#   1 GPU with ≥48 GB VRAM — llama3.1:70b 4-bit requires ~40 GB
#   64 GB RAM              — for embeddings + Neo4j client + Python overhead
#   4 hours                — Zone 2 (~90 min) + Zone 3 (~20 min) + eval (~30 min)
#
# AFTER JOB COMPLETES — per cluster policy:
#   1. Fetch results:  bash scripts/sync_to_cluster.sh <netid> --fetch
#   2. Delete scratch: ssh -J <netid>@lab0z... turinglogin 'rm -rf /local/scratch/<netid>/project'
# =============================================================================

#SBATCH --job-name=cs584_z2z3_70b
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

set -e

SCRATCH=/local/scratch/$USER
PROJECT=$SCRATCH/project
MODEL=llama3.1:70b

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export PATH=$SCRATCH/bin:$PATH          # ollama binary installed here (no sudo)
export XDG_CACHE_HOME=$SCRATCH/.cache
export HF_HOME=$SCRATCH/.cache/huggingface
export OLLAMA_MODELS=$SCRATCH/models
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
# Ollama ships CUDA libs in lib/ollama/ alongside the binary — expose them at runtime
export LD_LIBRARY_PATH=$SCRATCH/lib/ollama:${LD_LIBRARY_PATH:-}
# Limit context window to 4096 tokens — our chunks are ~750 tokens + few-shot ~1000 tokens.
# Default (auto) = 131072 → 40 GB KV cache → only 29/81 layers on GPU.
# At 4096: KV cache ~1.3 GB → 46 GB free → all 81 layers on GPU → full speed.
export OLLAMA_CONTEXT_LENGTH=4096

source $SCRATCH/venv/bin/activate
cd $PROJECT

mkdir -p $SCRATCH/logs

# Install/verify Python dependencies (fast no-op if already installed)
echo "[setup] Installing Python dependencies..."
pip install -r $PROJECT/requirements.txt --quiet
echo "[setup] Dependencies ready"

echo "============================================================"
echo "CS584 Zone 2 + Zone 3 — GPU Run"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Host:    $(hostname)"
echo "  Model:   $MODEL"
echo "  Start:   $(date)"
echo "============================================================"
echo ""

# GPU info — verify we have enough VRAM for llama3.1:70b 4-bit (~40 GB)
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -lt 40000 ]; then
    echo "ERROR: GPU has only ${GPU_MEM} MB VRAM — llama3.1:70b requires ~40000 MB. Exiting."
    exit 1
fi
echo "[setup] GPU VRAM: ${GPU_MEM} MB — sufficient for llama3.1:70b"

# ---------------------------------------------------------------------------
# Start Ollama server (uses GPU automatically)
# ---------------------------------------------------------------------------
echo ""
echo "[setup] Starting Ollama server..."
# Log to scratch (not /tmp which may not be on compute node) for easier inspection
ollama serve &>$SCRATCH/logs/ollama_$SLURM_JOB_ID.log &
OLLAMA_PID=$!

# Wait for Ollama to become ready (up to 30s)
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "[setup] Ollama ready (${i}s)"
        break
    fi
    sleep 1
done

# Warm up model (load weights into GPU VRAM)
# Output visible so GPU vs CPU can be confirmed; timeout=180s (CPU fallback would hang for hours)
echo "[setup] Loading $MODEL into GPU..."
timeout 180 ollama run $MODEL "Hello" --nowordwrap 2>&1 | head -5 || true
echo "[setup] Model loaded (check above — should show GPU layers, not CPU)"

# ---------------------------------------------------------------------------
# Zone 2: Domain-agnostic Open IE
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "ZONE 2 — Open IE extraction with $MODEL"
echo "================================================================"
# Delete cached vocab to regenerate with 70B (8B vocab may differ)
rm -f $PROJECT/data/results/zone2_vocab.json
python3 zone2/pipeline.py --model $MODEL
echo "Zone 2 complete at $(date)"

# ---------------------------------------------------------------------------
# Zone 3: Leiden ontology induction
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "ZONE 3 — Leiden ontology induction with $MODEL"
echo "================================================================"
python3 zone3/pipeline.py --model $MODEL
echo "Zone 3 complete at $(date)"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "EVALUATION — suffix=zone3_multileiden_70b"
echo "================================================================"
python3 baseline/eval.py --suffix zone3_multileiden_70b --riskine --model $MODEL
echo "Evaluation complete at $(date)"

# ---------------------------------------------------------------------------
# Cleanup — kill Ollama server
# ---------------------------------------------------------------------------
kill $OLLAMA_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo "Job complete: $(date)"
echo "Results saved to: $PROJECT/data/results/"
echo "  zone2_run_summary.json"
echo "  zone3_run_summary.json"
echo "  baseline_eval_results_zone3_multileiden_70b.json"
echo "  riskine_eval_zone3_multileiden_70b.json"
echo ""
echo "NEXT STEPS (cluster policy — do this promptly):"
echo "  1. Fetch results to local machine:"
echo "     bash scripts/sync_to_cluster.sh $USER --fetch"
echo "  2. Delete project from scratch (required per policy):"
echo "     rm -rf $PROJECT"
echo "     rm -rf $SCRATCH/models   # optional: frees ~40 GB model weights"
echo "     rm -rf $SCRATCH/venv     # optional: frees Python env"
echo "============================================================"
