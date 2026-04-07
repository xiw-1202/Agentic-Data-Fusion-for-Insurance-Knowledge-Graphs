#!/usr/bin/env bash
# =============================================================================
# _env.sh — Shared environment for all SLURM jobs
#
# Sourced by: slurm_pipeline.sh, slurm_zone2.sh, slurm_zone3.sh, slurm_eval.sh
# Sets up: Ollama, Python venv, GPU config, data paths
# =============================================================================

set -euo pipefail

SCRATCH=/local/scratch/$USER
PROJECT=$SCRATCH/project

# --- Configurable via --export=ALL,VAR=value ---
MODEL=${MODEL:-qwen2.5:72b}
PASSES=${PASSES:-1}
DATA_DIR=${DATA_DIR:-data/flood}
JUDGE_MODEL=${JUDGE_MODEL:-$MODEL}

# --- Auto-derive RESULTS_DIR and SUFFIX from DATA_DIR ---
DATASET=$(basename "$DATA_DIR")
case $DATASET in
    flood)              RESULTS_DIR=${RESULTS_DIR:-data/results/flood} ;;
    Emory_Spring2026)   RESULTS_DIR=${RESULTS_DIR:-data/results/emory} ;;
    *)                  RESULTS_DIR=${RESULTS_DIR:-data/results/$DATASET} ;;
esac

MODEL_SHORT=$(echo "$MODEL" | tr ':.' '_')
SUFFIX=${SUFFIX:-${DATASET}_${MODEL_SHORT}}

CHUNKS_FILE=${DATA_DIR}/processed/zone1_chunks.json

# --- Environment ---
export PATH=$SCRATCH/bin:$PATH
export XDG_CACHE_HOME=$SCRATCH/.cache
export HF_HOME=$SCRATCH/.cache/huggingface
export OLLAMA_MODELS=$SCRATCH/models
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
export LD_LIBRARY_PATH=$SCRATCH/lib/ollama:${LD_LIBRARY_PATH:-}

# Suppress Ollama HTTP logs
export OLLAMA_DEBUG=false
export GIN_MODE=release

# Context window: 8192 for system prompt + few-shot + chunk + output
export OLLAMA_CONTEXT_LENGTH=8192
export CUDA_VISIBLE_DEVICES=0,1
export OLLAMA_NUM_GPU=999

# --- Activate environment ---
source "$SCRATCH/venv/bin/activate"
cd "$PROJECT"
mkdir -p "$SCRATCH/logs" "$RESULTS_DIR" "${RESULTS_DIR}/eval"
pip install -r "$PROJECT/requirements.txt" --quiet

# --- Print config ---
echo "================================================================"
echo "  MODEL:       $MODEL"
echo "  DATA_DIR:    $DATA_DIR"
echo "  RESULTS_DIR: $RESULTS_DIR"
echo "  SUFFIX:      $SUFFIX"
echo "  CHUNKS:      $CHUNKS_FILE"
echo "  JUDGE_MODEL: $JUDGE_MODEL"
echo "  Started:     $(date)"
echo "================================================================"

# --- Start Ollama ---
_start_ollama() {
    echo "[setup] Starting Ollama server..."
    $SCRATCH/bin/ollama serve &
    OLLAMA_PID=$!
    sleep 5
    echo "[setup] Loading $MODEL into GPU..."
    $SCRATCH/bin/ollama run "$MODEL" "hello" > /dev/null 2>&1
    if [ "$JUDGE_MODEL" != "$MODEL" ]; then
        echo "[setup] Loading judge model $JUDGE_MODEL..."
        $SCRATCH/bin/ollama run "$JUDGE_MODEL" "hello" > /dev/null 2>&1
    fi
    echo "[setup] Models ready."
}

_stop_ollama() {
    kill $OLLAMA_PID 2>/dev/null || true
}

# --- Source .env for Neo4j credentials ---
source .env 2>/dev/null || true
