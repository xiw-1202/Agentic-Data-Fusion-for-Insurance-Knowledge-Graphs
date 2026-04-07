#!/bin/bash
#SBATCH --job-name=eval_only
#SBATCH --output=/local/scratch/%u/logs/eval_%j.out
#SBATCH --error=/local/scratch/%u/logs/eval_%j.err
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail

SCRATCH=/local/scratch/$USER
PROJECT=$SCRATCH/project

MODEL=${MODEL:-qwen2.5:72b}
SUFFIX=${SUFFIX:-eval_rerun}
DATA_DIR=${DATA_DIR:-data/flood}
RESULTS_DIR=${RESULTS_DIR:-data/results}

export PATH=$SCRATCH/bin:$PATH
export OLLAMA_MODELS=$SCRATCH/models
export OLLAMA_CONTEXT_LENGTH=8192
export CUDA_VISIBLE_DEVICES=0,1
export OLLAMA_NUM_GPU=999
export LD_LIBRARY_PATH=$SCRATCH/lib/ollama:${LD_LIBRARY_PATH:-}
export HF_HOME=$SCRATCH/.cache/huggingface

source "$SCRATCH/venv/bin/activate"
cd "$PROJECT"
source .env 2>/dev/null || true

mkdir -p "$SCRATCH/logs"
pip install -r requirements.txt --quiet

echo "================================================================"
echo "EVAL ONLY — $MODEL"
echo "  DATA_DIR: $DATA_DIR"
echo "  RESULTS_DIR: $RESULTS_DIR"
echo "  SUFFIX: $SUFFIX"
echo "  Started: $(date)"
echo "================================================================"

# Start Ollama
$SCRATCH/bin/ollama serve &
OLLAMA_PID=$!
sleep 5

JUDGE_MODEL=${JUDGE_MODEL:-$MODEL}

# Warm up models
echo "Loading models into GPU..."
$SCRATCH/bin/ollama run $MODEL "hello" > /dev/null 2>&1
if [ "$JUDGE_MODEL" != "$MODEL" ]; then
    $SCRATCH/bin/ollama run $JUDGE_MODEL "hello" > /dev/null 2>&1
fi
echo "Models ready."

# Extraction quality eval
echo ""
echo "=== Extraction Quality Eval ==="
echo "  Extraction model: $MODEL"
echo "  Judge model: $JUDGE_MODEL"
JUDGE_FLAG=""
if [ "$JUDGE_MODEL" != "$MODEL" ]; then
    JUDGE_FLAG="--judge-model $JUDGE_MODEL"
fi
python3 evaluation/extraction_quality.py \
    --suffix $SUFFIX --model $MODEL --sample-size 100 \
    --chunks ${DATA_DIR}/processed/zone1_chunks.json \
    --results-dir $RESULTS_DIR $JUDGE_FLAG

echo ""
echo "=== Done: $(date) ==="

kill $OLLAMA_PID 2>/dev/null || true
