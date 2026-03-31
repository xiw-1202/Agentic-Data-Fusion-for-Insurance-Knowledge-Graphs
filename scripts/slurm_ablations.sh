#!/bin/bash
#SBATCH --job-name=svloi_ablations
#SBATCH --output=/local/scratch/xwa2284/logs/ablations_%j.log
#SBATCH --error=/local/scratch/xwa2284/logs/ablations_%j.err
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# SV-LOI ablation study — all 5 variants + full Riskine eval
# Submit: sbatch scripts/slurm_ablations.sh

set -e
cd /local/scratch/xwa2284/CS584_AI_Capstone

MODEL="qwen2.5:72b"
echo "Starting ablation study at $(date)"
echo "Model: $MODEL"

# Ensure Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama..."
    nohup ollama serve &
    sleep 10
fi

# Run all ablation variants
bash scripts/run_ablations.sh "$MODEL"

# Run evaluation with ALL 26 Riskine classes for each variant
echo ""
echo "================================================================"
echo "Running evaluation with full 26-class Riskine..."
echo "================================================================"

for suffix in zone3_svloi_full zone3_svloi_noverify zone3_svloi_noarb zone3_svloi_nocons zone3_svloi_llmonly; do
    echo ""
    echo "=== Evaluating: $suffix ==="
    python3 baseline/eval.py --suffix "$suffix" --riskine --all-classes --model "$MODEL"
done

# Also run existing methods for comparison
echo ""
echo "=== Evaluating Leiden baseline ==="
python3 zone3/pipeline.py --model "$MODEL" --suffix zone3_leiden_rerun
python3 baseline/eval.py --suffix zone3_leiden_rerun --riskine --all-classes --model "$MODEL"

echo ""
echo "=== Evaluating RSI-LCR ==="
python3 zone3/rsi_lcr.py --model "$MODEL" --suffix zone3_rsi_rerun
python3 baseline/eval.py --suffix zone3_rsi_rerun --riskine --all-classes --model "$MODEL"

echo ""
echo "Ablation study complete at $(date)"
echo "Results in data/results/"
