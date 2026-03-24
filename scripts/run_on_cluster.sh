#!/usr/bin/env bash
# =============================================================================
# run_on_cluster.sh — Quick-reference script: push code + submit job in one go
#
# Usage:
#   bash scripts/run_on_cluster.sh                          # qwen2.5:72b (default)
#   bash scripts/run_on_cluster.sh llama3.3:70b             # alternative model
#   bash scripts/run_on_cluster.sh qwen2.5:72b --setup      # first-time setup + run
#   bash scripts/run_on_cluster.sh --fetch                   # fetch results back
#
# Pre-requisites:
#   - Emory VPN active (if remote)
#   - SSH key configured for lab0z.mathcs.emory.edu jump host
# =============================================================================

set -euo pipefail

NETID=${NETID:-xwa2284}
MODEL=${1:-qwen2.5:72b}
MODE=${2:-""}
REMOTE="$NETID@turinglogin.mathcs.emory.edu"
SCRATCH="/local/scratch/$NETID"
SSH_JUMP="-J ${NETID}@lab0z.mathcs.emory.edu"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Fetch mode ---
if [ "$MODEL" = "--fetch" ] || [ "$MODE" = "--fetch" ]; then
    echo "=== Fetching results from Turing ==="
    bash "$PROJECT_ROOT/scripts/sync_to_cluster.sh" $NETID --fetch
    exit 0
fi

# --- Push code ---
echo "=== Step 1: Syncing code to Turing ==="
bash "$PROJECT_ROOT/scripts/sync_to_cluster.sh" $NETID

# --- Copy .env ---
echo ""
echo "=== Step 2: Copying .env ==="
scp $SSH_JUMP "$PROJECT_ROOT/.env" "$REMOTE:$SCRATCH/project/.env" 2>/dev/null || {
    echo "WARNING: .env copy failed. Make sure it exists locally and SSH keys are set."
    echo "Manual: scp $SSH_JUMP .env $REMOTE:$SCRATCH/project/.env"
}

# --- First-time setup ---
if [ "$MODE" = "--setup" ]; then
    echo ""
    echo "=== Step 3: Running first-time cluster setup ==="
    echo "  This will install Ollama + pull models (~40 min)."
    echo "  Running in background via Slurm interactive session..."
    ssh $SSH_JUMP $REMOTE "cd $SCRATCH/project && bash scripts/cluster_setup_v2.sh"
    echo "Setup complete."
fi

# --- Determine suffix based on model ---
case "$MODEL" in
    qwen2.5:72b)     SUFFIX="zone3_v2_qwen72b" ;;
    llama3.3:70b)     SUFFIX="zone3_v2_llama33" ;;
    llama3.1:70b)     SUFFIX="zone3_v2_llama31_70b" ;;
    llama3.1:8b)      SUFFIX="zone3_v2_8b" ;;
    *)                SUFFIX="zone3_v2_custom" ;;
esac

# --- Submit Slurm job ---
echo ""
echo "=== Step 3: Submitting Slurm job ==="
echo "  Model:  $MODEL"
echo "  Suffix: $SUFFIX"
echo ""

PASSES=${PASSES:-1}
ssh $SSH_JUMP $REMOTE "cd $SCRATCH/project && sbatch --export=ALL,MODEL=$MODEL,SUFFIX=$SUFFIX,PASSES=$PASSES scripts/slurm_pipeline_v2.sh"

echo ""
echo "=== Job submitted! ==="
echo ""
echo "Monitor:"
echo "  ssh $SSH_JUMP $REMOTE 'squeue -u $NETID'"
echo ""
echo "Logs:"
echo "  ssh $SSH_JUMP $REMOTE 'tail -f $SCRATCH/logs/\$(ls -t $SCRATCH/logs/*.out | head -1)'"
echo ""
echo "Fetch results when done:"
echo "  bash scripts/run_on_cluster.sh --fetch"
