#!/usr/bin/env bash
# =============================================================================
# run_on_cluster.sh — Push code + submit SLURM job to Emory Turing cluster
#
# Usage:
#   bash scripts/run_on_cluster.sh                                    # full pipeline, flood
#   bash scripts/run_on_cluster.sh --data Emory_Spring2026            # full pipeline, emory
#   bash scripts/run_on_cluster.sh --mode zone2 --data Emory_Spring2026
#   bash scripts/run_on_cluster.sh --mode zone3 --data Emory_Spring2026
#   bash scripts/run_on_cluster.sh --mode eval --judge gemma4:31b
#   bash scripts/run_on_cluster.sh --model gemma4:31b --data Emory_Spring2026
#   bash scripts/run_on_cluster.sh --force-zone1 --data Emory_Spring2026  # re-run zone1 (cache busted)
#   bash scripts/run_on_cluster.sh --fetch                            # fetch results
#   bash scripts/run_on_cluster.sh --setup                            # first-time setup
# =============================================================================

set -euo pipefail

NETID=${NETID:-xwa2284}
REMOTE="$NETID@turinglogin.mathcs.emory.edu"
SCRATCH="/local/scratch/$NETID"
SSH_JUMP="-J ${NETID}@lab0z.mathcs.emory.edu"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Parse arguments ---
MODE="full"
MODEL="qwen2.5:72b"
DATA=""
JUDGE=""
ACTION="submit"
FORCE_ZONE1=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fetch)        ACTION="fetch"; shift ;;
        --setup)        ACTION="setup"; shift ;;
        --mode)         MODE="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --data)         DATA="$2"; shift 2 ;;
        --judge)        JUDGE="$2"; shift 2 ;;
        --force-zone1)  FORCE_ZONE1=1; shift ;;
        *)              MODEL="$1"; shift ;;  # backward compat: positional = model
    esac
done

# Map --data shorthand to DATA_DIR
case "$DATA" in
    ""|flood)           DATA_DIR="data/flood" ;;
    emory|Emory*)       DATA_DIR="data/Emory_Spring2026" ;;
    *)                  DATA_DIR="data/$DATA" ;;
esac

# Map --mode to script
case "$MODE" in
    full)       SCRIPT="scripts/slurm_pipeline.sh" ;;
    zone2)      SCRIPT="scripts/slurm_zone2.sh" ;;
    zone3)      SCRIPT="scripts/slurm_zone3.sh" ;;
    eval)       SCRIPT="scripts/slurm_eval.sh" ;;
    *)          echo "ERROR: Unknown mode '$MODE'. Use: full, zone2, zone3, eval"; exit 1 ;;
esac

# --- Fetch mode ---
if [ "$ACTION" = "fetch" ]; then
    echo "=== Fetching results from Turing ==="
    bash "$PROJECT_ROOT/scripts/sync_to_cluster.sh" "$NETID" --fetch
    exit 0
fi

# --- Push code ---
echo "=== Step 1: Syncing code to Turing ==="
bash "$PROJECT_ROOT/scripts/sync_to_cluster.sh" "$NETID"

# --- Copy .env ---
echo ""
echo "=== Step 2: Copying .env ==="
scp $SSH_JUMP "$PROJECT_ROOT/.env" "$REMOTE:$SCRATCH/project/.env" 2>/dev/null || {
    echo "WARNING: .env copy failed."
}

# --- First-time setup ---
if [ "$ACTION" = "setup" ]; then
    echo ""
    echo "=== Running first-time cluster setup ==="
    ssh $SSH_JUMP "$REMOTE" "cd $SCRATCH/project && bash scripts/cluster_setup_v2.sh"
    echo "Setup complete."
    exit 0
fi

# --- Build export vars ---
EXPORT_VARS="ALL,MODEL=$MODEL,DATA_DIR=$DATA_DIR"
[ -n "$JUDGE" ] && EXPORT_VARS="$EXPORT_VARS,JUDGE_MODEL=$JUDGE"
[ -n "$FORCE_ZONE1" ] && EXPORT_VARS="$EXPORT_VARS,SKIP_ZONE1=0"

# --- Submit ---
echo ""
echo "=== Step 3: Submitting SLURM job ==="
echo "  Script: $SCRIPT"
echo "  Mode:   $MODE"
echo "  Model:  $MODEL"
echo "  Data:   $DATA_DIR"
[ -n "$JUDGE" ] && echo "  Judge:  $JUDGE"
echo ""

ssh $SSH_JUMP "$REMOTE" "cd $SCRATCH/project && sbatch --export=$EXPORT_VARS $SCRIPT"

echo ""
echo "=== Job submitted! ==="
echo ""
echo "Monitor:  ssh $SSH_JUMP $REMOTE 'squeue -u $NETID'"
echo "Fetch:    bash scripts/run_on_cluster.sh --fetch"
