#!/usr/bin/env bash
# =============================================================================
# sync_to_cluster.sh — Sync code to / fetch results from Turing scratch area
#
# Usage:
#   bash scripts/sync_to_cluster.sh <netid>           # push code TO cluster
#   bash scripts/sync_to_cluster.sh <netid> --fetch   # pull results FROM cluster
#
# Cluster policy reminders:
#   - Use /local/scratch/<netid> — NOT /home/<netid>
#   - Delete scratch data after fetching results (it is not backed up)
#   - Use -J flag (this script does) — never stage through /home
# =============================================================================

NETID=${1:-$USER}
MODE=${2:-""}
REMOTE="$NETID@turinglogin.mathcs.emory.edu"
SCRATCH="/local/scratch/$NETID"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_JUMP="-J ${NETID}@lab0z.mathcs.emory.edu"

# ---------------------------------------------------------------------------
# --fetch: pull results back to local data/results/
# ---------------------------------------------------------------------------
if [ "$MODE" = "--fetch" ]; then
    echo "Fetching results $REMOTE:$SCRATCH/project/data/results/ → $PROJECT_ROOT/data/results/"
    echo ""
    rsync -avz --progress \
        -e "ssh $SSH_JUMP" \
        "$REMOTE:$SCRATCH/project/data/results/" \
        "$PROJECT_ROOT/data/results/"
    echo ""
    echo "Fetch complete."
    echo ""
    echo "IMPORTANT — cluster policy: delete scratch data now that you have results:"
    echo "  ssh $SSH_JUMP $REMOTE 'rm -rf $SCRATCH/project'"
    echo "  (optionally also: rm -rf $SCRATCH/models  # frees ~40 GB)"
    exit 0
fi

# ---------------------------------------------------------------------------
# Default: push code TO cluster
# ---------------------------------------------------------------------------
echo "Syncing $PROJECT_ROOT → $REMOTE:$SCRATCH/project"
echo ""

rsync -avz --progress \
    --exclude='.claude/' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='.env' \
    --exclude='.DS_Store' \
    --exclude='data/flood/*.pdf' \
    --exclude='data/auto/*.pdf' \
    --exclude='data/riskine/' \
    --include='data/' \
    --include='data/results/' \
    -e "ssh $SSH_JUMP" \
    "$PROJECT_ROOT/" \
    "$REMOTE:$SCRATCH/project/"

echo ""
echo "Sync complete. Now copy your .env:"
echo "  scp $SSH_JUMP .env $REMOTE:$SCRATCH/project/.env"
echo ""
echo "Then submit the job:"
echo "  ssh $SSH_JUMP $REMOTE 'sbatch $SCRATCH/project/scripts/slurm_zone2_zone3.sh'"
