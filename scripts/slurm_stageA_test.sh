#!/bin/bash
# =============================================================================
# slurm_stageA_test.sh — Verify Stage A (relation_raw + skip flags) on cluster.
#
# Runs Zone 1 + Zone 2 against the default flood data with FULL ISOLATION:
#
#   • Zone 1 chunks       → data/flood-stageA-<ts>/processed/zone1_chunks.json
#   • Zone 2 results dir  → data/results/flood-stageA-<ts>/
#   • Neo4j               → URI overridden to localhost:0; insert_to_neo4j
#                            fails fast, existing AuraDB graph is untouched
#                            (zone2_run_summary.json still gets written)
#
# Existing artefacts that are NOT TOUCHED:
#   • data/flood/processed/zone1_chunks.json
#   • data/results/flood/zone2_run_summary.json
#   • data/results/flood/zone2_vocab.json
#   • AuraDB Neo4j graph
#
# Usage:
#   sbatch scripts/slurm_stageA_test.sh
#   sbatch --export=ALL,MODEL=qwen2.5:7b scripts/slurm_stageA_test.sh
#
# After completion, run the spot-check commands printed at the end of the log.
# =============================================================================

#SBATCH --job-name=stageA-test
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

set -euo pipefail

SCRATCH=/local/scratch/$USER
source "$SCRATCH/project/scripts/_env.sh"

# --- Isolated output paths (timestamped) ---
SUFFIX="stageA-$(date +%Y%m%d-%H%M)"
ISO_CHUNKS="data/flood-${SUFFIX}/processed/zone1_chunks.json"
ISO_RESULTS="data/results/flood-${SUFFIX}"

mkdir -p "$(dirname "$ISO_CHUNKS")" "$ISO_RESULTS"

echo "============================================================"
echo "Stage A verification run"
echo "  Branch        : $(git -C "$PROJECT" rev-parse --abbrev-ref HEAD)"
echo "  Commit        : $(git -C "$PROJECT" rev-parse --short HEAD)"
echo "  Model         : $MODEL"
echo "  Passes        : $PASSES"
echo "  Chunks output : $ISO_CHUNKS"
echo "  Results output: $ISO_RESULTS"
echo "============================================================"

_start_ollama

# --- Zone 1 — read default flood raw, write isolated chunks ---
echo ""
echo "===== ZONE 1 ====="
ZONE1_START=$(date +%s)
python3 zone1/ingestion.py \
    --output "$ISO_CHUNKS" \
    --model "$MODEL"
echo "Zone 1 complete in $(($(date +%s) - ZONE1_START))s"

# --- Zone 2 — Stage A flags + Neo4j unreachable URI ---
# NEO4J_URI override makes insert_to_neo4j fail fast on connect.  The
# try/except inside that node logs the error and the run continues —
# zone2_run_summary.json is still written with the full triple list.
echo ""
echo "===== ZONE 2 (--skip-canonicalize --skip-normalize) ====="
ZONE2_START=$(date +%s)
NEO4J_URI="bolt://localhost:0" python3 zone2/pipeline.py \
    --model "$MODEL" --passes "$PASSES" \
    --chunks "$ISO_CHUNKS" \
    --results-dir "$ISO_RESULTS" \
    --skip-canonicalize \
    --skip-normalize
echo "Zone 2 complete in $(($(date +%s) - ZONE2_START))s"

# --- Spot-check: relation_raw populated, lob populated, no canon/normalize ran ---
echo ""
echo "===== STAGE A VERIFICATION ====="
python3 - <<PYEOF
import json, sys
path = "${ISO_RESULTS}/zone2_run_summary.json"
data = json.load(open(path))
t = data.get("triples", [])
n = len(t)

with_raw = sum(1 for x in t if x.get("relation_raw"))
with_lob = sum(1 for x in t if x.get("lob"))
canon_distinct = len({x["relation"]      for x in t})
raw_distinct   = len({x.get("relation_raw", x["relation"]) for x in t})

print(f"  Total triples         : {n}")
print(f"  With relation_raw     : {with_raw}/{n}")
print(f"  With lob              : {with_lob}/{n}")
print(f"  Distinct relation     : {canon_distinct}")
print(f"  Distinct relation_raw : {raw_distinct}")
print()

if with_raw < n:
    print(f"  ✗ FAIL: {n - with_raw} triples missing relation_raw", file=sys.stderr)
    sys.exit(1)
if with_lob < n:
    print(f"  ✗ FAIL: {n - with_lob} triples missing lob", file=sys.stderr)
    sys.exit(1)
# Under --skip-canonicalize --skip-normalize, distinct(relation) should
# equal or nearly equal distinct(relation_raw) — neither merging step ran.
if canon_distinct < raw_distinct * 0.95:
    print(f"  ✗ FAIL: relation diversity collapsed despite skip flags "
          f"({canon_distinct} canon vs {raw_distinct} raw)", file=sys.stderr)
    sys.exit(1)
print("  ✓ All checks passed.")
PYEOF

echo ""
echo "============================================================"
echo "Stage A verification done."
echo "  Inspect: $ISO_RESULTS/zone2_run_summary.json"
echo "  Sample : python3 -c \"import json; d=json.load(open('$ISO_RESULTS/zone2_run_summary.json')); print(json.dumps(d['triples'][:3], indent=2))\""
echo "============================================================"
