#!/bin/bash
# =============================================================================
# slurm_zone3_fbi.sh — Zone 3 FBI (Fingerprint-Based Ontology Induction)
#
# Runs Phases 1-3 (no Neo4j needed). Phase 4 requires Neo4j separately.
#
# Usage:
#   sbatch --export=ALL,DATA_DIR=data/Emory_Spring2026 scripts/slurm_zone3_fbi.sh
#   sbatch --export=ALL,MODEL=gemma4:31b,DATA_DIR=data/Emory_Spring2026 scripts/slurm_zone3_fbi.sh
# =============================================================================

#SBATCH --job-name=zone3-fbi
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/local/scratch/%u/logs/%j.out
#SBATCH --error=/local/scratch/%u/logs/%j.err

SCRATCH=/local/scratch/$USER
source "$SCRATCH/project/scripts/_env.sh"

_start_ollama

echo ""
echo "===== ZONE 3 — FBI Ontology Induction ($MODEL) ====="
echo "  Data dir:    $DATA_DIR"
echo "  Results dir: $RESULTS_DIR"
echo ""

FBI_START=$(date +%s)
python3 -m zone3.fbi.pipeline \
    --data-dir "$DATA_DIR" \
    --model "$MODEL" \
    --output-dir "$RESULTS_DIR" \
    --skip-neo4j
echo "FBI complete: $(($(date +%s) - FBI_START))s"

echo ""
echo "===== FBI RESULTS ====="
echo "Fingerprints:"
python3 -c "
import json
with open('${RESULTS_DIR}/fbi_fingerprints.json') as f:
    fps = json.load(f)
for fp in fps:
    if fp['file_type'] == 'csv':
        print(f\"  {fp['file']}: {len(fp['headers_raw'])} headers, tokens={fp['filename_tokens']}\")
    else:
        print(f\"  {fp['file']}: {len(fp.get('sections',[]))} sections, tokens={fp['filename_tokens']}\")
"

echo ""
echo "Classes:"
python3 -c "
import json
def print_tree(cls, indent=0):
    name = cls.get('name', cls.get('prefix', '?'))
    n_headers = len(cls.get('headers', []))
    srcs = ', '.join(cls.get('source_files', []))
    print(f\"{'  ' * indent}{name} ({n_headers} headers) [{srcs}]\")
    for child in cls.get('children', []):
        print_tree(child, indent + 1)

with open('${RESULTS_DIR}/fbi_classes.json') as f:
    classes = json.load(f)
for cls in classes:
    print_tree(cls)
print(f'\nTotal top-level classes: {len(classes)}')
"

echo ""
echo "Relationships:"
python3 -c "
import json
with open('${RESULTS_DIR}/fbi_relationships.json') as f:
    rels = json.load(f)
for r in rels:
    print(f\"  {r['source']} --{r['name']}--> {r['target']} (via {r['bridge']})\")
print(f'\nTotal relationships: {len(rels)}')
"

_stop_ollama

echo ""
echo "===== ZONE 3 FBI COMPLETE — $(date) ====="
echo "  Results: $RESULTS_DIR/"
