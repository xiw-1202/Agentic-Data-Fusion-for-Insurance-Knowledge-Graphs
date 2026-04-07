#!/bin/bash
# Run all SV-LOI ablation variants for the paper.
# Usage: bash scripts/run_ablations.sh [MODEL]
# Default model: qwen2.5:72b

MODEL="${1:-qwen2.5:72b}"
echo "Running SV-LOI ablation study with model: $MODEL"
echo "================================================================"

# Full SV-LOI (all components enabled)
echo ""
echo "=== Variant 1/5: Full SV-LOI ==="
python3 zone3/sv_loi.py --model "$MODEL" --suffix zone3_svloi_full --seed 42

# No structural verification (Phase 2 disabled)
echo ""
echo "=== Variant 2/5: No Verification ==="
python3 zone3/sv_loi.py --model "$MODEL" --suffix zone3_svloi_noverify --skip-verify --seed 42

# No arbitration (Phase 3 disabled)
echo ""
echo "=== Variant 3/5: No Arbitration ==="
python3 zone3/sv_loi.py --model "$MODEL" --suffix zone3_svloi_noarb --skip-arbitrate --seed 42

# No consolidation (Phase 4a disabled)
echo ""
echo "=== Variant 4/5: No Consolidation ==="
python3 zone3/sv_loi.py --model "$MODEL" --suffix zone3_svloi_nocons --skip-consolidate --seed 42

# LLM-only (no verify + no arbitrate = pure LLM typing)
echo ""
echo "=== Variant 5/5: LLM-Only (no verify, no arbitrate) ==="
python3 zone3/sv_loi.py --model "$MODEL" --suffix zone3_svloi_llmonly --skip-verify --skip-arbitrate --seed 42

echo ""
echo "================================================================"
echo "All ablations complete. Run evaluation:"
echo ""
for suffix in zone3_svloi_full zone3_svloi_noverify zone3_svloi_noarb zone3_svloi_nocons zone3_svloi_llmonly; do
    echo "  python3 baseline/eval.py --suffix $suffix --riskine --all-classes --model $MODEL"
done
