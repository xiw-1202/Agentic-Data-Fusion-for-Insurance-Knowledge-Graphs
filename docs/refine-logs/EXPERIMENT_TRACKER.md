# Experiment Tracker: SV-LOI

## Status

| Run ID | Variant | Method | Status | Member F1 | Name F1 | Type Incon. | Purity | Notes |
|--------|---------|--------|:------:|:---------:|:-------:|:-----------:|:------:|-------|
| A-01 | A | Leiden baseline (72b) | ✅ Done | 0.234 | 0.000 | 8.3% | ~40% | 11 classes |
| B-01 | B | RSI-LCR (structural) | ⏳ Pending | — | — | — | — | Priority 1 |
| C-01 | C | LLM typing, no consensus | ⏳ Pending | — | — | — | — | Priority 1 |
| D-01 | D | SV-LOI full | ⏳ Pending | — | — | — | — | Priority 1 |
| D-02 | D-abl | SV-LOI minus arbitration | ⏳ Pending | — | — | — | — | Block 2 |
| D-03 | D-abl | SV-LOI minus sparse handling | ⏳ Pending | — | — | — | — | Block 2 |
| D-04 | D-abl | SV-LOI minus relation preproc | ⏳ Pending | — | — | — | — | Block 2 |
| C'-01 | C' | ETC-SCV (type consolidation) | ⏳ Pending | — | — | — | — | Block 2 |
| D-05 | D | SV-LOI on auto insurance | ⏳ Pending | — | — | — | — | Block 3 |
| A-02 | A | Leiden on auto insurance | ⏳ Pending | — | — | — | — | Block 3 |

## Implementation Status

| Component | File | Status |
|-----------|------|:------:|
| RSI-LCR (Variant B) | `zone3/rsi_lcr.py` | ✅ Implemented |
| SV-LOI (Variant D) | `zone3/sv_loi.py` | ⏳ Not started |
| LLM Typing only (Variant C) | ablation of D | ⏳ Not started |
| ETC-SCV (Variant C') | ablation of D | ⏳ Not started |
| Evaluation (member F1, purity) | `evaluation/riskine_eval.py` | ✅ Implemented |
| Cross-domain (auto data) | `data/auto/` | ⏳ Need data |

## Log

- **2026-03-22**: RSI-LCR implemented and IDEA_REPORT drafted
- **2026-03-24**: SV-LOI proposal finalized after idea discovery pipeline
  - Evolved from RSI-LCR-only → LLM Conceptualization + Structural Consensus
  - 4-variant ablation design (A/B/C/D)
  - Reviewer concerns addressed (sparse entities, arbitration, cross-domain)
