# Experiment Plan: SV-LOI Validation

## Overview

Validate SV-LOI through:
1. Head-to-head comparison of 4 variants (A/B/C/D)
2. Ablation of each component
3. Cross-domain transfer test (flood → auto)
4. Sensitivity analysis (model size, batch size, threshold)

## Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Member F1** (primary) | F1 between induced class members and Riskine class members via embedding similarity | > 0.90 |
| **Name F1** (secondary) | F1 between induced class names and Riskine class names | > 0.50 |
| **Type Inconsistency** | % of entities with >1 conflicting ontology label | < 5% |
| **Class Purity** | % of entities in each induced class that belong to the same reference class | > 85% |
| **Coverage** | # of Riskine classes with at least 1 matching induced class | ≥ 8/10 |
| **CTR** (cross-domain) | Class Transfer Rate — how many classes appear in both flood and auto | > 60% |
| **PTR** (cross-domain) | Property Transfer Rate — how many relations appear in both domains | > 50% |

## Experiment Blocks

### Block 1: Core Comparison (MUST-RUN)

| Run ID | Variant | Method | Model | Priority |
|--------|---------|--------|-------|----------|
| A-01 | A | Leiden baseline | qwen2.5:72b | ✅ Done (F1=0.234) |
| B-01 | B | RSI-LCR (structural only) | qwen2.5:72b | P1 |
| C-01 | C | LLM typing, no consensus | qwen2.5:72b | P1 |
| D-01 | D | SV-LOI full | qwen2.5:72b | P1 |

**Goal**: Show D > C > B > A progression.
**GPU hours**: ~4h total (most time is LLM inference for C and D).

### Block 2: Component Ablation (MUST-RUN)

| Run ID | Ablation | What's removed | Expected impact |
|--------|----------|---------------|-----------------|
| D-02 | D minus arbitration | No Phase 4 re-query; keep LLM typing + structural flagging only | F1 drops ~0.05-0.10 |
| D-03 | D minus sparse handling | No Phase 5; sparse entities use structural consensus anyway | F1 drops ~0.02-0.05 |
| D-04 | D minus relation pre-processing | Keep all 200+ raw relation types | F1 drops if noise hurts signatures |
| C'-01 | Type consolidation only (ETC-SCV) | Just consolidate 20 types, no per-entity LLM re-typing | Shows if extraction types suffice |

**Goal**: Quantify each component's contribution. Show structural consensus (D vs C) adds the most value.
**GPU hours**: ~3h total.

### Block 3: Cross-Domain Transfer (MUST-RUN)

| Run ID | Domain | Method | Notes |
|--------|--------|--------|-------|
| D-05 | Auto insurance | SV-LOI full | Zero code changes from flood |
| A-02 | Auto insurance | Leiden baseline | Comparison |

**Goal**: Show domain-agnosticity. Report CTR and PTR.
**GPU hours**: ~3h (need Zone 2 extraction on auto data first).

### Block 4: Sensitivity Analysis (NICE-TO-HAVE)

| Run ID | Variable | Values | Notes |
|--------|----------|--------|-------|
| D-06 | Model size | qwen2.5:7b | Data leakage check + cost analysis |
| D-07 | Batch size | 10, 20, 40 | Does batch size affect typing quality? |
| D-08 | Flagging threshold | 1σ, 2σ, 3σ | How much arbitration is optimal? |
| D-09 | Expand to 27 Riskine classes | Full Riskine eval | Does F1 hold with more classes? |

**GPU hours**: ~6h total.

## Run Order (Critical Path)

```
Phase 1: Block 1 in parallel (B-01, C-01, D-01)         ~4h
Phase 2: Block 2 ablations (D-02 through C'-01)          ~3h
Phase 3: Block 3 cross-domain (D-05, A-02)               ~3h
Phase 4: Block 4 sensitivity (if time permits)            ~6h
                                                    Total: ~16h
```

**Minimum viable**: Blocks 1-2 (7h) give a complete paper. Block 3 strengthens the domain-agnostic claim.

## Expected Results Table (for paper)

| Method | Member F1 | Name F1 | Type Incon. | Purity | # Classes |
|--------|:---------:|:-------:|:-----------:|:------:|:---------:|
| Leiden (A) | 0.234 | 0.000 | 8.3% | ~40% | 11 |
| RSI-LCR (B) | 0.50-0.70 | 0.10-0.30 | <5% | ~60% | 12-18 |
| LLM typing (C) | 0.70-0.85 | 0.30-0.50 | <3% | ~75% | 10-15 |
| **SV-LOI (D)** | **0.85-0.95** | **0.40-0.60** | **<3%** | **>85%** | **10-15** |
| ETC-SCV (C') | 0.60-0.80 | 0.20-0.40 | <5% | ~65% | 10-15 |

## Compute Budget

| Resource | Budget | Notes |
|----------|--------|-------|
| GPU | 1× A100 (cluster) or local 72b via Ollama | qwen2.5:72b requires ~40GB VRAM |
| LLM calls per D run | ~100 (70 typing + 20 arbitration + 10 misc) | ~30 min at 72b speed |
| Neo4j | AuraDB (existing) | No additional cost |
| Total GPU hours | ~16h (all blocks) or ~7h (minimum viable) | |

## Success Criteria

**Minimum for paper**:
- D-01 member F1 > 0.80
- D-01 > C-01 (structural consensus helps)
- C-01 > B-01 (LLM typing helps)
- B-01 > A-01 (relation signatures help)

**Stretch goals**:
- D-01 member F1 > 0.90
- Cross-domain CTR > 60%
- Per-class F1 breakdown showing consistent improvement across all Riskine classes

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|------------|------------|
| LLM typing accuracy too low | LOW (72b is strong) | Increase batch context; add few-shot examples |
| Structural signal too noisy (200+ relations) | MEDIUM | Relation pre-processing; test with/without (D-04) |
| Sparse entities dominate | MEDIUM | Measure sparse entity % first; tune threshold |
| Cross-domain fails (auto too different) | LOW | Same pipeline design; types will differ but method won't |
| F1 saturates below 0.85 | MEDIUM | Combine with ETC-SCV (type consolidation as init) |
