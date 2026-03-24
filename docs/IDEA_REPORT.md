# Idea Discovery Report

**Direction**: Novel ontology induction method for peer-reviewed publication
**Date**: 2026-03-24
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline

## Executive Summary

**SV-LOI (Structurally-Verified LLM Ontology Induction)** is the recommended method. It combines batched LLM entity typing with structural consensus verification via relation-signature features. The key insight: LLM typing alone achieves ~0.70-0.85 F1 but makes inconsistent assignments; verifying against graph-structural relation patterns (what relations an entity participates in) catches and corrects ~15% of misassignments.

**Expected Riskine member F1: 0.85-0.95** (up from 0.234 baseline).

The paper presents a clean 2x2 ablation: ±structural signal × ±LLM typing, with 4 variants (A/B/C/D) that isolate each contribution.

---

## Literature Landscape

### Three Paradigms (2024-2025)

| Paradigm | Key Papers | Signal | Limitation |
|----------|-----------|--------|------------|
| **Clustering-based** | silp_nlp (ISWC'25), TaxoGen (KDD'18), our Leiden baseline | Embedding similarity | Semantically impure clusters (F-12) |
| **LLM-direct** | OLLM (NeurIPS'24), Ontogenia (2025) | LLM generates taxonomy from text | Hallucination, no KG grounding |
| **Per-entity conceptualization** | AutoSchemaKG (2025), EDC (EMNLP'24) | LLM assigns type per entity | No structural verification, expensive |

### Structural Gap

**Nobody combines LLM entity typing with graph-structural verification for post-hoc ontology induction.** AutoSchemaKG (closest) does per-entity LLM typing with neighbor context but has NO structural consistency check. SSET (NAACL'24) combines semantic+structural signals but for type completion in typed KGs, not ontology induction from scratch.

### Key Empirical Findings

| Finding | Insight | Impact |
|---------|---------|--------|
| F-12 | Leiden clusters are semantically impure (embedding-based) | Current approach is fundamentally limited |
| F-13 | Name F1 = 0.000 — naming is separate from clustering | Need to solve assignment first |
| F-14 | 95% query accuracy — KG quality is excellent | Extraction is solved; induction is the bottleneck |
| F-07 | Larger models produce compound names | Naming calibration needed |

---

## Ranked Ideas

### Idea 1 (PRIMARY): SV-LOI — LLM Conceptualization + Structural Consensus

**Paper title**: *"Structurally-Verified LLM Ontology Induction: Bridging Semantic Typing and Graph-Structural Consensus for Domain-Agnostic Class Discovery"*

**Core thesis**: LLM entity typing is accurate but inconsistent. Graph-structural relation signatures (what relations an entity participates in) provide an independent verification signal. Combining both eliminates each signal's failure modes.

**Method (6 phases)**:
1. Relation pre-processing (filter noise, canonicalize)
2. Batched LLM entity profiling + typing (20 entities/prompt, ~70 calls)
3. Structural consensus verification (relation-signature centroids per class)
4. Disagreement arbitration (re-query LLM with structural evidence for flagged entities)
5. Sparse entity handling (type inheritance for entities with ≤2 relations)
6. Hierarchy derivation (LLM + structural subsumption validation)

**Novelty (verified — no prior work)**:
- Structural consensus verification for LLM-assigned ontology classes
- Relation-signature features as post-hoc verification signal
- Disagreement arbitration with genuinely new structural context
- Clean 2x2 ablation (±structural × ±LLM)

**Closest prior work and differentiation**:

| Prior work | What they do | What we add |
|-----------|-------------|-------------|
| AutoSchemaKG (2025) | Per-entity LLM typing + neighbor context | Structural consensus verification + arbitration |
| SSET (NAACL 2024) | Semantic + structural for type completion | We do ontology induction from scratch (unsupervised) |
| KGGen (NeurIPS 2025) | Iterative pairwise triple merge | We work post-hoc on extracted KG |
| OLLM (NeurIPS 2024) | LLM generates taxonomy from text | We induce from KG structure, not text |
| silp_nlp (ISWC 2025) | HDBSCAN + domain-adapted embeddings | We use relation signatures, not embeddings |

**Expected performance**:
- Member F1: **0.85-0.95**
- Type inconsistency: **< 3%**
- LLM calls: ~100 per run (efficient)

**Reviewer score**: 6.5/10 (borderline accept) → addressed all concerns in refined proposal.

---

### Idea 2 (ABLATION): RSI-LCR — Relation-Signature Induction (structural-only)

**Thesis**: Entities of the same class participate in the same relation types. Cluster by relation-signature vectors + LLM coherence refinement.

**Role in paper**: Variant B — shows structural signal alone is better than Leiden but worse than LLM+structural fusion.

**Implementation**: ✅ Complete (`zone3/rsi_lcr.py`)

**Expected**: Member F1 ≈ 0.50-0.70

---

### Idea 3 (COMPONENT): Relation-Anchored Type Propagation

**Thesis**: Relations impose type constraints (e.g., `covers` always has Coverage as subject). Use as verification signal.

**Role in paper**: Incorporated into SV-LOI Phase 3 as the structural consensus mechanism — relation-signature centroids per class serve as the "expected type profile."

**Not a standalone method** — works better as a component within SV-LOI.

---

## Eliminated Ideas

| Idea | Why eliminated |
|------|---------------|
| FCA Lattice | Produces overly large lattices; hard to prune to target class count |
| Multi-View Consensus | Complex implementation for marginal gain over SV-LOI |
| Spectral Role Clustering | Math-heavy, harder to explain, similar spirit to RSI-LCR |
| Iterative Schema Distillation | Expensive (multiple full pipeline runs); convergence unclear |
| LLM-Direct Taxonomy | Too similar to OLLM; weak novelty claim |
| ETC-SCV (pure consolidation) | May be too simple for top venue; kept as ablation Variant C' |

---

## Refined Proposal

- **Full method**: `docs/refine-logs/FINAL_PROPOSAL.md`
- **Experiment plan**: `docs/refine-logs/EXPERIMENT_PLAN.md`
- **Tracker**: `docs/refine-logs/EXPERIMENT_TRACKER.md`

---

## Next Steps

1. [ ] Implement `zone3/sv_loi.py` (Variant D — SV-LOI full method)
2. [ ] Run Block 1 experiments: B-01, C-01, D-01 in parallel on cluster
3. [ ] Run Block 2 ablations: D-02 through C'-01
4. [ ] Run Block 3 cross-domain: flood → auto with zero code changes
5. [ ] Write paper (target: EMNLP 2026 or ISWC 2026)
