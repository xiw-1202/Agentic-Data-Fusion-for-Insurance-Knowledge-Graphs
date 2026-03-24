# Final Proposal: Structurally-Verified LLM Ontology Induction (SV-LOI)

## Problem Anchor (FROZEN)

**Problem**: Given an extracted KG with N entities, M relations, and T bootstrapped entity types, induce an ontology (set of classes + SUBCLASS_OF hierarchy + entity-to-class assignments) that aligns with a held-out reference ontology — without ever seeing the reference.

**Input**: Neo4j graph from Zone 2 (1,351 entities, 5,421 triples, 20 entity types, ~200 relation types)
**Output**: 10-30 ontology classes with SUBCLASS_OF hierarchy + entity assignments
**Metric**: Riskine member F1 (primary), name F1 (secondary), type inconsistency rate
**Target**: Member F1 > 0.9, type inconsistency < 5%
**Constraint**: Domain-agnostic — zero code changes between flood and auto insurance

---

## Method: SV-LOI (4 Variants for Ablation)

### Variant A: Leiden Baseline (existing Zone 3)
Multi-resolution Leiden on 4D similarity graph → LLM naming.
**Expected**: Member F1 ≈ 0.234 (measured)

### Variant B: RSI-LCR (structural-only signal)
Relation-signature HAC → silhouette-guided k → LLM coherence refinement → LLM naming.
**Expected**: Member F1 ≈ 0.50-0.70

### Variant C: LLM Typing without Consensus (≈AutoSchemaKG ablation)
Batched LLM entity typing (20 per prompt, with relation context) → type consolidation → hierarchy.
No structural verification step.
**Expected**: Member F1 ≈ 0.70-0.85

### Variant D: SV-LOI Full Method (PRIMARY CONTRIBUTION)

**Phase 1 — Relation Pre-processing**:
Filter relation types: remove count=1 singletons and obvious hallucination chains.
Target: ~40-60 canonical relation types from ~200 raw types.
Build relation-signature feature matrix (same as RSI-LCR Step 2).

**Phase 2 — LLM Entity Profiling + Typing** (batched):
For each batch of 20 entities, build profiles:
- Entity name + bootstrapped type from Zone 2
- Top 5 outgoing relations with target entity names
- Top 5 incoming relations with source entity names
LLM assigns an ontology class to each entity.
~70 LLM calls for 1,351 entities.

**Phase 3 — Structural Consensus Verification**:
For each induced class C:
- Compute relation-signature centroid from RSI features of all members
- For each entity in C, compute cosine similarity to C's centroid
- Flag entities where: (a) sim < class_mean - 2σ, OR (b) entity is structurally closer to a DIFFERENT class centroid

**Phase 4 — Disagreement Arbitration**:
For flagged entities (~10-25% of total), build an enriched prompt:
- Entity profile (same as Phase 2)
- "Your initial assignment: class X"
- "Structural evidence: class X members typically have {top 5 relations}. You have {your relations}."
- "Alternative: class Y members have {top 5 relations}. Your signature is closer to Y."
- "Which class is correct? X or Y?"
This provides genuinely NEW information vs Phase 2 (structural context the LLM didn't see before).

**Phase 5 — Sparse Entity Handling**:
Entities with ≤2 relations (sparse signature): skip structural verification.
Use type inheritance: class = LLM-assigned class from Phase 2 (no override).
If entity type is "Unknown", use name-embedding similarity to class centroids as tiebreaker.

**Phase 6 — Hierarchy Derivation**:
LLM proposes SUBCLASS_OF from the consolidated class list (1 LLM call).
Validate with structural subsumption: if class A's relation signature is a subset of class B's signature, then A is a candidate SUBCLASS_OF B.

**Expected**: Member F1 ≈ 0.85-0.95

---

## Ablation Story (clean 2×2 design)

|  | No structural signal | With structural signal |
|--|---------------------|----------------------|
| **No LLM typing** | Variant A (Leiden) = 0.234 | Variant B (RSI-LCR) = 0.50-0.70 |
| **With LLM typing** | Variant C (LLM only) = 0.70-0.85 | **Variant D (SV-LOI)** = 0.85-0.95 |

This cleanly shows: structural adds value, LLM adds value, combining both adds the most.

---

## Novelty Claims

1. **Structural consensus verification for LLM-assigned ontology classes** — AutoSchemaKG trusts LLM typing blindly; we verify against graph-structural relation signatures and arbitrate disagreements with enriched structural context.

2. **Relation-signature features as post-hoc verification signal** — bridging regular equivalence from network science with LLM conceptualization. Novel application: structural features verify/correct LLM assignments rather than being the primary signal.

3. **Disagreement arbitration with genuinely new information** — the LLM is re-queried only when structural evidence contradicts its initial typing, and the re-query prompt includes structural context absent from the original prompt.

4. **Sparse entity handling via type inheritance** — practical solution for the long tail of entities with insufficient structural signal.

5. **Clean 2×2 ablation** isolating structural vs. semantic signal contributions.

---

## Addressing Reviewer Concerns

| Concern (from Phase 4) | Resolution |
|------------------------|------------|
| W1: Sparse entities weaken consensus | Phase 5: type inheritance for entities with ≤2 relations |
| W2: Single evaluation domain | Cross-domain: flood + auto insurance with identical code |
| W3: Arbitration prompt underspecified | Phase 4: enriched prompt with class relation profiles + structural alternative |
| W4: No head-to-head vs AutoSchemaKG | Variant C = AutoSchemaKG ablation (LLM typing, no consensus) |
| W5: Type consolidation may suffice | Add Variant C': ETC-SCV (just consolidate 20 types, no re-typing) |
| W6: Data leakage | Test with 7b model; report per-class F1 to detect memorization |

---

## Key Design Decisions

1. **Batch size = 20**: Balances context richness vs. prompt length. 20 entities × ~5 lines = ~100 lines.
2. **Flagging threshold**: 2σ below class mean cosine. Estimated ~15% of entities flagged.
3. **Relation pre-processing**: Remove count=1 singletons and hallucination chains (HAS_FLOOD_ZONE_VEL_VEL_VEL_...). Target ~40-60 clean relation types.
4. **LLM model**: qwen2.5:72b (primary), qwen2.5:7b (ablation).
5. **No reference ontology in pipeline**: Riskine is ONLY used in evaluation.
6. **Entity type inventory**: 20 types from 72b extraction (not the 52 from 8b — 72b is more disciplined).

---

## Comparison to Prior Work

| Method | Signal | Structural verification | Post-hoc (on existing KG) | Domain-agnostic |
|--------|--------|:-:|:-:|:-:|
| AutoSchemaKG (2025) | LLM per-entity | ✗ | ✗ (during extraction) | ✓ |
| KGGen (NeurIPS 2025) | Pairwise merge | ✗ | ✗ (during extraction) | ✓ |
| OLLM (NeurIPS 2024) | LLM taxonomy | ✗ | ✗ (from text) | ✓ |
| SSET (NAACL 2024) | Semantic+Structural | ✓ (for type completion) | ✓ | ✗ (supervised) |
| silp_nlp (ISWC 2025) | HDBSCAN embeddings | ✗ | ✗ | ✓ |
| **SV-LOI (ours)** | **LLM + relation signatures** | **✓** | **✓** | **✓** |
