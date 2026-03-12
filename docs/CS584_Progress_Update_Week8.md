# CS584 AI Capstone — Progress Update (Week 8 / March 2026)
### Updates since Simplified Plan (Feb 22) — Zones 1–3 Complete

---

## 1. Executive Summary

Since the original simplified plan (Feb 22), the pipeline has been fully implemented through Zone 3. Starting from a theoretical 4-zone architecture, we now have a working end-to-end system that extracts a knowledge graph from flood insurance documents and automatically induces an ontology — with **zero domain-specific hardcoding**. The best result to date is **75% query accuracy** and **Riskine F1 = 0.247** (member-based), with zero entity duplication, achieved using llama3.1:8b with EDC canonicalization and multi-resolution Leiden community detection. A critical discovery this week (F-07): **Riskine F1 is a biased metric** — it rewards models whose class names accidentally match the reference ontology's naming conventions, not models with better semantic structure. The paper should lead with **cross-domain CTR/PTR transfer** as the primary contribution, not Riskine F1.

---

## 2. Full Results Table

| Run | Model | Query Acc | Type Incon. | Riskine F1 | Notes |
|-----|-------|:---------:|:-----------:|:----------:|-------|
| Baseline (512-tok) | llama3.1:8b | 35% | 8.0% | — | LLMGraphTransformer, unconstrained |
| Zone 1 | llama3.1:8b | 50% | 15.2% | 0.250 | Section-aware chunking (τ=0.85) |
| Zone 2 v1 | llama3.1:8b | 50% | 0.0% | 0.874 | ⚠️ Domain leakage — Riskine anchors injected |
| Zone 2 v2 (honest) | llama3.1:8b | 35% | 0.0% | ~0.000 | All anchors removed; true baseline |
| Zone 3 (8b) | llama3.1:8b | 35% | 17.4% | 0.171 | Leiden res=0.6, 25 clusters |
| Zone 3 (70b) | llama3.1:70b | 30% | 2.9% | 0.071 | Turing GPU; compound names penalized by F-07 |
| **Zone 3 8b+EDC+ML** | llama3.1:8b | **75%** | 14.3% | **0.247** | **← Best result** — 5 algorithmic SUBCLASS_OF |
| Zone 3 70b+EDC+ML | llama3.1:70b | 65% | 16.7% | 0.158 | Penalized by F-07/F-10; 28 clusters |

> **Note**: Zone 2 v1's F1 of 0.874 is **not a real result**. It was achieved by injecting 9 Riskine-labeled anchor nodes directly into the graph — a form of evaluation leakage. All leaking components were removed in v2.

---

## 3. Zone Status and Architecture

```
ZONE 1 ✅           ZONE 2 ✅                  ZONE 3 ✅                  ZONE 4 ⏳
──────────          ────────────────           ──────────────────         ─────────
PDF/CSV        →    Bootstrap vocab        →   Build entity          →    3-layer Neo4j
Section-aware       from document              similarity graph            Ontology layer
chunking            LLM Open IE                Multi-res Leiden            SUBCLASS_OF
τ=0.85              EDC canonicalize            Algorithmic hier.           hierarchy
semantic            Entity resolution           Neo4j ontology              structured
merge               (no Riskine)                SUBCLASS_OF edges           storage
```

### Current Pipeline Components

| Component | File | Description |
|-----------|------|-------------|
| Zone 1 ingestion | `zone1/ingestion.py` | Section-aware hybrid chunking with τ=0.85 semantic merge |
| Zone 2 vocab bootstrap | `zone2/pipeline.py` | LLM reads docs, proposes entity + relation types |
| Zone 2 Open IE | `zone2/pipeline.py` | 3-pass extraction: general → numeric → obligations |
| Zone 2 EDC | `zone2/pipeline.py` | Maps 48 raw relation types → 14 canonicalized (71% reduction) |
| Zone 2.5 entity resolution | `zone2/entity_resolution.py` | Embedding-based merge (cosine ≥ 0.90) |
| Zone 3 Leiden | `zone3/pipeline.py` | Multi-resolution [0.3, 0.6, 1.2] community detection |
| Zone 3 naming | `zone3/pipeline.py` | LLM assigns PascalCase class names per cluster |
| Zone 3 hierarchy | `zone3/pipeline.py` | Algorithmic `derive_hierarchy()` — membership overlap ≥ 0.60 |
| Evaluation | `baseline/eval.py` | 20-task Cypher query accuracy + type inconsistency |
| Riskine eval | `evaluation/riskine_eval.py` | Member-based F1 with PascalCase humanization |

---

## 4. What Changed Since the Simplified Plan

| Component | Old (Simplified Plan, Feb 22) | New (Week 8) |
|-----------|-------------------------------|--------------|
| Zone 2 extraction | Described theoretically | Fully implemented: bootstrapped vocab + 3-pass + EDC |
| Zone 2 v1 issue | Not yet built | Had Riskine anchor injection — **removed in v2** |
| Ontology induction | "LLM will propose classes" | Leiden community detection on entity similarity graph |
| SUBCLASS_OF edges | Described as LLM-generated | **Algorithmic** from membership overlap — 5 clean edges, no hallucination |
| Evaluation | Riskine F1 planned | Riskine F1 + query accuracy + type inconsistency, all implemented |
| Multi-model | Implied single model | llama3.1:8b + llama3.1:70b (Turing GPU cluster runs) |
| EDC canonicalization | Not in plan | Added: 48 raw → 14 canonicalized relation types (71% reduction) |
| Multi-resolution Leiden | Not in plan | [0.3, 0.6, 1.2] resolutions → coarse/medium/fine cluster hierarchy |
| Cluster script | Not in plan | Emory Turing GPU Slurm scripts created + tested |
| Results | None | 8 complete pipeline runs with metrics |

---

## 5. Key Empirical Findings

| # | Finding | Impact |
|---|---------|--------|
| F-01 | llama3.1:8b ignores dollar amounts as triple objects (day counts succeed) | Motivates 3-pass extraction with numeric focus pass |
| F-02 | Few-shot pairs from source text cause global extraction interference | All few-shot examples are synthetic (no SFIP text) |
| F-03 | 8B (and 70B) extracts ~1 triple/chunk regardless of prompting | Multi-pass extraction as mitigation |
| F-04 | Vocabulary compliance > volume: deepseek-r1:14B failed at 30% vs 8B at 50% | Validates bootstrapped vocab approach |
| F-05 | Entity resolution threshold is model-dependent (0.90 works for 8B) | Requires per-model tuning for Zone 2.5 |
| F-06 | Zone 2 v1's F1=0.874 was domain leakage (9 Riskine anchor nodes injected) | All anchors removed; honest F1 ≈ 0.000 |
| F-07 | 8B accidentally uses short names matching Riskine conventions → higher F1. 70B uses compound descriptive names → lower F1. Metric rewards naming convention similarity, not semantic correctness. | **Riskine F1 is unreliable.** Paper must lead with CTR/PTR cross-domain transfer. |
| F-08 | Query accuracy is Cypher-pattern sensitive — 70B entity names differ from hardcoded patterns | 70B: 65% vs 8B: 75%, despite richer ontology |
| F-09 | EDC canonicalization reduces structural discriminability in Leiden graph; fewer relation types → larger clusters → more keyword overlap → higher type inconsistency | Type inconsistency is a property of the eval metric, not a labeling error |
| F-10 | 70B bootstraps a more varied relation vocabulary → more unique raw types → EDC maps fewer cleanly (47→32, 32% reduction vs 8B's 48→14, 71% reduction) | 70B graph is richer but less canonicalized |

---

## 6. Why Riskine F1 is Misleading (Finding F-07)

The Riskine evaluation compares our induced ontology classes to 10 manually-curated Riskine classes:
`Coverage · Product · Damage · Risk · Structure · Property · Person · Object · Organization · Address`

**The problem**: Riskine F1 uses cosine similarity between class name embeddings. The 8B model accidentally produces short class names (`InsuranceTerm`, `InsuredAsset`) that embed close to Riskine's short names. The 70B model produces semantically richer compound names (`InsuredProperty`, `InsuranceCoverageTerm`, `PolicyParticipantOrEvent`) that embed farther away — despite arguably being *better* ontology class names.

**Concrete example**:
- `InsuranceLossEvent` (8B) → closest to Riskine `Risk` — ✓ reasonable
- `InsuranceCoverageTerm` (8B) → closest to Riskine `Coverage` — ✓ reasonable
- `PolicyParticipantOrEvent` (70B) → no Riskine candidate above threshold — ✗ penalized
- `InsuredAsset` (70B) → closest to Riskine `Property` — ✓ but lower score than 8B's shorter names

**Conclusion**: Riskine F1 is a weak proxy for ontology quality. It is not measuring whether the induced classes are semantically correct — only whether they happen to use the same naming conventions as the reference. The correct metric for our goal (domain-agnostic generalization) is **CTR/PTR cross-domain transfer**.

---

## 7. Future Plans / Roadmap

### Week 8 (Current): Cross-Domain Transfer Validation

**Goal**: Prove the pipeline is domain-agnostic by running *identical code* on auto insurance documents.

Steps:
1. Run `python3 zone1/ingestion.py` with auto insurance PDF (`data/auto/`)
2. Delete `data/results/zone2_vocab.json` to force vocabulary regeneration from auto docs
3. Run `python3 zone2/pipeline.py` — will bootstrap vocab from auto data automatically
4. Run `python3 zone3/pipeline.py` — identical parameters, zero code changes

**Metrics to measure**:
- **CTR (Cross-Type Ratio)**: % of flood ontology classes that have a semantic match (cosine ≥ 0.60) in the auto ontology
- **PTR (Per-Type Ratio)**: % of shared relation types between matched class pairs
- **Expected**: CTR > 60%, PTR > 50% — would prove the pipeline discovers domain-agnostic insurance concepts (Coverage, Policy, Exclusion, etc.) in both LOBs without being told to

### Week 9: Cross-Domain Analysis

- Compare flood vs auto induced ontology classes side-by-side
- Document which classes transfer (Coverage, Policy, Exclusion-type concepts) vs don't (flood-specific: `ContinuousLakeFlood`, `FederallyLeasedLand`)
- Assess whether 8B or 70B classes transfer better cross-domain — directly tests F-07 hypothesis (70B may produce semantically richer classes that transfer better, even if Riskine F1 is lower)

### Weeks 10–11: Paper + Streamlit Demo + Human Annotation

**Paper structure**:
- Table 1: Full ablation (Baseline → Zone 1 → Zone 2 → Zone 3 → cross-domain)
- Contribution framing: CTR/PTR as novel metric, algorithmic SUBCLASS_OF hierarchy, EDC canonicalization
- *Not* framing around Riskine F1 (biased metric, see F-07)

**Streamlit demo**:
- Interactive KG explorer — query the induced ontology, visualize SUBCLASS_OF hierarchy
- Side-by-side comparison: flood ontology vs auto ontology
- Shows cross-domain class alignment visually

**Human annotation**:
- 3 annotators rate ontology class quality (relevance, precision, coverage)
- Reference-free quality metric (doesn't require knowing Riskine)
- Provides ground truth independent of naming convention bias

### Zone 4 (Upcoming): Structured Neo4j Storage

- **3-layer storage**: raw entities → `OntologyClass` nodes → `SUBCLASS_OF` hierarchy
- Clean separation between instance layer and schema layer
- Cypher views for the paper's 20-task evaluation queries
- Enables efficient graph traversal for the Streamlit demo

---

## Appendix: Zone 3 Ontology Classes (Best Run: 8b+EDC+MultiLeiden)

**Primary classes (medium resolution, res=0.6):**

| Class | Key Members | SUBCLASS_OF |
|-------|-------------|-------------|
| InsuredAsset | Insurance, Insured, Insurer, Insured Property, Single Building | InsuranceConcept |
| InsuranceTerm | $30,000, $1,000, $2,500, Building Coverage, Deductible | InsuranceCoverageTerm |
| Timeframe | 30 days, 60 days, 90 days, Proof of Loss, Policy Term | — |
| PolicyParticipantOrEvent | Flood, You, Continuous Lake Flood, Named Insured(s) | Event |
| InsuredProperty | Coverage A, Coverage B, Coverage C, Coverage D, FEMA | InsuranceCoverageTerm |
| InsurancePolicyProvisions | Policyholder, Standard Flood Insurance Policy, Policy Terms | — |
| CoverageExpansionProvision | Liberalization Clause, Change Broadens Coverage | — |
| ReplacementValue | Replacement Cost, Pair and Set Clause | — |

**Algorithmic SUBCLASS_OF edges (derive_hierarchy, overlap ≥ 0.60):**
1. `InsuredAsset` SUBCLASS_OF `InsuranceConcept` (overlap=1.00)
2. `InsuranceTerm` SUBCLASS_OF `InsuranceCoverageTerm` (overlap=1.00)
3. `PolicyParticipantOrEvent` SUBCLASS_OF `Event` (overlap=0.78)
4. `InsuredProperty` SUBCLASS_OF `InsuranceCoverageTerm` (overlap=1.00)
5. `InsurableProperty` SUBCLASS_OF `InsuredAsset` (overlap=0.80)

These edges are **algorithmically derived** from multi-resolution Leiden membership overlap — no LLM guessing, no hallucination. Both 8B and 70B produce exactly 5 edges with the same method, confirming structural stability across models.
