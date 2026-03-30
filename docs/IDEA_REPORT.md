# Idea Discovery Report

**Direction**: Improving extraction richness — extracting key identifiable entities (claim IDs, policy numbers, etc.) for queryable insurance KGs
**Date**: 2026-03-29
**Pipeline**: research-lit -> idea-creator -> novelty-check -> research-review -> research-refine

## Executive Summary

**Problem**: Current Zone 2 extraction produces 5,421 triples but they are "rule-rich, identity-poor" — the KG captures coverage rules, exclusions, and definitions but cannot answer "What happened to Claim X?" because structured records (CSV claims/policies) lose their individual identity during extraction.

**Recommended method**: **SEAF-KG** (Source-calibrated Evidence Aggregation and Fusion for KG Construction) — a refined version of the initial SAMGE idea, reframed from "source-adaptive routing" to "confidence-aware cross-source fusion" as the core research contribution.

**Key insight**: Different source types (structured records vs. unstructured text) produce entity/relation hypotheses with fundamentally different error profiles. The research contribution is not routing chunks to different prompts, but a **formal evidence fusion framework** that normalizes heterogeneous extractions, links entities across sources with calibrated confidence, and abstains when uncertain.

## Literature Landscape

### State of the Art (2024-2025)

| Approach | Key Paper | Gap for Our Problem |
|----------|-----------|---------------------|
| Schema-optimized extraction | [PARSE (2025)](https://arxiv.org/abs/2510.08623) | Web data focus, no mixed structured/unstructured |
| Two-phase entity->relation | [KGGen (NeurIPS 2025)](https://arxiv.org/html/2502.09956v1) | Plain text only, no source-type adaptation |
| Ontology-guided extraction | [ODKE+ (2025)](https://arxiv.org/html/2509.04696v1) | Requires pre-existing ontology |
| Multi-agent enrichment | [KARMA (2025)](https://arxiv.org/pdf/2502.06472) | Enterprise focus, not domain-agnostic |
| Incremental KG | [iText2KG (2024)](https://arxiv.org/html/2409.03284v1) | Text-only, no structured data handling |
| Tabular->KG matching | [SemTab Challenge (2025)](https://sem-tab-challenge.github.io/2025/) | Maps TO existing KGs (reverse direction) |
| LLM KG survey | [LLM-empowered KG (2025)](https://arxiv.org/html/2510.20345v1) | Survey identifies gap but no solution |
| Practical GraphRAG | [GraphRAG (2025)](https://arxiv.org/html/2507.03226v2) | Hybrid retrieval but text-only extraction |

### Structural Gap

All existing LLM-based KG extraction treats input as homogeneous text. No published work:
1. Adapts extraction strategy based on detected source structure
2. Performs formal cross-source entity linking between record-derived and text-derived entities
3. Does confidence-aware evidence fusion from heterogeneous sources
4. Maintains domain-agnosticism across all of the above

## Ranked Ideas

### Idea 1 (RECOMMENDED): SEAF-KG — Source-calibrated Evidence Aggregation and Fusion

**One-sentence contribution**:
> We introduce a source-calibrated evidence fusion framework for KG construction from heterogeneous structured and unstructured sources, combining deterministic schema-grounded extraction, LLM-based semantic proposal, and confidence-aware cross-source linking with abstention.

**Refined Method (4 stages)**:

#### Stage 1: Source-Specific Candidate Extraction

**Structured path** (CSV-derived chunks):
- **Deterministic schema-to-graph conversion**: Each row becomes an entity bundle. Columns map to relation templates:
  - Identifier-like columns -> entity IDs / composite keys
  - Categorical columns -> attribute relations
  - Numeric/date columns -> typed literals
- **Optional LLM refinement** (only where needed):
  - Column semantic typing when header meaning is ambiguous
  - Normalization of free-text cells
  - Suggesting relation labels from header names
  - Do NOT use LLM to reconstruct obvious deterministic triples

**Unstructured path** (PDF-derived chunks):
- Keep existing domain-agnostic Open IE with bootstrapped entity/relation types
- Extract semantic rules, definitions, coverage conditions as triples with provenance

#### Stage 2: Canonical Evidence Representation

Every extracted item becomes a standardized evidence tuple:
```
z = (subject, relation, object, attrs, provenance, source_type, local_confidence)
```
- Normalize names, values, dates, units
- Assign coarse types (bootstrapped from data)
- Compute embeddings for entity mentions

#### Stage 3: Cross-Source Entity Linking (CORE CONTRIBUTION)

**3a. Candidate Generation (blocking)**:
- Same coarse type
- Lexical overlap on normalized tokens (Jaccard > threshold)
- Character n-gram similarity
- Optional ANN nearest neighbors on embeddings

**3b. Pairwise Scoring**:
```
s(e_i, e_j) = sigma(w^T * phi(e_i, e_j))
```
Feature vector phi contains domain-agnostic features:
- Lexical: normalized Levenshtein, token Jaccard, abbreviation match
- Semantic: cosine(emb(e_i), emb(e_j))
- Type: exact match / compatible type indicator
- Context: overlap between neighboring attributes, headings
- Value: matching numeric/date/location values
- Source interaction: structured-unstructured pair indicator

**3c. Threshold with Abstention** (two thresholds):
- s >= tau_merge: merge entities
- tau_review <= s < tau_merge: uncertain, keep separate
- s < tau_review: reject

**3d. Cluster Construction**:
- Union-find with guard preventing merges on high-confidence attribute conflicts

#### Stage 4: Fact Fusion

Aggregate evidence from all mentions into canonical triples:
```
Score(f) = 1 - product_over_z(1 - c(z))   for each z supporting f
```
- Resolve conflicts using confidence and provenance
- Output final triples with support counts and confidence

**Why this is research, not engineering**:
- Formal cross-source linking model with calibrated scoring
- Source-calibrated confidence handling (structured evidence != text evidence)
- Abstention mechanism prevents catastrophic false merges
- Evidence aggregation with provenance tracking

**Novelty**: CONFIRMED — no existing work combines these in a unified domain-agnostic framework.

**Feasibility**: ~2 weeks implementation. Priority order:
1. Deterministic structured mapper (3 days)
2. Cross-source linker with features (3 days)
3. Threshold tuning + abstention (1 day)
4. Fact evidence aggregation (2 days)
5. Ablations (2 days)
6. Auto insurance transfer test (2 days)

---

### Idea 2 (BACKUP): Record-Entity Promotion with Composite Keys (REP-CK)

Simpler version — after standard extraction, post-hoc identification and promotion of record entities to first-class nodes. Less novel but faster to implement.

**Status**: Subsumed by SEAF-KG Stage 1 (deterministic structured mapping)

## Eliminated Ideas

| Idea | Phase Eliminated | Reason |
|------|:----------------:|--------|
| SGIE (Schema-Guided) | Phase 4 | Similar to PARSE; reviewer says deterministic > LLM for structured |
| TPIF (Two-Phase Identity-First) | Phase 2 | KGGen already does entity->relation decomposition |
| Adaptive Prompt Routing | Phase 2 | Too shallow — "just engineering" per reviewer |
| HETI (Hierarchical Types) | Phase 2 | Incorporated into SEAF-KG Stage 2, not standalone |
| Record-Aware Chunking alone | Phase 2 | Infrastructure, not contribution — built into Stage 1 |

## Experiment Plan

### Experiment 1: Main End-to-End Comparison (Flood Insurance)

**Baselines**:
1. **Text-only**: Convert CSV rows to templated text, run same Open IE pipeline (sanity check)
2. **Deterministic-only**: Schema-to-graph for CSVs, no text extraction
3. **Naive union**: Independent extraction from both sources, no linking
4. **Current pipeline**: Zone 2 Open IE (baseline — what we have now)
5. **SEAF-KG**: Full method

**Metrics**:
- Entity identity preservation: P/R/F1 on reconstructing distinct claim/policy entities
- Triple quality: precision/recall on sampled triples (human annotation, ~100 samples)
- Cross-source link quality: P/R/F1 on record-to-rule links (annotate ~50 links)
- Queryability: Instance-centric queries ("What rules apply to records in zone A?")
- Riskine alignment: Name F1, BERTScore F1, Graph F1 (existing metrics)

### Experiment 2: Linker Ablation

Ablate features of the cross-source linker:
1. Lexical only
2. Lexical + type
3. Lexical + type + embedding
4. Full source-calibrated scorer
5. Full scorer + abstention

**Metric**: Entity linking P/R/F1, false merge rate

### Experiment 3: Structured Extraction Ablation

Compare:
1. Deterministic only (schema-to-graph)
2. LLM only (current Open IE on CSV chunks)
3. Hybrid (deterministic + LLM refinement)

**Metric**: Field-to-property accuracy, type assignment accuracy, triple count

### Experiment 4: Cross-Domain Transfer (Auto Insurance)

- Tune thresholds on flood insurance
- Evaluate on auto insurance with zero code changes
- Show linker works, structured mapper functions, method degrades gracefully

**Metric**: Same as Experiment 1 but on auto data

### Annotation Budget

| What | Size | Estimated Time |
|------|:----:|:--------------:|
| Entity identity gold set | 100 records | 2 hours |
| Triple quality sample | 100 triples | 2 hours |
| Cross-source links | 50 links | 1 hour |
| Instance-centric queries | 20 queries | 1 hour |

## Impact on Existing Pipeline

### What Changes

| Component | Current | After SEAF-KG |
|-----------|---------|---------------|
| Zone 1 chunking | Batches 4-7 CSV rows per chunk | Preserves record boundaries, adds record metadata |
| Zone 2 extraction | Same Open IE for all chunks | Source-detected dual path + canonical evidence format |
| Zone 2.5 entity resolution | Embedding-based clustering | Enhanced with cross-source linker features |
| Zone 3 ontology induction | SV-LOI on Open IE triples | SV-LOI on fused evidence (richer input) |
| Evaluation | Riskine alignment only | + identity preservation + link quality + queryability |

### What Stays the Same

- Zone 3 SV-LOI method (novel ontology induction) — still the primary Zone 3 contribution
- Domain-agnostic design principle — no Riskine in pipeline
- Evaluation against Riskine reference ontology
- Neo4j storage layer

## Paper Narrative

**Before**: "We propose SV-LOI for ontology induction and show it outperforms Leiden and RSI-LCR."

**After (enhanced)**: "We present SEAF-KG, a source-calibrated evidence fusion framework that constructs queryable, identity-grounded KGs from mixed structured and unstructured sources. Combined with SV-LOI ontology induction, the pipeline produces KGs where both individual records (claims, policies) AND semantic concepts (coverage rules, exclusions) are represented, linked, and queryable — all without domain-specific engineering."

## What NOT to Claim

- "Fully domain-agnostic" in the strong sense -> instead: "ontology-independent, demonstrated transfer across two insurance domains"
- "Novel multi-source pipeline" as main novelty -> instead: "source-calibrated evidence fusion with formal cross-source linking"
- "Composite keys solve identity" -> instead: "composite keys are one feature for entity linking, not the identity mechanism"
- "LLMs are necessary for structured data" -> instead: "deterministic extraction with optional LLM refinement"

## Reviewer Feedback Summary

**External review (GPT-5.4 via Codex)**: Novelty 6/10, Significance 7/10, Feasibility 6/10

**Key criticisms addressed**:
1. "Pipeline engineering" -> Reframed as formal evidence fusion framework with calibrated linker
2. "Composite keys are brittle" -> Keys are now one feature among many, not the identity mechanism
3. "Cross-source linker underspecified" -> Formal blocking + scoring + abstention with two thresholds
4. "Structured path overkill" -> Hybrid: deterministic first, LLM only for ambiguous cells
5. "Domain-agnostic claim needs evidence" -> Experiment 4 tests zero-change transfer to auto insurance

## Next Steps

- [ ] Implement deterministic structured mapper (Stage 1)
- [ ] Implement cross-source linker with feature scoring (Stage 3)
- [ ] Implement evidence aggregation (Stage 4)
- [ ] Run Experiment 1 (end-to-end comparison)
- [ ] Run Experiment 2 (linker ablation)
- [ ] Run Experiment 3 (structured extraction ablation)
- [ ] Run Experiment 4 (auto insurance transfer)
- [ ] Annotate gold standard samples for evaluation

## Sources

- [PARSE (2025)](https://arxiv.org/abs/2510.08623) — Schema optimization for entity extraction
- [KGGen (NeurIPS 2025)](https://arxiv.org/html/2502.09956v1) — Two-phase KG extraction from text
- [ODKE+ (2025)](https://arxiv.org/html/2509.04696v1) — Ontology-guided extraction
- [KARMA (2025)](https://arxiv.org/pdf/2502.06472) — Multi-agent KG enrichment
- [iText2KG (2024)](https://arxiv.org/html/2409.03284v1) — Incremental KG construction
- [SemTab 2025](https://sem-tab-challenge.github.io/2025/) — Tabular data to KG matching challenge
- [LLM-empowered KG Survey (2025)](https://arxiv.org/html/2510.20345v1) — Comprehensive survey
- [GraphRAG (2025)](https://arxiv.org/html/2507.03226v2) — Efficient KG construction + hybrid retrieval
- [CDL 2024 ER Masterclass](https://github.com/DerwenAI/cdl2024_masterclass) — Mixed data KG with entity resolution
- [Insurance KG Modeling](https://memgraph.com/blog/how-to-model-insurance-data-as-a-graph) — Insurance graph data modeling
- [NER + GenAI for Insurance](https://insurants.com/extraction-of-entities-from-unstructured-commercial-insurance-documents-leveraging-ner-and-genai-algorithms/) — Insurance entity extraction
