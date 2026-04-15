# Idea Discovery Report

**Direction**: Ontology hierarchy depth — enriching flat induced ontologies with subclasses
**Date**: 2026-04-14
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → refine

## Executive Summary

Current ontology induction (SV-LOI) produces 10 flat classes with mostly incorrect IS-A edges (e.g., Person SUBCLASS_OF Coverage). The core problem is **ontological heterogeneity**: induced classes contain structurally distinct subgroups that should be separate subclasses. We propose **Structural Ontological Heterogeneity Detection (SOHD)** — a principled method to detect when a class contains distinct subgroups via relation profile analysis, then split and create IS-A edges. This addresses a real gap: no existing method uses within-class structural analysis to deepen induced ontologies.

## Literature Landscape

### Key Papers (Taxonomy/Hierarchy Induction)

| Paper | Venue | Year | Key Technique | Relevance |
|-------|-------|------|---------------|-----------|
| Chain-of-Layer | CIKM | 2024 | Layer-by-layer LLM taxonomy construction | Architecture inspiration |
| AutoSchemaKG | arXiv | 2025 | Neighbor-context LLM conceptualization | Variant B baseline |
| OLLM | NeurIPS | 2024 | End-to-end LLM ontology learning | Variant D baseline |
| Pietrasik & Reformat | ESWC | 2020 | Relation co-occurrence for taxonomy induction | Signal 1 inspiration |
| Ren & Paulheim | CIKM | 2022 | Within-relation entity clustering for sub-relations | **Dual operation** |
| LLMs4Life | EKAW | 2024 | Re-prompting + ontology reuse for depth | Hierarchy deepening |
| SI-LLM | arXiv | 2025 | Schema inference from tabular data | Tabular hierarchy |
| GeTT | ESWC | 2025 | Iterative LLM taxonomy from tables | Tabular hierarchy |
| TaxoAdapt | ACL | 2025 | Iterative width+depth expansion | Adaptive deepening |
| Doubly-Checked | CCKS | 2024 | Bidirectional IS-A verification | Validation method |
| OnEFET | KDD | 2024 | Instance-enriched ontology for entity typing | Instance-driven |
| Refining Wikidata | CIKM | 2024 | LLM + graph mining taxonomy cleanup | Deep-level accuracy |

### Key Gaps Identified

1. **No one combines structural (relation-profile) signals with LLM validation for hierarchy deepening**
2. **Within-class distributional splitting is the dual of fine-grained relation discovery (CIKM 2022) but has never been applied to ontology classes**
3. **Insurance domain ontology induction is entirely unexplored** — all existing insurance ontologies (Riskine, ACORD) are manually built
4. **No principled criterion exists for "when to split a class into subclasses"** — current methods either always split (forced depth) or never split (flat)

### Insurance Ontology Standards

- **Riskine**: 10 flat classes (Coverage, Product, Damage, Risk, Structure, Property, Person, Object, Organization, Address)
- **ACORD**: ~1000 entities, 2-3 level depth, proprietary
- **FIBO**: 700+ financial classes, 4-5 levels (banking-focused)
- Insurance ontologies are typically flat (breadth > depth). Building data-driven 3-4 level hierarchy is a genuine contribution.

## Ranked Ideas

### Idea 1: SOHD — Structural Ontological Heterogeneity Detection — RECOMMENDED

**Status**: Refined after GPT-5.4 review (initial score 4/10 → refined to address all weaknesses)

**Scientific Question**: *When is an induced ontology class ontologically heterogeneous, and how do we detect and resolve it using only structural KG evidence?*

**Method**:
1. Start with flat classes from existing induction (10 classes from SV-LOI)
2. For each class with N > threshold members:
   a. Compute per-instance relation profile vectors (binary: participates in relation r)
   b. Test heterogeneity via spectral clustering on relation profile matrix
   c. Criterion: silhouette score > 0.3 AND Jensen-Shannon divergence between subgroup profiles > threshold
   d. If heterogeneous: promote subgroups to subclasses
3. Validate IS-A: subgroup's defining relations should be a *proper subset* of parent's defining relations (plus distinguishing relations)
4. Optional: LLM names subclasses for interpretability
5. Recurse on new subclasses until no more splits are evidence-supported
6. Enforce DAG + bidirectional LLM verification

**Novelty**:
- First principled method for detecting ontological heterogeneity in induced KG classes
- Dual of fine-grained relation discovery (Ren & Paulheim, CIKM 2022): they split relations by entity types; we split classes by relation profiles
- Information-theoretic split criterion prevents forced/artificial hierarchy

**Evaluation Plan (6 ablations)**:
- A: Flat baseline (current SV-LOI, no hierarchy deepening)
- B: LLM-only hierarchy (Chain-of-Layer prompting, no structural signals)
- C: Relation-signature subsumption only (Pietrasik-style)
- D: **SOHD without LLM naming** (core structural method)
- E: **SOHD with LLM naming** (full method)
- F: SOHD + entity-type seeds (ceiling ablation, to quantify leakage)

**Metrics**:
- Member-based F1 vs Riskine (current: 0.234)
- Name-based F1 vs Riskine (current: 0.000)
- Hierarchy depth (current: 1-2, target: 3-4)
- OntoQA structural metrics (depth, breadth, inheritance richness)
- Hierarchy-aware BDM (Balanced Distance Metric)
- IS-A edge precision (human evaluation of sample)
- Query accuracy on 20 insurance questions (current: 95%)

**Multi-domain evaluation**:
- Primary: Flood insurance KG (7,415 entities, 1,749 relationships)
- Secondary: Auto/mobile/home insurance (same pipeline, different LOB)
- Tertiary: Public KG benchmark from LLMs4OL (generalization test)

**Estimated effort**: ~20h implementation, ~10h evaluation

**Reviewer feedback addressed**:
| Weakness | How addressed |
|----------|--------------|
| Too heuristic/pipeline-like | Formalized with JS-divergence + silhouette criteria |
| Signal 2 leakage | Dropped from core method; only in ceiling ablation (F) |
| Single domain | Flood + auto/mobile/home + LLMs4OL benchmark |
| LLM-as-oracle | Core method (D) runs WITHOUT LLM; LLM is optional naming |
| Forced hierarchy | Information-theoretic criterion prevents artificial splits |
| Evaluation narrow | 7 ablations + 7 metrics + human eval + multi-domain |

**Target venues**: ESWC 2027, ISWC 2026, K-CAP 2027, WWW workshop

---

### Idea 2: Multi-Signal Ensemble (MSIHC) — BACKUP

**Status**: Original idea before refinement. Still viable as extended version.

**Method**: Ensemble 3 signals (relation signatures + entity-type seeds + distributional clustering) with voting + LLM layer-by-layer validation.

**Why backup**: Reviewer found it "too pipeline/heuristic." SOHD is the refined, focused version. MSIHC could be the extended journal version with all 3 signals.

**Novelty**: CONFIRMED — no existing paper combines all 3 signals.

---

### Idea 3: RelSig + Chain-of-Layer — ABLATION BASELINE

**Method**: Combine Pietrasik's relation signatures with CoL prompting.

**Role in paper**: Ablation C — tests whether structural subsumption alone is enough.

---

### Idea 4: Entity-Type Seeded Hierarchy — CEILING ABLATION

**Method**: Reuse Zone 2 entity_types as subclass candidates.

**Role in paper**: Ablation F — measures ceiling when extraction types are available. Also tests leakage.

---

### Idea 5: Bidirectional Hierarchy — ELIMINATED

**Why killed**: Reconciliation logic is complex with unclear benefit over SOHD. Top-down + bottom-up creates implementation complexity without clear scientific contribution.

## Eliminated Ideas

| Idea | Phase Killed | Reason |
|------|-------------|--------|
| TabOnt (schema from CSV) | Phase 2 | Limited novelty, depends on CSV quality |
| XferHier (cross-domain transfer) | Phase 2 | Requires full pipeline on additional domains first |
| ODP-Guide (design patterns) | Phase 2 | Too domain-specific, limited generalization |
| HyperHier (hyperbolic embeddings) | Phase 2 | Complex math, unclear improvement over simpler methods |
| BiHier (bidirectional) | Phase 4 | Complex reconciliation, no clear advantage |

## Experiment Plan

### Phase 1: Implement SOHD core (8h)
- [ ] Add `detect_heterogeneity()` function to sv_loi.py
- [ ] Implement spectral clustering on relation profile matrix
- [ ] Implement JS-divergence + silhouette criterion
- [ ] Add `split_class()` function that creates SUBCLASS_OF edges
- [ ] Add recursive deepening with depth limit

### Phase 2: Implement ablation baselines (6h)
- [ ] Ablation B: Chain-of-Layer LLM-only hierarchy (adapt CoL prompting for local LLM)
- [ ] Ablation C: Relation-signature subsumption only
- [ ] Ablation F: Entity-type seed ceiling

### Phase 3: Run on flood insurance data (4h)
- [ ] Run all 6 configurations (A-F)
- [ ] Compute all 7 metrics for each
- [ ] Generate comparison table

### Phase 4: Multi-domain evaluation (6h)
- [ ] Run on auto/mobile/home insurance data
- [ ] Run on LLMs4OL benchmark (if feasible)
- [ ] Cross-domain consistency analysis

### Phase 5: Human evaluation (4h)
- [ ] Sample 50 IS-A edges from SOHD output
- [ ] Have domain expert (Dr. Kang?) evaluate correctness
- [ ] Compute IS-A precision

### Phase 6: Write paper (8h)
- [ ] Introduction + problem formalization
- [ ] Method section with algorithm
- [ ] Experiments + ablation analysis
- [ ] Related work positioning

## Next Steps

1. **Implement SOHD** — add heterogeneity detection to `zone3/sv_loi.py`
2. **Run ablations** on flood insurance data
3. **Evaluate** with full metric suite
4. **Expand** to auto/mobile/home insurance
5. **Write paper** targeting ESWC 2027 or ISWC 2026
