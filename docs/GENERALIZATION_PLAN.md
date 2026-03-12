# Generalization Plan — CS584 AI Capstone

**Date:** March 9, 2026
**Authors:** Xiaofei Wang · Zechary Chou
**Status:** Approved — implementation starting Week 6

---

## 1. Problem: Domain Leakage

An audit of the Zone 2 pipeline (March 9, 2026) revealed that the reference ontology
(Riskine) had leaked from evaluation into the extraction pipeline itself, undermining
the project's core claim of domain-agnostic ontology induction.

### Specific violations found in `zone2/pipeline.py`:

| Component | What it does | Why it's a problem |
|-----------|-------------|-------------------|
| `ANCHORS` (9 entries) | Injects nodes like `("NFIP Policy", "Product")`, `("Flood", "Risk")`, `("FEMA", "Organization")` | Hardcodes Riskine class labels directly into the extraction graph. Running on auto insurance would still create a "Flood" node labeled "Risk." |
| `RELATION_ROLE_MAP` (15 entries) | Maps e.g. `COVERS → (Coverage, Property)`, `MUST_NOTIFY → (Person, Organization)` | Assumes Riskine's domain model. The pipeline assigns Riskine classes during extraction, not as a post-hoc evaluation. |
| `EVAL_SEEDS` (9 entries) | Forces relations like `EXCLUDED_FROM`, `MUST_NOTIFY`, `HAS_DEADLINE` | Designed to match the 20 hand-written Cypher evaluation queries, not to capture the document's actual structure. |
| `label_nodes()` step | Applies Riskine labels to nodes after extraction | Mixes evaluation into the pipeline. `db.labels()` returns Riskine classes because they were injected, not induced. |
| `FEW_SHOT_PAIRS` (13 pairs) | Uses exact SFIP text: earth movement exclusions, ICC coverage, proof of loss deadlines | Teaches the model NFIP-specific extraction patterns. On an auto insurance PDF, these examples are irrelevant noise. |
| `_BOOTSTRAP_SECTION_GROUPS` | Stratified sampling with keywords: `"coverage a"`, `"proof of loss"`, `"icc"` | SFIP-specific section names. A health insurance PDF has none of these sections. |

### Impact on reported metrics:

The Zone 2 Riskine F1 of **0.874** is inflated. The `label_nodes()` step guarantees
9/10 Riskine classes appear in `db.labels()` by injecting anchor nodes. The actual
extraction-only F1 (without anchors) would be ~0.35-0.45.

---

## 2. Design Principle: Pipeline ≠ Evaluation

```
CORRECT separation:
┌──────────────────────────────────────────────────────────┐
│ PIPELINE (zone1/ → zone2/ → zone3/ → zone4/)            │
│                                                          │
│ Input:  ANY insurance documents (PDF, CSV)               │
│ Output: Induced ontology with classes + SUBCLASS_OF      │
│                                                          │
│ Must NOT contain: Riskine class names, SFIP text,        │
│ flood-specific keywords, anchor nodes                    │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼ (output ontology)
┌──────────────────────────────────────────────────────────┐
│ EVALUATION (evaluation/)                                 │
│                                                          │
│ Compares: induced ontology classes → Riskine classes     │
│ Metrics:  Precision, Recall, F1 via embedding + LLM judge│
│                                                          │
│ Riskine lives HERE and ONLY here.                        │
└──────────────────────────────────────────────────────────┘
```

The pipeline should produce an ontology that is then *compared to* Riskine,
not *built from* Riskine.

---

## 3. Zone 2 Redesign: Domain-Agnostic Open IE

### 3.1 What to REMOVE

| Item | File | Action |
|------|------|--------|
| `EVAL_SEEDS` | zone2/pipeline.py | Delete entirely |
| `RELATION_ROLE_MAP` | zone2/pipeline.py | Delete entirely |
| `ANCHORS` | zone2/pipeline.py | Delete entirely |
| `label_nodes()` | zone2/pipeline.py | Delete node from LangGraph pipeline |
| `_apply_role_labels()` | zone2/pipeline.py | Delete function |
| `_apply_anchor_labels()` | zone2/pipeline.py | Delete function |
| `_BOOTSTRAP_SECTION_GROUPS` | zone2/pipeline.py | Replace with generic keywords |
| `FEW_SHOT_PAIRS` (13 SFIP pairs) | zone2/prompts.py | Replace with 4-5 synthetic generic pairs |
| `BOOTSTRAP_PROMPT` categories | zone2/prompts.py | Generalize category names |
| `SYSTEM_PROMPT_TEMPLATE` named concepts | zone2/prompts.py | Remove "Building", "Flood" etc. |

### 3.2 What to KEEP (already general)

| Item | Why it's general |
|------|-----------------|
| `bootstrap_vocab()` concept | LLM reads sample chunks → proposes relation types. This IS general. |
| `GENERIC_BLACKLIST` | Filtering HAS/IS/CONTAINS applies to any domain. |
| `RELATION_NORMALIZATIONS` | Synonym consolidation is general (entries need generalization). |
| Entity resolution (zone 2.5) | Embedding-based dedup is domain-agnostic. |
| JSON parsing, sanitization, confidence filtering | Infrastructure, not domain logic. |
| `keep_triple()` | Quality filter is domain-agnostic. |

### 3.3 New Components

**a) Entity Type Bootstrapping**

Current Zone 2 only bootstraps relation types. Add entity type bootstrapping:

```python
ENTITY_BOOTSTRAP_PROMPT = """You are a knowledge graph expert.
Read these sample passages from an insurance domain document.
Propose 8-15 PascalCase entity TYPE names that would be useful
for organizing the concepts in these passages.

Sample passages:
{samples}

Respond with ONLY a JSON array of strings:
["InsurancePolicy", "CoverageType", "ExcludedPeril", ...]"""
```

The LLM reads the actual documents and proposes entity types — different
docs yield different types. Flood docs → FloodZone, InsuredBuilding.
Auto docs → Vehicle, DrivingRecord, CollisionCoverage.

**b) Synthetic Few-Shot Examples**

Replace 13 SFIP-specific pairs with 4-5 generic insurance patterns:

| Pattern | Example (generic, not from any real document) |
|---------|-----------------------------------------------|
| Definition | "The term 'peril' means any event that may result in a loss." |
| Coverage | "This policy covers damage to the insured dwelling up to $300,000." |
| Exclusion | "This policy does not cover losses caused by war or intentional acts." |
| Obligation | "The insured must report any claim within 30 days of the loss." |
| Limit | "The maximum deductible for this coverage type is $5,000." |

These patterns exist in EVERY insurance LOB. No SFIP text, no NFIP references.

**c) Generic Stratified Sampling**

```python
# OLD (flood-specific):
["coverage a", "building property", "proof of loss", "icc"]

# NEW (any insurance LOB):
["coverage", "insure", "covered", "protect"],
["exclusion", "not covered", "does not", "except"],
["definition", "means", "defined as", "term"],
["claim", "loss", "notify", "report", "file"],
["condition", "requirement", "must", "shall"],
["limit", "maximum", "deductible", "amount"],
["period", "effective", "expir", "cancel", "renew"],
["premium", "payment", "cost", "rate"],
```

---

## 4. Zone 3 Redesign: Bottom-Up Ontology Induction

### 4.1 Why Leiden gets its purpose back

The original Zone 3 design assumed Zone 2 would produce ~44 free-form labels with
type inconsistency. The domain-coupled Zone 2 v1 solved this "too well" by hardcoding
10 Riskine labels, leaving Leiden with nothing to cluster.

With the generalized Zone 2, extraction produces 40-80 raw entity types with free-form
names (because no RELATION_ROLE_MAP forces them into Riskine classes). This is exactly
the scenario Leiden was designed for.

### 4.2 Pipeline

```
Zone 2 output: ~60+ entity nodes with free-form names
    │
    ▼
Step 3a: Build entity-similarity graph
    │   For each pair of entities, compute composite weight:
    │   w = 0.5 × embedding_similarity
    │     + 0.3 × structural_similarity (shared relation types)
    │     + 0.2 × co_occurrence (appear in same chunks)
    │
    ▼
Step 3b: Leiden community detection → 15-30 clusters
    │   Each cluster = a candidate ontology class
    │   Resolution parameter tuned for target cluster count
    │
    ▼
Step 3c: LLM names each cluster (NO Riskine reference)
    │   Input: ["Building Coverage", "Contents Coverage", "Coverage A", "Coverage B"]
    │   Output: "InsuranceCoverage"
    │   The LLM INVENTS the name — it does NOT choose from Riskine classes
    │
    ▼
Step 3d: LLM proposes SUBCLASS_OF between clusters
    │   Input: All cluster names + member entities
    │   Output: "FloodCoverage SUBCLASS_OF InsuranceCoverage"
    │
    ▼
Step 3e: Write ontology to Neo4j
    │   - Ontology class nodes (one per cluster)
    │   - INSTANCE_OF edges (entity → class)
    │   - SUBCLASS_OF edges (class → class)
    │
    ▼
EVALUATION (evaluation/riskine_eval.py — separate from pipeline):
    Compare induced class names → Riskine class names
    via embedding similarity + LLM judge
```

### 4.3 What makes this general

The SAME pipeline produces different ontologies for different LOBs:

| LOB | Induced classes (examples) |
|-----|---------------------------|
| Flood | FloodCoverage, PropertyExclusion, ClaimsProcedure, InsuredStructure |
| Auto | VehicleCoverage, LiabilityLimit, AccidentClaim, InsuredVehicle |
| Health | MedicalCoverage, PreExistingCondition, ProviderNetwork |

Then cross-domain transfer (Zone 3b) asks: do flood and auto share a common
"Coverage" superclass? A common "Exclusion" pattern? That question can ONLY
be answered honestly if the pipeline doesn't already know the answer.

---

## 5. Impact on Results

### What we lose

| Metric | Zone 2 v1 (inflated) | Zone 2 v2 (honest) | Why |
|--------|:--------------------:|:------------------:|-----|
| Riskine F1 | 0.874 | ~0.35-0.45 | No anchor nodes, no RELATION_ROLE_MAP |
| Type inconsistency | 0.0% | ~15-20% | No hardcoded labels — Zone 3 must fix this |

### What we gain

| Benefit | Description |
|---------|-------------|
| True generality | Same code runs on flood, auto, health with zero changes |
| Honest evaluation | Riskine F1 reflects actual induction quality |
| Meaningful Zone 3 | Leiden has 40-80 labels to cluster, not 10 trivial ones |
| Cross-domain transfer | CTR/PTR comparison is genuine (pipeline didn't cheat) |
| Stronger paper | "General ontology induction" > "flood insurance KG builder" |
| F-03 finding validated | Extraction sparsity is a real problem to solve, not masked by hardcoding |

### Expected progression (honest)

| Metric | Baseline | Zone 1 | Zone 2 v2 | Zone 3 |
|--------|:--------:|:------:|:---------:|:------:|
| Query accuracy | 35% | 50% | ~45-55% | > 65% |
| Type inconsistency | 8% | 15.2% | ~15-20% | < 5% |
| Riskine F1 | — | 0.250 | ~0.35-0.45 | > 0.55 |

Each zone shows clear, attributable improvement — a better story for the paper.

---

## 6. Implementation Timeline

| Week | Task | Key Deliverable |
|------|------|-----------------|
| **6** | Generalize Zone 2 | Remove ANCHORS/RELATION_ROLE_MAP/EVAL_SEEDS, synthetic few-shot, entity type bootstrap |
| **7** | Implement Zone 3 | Leiden clustering, LLM cluster naming, SUBCLASS_OF inference |
| **8** | Auto insurance run | Run identical pipeline on auto data, zero code changes |
| **9** | Cross-domain transfer | CTR/PTR measurement, ontology merging analysis |
| **10-11** | Paper + Demo | Streamlit demo, human annotation, 10-15 page report |

---

## 7. Files to Modify

| File | Changes |
|------|---------|
| `zone2/pipeline.py` | Remove EVAL_SEEDS, RELATION_ROLE_MAP, ANCHORS, label_nodes(), _apply_*. Add entity type bootstrapping. Replace _BOOTSTRAP_SECTION_GROUPS. |
| `zone2/prompts.py` | Replace 13 SFIP FEW_SHOT_PAIRS with 4-5 synthetic. Generalize BOOTSTRAP_PROMPT and SYSTEM_PROMPT_TEMPLATE. |
| `zone3/pipeline.py` | NEW: Leiden clustering, LLM naming, SUBCLASS_OF, Neo4j ontology layer. |
| `evaluation/riskine_eval.py` | Adapt to compare Zone 3's INDUCED class names (not hardcoded Riskine labels). |
| `CLAUDE.md` | Updated with generalization principle and revised targets. |
| `config.py` | Add auto insurance data paths. |

---

*Approved: March 9, 2026*
