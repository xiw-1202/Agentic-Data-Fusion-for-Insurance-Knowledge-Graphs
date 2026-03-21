# CLAUDE.md — CS584 AI Capstone

## Project Overview

**Goal:** Build a general-purpose, domain-agnostic pipeline that automatically induces ontologies from ANY insurance Line of Business (flood, auto, health, liability) — without manual ontology engineering or hardcoded domain knowledge.

**Thesis:** LLM-based KG extraction (LLMGraphTransformer) produces free-form labels with no ontology structure. Our pipeline adds: structured chunking (Zone 1) → domain-agnostic Open IE with bootstrapped schema (Zone 2) → bottom-up ontology induction via Leiden community detection (Zone 3) → structured Neo4j storage (Zone 4). The pipeline never sees the reference ontology — Riskine is used ONLY for evaluation. Generality is proven by running the identical pipeline on flood and auto insurance with zero code changes.

**Paper title:** *Schema-Evolving Knowledge Graphs for Insurance: Automated Ontology Induction with Cross-Domain Transfer Learning*

---

## ⚠️ CRITICAL DESIGN PRINCIPLE: No Domain Leakage

**Riskine (the reference ontology) must NEVER appear in the pipeline code.**
It belongs ONLY in `evaluation/riskine_eval.py`.

The pipeline must produce an ontology that is then *compared to* Riskine, not *built from* Riskine. The following are PROHIBITED in zone2/, zone3/, zone4/:
- Hardcoded Riskine class names (Coverage, Product, Damage, Risk, etc.)
- Anchor nodes that inject reference ontology entities
- Relation-to-class mappings that assume the target schema
- Few-shot examples using actual SFIP/NFIP document text
- Section keywords specific to any single insurance LOB

See `docs/GENERALIZATION_PLAN.md` for the full rationale and audit.

---

## Environment

```
Neo4j AuraDB   — credentials in .env (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)
Ollama         — http://localhost:11434  |  default model: llama3.1:8b
Python         — 3.11+, run from project root
```

All scripts must be run from the **project root** (`~/Documents/School/Emory/CS584_AI_Capstone/`).

---

## Key Commands

```bash
# Baseline pipeline (comparison — uses LLMGraphTransformer, unconstrained)
python3 baseline/pipeline.py --zone1
python3 baseline/eval.py --suffix zone1 --riskine

# Zone 2 pipeline — domain-agnostic Open IE (bootstrapped schema, no hardcoding)
python3 zone2/pipeline.py
python3 zone2/pipeline.py --model qwen2.5:7b

# Zone 3 pipeline — Leiden ontology induction (upcoming)
python3 zone3/pipeline.py

# Evaluate any zone
python3 baseline/eval.py --suffix zone2 --riskine
python3 baseline/eval.py --suffix zone3 --riskine

# Zone 1 chunking (re-chunk any PDF/CSV)
python3 zone1/ingestion.py

# Download FEMA data
python3 scripts/data_download.py
```

---

## Directory Structure

```
baseline/          pipeline.py, eval.py, ontology_induction.py, pdf_loader.py
zone1/             ingestion.py — section-aware hybrid chunking (general)
zone2/             Domain-agnostic Open IE (pipeline.py, prompts.py, entity_resolution.py)
zone3/             Leiden ontology induction — bottom-up class discovery (upcoming)
zone4/             Structured Neo4j storage with SUBCLASS_OF hierarchy (upcoming)
evaluation/        riskine_eval.py, riskine_loader.py, visualize_results.py, compare_results.py
scripts/           data_download.py
config.py          all paths + credentials (loaded from .env)

data/flood/        NFIP flood insurance data (primary LOB)
data/auto/         Auto insurance data (cross-domain transfer LOB)
data/riskine/      Reference ontology (EVALUATION ONLY — never imported by pipeline)
data/results/      All eval output JSONs + HTML
```

---

## Current Results (do not overwrite without intent)

| Run | Query Acc | Type Incon. | Riskine F1 | File | Notes |
|-----|:---------:|:-----------:|:----------:|------|-------|
| Baseline (512-tok) | 35% | 8.0% | — | `baseline_eval_results_original.json` | |
| Zone 1 (llama3.1:8b) | **50%** | 15.2% | **0.250** | `baseline_eval_results_zone1.json` | |
| Zone 1 (qwen2.5:7b) | 35% | 7.9% | 0.11 | `baseline_eval_results_zone1_qwen.json` | |
| Zone 2 v1 (domain-coupled) | 50% | 0.0% | 0.874 | `baseline_eval_results_zone2.json` | ⚠️ Inflated — had ANCHORS + RELATION_ROLE_MAP |
| Zone 2 v2 (domain-agnostic) | 35% | 0.0% | ~0.000 | `baseline_eval_results_zone2.json` | Honest baseline — no anchors |
| Zone 3 (Leiden, 8b) | 35% | 17.4% | **0.171** | `baseline_eval_results_zone3.json` | P=0.120, R=0.300; threshold=0.40, resolution=0.6 |
| Zone 3 (Leiden, 70b) | 30% | **2.9%** | 0.071 | `baseline_eval_results_zone3_70b.json` | Turing GPU; descriptive names → low Riskine F1 (see F-07) |

**Zone 2 v1 Riskine F1 of 0.874 is NOT a real result.** It was achieved by injecting 9 Riskine-labeled anchor nodes directly into the graph. Zone 2 v2 removes all domain leakage. Zone 3 70B achieved **2.9% type inconsistency** — the real structural win — but Riskine F1 dropped because 70B produces descriptive compound class names (`PropertyCoverageComponent`, `InsuranceParty`) that don't match Riskine's simple convention-names (`Coverage`, `Person`). See F-07.

---

## Critical Rules

- **Never re-run the baseline pipeline** unless the user explicitly asks — it wipes Neo4j and overwrites `data/results/` eval files
- **Riskine is EVALUATION ONLY** — never import riskine_loader or riskine class names in zone2/, zone3/, zone4/ code
- **Pipeline must be domain-agnostic** — the same code must run on flood AND auto with zero changes
- `config.py` loads all paths from `.env`; never hardcode paths in scripts
- Result files are named with a `--suffix` — always pass the correct suffix to `eval.py`

---

## Zone Status

| Zone | Status | What it does |
|------|:------:|---|
| Zone 1 | ✅ Done | Section-aware hybrid PDF chunking (τ=0.85 semantic merge) — general |
| Baseline | ✅ Done | LLMGraphTransformer → Neo4j, 20-task eval — comparison only |
| Zone 2 v1 | ⚠️ Domain-coupled | Few-shot Open IE with SFIP-specific prompts + Riskine anchors — NEEDS REWORK |
| Zone 2 v2 | ✅ Done | Domain-agnostic: bootstrapped entity+relation types, synthetic few-shot, no anchors |
| Zone 2.5 | ✅ Done | Entity resolution: embed + cluster near-duplicate nodes — general |
| Zone 3 | ✅ Done | Leiden community detection on ENTITIES → bottom-up ontology classes + SUBCLASS_OF |
| Zone 4 | ⏳ Upcoming | Structured Neo4j storage with ontology layer |

---

## Pipeline Architecture (Generalized)

```
ZONE 1: Ingestion        ZONE 2: Open IE (general)       ZONE 3: Ontology Induction      ZONE 4: Storage
─────────────────        ────────────────────────         ──────────────────────────       ──────────────
ANY insurance docs       Bootstrap from docs:             Leiden on extracted entities:    3-layer Neo4j:
PDF + CSV           →      LLM proposes entity types →     Build similarity graph     →    Ontology classes
Section-aware chunk        LLM proposes relation types     Leiden → 15-30 clusters         SUBCLASS_OF
τ=0.85 merge               Synthetic few-shot (generic)    LLM names each cluster          Entity instances
                           Extract (s,r,o,confidence)      LLM proposes SUBCLASS_OF
                           Entity resolution               NO reference ontology used

                           ┌──────────────────────────┐
                           │ EVALUATION (separate)     │
                           │ Induced ontology → Riskine│
                           │ P / R / F1 / CTR / PTR   │
                           └──────────────────────────┘
```

---

## Zone 2 Generalization — What Changed

### REMOVED (domain-specific):
- `EVAL_SEEDS` — 9 hand-picked relation types matching 20 Cypher queries
- `RELATION_ROLE_MAP` — hardcoded relation→Riskine-class mapping
- `ANCHORS` — 9 injected Riskine-labeled nodes ("NFIP Policy"→Product, "Flood"→Risk, etc.)
- `label_nodes()` pipeline step — Riskine labeling was pipeline, not evaluation
- `_BOOTSTRAP_SECTION_GROUPS` with SFIP keywords ("coverage a", "proof of loss")
- `FEW_SHOT_PAIRS` with exact SFIP text (13 flood-specific examples)

### REPLACED WITH (domain-agnostic):
- Entity type bootstrapping — LLM reads YOUR docs, proposes entity types
- Relation type bootstrapping — LLM reads YOUR docs, proposes relation types (already existed)
- Synthetic few-shot examples — 4-5 patterns: definition, coverage, exclusion, obligation, limit
  (uses generic insurance language, not any specific LOB text)
- Generic section sampling keywords: "coverage", "exclusion", "definition", "claim", "limit"

### KEPT (already general):
- `bootstrap_vocab()` concept, `GENERIC_BLACKLIST`, entity resolution, JSON parsing
- `RELATION_NORMALIZATIONS` (generalized: synonym consolidation applies to any LOB)

---

## Riskine Ontology (10 Classes) — EVALUATION REFERENCE ONLY

`Coverage · Product · Damage · Risk · Structure · Property · Person · Object · Organization · Address`

Schemas in `data/riskine/schemas/*.json`. Used ONLY in `evaluation/riskine_eval.py`.
Never imported by zone2/, zone3/, or zone4/ code.

---

## Novel Pipeline Targets (for paper)

| Metric | Baseline | Zone 1 | Zone 2 (general) | Zone 3 (Leiden) | Notes |
|--------|:--------:|:------:|:-----------------:|:---------------:|-------|
| Query accuracy | 35% | 50% | ~45-55% | > 65% | Flood-specific eval tasks |
| Type inconsistency | 8% | 15.2% | ~15-20% | < 5% | Zone 3 is the fix |
| Riskine F1 | — | 0.250 | ~0.35-0.45 | > 0.55 | Honest — no anchor injection |
| Entity duplication | ~0% | ~0% | ~0% | < 5% | Grows at 150K scale |
| CTR (cross-domain) | — | — | — | > 60% | Same pipeline on auto data |
| PTR (cross-domain) | — | — | — | > 50% | Shared properties flood↔auto |

**These targets are honest.** Zone 2 v1's 0.874 F1 was inflated by anchor injection.
The real test is: does Zone 3 Leiden discover classes that align with Riskine WITHOUT ever seeing Riskine?

---

## Empirical Findings (documented in docs/FINDINGS.md)

| Finding | Summary | Impact on Pipeline |
|---------|---------|-------------------|
| F-01 | Llama 3.1 8B ignores dollar amounts as triple objects | Motivates regex post-extraction numeric linking |
| F-02 | Few-shot pairs from source text cause global interference | Motivates synthetic (not source-derived) few-shot examples |
| F-03 | 8B model extracts exactly 1 triple/chunk regardless of prompting | Motivates multi-pass extraction or chunk decomposition |
| F-04 | Vocab compliance > extraction volume for query accuracy | Validates bootstrapped vocabulary approach |
| F-05 | Entity resolution threshold is model-dependent | 0.90 works for 8B; needs adaptation for larger models |
| F-06 | Zone 2 v1's 0.874 F1 was domain leakage (Riskine anchor nodes) | Removed in v2; honest baseline is ~0.000 |
| F-07 | Larger models produce descriptive compound class names (70B: `PropertyCoverageComponent`) that don't align with reference ontology naming conventions (Riskine: `Coverage`), while smaller models accidentally produce shorter names that embed closer to Riskine | Proves Riskine F1 is a poor metric for domain-agnostic ontology evaluation — rewards naming convention similarity not semantic correctness. The correct metric is cross-domain CTR/PTR. |

---

## Timeline (Weeks 6-11)

| Week | Task | Deliverable |
|------|------|-------------|
| 6 | Generalize Zone 2: remove all domain hardcoding | zone2/ v2 with synthetic few-shot |
| 7 | Implement Zone 3: Leiden + LLM naming + SUBCLASS_OF | zone3/pipeline.py |
| 8 | Run identical pipeline on auto insurance (zero changes) | data/auto/ results |
| 9 | Cross-domain transfer: CTR/PTR measurement | Transfer evaluation |
| 10-11 | Paper, Streamlit demo, human annotation | Final deliverables |
