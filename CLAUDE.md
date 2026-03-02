# CLAUDE.md — CS584 AI Capstone

## Project Overview

**Goal:** Automated insurance ontology induction from NFIP flood insurance documents using a 4-zone LangGraph pipeline.

**Thesis:** LLM-based KG extraction (LLMGraphTransformer) produces free-form labels with no ontology structure. Our pipeline adds: structured chunking (Zone 1) → constrained Open IE (Zone 2) → Leiden community clustering for ontology induction (Zone 3) → Neo4j structured storage (Zone 4). Evaluated against Riskine insurance ontology (10 classes).

**Paper title:** *Schema-Evolving Knowledge Graphs for Insurance: Automated Ontology Induction with Cross-Domain Transfer Learning*

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
# Baseline pipeline (3-step, no induction — default/stable)
python3 baseline/pipeline.py --zone1

# Baseline pipeline with LLM ontology induction (use only for final comparison)
python3 baseline/pipeline.py --zone1 --induce-ontology

# Evaluate baseline
python3 baseline/eval.py --suffix zone1
python3 baseline/eval.py --suffix zone1 --riskine   # + Riskine P/R/F1

# HTML report
python3 evaluation/visualize_results.py --suffix zone1 --html

# Zone 1 chunking (re-chunk the PDF/CSV)
python3 zone1/ingestion.py

# Download FEMA data
python3 scripts/data_download.py
```

---

## Directory Structure

```
baseline/          pipeline.py, eval.py, ontology_induction.py, pdf_loader.py
zone1/             ingestion.py — section-aware hybrid chunking
zone2/             Open IE with few-shot prompting (IN PROGRESS)
zone3/             Leiden ontology induction (upcoming)
zone4/             Neo4j structured storage (upcoming)
evaluation/        riskine_eval.py, riskine_loader.py, visualize_results.py, compare_results.py
scripts/           data_download.py
config.py          all paths + credentials (loaded from .env)

data/flood/raw/pdf/                       SFIP PDF (primary source)
data/flood/raw/openfema/                  500-record policy + claims CSVs
data/flood/processed/zone1_chunks.json   69 Zone 1 section-aware chunks
data/riskine/schemas/                     10 Riskine class schemas (JSON)
data/results/                            all eval output JSONs + HTML
```

---

## Current Results (do not overwrite without intent)

| Run | Query Acc | Type Incon. | Riskine F1 | File |
|-----|:---------:|:-----------:|:----------:|------|
| Original 512-token | 35% | 8.0% | — | `baseline_eval_results_original.json` |
| Zone 1 (llama3.1:8b) | **50%** | 15.2% | **0.250** | `baseline_eval_results_zone1.json` |
| Zone 1 (qwen2.5:7b) | 35% | 7.9% | 0.11 | `baseline_eval_results_zone1_qwen.json` |

**These are the stable comparison baseline numbers.** Do not re-run `baseline/pipeline.py --zone1` (without being asked) as it clears Neo4j and overwrites results.

---

## Critical Rules

- **Never re-run the baseline pipeline** unless the user explicitly asks — it wipes Neo4j and overwrites `data/results/` eval files
- **`--induce-ontology` is OFF by default** — only use it for the final fair comparison after Zone 2–4 are complete
- Result files are named with a `--suffix` (e.g., `zone1`, `original`) — always pass the correct suffix to `eval.py` and `visualize_results.py`
- `config.py` loads all paths from `.env`; never hardcode paths in scripts

---

## Zone Status

| Zone | Status | What it does |
|------|:------:|---|
| Zone 1 | ✅ Done | Section-aware hybrid PDF chunking (τ=0.85 semantic merge) |
| Baseline | ✅ Done | LLMGraphTransformer → Neo4j, 20-task eval, Riskine alignment |
| Baseline+Induction | ✅ Done | `--induce-ontology` LLM label→Riskine mapping (opt-in) |
| Zone 2 | 🔄 In Progress | Few-shot Open IE: typed negation, numerical facts, procedures |
| Zone 2.5 | ⏳ Upcoming | Entity resolution: embed + cluster near-duplicate nodes |
| Zone 3 | ⏳ Upcoming | Leiden community detection → canonical ontology classes |
| Zone 4 | ⏳ Upcoming | Structured Neo4j storage with SUBCLASS_OF hierarchy |

---

## Riskine Ontology (10 Classes)

`Coverage · Product · Damage · Risk · Structure · Property · Person · Object · Organization · Address`

Schemas in `data/riskine/schemas/*.json`. Evaluation via embedding cosine similarity + LLM judge.

---

## Novel Pipeline Targets (for paper)

| Metric | Current Baseline | Novel Target |
|--------|:----------------:|:------------:|
| Query accuracy | 50% | > 75% |
| Type inconsistency | 15.2% | < 2% |
| Riskine F1 | 0.250 | > 0.75 |
| Entity duplication | ~0% | < 5% |
| CTR (cross-domain) | — | > 60% |
