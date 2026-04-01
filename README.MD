# SEAF-KG: Schema-Evolving Agentic Framework for Insurance Knowledge Graphs

**Automated Ontology Induction with Cross-Domain Transfer Learning**

Xiaofei Wang, Zechary Chou
CS 584 AI Capstone, Emory University, Spring 2026
Advisors: Dr. Eugene Agichtein (Emory), Dr. Yingying Kang (Assurant)

---

## Problem

Insurance companies entering new Lines of Business face a **3-6 month ontology engineering bottleneck** per domain. Existing tools like LangChain's `LLMGraphTransformer` extract KG triples but produce flat, duplicated, schema-less graphs. We automate the full pipeline from raw documents to a structured, hierarchical ontology -- with zero manual engineering.

## Solution

A 4-zone pipeline that is **fully domain-agnostic** -- the same code runs on flood, auto, or any insurance LOB with no changes:

```
Zone 1              Zone 2                   Zone 3                    Zone 4
Ingestion           Open IE                  Ontology Induction        Storage
-----------         ----------------         ----------------------    ---------------
PDF + CSV     -->   Bootstrap vocab    -->   SV-LOI (novel):     -->   Neo4j:
Section-aware       from documents           - Class discovery          OntologyClass
chunking            LLM extraction           - Entity typing            SUBCLASS_OF
                    Entity resolution        - Structural verify        Instances
                                             - Consolidation
```

The pipeline never sees the reference ontology. [Riskine](https://github.com/riskine/ontology) (26 classes) is used **only for evaluation**.

## Novel Contribution: SV-LOI

**Structurally-Verified LLM Ontology Induction** -- a multi-view consensus framework:

1. **Two-stage class discovery**: detect domain from entities, then propose 12-25 ontology classes
2. **Batched LLM entity typing**: classify ~7K entities in 15-entity batches
3. **Structural verification**: relation-signature vectors flag outliers >2sigma from class centroid
4. **Disagreement arbitration**: re-query LLM with both semantic and structural evidence
5. **Conservative consolidation**: merge only true synonyms (>90% confidence)

**Key insight**: LLM typing fails on ambiguous names; structural clustering fails on topologically similar roles. Fusing both with disagreement arbitration eliminates each signal's failure mode.

---

## Results

### Ontology Induction (Zone 3)

Three methods compared on the same extraction (qwen2.5:72b, 7,415 entities). Evaluated against the **full 26-class Riskine ontology** using metrics from [OLLM](https://arxiv.org/abs/2410.XXXXX) (NeurIPS 2024) and AutoSchemaKG (2025):

| Metric | Leiden (baseline) | RSI-LCR | **SV-LOI** |
|--------|:-:|:-:|:-:|
| Graph F1 (SGC) | 0.441 | 0.501 | **0.746** |
| Continuous F1 | 0.069 | 0.081 | **0.288** |
| Wu-Palmer | 0.653 | 0.723 | **0.812** |
| BERTScore F1 | 0.451 | 0.577 | **0.592** |
| Induced classes | 11 | 30 | **24** |

### Extraction Quality (Zone 2)

| Metric | Value |
|--------|:-----:|
| Triple Precision (LLM-as-judge) | **78.7%** |
| Fact Recall | **57.2%** |
| Source Grounding Rate | **92.0%** |
| Entities | 7,407 (208 concept + 7,199 structured) |
| Relationships | 27,150 |

### Pipeline Progression

| Stage | Query Acc | Entities | Graph F1 |
|-------|:---------:|:--------:|:--------:|
| Baseline (LLMGraphTransformer) | 35% | ~60 | -- |
| + Zone 1 (section-aware chunking) | 50% | ~60 | -- |
| + Zone 2 (domain-agnostic Open IE) | 56.7% | 7,407 | 0.404 |
| + Zone 3 SV-LOI | **82.5%** | 7,415 | **0.746** |

Query accuracy measured on 40 tasks: PDF knowledge (Q1-20), structured CSV lookups (Q21-30), cross-source linkage (Q31-40).

---

## Quick Start

```bash
pip install -r requirements.txt

# Requires: Neo4j AuraDB (.env), Ollama (localhost:11434)
ollama pull qwen2.5:72b

# Run pipeline
python3 zone1/ingestion.py
python3 zone2/pipeline.py --model qwen2.5:72b
python3 zone3/sv_loi.py --model qwen2.5:72b --suffix zone3_svloi

# Evaluate
python3 baseline/eval.py --suffix zone3_svloi --riskine --all-classes --model qwen2.5:72b

# Compare methods
python3 evaluation/compare_results.py --zone3
```

### Ablation Study

```bash
bash scripts/run_ablations.sh qwen2.5:72b
# Variants: full, no-verify, no-arbitrate, no-consolidate, LLM-only
```

---

## Project Structure

```
zone1/         Section-aware hybrid chunking (PDF + CSV)
zone2/         Domain-agnostic Open IE (bootstrapped vocab, entity resolution)
zone3/         Ontology induction: pipeline.py (Leiden), rsi_lcr.py, sv_loi.py
zone4/         Structured Neo4j storage (upcoming)
baseline/      LLMGraphTransformer baseline + 40-query eval harness
evaluation/    Riskine alignment, ontology metrics (OLLM/AutoSchemaKG), comparison tools
scripts/       Slurm cluster scripts, sync tools
data/          flood/ (NFIP), auto/ (cross-domain), riskine/ (eval only), results/
```

## Datasets

| Source | Format | Size | Purpose |
|--------|--------|------|---------|
| SFIP General Property Policy F-123 | PDF | 27 pages | Primary knowledge source |
| OpenFEMA Policies | CSV | 500 records | Structured policy data |
| OpenFEMA Claims | CSV | 500 records | Structured claims data |
| Kaggle Auto Insurance | CSV | 60K records | Cross-domain transfer |
| Riskine Ontology | JSON Schema | 26 classes | Evaluation reference only |

## Evaluation Metrics

| Metric | Source | Measures |
|--------|--------|----------|
| Graph F1 | OLLM (NeurIPS 2024) | Structure-aware node matching via SGC |
| Continuous F1 | OLLM (NeurIPS 2024) | Optimal 1-to-1 edge matching (Hungarian) |
| BERTScore F1 | AutoSchemaKG (2025) | Threshold-free class name alignment |
| Wu-Palmer | WordNet literature | Taxonomy hierarchy similarity |
| Entity Assignment F1 | Ours | Entity-to-class placement quality |
| Query Accuracy | Ours | 40-query functional KG evaluation |

---

## Key Findings

| ID | Finding | Impact |
|----|---------|--------|
| F-03 | 8B models extract ~1 triple/chunk; 72B extracts 8-50 | Large models critical for extraction |
| F-06 | Early 0.874 F1 was domain leakage (Riskine anchors) | Removed; honest baseline ~0.000 |
| F-07 | Larger models produce compound names misaligned with reference | Motivates Graph F1 over Name F1 |
| F-15 | Two-stage discovery eliminates data-type classes | Domain detection grounds proposals |
| F-16 | LLM consolidation raises Name F1 from 0.214 to 0.563 | Consolidation as important as discovery |
| F-17 | Person, Organization, Object absent from flood data | Max achievable recall ~0.6-0.7 |

Full findings: [`docs/FINDINGS.md`](docs/FINDINGS.md)

---

Academic project -- Emory University CS 584, Spring 2026.
