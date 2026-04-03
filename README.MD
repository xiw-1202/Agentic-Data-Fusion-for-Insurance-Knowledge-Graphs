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
                                             - Schema mapping
                                             - LLM consolidation
```

The pipeline never sees the reference ontology. [Riskine](https://github.com/riskine/ontology) (26 classes) is used **only for evaluation**.

## Novel Contribution: SV-LOI

**Structurally-Verified LLM Ontology Induction** -- fuses LLM semantic typing with graph-structural verification:

1. **Two-stage class discovery**: detect domain from entities, then propose ontology classes
2. **Batched LLM entity typing**: classify ~7,600 entities in batches with relation context
3. **Structural verification**: relation-signature vectors flag outliers >2sigma from class centroid
4. **Disagreement arbitration**: re-query LLM with both semantic and structural evidence
5. **Schema mapping**: LLM maps tabular record types to induced classes via relation profiles (inspired by R2RML/OBDA)
6. **LLM-guided consolidation**: structural evidence (member count, leaf fraction, relational diversity) informs LLM keep/merge decisions, with structural veto on record-backed classes

**Key insight**: LLM typing is accurate but inconsistent; structural clustering is consistent but over-fragments. Fusing both with disagreement arbitration eliminates each signal's failure mode.

---

## Results

### Extraction Quality (Zone 2 v4)

| Metric | Value |
|--------|:-----:|
| Triple Precision (LLM-as-judge) | **78.9%** |
| Fact Recall (G-BERTScore) | **81.6%** |
| Source Grounding Rate | **81.0%** |
| Entities | 7,594 |
| Relationships | 25,512 |
| Entity types | 103 |

Full extraction version history: [`docs/ZONE2_VERSIONS.md`](docs/ZONE2_VERSIONS.md)

### Ontology Induction (Zone 3)

Best results across SV-LOI versions. Evaluated against the **full 26-class Riskine ontology**:

| Metric | Best | Version | Notes |
|--------|:----:|:-------:|-------|
| Name F1 | **0.441** | v8 | 9 induced vs 26 reference classes |
| BERTScore F1 | **0.703** | v8 | Threshold-free alignment |
| EA F1 (present-class) | **0.582** | v6 | Excludes absent Riskine classes |
| EA F1 (full 26-class) | **0.414** | v6 | Against all 26 Riskine classes |
| Wu-Palmer | **0.775** | v6 | Taxonomy hierarchy similarity |
| Continuous F1 | **0.222** | v6 | Soft assignment score |
| Riskine classes covered | **10/26** | v5-v6 | 17 classes absent from flood data |
| Query Accuracy | **87.5%** | v4+ | 40-task functional eval |
| Type Inconsistency | **0.0%** | v5+ | Zero duplicate type labels |

Full induction version history: [`docs/ZONE3_VERSIONS.md`](docs/ZONE3_VERSIONS.md)

### Method Comparison (Zone 3)

| Metric | Leiden (baseline) | RSI-LCR | **SV-LOI** (best) |
|--------|:-:|:-:|:-:|
| BERTScore F1 | 0.451 | 0.577 | **0.703** |
| Graph F1 | 0.441 | 0.501 | **0.714** |
| Wu-Palmer | 0.653 | 0.723 | **0.775** |
| Induced classes | 11 | 30 | 9-21 |
| Query Accuracy | 95% | 95% | 87.5% |

### Pipeline Progression

| Stage | Query Acc | Entities | Notes |
|-------|:---------:|:--------:|-------|
| Baseline (LLMGraphTransformer) | 35% | ~60 | Flat, schema-less graph |
| + Zone 1 (section-aware chunking) | 50% | ~60 | Better context boundaries |
| + Zone 2 (domain-agnostic Open IE) | 87.5% | 7,594 | 25K triples, 79% precision |
| + Zone 3 (SV-LOI) | 87.5% | 7,594 | Structured ontology layer |

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
docs/          Findings, version histories, experiment logs
data/          flood/ (NFIP), auto/ (cross-domain), riskine/ (eval only), results/
```

## Datasets

| Source | Format | Size | Purpose |
|--------|--------|------|---------|
| SFIP General Property Policy F-123 | PDF | 27 pages | Primary knowledge source |
| OpenFEMA Policies | CSV | 500 records | Structured policy data |
| OpenFEMA Claims | CSV | 500 records | Structured claims data |
| Riskine Ontology | JSON Schema | 26 classes | Evaluation reference only |

## Evaluation Metrics

| Metric | Source | Measures |
|--------|--------|----------|
| BERTScore F1 | AutoSchemaKG (2025) | Threshold-free class name alignment |
| Graph F1 | OLLM (NeurIPS 2024) | Structure-aware node matching via SGC |
| Continuous F1 | OLLM (NeurIPS 2024) | Optimal 1-to-1 edge matching (Hungarian) |
| Wu-Palmer | WordNet literature | Taxonomy hierarchy similarity |
| Entity Assignment F1 | Ours | Entity-to-class placement quality |
| Query Accuracy | Ours | 40-query functional KG evaluation |

## Key Findings

| ID | Finding | Impact |
|----|---------|--------|
| F-03 | 8B models extract ~1 triple/chunk; 72B extracts 8-50 | Large models critical for extraction |
| F-06 | Early 0.874 F1 was domain leakage (Riskine anchors) | Removed; honest baseline ~0.000 |
| F-15 | Two-stage discovery eliminates data-type classes | Domain detection grounds proposals |
| F-16 | LLM consolidation raises Name F1 from 0.214 to 0.563 | Consolidation as important as discovery |
| F-17 | Person, Organization, Object absent from flood data | Max achievable recall ~0.6-0.7 |
| F-19 | EA eval undercounts with raw member centroids | Enriched + dual representation triples EA F1 |

Full findings: [`docs/FINDINGS.md`](docs/FINDINGS.md)

---

Academic project -- Emory University CS 584, Spring 2026.
