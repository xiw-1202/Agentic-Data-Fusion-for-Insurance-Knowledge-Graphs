# SEAF-KG: Schema-Evolving Agentic Framework for Insurance Knowledge Graphs

**Automated Ontology Induction with Cross-Domain Transfer Learning**

**Authors:** Xiaofei Wang, Zechary Chou
**Course:** CS 584 AI Capstone -- Emory University, Spring 2026
**Advisors:** Dr. Eugene Agichtein (Emory), Dr. Yingying Kang (Assurant)

---

## Overview

Insurance companies expanding into new Lines of Business (flood, auto, health, liability) face a **3-6 month ontology engineering bottleneck per domain**. We build a fully automated pipeline that induces ontologies from raw insurance documents -- without manual ontology engineering or hardcoded domain knowledge.

Our pipeline: **structured chunking (Zone 1) -> domain-agnostic Open IE (Zone 2) -> bottom-up ontology induction (Zone 3) -> structured Neo4j storage (Zone 4)**. The pipeline never sees the reference ontology -- Riskine is used ONLY for evaluation.

### Novel Contribution: SV-LOI

**SV-LOI (Structurally-Verified LLM Ontology Induction)** is a multi-view consensus framework that fuses:
1. **LLM semantic typing** -- two-stage class discovery (detect domain, then propose 12-25 classes) + batched entity classification
2. **Graph structural verification** -- relation-signature vectors verify assignments, flag outliers >2 sigma from class centroid
3. **Disagreement arbitration** -- re-query LLM with both semantic and structural evidence for flagged entities
4. **LLM-guided consolidation** -- conservative merging of synonym classes (>90% confidence only)

Key insight: LLM typing fails on entities with ambiguous names but clear structural patterns; structural clustering fails on topologically similar but semantically different roles. Fusing both eliminates each signal's failure mode.

---

## Results

### Zone 3 Method Comparison (qwen2.5:72b)

Evaluated against the **full 26-class Riskine ontology** using standard metrics from OLLM (NeurIPS 2024) and AutoSchemaKG (2025):

| Metric | Leiden (baseline) | RSI-LCR | **SV-LOI** (novel) |
|--------|:-:|:-:|:-:|
| BERTScore F1 | 0.451 | 0.577 | **0.592** |
| Graph F1 (SGC) | 0.441 | 0.501 | **0.746** |
| Continuous F1 | 0.069 | 0.081 | **0.288** |
| Wu-Palmer | 0.653 | 0.723 | **0.812** |
| Entity Assignment F1 (present) | 0.293 | 0.344 | **0.264** |
| Induced classes | 11 | 30 | **24** |
| Query accuracy (40Q) | -- | -- | **82.5%** |
| Type inconsistency | 8.3% | 2.1% | **8.7%** |

**SV-LOI achieves the best Graph F1 (0.746), Continuous F1 (0.288), and Wu-Palmer (0.812)** -- the three structure-aware metrics from OLLM. The 40-query evaluation tests PDF knowledge retrieval (Q1-20), structured CSV data lookups (Q21-30), and cross-source entity linkage (Q31-40).

### Pipeline Progression

| Stage | Query Acc | Entities | Key Improvement |
|-------|:---------:|:--------:|-----------------|
| Baseline (LLMGraphTransformer) | 35% | ~60 | Fixed 512-token chunks |
| + Zone 1 (section-aware chunking) | 50% | ~60 | tau=0.85 semantic merge |
| + Zone 2 (domain-agnostic Open IE) | 95% | 1,351 | Bootstrapped vocab, 72b model |
| + Zone 2 SEAF (PDF + CSV) | 95% | 7,415 | Claims + policy CSV ingestion |
| + Zone 3 SV-LOI | 82.5% | 7,415 | 24-class ontology (harder 40Q eval) |

---

## Architecture

```
ZONE 1: Ingestion        ZONE 2: Open IE              ZONE 3: Ontology Induction    ZONE 4: Storage
-----------------        ---------------              -------------------------     ---------------
ANY insurance docs       Bootstrap from docs:          SV-LOI (novel):               3-layer Neo4j:
PDF + CSV           ->     LLM proposes entity    ->    Two-stage class discovery ->   OntologyClass
Section-aware chunk        + relation types             Batched LLM entity typing     SUBCLASS_OF
tau=0.85 merge             Synthetic few-shot           Structural verification       Entity instances
                           Extract (s,r,o,conf)         Disagreement arbitration
                           Entity resolution            LLM consolidation

                           EVALUATION (separate)
                           Induced ontology -> Riskine (26 classes)
                           BERTScore / Graph F1 / Continuous F1 / Wu-Palmer
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.11+
pip install -r requirements.txt

# Neo4j AuraDB — credentials in .env
# Ollama — http://localhost:11434
ollama pull qwen2.5:72b  # or llama3.1:8b for testing
```

### Run Pipeline

```bash
# Zone 1 — chunking
python3 zone1/ingestion.py

# Zone 2 — domain-agnostic Open IE
python3 zone2/pipeline.py --model qwen2.5:72b

# Zone 3 — ontology induction (choose one)
python3 zone3/pipeline.py                                          # Leiden (baseline)
python3 zone3/rsi_lcr.py --model qwen2.5:72b --suffix zone3_rsi   # RSI-LCR
python3 zone3/sv_loi.py --model qwen2.5:72b --suffix zone3_svloi  # SV-LOI (novel)

# Evaluation
python3 baseline/eval.py --suffix zone3_svloi --riskine --all-classes --model qwen2.5:72b
```

### Ablation Study

```bash
# Run all 5 SV-LOI ablation variants
bash scripts/run_ablations.sh qwen2.5:72b

# On Turing cluster
sbatch --export=ALL,MODEL=qwen2.5:72b,SUFFIX=zone3_svloi_seaf scripts/slurm_sv_loi.sh
sbatch --export=ALL,MODEL=qwen2.5:72b,SUFFIX=zone3_svloi_seaf scripts/slurm_eval_only.sh
```

### Compare Results

```bash
python3 evaluation/compare_results.py --zone3
```

---

## Directory Structure

```
baseline/          pipeline.py, eval.py (40-query harness), ontology_induction.py
zone1/             ingestion.py — section-aware hybrid chunking
zone2/             pipeline.py, prompts.py, entity_resolution.py — domain-agnostic Open IE
zone3/             pipeline.py (Leiden), rsi_lcr.py (RSI-LCR), sv_loi.py (SV-LOI), graph_cache.py
zone4/             Structured Neo4j storage (upcoming)
evaluation/        riskine_eval.py, ontology_metrics.py, riskine_loader.py, compare_results.py
scripts/           Slurm scripts, cluster setup, sync tools
config.py          Paths + credentials (from .env)
data/flood/        NFIP flood insurance data (primary LOB)
data/auto/         Auto insurance data (cross-domain transfer LOB)
data/riskine/      Reference ontology (EVALUATION ONLY)
data/results/      All evaluation output JSONs
docs/              FINDINGS.md (F-01 through F-18), research documentation
```

---

## Evaluation Metrics

Standard metrics from the ontology induction literature:

| Metric | Source | What It Measures |
|--------|--------|-----------------|
| BERTScore F1 | AutoSchemaKG 2025 | Threshold-free class name alignment |
| Graph F1 | OLLM NeurIPS 2024 | Structure-aware node matching (SGC embeddings) |
| Continuous F1 | OLLM NeurIPS 2024 | Optimal 1-to-1 edge matching (Hungarian) |
| Wu-Palmer | WordNet literature | Taxonomy hierarchy similarity |
| Entity Assignment F1 | Ours | Entity-to-class placement quality |
| Query Accuracy | Ours | 40-query functional KG evaluation |

Evaluated against the **full Riskine ontology** (26 classes). Since flood insurance data covers only ~10 of these classes, we report both full-ontology and present-class metrics.

---

## Riskine Reference Ontology

26 core classes: Address, Animal, BankAccount, BusinessProcess, Coverage, CreditCard, Damage, DataProcessing, DrivingLicense, Education, Employee, Finances, Identification, Object, Organization, Person, Preference, Product, Profession, Property, Revenue, Risk, SecurityMeasure, Site, Structure, Vehicle

Source: [github.com/riskine/ontology](https://github.com/riskine/ontology)

**Used ONLY for evaluation** -- never imported by pipeline code (zone2/, zone3/, zone4/).

---

## Empirical Findings

Key findings documented in `docs/FINDINGS.md`:

| ID | Finding | Impact |
|----|---------|--------|
| F-03 | 8B models extract ~1 triple/chunk; 72B extracts 8-50 | Motivates large model for extraction |
| F-06 | Zone 2 v1's 0.874 F1 was domain leakage | Removed anchors; honest baseline ~0.000 |
| F-07 | Larger models produce compound names that don't align with reference | Motivates BERTScore/Graph F1 over Name F1 |
| F-15 | SV-LOI two-stage discovery eliminates data-type classes | Domain detection grounds class proposals |
| F-16 | LLM consolidation raises Name F1 from 0.214 to 0.563 | Consolidation is as important as discovery |
| F-17 | Person, Organization, Object absent from flood data | Max achievable full-Riskine recall ~0.6-0.7 |

---

## Cluster Setup (Emory Turing)

- 2x 48GB Quadro RTX 8000 GPUs
- Models: qwen2.5:72b (primary), llama3.3:70b (backup)
- Scripts: `scripts/cluster_setup_v2.sh`, `scripts/slurm_sv_loi.sh`, `scripts/slurm_eval_only.sh`

```bash
# Sync to cluster
bash scripts/sync_to_cluster.sh xwa2284

# Fetch results
bash scripts/sync_to_cluster.sh xwa2284 --fetch
```

---

## Timeline

| Week | Milestone |
|------|-----------|
| 1-3 | Baseline pipeline + Zone 1 chunking |
| 4-5 | Zone 2 domain-agnostic Open IE + entity resolution |
| 6-7 | Zone 3 Leiden baseline + 72b extraction |
| 8-9 | RSI-LCR + SV-LOI + comprehensive evaluation suite |
| 10 | Full 26-class eval, ablation framework, 40-query harness |
| 11 | Paper writing, cross-domain transfer, final deliverables |

---

## License

Academic project -- Emory University CS 584, Spring 2026.
