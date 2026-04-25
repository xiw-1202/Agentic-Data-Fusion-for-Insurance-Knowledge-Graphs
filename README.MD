# SEAF-KG: Structurally-Enhanced Agentic Fusion for Insurance Knowledge Graphs

**Automated domain-ontology induction with cross-domain validation, evaluated on two independent insurance corpora.**

Xiaofei Wang
CS 584 AI Capstone · Emory University · Spring 2026
Advisors: Dr. Eugene Agichtein (Emory) · Dr. Yingying Kang (Assurant)

[![KG demo](https://img.shields.io/badge/demo-text--to--cypher%20chatbot-blue)](chatbot/app.py)

---

## Problem

Insurance carriers entering a new Line of Business face a **3–6 month ontology-engineering bottleneck** per domain. Off-the-shelf LLM KG extractors (e.g. LangChain's `LLMGraphTransformer`) produce flat, duplicated, schema-less graphs. SEAF-KG automates the full pipeline from raw documents and tabular records to a queryable knowledge graph with an **induced domain ontology** — zero manual schema engineering.

## Solution — 4-zone pipeline

The same code runs on any insurance LOB without modification:

```
Zone 1              Zone 2                  Zone 3                      Zone 4
Ingestion           Open IE + Mapper        SV-LOI Ontology Induction   Storage + Demo
-----------         ---------------         --------------------------  ------------------
PDF + CSV     -->   LLM extraction    -->   Class discovery         --> Neo4j (Docker/Aura)
section-aware       + deterministic         Batched entity typing       Class / SUBCLASS_OF
chunking            structured mapper       Structural verification     INSTANCE_OF edges
                    + entity resolution     Schema mapping              Text-to-Cypher
                    + property collapse     LLM consolidation           chatbot (Streamlit)
```

Reference ontology [Riskine](https://github.com/riskine/ontology) is used **only for evaluation**; the pipeline never sees it.

## Novel contribution — SV-LOI

**Structurally-Verified LLM Ontology Induction** fuses LLM semantic typing with graph-structural verification:

1. **Two-stage class discovery** — detect domain from entities, then propose ontology classes.
2. **Batched LLM entity typing** — relation context in each prompt; 15 entities per batch.
3. **Structural verification** — relation-signature vectors flag outliers >2σ from class centroid.
4. **Disagreement arbitration** — re-query LLM with both semantic and structural evidence.
5. **Schema mapping with `Other` escape hatch** — map tabular record types to induced classes (R2RML/OBDA inspired). If no class is a clear fit, the LLM answers "Other" rather than polluting a real class.
6. **Consensus-filtered hierarchy** — candidate SUBCLASS_OF edges from 4 sources are kept only when ≥1 trusted source *or* ≥2 weak sources agree. Eliminates single-source LLM false positives.
7. **Temporal / value-entity source-class fallback** — date and numeric values that the LLM labels `Other` inherit their record's class (claim dates → Claim, policy dates → Policy), raising coverage from ~45 % to ~99 %.

**Key insight**: LLM typing is accurate but inconsistent; graph structure is consistent but over-fragments. Fusing both with arbitration and consensus filtering eliminates each signal's failure mode.

---

## Cross-domain results (2026-04-24)

Two independently-run datasets, **same pipeline binary**, `qwen2.5:72b` for extraction on 2× Quadro RTX 8000.

**Evaluation methodology:** all LLM-as-judge metrics (triple precision, source grounding, Riskine class alignment) are scored by an **independent judge model — `gemma4:31b`** — *different* from the extraction model `qwen2.5:72b`. Using a different model for evaluation prevents self-evaluation bias, where a model rating its own output tends to inflate scores.

| Metric | Flood (NFIP + OpenFEMA) | Emory Spring 2026 (GEICO + T-Mobile + Auto) |
|---|---:|---:|
| Chunks processed | 56 | 34 |
| Raw triples (LLM + structured) | 72,663 | 31,489 |
| Relation types | 186 | 321 |
| **Triple precision** (gemma judge, 100-triple sample) | **0.780** | **0.745** |
| Fact recall (BERTScore, 30-chunk sample) | 0.740 | 0.730 |
| Source grounding rate | 0.680 | 0.800 |
| Entities | 12,326 | 3,666 |
| **Induced classes** | **10** | **10** |
| **Labeling coverage** | **99.7 %** | **98.6 %** |
| SUBCLASS_OF / ASSOCIATED_WITH edges | 4 / 27 | 3 / 21 |
| **Type inconsistency** | 9.1 % | **0.0 %** |
| Duplication rate | 0.0 % | 0.0 % |
| Name F1 vs Riskine | 0.440 | 0.427 |
| BERTScore F1 vs Riskine | 0.615 | 0.544 |
| Wu-Palmer taxonomy similarity | 0.621 | 0.667 |
| Entity-assignment F1 vs Riskine (full 26 / evidenced) | 0.252 / 0.388 | 0.252 / 0.482 |
| Query accuracy (dataset-appropriate 40-Q benchmark)¹ | **87.5 %** | **90.0 %** |

¹ *Each dataset is evaluated against its own 40-question benchmark, **derived from the raw source files not from KG inspection** ([`baseline/eval_tasks_flood.py`](baseline/eval_tasks_flood.py) for NFIP / OpenFEMA / SFIP PDF; [`baseline/eval_tasks_emory.py`](baseline/eval_tasks_emory.py) for Auto Service PDF + T-Mobile + GEICO renters CSVs). Questions probe column names, PDF section titles, and actual source values — keywords come from the data, not from the KG. The picker auto-selects from the suffix or via an explicit `--dataset` flag. Emory's 4 misses (of 40, on the cluster run): Q14 payment-plan column preservation, Q16 claim-status keyword surfacing, Q34 auto-service-to-mobile concept sharing, Q40 county column capture.*

### Classes induced

| Shared core (7) | Flood-only (3) | Emory-only (3) |
|---|---|---|
| Claim · Coverage · Organization · Person · Policy · Procedure · Property | InsuranceCoverage · InsurancePayoutCoverage · Risk | Survey · ClientJourneySurvey · WarrantyServiceProcedure |

Both runs *independently* discovered the same 7-class insurance backbone plus **domain-specific** classes for their respective corpora. Two of Emory's three SUBCLASS_OF edges are semantically correct IS-A relations discovered without manual labels:
- `ClientJourneySurvey` ⊂ `Survey`
- `WarrantyServiceProcedure` ⊂ `ServiceProcedure`

### Temporal-fix verification

Before/after the source-class fallback:

| | Flood | Emory |
|---|---:|---:|
| `Other` label before fallback | 82 % | 77 % |
| `Other` label after fallback | **0.3 %** | **3 %** |
| Date entities re-typed | 3,248 / 3,248 | 345 / 345 |
| Numeric entities re-typed | 6,046 / 6,047 | 1,493 / 1,518 |

Live verification via Cypher: *"Show me claims in May 2023"* returns 4 concrete claims with device type joined from the structured path.

See [`docs/final_analysis.md`](docs/final_analysis.md) for full metric breakdowns, semantic-sanity sampling per class, and the result-to-claim verdict (`partial, confidence medium`).

---

## Live KG chatbot

`chatbot/` is a Streamlit app that translates English questions into Cypher against the induced ontology, runs them on Neo4j, and picks the best visualization (table / bar / line / pie / graph / scalar) per question. Uses Anthropic Claude with prompt caching on the schema + few-shot block.

```bash
# Spin up isolated Neo4j containers for each dataset
cd zone4 && docker compose up -d && cd ..

# Load fresh results into the containers
cp zone4/env.emory.example .env
# add ANTHROPIC_API_KEY=... to .env
pip install -r chatbot/requirements.txt
python3 -m zone4.load_to_neo4j --results data/results/emory

# Launch chatbot
streamlit run chatbot/app.py
# → http://localhost:8501
```

The chatbot surfaces the generated Cypher, the rows it returned, and a natural-language interpretation — so demos make the pipeline's value visible (structured query against an induced ontology, not a vector-RAG bypass).

---

## Quick start — full pipeline

```bash
pip install -r requirements.txt

# Requires: Neo4j (Aura or local Docker), Ollama
ollama pull qwen2.5:72b

# Full pipeline on flood data
python3 zone1/ingestion.py --data-dir data/flood
python3 zone2/pipeline.py --model qwen2.5:72b --passes 1
python3 zone3/sv_loi.py   --model qwen2.5:72b --suffix flood_qwen2_5_72b

# Evaluate
python3 evaluation/extraction_quality.py --suffix flood_qwen2_5_72b --sample-size 100
python3 baseline/eval.py --suffix flood_qwen2_5_72b --riskine --all-classes
```

Cluster (Emory Turing) shortcut:

```bash
bash scripts/sync_to_cluster.sh <netid>
ssh -J <netid>@lab0z... <netid>@turinglogin... \
  'cd /local/scratch/<netid>/project && sbatch scripts/slurm_pipeline.sh'
# Emory dataset:
#   sbatch --export=ALL,DATA_DIR=data/Emory_Spring2026 scripts/slurm_pipeline.sh
bash scripts/sync_to_cluster.sh <netid> --fetch
```

Zone 1 auto-skips when `$CHUNKS_FILE` already exists, so fast Zone-2+3 re-runs are one sbatch.

---

## Project structure

```
zone1/          Section-aware hybrid chunking (PDF + CSV)
zone2/          Open IE: LLM extraction + deterministic structured mapper
zone3/          sv_loi.py facade + _svloi/ (pipeline, typing, hierarchy, sohd, records, …)
zone4/          Neo4j loader, docker-compose, env templates for dual-DB demo
chatbot/        Streamlit text-to-Cypher QA against the induced KG
baseline/       LLMGraphTransformer baseline + 40-query eval harness
evaluation/     Extraction quality, Riskine alignment, ontology metrics (OLLM/AutoSchemaKG)
scripts/        Slurm submission, cluster sync, data tools
docs/           final_analysis.md, FINDINGS.md, version histories, superpowers/plans/
data/           flood/ · Emory_Spring2026/ (gitignored, proprietary) · riskine/ · results/
```

## Datasets

| Source | Format | Records | Domain |
|---|---|---:|---|
| FEMA SFIP General Property (F-123) | PDF | 27 pages | Flood |
| OpenFEMA Policies / Claims | CSV | 500 + 500 | Flood |
| GEICO Renters: policies / claims / 3 surveys / cancel survey | CSV | 100 each | Auto/Home |
| T-Mobile: claims / survey / chat survey | CSV | 100 each | Mobile device |
| Auto Service Contract (Assurant form) | PDF | 24 pages | Auto |
| [Riskine Ontology](https://github.com/riskine/ontology) | JSON Schema | 26 classes | **Evaluation reference only** |

## Evaluation metrics

| Metric | Source | Measures |
|---|---|---|
| Triple Precision | Ours, LLM-as-judge | Factual correctness of extracted triples |
| Fact Recall | G-BERTScore | Source-document coverage |
| Source Grounding | Ours | Triple traceability to evidence |
| Entity Assignment F1 | Ours | Entity-to-class placement vs reference |
| BERTScore F1 | AutoSchemaKG 2025 | Class-name semantic alignment |
| Graph F1 | OLLM NeurIPS 2024 | SGC-based structure match |
| Wu-Palmer | WordNet | Taxonomy hierarchy similarity |
| Query Accuracy | Ours | 40 analytical-query functional eval |

## Key findings

| ID | Finding | Impact |
|---|---|---|
| F-03 | 8B extracts ~1 triple/chunk; 72B extracts 8–50 | Large models critical for extraction |
| F-13 | 72B name-F1 vs Riskine = 0.000 (naming style mismatch) | Domain ontologies diverge from flood references |
| F-17 | Riskine is flood-biased; Emory alignment under-states quality | Build domain-matched benchmarks |
| F-20 | Source-class fallback raises labeling coverage 45 % → 99 % | Temporal/value entities inherit record context |
| F-21 | Consensus filter on IS-A edges cuts false positives 80 % → 33 % | Multi-source voting beats single-LLM judgment |
| F-22 | Explicit `Other` option in schema mapping lets domain classes emerge | `SurveyRecord` → `Survey`, not `Document` |

Full list: [`docs/FINDINGS.md`](docs/FINDINGS.md)

## Honest limitations

- **Single run per dataset**, no seed-variance reported yet.
- **Flood-authored benchmark** — the 40-query set doesn't transfer to Emory; a domain-matched Emory benchmark is needed.
- **`Coverage → Policy` SUBCLASS_OF edge** remains debatable (part-of vs IS-A); consensus filter kept it but it should probably be an ASSOCIATED_WITH edge.
- **Source data** for Emory has no shared policy numbers between `policies.csv` and `claims.csv` (independent 100-row samples) — cross-source joins on policy ID return empty.
- Planned variant baselines (AutoSchemaKG, KGGen, OLLM) not yet executed.

See [`docs/final_analysis.md`](docs/final_analysis.md) §7 for the full prioritized improvement backlog.

---

Academic project — Emory University CS 584, Spring 2026.
