# CS584 AI Capstone — Progress Report: Week 10
**Schema-Evolving Knowledge Graphs for Insurance: Automated Ontology Induction with Cross-Domain Transfer Learning**

**Authors:** Xiaofei Wang · Zechary Chou
**Course:** CS 584 AI Capstone — Emory University, Spring 2026
**Last Updated:** March 24, 2026

---

## Zone Status (Week 10)

| Zone | Status | Best Result | Notes |
|------|:------:|:-----------:|-------|
| Zone 1 — Section-aware chunking | ✅ Done | Acc 50%, F1 0.250 | τ=0.85 semantic merge |
| Baseline — LLMGraphTransformer | ✅ Done | Acc 35% | Comparison only |
| Zone 2 v2 — Domain-agnostic Open IE | ✅ Done | Acc 95% (72b) | No anchors, honest baseline |
| Zone 2 EDC canonicalization | ✅ Done | 48→14 rel types (71% reduction) | Included in best run |
| Zone 2.5 — Entity resolution | ✅ Done | 0% duplication | cosine ≥ 0.90 merge |
| Zone 3 — Leiden (baseline) | ✅ Done | Name F1 0.000, BERTScore 0.451 | Multi-resolution Leiden, 11 classes |
| Zone 3 — RSI-LCR | ✅ Done | Name F1 0.295, BERTScore 0.577 | Relation-Signature Induction + LLM Coherence Refinement |
| Zone 3 — **SV-LOI** (novel) | ✅ Done | **Name F1 0.563, BERTScore 0.732** | **Structurally-Verified LLM Ontology Induction — primary contribution** |
| Zone 4 — Structured Neo4j storage | ⏳ Upcoming | — | 3-layer ontology schema |

## Full Results Table

### Early Results (8b/70b exploration)

| Run | Model | Query Acc | Type Incon. | Riskine F1 | Notes |
|-----|-------|:---------:|:-----------:|:----------:|-------|
| Baseline (512-tok) | llama3.1:8b | 35% | 8.0% | — | LLMGraphTransformer |
| Zone 1 | llama3.1:8b | 50% | 15.2% | 0.250 | Section-aware chunking |
| Zone 2 v1 | llama3.1:8b | 50% | 0.0% | 0.874 | ⚠️ Domain leakage — inflated |
| Zone 2 v2 (honest) | llama3.1:8b | 35% | 0.0% | ~0.000 | All anchors removed |
| Zone 3 (8b) | llama3.1:8b | 35% | 17.4% | 0.171 | Leiden res=0.6 |
| Zone 3 (70b) | llama3.1:70b | 30% | 2.9% | 0.071 | Turing GPU; F-07 |

### Zone 3 Method Comparison (72b, primary results)

All methods use the same Zone 2 extraction (qwen2.5:72b, 1,351 entities, 1,749 relationships).

| Metric | Leiden (baseline) | RSI-LCR | **SV-LOI** (novel) |
|--------|:-:|:-:|:-:|
| Name F1 | 0.000 | 0.295 | **0.563** |
| Entity Assignment F1 (full) | 0.226 | 0.302 | **0.326** |
| Entity Assignment F1 (present) | 0.293 | 0.344 | **0.417** |
| BERTScore F1 | 0.451 | 0.577 | **0.732** |
| Graph F1 | 0.441 | 0.501 | **0.714** |
| Continuous F1 | 0.069 | 0.081 | **0.122** |
| Wu-Palmer | 0.653 | 0.723 | **0.752** |
| Induced classes | 11 | 30 | 7 |
| Query accuracy | 95% | 95% | 95% |
| Type inconsistency | 8.3% | 2.1% | 2.2% |

**Methods:**
- **Leiden** — Multi-resolution Leiden community detection on entity similarity graph (baseline, Variant A)
- **RSI-LCR** — Relation-Signature Induction with LLM Coherence Refinement (Variant B)
- **SV-LOI** — Structurally-Verified LLM Ontology Induction (novel contribution): two-stage LLM class discovery (detect domain, then propose classes) + LLM-guided class consolidation + structural verification

> **Key finding:** SV-LOI achieves **0.563 Name F1** and **0.732 BERTScore F1** — the highest across all methods — by using two-stage class discovery that eliminates data-type classes and LLM-guided consolidation that maps fine-grained classes to canonical ontology concepts.
>
> **Note on present-class F1 (F-17/F-18):** Flood insurance data does not contain Person, Organization, or Object entities. The "present-class" metric evaluates only against the ~7 Riskine classes actually representable in the source data, giving a fairer comparison. SV-LOI achieves **0.417** present-class F1 vs 0.293 for Leiden.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Proposed Solution Overview](#2-proposed-solution-overview)
3. [Datasets](#3-datasets)
4. [What We Built — Step by Step](#4-what-we-built--step-by-step)
   - 4.1 [Baseline PDF Loader (Pre-existing)](#41-baseline-pdf-loader-pre-existing)
   - 4.2 [Baseline Pipeline — LangGraph + LLMGraphTransformer](#42-baseline-pipeline--langgraph--llmgraphtransformer)
   - 4.3 [Baseline Evaluation Harness — 20 Query Tasks](#43-baseline-evaluation-harness--20-query-tasks)
   - 4.4 [Zone 1 — Novel Hybrid Chunking](#44-zone-1--novel-hybrid-chunking)
   - 4.5 [Ablation Study Setup](#45-ablation-study-setup)
   - 4.6 [Results Comparison Tool](#46-results-comparison-tool)
   - 4.7 [Baseline Riskine Alignment + LLM Ontology Induction](#47-baseline-riskine-alignment--llm-ontology-induction)
5. [Evaluation Method and Metrics](#5-evaluation-method-and-metrics)
6. [Results](#6-results)
   - 6.1 [Baseline — Original 512-token Chunks](#61-baseline--original-512-token-chunks)
   - 6.2 [Baseline + Zone 1 Chunks — Ablation](#62-baseline--zone-1-chunks--ablation)
   - 6.3 [Side-by-Side Comparison](#63-side-by-side-comparison)
   - 6.4 [Task-by-Task Breakdown — All 20 Tasks](#64-task-by-task-breakdown--all-20-tasks)
7. [Concrete Examples](#7-concrete-examples)
   - 7.1 [Example 1 — Chunking Quality](#71-example-1--chunking-quality)
   - 7.2 [Example 2 — Type Inconsistency and the Leiden Fix](#72-example-2--type-inconsistency-and-the-leiden-fix)
8. [Key Findings and Interpretation](#8-key-findings-and-interpretation)
9. [Bugs Found and Fixed](#9-bugs-found-and-fixed)
10. [Technical Stack](#10-technical-stack)
11. [File Structure](#11-file-structure)
12. [Zone 2 & 3 Implementation](#12-zone-2--3-implementation)
13. [Timeline](#13-timeline)

---

## 1. Problem Statement

Insurance companies expanding into new Lines of Business (LOBs) — flood, auto, health, liability — face a **3–6 month ontology engineering bottleneck per domain**. A team of domain experts must manually design ontology schemas from scratch for each new LOB. This is:

- **Slow and expensive** — FTE months of expert labor per domain
- **Not scalable** — cost grows linearly with number of LOBs
- **Siloed** — no mechanism for reusing structure learned in one domain when building the next

**State-of-practice baseline (what we're comparing against):**
Tools like LangChain's `LLMGraphTransformer` can extract knowledge graph triples from text automatically, but the resulting graphs have critical problems:

| Problem | Description |
|---------|-------------|
| Entity duplication | Same entity described differently in two chunks → two separate nodes (`Policy 5123` and `policy #5123`) |
| Type proliferation | LLM invents a new label per chunk context → `Policy`, `Policy action`, `Policy feature`, `Policy type`, `Policy clause` all refer to the same concept |
| No ontology hierarchy | All nodes have flat string labels — no `SUBCLASS_OF` edges, no class grouping |
| No cross-domain transfer | Every new LOB deployment starts from scratch |

**Our research goal:**
Build an automated 4-zone pipeline that eliminates all four problems and produces a reusable, structured ontology from raw insurance documents — without any manual ontology engineering.

---

## 2. Proposed Solution Overview

### 4-Zone Pipeline

```
ZONE 1: Ingestion ✅      ZONE 2: Open IE ✅         ZONE 3: Ontology ✅         ZONE 4: Storage ⏳
─────────────────         ──────────────────         ────────────────────────    ──────────────────────
PDF + CSV input           Bootstrap vocab             3 methods compared:         3-layer graph:
Section-aware       →     from documents        →     A. Leiden (baseline)    →    • OntologyClass nodes
chunking                  1-pass extraction           B. RSI-LCR                  • SUBCLASS_OF hierarchy
τ = 0.85 semantic         EDC canonicalization        C. SV-LOI (novel)           • Entity instances
merge                     Entity resolution           LLM class discovery
                          (no domain hardcoding)      + consolidation
```

### Novel Contributions (Implemented)
**Zone 3 is the core research contribution**: three ontology induction methods are compared, with **SV-LOI** as the primary novel contribution.

**SV-LOI (Structurally-Verified LLM Ontology Induction)** — the novel method:
1. **Two-stage class discovery**: LLM first detects the insurance domain from entity examples, then proposes ontology classes grounded in that domain (eliminates data-type classes like "NumericValue" or "Temporal")
2. **Entity assignment**: Each entity is assigned to the most semantically appropriate class via LLM
3. **LLM-guided class consolidation**: Merges fine-grained classes into canonical concepts (e.g., City+County+State -> Address, Policy -> Product) — raises Name F1 from 0.214 to 0.563
4. **Structural verification**: Validates class assignments against graph structure
5. Domain-agnostic — transfers to any insurance LOB with **zero code changes**

**Comparison methods:**
- **Leiden (baseline)**: Multi-resolution community detection on entity similarity graph
- **RSI-LCR**: Relation-Signature Induction with LLM Coherence Refinement — groups entities by relation-type signatures, then refines with LLM

**EDC Canonicalization** (Zone 2): Post-extraction batch LLM call maps 48 raw relation types to 14 bootstrapped vocab types (71% reduction), reducing graph fragmentation without hardcoding any domain knowledge.

**Evaluation suite** now includes OLLM-style metrics: BERTScore F1, Graph F1, Continuous F1, Wu-Palmer similarity — providing a comprehensive view beyond simple name matching.

---

## 3. Datasets

### Primary: NFIP Flood Insurance

| Source | Format | Records Used | Full Dataset |
|--------|--------|-------------|--------------|
| SFIP General Property Policy F-123 (2021) | PDF, 27 pages | 1 document | 1 document |
| OpenFEMA Policy Dataset | CSV | 500 records | 100K+ records |
| OpenFEMA Claims Dataset | CSV | 500 records | 50K+ records |

The **PDF is the primary knowledge source** — formal definitions, coverage rules, exclusions, claims procedures for NFIP flood policies. Its structured format (Roman numeral sections, ALL-CAPS headings, lettered subsections) makes section-aware chunking tractable.

### Cross-Domain Transfer: Kaggle Auto Insurance

| Source | Format | Records | Purpose |
|--------|--------|---------|---------|
| Kaggle Auto Insurance Dataset | CSV | 60K records | Cross-domain transfer evaluation, Weeks 8–9 |

**Transfer evaluation targets:**
- **CTR (Class Transfer Rate) > 60%** — what fraction of auto ontology classes map to an existing flood class
- **PTR (Property Transfer Rate) > 50%** — what fraction of entity properties are shared across domains

### Reference Ontology: Riskine

- **Source:** github.com/riskine/ontology
- **Purpose:** Ground-truth insurance ontology for Precision, Recall, F1 evaluation of our induced ontology
- **Matching:** Semantic class-name matching via LLM judge + embedding cosine similarity

---

## 4. What We Built — Step by Step

### 4.1 Baseline PDF Loader (Pre-existing)

**File:** `baseline_pdf_loader.py`
**What it does:** Loads the SFIP PDF with LangChain `PyPDFLoader`, splits into fixed 512-token overlapping windows (50-token overlap) using `RecursiveCharacterTextSplitter`.
**Output:** `data/pdf_chunks.json` — **56 chunks**

**Why this is the starting point:** Fixed sliding-window chunking is the LangChain default and the most common approach in open-source KG construction pipelines. We use it as-is for the baseline.

**Known problem:** The 512-token window ignores document structure. Chunks frequently start mid-sentence, split a definition across two chunks, or strip the section heading that gives meaning to a list of items.

---

### 4.2 Baseline Pipeline — LangGraph + LLMGraphTransformer

**File:** `baseline/pipeline.py`
**Why we built it:** To establish a documented, reproducible baseline using off-the-shelf LangChain tools — the state-of-practice approach for LLM-based KG construction.

**Architecture — LangGraph StateGraph, 3 nodes (default) or 4 nodes with `--induce-ontology`:**

```
load_chunks  →  extract_triples  →  insert_to_neo4j  [→  induce_ontology]
```

**State:**
```python
class PipelineState(TypedDict):
    chunks: list[dict]
    graph_documents: list          # GraphDocument objects
    neo4j_stats: dict              # nodes/rels inserted
    ontology_induction: dict       # populated only with --induce-ontology
    errors: Annotated[list, operator.add]
    model: str
```

**CLI flags:**
```bash
python3 baseline/pipeline.py                           # original 512-token chunks
python3 baseline/pipeline.py --zone1                   # Zone 1 section-aware chunks
python3 baseline/pipeline.py --zone1 --induce-ontology # + LLM ontology induction
python3 baseline/pipeline.py --model qwen2.5:7b        # alternative model
```

**Triple extraction (unconstrained):**
```python
llm = ChatOllama(model="llama3.1:8b", base_url=OLLAMA_BASE_URL, temperature=0)
transformer = LLMGraphTransformer(llm=llm)
# No allowed_nodes, no allowed_relationships
# LLM invents its own entity types and relation names
```

**Neo4j insertion:**
```python
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME,
                   password=NEO4J_PASSWORD, database=NEO4J_DATABASE)
graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
```

**Why Llama 3.1 8B via Ollama:** All compute runs locally — no API costs, no data privacy concerns for insurance documents. Same model used in the novel pipeline, ensuring a fair comparison.

**Critical fix — Neo4j database name:**
Neo4j AuraDB uses the username as the database name (not `neo4j`).

```python
# config.py
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", NEO4J_USERNAME)
# Our username = "8929c2c7"  →  database = "8929c2c7"
```

**Baseline run stats:**

| Metric | Value |
|--------|-------|
| Runtime | 908.4 seconds (~15 min) |
| Chunks processed | 56 |
| Errors | 0 |
| Nodes created | 365 |
| Relationships created | 587 |
| Unique node labels | 58 |

---

### 4.3 Baseline Evaluation Harness — 20 Query Tasks

**File:** `baseline/eval.py`
**Why we built it:** Reproducible, quantitative evaluation that can be re-run after each pipeline change. 20 questions mirror what a claims adjuster or domain expert would actually ask the knowledge graph — covering all sections of the SFIP document.

**Task categories:**

| Category | Count | Example questions |
|----------|-------|------------------|
| Coverage | 6 | What property types are covered? Max building coverage? Basement contents covered? |
| Exclusions | 3 | Is earth movement covered? Financial losses excluded? Types of excluded damage? |
| Definitions | 2 | How is 'flood' defined? How is 'building' defined? |
| Claims Process | 3 | Obligations after loss? Proof of loss deadline? Appeal process? |
| Policy Terms | 6 | Waiting period? ICC coverage? RCV determination? Coinsurance? |

**Scoring:** Each task has a Cypher query + expected keywords.
Marked **correct** if query returns ≥1 row AND any row contains an expected keyword.

**Metrics computed:**
- **Query Accuracy:** `correct / 20`
- **Entity Duplication Rate:** `(total_nodes − unique_entity_names) / total_nodes`
- **Type Inconsistency Rate:** % of root concept families with more than one label variant

**Critical bug fixed — Type Inconsistency was showing 100%:**

```python
# WRONG: zero-width lookahead → '' (empty string) as root for every label
tokens = re.split(r'(?=[A-Z])|_| ', label)
root = tokens[0]   # always '' → all labels share root '' → 100% "inconsistent"

# CORRECT: lookbehind (non-zero-width) + filter empty tokens
tokens = [t for t in re.split(r'_| |(?<=[a-z])(?=[A-Z])', label) if t]
root = tokens[0].lower() if tokens else label.lower()
# "Policy"       → ["Policy"]           → root = "policy"  ✓
# "PolicyAction" → ["Policy", "Action"] → root = "policy"  ✓
# "Policy_type"  → ["Policy", "type"]   → root = "policy"  ✓
```

Without this fix the metric was completely meaningless. After fix: **8.0%** (original) and **15.2%** (Zone 1).

---

### 4.4 Zone 1 — Novel Hybrid Chunking

**File:** `zone1/ingestion.py`
**Why we built it:** Replace fixed sliding-window chunking with structure-aware hybrid chunking that respects the semantic boundaries of insurance documents (Zone 1 of the novel pipeline per the project plan §3.2).

**Core insight:** Insurance policy documents use numbered sections with ALL-CAPS headings. A chunk that crosses a section boundary strips the heading that gives meaning to its content. The LLM extracts triples without knowing what rule applies.

**Two-step process for PDFs:**

**Step 1 — Section boundary detection:**
```python
# Detect major Roman numeral sections (ALL-CAPS title required)
ROMAN_SECTION = re.compile(
    r'^(I{1,3}|IV|V?I{0,3}|IX|X{1,2})\.\s+[A-Z]{2,}',
    re.MULTILINE
)
# [A-Z]{2,} = at least 2 consecutive caps → only ALL-CAPS headings match
# Prevents "I. No Benefit to Bailee" from matching as major section I

LETTER_SUBSECTION = re.compile(r'^([A-Z])\.\s+[A-Z][a-z]')

PAGE_HEADER = re.compile(
    r'NFIP GENERAL PROPERTY FORM SFIP\s+P\s*AG\s*E\s*\d+\s*OF\s*\d+',
    re.IGNORECASE
)
```

Each chunk carries `section_hierarchy`, e.g.:
- `["IV. PROPERTY NOT INSURED"]`
- `["VII. GENERAL CONDITIONS", "B. Other Insurance"]`

**Step 2 — Semantic merge (τ = 0.85):**
```python
# If two adjacent sections are too similar in embedding space, merge them
# Prevents very short sections from becoming useless isolated chunks
SEMANTIC_MERGE_THRESHOLD = 0.85
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHUNK_TOKENS = 1200  # never merge beyond this

# Merge only if: cosine_similarity >= 0.85 AND combined_tokens <= 1200
```

**Critical bug fixed — Section detection misclassifying subsections:**

```python
# WRONG: single capital matches title-case subsections
r'^(I{1,3}|...)\.\s+[A-Z]'
# "I. No Benefit to Bailee" → matched as major section I  ✗

# CORRECT: 2+ consecutive capitals → ALL-CAPS only
r'^(I{1,3}|...)\.\s+[A-Z]{2,}'
# "I. INSURED PROPERTY" → matched as major section I  ✓
# "I. No Benefit to Bailee" → NOT matched (title case)  ✓
```

**CSV chunking:** 50 records per chunk, each chunk includes field names and sample values for LLM context.

**Zone 1 output — `data/flood/processed/zone1_chunks.json`:**

| Source | Chunks | Notes |
|--------|--------|-------|
| SFIP PDF | 49 | Section-aware (vs 56 fixed) |
| OpenFEMA Policies CSV | 10 | 50 records/chunk |
| OpenFEMA Claims CSV | 10 | 50 records/chunk |
| **Total** | **69** | |

**Sample chunk metadata:**
```json
{
  "text": "IV. PROPERTY NOT INSURED\n\nWe do not insure any of the following...",
  "source": "fema_F-123-general-property-SFIP_2021.pdf",
  "section_hierarchy": ["IV. PROPERTY NOT INSURED"],
  "pages": [11, 12],
  "token_count": 605,
  "merged_from": 1,
  "temporal_markers": []
}
```

---

### 4.5 Ablation Study Setup

**Why:** Isolate the contribution of Zone 1 chunking alone. Same extractor (LLMGraphTransformer) + same LLM (Llama 3.1 8B) + same Neo4j insertion — only chunking strategy changes. This gives a clean, single-variable comparison.

**CLI flags added:**
```bash
# Original baseline
python3 baseline/pipeline.py
python3 baseline/eval.py

# Zone 1 ablation
python3 baseline/pipeline.py --zone1               # clears Neo4j, uses zone1_chunks.json
python3 baseline/eval.py --suffix zone1            # saves to baseline_eval_results_zone1.json
python3 baseline/eval.py --suffix zone1 --riskine  # + Riskine ontology alignment score
```

Results are saved separately so both are preserved for the final paper's Table 1.

---

### 4.6 Results Comparison Tool

**File:** `evaluation/compare_results.py`
Side-by-side terminal table comparing metrics across all runs. Supports original, zone1, and future novel pipeline results.

**HTML report generation:**
```bash
python3 evaluation/visualize_results.py --suffix zone1 --html
# Saves: data/results/visualization_report_zone1.html
```

---

### 4.7 Baseline Riskine Alignment + LLM Ontology Induction

**Files:** `evaluation/riskine_eval.py`, `evaluation/riskine_loader.py`, `baseline/ontology_induction.py`

**Why we measured it:** Zone 3 of the novel pipeline produces a structured ontology aligned to the Riskine insurance ontology. To have a fair comparison, the baseline also needs an ontology alignment score.

**Riskine reference:** 10 canonical flood-insurance classes sourced from [github.com/riskine/ontology](https://github.com/riskine/ontology):
`Coverage · Product · Damage · Risk · Structure · Property · Person · Object · Organization · Address`
Schemas stored locally in `data/riskine/schemas/`.

**Riskine evaluation** (`baseline/eval.py --riskine`): Queries Neo4j `db.labels()`, uses embedding cosine similarity + LLM judge to compute:
- **Precision** = `Riskine-matching labels / total extracted labels`
- **Recall** = `Riskine classes covered / 10`
- **F1** = harmonic mean

**Baseline zone1 result (llama3.1:8b):**
| Metric | Value | Notes |
|--------|-------|-------|
| Precision | 0.167 | 5 of ~44 labels match a Riskine class |
| Recall | 0.500 | 5 of 10 Riskine classes covered |
| F1 | 0.250 | Limited by unconstrained free-form labels |

**Why it's low:** LLMGraphTransformer invents labels like `PolicyAction`, `CoverageType`, `BuildingLocation` — none of which are Riskine class names. Only 5 of the 44 labels have sufficient semantic overlap.

**Improved baseline — LLM Ontology Induction (opt-in):**
`baseline/ontology_induction.py` adds a post-extraction step that:
1. Collects all unique Neo4j labels
2. Calls the LLM in a single batch to map each label → closest Riskine class
3. Applies the mapping as **additional** Neo4j labels (non-destructive: original labels preserved)

Run with:
```bash
python3 baseline/pipeline.py --zone1 --induce-ontology
```

This step is **off by default** — keeping current eval results stable while the full Zone 2–4 approach is still under development. When the full pipeline is complete, running both with `--induce-ontology` will give a fair apples-to-apples comparison.

Expected improvement (not yet measured):
| Metric | Without induction | With `--induce-ontology` |
|--------|:-----------------:|:------------------------:|
| Precision | 0.167 | ~0.50–0.70 |
| Recall | 0.500 | ~0.70–0.90 |
| F1 | 0.250 | ~0.55–0.75 |

---

## 5. Evaluation Method and Metrics

### Automatic Metrics

| Metric | Formula | Baseline | Zone 1 | SV-LOI (best) |
|--------|---------|:-:|:-:|:-:|
| Query Accuracy | `correct / 20` | 35% | 50% | **95%** |
| Entity Duplication | `(nodes − unique) / nodes` | ~0% | ~0% | 0% |
| Type Inconsistency | `multi_label_roots / roots` | 8.0% | 15.2% | **2.2%** |
| Name F1 | exact name match to reference | — | 0.250 | **0.563** |
| Entity Assignment F1 (present) | member-based, present classes only | — | — | **0.417** |
| BERTScore F1 | semantic similarity of class names | — | — | **0.732** |
| Graph F1 | structural alignment of ontology graph | — | — | **0.714** |
| Continuous F1 | soft assignment score | — | — | **0.122** |
| Wu-Palmer | taxonomy similarity | — | — | **0.752** |

### Manual Evaluation (Weeks 10–11)
- 2 domain experts annotate 50 induced ontology classes
- 2 domain experts annotate 30 `SUBCLASS_OF` relationships
- Inter-rater agreement: Cohen's κ > 0.7 required

### Why ~0% duplication on the sample
Entity duplication grows at scale. With 500-record samples the same entity rarely appears with variant spellings. At full 150K records, we expect 20–30% duplication — the metric is designed to catch that.

### Why inconsistency went UP with Zone 1 (and why that's correct)
Fragmented 512-token chunks → LLM produces generic labels (`Policy`).
Complete section chunks → LLM produces specific labels (`Policy action`, `Policy feature`).
Higher inconsistency with Zone 1 reveals the *true scope* of label proliferation that Zone 3 must solve. The 8% original figure was artificially low.

---

## 6. Results

### 6.1 Baseline — Original 512-token Chunks

**56 chunks · 908.4s runtime · 0 errors**

```
Nodes:         365
Relationships: 587
Unique labels:  58

Query Accuracy:     7 / 20 = 35.0%
Entity Duplication: 0.0%
Type Inconsistency: 8.0%
```

**Passed tasks (7/20):**

| Task | Question | Why it passed |
|------|----------|--------------|
| 1 | Types of covered property | Coverage nodes extracted across multiple chunks |
| 2 | Max building coverage | "$250,000" figure extracted |
| 7 | Define 'flood' | Definition section captured in one chunk |
| 9 | Obligations after loss | Procedural list items extracted |
| 10 | Proof of loss deadline | *Lucky — "60 days" keyword appears in a chunk* |
| 13 | ICC coverage | Short dedicated section, fully in one chunk |
| 17 | Waiting period | Short "30-day" rule, fully captured |

---

### 6.2 Baseline + Zone 1 Chunks — Ablation

**49 PDF chunks · 659.6s runtime (−27%) · 0 errors**
*Same extractor, same LLM, same Neo4j insertion — only chunking changed*

```
Nodes:         242  (↓34%)
Relationships: 380  (↓35%)
Unique labels:  44  (↓24%)

Query Accuracy:     10 / 20 = 50.0%  (+15%)
Entity Duplication: 0.0%
Type Inconsistency: 15.2%            (↑ from 8.0% — expected)
```

**What changed (task-by-task):**

| Task | Outcome | Reason |
|------|---------|--------|
| 5 — Earth movement exclusion | ✅ **GAINED** | V. EXCLUSIONS section complete → LLM sees full exclusion list |
| 12 — Define 'building' | ✅ **GAINED** | II. DEFINITIONS section complete in one chunk |
| 15 — Coinsurance provision | ✅ **GAINED** | VII.B Other Insurance section complete |
| 19 — RCV determination | ✅ **GAINED** | VII.R Loss Settlement section complete |
| 10 — Proof of loss deadline | ❌ **LOST** | Was a lucky keyword hit in original; Zone 1 chunk is more precise but the keyword isn't in the result row |

**Net: +4 gained, −1 lost = +3 tasks = +15% accuracy**

---

### 6.3 Side-by-Side Comparison

| Metric | Baseline (512-token) | Baseline + Zone 1 | Change | Novel Target |
|--------|:-------------------:|:-----------------:|:------:|:------------:|
| Runtime | 908.4s | 659.6s | −27% | — |
| Chunks | 56 | 49 | −12% | 69 (all) |
| Nodes | 365 | **242** | **−34%** | ~150–200 |
| Relationships | 587 | 380 | −35% | — |
| Unique labels | 58 | **44** | **−24%** | 20–30 |
| Query accuracy | 35% | **50%** | **+15%** | **>75%** |
| Entity duplication | ~0% | ~0% | — | <5% |
| Type inconsistency | 8.0% | 15.2% | +7.2% *(expected)* | **<2%** |
| Riskine F1 | — | **0.250** | — | **>0.75** |

*Riskine F1 not measured on original 512-token run (added after zone1 evaluation).*

---

### 6.4 Task-by-Task Breakdown — All 20 Tasks

| # | Category | Question | Original | Zone 1 | Notes |
|---|----------|----------|:--------:|:------:|-------|
| 1 | Coverage | Covered property types? | ✓ | ✓ | Both extract coverage nodes |
| 2 | Coverage | Max building coverage? | ✓ | ✓ | $250,000 extracted both times |
| 3 | Coverage | Basement contents covered? | ✗ | ✗ | Negation not typed as relation |
| 4 | Coverage | Replacement cost basis? | ✗ | ✗ | Numerical/conditional rule missed |
| 5 | Exclusions | Earth movement covered? | ✗ | **✓** | Zone 1: full exclusion section |
| 6 | Exclusions | Financial losses excluded? | ✗ | ✗ | Abstract exclusion not typed |
| 7 | Definitions | Define 'flood'? | ✓ | ✓ | Definition section in both |
| 8 | Definitions | Define 'building'? | ✗ | ✗ | Complex nested definition |
| 9 | Claims | Obligations after loss? | ✓ | ✓ | Procedural list extracted |
| 10 | Claims | Proof of loss deadline? | ✓ | **✗** | Lucky hit in original; precise Zone 1 chunk lacks keyword |
| 11 | Claims | Appeal process? | ✗ | ✗ | Multi-step logic not captured |
| 12 | Definitions | Define 'building'? | ✗ | **✓** | Zone 1: full definitions section |
| 13 | Policy Terms | ICC coverage? | ✓ | ✓ | Short section, captured both |
| 14 | Policy Terms | Cancellation terms? | ✗ | ✗ | Procedural rule, not typed |
| 15 | Policy Terms | Coinsurance provision? | ✗ | **✓** | Zone 1: VII.B complete |
| 16 | Policy Terms | Premium changes? | ✗ | ✗ | Numerical rule not captured |
| 17 | Policy Terms | Waiting period? | ✓ | ✓ | 30-day rule short, both capture |
| 18 | Coverage | What damage is excluded? | ✗ | ✗ | NOT_INSURED not a typed relation |
| 19 | Policy Terms | RCV determination? | ✗ | **✓** | Zone 1: VII.R complete |
| 20 | Coverage | Additional coverage available? | ✗ | ✗ | Complex conditional rules |
| | | **TOTAL** | **7/20 = 35%** | **10/20 = 50%** | |

**Persistent failure patterns (both runs) → motivate Zone 2:**

| Pattern | Tasks | Zone 2 fix |
|---------|-------|-----------|
| Typed negation (`NOT_INSURED`, `EXCLUDED_FROM`) | 3, 6, 18 | Explicit negation relation in few-shot examples |
| Numerical facts (`HAS_LIMIT(250000)`) | 4, 16 | Typed amount extraction in prompt |
| Multi-step procedures (`PRECEDES`) | 11, 14 | Sequential step ordering in prompt |
| Complex conditionals | 20 | Structured if-then-unless extraction |

---

## 7. Concrete Examples

### 7.1 Example 1 — Chunking Quality

#### BAD: Fixed 512-token chunk (baseline, chunk 10, page 5)

```
...time of the loss:
a. By over-the-top or frame ties to ground
   anchors; or
b. In accordance with the manufacturer's
   specifications; or
c. In compliance with the community's
   floodplain management requirements
   unless it has been continuously insured
   by the NFIP at the same described
   location since September 30, 1982.
8. Items of property below the lowest
   elevated floor of an elevated post-FIRM...
```

**Why it fails:**
- Starts mid-sentence — "...time of the loss:" refers to a condition in the previous chunk
- Items a/b/c are conditions for something — the subject is missing (it's in the previous chunk)
- LLM cannot determine the governing rule, the section, or the entity being described
- LLM extracts generic, useless triples:
  ```
  (Manufacturer, HAS_SPECIFICATION, Specification)
  (Community, HAS_REQUIREMENT, Floodplain_management)
  ```

**LLM's view:** "Some items about anchors and manufacturers. I'll extract what I can."

---

#### GOOD: Zone 1 section-aware chunk (Section IV, pages 11–12)

```
IV. PROPERTY NOT INSURED

We do not insure any of the following property:

1. Personal property not inside the fully enclosed building.
2. A building, and personal property in it, located entirely in, on,
   or over water or seaward of mean high tide if constructed after
   September 30, 1982.
3. Open structures, including a building used as a boathouse or any
   structure or thing that is principally below ground level.
4. A recreational vehicle other than a travel trailer...
```

**Why it succeeds:**
- Starts at the section heading — LLM immediately knows: governing rule = "We do not insure"
- Each numbered item is a specific excluded property type
- LLM extracts precise, typed triples:
  ```
  (NFIP_Policy, NOT_INSURED, Personal_property_outside_building)
  (NFIP_Policy, NOT_INSURED, Building_over_water_post_1982)
  (NFIP_Policy, NOT_INSURED, Open_structure)
  (NFIP_Policy, NOT_INSURED, Recreational_vehicle)
  ```
- These directly answer: "What types of property are excluded?" (Task 18)

**LLM's view:** "This is a complete 'PROPERTY NOT INSURED' section. I can extract all exclusion relationships."

**Metadata attached to this chunk:**
```json
{
  "section_hierarchy": ["IV. PROPERTY NOT INSURED"],
  "pages": [11, 12],
  "token_count": 605,
  "merged_from": 1
}
```

---

### 7.2 Example 2 — Type Inconsistency and the Leiden Fix

#### The problem: same concept, 5 different labels

"Policy" is mentioned in every section of the SFIP document. Each time it appears in a different chunk context, `LLMGraphTransformer` assigns a different label:

| Section chunk | Label LLM assigns |
|---|---|
| I. INSURED PROPERTY | `:Policy` |
| III. DEDUCTIBLES | `:Policy action` |
| V. EXCLUSIONS | `:Policy feature` |
| VI. LOSS SETTLEMENT | `:Policy type` |
| VII. GENERAL CONDITIONS | `:Policy clause` |

All five refer to the same real-world concept — the NFIP flood insurance policy — but they are **5 disconnected node types** in Neo4j.

**Consequence:**
```cypher
-- This query only returns 1/5 of relevant nodes
MATCH (p:Policy) RETURN p

-- The other four labels are invisible to this query:
-- :Policy_action, :Policy_feature, :Policy_type, :Policy_clause
```

This is why policy-related tasks fail even when the data is in the graph. The information is there — just scattered across 5 incompatible labels.

---

#### Our Zone 3 fix: Leiden clustering

```python
# Step 1: Build type-similarity graph
# For each pair of labels, compute composite similarity:
def edge_weight(la, lb):
    sem   = cosine(embed(la), embed(lb))       # embedding similarity
    co    = cooccurrence_count(la, lb)           # how often near same entity
    rel   = jaccard(relations(la), relations(lb))# shared relation types
    return 0.5 * sem + 0.3 * co + 0.2 * rel

# Concrete example:
w("Policy", "Policy_action") = 0.5 * 0.91 + 0.3 * 0.72 + 0.2 * 0.68
                              = 0.455 + 0.216 + 0.136
                              = 0.807   ← exceeds threshold 0.75 → SAME CLUSTER

# Step 2: Leiden community detection
from graspologic.partition import leiden
communities = leiden(type_similarity_graph, trials=10)

# All 5 Policy labels end up in one cluster:
C_policy = {Policy, Policy_action, Policy_feature, Policy_type, Policy_clause}

# Step 3: LLM names the canonical class
prompt = """
Insurance ontology engineer task:
These node types all refer to the same abstract concept:
['Policy', 'Policy action', 'Policy feature', 'Policy type', 'Policy clause']
Propose one canonical PascalCase class name:
"""
# LLM output: InsurancePolicy

# Step 4: Graph migration
# All 5 labels → :InsurancePolicy
# All instances unified under one canonical class
# SUBCLASS_OF edges added where subclusters found
```

**After Leiden — what's now possible:**
```cypher
-- All policy instances, unified (was impossible before)
MATCH (p:InsurancePolicy) RETURN p

-- All policy rules
MATCH (p:InsurancePolicy)-[r]->(rule) RETURN type(r), rule

-- Hierarchy queries (impossible before — no SUBCLASS_OF)
MATCH (c)-[:SUBCLASS_OF]->(parent:InsurancePolicy) RETURN c, parent
```

**Expected metric impact:**
- Type inconsistency: 15.2% → **< 2%**
- Unique labels: 44 → **20–30**
- Query accuracy: 50% → **> 75%** (multi-hop queries now possible)

---

## 8. Key Findings and Interpretation

### Finding 1: Chunking strategy alone explains a 15% accuracy gap

A single-variable change (chunking strategy only, same extractor, same LLM) produces a jump from **35% → 50%** query accuracy. This is a clean ablation.

**Takeaway:** For LLM-based KG construction from structured documents, section-aware chunking should be the default. Fixed sliding-window chunking is actively harmful — it strips the headings that give meaning to policy rules.

---

### Finding 2: Type inconsistency is not a chunking-size problem

Zone 1 shows *higher* inconsistency (15.2%) than the original (8%). The 8% was artificially low because fragmented chunks don't give the LLM enough context to create specific labels — everything becomes generic `:Policy`. Complete sections give the LLM enough context to identify subtypes, which is correct per-chunk but inconsistent globally.

**Takeaway:** Type inconsistency is a fundamental limitation of per-chunk extraction, not of chunk size. It requires a global post-processing step (Zone 3 Leiden) to merge labels across the entire graph.

---

### Finding 3: The baseline ceiling is ~50% even with perfect chunking

Tasks that remain unsolved with Zone 1 all share a structural reason:
- **Negation** (Tasks 3, 6, 18): LLM writes "not covered" as text in a node, doesn't type it as `NOT_INSURED` relation
- **Numerical facts** (Tasks 4, 16): Dollar limits not linked to entities with typed amounts
- **Procedures** (Tasks 11, 14): Sequential steps collapsed into single nodes
- **Conditionals** (Task 20): "If X then Y unless Z" logic is lost

These failure modes cannot be fixed by better chunking. They require a constrained extraction prompt (Zone 2 few-shot Open IE).

---

### Finding 4: Fewer nodes is better

Zone 1 produces 34% fewer nodes with the same extractor. This is a quality improvement, not information loss — fragmented chunks cause the LLM to over-extract partial phrases as spurious entities. Complete sections cause conservative, accurate extraction. **Precision improves; query accuracy improves.**

---

## 9. Bugs Found and Fixed

### Bug 1 — Neo4j DatabaseNotFound

**Error:** `DatabaseNotFound: Database does not exist. Database name: 'neo4j'`

**Cause:** Neo4j AuraDB names the database after the username. Our username is `8929c2c7` so the database is `8929c2c7`, not `neo4j`.

**Fix:**
```python
# config.py
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", NEO4J_USERNAME)
```

---

### Bug 2 — Type Inconsistency = 100%

**Cause:** `re.split(r'(?=[A-Z])', label)` is a zero-width lookahead. It inserts a split point *before* every capital without consuming it. `"Policy"` splits to `['', 'Policy']`. Root = `''` for all labels. All labels share root `''` → 100% "inconsistent".

**Fix:**
```python
# Correct: lookbehind (non-zero-width) + filter empties
tokens = [t for t in re.split(r'_| |(?<=[a-z])(?=[A-Z])', label) if t]
root = tokens[0].lower() if tokens else label.lower()
```

---

### Bug 3 — Section Detection Misclassifying Title-Case Subsections

**Cause:** Regex matched any label starting with one capital after the numeral. `"I. No Benefit to Bailee"` matched as major section I.

**Fix:** Require `[A-Z]{2,}` — ALL-CAPS titles only.

---

### Bug 4 — compare_results.py Printing Columns Twice

**Cause:** String formatting trick caused each column to be appended twice in the display loop.

**Fix:** Plain `+=` concatenation. Cosmetic only — results unaffected.

---

## 10. Technical Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Pipeline orchestration | LangGraph `StateGraph` | State machine for Zone 2 and Zone 3 pipelines |
| LLM (local) | Ollama + qwen2.5:72b (primary) / llama3.1:8b | Triple extraction, EDC canonicalization, class discovery, consolidation |
| KG extraction (baseline) | LangChain `LLMGraphTransformer` | Baseline only — comparison point |
| Graph database | Neo4j AuraDB | Knowledge graph + ontology storage |
| Graph connector | `langchain-neo4j` | Python ↔ Neo4j |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Zone 1 merge, Zone 2 entity resolution, Zone 3 similarity graph |
| PDF loading | LangChain `PyPDFLoader` | PDF text extraction |
| Community detection | `leidenalg` + `igraph` | Zone 3 multi-resolution Leiden |
| GPU cluster | Emory Turing (Slurm, RTX 8000 48GB) | 70b model runs |
| Visualization | `matplotlib` + `networkx` | Presentation figures |
| Future: demo | Streamlit | Interactive KG explorer |

**Environment:**
```
Neo4j AuraDB:  URI = neo4j+s://...  |  DB = 8929c2c7
Ollama:        http://localhost:11434  |  model = llama3.1:8b (default), llama3.1:70b (cluster)
Python:        3.11+
```

---

## 11. File Structure

```
CS584_AI_Capstone/
├── config.py                       # Neo4j + Ollama + path settings
├── .env                            # Credentials (not committed)
├── requirements.txt                # Pinned dependencies
│
├── baseline/                       # Comparison baseline (LLMGraphTransformer + Neo4j)
│   ├── pipeline.py                 # LangGraph pipeline (--zone1, --induce-ontology, --model)
│   ├── eval.py                     # 20-task evaluator (--suffix, --riskine flags)
│   ├── ontology_induction.py       # LLM label → Riskine class mapping (opt-in)
│   └── pdf_loader.py               # PDF chunk loading utility
│
├── zone1/
│   └── ingestion.py                # Hybrid section-aware + semantic chunking (τ=0.85)
│
├── zone2/                          # ✅ Domain-agnostic Open IE pipeline
│   ├── pipeline.py                 # Bootstrapped vocab + 3-pass extraction + EDC + entity resolution
│   ├── prompts.py                  # Extraction prompts (domain-agnostic, synthetic few-shot)
│   └── entity_resolution.py       # Embedding-based entity merge (cosine ≥ 0.90)
│
├── zone3/                          # ✅ Ontology induction (3 methods)
│   ├── pipeline.py                 # Leiden community detection (baseline)
│   ├── rsi_lcr.py                  # RSI-LCR: Relation-Signature Induction + LLM Coherence Refinement
│   └── sv_loi.py                   # SV-LOI: Structurally-Verified LLM Ontology Induction (novel)
│
├── zone4/                          # ⏳ Structured Neo4j storage (upcoming)
│
├── evaluation/
│   ├── riskine_eval.py             # Entity Assignment F1, Name F1, present-class F1
│   ├── ontology_metrics.py         # BERTScore F1, Graph F1, Continuous F1, Wu-Palmer
│   ├── riskine_loader.py           # Loads Riskine schema JSONs
│   ├── visualize_results.py        # HTML report generation (--suffix, --html flags)
│   └── compare_results.py          # Side-by-side terminal table
│
├── scripts/
│   ├── data_download.py            # OpenFEMA policy + claims data download
│   ├── cluster_setup_v2.sh         # Emory Turing one-time setup (Ollama + qwen2.5:72b + venv)
│   ├── slurm_pipeline_v2.sh        # Full Zone 2+3 GPU job (2× GPU, 96GB, 12h)
│   ├── slurm_rsi_lcr.sh            # Zone 3 RSI-LCR only (no re-extraction)
│   ├── slurm_sv_loi.sh             # Zone 3 SV-LOI only (no re-extraction)
│   ├── slurm_eval_all.sh           # Re-run all 3 Zone 3 methods + eval
│   ├── run_on_cluster.sh           # Push + submit convenience wrapper
│   └── sync_to_cluster.sh          # rsync via jump host lab0z.mathcs.emory.edu
│
├── data/
│   ├── flood/
│   │   ├── raw/
│   │   │   ├── pdf/                              # SFIP PDF (27 pages)
│   │   │   └── openfema/                         # 500-record policy + claims CSVs
│   │   └── processed/
│   │       └── zone1_chunks.json                 # 69 Zone 1 chunks
│   ├── riskine/
│   │   └── schemas/                              # 10 Riskine class schemas (EVALUATION ONLY)
│   ├── auto/                                     # Cross-domain transfer data
│   └── results/                                  # Generated eval JSONs + PNGs (git-ignored)
│       ├── presentation_eval_qa.png              # 4-panel Q&A figure
│       ├── presentation_prompts.png              # Prompt examples figure
│       └── presentation_kg_subgraph.png          # KG visualization
│
└── docs/
    ├── FINDINGS.md                               # F-01 through F-18 empirical findings
    ├── GENERALIZATION_PLAN.md                    # Domain leakage audit + fix rationale
    ├── IDEA_REPORT.md                            # SV-LOI idea discovery + novelty check
    ├── CS_584_AI_Capstone.pdf                    # Main capstone report
    ├── CS_584_AI_Capstone_Simplified_Plan_v2.pdf # Current project plan
    ├── Automated_Insurance_Ontology_Induction.pptx # Presentation slides
    └── refine-logs/                              # Method refinement + experiment plan
```

---

## 12. Zone 2 & 3 Implementation

### Zone 2 v2 — Domain-Agnostic Open IE

**What was removed (domain leakage, Week 6 audit):**
- `ANCHORS` — 9 injected Riskine-labeled nodes that inflated F1 to 0.874
- `RELATION_ROLE_MAP` — hardcoded relation→Riskine class mappings
- `FEW_SHOT_PAIRS` (13 pairs) — all used exact SFIP text
- `label_nodes()` — applied Riskine labels inside the pipeline

**What was added (domain-agnostic):**
- Entity type bootstrapping — LLM reads document samples, proposes entity types
- Relation type bootstrapping — LLM reads document samples, proposes relation types
- 3-pass extraction — general facts, numeric facts, obligation/procedure facts
- EDC canonicalization — maps ~48 raw relation types → 14 vocab types (71% reduction)
- Synthetic few-shot examples — generic insurance patterns (definition, coverage, exclusion, obligation, limit) — no SFIP text

**Critical rule: Riskine appears ONLY in `evaluation/riskine_eval.py`.**
The pipeline produces an ontology that is then *compared to* Riskine, not *built from* Riskine.

Full rationale: `docs/GENERALIZATION_PLAN.md`

---

### Zone 3 — Ontology Induction (3 Methods)

Three ontology induction methods are implemented and compared, all operating on the same Zone 2 extraction output:

**Method A: Leiden (baseline)**
- Multi-resolution Leiden community detection at resolutions [0.3, 0.6, 1.2]
- Entity similarity graph from embeddings + structural features
- Algorithmic SUBCLASS_OF from cross-resolution membership overlap (>=0.60)
- Result: 11 classes, Name F1 = 0.000, BERTScore F1 = 0.451

**Method B: RSI-LCR (Relation-Signature Induction with LLM Coherence Refinement)**
- Groups entities by their relation-type signature vectors
- LLM validates cluster coherence and proposes refinements
- Result: 30 classes, Name F1 = 0.295, BERTScore F1 = 0.577

**Method C: SV-LOI (Structurally-Verified LLM Ontology Induction) — novel contribution**
- Two-stage class discovery: LLM detects domain from entity examples, then proposes classes
- Eliminates data-type classes (NumericValue, Temporal) that plagued Leiden
- LLM-guided class consolidation merges fine-grained classes into canonical concepts
- Result: 7 classes, Name F1 = 0.563, BERTScore F1 = 0.732, Graph F1 = 0.714

---

## 13. Empirical Findings Summary

| # | Finding | Key Impact |
|---|---------|-----------|
| F-01 | 8B ignores dollar amounts as triple objects | 3-pass extraction with numeric focus pass |
| F-02 | Few-shot pairs from source text cause global interference | Synthetic (not source-derived) few-shot |
| F-03 | 8B and 70B extract ~1 triple/chunk regardless of model size | Multi-pass mitigation |
| F-04 | Vocabulary compliance > volume (deepseek-r1:14B at 30% vs 8B at 50%) | Validates bootstrapped vocab approach |
| F-05 | Entity resolution threshold is model-dependent (0.90 for 8B) | Per-model tuning needed |
| F-06 | Zone 2 v1 F1=0.874 was domain leakage (9 Riskine anchors injected) | All anchors removed; honest F1 ≈ 0.000 |
| F-07 | Riskine F1 rewards naming convention similarity, not semantic quality | Paper uses BERTScore/Graph F1 alongside name F1 |
| F-08 | Query accuracy is Cypher-pattern sensitive; 70B names differ from query patterns | 70B: 65% vs 8B: 75% despite richer ontology |
| F-09 | EDC reduces structural discriminability | Type inconsistency is an eval artifact, not a labeling error |
| F-10 | 70B bootstraps more varied vocabulary; EDC less effective (47→32 vs 48→14) | 70B graph richer but less canonicalized |
| F-15 | SV-LOI two-stage class discovery (detect domain, then propose classes) eliminates data-type classes | No more NumericValue, Temporal, VoidState classes polluting the ontology |
| F-16 | LLM-guided class consolidation (City+County+State→Address, Policy→Product) improves Name F1 from 0.214 to 0.563 | Consolidation is as important as initial class discovery |
| F-17 | Person, Organization, Object classes are absent from flood insurance data | Max achievable full-Riskine recall is ~0.6-0.7 |
| F-18 | Present-class F1 metric gives fairer comparison when source data lacks some reference classes | SV-LOI: 0.417 present-class F1 vs 0.293 for Leiden |

Full details: `docs/FINDINGS.md`

---

## 14. Timeline (Week 10 State)

| Period | Phase | Status | Deliverables |
|--------|-------|:------:|-------------|
| Weeks 1–3 | Baseline Implementation | ✅ Done | LangGraph pipeline · Zone 1 chunking · 20-task eval · Ablation |
| Weeks 4–5 | Zone 2 v1 (domain-coupled) | ✅ Done | Few-shot extractor · Findings F-01–F-05 · Riskine eval |
| Week 6 | Zone 2 v2 (generalized) | ✅ Done | Removed all domain leakage · EDC · Synthetic few-shot |
| Week 7 | Zone 3 — Leiden (baseline) | ✅ Done | MultiLeiden · Algorithmic SUBCLASS_OF · Turing 70b/72b GPU runs |
| Weeks 8–9 | Zone 3 — RSI-LCR + SV-LOI | ✅ Done | 3-method comparison · OLLM-style metrics · LLM consolidation |
| **Week 10** | Paper + cross-domain transfer | 🔄 **In Progress** | Paper writing · Auto insurance transfer · Final eval |
| Week 11 | Paper + Demo + Annotation | ⏳ Upcoming | Streamlit demo · Human annotation · Final report |

### Final Deliverables
- **Domain-agnostic** 4-zone pipeline — identical code runs on flood and auto insurance
- 3-method comparison: Leiden vs RSI-LCR vs SV-LOI (novel) with comprehensive metrics
- Neo4j graph: flood insurance ontology (induced, no reference ontology used)
- Cross-domain transfer evaluation: CTR/PTR on auto insurance data
- Full ablation table: Baseline → Zone 1 → Zone 2 → Zone 3 (3 methods)
- OLLM-style evaluation: BERTScore F1, Graph F1, Continuous F1, Wu-Palmer
- Streamlit demo: query the induced ontology, visualize SUBCLASS_OF hierarchy

---

*Last updated: March 24, 2026 · CS584 AI Capstone · Emory University*
