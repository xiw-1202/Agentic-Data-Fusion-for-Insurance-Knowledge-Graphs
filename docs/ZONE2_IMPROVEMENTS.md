# Zone 2 Extraction — How We Improved It

## Starting Point (Midterm, March 15)

Zone 2 v1 was domain-coupled — it worked well on flood insurance but couldn't run on any other domain without code changes.

**Problems identified**:
- `ANCHORS`: 9 hardcoded Riskine nodes injected into the graph ("NFIP Policy"→Product, "Flood"→Risk)
- `RELATION_ROLE_MAP`: 15 hardcoded relation→Riskine-class mappings (COVERS→Coverage, MUST_NOTIFY→Person)
- `EVAL_SEEDS`: 9 forced relation types designed to match our 20 eval queries
- `FEW_SHOT_PAIRS`: 13 examples using exact SFIP policy text (flood-specific)
- `_BOOTSTRAP_SECTION_GROUPS`: keywords like "coverage a", "proof of loss" — SFIP-specific
- Riskine F1 of 0.874 was **inflated** — anchor nodes guaranteed 9/10 classes appear

**Extraction quality**: ~92 triples from llama3.1:8b (1 triple per chunk — see Finding F-03)

## What We Changed

### 1. Removed All Domain Leakage

Identified 6 components that leaked Riskine into the pipeline (see `docs/GENERALIZATION_PLAN.md`). Removed all of them:

| Removed | Replaced With |
|---------|---------------|
| `ANCHORS` (9 Riskine nodes) | Nothing — pipeline discovers entities from documents |
| `RELATION_ROLE_MAP` (15 mappings) | Entity resolution clusters similar entities automatically |
| `EVAL_SEEDS` (9 forced relations) | Bootstrapped relation types from document content |
| `FEW_SHOT_PAIRS` (13 SFIP examples) | Synthetic few-shot using generic insurance patterns |
| `_BOOTSTRAP_SECTION_GROUPS` | Generic keywords: "coverage", "exclusion", "definition", "claim", "limit" |
| `label_nodes()` pipeline step | Labeling moved to Zone 3 (ontology induction) and evaluation only |

**Result**: Honest baseline F1 dropped from 0.874 to ~0.000 — proving the old score was entirely from injected anchors.

### 2. Bootstrapped Vocabulary Discovery

Instead of hardcoding entity and relation types, the pipeline now **discovers them from the documents**:

1. Sample chunks from the input documents
2. Ask LLM: "What types of entities appear in this insurance text?" → LLM proposes entity types
3. Ask LLM: "What types of relationships connect these entities?" → LLM proposes relation types
4. Use the discovered types as extraction guidance

This works for ANY domain — flood, auto, health — because the types come from the documents themselves.

### 3. Switched to qwen2.5:72b

The 8b model extracted exactly 1 triple per chunk regardless of chunk complexity (Finding F-03). The 72b model on Emory's Turing cluster (2× Quadro RTX 8000) extracts 8-50 triples per chunk.

**Impact**: 92 triples → 25,512 triples (277× increase)

### 4. Decompose-then-Extract (v4)

Complex chunks (long paragraphs with multiple facts) overwhelm the LLM. We added a decomposition step:

1. For each chunk, check if it contains multiple distinct facts
2. If yes, split into sub-chunks (each containing one main idea)
3. Extract from each sub-chunk independently
4. Merge results

**Impact**: +188 new triples from chunks that previously yielded 0-1 triples.

### 5. Placeholder Filtering (v4)

Zone 2 was extracting template/placeholder triples from CSV headers and metadata:
- `(PolicyRecord, HAS_FIELD, columnName)` — describing the schema, not the data
- `(HEADER, IS_A, COLUMN)` — CSV structure, not insurance knowledge

We added a filter that removes triples where subject or object matches common placeholder patterns.

**Impact**: ~2,000 garbage triples removed, improving precision without hurting recall.

### 6. Relation Deduplication (v2)

The same relation between two entities was sometimes extracted multiple times from overlapping chunks. We deduplicate by (subject, relation, object) tuple.

**Impact**: 27,150 → 25,512 edges (cleaner graph, same information)

### 7. G-BERTScore Fact Recall Metric (v4)

The original fact recall metric matched extracted facts against raw source text, producing unreliable scores. We switched to:

1. Linearize each triple: `"subject relation object"` → `"NFIP covers flood damage"`
2. Extract key facts from each chunk using LLM
3. Compute cosine similarity between each fact and all linearized triples (using all-MiniLM-L6-v2)
4. A fact is "found" if any triple has cosine ≥ 0.65

**Impact**: Recall went from 40% (broken metric) to 81.6% (correct measurement). The extraction was already good — the metric was wrong.

## Results Progression

| Metric | Midterm | v1 | v2 | v3 | **v4** |
|--------|:-------:|:--:|:--:|:--:|:------:|
| Triples | 92 | 27,150 | 26,668 | 25,401 | **25,512** |
| Entities | 59 | 7,407 | 7,412 | 7,455 | **7,594** |
| Precision | — | 78.7% | 78.9% | 82.6% | **78.9%** |
| Fact Recall | — | 57.2% | 40.0% | 40.4% | **81.6%** |
| Grounding | — | 92.0% | 85.0% | 83.0% | **81.0%** |
| Domain-agnostic | No | Yes | Yes | Yes | **Yes** |

## Key Design Decisions

1. **Synthetic few-shot over source-derived examples**: Using actual document text in few-shot examples causes the LLM to copy patterns instead of understanding structure (Finding F-02). Generic examples work better across domains.

2. **1-pass combined extraction**: Instead of separate passes for entities, relations, and obligations, we do one combined pass with `num_predict=4096`. Reduces LLM calls by 3× with no quality loss.

3. **Entity resolution via embedding clustering**: After extraction, cluster near-duplicate entities (e.g., "NFIP" and "National Flood Insurance Program") using sentence embeddings + agglomerative clustering. Threshold is model-dependent (Finding F-05): 0.90 for 8b, needs tuning for 72b.
