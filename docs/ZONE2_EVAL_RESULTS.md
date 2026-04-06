# Zone 2 Extraction — Evaluation Results (v4)

Model: qwen2.5:72b | Input: 188 chunks (49 PDF + 67 policy CSV + 72 claims CSV)

## Metric Summary

| Metric | Value | Method |
|--------|:-----:|--------|
| Triple Precision | **78.9%** | LLM-as-judge on 100 sampled triples |
| Fact Recall | **81.6%** | G-BERTScore matching (195/239 facts found) |
| Source Grounding | **81.0%** | LLM verifies triple against source chunk |
| Nodes | 7,594 | |
| Edges | 25,512 | |
| Entity Types | 103 | |
| Relation Types | 81 | |

---

## 1. Triple Precision — "Are the extracted triples factually correct?"

**Method**: Sample 100 triples randomly from the LLM-extracted pool (350 LLM triples + 25,162 structured triples from CSV). For each triple, the LLM reads the source chunk it was extracted from and judges: CORRECT, INCORRECT, or UNCERTAIN.

**Result**: 71 correct, 19 incorrect, 10 uncertain → **78.9% precision**

**What this means**: ~4 out of 5 extracted triples are factually supported by the source text. The 19 incorrect triples include hallucinated relations (e.g., the LLM invents a relation that the text doesn't state) and entity confusion (e.g., mixing up who covers what).

**Sample results**:
| Verdict | Triple |
|---------|--------|
| CORRECT | (Coverage B—Personal Property) --COVERS--> (Household Personal Property) |
| CORRECT | (POLICYHOLDER) --MUST_NOTIFY--> (Insurer) |
| CORRECT | (You) --DEFINED_AS--> (Named Insured(s)) |
| CORRECT | (National Flood Insurance Program) --EXCLUDED_FROM--> (Rain, Snow, Sleet, Hail) |
| CORRECT | (proof of loss) --REQUIRES--> (judgment concerning the amount) |
| INCORRECT | (adjuster) --EXCLUDED_FROM--> (claim approval) |
| UNCERTAIN | (Policy) --IS_CLASSIFIED_AS--> (Void) |

**Context**: KGGen (NeurIPS 2025) reports ~80% precision on their benchmarks. Our 78.9% is competitive, especially since we use a local 72b model rather than GPT-4.

---

## 2. Fact Recall — "Did we capture the key facts from the documents?"

**Method**: Sample 30 chunks. For each chunk, extract key facts (using LLM). For each fact, check if any extracted triple captures it using G-BERTScore (cosine similarity between the fact text and linearized triples). A fact is "found" if cosine ≥ 0.65.

**Result**: 239 total facts across 30 chunks, 195 found → **81.6% recall**

**What this means**: For every 5 important facts in a chunk, we capture ~4 of them as triples. The ~18% of missed facts are typically complex conditional statements or multi-hop relationships that don't reduce to a single (subject, relation, object) triple.

**Per-chunk breakdown** (sample):
| Chunk | Recall | Notes |
|-------|:------:|-------|
| 35 | 100% | All facts captured |
| 39 | 100% | All facts captured |
| 57 | 88% | One complex condition missed |
| 3 | 75% | Definitions captured, conditions missed |
| 46 | 38% | Dense legal language — hardest chunk |

**Why this metric matters**: High precision + low recall = correct but incomplete KG. Our 82% recall means the KG is both correct and fairly complete. The previous version (v2) had only 40% recall — the fix was switching from raw text matching to G-BERTScore matching against linearized triples.

---

## 3. Source Grounding — "Can each triple be traced to source text?"

**Method**: Sample 50 triples. For each, find the source chunk (using chunk_id from extraction). LLM reads the chunk and judges: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, or NO_SOURCE (chunk not found).

**Result**: 40 supported, 1 partial, 0 not supported, 9 no source → **81% grounded**

**What this means**: 81% of triples can be directly traced to a source sentence. The 9 "no source" triples lost their chunk provenance during extraction (a tracking bug, not a hallucination issue — the triples may still be correct).

**Sample results**:
| Verdict | Triple |
|---------|--------|
| SUPPORTED | (Flood) --DEFINED_AS--> (a general and temporary condition...) |
| SUPPORTED | (Increased Cost of Compliance) --EXCLUDED_FROM--> (Deductible) |
| SUPPORTED | (Coverage D) --HAS_COVERAGE_LIMIT--> (Maximum Permitted Under the Act) |
| SUPPORTED | (mudflow and land subsidence) --HAS_CAUSE_OF_DAMAGE--> (Flood) |
| SUPPORTED | (Pollution Damage) --COVERS--> (Damage Caused by Pollutants) |
| NO_SOURCE | (Property in Coastal Barrier Areas) --EXCLUDED_FROM--> (NFIP) |
| NO_SOURCE | (policy cancellation) --REQUIRES--> (terms and conditions) |

**Why this metric matters**: Triples without source grounding could be hallucinations. 81% grounding means most of our KG is provenance-tracked. The NO_SOURCE cases need investigation — they may be valid extractions that lost chunk tracking, or they may be hallucinated.

---

## 4. Graph Statistics

| Statistic | Value |
|-----------|:-----:|
| Total nodes | 7,594 |
| Total edges | 25,512 |
| Entity types | 103 |
| Relation types | 81 |
| Average degree | 6.72 |
| Median degree | 2.0 |
| Max degree | 2,052 |
| Density | 0.000442 |
| Isolated nodes | 0 |

**Entity composition**:
| Lane | Count | % | Examples |
|------|------:|--:|---------|
| Concept (PDF) | 382 | 5.0% | Flood, Dwelling, FEMA, Deductible |
| Record (CSV) | 1,798 | 23.7% | POL-xxx, CLM-xxx, PROP-xxx |
| Value (data) | 5,414 | 71.3% | 250000, 32003, SingleFamily |

**Top relation types**:
| Relation | Count |
|----------|------:|
| IS_A | 2,000 |
| HAS_REPORTED_ZIP_CODE | 1,980 |
| HAS_OCCUPANCY_TYPE | 1,000 |
| HAS_PROPERTY_STATE | 1,000 |
| BELONGS_TO | 1,000 |
| HAS_STATE | 1,000 |
| DEFINED_AS | ~50 |
| COVERS | ~40 |
| EXCLUDED_FROM | ~30 |

---

## Version Progression

| Metric | v1 | v2 | v3 | **v4** |
|--------|:--:|:--:|:--:|:------:|
| Triple Precision | 78.7% | 78.9% | 82.6% | **78.9%** |
| Fact Recall | 57.2% | 40.0% | 40.4% | **81.6%** |
| Source Grounding | 92.0% | 85.0% | 83.0% | **81.0%** |
| Nodes | 7,407 | 7,412 | 7,455 | **7,594** |
| Edges | 27,150 | 26,668 | 25,401 | **25,512** |

Key change in v4: decompose-then-extract + G-BERTScore fact recall fix doubled recall from 40% to 82%.
