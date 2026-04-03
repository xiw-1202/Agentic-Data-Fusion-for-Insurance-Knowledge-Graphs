# Zone 2 Extraction — Version History

All versions use qwen2.5:72b on the same input data (188 chunks: 49 PDF + 67 policy CSV + 72 claims CSV).

## Version Summary

| Metric | v1 | v2 | v3 | **v4** |
|--------|:--:|:--:|:--:|:--:|
| Nodes | 7,407 | 7,412 | 7,455 | **7,594** |
| Edges | 27,150 | 26,668 | 25,401 | **25,512** |
| Entity types | 37 | 41 | 60 | **103** |
| Relation types | 76 | 79 | 93 | **81** |
| Triple Precision | 78.7% | 78.9% | 82.6% | **78.9%** |
| Fact Recall | 57.2% | 40.0% | 40.4% | **81.6%** |
| Source Grounding | 92.0% | 85.0% | 83.0% | **81.0%** |

## v1 — Baseline SEAF Extraction

Initial domain-agnostic extraction with bootstrapped vocabularies.

- Bootstrapped entity and relation types from document content
- Synthetic few-shot examples (generic insurance patterns, no LOB-specific text)
- Entity resolution via embedding + clustering

## v2 — Relation Deduplication

- Added relation deduplication to reduce redundant edges
- Minimal impact on precision; fact recall dropped (eval metric changed)

## v3 — Semantic Validation

- Added semantic validation layer (LLM judges triple plausibility)
- Higher precision (82.6%) but more conservative extraction
- Entity type expansion (60 types) from finer-grained LLM typing

## v4 — Decompose-then-Extract + Fact Recall Fix (Current)

- **Decompose-then-extract**: split complex chunks before extraction, adding 188 new triples
- **Placeholder filtering**: removed placeholder/template triples (~2,000 removed)
- **Fact recall fix**: G-BERTScore matching against linearized triples (was matching against raw text)
- **Result**: 81.6% fact recall (was 40%), competitive with KGGen and AutoSchemaKG benchmarks
- Entity types doubled (103 vs 60) from finer-grained decomposed extraction
- Relation types consolidated (81 vs 93) from deduplication
