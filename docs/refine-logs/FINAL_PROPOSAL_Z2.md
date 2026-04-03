# Zone 2 Fix: Decompose-then-Extract for Prose Chunks

## Root Cause (empirically confirmed)
- `json_mode=True` + `temp=0` → qwen2.5:72b produces 1-item JSON arrays for prose
- 35/46 PDF chunks get exactly 1 triple; 8 list/enumeration chunks get 10-26
- Recall pass (showing prior triples) also gets +1 per chunk — same grammar constraint
- Known Ollama/llama.cpp behavior: JSON grammar biases toward early `]` at greedy decoding

## Literature
- **CoDe-KG (EMNLP 2025)**: Sentence decomposition → +20% recall on rare relations, frozen weights
- **SCIR (Dec 2025)**: Iterative refinement with dual-path detection, +5% F1
- Codex reviewer confirmed: decomposition is the right fix, temp alone is unreliable

## Method: Two-Stage Prompting

### Stage 1 — Fact Decomposition (NO json_mode)
```
Read this insurance policy passage and list ALL factual statements as numbered items.
Each fact must be an explicit statement from the text — quote or anchor to specific text.
Include: definitions, coverage rules, exclusions, conditions, obligations, deadlines.
List EACH fact separately, one per line. Do not summarize or generalize.

Passage: {chunk_text}

Facts:
1.
```
Free-form output → no grammar constraint → model lists 5-15 facts instead of stopping at 1.

### Stage 2 — Per-Fact Triple Extraction (json_mode OK)
```
Extract one (subject, relation, object) triple from this fact.
Relation types: {vocab}
Fact: "{single_fact}"
Output: {"subject": "...", "relation": "...", "object": "...", "span": "...", "confidence": 0.9}
```
json_mode for single object (not array) works fine — minimal-output bias is correct here.

### Routing
- **List chunks** (Pass 1 extracted ≥5 triples): keep current extraction (already works)
- **Prose chunks** (Pass 1 extracted ≤2 triples): run decompose-then-extract
- This avoids wasting LLM calls on chunks that already extract well

### Grounding Verification
- Stage 1 facts filtered: reject facts < 10 words or not anchored in source text
- Stage 2 triples verified via existing span-grounding check
- Hallucination mitigation: "quote or anchor each fact to exact text" in Stage 1 prompt

## Expected Outcome
- 35 prose chunks × ~8 facts each = ~280 new triples (vs current 35)
- Fact Recall: 40% → **55-60%**
- Triple Precision: 82.6% → **≥78%** (grounded facts → grounded triples)
- Compute: +10 min (~315 additional LLM calls at ~2s each)

## Implementation
1. Add `DECOMPOSITION_PROMPT` and `SINGLE_FACT_EXTRACTION_PROMPT` to `prompts.py`
2. Add `_decompose_then_extract()` function to `pipeline.py`
3. Route prose vs list chunks based on Pass 1 triple count
4. Replace the failed recall pass with decompose-then-extract
