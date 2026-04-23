# SV-LOI Refactor — Design

**Date:** 2026-04-21
**Target:** `zone3/sv_loi.py` (4188 lines)
**Motivation:** Developer velocity (B) + code quality (C). Final presentation in 2 weeks.
**Scope:** `sv_loi.py` only — no changes to `recursive_induction.py`, `graph_cache.py`, `rsi_lcr.py`, `pipeline.py`.

## Problem

`zone3/sv_loi.py` is 4188 lines in a single module. It contains ~30 top-level functions across 8 pipeline phases plus a ~900-line SOHD heterogeneity-detection block. The file is edited in nearly every commit touching zone3 and is slow to navigate, hard to hold in context, and mixes concerns (Neo4j IO, LLM prompts, graph math, orchestration).

`recursive_induction.py` imports 9 symbols from it, so any refactor must preserve the public API exactly.

There are no tests. Validation is by rerunning the pipeline end-to-end and diffing Neo4j output against a baseline.

## Goals

1. Split `sv_loi.py` into ~8 focused modules, each 250–900 lines.
2. Preserve the public API: every symbol currently imported by `recursive_induction.py` (and `run_sv_loi`, `__main__`) remains importable from `zone3.sv_loi`.
3. Land incrementally — each migration step is an independently validatable commit.

## Non-Goals

- No logic changes. Function bodies are moved verbatim; only import paths change.
- No signature changes. Every function keeps its name, arguments, and return type.
- No bug fixes or feature work mixed in. If a bug is spotted, it is logged and fixed in a separate follow-up PR.
- No new tests. A baseline Neo4j + F1 snapshot is the regression check.
- `recursive_induction.py` is not modified.

## Architecture

New layout: `zone3/_svloi/` (leading underscore signals "internal, import via facade").

```
zone3/
├── sv_loi.py              # ~30-line facade: re-exports the public API
└── _svloi/
    ├── __init__.py
    ├── constants.py       # ~100 lines — vocab maps, prefixes, forbidden names
    ├── utils.py           # ~200 lines — get_llm, get_neo4j_graph, _sanitize_*, _parse_json_safely, _invoke_llm, load_entities
    ├── typing.py          # ~900 lines — Phase 1+2+3: analyze_record_evidence, discover_class_vocabulary, batch_type_entities, rescue_other_entities, type_value_entities, rebalance_mega_classes, propagate_to_records
    ├── structural.py      # ~400 lines — build_structural_signatures, structural_consensus_check, arbitrate_disagreements
    ├── hierarchy.py       # ~700 lines — infer_class_relations, merge_small_classes, merge_leaf_classes, derive_interclass_edges, derive_hierarchy, derive_taxonomy, derive_taxonomy_llm_pairwise, naming helpers
    ├── sohd.py            # ~900 lines — detect_and_split_heterogeneous_classes and its helpers (_build_class_relation_profiles, _cosine_similarity_matrix, _js_divergence, _top_distinguishing_relations, _name_subclass_llm, _auto_name_subclass)
    ├── records.py         # ~250 lines — decompose_records, write_record_decomposition
    ├── writer.py          # ~350 lines — validate_backbone, write_ontology, _compute_intrinsic_quality, _flush_print
    └── pipeline.py        # ~600 lines — run_sv_loi orchestrator, CLI entry (__main__)
```

### Why this grouping

- **SOHD isolated** (900 lines rarely touched) — biggest velocity win; removes it from every unrelated edit context.
- **Typing stays together** — Phases 1+2+3 share class vocabulary state; splitting them would create artificial boundaries.
- **Structural = consensus + arbitration** — both consume relation signatures; arbitration is a short LLM step that logically follows consensus flagging.
- **Hierarchy = merge + derive** — merging small/leaf classes is part of the IS-A derivation flow.
- **Writer separate from records** — `records.py` handles record decomposition (Phase 7); `writer.py` handles final Neo4j ontology write (Phase 15). Different output targets.

## Public API (Facade)

`zone3/sv_loi.py` becomes a thin re-export facade:

```python
# zone3/sv_loi.py (post-refactor)
"""Facade preserving the public API. See zone3/_svloi/ for implementation."""
from zone3._svloi.utils import (
    get_llm, get_neo4j_graph, _invoke_llm, _sanitize_label, _parse_json_safely,
)
from zone3._svloi.typing import type_value_entities, propagate_to_records
from zone3._svloi.hierarchy import derive_interclass_edges
from zone3._svloi.writer import write_ontology
from zone3._svloi.pipeline import run_sv_loi

__all__ = [
    "get_llm", "get_neo4j_graph", "_invoke_llm", "_sanitize_label", "_parse_json_safely",
    "type_value_entities", "propagate_to_records", "derive_interclass_edges",
    "write_ontology", "run_sv_loi",
]

if __name__ == "__main__":
    from zone3._svloi.pipeline import main
    main()
```

The 9 symbols imported by `recursive_induction.py` (`get_llm`, `get_neo4j_graph`, `_invoke_llm`, `_sanitize_label`, `_parse_json_safely`, `write_ontology`, `derive_interclass_edges`, `propagate_to_records`, `type_value_entities`) are all re-exported. `recursive_induction.py` is not modified.

## Migration Strategy

Risk-ordered, one commit per step, each independently validatable.

### Step 0 — Baseline capture

Before any code moves:

1. Run `python3 zone3/sv_loi.py` with current code on the standard dev dataset.
2. Save to `docs/superpowers/baselines/2026-04-21-svloi-baseline/`:
   - `classes.txt` — Cypher dump of class list + member counts
   - `f1.json` — final Riskine F1 score from the eval harness
   - `run.log` — full stderr log (catches accidental prompt/string edits)
3. Commit baseline artifacts so every subsequent migration step can be diffed against the same reference.

### Step 1 — Extract `constants.py` and `utils.py`

Lowest risk. No logic. Move constants block and utility functions. Update `sv_loi.py` to re-import them.

**Validation gate:** rerun pipeline; diff classes/F1/log against baseline. Expect zero-diff on classes (LLM non-determinism aside). Any diff here is a sign of accidental edit.

### Step 2 — Extract `sohd.py`

Self-contained block (lines ~2391–3292 in current file). No shared state with earlier phases. Moving this alone deletes ~900 lines from the main file.

**Validation gate:** rerun; diff.

### Step 3 — Extract `writer.py` and `records.py`

Output-side boundary. `writer.py` owns the final Neo4j write; `records.py` owns record decomposition.

**Validation gate:** rerun; diff.

### Step 4 — Extract `structural.py` and `hierarchy.py`

Internal compute modules. Structural first (leaf dependency), then hierarchy (depends on structural for consensus checks).

**Validation gate:** rerun; diff.

### Step 5 — Extract `typing.py`

Largest and most tangled block. Saved for last because it is easiest to debug when it is the only remaining unknown.

**Validation gate:** rerun; diff.

### Step 6 — Trim `sv_loi.py` to facade, move `run_sv_loi` to `pipeline.py`

Final cleanup. `sv_loi.py` is now ~30 lines. `run_sv_loi` and CLI live in `_svloi/pipeline.py`.

**Validation gate:** rerun; diff. Final sanity check: `python3 -c "from zone3.sv_loi import *; print(len(__all__))"` returns 10.

## Validation

End-to-end rerun after each step. Non-determinism in LLM outputs means F1 may drift ±0.01 between runs; anything larger than that threshold triggers investigation before proceeding. Class-list membership drift of more than ~5 entities per class is similarly a red flag.

If a step fails validation:
- `git revert <step-commit>` to roll back just that step.
- Investigate before re-attempting.

## Risks

| Risk | Mitigation |
|------|-----------|
| Circular imports between new modules | Bottom-up extraction order (utils → leaf phases → pipeline) surfaces cycles at the earliest step they appear. |
| LLM prompt strings accidentally edited during move | Each migration commit is reviewed for string diffs; run log diff in validation gate catches behavioral change even if the visual diff is missed. |
| Hidden module-level side effects | Before Step 1, grep for top-level statements beyond imports/constants/def; confirm nothing executes at import time. |
| Step 5 (typing) harder than expected, blocks deadline | Steps 1–4 are standalone wins. If Step 5 stalls past day 10, freeze the refactor: typing stays as-is in `sv_loi.py` alongside the new package; presentation still benefits from the other 3000 lines being split out. |
| Private-symbol (`_invoke_llm` etc.) imports from `recursive_induction.py` break if facade misses one | `__all__` in the facade is checked against a grep of `from zone3.sv_loi import` in the whole repo before Step 6 lands. |

## Success Criteria

1. `zone3/sv_loi.py` is < 50 lines (facade only).
2. Every file in `zone3/_svloi/` is < 1000 lines.
3. `recursive_induction.py` is unchanged, and `from zone3.sv_loi import ...` continues to resolve all 9 symbols it uses.
4. End-to-end pipeline rerun produces class list and F1 within tolerance of the Step 0 baseline.
5. All migration commits are reverted-cleanly independent (no commit depends on a later commit to be valid).
