# Zone 3 SV-LOI Session Changes (2026-04-01 to 2026-04-02)

## Summary
Major architectural refinement of SV-LOI ontology induction. Started from auto-review-loop (4 rounds, score 4.0→7.0), then deep analysis and implementation of 6 architectural changes.

## Auto-Review Loop (4 rounds)
- Round 1: 4.0/10 → Added AUC-ROC, 26-class Riskine eval, ablation variants
- Round 2: 5.5/10 → Multi-view consensus formulation
- Round 3: 6.0-6.5/10 → Venue targeting (ISWC/AAAI applied)
- Round 4: 7.0/10 → Code mature, experiments pending

## Architectural Changes Implemented

### Change 1: `get_entity_lane()` helper (graph_cache.py)
- Centralized concept/record/value classification
- Replaces scattered `is_concept_entity()` + prefix checks

### Change 2: Rebalance denominator fix
- Was: 25% of total entities (7,415) — records dominate
- Now: 40% of concept entities only (~208) with min 50 absolute guard
- Ablation: `--use-old-rebalance`

### Change 3: Record evidence for class discovery (Phase 0.5)
- `analyze_record_evidence()` summarizes record relation patterns
- Feeds into Stage 2 class discovery prompt as structural evidence
- Domain-agnostic — pattern-based, no hardcoded field names

### Change 4: Generalized value entity typing (Phase 1f)
- Relation-range induction: learns `relation_type → class` from typed entities
- Builds (76 × 22) score matrix, computes P(class | relation)
- Replaces hardcoded LOCATION_RELATIONS heuristic
- Runs AFTER record propagation (Phase 1e) so values see record neighbors

### Change 5: Pipeline restructure — concept-first verification
- Phase 2a: Verify concepts ONLY with clean centroids
- Phase 1e: `propagate_to_records()` — neighbor-majority voting (>50%)
- Phase 2b: Full verification on all entities
- Prevents centroid pollution from unverified record pre-assignments
- Ablation: `--skip-record-propagation`

### Change 6: 5-way typed relation inference (consolidated Phase 4)
- Replaces merge-or-nothing consolidation + separate hierarchy
- 5 types: equivalent / parent / child / overlapping / distinct
- Protected class names (PROTECTED_CLASS_NAMES) cannot be renamed
- Two-lane: concept entities drive consolidation, records excluded

### Additional fixes:
- Data-driven inter-class edges replace LLM hierarchy guessing
- Protected classes exempt from `merge_small_classes()`
- Markdown code fence stripping for LLM JSON parsing
- Standard-term renaming: Policy→Product, Hazard→Risk, etc.
- Removed all NFIP/flood-specific references — fully domain-agnostic
- Audit confirmed zero reference ontology leakage

## Evaluation Additions
- 26-class Riskine evaluation (was 10)
- AUC-ROC with LLM judge as independent GT
- Per-class confusion analysis
- Intrinsic ontology quality metrics (coverage, coherence, balance, hierarchy depth)
- 10 new structured data queries (31-40)
- `compare_results.py --zone3` comparison tool
- `scripts/slurm_eval_only.sh` for eval-only cluster runs

## Results Progression
| Metric | Run 1 | Run 5 (best) | Latest |
|--------|:--:|:--:|:--:|
| Name F1 | 0.306 | 0.363 | 0.328 |
| BERTScore F1 | 0.592 | 0.629 | 0.606 |
| Graph F1 | 0.746 | 0.648 | 0.593 |
| EA F1 (present) | 0.264 | 0.341 | 0.320 |
| Coverage Recall | ? | 8/9 | 7/9 |
| Class coverage | ? | 85.9% | 26.9% |
| Query accuracy | 82.5% | 82.5% | 82.5% |

## Known Issues (for next session)
1. Record propagation gets 0/1798 — records don't connect to concept entities
2. Value typing learns few mappings — needs Zone 2 concept linking (Priority 3)
3. Product/Address still fragile with small member counts
4. Graph F1 dropped with data-driven edges — Riskine uses $ref associations, not IS-A
5. Zone 2 v4 now available — should improve all metrics with better extraction

## Files Modified
- `zone3/sv_loi.py` — major rewrite (~300 lines added/modified)
- `zone3/graph_cache.py` — `get_entity_lane()` helper
- `evaluation/ontology_metrics.py` — AUC, intrinsic metrics, per-class confusion
- `evaluation/riskine_eval.py` — 26-class support, AUC display
- `evaluation/riskine_loader.py` — ALL_SCHEMAS (26 classes)
- `evaluation/compare_results.py` — Zone 3 comparison tool
- `baseline/eval.py` — `--all-classes`, queries 31-40
- `scripts/slurm_sv_loi.sh`, `slurm_eval_only.sh`, `slurm_ablations.sh`, `run_ablations.sh`
