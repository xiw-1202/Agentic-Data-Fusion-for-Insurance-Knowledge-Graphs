# Zone 3 SV-LOI — Version History

All versions evaluated against the Riskine reference ontology. Original used 10-class eval; all others use 26-class (full Riskine).

## Version Summary

| Metric | orig | seaf | v4 | v5 | v6 | v7 | v8 |
|--------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Eval scope | 10 | 26 | 26 | 26 | 26 | 26 | 26 |
| Induced classes | 7 | 20 | 21 | 20 | 19 | 6 | 9 |
| Name F1 | 0.562 | 0.328 | 0.405 | 0.415 | 0.380 | 0.250 | **0.441** |
| BERTScore F1 | **0.732** | 0.606 | 0.647 | 0.655 | 0.639 | 0.600 | 0.703 |
| Graph F1 | **0.714** | 0.593 | 0.621 | 0.629 | 0.622 | 0.307 | 0.453 |
| Continuous F1 | 0.122 | 0.157 | 0.157 | 0.167 | **0.222** | 0.076 | 0.183 |
| Wu-Palmer | 0.752 | 0.751 | 0.735 | 0.754 | **0.775** | 0.706 | 0.656 |
| EA F1 (full 26) | 0.326 | 0.174 | 0.128 | 0.404 | **0.414** | 0.298 | 0.374 |
| EA F1 (present) | 0.417 | 0.320 | 0.231 | 0.533 | 0.582 | **0.741** | 0.684 |
| Riskine covered | 3 | 4 | 3 | **10** | **10** | 5 | 7 |
| Query Accuracy | **95.0%** | 82.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% |

Note: orig was evaluated on 10 Riskine classes (flood-relevant subset). BERTScore/Graph F1 are not directly comparable with 26-class runs.

## Version Details

### orig — Original SV-LOI (10-class eval)
- First SV-LOI implementation with 7 induced classes
- 10-class Riskine eval (flood-relevant subset)
- High BERTScore/Graph F1 (easier eval target)

### seaf — Architectural Refinement (26-class eval)
- 6 architectural changes: concept-first verification, record propagation, 5-way consolidation, relation-range induction, protected classes, data-driven edges
- Switched to 26-class Riskine eval (full ontology)
- Metrics dropped due to harder eval, not worse algorithm

### v4 — Zone 2 v4 Extraction
- Same SV-LOI code on improved Zone 2 extraction (7,594 nodes vs 7,415)
- Better entity types (103 vs 60) gave richer signals for class discovery
- 10 exact Riskine name matches in discovered vocabulary

### v5 — Eval Fix: Enriched + Dual Representation
- **EA metric fix**: enriched class representation ("ClassName: member1, member2, ...") instead of raw member centroid averaging
- **Dual Riskine representation**: embed both property descriptions AND class names, take element-wise max
- **Schema mapping (partial)**: ClaimRecord→Claim worked, but PolicyRecord→PolicyRecord (tautological)
- **Result**: EA F1 tripled (0.128→0.404), 10 Riskine classes covered (was 3)

### v6 — Schema Mapping Complete
- **Fixed tautological mapping**: excluded record-type names from LLM target classes
- PolicyRecord→Coverage, ClaimRecord→Claim, Property→Structure
- Zero record-type classes in final ontology
- Best EA F1 full (0.414) and Wu-Palmer (0.775)

### v7 — LLM-Guided Class Validation (Over-merged)
- Replaced hardcoded `PROTECTED_CLASS_NAMES` (domain leakage) with LLM-guided class validation
- LLM decides keep/merge for each small class based on structural evidence
- **Problem**: only 3 large target classes (Structure, Claim, Coverage) → LLM merged everything into them
- 21→6 classes, lost Risk, Damage, Product

### v8 — Fixed Prompt (Better but Claim→Coverage)
- Presented ALL classes to LLM (not just small ones)
- Improved prompt: "KEEP if members are real-world THINGS, MERGE only if MEASUREMENTS/ATTRIBUTES"
- Risk, Damage, Product correctly kept
- **Problem**: Claim (485 records) merged into Coverage (both record-backed)
- Best Name F1 (0.441) and BERTScore (0.703)

### v9 — Structural Veto (Pending)
- Added structural veto: never merge two record-backed classes (distinct data schemas)
- Record-aware prompt: classes marked [RECORDS], explicit rule against merging them
- Expected: v8's correct keep decisions + Claim preserved
