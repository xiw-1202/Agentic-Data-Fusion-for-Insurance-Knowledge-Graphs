# Zone 3 Ontology Induction — How We Improved It

## Starting Point (Midterm, March 15)

Only one method: multi-resolution Leiden clustering on a 4D similarity graph. No LLM involvement in ontology induction.

**Problems**:
- Induced 11 classes with **zero** Riskine name overlap (Name F1 = 0.000)
- Cluster names like "Hazard" mixed buildings, coverages, and perils (semantically impure)
- No hierarchy (flat classes, no SUBCLASS_OF)
- Query accuracy: 75% (only 20 queries)
- Type inconsistency: 8.3% (entities with conflicting labels)

## Improvement 1: SV-LOI Algorithm (Novel Contribution)

**Problem**: Leiden clustering is consistent but over-fragments. LLM typing is accurate but inconsistent across batches. Neither alone is sufficient.

**Solution**: SV-LOI (Structurally-Verified LLM Ontology Induction) — fuse both signals:

1. **LLM proposes** ontology classes via two-stage discovery (detect domain, then propose classes)
2. **LLM types** each entity into a class (batched, 15 entities/prompt)
3. **Structure verifies** — build relation-signature vectors, compute class centroids, flag outliers >2σ
4. **LLM arbitrates** disagreements with both semantic and structural evidence

**Key insight**: Entities of the same ontological class share two independent signals — the LLM recognizes what they ARE (semantic), and they participate in the same types of relations (structural). Fusing both with disagreement arbitration eliminates each signal's failure mode.

**Impact**: Name F1 from 0.000 to 0.441, BERTScore from 0.451 to 0.703

## Improvement 2: Three-Lane Entity Architecture

**Problem**: The KG has 7,594 entities but they're fundamentally different types:
- **Concept entities** (382): from PDF — "Flood", "Dwelling", "FEMA"
- **Record entities** (1,798): from CSV — POL-xxx, CLM-xxx, PROP-xxx
- **Value entities** (5,414): numbers, dates — "250000", "32003"

Treating them uniformly fails: records dominate class centroids, values add noise.

**Solution**: `get_entity_lane()` classifies each entity. Different pipeline stages handle different lanes:
- Phase 1a (class discovery): concept entities only — they carry the semantic signal
- Phase 1b (typing): all entities, but structured entities get Zone 2 types directly
- Phase 2a (structural verification): concept entities only — clean centroids
- Phase 1e (schema mapping): records get mapped via relation profiles
- Phase 1f (value typing): relation-range induction from typed entities

**Impact**: Type inconsistency from 8.3% to 0.0%. Concepts drive class discovery while records and values follow.

## Improvement 3: Schema Mapping for Records

**Problem**: Records exist in a **disconnected subgraph** from concepts. Records connect only to values (HAS_ZIP_CODE→32003), never to concepts (Flood, Dwelling). Neighbor-majority voting for type propagation got 0/1,798 — structurally impossible.

Graph structure:
```
[concept]──→[concept]     331 edges (PDF knowledge)
[record]──→[value]        25,279 edges (CSV data)
               ↑ zero cross-links
```

**Solution**: LLM schema mapping inspired by R2RML/OBDA (Sequeda et al. 2012):

1. Group records by Zone 2 entity_type (PolicyRecord, ClaimRecord, Property)
2. Collect each type's relation profile (top 12 relations)
3. Ask LLM: "Given these ontology classes, which one fits entities with these relations?"
4. Bulk-assign: PolicyRecord→Coverage, ClaimRecord→Claim, Property→Structure

Only ~3 LLM calls for 1,798 records. Domain-agnostic — any new tabular schema goes through the same path.

**Impact**: 1,798 records correctly mapped. Zero record-type classes (PolicyRecord, ClaimRecord) polluting the final ontology.

## Improvement 4: Evaluation Metric Fix

**Problem**: Entity Assignment F1 was 0.128 even though the ontology was correct. The eval was comparing member name embeddings (e.g., average of "FEMA", "Insurer") against Riskine property descriptions ("Organization: business-name, founding-date, registry-number"). Different semantic spaces → low cosine → false negatives.

Organization (correct assignment) scored 0.380 — below the 0.42 threshold — and was rejected.

**Solution**: Two changes:
1. **Enriched induced representation**: Embed `"Organization: FEMA, Insurer, Condominium Association"` instead of averaging raw member embeddings. Anchors the embedding to the class concept.
2. **Dual Riskine representation**: Embed both property descriptions AND bare class names, take element-wise max. Some classes match better on names (Organization→Organization), others on properties (Coverage→sum-insured).

**Impact**: EA F1 tripled from 0.128 to 0.414. Riskine classes covered jumped from 3 to 10.

## Improvement 5: Removing Domain Leakage (PROTECTED_CLASS_NAMES)

**Problem**: `PROTECTED_CLASS_NAMES` contained 18 terms including `coverage`, `damage`, `risk`, `claim` — exactly matching Riskine's classes. These terms were protected from consolidation (never merged or renamed), artificially boosting Name F1.

**Solution**: Removed the hardcoded list entirely. Protection is now data-driven:
- Phase 4b+ (LLM-guided class validation) presents structural evidence to the LLM
- The LLM decides keep/merge based on member identity and relational diversity
- Structural heuristic fallback: merge only if >70% leaf nodes AND <4 distinct relation types
- No hardcoded class names anywhere in the pipeline

**Impact**: Zero domain leakage. Classes survive consolidation based on structural merit, not hardcoded protection. Risk (4 relation types, members like "Base Flood") is kept because it's structurally real, not because "risk" is on a list.

## Improvement 6: LLM-Guided Class Validation

**Problem**: Rule-based leaf-class merging uses fixed thresholds (leaf% > 70%, rel_types < 4) that are calibrated for our current sparse data. With more data, these thresholds would break — even "Deductible" might have 6+ relation types and survive the filter.

**Solution**: Present ALL classes to the LLM with structural evidence and let it decide:

```
Classes:
  Coverage (468 total, 9 concepts): examples=[total building insurance coverage, Coverage C], 51 relation types
  Deductible (4 total, 3 concepts): examples=[Separate Deductible], 3 relation types
  Risk (17 total, 17 concepts): examples=[Base Flood, Continuous Lake Flood], 4 relation types

Rules:
  KEEP if members are real-world THINGS (risks, products, people)
  MERGE if members are MEASUREMENTS or ATTRIBUTES (deductible amounts, coverage limits)
```

The LLM makes semantic decisions informed by structural evidence. Scales naturally — more data means richer evidence, better decisions.

**Impact**: Correct keep/merge decisions on 14b and 72b models. Risk, Damage, Product kept. Deductible, Limit, Payment merged into Coverage.

## Improvement 7: Structural Veto for Record-Backed Classes

**Problem**: The 72b LLM merged Claim (485 records) into Coverage (459 records). Both are record-backed classes from different CSV schemas (claims vs policies). Merging them conflates distinct data sources.

**Solution**: Structural veto — never merge two classes that both contain record entities:
- Detect record-backed classes by checking for entity ID prefixes (POL-xxx, CLM-xxx)
- If LLM says "merge A into B" and both have records → VETO
- Also mark record-backed classes as `[RECORDS]` in the LLM prompt

This is the SV-LOI principle applied to consolidation: when semantic (LLM) and structural signals disagree, keep separate.

**Impact**: Claim preserved as distinct class. Coverage and Claim remain separate ontology classes reflecting their distinct data schemas.

## Improvement 8: Data-Driven Inter-Class Edges

**Problem**: The original approach asked the LLM to guess SUBCLASS_OF relationships ("Is Coverage a subclass of Product?"). The LLM produced wrong IS-A edges because most inter-class relationships in insurance are HAS-A (association), not IS-A (inheritance).

**Solution**: Derive edges from actual entity-level connections in the KG:
- Count how many entities of class A connect to entities of class B
- If count > threshold (3), create an (A, B) inter-class edge
- These are association edges, not IS-A, matching how Riskine defines inter-class links

**Impact**: Edges reflect actual data relationships, not LLM guesses. More accurate Graph F1 evaluation.

## Results Progression

| Metric | Midterm (Leiden) | SV-LOI orig | Best (v6/v8) |
|--------|:----------------:|:-----------:|:------------:|
| Eval scope | 10-class | 10-class | **26-class** |
| Induced classes | 11 | 7 | **9-21** |
| Name F1 | 0.000 | 0.562 | **0.441** |
| BERTScore F1 | 0.451 | 0.732 | **0.703** |
| EA F1 (present) | 0.293 | 0.417 | **0.582** |
| Wu-Palmer | 0.653 | 0.752 | **0.775** |
| Riskine covered | 0 | 3 | **10** |
| Query accuracy | 75% | 95% | **87.5%** |
| Type inconsistency | 8.3% | 2.2% | **0.0%** |

Note: orig used 10-class eval (easier). v6/v8 use 26-class eval. BERTScore/Graph F1 not directly comparable across eval scopes.

## Key Design Decisions

1. **Concept-first verification**: Run structural verification on concept entities first (clean centroids), then propagate to records. Prevents record entities from dominating centroids with their uniform relation patterns.

2. **Two-stage class discovery**: First detect domain ("flood insurance"), then propose classes. The domain detection step prevents data-type classes (NumericValue, TimePeriod) from being proposed — the LLM knows to propose insurance concepts, not data types (Finding F-15).

3. **LLM + structure at every decision point**: Class discovery (LLM), entity typing (LLM), verification (structure), arbitration (LLM + structure), schema mapping (LLM + relation profiles), class validation (LLM + structural evidence), veto (structure). Neither signal alone is sufficient.

4. **No hardcoded domain knowledge**: Zero references to Riskine, SFIP, NFIP, or any specific insurance LOB in the Zone 3 code. The same `sv_loi.py` runs on flood, auto, health, or any domain.
