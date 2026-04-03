# SV-LOI Architecture Deep Analysis

Date: 2026-04-01
Status: Design analysis for paper refinement

---

## 1. Current Method Summary

SV-LOI (Structurally-Verified LLM Ontology Induction) is a 5-phase pipeline:

**Phase 0**: Load 1,351 entities from graph cache. Separate into ~208 concept entities (from PDF) and ~1,143 value/record entities (from CSV).

**Phase 1a** (Class Discovery): Two-stage LLM prompting on concept entities only. Stage 1 detects domain ("flood insurance"). Stage 2 proposes 12-25 classes grounded in that domain. Post-process: forbidden name filter, STANDARD_RENAMES mapping, protected class injection.

**Phase 1b** (Entity Typing): Three-lane assignment:
- Structured entities (POL-/CLM-/REC-/PER-/PROP-): pre-assigned from `entity_type`
- Value entities (Numeric, Date, Text): pre-assigned to "Other" (with location-relation heuristic for Address)
- Concept entities (~208): LLM batch classification (BATCH_SIZE=15, ~14 calls)

**Phase 1c** (Rebalance): Split any class exceeding 25% of total entities.

**Phase 1d** (Rescue): Re-type concept-only "Other" entities with class examples as context.

**Phase 2** (Structural Verification): Build relation-signature feature vectors. Compute class centroids. Flag entities deviating >2 sigma from their class centroid. Two rounds.

**Phase 3** (Arbitration): Re-query LLM for flagged entities with enriched context showing the conflict between semantic and structural signals.

**Phase 4** (Consolidation + Hierarchy): 5-way pairwise class relation inference (equivalent/parent/child/overlap/distinct). Merge small classes (<3 members). Dedicate hierarchy derivation pass.

**Phase 5**: Write to Neo4j.

Current results: 7 final classes, Name F1=0.563, BERTScore F1=0.732, Graph F1=0.714.

---

## 2. Question-by-Question Analysis

### Q1: Phase 0 — Why Only Concepts Drive Ontology?

**Current behavior**: `get_concept_entities()` filters to ~208 entities for class discovery. The 1,143 value/record entities are excluded from Phase 1a entirely and pre-assigned in Phase 1b.

**Is concept-only fundamentally flawed?** No, but it is incomplete.

The concept-only approach is *correct in principle*: ontology classes should be derived from domain concepts, not from data values. A dollar amount IS NOT an ontology class — it's an attribute. A zip code IS NOT an ontology class — it's an instance of Address.

However, the current implementation has a blind spot: **records carry structural information that could inform class discovery without directly participating in it**.

**What records reveal:**
- The existence of ClaimRecord and PolicyRecord entities tells us the domain has "Claims" and "Policies" (which should map to Product and potentially a Claim class)
- The ADDRESS-related fields (cities, states, zips) connected via HAS_PROPERTY_STATE, HAS_REPORTED_CITY confirm Address as a class
- Payment amounts connected via HAS_AMOUNT confirm a financial/monetary concept
- Building values connected via HAS_PROPERTY_VALUE confirm Property/Structure concepts

**What records should NOT do:**
- Records should NOT vote in class discovery prompts (they'd overwhelm concept entities 5:1)
- Records should NOT participate in naming — "POL-12345" is not a useful class exemplar
- Records should NOT be counted in class distribution for rebalancing (see Q2)

**Recommendation: Records as Class Evidence, Not Class Voters**

Add a "record signature analysis" step between Phase 0 and Phase 1a:

```python
def analyze_record_signatures(entities: list[dict]) -> dict[str, list[str]]:
    """Extract domain signals from record entities.

    Returns evidence like:
        {"suggests_address": ["HAS_PROPERTY_STATE seen 400x", "HAS_REPORTED_CITY seen 350x"],
         "suggests_claim": ["CLM- prefix seen 600x", "HAS_DATE_OF_LOSS seen 590x"],
         ...}
    """
```

Then feed these signals into the Phase 1a class discovery prompt as structural evidence:

> "The graph also contains 600 claim records with DATE_OF_LOSS and AMOUNT relationships, and 400 property records with STATE, CITY, and ZIP_CODE relationships. Consider whether these patterns suggest additional ontology classes."

This preserves the concept-only voting principle while using record patterns as auxiliary evidence.

**Code-level change**: Modify `discover_class_vocabulary()` to accept a `record_evidence: dict` parameter. Compute evidence from record relation signatures before class discovery. Include 3-5 lines of evidence in the Stage 2 prompt. Do NOT include record entity names.

**Impact**: Likely adds 1-2 classes that are currently missing (Claim, Payment) and strengthens evidence for Address and Property.

---

### Q2: Phase 1c — Why Rebalance at 25%?

**Current behavior**: `MAX_CLASS_FRACTION = 0.25`. Any class exceeding 25% of **total entities** (1,351 * 0.25 = 337) gets split by LLM.

**The fundamental problem**: The 25% threshold is computed over ALL entities, but class sizes are dominated by records, not concepts. Consider the entity distribution:

- ~600 CLM- records → likely all map to one class (Claim/Product)
- ~400 POL- records → likely all map to one class (Product)
- ~150 value entities (dates, amounts) → "Other"
- ~208 concept entities → spread across 7-12 classes

This means ~1,000 out of 1,351 entities SHOULD concentrate in 2-3 classes. The 25% threshold is wrong because it treats records and concepts as equally important for class balance.

**Is rebalancing even needed?** Not in the current form. The rebalancing step addresses a problem that doesn't actually exist in the concept entity space.

**What the rebalance step actually does wrong:**
1. It measures class size over ALL entities, including records that naturally cluster
2. It splits classes that are correctly large (e.g., Product containing all POL- records)
3. The LLM splits then produce artificial sub-classes with no ontological justification

**Recommendation: Two options**

**Option A (preferred): Rebalance on concept entities only.**

```python
def rebalance_mega_classes(...):
    # Count only concept entities per class
    concept_counts = Counter(
        cls for eid, cls in assignments.items()
        if is_concept_entity(entity_map.get(eid, ...))
    )
    threshold = int(len(concept_entities) * max_fraction)  # 208 * 0.25 = 52
```

This means a class can have 600 records + 30 concepts and NOT trigger rebalancing, because only 30/208 concepts are in it.

**Option B: Remove rebalancing entirely.** The current results show it either doesn't fire (no mega-classes in concept space) or produces bad splits when it does. The 25% threshold has no theoretical justification. In insurance ontology, Coverage IS legitimately the largest class because most policy language is about coverage.

**My recommendation**: Implement Option A and raise the threshold to 40% for concept entities. If a single class has 40%+ of the ~208 concept entities (>83), that is suspicious and worth splitting. But at 25% (>52), a Coverage class with 55 concept members is perfectly natural.

**Code-level change**: In `rebalance_mega_classes()`, change `total = len(assignments)` to count only concept entities. Change `MAX_CLASS_FRACTION = 0.40`. Add a guard: if the class has <50 concept members absolute, skip regardless of percentage.

---

### Q3: Phase 1d — Rescue Improvements

**Current behavior**: Rescue only operates on concept entities in "Other" (correct). But typically only 0-5 concept entities are unclassified, making rescue a near-no-op.

**The real rescue gap**: Value entities that SHOULD belong to a class are permanently stuck in "Other":

| Entity type | Count (approx) | Correct class | Current assignment |
|-------------|:-:|:-:|:-:|
| City names (FL, TX, ...) | ~50 | Address | Other (unless location-relation heuristic fires) |
| State codes | ~30 | Address | Other |
| Zip codes | ~40 | Address | Other |
| Dollar amounts ($250K, ...) | ~100 | Other (correct) | Other |
| Dates (2020-01-15, ...) | ~80 | Other (correct) | Other |
| Occupancy types | ~20 | Structure (arguably) | Other |

**The location-relation heuristic** (lines 466-478) partially addresses this: value entities connected via HAS_PROPERTY_STATE, HAS_REPORTED_CITY, etc. get assigned to Address. But this heuristic is brittle:
- It depends on exact relation name strings (LOCATION_RELATIONS set)
- It only handles Address, not other value-to-class mappings
- It doesn't cover all location relations that might exist in other LOBs

**Recommendation: Generalize the relation-based value typing**

Replace the hardcoded LOCATION_RELATIONS heuristic with a **relation-signature classifier for value entities**:

```python
def type_value_entities(
    value_entities: list[dict],
    class_vocab: list[str],
    concept_assignments: dict[str, str],
) -> dict[str, str]:
    """Assign value entities to classes based on what they connect to.

    If a value entity is the target of a relation from a concept entity
    of class X, the value entity likely belongs to class X or a related class.

    Example: "FL" <--HAS_PROPERTY_STATE-- (entity in Structure class)
             => "FL" is likely an Address or part of a Structure.
    """
```

The algorithm:
1. For each value entity, collect the classes of its connected concept entities
2. If >70% of connections point to a single class, assign the value entity to that class
3. If mixed, assign to "Other" (correct — it's ambiguous)
4. Special case: if ALL incoming relations are geographic (state/city/zip patterns), assign to Address regardless of connected class

**What NOT to change**: Dollar amounts, dates, and bare numbers should STAY in "Other". They are attribute values, not domain concepts. The pipeline is correct to exclude them.

**Impact on evaluation**: This could improve Entity Assignment F1 by routing ~120 location-related value entities to Address instead of "Other". Since Address IS a Riskine class, this improves both precision and recall.

**Code-level change**: Replace lines 466-479 (LOCATION_RELATIONS heuristic) with the generalized `type_value_entities()` function. Call it after Phase 1b, before rebalancing.

---

### Q4: Overall Architecture Assessment

**Is SV-LOI's architecture fundamentally sound?** Yes. The core insight is correct and novel: fusing LLM semantic typing (Phase 1) with structural verification (Phase 2-3) produces better ontology classes than either signal alone. The ablation results confirm this — removing verification drops quality.

**What works well:**
1. Two-stage class discovery (F-15: eliminates data-type classes)
2. LLM-guided consolidation (F-16: doubles Name F1)
3. Structural verification as a safety net (catches ~10-20% mistyped entities)
4. Protected class names (prevents destroying good name matches)
5. Decision provenance tracking (good for paper's error analysis)
6. Ablation flags (enables controlled experiments)

**The ONE biggest architectural weakness: The concept/record/value separation is too rigid and too early.**

The three-lane split (line 451-460) happens BEFORE any class information exists. This creates a chicken-and-egg problem:
- Records are pre-assigned based on `entity_type` (from Zone 2), which may not match the ontology
- Value entities are blanket-assigned to "Other", losing location/occupancy/etc. information
- Only concepts go through the LLM, but concepts are only 15% of the graph

**The consequence**: The ontology is built from 15% of the data. 85% of entities get assigned to classes they never helped discover and never validated. The structural verification (Phase 2) uses ALL entities' relation signatures, but the class vocabulary was discovered from only 15% of them. This mismatch means structural centroids are distorted by the majority of entities (records) that were force-assigned.

**The fix: Two-phase typing with structural feedback**

Instead of the current flow:

```
Discover classes (concepts only) → Assign ALL entities → Verify → Arbitrate
```

Use:

```
Discover classes (concepts + record evidence) → Assign concepts → Verify concepts →
→ Propagate to records via relation patterns → Verify ALL → Final arbitration
```

Concretely:

1. **Phase 1a** (unchanged but enriched): Class discovery from concept entities + record structural evidence
2. **Phase 1b** (concept-only typing): LLM types only the ~208 concept entities
3. **Phase 2-first** (concept verification): Structural verification on concept entities only
4. **Phase 1b'** (record propagation): For each record, find its connected concept entities. Majority-vote their classes to assign the record. This is the key change — records get typed by their concept neighbors, not by their Zone 2 entity_type.
5. **Phase 2-full** (full verification): Now verify ALL entities with proper centroids
6. **Phase 3** (arbitration): As before
7. **Phase 4** (consolidation + hierarchy): As before, but with concept-only voting for merge decisions

This fixes the centroid distortion problem: record entities get assigned AFTER concept entities are verified, so they reinforce correct class boundaries rather than polluting them.

---

## 3. Proposed Refined Architecture

```
Phase 0:  Load entities → separate concept/record/value
Phase 0.5: Analyze record relation signatures → extract domain evidence
Phase 1a: Discover classes (concepts + record evidence signals)
Phase 1b: LLM batch-type CONCEPT entities only (~208, ~14 LLM calls)
Phase 1c: Structural verification on concepts only (small feature matrix)
Phase 1d: Rescue unclassified concept entities
Phase 1e: Propagate concept types to records via neighbor-majority voting
Phase 1f: Type value entities via relation-signature classifier
Phase 1g: Rebalance (concept-count only, threshold=40%)
Phase 2:  Full structural verification (all 1,351 entities, proper centroids)
Phase 3:  Arbitration on flagged entities
Phase 4:  5-way class relation inference (concept-only voting)
Phase 4a: Merge small classes
Phase 4b: Hierarchy derivation
Phase 5:  Write to Neo4j
```

Key differences from current:
- Record evidence feeds into class discovery (Phase 0.5)
- Verification happens BEFORE record propagation (Phase 1c)
- Records get typed by concept neighbors, not by Zone 2 entity_type (Phase 1e)
- Value entities get relation-based typing, not hardcoded heuristic (Phase 1f)
- Rebalancing uses concept counts only (Phase 1g)
- Full verification uses correct centroids (Phase 2)

---

## 4. What to Keep vs What to Change

### KEEP (working well)
- Two-stage class discovery (domain detection then class proposal)
- FORBIDDEN_CLASS_NAMES filter
- STANDARD_RENAMES mapping
- PROTECTED_CLASS_NAMES mechanism
- 5-way class relation inference (equivalent/parent/child/overlap/distinct)
- Structural signature vectors (relation-type OUT/IN counts + entity-type one-hot)
- Decision provenance tracking
- Ablation flag infrastructure
- Batch LLM prompting with `entity -> ClassName` output format
- `_parse_json_safely()` with fallback extractors

### CHANGE
| Component | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| Record role in discovery | Excluded entirely | Provide relation-signature evidence | Records confirm domain concepts |
| Entity typing order | All at once, 3-lane split | Concepts first, verify, then propagate to records | Prevents centroid pollution |
| Value entity typing | Hardcoded LOCATION_RELATIONS | Relation-signature classifier | Generalizes to any LOB |
| Rebalance threshold | 25% of total entities | 40% of concept entities only | Records naturally cluster |
| Rebalance denominator | Total entities (1,351) | Concept entities (~208) | Correct unit of analysis |
| Rescue scope | Concept-only (correct but small) | Keep concept-only, add value typing as separate step | Separates concerns |
| Structural verification timing | After all typing | After concept typing, before record propagation | Clean centroids |

### REMOVE (if simplifying)
- The current `LOCATION_RELATIONS` hardcoded set (replaced by generalized classifier)
- The concept-count guard in rescue (it's a near-no-op; keep or drop)

---

## 5. Priority-Ordered Action Items

### P0: Critical for paper quality
1. **Implement record evidence for class discovery** (Phase 0.5). Low effort, high impact. Analyze record relation signatures, feed 5-line summary into Stage 2 prompt. Does NOT change the algorithm structure, only enriches the discovery prompt.

2. **Fix rebalance to use concept-count denominator**. Change 3 lines in `rebalance_mega_classes()`. Prevents incorrect splits of naturally large classes. Raise threshold from 0.25 to 0.40.

### P1: Important for completeness
3. **Generalize value entity typing**. Replace LOCATION_RELATIONS heuristic with neighbor-class majority voting. Affects ~120 entities (location values). Improves Address recall.

4. **Reorder verification before record propagation**. Structural change to `run_sv_loi()`. Concepts get verified first, then records get typed by verified concept neighbors. Prevents centroid pollution. Requires running Phase 2 twice (concept-only, then full).

### P2: Nice to have for ablation paper
5. **Add record-propagation ablation flag** (`--skip-record-propagation`). Compare: (a) current Zone 2 entity_type assignment, (b) neighbor-majority propagation. Shows whether propagation helps.

6. **Measure concept-count vs total-count rebalancing**. Run ablation with old threshold (0.25/total) vs new (0.40/concept). Show rebalancing was previously inactive or harmful.

### P3: Future work
7. **Cross-domain validation**. Run refined SV-LOI on auto insurance data to confirm the generalized value typing and record evidence work without modification.

8. **Hierarchical class discovery**. Instead of flat discovery + post-hoc hierarchy, discover classes in a taxonomy-aware way (parent classes first, then subclasses). Out of scope for current paper.
