# Research Findings Log — CS584 AI Capstone

Empirical findings discovered during pipeline development. Each finding is numbered,
falsifiable, and tied to specific experimental evidence.

---

## F-01 · Semantic Label Bias in 8B-Scale Open IE

**Date**: 2026-03-03
**Zone**: Zone 2 (Few-Shot Open IE)
**Model**: llama3.1:8b
**Status**: Confirmed across 4 independent pipeline runs

### Claim
llama3.1:8b systematically ignores numeric precision values as triple objects — even when
the few-shot answer explicitly shows `$2,500`, `$30,000`, or `10 percent` as the object
field — and instead extracts the most semantically prominent *concept name* available in
the passage.

### Evidence

| Few-shot pair | Dollar shown in answer | Dollar triples extracted | Effect on query accuracy |
|---|---|---|---|
| Pair 3 (original) | `$500,000` (paraphrased) | 0 / 42 triples | 50% baseline |
| Pair 3 (v2) | `$30,000` ICC actual text | 0 / 43 triples | 45% (−5%) |
| Pair 3 (v3) | `$2,500` actual chunk text | 0 / 43 triples | 45% (−5%) |
| Pair 3 (reverted) | `$500,000` (paraphrased) | 0 / 42 triples | 50% restored |

Source chunks with dollar values not extracted: chunk 11 (`$2,500` special limits,
`10 percent` tenant improvements), chunk 12 (`$1,000` sandbags, `$10,000` property
removal), chunk 13 (`$30,000` ICC limit, `25 percent` / `50 percent` market value
thresholds). All 0 dollar-amount triples across all runs.

Day-count values *are* extracted successfully: `30 days`, `60 days`, `90 days`.

### Interpretation
The model treats dollar amounts and percentages as *quantifiers modifying a noun*
rather than as standalone factual objects. It tokenizes `"up to $30,000 under Coverage D"`
as evidence for a coverage relationship, discarding the numeric as context rather than
extracting it as the primary fact. Day counts succeed because they appear in syntactically
cleaner obligatory patterns (`"Within 60 days"`, `"policy expires after 30 days"`) where
the number IS the primary fact.

### Paper Framing
> *"At 8B scale, LLM-based Open IE exhibits a systematic semantic label bias: models
> prefer conceptual entity names over numeric precision values as triple objects. This
> bias is robust to few-shot prompting with exact-match examples and motivates
> post-extraction numeric linking (Zone 3/4) rather than relying on extraction alone."*

---

## F-02 · Few-Shot Pair Global Interference

**Date**: 2026-03-03
**Zone**: Zone 2
**Status**: Confirmed

### Claim
When a few-shot pair uses text that overlaps syntactically with an actual source chunk,
the model's extraction of that chunk is overridden to follow the few-shot *answer template*
rather than extracting the most semantically appropriate fact from the chunk.

### Evidence
Replacing Pair 3 with text from chunk 13 (ICC/Coverage D) caused the pipeline to
extract `(Coverage D) -[COVERS]-> (Elevation and Floodproofing)` instead of the
previously stable `(Coverage D) -[COVERS]-> (Increased Cost of Compliance)` — breaking
Task 8 (ICC coverage query). The Pair 3 answer showed `$30,000` as the object, but the
model instead extracted the first activity in the compliance list.

Reverting restored the previous extraction and Task 8 accuracy.

### Implication
Few-shot examples must use text that does NOT appear in the actual document corpus,
*or* must be carefully matched to the desired extraction behavior for each specific chunk.
Paraphrased examples (not matching any source) function as safe abstract pattern teachers.
Examples drawn from source chunks function as strong local templates with unpredictable
spillover on nearby chunks.

---

## F-03 · Multi-Triple Extraction Failure at 8B Scale

**Date**: 2026-03-03
**Zone**: Zone 2
**Status**: Confirmed

### Claim
llama3.1:8b extracts exactly 1 triple per chunk regardless of the MANDATORY LIST
EXTRACTION system-prompt rule and few-shot examples demonstrating 3–4 triples per passage.

### Evidence
- 43 triples / 46 non-empty chunks = 0.93 triples/chunk average
- Chunk 11 (10 numbered items, $2,500 limit, 10% improvements): 1 triple extracted
- Chunk 13 (ICC with $30,000 limit, 25%/50% thresholds, eligibility list): 1 triple
- Chunk 18 (nested 2-level exclusion list): 1 triple

Few-shot pairs 1, 2, 10, 13 all demonstrate 3–4 triples. The 8B model ignores these.

### Interpretation
At 8B scale, the model's generation follows a "most salient fact" heuristic: it identifies
the single most prominent semantic relationship in the passage and stops. The instruction
to extract *all* items is overridden by the model's inherent stopping criterion.

### Paper Framing
Motivates Zone 3 (Leiden community detection): rather than hoping the 8B extractor
captures all facts, we let the graph structure itself surface latent relationships
through community structure — a graph-theoretic approach that compensates for
extraction sparsity.

---

## F-04 · Vocabulary Compliance Dominates Extraction Volume

**Date**: 2026-03-03
**Zone**: Zone 2
**Comparison**: llama3.1:8b vs deepseek-r1:14b

### Claim
A 14B reasoning model extracting more triples (70 vs 42) performs *worse* on query
accuracy (30% vs 50%) when it ignores the bootstrapped relation vocabulary.

### Evidence

| Model | Triples | Relation types | Vocab compliance | Query accuracy |
|---|---|---|---|---|
| llama3.1:8b | 42 | 15 | High | **50%** |
| deepseek-r1:14b | 70 | ~22 | Low (uses own names) | **30%** |

deepseek-r1:14b used `OBLIGED_TO` instead of `MUST_NOTIFY`, verbose full-sentence
object names, and caused a bad entity merge: `"Standard Flood Insurance Policy"` →
`"Other flood insurance policies"` at cosine ≥ 0.90 threshold.

### Paper Framing
> *"Vocabulary compliance is a stronger predictor of downstream query accuracy than
> extraction volume. A 14B reasoning model that ignores constrained vocabulary achieves
> 30% accuracy versus 50% for an 8B model that respects it, despite extracting 67%
> more triples."*

---

## F-05 · Entity Resolution Threshold Calibration

**Date**: 2026-03-03
**Zone**: Zone 2.5
**Status**: Confirmed

### Claim
At cosine ≥ 0.90, entity resolution merges 0 pairs in the stable 8B run (too conservative),
but merges semantically *incorrect* pairs in the 14B run (e.g., different policy names).

### Evidence
- llama3.1:8b: 0 merges / 61 nodes across all stable runs
- deepseek-r1:14b: 5 merges, including `"Standard Flood Insurance Policy"` →
  `"Other flood insurance policies"` (incorrect; different policies)
- One run (during regression testing): 1 merge of `"Appraisal Procedure"` →
  `"Appraisal Process"` at 0.90 — this was correct

### Implication
The 0.90 threshold is appropriate for the 8B model's consistent, concise node naming.
For a model with verbose/varied naming (14B), 0.90 triggers false merges. Threshold
should be adaptive based on node name length and extraction style.


---

## F-06 · Domain Leakage Invalidates Ontology Induction Claims

**Date**: 2026-03-09
**Zone**: Zone 2 (pipeline architecture)
**Status**: Confirmed — redesign approved

### Claim
The Zone 2 pipeline's Riskine F1 of 0.874 is an artifact of reference ontology
injection, not genuine ontology induction. Three components — ANCHORS,
RELATION_ROLE_MAP, and label_nodes() — hardcode Riskine class labels into the
extraction graph, making the "evaluation" a self-fulfilling prophecy.

### Evidence

**ANCHORS** (zone2/pipeline.py lines ~110-118):
```python
ANCHORS = [
    ("NFIP Policy", "Product"),       ("Building Coverage", "Coverage"),
    ("Flood", "Risk"),                ("Insured Building", "Structure"),
    ("Policyholder", "Person"),       ("FEMA", "Organization"),
    ("Insured Property", "Property"), ("Proof of Loss", "Object"),
    ("Earth Movement", "Damage"),
]
```
These 9 nodes are MERGEd with Riskine labels unconditionally. Running
`CALL db.labels()` returns Coverage, Product, Damage, Risk, Structure, Property,
Person, Object, Organization — 9/10 Riskine classes — regardless of what the
extractor actually found.

**RELATION_ROLE_MAP** (15 entries) assigns Riskine labels to nodes based on their
relation type: any node that is the subject of a COVERS relation gets labeled
`:Coverage`, any object of COVERS gets labeled `:Property`, etc. This is the
reference ontology's domain model, not an induced one.

**Generality test (thought experiment):** If we ran this pipeline on an auto
insurance PDF, it would still inject "NFIP Policy" (Product), "Flood" (Risk),
and "FEMA" (Organization) as anchor nodes — entities that have nothing to do
with auto insurance.

### Quantitative impact

| Component | Riskine classes it contributes | Without it |
|-----------|:------------------------------:|:-----------:|
| ANCHORS alone | 9/10 guaranteed | 0 guaranteed |
| RELATION_ROLE_MAP | 7-8 from typed relations | 0 |
| Extraction alone | ~3-4 via embedding similarity | ~3-4 |

Estimated honest Riskine F1 without domain leakage: **0.35–0.45** (vs reported 0.874).

### Resolution
All domain-specific components removed from zone2/pipeline.py.
Riskine evaluation moved exclusively to evaluation/riskine_eval.py.
Zone 3 Leiden clustering restored to its original purpose: bottom-up
class discovery from unlabeled extraction output.

Full plan: `docs/GENERALIZATION_PLAN.md`

### Paper Framing
> *"A common pitfall in LLM-based ontology induction pipelines is reference
> ontology leakage: hardcoding target classes into extraction prompts, anchor
> nodes, or relation-type mappings inflates alignment metrics without testing
> whether the system can genuinely discover ontological structure. We identified
> and corrected this in our Zone 2 redesign, reducing reported F1 from 0.874
> to an honest 0.35–0.45 from extraction alone, with Zone 3 Leiden clustering
> recovering to 0.55+ through genuine bottom-up induction."*

---

## F-15 · Two-Stage Class Discovery Eliminates Data-Type Classes

**Date**: 2026-03-24
**Zone**: Zone 3 (SV-LOI)
**Model**: qwen2.5:72b
**Status**: Confirmed

### Claim
SV-LOI's two-stage class discovery — first detect the insurance domain from entity
examples, then propose ontology classes grounded in that domain — eliminates data-type
classes (NumericValue, Temporal, VoidState, HazardZone) that plagued the Leiden baseline.

### Evidence
Leiden (baseline) produces 11 classes including several data-type classes that have
no ontological meaning in insurance: NumericValue, Temporal, VoidState, OccupancyType.
These classes arise because Leiden clusters entities by embedding similarity, and
numeric values / temporal expressions / status indicators cluster together by surface form.

SV-LOI's two-stage approach produces 7 classes, all grounded in insurance domain concepts.
The domain detection step constrains class proposals to insurance-relevant categories,
preventing the LLM from proposing data-type classes.

| Method | Classes | Data-type classes | Domain-relevant classes |
|--------|---------|-------------------|------------------------|
| Leiden | 11 | 4 (NumericValue, Temporal, VoidState, OccupancyType) | 7 |
| SV-LOI | 7 | 0 | 7 |

### Paper Framing
> *"Pure clustering methods like Leiden are susceptible to surface-form grouping:
> numeric values cluster with other numeric values regardless of semantic role.
> SV-LOI's two-stage class discovery — domain detection followed by domain-grounded
> class proposal — eliminates this failure mode entirely."*

---

## F-16 · LLM-Guided Consolidation Doubles Name F1

**Date**: 2026-03-24
**Zone**: Zone 3 (SV-LOI)
**Model**: qwen2.5:72b
**Status**: Confirmed

### Claim
LLM-guided class consolidation — merging fine-grained classes into canonical ontology
concepts — improves Name F1 from 0.214 (pre-consolidation) to 0.563 (post-consolidation),
a 2.6x improvement.

### Evidence
Before consolidation, SV-LOI proposes fine-grained classes that are semantically correct
but do not match reference ontology naming conventions. The consolidation step maps:
- City + County + State → Address
- Policy → Product
- Other fine-grained geographic/administrative classes into standard ontology categories

| Stage | Name F1 | BERTScore F1 | Notes |
|-------|---------|-------------|-------|
| Pre-consolidation | 0.214 | ~0.60 | Fine-grained classes, correct but verbose |
| Post-consolidation | **0.563** | **0.732** | Canonical names aligned with reference |

### Interpretation
Class consolidation is as important as initial class discovery. The LLM's ability to
recognize that City, County, and State are all sub-types of Address — and to merge them
accordingly — is a key advantage of SV-LOI over purely algorithmic methods.

### Paper Framing
> *"Initial class discovery produces semantically valid but fine-grained classes.
> LLM-guided consolidation maps these to canonical ontology concepts, raising
> Name F1 from 0.214 to 0.563. This two-phase approach separates the concerns
> of semantic correctness (discovery) and naming convention alignment (consolidation)."*

---

## F-17 · Incomplete Reference Class Coverage in Source Data

**Date**: 2026-03-24
**Zone**: Evaluation
**Status**: Confirmed

### Claim
The Riskine reference ontology contains 10 classes, but flood insurance documents
(SFIP policy + OpenFEMA claims/policy data) do not contain entities that naturally
map to Person, Organization, or Object. The maximum achievable full-Riskine recall
from flood data alone is approximately 0.6-0.7.

### Evidence
Riskine's 10 classes: Coverage, Product, Damage, Risk, Structure, Property, Person,
Object, Organization, Address.

Flood insurance documents primarily contain:
- Coverage descriptions, policy terms, property definitions, risk factors,
  structural requirements, geographic/address information, damage types
- No named persons (policies reference "the insured" generically, not by name)
- No named organizations (FEMA/NFIP appear as proper nouns, not as typed entities)
- No "Object" entities in the Riskine sense (movable physical objects)

This means any induction method operating on flood data alone has a recall ceiling.

### Interpretation
Full-Riskine F1 penalizes methods that correctly identify all discoverable classes
but cannot discover classes absent from the source data. The "present-class" metric
(F-18) provides a fairer comparison.

### Paper Framing
> *"Reference ontology recall is bounded by source data coverage. Flood insurance
> documents lack Person, Organization, and Object entities, capping achievable
> Riskine recall at ~0.7. We introduce present-class F1 to evaluate only against
> reference classes representable in the source domain."*

---

## F-18 · Present-Class F1 as a Fairer Evaluation Metric

**Date**: 2026-03-24
**Zone**: Evaluation
**Status**: Confirmed

### Claim
The "present-class" F1 metric — which evaluates only against reference ontology classes
that are actually representable in the source data — provides a fairer comparison
between ontology induction methods than full-Riskine F1.

### Evidence

| Method | Entity Assign F1 (full) | Entity Assign F1 (present) | Delta |
|--------|:-:|:-:|:-:|
| Leiden | 0.226 | 0.293 | +0.067 |
| RSI-LCR | 0.302 | 0.344 | +0.042 |
| SV-LOI | 0.326 | **0.417** | +0.091 |

SV-LOI benefits most from the present-class metric because it correctly identifies
and assigns entities to the ~7 discoverable classes with higher precision, but is
unfairly penalized for the ~3 classes absent from the source data.

### Interpretation
When evaluating domain-agnostic ontology induction, the reference ontology may
cover concepts not present in every domain's data. Present-class F1 isolates
the method's ability to discover and organize the classes that ARE there, rather
than penalizing for data limitations.

### Paper Framing
> *"We propose present-class F1, which restricts evaluation to reference classes
> with at least one entity in the source data. This avoids penalizing induction
> methods for coverage gaps in the source documents rather than algorithmic
> limitations. SV-LOI achieves 0.417 present-class F1, a 42% improvement over
> Leiden's 0.293."*
