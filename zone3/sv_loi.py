"""Zone 3 — SV-LOI: Structurally-Verified LLM Ontology Induction

Novel ontology induction method that fuses LLM semantic typing with
graph-structural verification. Neither signal alone is sufficient:
LLM typing is accurate but inconsistent across batches; structural
clustering is consistent but over-fragments.

Key insight: entities of the same ontological class share two independent
signals — (1) the LLM recognizes what they ARE (semantic), and
(2) they participate in the same types of relations (structural).
Fusing both with disagreement arbitration eliminates each signal's
failure mode.

Algorithm:
  Phase 1 — LLM Entity Typing:
    Batch entities (20/prompt) with name + type + top relations.
    LLM assigns ontology class from a discovered class vocabulary.
    ~70 LLM calls for 1,351 entities.

  Phase 2 — Structural Consensus Verification:
    Build relation-signature vectors per entity.
    Compute class centroid for each LLM-assigned class.
    Flag entities whose signature deviates >2σ from their class centroid.

  Phase 3 — Disagreement Arbitration:
    For flagged entities, re-query LLM with enriched structural context:
    "You typed this as X, but it structurally resembles Y. Which is correct?"

  Phase 4 — Hierarchy Derivation:
    LLM proposes SUBCLASS_OF from final class set.
    Write ontology layer to Neo4j.

Usage:
  python3 zone3/sv_loi.py
  python3 zone3/sv_loi.py --model qwen2.5:72b
  python3 zone3/sv_loi.py --model qwen2.5:72b --suffix zone3_svloi

Pre-requisite: Zone 2 must have run first (Neo4j populated with :Entity nodes).
"""

from __future__ import annotations

import json
import random
import re
import time
import os
import sys
import argparse
from typing import Any, Optional, Union
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

import config
from zone3.graph_cache import (
    load_cached_entities,
    get_concept_entities,
    get_entity_lane,
    is_concept_entity,
    STRUCTURED_PREFIXES,
)
from zone3._svloi.constants import (
    BATCH_SIZE,
    MAX_MEMBERS_IN_PROMPT,
    MIN_CLASS_SIZE,
    DEVIATION_THRESHOLD,
    MAX_ARBITRATION_BATCH,
    MAX_CLASS_FRACTION,
    _STRUCTURED_PREFIXES,
    TARGET_CLASSES_MIN,
    TARGET_CLASSES_MAX,
    PROTECTED_CLASS_NAMES,
    FORBIDDEN_CLASS_NAMES,
    ZONE2_TYPE_NORMALIZATION,
)
from zone3._svloi.utils import (
    get_llm,
    get_neo4j_graph,
    _sanitize_label,
    _sanitize_rel_type,
    _parse_json_safely,
    _invoke_llm,
    load_entities,
)
from zone3._svloi.sohd import detect_and_split_heterogeneous_classes
from zone3._svloi.records import decompose_records, write_record_decomposition
from zone3._svloi.writer import (
    validate_backbone,
    write_ontology,
    _flush_print,
    _compute_intrinsic_quality,
)
from zone3._svloi.structural import (
    build_structural_signatures,
    structural_consensus_check,
    arbitrate_disagreements,
)
from zone3._svloi.hierarchy import (
    infer_class_relations,
    merge_small_classes,
    merge_leaf_classes,
    derive_interclass_edges,
    derive_hierarchy,
    derive_taxonomy,
    derive_taxonomy_llm_pairwise,
)


# ---------------------------------------------------------------------------
# Phase 1: Record Evidence Analysis
# ---------------------------------------------------------------------------

def analyze_record_evidence(all_entities: list[dict]) -> str:
    """Analyze record entity relation patterns → natural-language evidence.

    Records (POL-xxx, CLM-xxx, etc.) carry domain signals that should inform
    class discovery without directly voting. This function summarizes their
    relation patterns into a short evidence block for the discovery prompt.

    Returns:
        5-8 line natural-language summary of record patterns, e.g.:
        "The graph also contains:
         - 485 claim records with HAS_DATE_OF_LOSS, HAS_BUILDING_DAMAGE_AMOUNT
         - 459 policy records with HAS_POLICY_COST, HAS_RATED_FLOOD_ZONE
         - 854 property records with HAS_PROPERTY_STATE, HAS_REPORTED_CITY"
    """
    # Group records by prefix type
    prefix_stats: dict[str, dict] = {}
    for e in all_entities:
        lane = get_entity_lane(e)
        if lane != "record":
            continue
        # Determine record type from ID prefix
        prefix = e["id"].split("-")[0] + "-" if "-" in e["id"] else "other"
        if prefix not in prefix_stats:
            prefix_stats[prefix] = {"count": 0, "rel_counts": Counter()}
        prefix_stats[prefix]["count"] += 1
        for rel_type in e.get("out_rel_counts", {}):
            if rel_type != "IS_A":  # skip generic IS_A
                prefix_stats[prefix]["rel_counts"][rel_type] += 1

    if not prefix_stats:
        return ""

    # Build summary lines
    lines = ["The graph also contains structured records with these patterns:"]
    prefix_labels = {"POL-": "policy", "CLM-": "claim", "PROP-": "property",
                     "REC-": "record", "PER-": "person"}
    for prefix, stats in sorted(prefix_stats.items(), key=lambda x: -x[1]["count"]):
        label = prefix_labels.get(prefix, prefix.rstrip("-"))
        top_rels = [rel for rel, _ in stats["rel_counts"].most_common(4)]
        if top_rels:
            lines.append(f"  - {stats['count']} {label} records with relations: {', '.join(top_rels)}")

    # Also summarize value entity patterns using domain-agnostic keyword matching
    geo_patterns = {"state", "city", "zip", "address", "county", "location", "community"}
    financial_patterns = {"amount", "cost", "payment", "premium", "fee", "price", "value", "coverage"}
    location_count = 0
    amount_count = 0
    for e in all_entities:
        if get_entity_lane(e) != "value":
            continue
        in_rels = set(r.lower().replace("_", " ") for r in e.get("in_rel_counts", {}).keys())
        in_rels_str = " ".join(in_rels)
        if any(p in in_rels_str for p in geo_patterns):
            location_count += 1
        if any(p in in_rels_str for p in financial_patterns):
            amount_count += 1

    if location_count > 0:
        lines.append(f"  - {location_count} location values (cities, states, zip codes) connected via geographic relations")
    if amount_count > 0:
        lines.append(f"  - {amount_count} financial values (coverage amounts, payments, costs)")

    lines.append("Consider whether these record patterns suggest additional ontology classes.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 2: Class Discovery + Phase 3: Batched LLM Entity Typing (concept entities)
# ---------------------------------------------------------------------------

def discover_class_vocabulary(
    entities: list[dict],
    llm: ChatOllama,
    record_evidence: str = "",
    all_entities: list[dict] | None = None,
) -> tuple[list[str], str]:
    """Two-stage class discovery: detect domain → propose domain-role classes.

    Stage 1: Ask LLM what domain this KG is about.
    Stage 2: Given domain, propose classes for real-world ROLES, not data types.
    Stage 3: Seed from structured record types with significant populations.
    Post-process: Filter out any forbidden data-type class names.

    Returns (class_list, detected_domain).
    """
    print("\n[Phase 2] Class vocabulary discovery (two-stage)", flush=True)

    # Load Zone 2 vocab for entity type hints
    rdir = config.RESULTS_DIR  # default; caller can override via all_entities
    vocab_path = os.path.join(rdir, "zone2_vocab.json")
    zone2_candidates: list[str] = []
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            zone2_types = json.load(f).get("entity_types", [])
        seen_norm: set[str] = set()
        for t in zone2_types:
            norm = ZONE2_TYPE_NORMALIZATION.get(t, t)  # passthrough if not in map
            if norm and norm not in seen_norm:
                zone2_candidates.append(norm)
                seen_norm.add(norm)
        print(f"  Loaded {len(zone2_types)} Zone 2 types → {len(zone2_candidates)} normalized candidates", flush=True)

    # Compute concept-only entity_type distribution (normalized)
    concept_type_counts: Counter = Counter()
    for e in entities:
        et = e.get("entity_type", "Unknown")
        if et not in ("Unknown", "Text", "Numeric", "Date"):
            norm = ZONE2_TYPE_NORMALIZATION.get(et, et)
            if norm:
                concept_type_counts[norm] += 1
    type_hint_block = ""
    if concept_type_counts:
        type_hint_block = "\n".join(
            f"  {t}: {c} concept entities" for t, c in concept_type_counts.most_common(15)
        )
        print(f"  Concept entity type distribution (top 10):", flush=True)
        for t, c in concept_type_counts.most_common(10):
            print(f"    {t}: {c}", flush=True)

    # Collect entity name samples (ungrouped — raw names)
    random.seed(42)
    all_names = [e["id"] for e in entities]
    name_sample = random.sample(all_names, min(80, len(all_names)))

    # Collect relation types
    rel_counts: Counter = Counter()
    for e in entities:
        for r in e.get("out_rel_counts", {}):
            rel_counts[r] += 1
        for r in e.get("in_rel_counts", {}):
            rel_counts[r] += 1
    top_rels = rel_counts.most_common(20)

    # Sample concrete triples with real entity names
    triple_examples = []
    sampled_ents = random.sample(entities, min(100, len(entities)))
    for e in sampled_ents:
        for r in e.get("out_rels", [])[:3]:
            target = r.get("target", "?")
            triple_examples.append(f"  {e['id']} --{r['rel']}--> {target}")
            if len(triple_examples) >= 25:
                break
        if len(triple_examples) >= 25:
            break

    # ---- Stage 1: Detect domain ----
    # Collect source chunk_ids from a small sample to show data breadth
    source_files: set[str] = set()
    sample_for_sources = random.sample(all_entities or entities,
                                       min(200, len(all_entities or entities)))
    for e in sample_for_sources:
        for r in e.get("out_rels", [])[:1]:
            cid = str(r.get("chunk_id", ""))
            if cid and cid != "None":
                source_files.add(cid.split("::")[0] if "::" in cid else cid.rsplit("_chunk_", 1)[0])
    source_hint = ""
    if source_files:
        source_hint = f"\n\nData sources (filenames): {', '.join(sorted(source_files)[:10])}"

    print("  Stage 1: Detecting domain...", flush=True)
    if source_hint:
        print(f"  Source files found: {sorted(source_files)[:10]}", flush=True)
    domain_prompt = f"""Look at these entity names from a knowledge graph:
{', '.join(name_sample[:40])}

And these relationship types:
{', '.join(r for r, _ in top_rels[:15])}{source_hint}

This knowledge graph may cover MULTIPLE lines of business or sub-domains.
What industry/domain(s) is this knowledge graph about?
Answer concisely (e.g., "Insurance (Auto, Renters, Mobile)")."""

    print("  Calling LLM for domain detection...", flush=True)
    domain_raw = _invoke_llm(llm, domain_prompt)
    domain = domain_raw.strip().strip('"').strip("'").strip(".")
    print(f"  Detected domain: {domain}", flush=True)

    # ---- Stage 2: Propose classes ----
    print("  Stage 2: Proposing ontology classes...", flush=True)

    # Inject record evidence if available
    evidence_block = ""
    if record_evidence:
        evidence_block = f"\n{record_evidence}\n"

    # Build type hints block for prompt
    type_hints_prompt = ""
    if type_hint_block:
        type_hints_prompt = f"""
Entity types discovered during extraction:
{type_hint_block}

These types suggest natural ontology classes. Use them as guidance but propose
your own class names based on real-world domain roles.
"""

    prompt = f"""You are designing a DOMAIN ONTOLOGY for a {domain} knowledge graph.
This data may span MULTIPLE lines of business (e.g., auto, renters, mobile insurance).
Propose classes that capture domain roles. Some classes may be shared across LOBs
(e.g., Policy, Claim, Coverage) while others may be LOB-specific if the data supports it.

Here are {len(name_sample)} entity names from the graph (domain concepts only):
{chr(10).join(f'  {n}' for n in name_sample)}

Relationship types:
{chr(10).join(f'  {r} ({c}x)' for r, c in top_rels)}

Example triples:
{chr(10).join(triple_examples)}
{evidence_block}{type_hints_prompt}
Propose {TARGET_CLASSES_MIN}-{TARGET_CLASSES_MAX} ontology classes that categorize these \
entities by their REAL-WORLD ROLE in {domain}.

For each entity, ask: "WHAT IS this thing in the real world?"
- A named coverage type → it IS a type of insurance coverage → Coverage
- A dollar amount → it IS a financial limit on a policy → Limit
- A company, agency, or service provider → it IS an organization → Organization
- A named individual or role → it IS a person → Person
- A street address, city, or zip code → it IS an address → Address
- A type of loss, cause of damage, or incident → it IS a damage type → Damage
- A hazard, risk factor, or threat → it IS a risk → Risk

RULES:
1. Classes = real-world roles, NOT data types
2. FORBIDDEN names: FinancialAmount, TimePeriod, Measurement, Amount, Number,
   Date, Text, Quantity, Value, Metric, Event, Condition, Location
3. Use SINGLE PascalCase words: Coverage, Person, Risk, Structure, Property
4. No class should contain >25% of entities — split large categories
5. Target {TARGET_CLASSES_MIN}-{TARGET_CLASSES_MAX} classes

Output ONLY a JSON array:
[{{"name": "ClassName", "definition": "what entities belong here"}}]
"""
    raw = _invoke_llm(llm, prompt)
    parsed = _parse_json_safely(raw)

    if isinstance(parsed, list):
        classes = [item["name"] for item in parsed if isinstance(item, dict) and "name" in item]
    elif isinstance(parsed, dict):
        classes = list(parsed.keys())
    else:
        classes = []

    # Fallback: extract class names from lines like "1. ClassName" or "- ClassName"
    if not classes:
        line_re = re.compile(
            r'(?:^|\n)\s*(?:\d+[\.\)]\s*|[-*]\s*)'  # "1. " or "- "
            r'\**([A-Z][A-Za-z]+)\**',                # PascalCase word (possibly bold)
        )
        classes = [m.group(1) for m in line_re.finditer(raw)]

    # Sanitize
    classes = [_sanitize_label(c) for c in classes if c]

    # Filter out forbidden data-type names
    classes = [c for c in classes if c.lower() not in FORBIDDEN_CLASS_NAMES]
    print(f"  After forbidden filter: {classes}", flush=True)

    # Deduplicate (case-insensitive)
    seen_lower: set[str] = set()
    deduped = []
    for c in classes:
        if c.lower() not in seen_lower:
            deduped.append(c)
            seen_lower.add(c.lower())
    classes = deduped

    # If LLM still produced too few valid classes, re-prompt with harder steering
    if len(classes) < TARGET_CLASSES_MIN:
        print(f"  Only {len(classes)} valid classes — re-prompting with stronger steering...", flush=True)
        # Collect relation types as hints (domain-agnostic — no hardcoded class names).
        rel_types_sample = sorted(set(
            r["rel"] for e in entities for r in e.get("out_rels", [])[:3] if r.get("rel")
        ))[:20]
        retry_prompt = f"""The previous attempt produced data-type classes that were rejected.

I need {TARGET_CLASSES_MIN}-{TARGET_CLASSES_MAX} ontology classes for a {domain} domain.

These entities exist in the graph:
{chr(10).join(f'  {n}' for n in name_sample[:50])}

Relation types in the graph: {', '.join(rel_types_sample)}

Propose classes that answer "what real-world thing IS this?" for each entity.
Think of chapter titles in a {domain} reference manual.

IMPORTANT: Every class must be a real-world domain concept, NOT a data type.
Look at the entities and relations above — what roles do these entities play
in the {domain} domain? Group them by their real-world function.

FORBIDDEN: Amount, Date, Number, Measurement, Event, Condition, Location, Text, Value

Output ONLY JSON: [{{"name": "ClassName", "definition": "..."}}]"""
        raw2 = _invoke_llm(llm, retry_prompt)
        parsed2 = _parse_json_safely(raw2)
        if isinstance(parsed2, list):
            extra = [item["name"] for item in parsed2 if isinstance(item, dict) and "name" in item]
        elif isinstance(parsed2, dict):
            extra = list(parsed2.keys())
        else:
            extra = []
        # Fallback: line-based extraction
        if not extra:
            line_re2 = re.compile(
                r'(?:^|\n)\s*(?:\d+[\.\)]\s*|[-*]\s*)'
                r'\**([A-Z][A-Za-z]+)\**',
            )
            extra = [m.group(1) for m in line_re2.finditer(raw2)]
        extra = [_sanitize_label(c) for c in extra if c]
        extra = [c for c in extra if c.lower() not in FORBIDDEN_CLASS_NAMES and c.lower() not in seen_lower]
        classes.extend(extra)
        for c in extra:
            seen_lower.add(c.lower())
        print(f"  After retry: {classes}", flush=True)

    # Augment LLM proposal with normalized Zone 2 type candidates.
    # Conservative: only add types with >= MIN_TYPE_SUPPORT concept entities
    # that the LLM didn't already propose.
    MIN_TYPE_SUPPORT = 10
    if zone2_candidates:
        z2_added = 0
        for candidate in zone2_candidates:
            sanitized = _sanitize_label(candidate)
            support = concept_type_counts.get(candidate, 0)
            if (sanitized.lower() not in seen_lower
                    and sanitized.lower() not in FORBIDDEN_CLASS_NAMES
                    and support >= MIN_TYPE_SUPPORT
                    and len(classes) < TARGET_CLASSES_MAX):
                classes.append(sanitized)
                seen_lower.add(sanitized.lower())
                z2_added += 1
                print(f"  [z2-seed] Added {sanitized} ({support} concept entities)", flush=True)
        if z2_added:
            print(f"  ✓ {z2_added} classes added from Zone 2 type hints", flush=True)

    # Fallback: if still below minimum, derive from raw entity_type distribution
    if len(classes) < TARGET_CLASSES_MIN:
        print(f"  WARNING: Only {len(classes)} classes. "
              f"Deriving fallbacks from graph entity types.", flush=True)
        for et_name, cnt in concept_type_counts.most_common(20):
            sanitized = _sanitize_label(et_name)
            if (sanitized.lower() not in seen_lower
                    and sanitized.lower() not in FORBIDDEN_CLASS_NAMES):
                classes.append(sanitized)
                seen_lower.add(sanitized.lower())
            if len(classes) >= TARGET_CLASSES_MIN:
                break

    # Data-driven record seeding: ensure classes exist for structured records
    # with significant populations. NOT leakage — reads from Zone 2 record
    # prefixes (POL-, CLM-, PER-, PROP-, SUR-), not from Riskine.
    RECORD_PREFIX_TO_CLASS = {
        "POL": "Policy", "CLM": "Claim", "PER": "Person", "PROP": "Property",
        "SUR": "Survey",
    }
    seed_entities = all_entities or entities
    for prefix, cls_name in RECORD_PREFIX_TO_CLASS.items():
        count = sum(1 for e in seed_entities if e["id"].startswith(f"{prefix}-"))
        if count > 30 and cls_name.lower() not in seen_lower:
            classes.append(cls_name)
            seen_lower.add(cls_name.lower())
            print(f"  [seed] Added {cls_name} from {count} {prefix}- records", flush=True)

    # Always include "Other" for unclassifiable entities
    if "Other" not in classes:
        classes.append("Other")

    print(f"  ✓ Discovered {len(classes) - 1} classes + Other:", flush=True)
    for c in classes:
        print(f"    - {c}", flush=True)
    return classes, domain


def batch_type_entities(
    entities: list[dict],
    class_vocab: list[str],
    llm: ChatOllama,
    results_dir: str | None = None,
) -> dict[str, str]:
    """Assign ontology class to each entity via batched LLM prompts.

    Structured entities (POL-xxx, CLM-xxx, etc.) are assigned directly
    from their existing entity_type — no LLM calls needed. Only
    LLM-extracted entities go through the batched classification.

    If a few_shot_corrections.json file exists in results_dir, up to 5
    correction examples are prepended to each typing prompt (feedback #7).

    Returns:
        {entity_id: class_name}
    """
    # Load few-shot corrections from previous eval runs (feedback loop)
    corrections_text = ""
    rdir = results_dir or config.RESULTS_DIR
    corrections_path = os.path.join(rdir, "few_shot_corrections.json")
    if os.path.exists(corrections_path):
        try:
            with open(corrections_path) as f:
                corrections = json.load(f)
            if corrections:
                examples = corrections[:5]
                lines = []
                for ex in examples:
                    lines.append(
                        f"- {ex['entity']} -> {ex['correct_class']} "
                        f"(NOT {ex['wrong_class']}: {ex.get('reason', 'based on relations')})"
                    )
                corrections_text = (
                    "\nLEARNED CORRECTIONS from previous runs:\n"
                    + "\n".join(lines) + "\n"
                )
                print(f"  Loaded {len(examples)} few-shot corrections from {corrections_path}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠ Could not load corrections: {e}")
    # Separate entities into 3 groups:
    # 1. Structured (POL-/CLM-/REC-/PER-/PROP-) — pre-assign from entity_type
    # 2. Value entities (Numeric, Date, Text, etc.) — pre-assign as "Other"
    # 3. Concept entities (~208) — these go through LLM classification
    structured_entities: list[dict] = []
    value_entities: list[dict] = []
    llm_entities: list[dict] = []
    for e in entities:
        lane = get_entity_lane(e)
        if lane == "record":
            structured_entities.append(e)
        elif lane == "value":
            value_entities.append(e)
        else:
            llm_entities.append(e)

    assignments: dict[str, str] = {}

    # Pre-assign value entities to "Other" — will be overridden by
    # type_value_entities() (Phase 9) using relation-range induction.
    for e in value_entities:
        assignments[e["id"]] = "Other"

    # Assign structured entities directly from their entity_type.
    # Assign structured entities to matching class in vocab, or "Other".
    # Previously auto-added unmatched entity_types as new classes, which caused
    # class explosion on multi-domain data (8 entity types for flood → fine;
    # 69 entity types for renters+device+auto+survey → 75 classes).
    # Now: unmatched structured entities go to "Other" and get rescued in Phase 5.
    for e in structured_entities:
        et = e.get("entity_type", "Other")
        sanitized_et = _sanitize_label(et) if et and et != "Unknown" else "Other"

        # Match against existing class vocab (case-insensitive).
        matched = False
        for cv in class_vocab:
            if cv.lower() == sanitized_et.lower():
                assignments[e["id"]] = cv
                matched = True
                break

        if not matched:
            assignments[e["id"]] = "Other"

    print(f"\n[Phase 3] Batched LLM entity typing "
          f"({len(llm_entities)} concept entities via LLM, "
          f"{len(structured_entities)} structured + {len(value_entities)} value pre-assigned, "
          f"batch={BATCH_SIZE})...")

    class_list = ", ".join(c for c in class_vocab if c != "Other")
    n_batches = (len(llm_entities) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        batch = llm_entities[start:start + BATCH_SIZE]

        entity_descriptions = []
        for e in batch:
            # Lead with entity name, de-emphasize noisy extraction type
            desc = f"- {e['id']}"
            if e["out_summary"]:
                desc += f"\n    connects to: {'; '.join(e['out_summary'][:5])}"
            if e["in_summary"]:
                desc += f"\n    connected from: {'; '.join(e['in_summary'][:3])}"
            entity_descriptions.append(desc)

        prompt = f"""Classify each entity into the ontology class that describes WHAT IT IS \
in the real world — its domain role, not its data type.

AVAILABLE CLASSES: {class_list}, Other

For each entity, ask: "What IS this thing in the real world?"
- A deductible is a feature of a product/coverage, not a financial amount
- A company or agency is an organization
- A named coverage type is a type of coverage
- A person's name refers to a person, not a text string
{corrections_text}
ENTITIES:
{chr(10).join(entity_descriptions)}

For each entity, output exactly one line: entity_name -> ClassName
Classify by DOMAIN ROLE. Use "Other" only if truly unclassifiable.
"""
        raw = _invoke_llm(llm, prompt)

        # Parse "entity_id -> ClassName" lines
        line_re = re.compile(r'^[- ]*(.+?)\s*->\s*(\w+)\s*$', re.MULTILINE)
        for m in line_re.finditer(raw):
            eid = m.group(1).strip().strip('"').strip("'")
            cls = m.group(2).strip()
            # Validate class name
            sanitized = _sanitize_label(cls)
            if sanitized in class_vocab or sanitized == "Other":
                assignments[eid] = sanitized
            else:
                # Find closest class by prefix match
                for cv in class_vocab:
                    if cv.lower() == sanitized.lower():
                        assignments[eid] = cv
                        break
                else:
                    assignments[eid] = "Other"

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"    Batch {batch_idx + 1}/{n_batches}: {len(assignments)} assigned so far")

    # Assign "Other" to any untyped entities
    for e in entities:
        if e["id"] not in assignments:
            assignments[e["id"]] = "Other"

    # Distribution
    dist = Counter(assignments.values())
    print(f"  ✓ Typed {len(assignments)} entities across {len(dist)} classes", flush=True)
    for cls, cnt in dist.most_common():
        print(f"    {cls}: {cnt}", flush=True)

    return assignments


def rescue_other_entities(
    assignments: dict[str, str],
    entities: list[dict],
    llm: ChatOllama,
    max_other_frac: float = 0.05,
) -> dict[str, str]:
    """Re-type "Other" entities using class examples as context.

    Shows the LLM what each discovered class looks like (name + example
    members + typical relations), then asks it to classify the remaining
    "Other" entities. Fully self-referential — uses only the pipeline's
    own induced classes and entities, no external ontology.
    """
    entity_map_full = {e["id"]: e for e in entities}

    # Only rescue CONCEPT entities — never reclassify dates, numbers, or records
    # that were correctly pre-assigned to "Other" by the value filter.
    other_ids = [
        eid for eid, cls in assignments.items()
        if cls == "Other" and is_concept_entity(entity_map_full.get(eid, {"id": eid, "entity_type": "Unknown"}))
    ]
    total = len(assignments)
    all_other = sum(1 for cls in assignments.values() if cls == "Other")
    target_max = int(total * max_other_frac)

    if len(other_ids) <= target_max:
        print(f"\n[Phase 5] Concept-Other={len(other_ids)}, Value-Other={all_other - len(other_ids)} "
              f"— concept entities below threshold, skipping rescue.", flush=True)
        return assignments

    print(f"\n[Phase 5] Rescuing {len(other_ids)} 'Other' CONCEPT entities "
          f"(skipping {all_other - len(other_ids)} value entities)...", flush=True)

    entity_map = entity_map_full

    # Build class descriptions from current assignments
    dist = Counter(v for v in assignments.values() if v != "Other")
    class_descriptions = []
    for cls, cnt in dist.most_common():
        members = [eid for eid, c in assignments.items() if c == cls]
        # Sample up to 8 member names
        random.seed(42)
        sample = random.sample(members, min(8, len(members)))
        # Get typical relations for this class
        rel_types: set[str] = set()
        for eid in members[:20]:
            e = entity_map.get(eid, {})
            rel_types.update(e.get("out_rel_counts", {}).keys())
            rel_types.update(e.get("in_rel_counts", {}).keys())
        top_rels = sorted(rel_types)[:5]

        class_descriptions.append(
            f"  {cls} ({cnt} members): {', '.join(sample)}\n"
            f"    Typical relations: {', '.join(top_rels) if top_rels else 'none'}"
        )

    class_list = ", ".join(c for c, _ in dist.most_common())

    # Re-type in batches
    updated = dict(assignments)
    rescued = 0
    n_batches = (len(other_ids) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        batch = other_ids[start:start + BATCH_SIZE]

        entity_descs = []
        for eid in batch:
            e = entity_map.get(eid, {})
            desc = f"- {eid}"
            if e.get("out_summary"):
                desc += f"\n    connects to: {'; '.join(e['out_summary'][:4])}"
            if e.get("in_summary"):
                desc += f"\n    connected from: {'; '.join(e['in_summary'][:3])}"
            entity_descs.append(desc)

        prompt = f"""These entities were not classified in the first pass. Using the class descriptions \
below, assign each entity to the BEST matching class.

DISCOVERED CLASSES (with examples and typical relations):
{chr(10).join(class_descriptions)}

UNCLASSIFIED ENTITIES:
{chr(10).join(entity_descs)}

For each entity, decide WHAT IT IS based on its name and relationships.
Match it to the class whose members it most resembles.
Only use "Other" if the entity truly does not fit ANY class.

Output one line per entity: entity_name -> ClassName
"""
        raw = _invoke_llm(llm, prompt)

        line_re = re.compile(r'^[- ]*(.+?)\s*->\s*(\w+)\s*$', re.MULTILINE)
        for m in line_re.finditer(raw):
            eid = m.group(1).strip().strip('"').strip("'")
            cls = _sanitize_label(m.group(2).strip())
            if eid in updated and updated[eid] == "Other":
                if cls in dist or cls == "Other":
                    if cls != "Other":
                        updated[eid] = cls
                        rescued += 1
                else:
                    # Try case-insensitive match
                    for known_cls in dist:
                        if known_cls.lower() == cls.lower():
                            updated[eid] = known_cls
                            rescued += 1
                            break

        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            remaining = sum(1 for v in updated.values() if v == "Other")
            print(f"    Batch {batch_idx + 1}/{n_batches}: rescued {rescued} so far, {remaining} Other remaining", flush=True)

    remaining_other = sum(1 for v in updated.values() if v == "Other")
    print(f"  ✓ Rescued {rescued} entities. Other: {len(other_ids)} → {remaining_other} ({100*remaining_other/total:.1f}%)", flush=True)

    new_dist = Counter(updated.values())
    for cls, cnt in new_dist.most_common():
        print(f"    {cls}: {cnt}", flush=True)

    return updated


def type_value_entities(
    assignments: dict[str, str],
    entities: list[dict],
    class_vocab: list[str],
    confidence_threshold: float = 0.50,
) -> dict[str, str]:
    """Type value entities via relation-range induction (domain-agnostic).

    Learns a relation_type → range_class mapping from already-typed entities,
    then uses incoming relation types to type untyped value entities.

    This is the standard KG approach: type objects from the predicate's range,
    not from the subject's class. Domain-agnostic because it learns from
    graph regularities, not hardcoded patterns.

    Algorithm:
        1. Build a (relation_type × class) score matrix from typed entities
        2. For each relation, compute P(class | relation) = range distribution
        3. For each value entity, aggregate P(class | incoming_relation) across edges
        4. Assign if confidence > threshold; otherwise stay "Other"

    Args:
        confidence_threshold: Min P(class|relation) to assign (default: 0.50)
    """
    print(f"\n[Phase 9] Value entity typing via relation-range induction...", flush=True)

    entity_map = {e["id"]: e for e in entities}
    updated = dict(assignments)

    # Step 1: Build relation → range-class score matrix from typed entities
    # For each relation type, count how often its TARGET belongs to each class
    rel_range: dict[str, Counter] = defaultdict(Counter)
    for e in entities:
        eid = e["id"]
        src_cls = assignments.get(eid, "Other")
        for rel in e.get("out_rels", []):
            rel_type = rel.get("rel", "")
            tgt_eid = rel.get("target", "")
            tgt_cls = assignments.get(tgt_eid, "Other")
            if tgt_cls != "Other" and rel_type:
                rel_range[rel_type][tgt_cls] += 1

    # Step 2: Compute P(class | relation) for each relation type
    rel_to_class: dict[str, tuple[str, float]] = {}  # rel_type → (best_class, confidence)
    for rel_type, class_counts in rel_range.items():
        total = sum(class_counts.values())
        if total < 2:
            continue  # too little evidence
        best_cls, best_count = class_counts.most_common(1)[0]
        confidence = best_count / total
        if confidence >= confidence_threshold and best_cls in class_vocab:
            rel_to_class[rel_type] = (best_cls, confidence)

    if rel_to_class:
        print(f"  Learned {len(rel_to_class)} relation→class mappings:", flush=True)
        for rt, (cls, conf) in sorted(rel_to_class.items(), key=lambda x: -x[1][1])[:15]:
            print(f"    {rt:<45} → {cls:<15} (conf={conf:.2f})", flush=True)

    # Step 3: Type value entities using their incoming relation's range class
    reclassified = 0
    class_gains: Counter = Counter()

    for e in entities:
        eid = e["id"]
        if get_entity_lane(e) != "value" or updated.get(eid) != "Other":
            continue

        # Aggregate evidence from incoming relations
        class_evidence: Counter = Counter()
        for rel_type, count in e.get("in_rel_counts", {}).items():
            if rel_type in rel_to_class:
                cls, conf = rel_to_class[rel_type]
                class_evidence[cls] += count * conf  # weight by confidence

        if not class_evidence:
            continue

        # Assign to highest-evidence class
        best_cls, best_score = class_evidence.most_common(1)[0]
        total_evidence = sum(class_evidence.values())
        fraction = best_score / total_evidence if total_evidence > 0 else 0

        if fraction >= confidence_threshold and best_cls in class_vocab:
            updated[eid] = best_cls
            reclassified += 1
            class_gains[best_cls] += 1

    if reclassified > 0:
        print(f"  ✓ Reclassified {reclassified} value entities from Other:", flush=True)
        for cls, cnt in class_gains.most_common():
            print(f"    → {cls}: +{cnt}", flush=True)
    else:
        print(f"  ✓ No value entities met the {confidence_threshold:.0%} threshold — all stay as Other", flush=True)

    return updated, rel_to_class


def rebalance_mega_classes(
    assignments: dict[str, str],
    entities: list[dict],
    class_vocab: list[str],
    llm: ChatOllama,
    max_fraction: float = 0.40,
    use_old_rebalance: bool = False,
) -> tuple[dict[str, str], list[str]]:
    """Split any class that exceeds max_fraction of CONCEPT entities.

    Uses concept-entity count as denominator (not total entities) because
    records naturally dominate 2-3 classes. A class with 600 records + 30
    concepts should NOT trigger rebalancing — only concept concentration matters.

    Args:
        max_fraction: Max fraction of concept entities per class (default: 0.40)
        use_old_rebalance: If True, use old behavior (total entities, 0.25 threshold)
    """
    entity_map = {e["id"]: e for e in entities}

    if use_old_rebalance:
        # Old behavior: count all entities, threshold=0.25
        total = len(assignments)
        threshold = int(total * MAX_CLASS_FRACTION)
        dist = Counter(assignments.values())
    else:
        # New behavior: count concept entities only
        concept_counts: Counter = Counter()
        for eid, cls in assignments.items():
            if get_entity_lane(entity_map.get(eid, {"id": eid, "entity_type": "Unknown"})) == "concept":
                concept_counts[cls] += 1
        total_concepts = sum(concept_counts.values())
        threshold = int(total_concepts * max_fraction)
        dist = concept_counts

    updated_vocab = list(class_vocab)

    # Fixed floor: don't split classes with <50 members regardless of dataset
    # size. Splitting 30 entities into sub-classes is never meaningful.
    # The real gate is max_fraction (40%) which IS percentage-based.
    MIN_CONCEPT_ABSOLUTE = 50
    mega_classes = [
        (cls, cnt) for cls, cnt in dist.items()
        if cnt > threshold and cnt >= MIN_CONCEPT_ABSOLUTE and cls != "Other"
    ]

    if not mega_classes:
        denom_type = "total" if use_old_rebalance else "concept"
        print(f"\n[Phase 4] No mega-classes detected ({denom_type} threshold={threshold}) — skipping rebalance.", flush=True)
        return assignments, updated_vocab

    print(f"\n[Phase 4] Rebalancing {len(mega_classes)} mega-classes (threshold={threshold})...", flush=True)
    entity_map = {e["id"]: e for e in entities}
    updated = dict(assignments)

    for mega_cls, mega_cnt in mega_classes:
        print(f"  Splitting {mega_cls} ({mega_cnt} members)...", flush=True)

        # Get members with context
        members = [eid for eid, c in updated.items() if c == mega_cls]
        # Sample for the LLM
        random.seed(42)
        sample = random.sample(members, min(40, len(members)))
        sample_descs = []
        for eid in sample:
            e = entity_map.get(eid, {})
            desc = f"  {eid}"
            if e.get("out_summary"):
                desc += f" (rels: {'; '.join(e['out_summary'][:2])})"
            sample_descs.append(desc)

        other_classes = [c for c in updated_vocab if c != mega_cls and c != "Other"]

        prompt = f"""The ontology class "{mega_cls}" contains {mega_cnt} entities — too many for a single class.

Here are example members:
{chr(10).join(sample_descs)}

Other existing classes: {', '.join(other_classes)}

These entities probably represent DIFFERENT real-world concepts that were lumped together. \
Propose 2-4 sub-classes to replace "{mega_cls}". Each sub-class should capture a distinct \
domain concept.

RULES:
- Use PascalCase single-word names
- Propose concepts for WHAT these entities ARE (e.g., Person, Product, Coverage, Risk, \
Structure, Document, Event) — not data types
- Some members might actually belong to existing classes listed above — that's fine
- Output JSON: [{{"name": "SubClass1", "definition": "...", "example_members": ["entity1", "entity2"]}}]
"""
        raw = _invoke_llm(llm, prompt)
        parsed = _parse_json_safely(raw)

        if not isinstance(parsed, list) or len(parsed) < 2:
            print(f"    ✗ Could not split {mega_cls} — LLM did not propose sub-classes", flush=True)
            continue

        new_classes = []
        for item in parsed:
            if isinstance(item, dict) and "name" in item:
                name = _sanitize_label(item["name"])
                if name and name != mega_cls and name not in updated_vocab:
                    new_classes.append(name)
                    updated_vocab.append(name)

        if not new_classes:
            print(f"    ✗ No valid new classes proposed for {mega_cls}", flush=True)
            continue

        # Remove mega_cls from vocab, replace with new classes
        if mega_cls in updated_vocab:
            updated_vocab.remove(mega_cls)

        # Re-type the mega-class members using new sub-classes + existing classes
        all_target_classes = new_classes + other_classes
        retype_vocab = all_target_classes + ["Other"]

        # Re-type in batches
        n_batches = (len(members) + BATCH_SIZE - 1) // BATCH_SIZE
        retype_class_list = ", ".join(all_target_classes)

        for bi in range(n_batches):
            batch = members[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            batch_descs = []
            for eid in batch:
                e = entity_map.get(eid, {})
                desc = f"- {eid} (type: {e.get('entity_type', '?')})"
                if e.get("out_summary"):
                    desc += f"\n    rels: {'; '.join(e['out_summary'][:4])}"
                batch_descs.append(desc)

            rprompt = f"""Classify each entity by WHAT IT IS in the real world.

CLASSES: {retype_class_list}, Other

ENTITIES:
{chr(10).join(batch_descs)}

Output: entity_name -> ClassName
"""
            raw2 = _invoke_llm(llm, rprompt)
            line_re = re.compile(r'^[- ]*(.+?)\s*->\s*(\w+)\s*$', re.MULTILINE)
            for m in line_re.finditer(raw2):
                eid = m.group(1).strip().strip('"').strip("'")
                cls = _sanitize_label(m.group(2).strip())
                if eid in updated and (cls in updated_vocab or cls == "Other"):
                    updated[eid] = cls

        new_dist = Counter(updated[m] for m in members)
        print(f"    ✓ Split into: {dict(new_dist.most_common())}", flush=True)

    return updated, updated_vocab


# ---------------------------------------------------------------------------
# Phase 6/10: Structural Consensus Verification
# ---------------------------------------------------------------------------

def propagate_to_records(
    concept_assignments: dict[str, str],
    entities: list[dict],
    entity_map: dict[str, dict],
    class_vocab: list[str],
    llm: Optional[ChatOllama] = None,
) -> dict[str, str]:
    """Map record entities to induced ontology classes via schema mapping.

    Records (POL-xxx, CLM-xxx, PROP-xxx) come from tabular data and exist in
    a disconnected subgraph from concept entities — they connect only to value
    entities (numbers, dates, categories), not to concepts. Neighbor-majority
    voting therefore cannot work.

    Instead, we use schema mapping: group records by their Zone 2 entity_type,
    collect each type's relation profile (what relations it participates in),
    and ask the LLM which induced ontology class best fits each record type.
    This is domain-agnostic — works for any tabular schema.

    Algorithm:
        1. Group record entities by Zone 2 entity_type
        2. For each type, collect top relation names as a profile
        3. Ask LLM: "Given classes [X, Y, Z], which class fits entities
           that have relations [HAS_COVERAGE, HAS_DAMAGE_AMOUNT, ...]?"
        4. Bulk-assign all records of that type to the mapped class

    This is analogous to R2RML/OBDA schema mapping (Sequeda et al. 2012),
    but learned from the induced ontology rather than a pre-existing one.

    Args:
        concept_assignments: verified concept entity → class mapping
        entities: all entities
        entity_map: {eid: entity_dict}
        class_vocab: list of induced ontology class names
        llm: LLM for schema mapping (optional; falls back to heuristic)

    Returns:
        (record_assignments, redirects) where:
          record_assignments: {record_eid: class_name}
          redirects: {old_class: new_class} for record-type names that were remapped
    """
    print("\n[Phase 8] Schema mapping: record types → induced classes...", flush=True)

    # Step 1: Group records by entity_type, collect relation profiles
    type_records: dict[str, list[str]] = defaultdict(list)
    type_relations: dict[str, Counter] = defaultdict(Counter)

    for e in entities:
        eid = e["id"]
        if get_entity_lane(e) != "record":
            continue
        etype = e.get("entity_type", "Unknown")
        type_records[etype].append(eid)
        for rel_type, count in e.get("out_rel_counts", {}).items():
            type_relations[etype][rel_type] += count

    if not type_records:
        print("  No record entities found.", flush=True)
        return {}

    print(f"  Found {len(type_records)} record types: "
          f"{', '.join(f'{t} ({len(eids)})' for t, eids in type_records.items())}",
          flush=True)

    # Step 2: Build schema mapping via LLM (or heuristic fallback)
    type_to_class: dict[str, str] = {}
    valid_classes = [c for c in class_vocab if c != "Other"]

    # Build target classes: exclude ONLY record-suffixed names (e.g., "PolicyRecord",
    # "ClaimRecord") but keep classes whose names match base record types
    # (e.g., keep "Person" even though "Person" is also a record type,
    # because Person IS a valid ontology class).
    record_suffixed_names = set()
    for et in type_records:
        sanitized = _sanitize_label(et).lower()
        if sanitized.endswith("record"):
            record_suffixed_names.add(sanitized)
    target_classes = [c for c in valid_classes
                      if c.lower() not in record_suffixed_names]

    if llm and target_classes:
        for etype, rel_counts in type_relations.items():
            top_rels = [r for r, _ in rel_counts.most_common(12)]
            rels_str = ", ".join(top_rels)

            # Check if record type name directly matches a class (strong prior)
            # This handles: Person→Person, Property→Property, Record→Other
            name_match = None
            etype_base = etype.replace("Record", "")  # ClaimRecord → Claim
            for c in target_classes:
                if c.lower() == etype.lower():
                    # Exact match: "Person" record type → "Person" class
                    name_match = c
                    break
                if c.lower() == etype_base.lower() and etype_base:
                    # Base match: "ClaimRecord" → "Claim" class
                    name_match = c
                    break

            # If exact name match exists, skip LLM — use it directly
            if name_match and etype.lower() == name_match.lower():
                type_to_class[etype] = name_match
                print(f"  Direct: {etype} → {name_match} (exact name match)", flush=True)
                continue

            prompt = (
                "Schema mapping task: assign a database record type to an ontology class.\n\n"
                f"Record type: {etype}\n"
                f"Typical relations: {rels_str}\n\n"
                f"Available ontology classes: {', '.join(target_classes)}\n\n"
                "Which ONE ontology class best describes what this record type IS?\n"
                "Think about what the record represents as a real-world concept.\n"
                + (f"\nNOTE: The record type name '{etype}' closely matches class '{name_match}'. "
                   f"Prefer '{name_match}' unless the relations strongly contradict this.\n"
                   if name_match else "")
                + "\nAnswer with ONLY the class name, nothing else."
            )

            try:
                response = llm.invoke(prompt)
                answer = response.content.strip().strip('"').strip("'")
                matched = None
                for c in target_classes:
                    if c.lower() == answer.lower():
                        matched = c
                        break
                if not matched:
                    for c in target_classes:
                        if answer.lower() in c.lower() or c.lower() in answer.lower():
                            matched = c
                            break
                if matched:
                    type_to_class[etype] = matched
                    print(f"  LLM: {etype} → {matched}", flush=True)
                else:
                    print(f"  LLM: {etype} → '{answer}' (no match in vocab)",
                          flush=True)
            except Exception as exc:
                print(f"  LLM error for {etype}: {exc}", flush=True)

    # Heuristic fallback for unmapped types
    for etype in type_records:
        if etype not in type_to_class:
            sanitized = _sanitize_label(etype)
            for c in target_classes:
                if c.lower() == sanitized.lower():
                    type_to_class[etype] = c
                    print(f"  Heuristic: {etype} → {c} (exact match)", flush=True)
                    break
            else:
                type_to_class[etype] = "Other"
                print(f"  Heuristic: {etype} → Other (no match)", flush=True)

    # Step 3: Bulk-assign records AND redirect any entities already typed
    # with the record-type class name (e.g., "PolicyRecord") to the mapped
    # ontology class (e.g., "Product"). This cleans up record-type names
    # that leaked into the class vocabulary during Phase 2 discovery.
    record_assignments: dict[str, str] = {}
    redirects: dict[str, str] = {}  # old_class → new_class
    mapped = 0
    unmapped = 0

    for etype, eids in type_records.items():
        cls = type_to_class.get(etype, "Other")
        # Record the redirect so caller can fix concept entities too
        sanitized = _sanitize_label(etype)
        if cls != "Other" and sanitized != cls:
            redirects[sanitized] = cls
        for eid in eids:
            record_assignments[eid] = cls
            if cls != "Other":
                mapped += 1
            else:
                unmapped += 1

    if redirects:
        print(f"  Class redirects: {redirects}", flush=True)

    print(f"  ✓ {mapped} records mapped to ontology classes, "
          f"{unmapped} unmapped", flush=True)
    return record_assignments, redirects



# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_sv_loi(
    model: str = config.OLLAMA_MODEL,
    suffix: str = "zone3_svloi",
    skip_verify: bool = False,
    skip_arbitrate: bool = False,
    skip_consolidate: bool = False,
    skip_record_propagation: bool = False,
    skip_sohd: bool = False,
    use_old_rebalance: bool = False,
    seed: int = 42,
    results_dir: str | None = None,
) -> dict:
    """Run the full SV-LOI pipeline (7 stages).

    Stage 1: Load + Prepare     (load cache, record evidence, structural sigs)
    Stage 2: Discover Classes   (domain detection, class vocabulary proposal)
    Stage 3: Classify Entities  (batch typing, rebalance, rescue)
    Stage 4: Verify + Propagate (concept verify, record propagation, value typing, full verify)
    Stage 5: Consolidate        (5-way class relations, LLM validation, structural merge)
    Stage 6: Build Structure    (LLM pairwise taxonomy, subsumption, associations, decomposition)
    Stage 7: Write + Report     (confidence scoring, quality metrics, Neo4j write)

    Ablation flags:
        skip_verify:              Skip structural consensus verification
        skip_arbitrate:           Skip disagreement arbitration
        skip_consolidate:         Skip LLM-guided class consolidation
        skip_record_propagation:  Skip record propagation (use Zone 2 entity_type)
        use_old_rebalance:        Use old rebalance (total entities, 25% threshold)
        seed:                     Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    rdir = results_dir or config.RESULTS_DIR
    os.makedirs(rdir, exist_ok=True)

    ablation_flags = []
    if skip_verify:
        ablation_flags.append("no-verify")
    if skip_arbitrate:
        ablation_flags.append("no-arbitrate")
    if skip_consolidate:
        ablation_flags.append("no-consolidate")
    if skip_record_propagation:
        ablation_flags.append("no-record-propagation")
    if skip_sohd:
        ablation_flags.append("no-sohd")
    if use_old_rebalance:
        ablation_flags.append("old-rebalance")
    ablation_str = f" [ABLATION: {', '.join(ablation_flags)}]" if ablation_flags else ""

    _flush_print("=" * 70)
    _flush_print("CS584 Capstone — Zone 3: SV-LOI (7-Stage Pipeline)")
    _flush_print(f"Structurally-Verified LLM Ontology Induction{ablation_str}")
    _flush_print(f"Model: {model} | Suffix: {suffix} | Seed: {seed}")
    _flush_print("=" * 70)

    start = time.time()

    # ===================================================================
    # Stage 1: Load + Prepare
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 1: Load + Prepare")
    _flush_print("─" * 50)

    entities = load_cached_entities(fmt="sv_loi", results_dir=rdir)
    if not entities:
        return {"error": "no entities"}

    llm = get_llm(model)
    entity_map_all = {e["id"]: e for e in entities}

    # Record evidence analysis (informs class discovery)
    record_evidence = analyze_record_evidence(entities)
    if record_evidence:
        _flush_print(f"\n  Record evidence:\n{record_evidence}")

    # Concept entities for class discovery
    concept_entities = get_concept_entities(entities)

    # Structural signatures (used in verify + consolidate stages)
    features, entity_ids, feature_names = build_structural_signatures(entities)

    _flush_print(f"  ✓ Loaded {len(entities)} entities "
                 f"({len(concept_entities)} concepts), "
                 f"{len(feature_names)} features")

    # ===================================================================
    # Stage 2: Discover Classes
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 2: Discover Classes")
    _flush_print("─" * 50)

    class_vocab, detected_domain = discover_class_vocabulary(
        concept_entities, llm, record_evidence=record_evidence,
        all_entities=entities,
    )

    # ===================================================================
    # Stage 3: Classify Entities (batch typing + rebalance + rescue)
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 3: Classify Entities")
    _flush_print("─" * 50)

    # Pass 1: LLM batch typing
    assignments = batch_type_entities(entities, class_vocab, llm, results_dir=rdir)

    # Pass 2: Rebalance mega-classes + rescue Other (combined post-pass)
    assignments, class_vocab = rebalance_mega_classes(
        assignments, entities, class_vocab, llm,
        use_old_rebalance=use_old_rebalance,
    )
    assignments = rescue_other_entities(assignments, entities, llm)

    # ===================================================================
    # Stage 4: Verify + Propagate
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 4: Verify + Propagate")
    _flush_print("─" * 50)

    # --- Decision provenance tracking ---
    provenance: dict[str, dict] = {}
    for eid, cls in assignments.items():
        provenance[eid] = {"llm_type": cls, "flagged": False, "arbitrated": False}

    total_flagged = 0

    # Sub-pass A: Concept-only verification (clean centroids, no record pollution)
    if skip_verify:
        _flush_print("\n  [ABLATION] Skipping structural verification")
    else:
        _flush_print("\n  Sub-pass A: Concept-only structural verification")
        concept_only_assignments = {
            eid: cls for eid, cls in assignments.items()
            if get_entity_lane(entity_map_all.get(eid, {"id": eid, "entity_type": "Unknown"})) == "concept"
        }
        class_stats, flagged = structural_consensus_check(
            concept_only_assignments, features, entity_ids,
        )
        total_flagged += len(flagged)

        for f_entry in flagged:
            eid = f_entry["entity_id"]
            if eid in provenance:
                provenance[eid]["flagged"] = True
                provenance[eid]["outlier_score"] = f_entry["similarity"]
                provenance[eid]["class_mean_sim"] = f_entry["class_mean_sim"]
                provenance[eid]["nearest_alt"] = f_entry["nearest_class"]

        if flagged and not skip_arbitrate:
            pre_arb = dict(assignments)
            assignments = arbitrate_disagreements(flagged, entities, assignments, class_vocab, llm)
            for eid in assignments:
                if eid in pre_arb and assignments[eid] != pre_arb[eid]:
                    if eid in provenance:
                        provenance[eid]["arbitrated"] = True
                        provenance[eid]["pre_arb_class"] = pre_arb[eid]
        elif flagged and skip_arbitrate:
            _flush_print(f"  [ABLATION] Skipping arbitration — {len(flagged)} flagged")

    # Sub-pass B: Record propagation + value typing
    if skip_record_propagation:
        _flush_print("\n  [ABLATION] Skipping record propagation (using Zone 2 types)")
    else:
        _flush_print("\n  Sub-pass B: Record propagation + value typing")
        concept_assignments_verified = {
            eid: cls for eid, cls in assignments.items()
            if get_entity_lane(entity_map_all.get(eid, {"id": eid, "entity_type": "Unknown"})) == "concept"
        }
        record_assignments, redirects = propagate_to_records(
            concept_assignments_verified, entities, entity_map_all,
            class_vocab=class_vocab, llm=llm,
        )
        for eid, cls in record_assignments.items():
            assignments[eid] = cls
        # Apply redirects: fix entities typed with record-type class names
        if redirects:
            redirected = 0
            for eid in list(assignments.keys()):
                old_cls = assignments[eid]
                if old_cls in redirects:
                    assignments[eid] = redirects[old_cls]
                    redirected += 1
            for old_cls, new_cls in redirects.items():
                if old_cls in class_vocab:
                    class_vocab.remove(old_cls)
                if new_cls not in class_vocab:
                    class_vocab.append(new_cls)
            _flush_print(f"  Redirected {redirected} entities, "
                         f"cleaned vocab: removed {list(redirects.keys())}")

    # Value typing AFTER record propagation (needs record neighbor classes)
    assignments, rel_to_class = type_value_entities(assignments, entities, class_vocab)

    # Sub-pass C: Full verification (all entities now typed)
    if not skip_verify:
        _flush_print("\n  Sub-pass C: Full structural verification")
        class_stats, flagged = structural_consensus_check(assignments, features, entity_ids)
        total_flagged += len(flagged)

        for f_entry in flagged:
            eid = f_entry["entity_id"]
            if eid in provenance:
                provenance[eid]["flagged"] = True
                provenance[eid]["outlier_score"] = f_entry["similarity"]

        if flagged and not skip_arbitrate:
            pre_arb = dict(assignments)
            assignments = arbitrate_disagreements(flagged, entities, assignments, class_vocab, llm)
            for eid in assignments:
                if eid in pre_arb and assignments[eid] != pre_arb[eid]:
                    if eid in provenance:
                        provenance[eid]["arbitrated"] = True
                        provenance[eid]["pre_arb_class"] = pre_arb[eid]

    # ===================================================================
    # Stage 5: Consolidate Classes (5-way inference + LLM validation + structural merge)
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 5: Consolidate Classes")
    _flush_print("─" * 50)

    # Two-lane: concept entities drive consolidation
    concept_assignments = {
        eid: cls for eid, cls in assignments.items()
        if get_entity_lane(entity_map_all.get(eid, {"id": eid, "entity_type": "Unknown"})) == "concept"
    }
    _flush_print(f"  {len(concept_assignments)} concepts drive consolidation")

    pre_consolidate = dict(assignments)
    if skip_consolidate:
        _flush_print("\n  [ABLATION] Skipping class relation inference")
        hierarchy = derive_hierarchy(assignments, llm)
    else:
        # 5-way class relation inference (concept entities only)
        concept_assignments, hierarchy = infer_class_relations(
            concept_assignments, entities, llm,
        )
        # Propagate concept consolidation to ALL entities
        remap_candidates: dict[str, set[str]] = defaultdict(set)
        for eid, new_cls in concept_assignments.items():
            old_cls = pre_consolidate.get(eid, "Other")
            if old_cls != new_cls and old_cls != "Other":
                remap_candidates[old_cls].add(new_cls)

        class_remap = {
            old: next(iter(news))
            for old, news in remap_candidates.items()
            if len(news) == 1
        }
        ambiguous = {old: news for old, news in remap_candidates.items() if len(news) > 1}
        if ambiguous:
            _flush_print(f"  ⚠ Ambiguous remaps skipped: {dict(ambiguous)}")

        if class_remap:
            _flush_print(f"  Propagating {len(class_remap)} class remaps to all entities...")
            for eid in list(assignments.keys()):
                old = assignments[eid]
                if old in class_remap:
                    assignments[eid] = class_remap[old]
        for eid, cls in concept_assignments.items():
            assignments[eid] = cls

    # Record consolidation changes in provenance
    for eid in assignments:
        if eid in pre_consolidate and assignments[eid] != pre_consolidate[eid]:
            if eid in provenance:
                provenance[eid]["consolidated_from"] = pre_consolidate[eid]

    # LLM class validation BEFORE structural merge (reordered from old pipeline)
    # This prevents merge_small_classes from merging classes that LLM would keep
    assignments = merge_leaf_classes(assignments, entities, llm=llm)

    # Structural merge of small classes AFTER LLM validation
    assignments = merge_small_classes(assignments, features, entity_ids)

    # ===================================================================
    # Stage 6: Build Ontology Structure (taxonomy + associations + decomposition)
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 6: Build Ontology Structure")
    _flush_print("─" * 50)

    # SOHD: Structural Ontological Heterogeneity Detection
    # Runs FIRST so taxonomy signals operate on the post-split class universe
    sohd_stats: dict = {}
    sohd_edges: list[tuple[str, str]] = []
    if skip_sohd:
        _flush_print("\n  [ABLATION] Skipping SOHD hierarchy deepening")
    else:
        _flush_print("\n  SOHD: Detecting structurally heterogeneous classes...")
        assignments, sohd_edges, sohd_stats = detect_and_split_heterogeneous_classes(
            assignments, entities, llm=llm, seed=seed,
        )
        _flush_print(f"  ✓ SOHD: {len(sohd_edges)} new IS-A edges from "
                     f"{len(set(c for c, _ in sohd_edges))} new subclasses")

    # Snapshot 5-way IS-A edges from Stage 5
    n_5way_edges = len(hierarchy)

    # IS-A taxonomy via LLM pairwise judgment (primary signal)
    # Now operates on post-SOHD assignments (includes new subclasses)
    taxonomy_llm_edges = derive_taxonomy_llm_pairwise(assignments, entities, llm)

    # Concept-only subsumption as structural validation (secondary signal)
    taxonomy_subsumption_edges = derive_taxonomy(assignments, entities, llm)

    # Collect all candidate IS-A edges from all sources
    all_candidate_isa: list[tuple[str, str]] = list(hierarchy)  # 5-way edges
    all_candidate_isa.extend(taxonomy_llm_edges)
    all_candidate_isa.extend(taxonomy_subsumption_edges)
    all_candidate_isa.extend(sohd_edges)

    # Global DAG enforcement across ALL sources:
    # 1. Filter out edges referencing unknown classes
    # 2. Remove cycles (keep first edge, reject contradicting edge)
    # 3. Enforce max depth 3
    final_classes = set(c for c in assignments.values() if c != "Other")
    MAX_DEPTH = 4

    # Filter unknown classes
    valid_edges = [(c, p) for c, p in all_candidate_isa
                   if c in final_classes and p in final_classes and c != p]
    n_unknown = len(all_candidate_isa) - len(valid_edges)
    if n_unknown > 0:
        _flush_print(f"  Filtered {n_unknown} edges referencing unknown classes")

    # Deduplicate
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for edge in valid_edges:
        if edge not in seen:
            deduped.append(edge)
            seen.add(edge)

    # DAG enforcement: no cycles, max depth, single parent per child
    parent_of: dict[str, str] = {}

    def _depth(node: str) -> int:
        d, cur, visited = 0, node, set()
        while cur in parent_of and cur not in visited:
            visited.add(cur)
            cur = parent_of[cur]
            d += 1
        return d

    hierarchy = []
    rejected_cycles = []
    for child, parent in deduped:
        # Cycle check: walk up from parent — if we reach child, it's a cycle
        cur, is_cycle, visited = parent, False, set()
        while cur in parent_of and cur not in visited:
            visited.add(cur)
            cur = parent_of[cur]
            if cur == child:
                is_cycle = True
                break
        if is_cycle:
            rejected_cycles.append((child, parent))
            continue
        # Depth check
        if _depth(parent) + 1 > MAX_DEPTH:
            continue
        # Single parent (first-come wins)
        if child not in parent_of:
            parent_of[child] = parent
            hierarchy.append((child, parent))

    if rejected_cycles:
        _flush_print(f"  Rejected {len(rejected_cycles)} cyclic edges:")
        for c, p in rejected_cycles:
            _flush_print(f"    {c} → {p} (would create cycle)")

    # Association edges — SEPARATE from IS-A (these are HAS-A / REFERENCES)
    _flush_print("\n  Association edges from entity connections...")
    data_edges = derive_interclass_edges(assignments, entities)
    existing_isa = set(hierarchy)
    # Deduplicate: remove any association that duplicates an IS-A edge
    associations = [e for e in data_edges
                    if e not in existing_isa
                    and e[0] in final_classes and e[1] in final_classes]

    n_llm_tax = len(taxonomy_llm_edges)
    n_sub_tax = len(taxonomy_subsumption_edges)
    n_assoc = len(associations)
    _flush_print(f"  IS-A edges: {len(hierarchy)} "
                 f"({n_llm_tax} LLM-pairwise + {n_sub_tax} subsumption + {n_5way_edges} from 5-way)")
    _flush_print(f"  Association edges: {n_assoc} (ASSOCIATED_WITH, not SUBCLASS_OF)")

    # Record decomposition (Q5 bridge fix)
    decomposition = decompose_records(assignments, entities, rel_to_class=rel_to_class)

    # Backbone validation
    backbone_report = validate_backbone(assignments, entities)

    # ===================================================================
    # Stage 7: Write + Report
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 7: Write + Report")
    _flush_print("─" * 50)

    # Confidence scoring
    for eid, cls in assignments.items():
        if eid in provenance:
            provenance[eid]["final_type"] = cls
            p = provenance[eid]
            if cls == "Other":
                p["confidence"] = 0.3
            elif p.get("arbitrated"):
                p["confidence"] = 0.5
            elif p.get("flagged"):
                p["confidence"] = 0.7
            elif p.get("consolidated_from"):
                p["confidence"] = 0.8
            else:
                p["confidence"] = 1.0

    # Intrinsic quality metrics
    final_dist = Counter(v for v in assignments.values() if v != "Other")
    quality_metrics = _compute_intrinsic_quality(
        assignments, entities, entity_map_all, final_dist,
    )

    # Write to Neo4j
    neo4j_stats = write_ontology(assignments, hierarchy, associations=associations)

    # Write decomposed record sub-nodes
    decomp_count = 0
    if decomposition:
        decomp_count = write_record_decomposition(decomposition)

    elapsed = time.time() - start

    # Summary
    _flush_print(f"\n{'=' * 70}")
    _flush_print(f"SV-LOI pipeline complete in {elapsed:.1f}s")
    _flush_print(f"  Method:            SV-LOI (Structurally-Verified LLM Ontology Induction)")
    _flush_print(f"  Entities:          {len(entities)}")
    _flush_print(f"  Classes:           {len(final_dist)}")
    _flush_print(f"  SUBCLASS_OF:       {len(hierarchy)} (IS-A)")
    _flush_print(f"  ASSOCIATED_WITH:   {len(associations)} (HAS-A)")
    _flush_print(f"  Flagged/Arbitrated:{total_flagged}")
    _flush_print(f"  Decomposed:        {decomp_count} sub-nodes")
    _flush_print(f"  Distribution:")
    for cls, cnt in final_dist.most_common():
        _flush_print(f"    {cls}: {cnt}")

    # Low-confidence entities
    low_conf = {
        eid: p for eid, p in provenance.items()
        if p.get("confidence", 1.0) < 0.5
    }

    # Save summary
    summary = {
        "mode": "zone3_sv_loi",
        "model": model,
        "suffix": suffix,
        "seed": seed,
        "domain": detected_domain,
        "elapsed_seconds": round(elapsed, 2),
        "entity_count": len(entities),
        "class_vocab_discovered": class_vocab,
        "classes_final": sorted(final_dist.keys()),
        "class_distribution": dict(final_dist),
        "flagged_count": total_flagged,
        "hierarchy": [{"child": c, "parent": p, "type": "SUBCLASS_OF"} for c, p in hierarchy],
        "associations": [{"source": s, "target": t, "type": "ASSOCIATED_WITH"} for s, t in associations],
        "neo4j_stats": neo4j_stats,
        "decomposition_count": decomp_count,
        "ablation": {
            "skip_verify": skip_verify,
            "skip_arbitrate": skip_arbitrate,
            "skip_consolidate": skip_consolidate,
            "skip_sohd": skip_sohd,
        },
        "sohd_stats": sohd_stats,
        "provenance_stats": {
            "total_entities": len(provenance),
            "flagged": sum(1 for p in provenance.values() if p.get("flagged")),
            "arbitrated": sum(1 for p in provenance.values() if p.get("arbitrated")),
            "consolidated": sum(1 for p in provenance.values() if p.get("consolidated_from")),
            "type_changed_by_verification": sum(
                1 for eid, p in provenance.items()
                if p.get("final_type") != p.get("llm_type")
            ),
            "low_confidence_count": len(low_conf),
            "confidence_distribution": {
                "high_1.0": sum(1 for p in provenance.values() if p.get("confidence", 0) == 1.0),
                "good_0.8": sum(1 for p in provenance.values() if p.get("confidence", 0) == 0.8),
                "medium_0.7": sum(1 for p in provenance.values() if p.get("confidence", 0) == 0.7),
                "low_0.5": sum(1 for p in provenance.values() if p.get("confidence", 0) == 0.5),
                "very_low_0.3": sum(1 for p in provenance.values() if p.get("confidence", 0) == 0.3),
            },
        },
        "intrinsic_quality": quality_metrics,
        "backbone": {
            "overall_connectivity": backbone_report.get("overall_connectivity", 0),
            "disconnected_classes": backbone_report.get("disconnected_classes", []),
        },
        "taxonomy_edges": len(taxonomy_llm_edges) + len(taxonomy_subsumption_edges) + len(sohd_edges),
        "taxonomy_edges_detail": {
            "llm_pairwise": len(taxonomy_llm_edges),
            "subsumption": len(taxonomy_subsumption_edges),
            "sohd": len(sohd_edges),
        },
    }

    # Save provenance
    prov_path = os.path.join(rdir, f"svloi_provenance_{suffix}.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    _flush_print(f"  ✓ Decision provenance saved → {prov_path}")

    # Save low-confidence entities
    if low_conf:
        lc_path = os.path.join(rdir, f"zone3_low_confidence_entities_{suffix}.json")
        with open(lc_path, "w") as f:
            json.dump(low_conf, f, indent=2, default=str)
        _flush_print(f"  ✓ {len(low_conf)} low-confidence entities saved → {lc_path}")

    # Print intrinsic quality metrics
    _flush_print(f"  Intrinsic quality:")
    for k, v in quality_metrics.items():
        _flush_print(f"    {k}: {v}")

    out_path = os.path.join(rdir, f"zone3_svloi_summary_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _flush_print(f"\n✓ Summary saved to {out_path}")
    _flush_print(f"\nNext steps:")
    _flush_print(f"  python3 baseline/eval.py --suffix {suffix} --riskine --model {model}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zone 3: SV-LOI — Structurally-Verified LLM Ontology Induction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Novel ontology induction: LLM assigns classes to entities, structural
signatures verify assignments, disagreements are arbitrated by LLM
with enriched context. Produces 10-20 clean ontology classes.

Pre-requisite: Run zone2/pipeline.py first to populate Neo4j with Entity nodes.

Examples:
  python3 zone3/sv_loi.py
  python3 zone3/sv_loi.py --model qwen2.5:72b
  python3 zone3/sv_loi.py --model qwen2.5:72b --suffix zone3_svloi

After running, evaluate with:
  python3 baseline/eval.py --suffix zone3_svloi --riskine
        """,
    )
    parser.add_argument(
        "--model", default=config.OLLAMA_MODEL,
        help=f"Ollama model name (default: {config.OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--suffix", default="zone3_svloi",
        help="Suffix for result files (default: zone3_svloi)"
    )
    parser.add_argument("--skip-verify", action="store_true",
                        help="[ABLATION] Skip Phase 2 structural verification")
    parser.add_argument("--skip-arbitrate", action="store_true",
                        help="[ABLATION] Skip Phase 3 disagreement arbitration")
    parser.add_argument("--skip-consolidate", action="store_true",
                        help="[ABLATION] Skip Phase 4 LLM-guided consolidation")
    parser.add_argument("--skip-record-propagation", action="store_true",
                        help="[ABLATION] Skip Phase 8 record propagation (use Zone 2 types)")
    parser.add_argument("--skip-sohd", action="store_true",
                        help="[ABLATION] Skip SOHD hierarchy deepening")
    parser.add_argument("--use-old-rebalance", action="store_true",
                        help="[ABLATION] Use old rebalance (total entities, 25% threshold)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--results-dir", default=None,
                        help="Results directory (default: config.RESULTS_DIR)")
    args = parser.parse_args()

    run_sv_loi(
        model=args.model,
        suffix=args.suffix,
        skip_verify=args.skip_verify,
        skip_arbitrate=args.skip_arbitrate,
        skip_consolidate=args.skip_consolidate,
        skip_record_propagation=getattr(args, 'skip_record_propagation', False),
        skip_sohd=getattr(args, 'skip_sohd', False),
        use_old_rebalance=getattr(args, 'use_old_rebalance', False),
        seed=args.seed,
        results_dir=args.results_dir,
    )
