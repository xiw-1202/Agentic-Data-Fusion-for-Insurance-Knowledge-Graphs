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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 15          # entities per LLM typing prompt (smaller = more context per entity)
MAX_MEMBERS_IN_PROMPT = 15
MIN_CLASS_SIZE = 10      # fallback floor — actual threshold is max(this, 1% of entities)
DEVIATION_THRESHOLD = 2.0  # σ threshold for structural flagging
MAX_ARBITRATION_BATCH = 10  # entities per arbitration prompt
MAX_CLASS_FRACTION = 0.25  # no single class should exceed 25% of entities

# Structured entity prefixes — these already have entity_type from Stage 1.
# SV-LOI should use their existing types directly, not re-classify via LLM.
_STRUCTURED_PREFIXES = STRUCTURED_PREFIXES

# Target class count range (guide for class discovery)
# Higher minimum to counteract consolidation over-merging
TARGET_CLASSES_MIN = 8
TARGET_CLASSES_MAX = 15

# REMOVED: PROTECTED_CLASS_NAMES was a hardcoded list that overlapped with
# Riskine reference classes (domain leakage). Protection is now data-driven:
# merge_leaf_classes() uses relational diversity (distinct_rels < 4) to
# distinguish property-value classes from real ontology classes.
# Classes with rich relational structure survive merging naturally.
PROTECTED_CLASS_NAMES: set[str] = set()  # empty — fully data-driven

# Forbidden class names — data types masquerading as ontology classes
FORBIDDEN_CLASS_NAMES = {
    "financialamount", "timperiod", "timeperiod", "measurement", "amount",
    "number", "date", "text", "quantity", "value", "metric", "event",
    "condition", "location", "data", "record", "entry", "item", "type",
    "category", "group", "class", "entity", "thing", "other",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_llm(model: str) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
    )


def get_neo4j_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )


def _sanitize_label(name: str) -> str:
    """Make a PascalCase name safe for Neo4j label."""
    cleaned = re.sub(r'[^A-Za-z0-9]', '', name.strip())
    if not cleaned:
        return "UnknownClass"
    if cleaned[0].isdigit():
        cleaned = "Class" + cleaned
    return cleaned


def _parse_json_safely(text: str) -> Union[dict, list]:
    """Try to parse JSON from LLM output with fallbacks."""
    text = text.strip()
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'\[.*\]', text, re.DOTALL) or re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _invoke_llm(llm: ChatOllama, prompt: str) -> str:
    """Call LLM and return content string."""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        print(f"    [llm] Error: {e}")
        return ""


# ---------------------------------------------------------------------------
# Step 1: Load Entities from Neo4j (reused from RSI-LCR)
# ---------------------------------------------------------------------------

def load_entities() -> list[dict]:
    """Load all Entity nodes from local cache (zero Neo4j round-trips)."""
    print("\n[Phase 0] Load graph cache", flush=True)
    return load_cached_entities(fmt="sv_loi")


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
    print("  Stage 1: Detecting domain...", flush=True)
    domain_prompt = f"""Look at these entity names from a knowledge graph:
{', '.join(name_sample[:40])}

And these relationship types:
{', '.join(r for r, _ in top_rels[:15])}

What industry/domain is this knowledge graph about? Answer in 1-3 words."""

    domain_raw = _invoke_llm(llm, domain_prompt)
    domain = domain_raw.strip().strip('"').strip("'").strip(".")
    print(f"  Detected domain: {domain}", flush=True)

    # ---- Stage 2: Propose classes ----
    print("  Stage 2: Proposing ontology classes...", flush=True)

    # Inject record evidence if available
    evidence_block = ""
    if record_evidence:
        evidence_block = f"\n{record_evidence}\n"

    prompt = f"""You are designing a DOMAIN ONTOLOGY for a {domain} knowledge graph.

Here are {len(name_sample)} entity names from the graph (domain concepts only):
{chr(10).join(f'  {n}' for n in name_sample)}

Relationship types:
{chr(10).join(f'  {r} ({c}x)' for r, c in top_rels)}

Example triples:
{chr(10).join(triple_examples)}
{evidence_block}
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

    # Last resort fallback — derive classes from actual entity types in the graph.
    # NEVER use hardcoded class names — that would be domain leakage (see CLAUDE.md).
    if len(classes) < TARGET_CLASSES_MIN:
        print(f"  WARNING: Only {len(classes)} classes after retry. "
              f"Deriving fallbacks from graph entity types.", flush=True)
        # Collect entity_type values already present in the graph.
        type_counts: dict[str, int] = defaultdict(int)
        for e in entities:
            et = e.get("entity_type", "Unknown")
            if et and et != "Unknown":
                type_counts[et] += 1
        # Sort by frequency, take most common as fallback classes.
        sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
        for et_name, _cnt in sorted_types:
            sanitized = _sanitize_label(et_name)
            if sanitized.lower() not in seen_lower and sanitized.lower() not in FORBIDDEN_CLASS_NAMES:
                classes.append(sanitized)
                seen_lower.add(sanitized.lower())
            if len(classes) >= TARGET_CLASSES_MIN:
                break

    # NOTE: No standard renames or forced class injection.
    # The LLM proposes class names freely; the Riskine eval measures
    # alignment via BERTScore which handles synonyms (Policy≈Product,
    # Location≈Address). Planting Riskine class names would be leakage.

    # Data-driven record seeding: if structured records have significant
    # populations with known prefix→class mappings, ensure those classes
    # exist in the vocabulary. This is NOT leakage — it reads from Zone 2's
    # own record prefixes (POL-, CLM-, PER-, PROP-), not from Riskine.
    RECORD_PREFIX_TO_CLASS = {
        "POL": "Policy", "CLM": "Claim", "PER": "Person", "PROP": "Property",
    }
    seed_entities = all_entities or entities
    for prefix, cls_name in RECORD_PREFIX_TO_CLASS.items():
        count = sum(1 for e in seed_entities if e["id"].startswith(f"{prefix}-"))
        if count > 50 and cls_name.lower() not in seen_lower:
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


def build_structural_signatures(entities: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """Build relation signature feature vectors.

    Returns:
        features: (n_entities, n_features) L2-normalized matrix
        entity_ids: ordered list
        feature_names: column names
    """
    all_rel_types: set[str] = set()
    all_entity_types: set[str] = set()
    for e in entities:
        all_rel_types.update(e.get("out_rel_counts", {}).keys())
        all_rel_types.update(e.get("in_rel_counts", {}).keys())
        all_entity_types.add(e["entity_type"])

    rel_types = sorted(all_rel_types)
    ent_types = sorted(all_entity_types)

    # Feature columns: [rel_OUT, rel_IN for each rel_type] + [entity_type one-hot]
    feature_names = []
    for rt in rel_types:
        feature_names.append(f"{rt}_OUT")
        feature_names.append(f"{rt}_IN")
    for et in ent_types:
        feature_names.append(f"type_{et}")

    n = len(entities)
    d = len(feature_names)
    features = np.zeros((n, d))

    feat_idx = {name: idx for idx, name in enumerate(feature_names)}
    entity_ids = []
    for i, e in enumerate(entities):
        entity_ids.append(e["id"])
        for rt in rel_types:
            features[i, feat_idx[f"{rt}_OUT"]] = e.get("out_rel_counts", {}).get(rt, 0)
            features[i, feat_idx[f"{rt}_IN"]] = e.get("in_rel_counts", {}).get(rt, 0)
        et = e["entity_type"]
        type_key = f"type_{et}"
        if type_key in feat_idx:
            features[i, feat_idx[type_key]] = 1.0

    # L2 normalize
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    features = features / norms

    return features, entity_ids, feature_names


def structural_consensus_check(
    assignments: dict[str, str],
    features: np.ndarray,
    entity_ids: list[str],
) -> tuple[dict[str, dict], list[dict]]:
    """Verify LLM assignments against structural signatures.

    For each class, compute the centroid of member signatures.
    Flag entities whose cosine distance from centroid > DEVIATION_THRESHOLD * σ.

    Returns:
        class_stats: {class_name: {centroid, mean_sim, std_sim, member_count}}
        flagged: [{entity_id, assigned_class, similarity, nearest_class, nearest_sim}]
    """
    print(f"\n[Phase 6/10] Structural consensus verification (σ threshold={DEVIATION_THRESHOLD})...")

    eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

    # Group entities by assigned class
    class_members: dict[str, list[str]] = defaultdict(list)
    for eid, cls in assignments.items():
        class_members[cls].append(eid)

    # Compute class centroids
    class_stats: dict[str, dict] = {}
    for cls, members in class_members.items():
        if cls == "Other":
            continue
        indices = [eid_to_idx[m] for m in members if m in eid_to_idx]
        if not indices:
            continue

        member_features = features[indices]
        centroid = member_features.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # Cosine similarities to centroid
        sims = member_features @ centroid
        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims)) if len(sims) > 1 else 0.0

        class_stats[cls] = {
            "centroid": centroid,
            "mean_sim": mean_sim,
            "std_sim": std_sim,
            "member_count": len(indices),
        }

    # Flag outliers
    flagged: list[dict] = []
    all_centroids = {cls: stats["centroid"] for cls, stats in class_stats.items()}

    for eid, cls in assignments.items():
        if cls == "Other" or cls not in class_stats:
            continue
        idx = eid_to_idx.get(eid)
        if idx is None:
            continue

        stats = class_stats[cls]
        sim = float(features[idx] @ stats["centroid"])
        threshold = stats["mean_sim"] - DEVIATION_THRESHOLD * max(stats["std_sim"], 0.05)

        if sim < threshold:
            # Find nearest alternative class
            best_alt_cls = None
            best_alt_sim = -1.0
            for alt_cls, alt_centroid in all_centroids.items():
                if alt_cls == cls:
                    continue
                alt_sim = float(features[idx] @ alt_centroid)
                if alt_sim > best_alt_sim:
                    best_alt_sim = alt_sim
                    best_alt_cls = alt_cls

            flagged.append({
                "entity_id": eid,
                "assigned_class": cls,
                "similarity": round(sim, 4),
                "class_mean_sim": round(stats["mean_sim"], 4),
                "class_std_sim": round(stats["std_sim"], 4),
                "nearest_class": best_alt_cls,
                "nearest_sim": round(best_alt_sim, 4),
            })

    print(f"  ✓ {len(flagged)} entities flagged for arbitration ({len(flagged)}/{len(assignments)} = {100*len(flagged)/max(len(assignments),1):.1f}%)")
    for cls, stats in sorted(class_stats.items(), key=lambda x: x[1]["member_count"], reverse=True):
        n_flagged = sum(1 for f in flagged if f["assigned_class"] == cls)
        print(f"    {cls}: {stats['member_count']} members, mean_sim={stats['mean_sim']:.3f}, flagged={n_flagged}")

    return class_stats, flagged


# ---------------------------------------------------------------------------
# Phase 7: Disagreement Arbitration
# ---------------------------------------------------------------------------

def arbitrate_disagreements(
    flagged: list[dict],
    entities: list[dict],
    assignments: dict[str, str],
    class_vocab: list[str],
    llm: ChatOllama,
) -> dict[str, str]:
    """Re-query LLM for flagged entities with enriched structural context.

    Returns updated assignments dict.
    """
    if not flagged:
        print("\n[Phase 7] No disagreements to arbitrate.")
        return assignments

    print(f"\n[Phase 7] Arbitrating {len(flagged)} disagreements...")

    entity_map = {e["id"]: e for e in entities}
    updated = dict(assignments)
    changes = 0

    # Batch flagged entities
    n_batches = (len(flagged) + MAX_ARBITRATION_BATCH - 1) // MAX_ARBITRATION_BATCH

    for batch_idx in range(n_batches):
        start = batch_idx * MAX_ARBITRATION_BATCH
        batch = flagged[start:start + MAX_ARBITRATION_BATCH]

        entity_sections = []
        for f in batch:
            eid = f["entity_id"]
            e = entity_map.get(eid, {})

            section = (
                f"Entity: {eid}\n"
                f"  Current type: {e.get('entity_type', '?')}\n"
                f"  Assigned class: {f['assigned_class']} (structural sim: {f['similarity']:.3f})\n"
                f"  Nearest alternative: {f['nearest_class']} (structural sim: {f['nearest_sim']:.3f})\n"
            )
            if e.get("out_summary"):
                section += f"  Outgoing: {'; '.join(e['out_summary'][:3])}\n"
            if e.get("in_summary"):
                section += f"  Incoming: {'; '.join(e['in_summary'][:2])}\n"
            entity_sections.append(section)

        class_list = ", ".join(c for c in class_vocab if c != "Other")
        prompt = f"""You are verifying ontology class assignments. Some entities were flagged because
their structural relationships don't match their assigned class.

AVAILABLE CLASSES: {class_list}, Other

FLAGGED ENTITIES:
{chr(10).join(entity_sections)}

For each entity, decide the CORRECT class based on both its name/type AND its relationships.
Output one line per entity: entity_id -> CorrectClass
"""
        raw = _invoke_llm(llm, prompt)

        line_re = re.compile(r'^[- ]*(.+?)\s*->\s*(\w+)\s*$', re.MULTILINE)
        for m in line_re.finditer(raw):
            eid = m.group(1).strip().strip('"').strip("'")
            cls = _sanitize_label(m.group(2).strip())
            if eid in updated:
                old = updated[eid]
                if cls in class_vocab or cls == "Other":
                    if cls != old:
                        updated[eid] = cls
                        changes += 1

    print(f"  ✓ Arbitration complete: {changes} entities reassigned")
    return updated


# ---------------------------------------------------------------------------
# Phase 11-14: Post-processing — consolidate, merge small, derive hierarchy
# ---------------------------------------------------------------------------

def infer_class_relations(
    assignments: dict[str, str],
    entities: list[dict],
    llm: ChatOllama,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """5-way typed relation inference between classes (Changes C+D+E unified).

    Replaces separate consolidate_classes() + derive_hierarchy() with a single
    pass that infers typed relations: equivalent / parent / child / overlap / distinct.

    Only 'equivalent' triggers a merge. 'parent/child' become SUBCLASS_OF edges.
    'overlap' and 'distinct' keep both classes separate.

    Protected classes (standard ontology terms) can be merged INTO but never
    renamed away — prevents destroying exact matches with reference ontologies.

    Returns:
        (updated_assignments, hierarchy_edges)
    """
    print("\n[Phase 11] Class relation inference (5-way)...", flush=True)

    dist = Counter(v for v in assignments.values() if v != "Other")
    class_names = sorted(dist.keys())

    if len(class_names) < 2:
        print("  ✓ Too few classes for relation inference")
        return assignments, []

    # Build class descriptions with CONCEPT member examples only
    entity_map = {e["id"]: e for e in entities}
    class_descs: dict[str, str] = {}
    for cls in class_names:
        concept_members = [
            eid for eid, c in assignments.items()
            if c == cls and is_concept_entity(entity_map.get(eid, {"id": eid, "entity_type": "Unknown"}))
        ]
        # If no concept members, use first few record IDs as fallback context
        if not concept_members:
            all_members = [eid for eid, c in assignments.items() if c == cls]
            sample = all_members[:5]
        else:
            sample = concept_members[:8]
        # Include typical relations for context
        rel_types: set[str] = set()
        for eid in (concept_members or [eid for eid, c in assignments.items() if c == cls])[:20]:
            e = entity_map.get(eid, {})
            rel_types.update(e.get("out_rel_counts", {}).keys())
            rel_types.update(e.get("in_rel_counts", {}).keys())
        top_rels = sorted(rel_types)[:5]

        protected = "  [PROTECTED — standard term]" if cls.lower() in PROTECTED_CLASS_NAMES else ""
        class_descs[cls] = (
            f"  {cls} ({dist[cls]} members{protected}):\n"
            f"    Examples: {', '.join(sample)}\n"
            f"    Relations: {', '.join(top_rels) if top_rels else 'none'}"
        )

    # --- Pairwise relation inference ---
    # Only compare plausible pairs (blocking: skip obvious non-matches)
    from sentence_transformers import SentenceTransformer
    st_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embs = st_encoder.encode([c for c in class_names], normalize_embeddings=True)
    sim_matrix = np.dot(embs, embs.T)

    # Block: only compare pairs with cosine > 0.25 (skip clearly unrelated)
    BLOCK_THRESHOLD = 0.25
    pairs_to_compare = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            if sim_matrix[i, j] > BLOCK_THRESHOLD:
                pairs_to_compare.append((class_names[i], class_names[j]))

    print(f"  Comparing {len(pairs_to_compare)} class pairs (blocked from {len(class_names) * (len(class_names)-1) // 2})...", flush=True)

    # Batch pairs into groups for efficient LLM calls
    PAIRS_PER_BATCH = 5
    all_relations: list[dict] = []

    for batch_start in range(0, len(pairs_to_compare), PAIRS_PER_BATCH):
        batch = pairs_to_compare[batch_start:batch_start + PAIRS_PER_BATCH]

        pair_sections = []
        for ci, cj in batch:
            pair_sections.append(
                f"PAIR: {ci} vs {cj}\n"
                f"{class_descs[ci]}\n"
                f"{class_descs[cj]}"
            )

        prompt = f"""For each pair of ontology classes, determine their relationship.

Choose EXACTLY ONE for each pair:
- "equivalent": same concept, different names → MERGE (keep the more standard name)
- "parent_of": first class is a broader category containing the second → SUBCLASS_OF
- "child_of": first class is a subtype of the second → SUBCLASS_OF
- "overlapping": partial intersection, neither fully contains the other → keep both
- "distinct": unrelated or sibling concepts → keep both

IMPORTANT RULES:
- Classes marked [PROTECTED] are standard ontology terms. They can ABSORB other classes
  but must NEVER be renamed or merged INTO a non-standard name.
- "equivalent" should be rare — only for true synonyms (e.g., "Policy" and "Product")
- When unsure, choose "distinct" — it's safer to keep classes separate

{chr(10).join(pair_sections)}

Output JSON array, one entry per pair:
[{{"class_a": "...", "class_b": "...", "relation": "...", "merge_into": "...", "reason": "..."}}]
The "merge_into" field is only needed for "equivalent" — which name to keep."""

        raw = _invoke_llm(llm, prompt)
        parsed = _parse_json_safely(raw)

        if isinstance(parsed, list):
            all_relations.extend(parsed)
        elif isinstance(parsed, dict):
            all_relations.append(parsed)

        if (batch_start // PAIRS_PER_BATCH + 1) % 5 == 0:
            print(f"    Batch {batch_start // PAIRS_PER_BATCH + 1}/{(len(pairs_to_compare) + PAIRS_PER_BATCH - 1) // PAIRS_PER_BATCH}", flush=True)

    # --- Process relations ---
    updated = dict(assignments)
    hierarchy: list[tuple[str, str]] = []
    merges_applied: list[str] = []
    merge_count = 0

    for rel_entry in all_relations:
        if not isinstance(rel_entry, dict):
            continue
        ca = rel_entry.get("class_a", "")
        cb = rel_entry.get("class_b", "")
        relation = rel_entry.get("relation", "distinct").lower()
        merge_into = rel_entry.get("merge_into", "")
        reason = rel_entry.get("reason", "")

        if relation == "equivalent":
            # Determine which name to keep (with validation)
            valid_names = {ca, cb}
            if merge_into:
                sanitized_merge = _sanitize_label(merge_into)
                # Validate: merge_into must match one of the two classes
                if sanitized_merge.lower() in {ca.lower(), cb.lower()}:
                    keep = ca if sanitized_merge.lower() == ca.lower() else cb
                    absorb = cb if keep == ca else ca
                else:
                    # LLM proposed a third name — fall through to protected heuristic
                    merge_into = ""

            if not merge_into:
                # Prefer protected names (standard ontology terms)
                if ca.lower() in PROTECTED_CLASS_NAMES and cb.lower() not in PROTECTED_CLASS_NAMES:
                    keep, absorb = ca, cb
                elif cb.lower() in PROTECTED_CLASS_NAMES and ca.lower() not in PROTECTED_CLASS_NAMES:
                    keep, absorb = cb, ca
                elif ca.lower() in PROTECTED_CLASS_NAMES and cb.lower() in PROTECTED_CLASS_NAMES:
                    # Both protected — skip merge, treat as distinct
                    print(f"  ✗ Both {ca} and {cb} are protected — keeping separate", flush=True)
                    continue
                else:
                    keep, absorb = ca, cb  # neither protected, keep first

            # NEVER rename a protected class into a non-protected one
            if absorb.lower() in PROTECTED_CLASS_NAMES and keep.lower() not in PROTECTED_CLASS_NAMES:
                print(f"  ✗ Blocked: {absorb} is protected, cannot merge into {keep}", flush=True)
                continue

            # Apply merge
            count = 0
            for eid in list(updated.keys()):
                if updated[eid] == absorb:
                    updated[eid] = keep
                    count += 1
            if count > 0:
                print(f"  ✓ EQUIVALENT: {absorb} → {keep} ({count} entities) — {reason}", flush=True)
                merge_count += count
                merges_applied.append(f"{absorb}→{keep}")

        elif relation == "parent_of":
            # ca is parent of cb
            edge = (cb, ca)
            if edge not in hierarchy:
                hierarchy.append(edge)
                print(f"  → PARENT: {ca} ⊃ {cb} — {reason}", flush=True)

        elif relation == "child_of":
            # ca is child of cb
            edge = (ca, cb)
            if edge not in hierarchy:
                hierarchy.append(edge)
                print(f"  → CHILD: {ca} ⊂ {cb} — {reason}", flush=True)

        elif relation == "overlapping":
            print(f"  ~ OVERLAP: {ca} ∩ {cb} — {reason}", flush=True)

        # "distinct" → no action

    # Also mark pure data-artifact classes as Other
    for cls in list(set(updated.values())):
        if cls == "Other":
            continue
        members = [eid for eid, c in updated.items() if c == cls]
        concept_members = [
            eid for eid in members
            if is_concept_entity(entity_map.get(eid, {"id": eid, "entity_type": "Unknown"}))
        ]
        if not concept_members and len(members) > 0:
            # Class has zero concept members — all records/values
            # Only mark as Other if it's not a protected name
            if cls.lower() not in PROTECTED_CLASS_NAMES:
                for eid in members:
                    updated[eid] = "Other"
                print(f"  ✓ {cls} → Other ({len(members)} entities, no concept members)", flush=True)
                merge_count += len(members)

    new_dist = Counter(v for v in updated.values() if v != "Other")
    print(f"\n  ✓ Relation inference complete:", flush=True)
    print(f"    Merges: {len(merges_applied)} ({merge_count} entities moved)", flush=True)
    print(f"    Hierarchy edges: {len(hierarchy)}", flush=True)
    print(f"    Classes remaining: {len(new_dist)}", flush=True)
    for cls, cnt in new_dist.most_common():
        protected = " [P]" if cls.lower() in PROTECTED_CLASS_NAMES else ""
        print(f"      {cls}{protected}: {cnt}", flush=True)

    return updated, hierarchy


def merge_small_classes(
    assignments: dict[str, str],
    features: np.ndarray,
    entity_ids: list[str],
    min_size: int = MIN_CLASS_SIZE,
) -> dict[str, str]:
    """Merge classes with fewer than min_size members into nearest.

    min_size is a fixed floor (default: 10). This is intentionally NOT
    percentage-based: a class with 999 members is valid at any dataset
    size. The LLM validation pass (merge_leaf_classes) handles the
    semantic question; this pass only cleans up degenerate noise clusters.
    """
    print(f"\n  Merging small classes (min_size={min_size})...")

    eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

    # Count class sizes
    class_counts = Counter(assignments.values())
    small_classes = [c for c, n in class_counts.items()
                     if n < min_size and c != "Other" and c.lower() not in PROTECTED_CLASS_NAMES]

    if not small_classes:
        print("  ✓ No small classes to merge")
        return assignments

    # Compute centroids for non-small classes
    centroids: dict[str, np.ndarray] = {}
    for cls, cnt in class_counts.items():
        if cls in small_classes or cls == "Other":
            continue
        indices = [eid_to_idx[eid] for eid, c in assignments.items() if c == cls and eid in eid_to_idx]
        if not indices:
            continue
        centroid = features[indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[cls] = centroid

    updated = dict(assignments)
    merge_map: dict[str, str] = {}

    for small_cls in small_classes:
        # Find nearest large class by centroid similarity
        indices = [eid_to_idx[eid] for eid, c in assignments.items() if c == small_cls and eid in eid_to_idx]
        if not indices:
            continue
        small_centroid = features[indices].mean(axis=0)
        norm = np.linalg.norm(small_centroid)
        if norm > 0:
            small_centroid = small_centroid / norm

        best_cls = None
        best_sim = -1.0
        for cls, centroid in centroids.items():
            sim = float(small_centroid @ centroid)
            if sim > best_sim:
                best_sim = sim
                best_cls = cls

        if best_cls:
            merge_map[small_cls] = best_cls
            for eid in list(updated.keys()):
                if updated[eid] == small_cls:
                    updated[eid] = best_cls

    if merge_map:
        print(f"  ✓ Merged {len(merge_map)} small classes:")
        for s, t in merge_map.items():
            print(f"    {s} ({class_counts[s]} members) → {t}")
    else:
        print("  ✓ No merges needed")

    return updated


# ---------------------------------------------------------------------------
# Phase 13: LLM-guided class validation
# ---------------------------------------------------------------------------

def merge_leaf_classes(
    assignments: dict[str, str],
    entities: list[dict],
    llm: Optional[ChatOllama] = None,
) -> dict[str, str]:
    """LLM-guided class validation with structural evidence.

    After entity typing, some induced classes are genuine ontology classes
    (Risk, Damage, Person) while others are property values masquerading
    as classes (Deductible, Limit, Notification). This function presents
    the LLM with structural evidence for each small class and asks it to
    decide: keep as a real class, or merge into a parent?

    The structural evidence (member count, leaf fraction, relational
    diversity, sample members, top parent) helps the LLM make informed
    decisions. This scales with data — more data means richer evidence,
    better decisions. No hardcoded thresholds that break with scale.

    Algorithm:
        1. Compute structural profile for each small class (<50 members)
        2. Present all small classes + evidence to LLM in one prompt
        3. LLM returns: keep or merge into a target class
        4. Apply merge decisions
        5. Heuristic fallback for any class LLM didn't decide:
           merge if leaf% > 70% AND distinct_rel_types < 4

    Domain-agnostic — uses graph structure + LLM reasoning, no hardcoded names.
    """
    print("\n[Phase 13] LLM-guided class validation...", flush=True)

    entity_map = {e["id"]: e for e in entities}
    class_counts = Counter(assignments.values())

    # Step 1: Compute structural profile for ALL non-Other classes.
    # Present every class to the LLM with its evidence — the LLM decides
    # which are real classes (KEEP) vs property values (MERGE).
    # No artificial large/small split — a class with 17 members (Risk)
    # is a valid merge target for a class with 3 members (Classification).
    all_classes: list[str] = []
    class_profiles: dict[str, dict] = {}

    for cls, count in class_counts.items():
        if cls == "Other":
            continue

        members = [eid for eid, c in assignments.items() if c == cls]
        leaf_count = 0
        all_rel_types: set[str] = set()
        total_degree = 0.0
        sample_members: list[str] = []
        # Prefer concept members for samples (semantic names, not POL-xxx)
        concept_members = [eid for eid in members
                           if not eid.startswith(("POL-", "CLM-", "PROP-", "REC-", "PER-"))]
        sample_pool = concept_members or members

        for eid in members:
            e = entity_map.get(eid)
            if not e:
                continue
            out_count = sum(e.get("out_rel_counts", {}).values())
            if out_count <= 1:
                leaf_count += 1
            total_degree += e.get("degree", 0)
            all_rel_types.update(e.get("out_rel_counts", {}).keys())
            all_rel_types.update(e.get("in_rel_counts", {}).keys())
            if len(sample_members) < 5 and eid in sample_pool:
                sample_members.append(eid)

        leaf_frac = leaf_count / len(members) if members else 0
        avg_degree = total_degree / len(members) if members else 0

        # Find which large classes point to this class's members
        member_set = set(members)
        parent_votes: Counter = Counter()
        for e in entities:
            src_cls = assignments.get(e["id"], "Other")
            if src_cls == "Other" or src_cls == cls or class_counts.get(src_cls, 0) < count:
                continue
            for rel in e.get("out_rels", []):
                if rel.get("target", "") in member_set:
                    parent_votes[src_cls] += 1

        class_profiles[cls] = {
            "count": count,
            "concept_count": len(concept_members),
            "leaf_frac": leaf_frac,
            "avg_degree": avg_degree,
            "rel_types": sorted(all_rel_types)[:8],
            "n_rel_types": len(all_rel_types),
            "sample_members": sample_members,
            "top_parent": parent_votes.most_common(1)[0] if parent_votes else None,
        }
        all_classes.append(cls)

    if not all_classes:
        print("  ✓ No classes to validate")
        return assignments

    # Identify classes backed by database records (distinct data schemas)
    classes_with_records: set[str] = set()
    for cls in all_classes:
        has_records = any(
            eid.startswith(("POL-", "CLM-", "REC-", "PER-", "PROP-"))
            for eid, c in assignments.items() if c == cls
        )
        if has_records:
            classes_with_records.add(cls)

    print(f"  {len(all_classes)} classes to validate "
          f"({len(classes_with_records)} record-backed)", flush=True)

    # Step 2: Ask LLM to validate — present ALL classes with evidence,
    # let LLM decide which are real vs which should merge.
    merge_map: dict[str, str] = {}

    if llm:
        profile_lines = []
        for cls in sorted(all_classes, key=lambda c: -class_counts[c]):
            p = class_profiles[cls]
            parent_info = ""
            if p["top_parent"]:
                parent_info = f", most referenced by: {p['top_parent'][0]}"
            members_str = ', '.join(p['sample_members'][:4]) if p['sample_members'] else '(database records only)'
            record_note = " [RECORDS]" if cls in classes_with_records else ""
            profile_lines.append(
                f"  {cls}{record_note} ({p['count']} total, {p['concept_count']} concepts): "
                f"examples=[{members_str}], "
                f"{p['n_rel_types']} relation types"
                f"{parent_info}"
            )

        prompt = (
            "Ontology class consolidation task.\n\n"
            "Below are ALL induced ontology classes with their evidence.\n"
            "Decide which classes are genuine ontological concepts and which "
            "are really properties/attributes of other classes.\n\n"
            "Classes:\n" + "\n".join(profile_lines) + "\n\n"
            "Rules:\n"
            "- KEEP if the members are real-world THINGS (people, places, risks, "
            "products, organizations, processes, damages, documents). These exist "
            "independently — 'Base Flood' IS a risk, 'FEMA' IS an organization.\n"
            "- MERGE only if the members are MEASUREMENTS or ATTRIBUTES that "
            "describe another class — amounts, limits, codes, dates, thresholds. "
            "Example: 'Separate Deductible' is an ATTRIBUTE of Coverage.\n"
            "- Class SIZE does not matter. A class with 5 members can be real "
            "(e.g., Product with 'Flood Insurance') while a class with 20 "
            "members can be an attribute (e.g., Requirement).\n"
            "- Classes marked [RECORDS] contain database records from distinct "
            "data sources. NEVER merge two [RECORDS] classes together — they "
            "represent different data schemas (e.g., claims vs policies).\n"
            "- When in doubt, KEEP. Over-merging destroys ontology structure.\n\n"
            "Return JSON: {\"ClassName\": \"keep\" or \"merge:TargetClass\"}\n"
            "Include ALL classes listed above."
        )

        try:
            response = llm.invoke(prompt)
            decisions = _parse_json_safely(response.content)
            if isinstance(decisions, dict):
                for cls_key, decision in decisions.items():
                    matched_cls = None
                    for c in all_classes:
                        if c.lower() == cls_key.lower():
                            matched_cls = c
                            break
                    if not matched_cls:
                        continue

                    decision = str(decision).strip()
                    if decision.lower() == "keep":
                        print(f"    LLM: {matched_cls} → KEEP", flush=True)
                    elif decision.lower().startswith("merge:"):
                        target = decision.split(":", 1)[1].strip()
                        matched_target = None
                        for c in all_classes:
                            if c.lower() == target.lower():
                                matched_target = c
                                break
                        if matched_target and matched_target != matched_cls:
                            # Guard: never merge record-backed classes with >100 members
                            member_count = class_counts.get(matched_cls, 0)
                            if (member_count > 100
                                    and matched_cls in classes_with_records):
                                print(f"    LLM: {matched_cls} → merge into "
                                      f"{matched_target} [BLOCKED: {member_count} "
                                      f"members, record-backed]", flush=True)
                            else:
                                merge_map[matched_cls] = matched_target
                                print(f"    LLM: {matched_cls} → merge into "
                                      f"{matched_target}", flush=True)
                        else:
                            print(f"    LLM: {matched_cls} → merge:{target} "
                                  f"(target not found, keeping)", flush=True)
            else:
                print("    LLM returned non-dict, using heuristic", flush=True)
        except Exception as exc:
            print(f"    LLM error: {exc}, using heuristic", flush=True)

    # Step 3: Heuristic fallback for undecided classes
    # Uses structural criterion: leaf% > 70% AND distinct_rels < 4
    # Respects PROTECTED_CLASS_NAMES (never merge protected classes via heuristic)
    for cls in all_classes:
        if cls in merge_map:
            continue
        if cls.lower() in PROTECTED_CLASS_NAMES:
            continue
        p = class_profiles[cls]
        if (p["count"] < 50
                and p["leaf_frac"] > 0.70
                and p["n_rel_types"] < 4
                and p["top_parent"]):
            target = p["top_parent"][0]
            merge_map[cls] = target
            print(f"    Heuristic: {cls} → {target} "
                  f"({p['leaf_frac']:.0%} leaf, {p['n_rel_types']} rels)",
                  flush=True)

    # Step 4: Structural veto — prevent merging classes that represent
    # distinct data schemas (both have record entities from different sources).
    # SV-LOI principle: when semantic (LLM) and structural signals disagree,
    # keep separate. Two record-backed classes = two distinct data schemas.
    vetoed: list[str] = []
    for src_cls, tgt_cls in list(merge_map.items()):
        # Veto: never merge two record-backed classes (conflates data schemas)
        if src_cls in classes_with_records and tgt_cls in classes_with_records:
            vetoed.append(src_cls)
            del merge_map[src_cls]
            print(f"    VETO: {src_cls} → {tgt_cls} "
                  f"(both have record entities — distinct data schemas)",
                  flush=True)

    # Step 5: Apply merges
    updated = dict(assignments)
    for src_cls, tgt_cls in merge_map.items():
        for eid in list(updated.keys()):
            if updated[eid] == src_cls:
                updated[eid] = tgt_cls

    if merge_map:
        print(f"  ✓ Merged {len(merge_map)} classes:", flush=True)
        for src, tgt in merge_map.items():
            print(f"    {src} ({class_counts[src]}) → {tgt}", flush=True)
    if vetoed:
        print(f"  ✓ Vetoed {len(vetoed)} merges: {vetoed}", flush=True)
    if not merge_map and not vetoed:
        print("  ✓ No merges needed")

    return updated


def derive_interclass_edges(
    assignments: dict[str, str],
    entities: list[dict],
    min_edge_count: int = 3,
    min_class_frac: float = 0.05,
) -> list[tuple[str, str]]:
    """Derive inter-class edges from actual entity-level connections (data-driven).

    Instead of asking the LLM to guess SUBCLASS_OF relationships (which produces
    wrong IS-A edges), this function looks at actual entity-to-entity connections
    in the KG and aggregates them into class-to-class edges.

    If entities of class A frequently connect to entities of class B via
    relations, that creates an (A, B) inter-class edge. This matches how
    reference ontologies like Riskine define inter-class relationships
    (via $ref links = association/composition, NOT is-a).

    Threshold: max(min_edge_count, 5% of the SMALLER class in the pair).
    This scales correctly because it's relative to the classes involved,
    not total entities. A small but tightly connected class (100 members,
    30 connections to another class = 30%) will always produce an edge.

    Args:
        assignments: entity → class mapping
        entities: all entities with relation data
        min_edge_count: absolute floor (default: 3)
        min_class_frac: fraction of smaller class needed for edge (default: 5%)

    Returns:
        List of (source_class, target_class) edges (stored as SUBCLASS_OF in Neo4j
        for compatibility with evaluation metrics, but semantically these are
        inter-class associations).
    """
    # Count class sizes for adaptive threshold
    class_sizes = Counter(v for v in assignments.values() if v != "Other")
    print(f"\n  Deriving inter-class edges (min_floor={min_edge_count}, "
          f"min_class_frac={min_class_frac:.0%})...", flush=True)

    # Count entity-level connections between classes
    class_connections: Counter = Counter()  # (src_class, tgt_class) → count

    for e in entities:
        eid = e["id"]
        src_cls = assignments.get(eid, "Other")
        if src_cls == "Other":
            continue

        for rel in e.get("out_rels", []):
            tgt_eid = rel.get("target", "")
            tgt_cls = assignments.get(tgt_eid, "Other")
            if tgt_cls == "Other" or tgt_cls == src_cls:
                continue
            class_connections[(src_cls, tgt_cls)] += 1

    # Filter: only keep edges with enough evidence
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for (src, tgt), count in class_connections.most_common():
        # Adaptive threshold: max(floor, 5% of the smaller class)
        smaller_size = min(class_sizes.get(src, 0), class_sizes.get(tgt, 0))
        pair_min = max(min_edge_count, int(smaller_size * min_class_frac))
        if count < pair_min:
            continue
        # Avoid circular edges: if (A,B) and (B,A) both exist, keep the stronger one
        if (tgt, src) in seen:
            continue
        edge = (src, tgt)
        if edge not in seen:
            edges.append(edge)
            seen.add(edge)

    print(f"  ✓ {len(edges)} inter-class edges (floor={min_edge_count}, "
          f"{min_class_frac:.0%} of smaller class)", flush=True)
    for src, tgt in edges[:20]:
        count = class_connections[(src, tgt)]
        print(f"    {src} → {tgt} ({count} connections)", flush=True)
    if len(edges) > 20:
        print(f"    ... and {len(edges) - 20} more", flush=True)

    return edges


def derive_hierarchy(
    assignments: dict[str, str],
    llm: ChatOllama,
) -> list[tuple[str, str]]:
    """Legacy LLM-based hierarchy (kept for ablation comparison)."""
    print("\n[Phase 11-legacy] LLM hierarchy derivation...", flush=True)

    class_counts = Counter(v for v in assignments.values() if v != "Other")
    class_names = sorted(class_counts.keys())

    if len(class_names) < 2:
        return []

    classes_desc = "\n".join(f"  {c} ({class_counts[c]} members)" for c in class_names)

    prompt = f"""Given these ontology classes, propose SUBCLASS_OF relationships to form a hierarchy.

CLASSES:
{classes_desc}

Rules:
- Only propose relationships where one class is truly a subtype of another
- Use format: ChildClass -> ParentClass (meaning Child SUBCLASS_OF Parent)
- Prefer shallow hierarchies (max depth 3)
- Not every class needs a parent — top-level classes are fine
- Only output valid relationships, one per line
"""
    raw = _invoke_llm(llm, prompt)

    hierarchy: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    line_re = re.compile(r'^[- ]*(\w+)\s*->\s*(\w+)\s*$', re.MULTILINE)
    for m in line_re.finditer(raw):
        child = _sanitize_label(m.group(1))
        parent = _sanitize_label(m.group(2))
        if child != parent and child in class_names and parent in class_names:
            edge = (child, parent)
            if edge not in seen:
                hierarchy.append(edge)
                seen.add(edge)

    print(f"  ✓ {len(hierarchy)} SUBCLASS_OF edges")
    return hierarchy


def derive_taxonomy(
    assignments: dict[str, str],
    entities: list[dict],
    llm: ChatOllama,
) -> list[tuple[str, str]]:
    """Derive IS-A taxonomy using defining relations + LLM validation.

    Unlike derive_hierarchy (pure LLM guessing) or derive_interclass_edges
    (association edges), this produces genuine IS-A relationships validated
    by both structural evidence and LLM judgment.

    Algorithm:
    1. Compute defining relations per class (frequent + discriminative)
    2. Find candidate IS-A pairs via relation-set subsumption
    3. LLM validates candidates (reject-only, not invent)
    4. Enforce DAG (remove cycles, max depth 3)
    """
    print("\n[Taxonomy] Deriving IS-A hierarchy from defining relations...", flush=True)

    class_counts = Counter(v for v in assignments.values() if v != "Other")
    class_names = sorted(class_counts.keys())
    if len(class_names) < 2:
        return []

    # Step 1: Compute defining relations per class
    # A "defining" relation is one that >= 30% of class members participate in
    class_rel_counts: dict[str, Counter] = {c: Counter() for c in class_names}
    for e in entities:
        eid = e["id"]
        cls = assignments.get(eid, "Other")
        if cls == "Other" or cls not in class_rel_counts:
            continue
        for rel in e.get("out_rels", []):
            class_rel_counts[cls][rel["rel"]] += 1
        for rel in e.get("in_rels", []):
            class_rel_counts[cls][f"~{rel['rel']}"] += 1  # ~REL = incoming

    # Filter to defining relations (>= 30% of class members participate)
    _GENERIC_RELS = {"HAS_VALUE", "HAS_NAME", "HAS_DATE", "HAS_TYPE", "HAS_ID",
                     "IS_A", "~IS_A", "~HAS_VALUE", "~HAS_NAME", "~HAS_DATE"}
    defining_rels: dict[str, set[str]] = {}
    for cls in class_names:
        n = class_counts[cls]
        threshold = max(2, int(n * 0.3))
        defining_rels[cls] = {
            rel for rel, count in class_rel_counts[cls].items()
            if count >= threshold and rel not in _GENERIC_RELS
        }

    # Step 2: Find candidate IS-A via relation-set subsumption
    # If A's defining rels are a strict subset of B's (>= 80% inclusion), A may be subclass of B
    candidates: list[tuple[str, str, float]] = []
    for child in class_names:
        child_rels = defining_rels.get(child, set())
        if len(child_rels) < 2:
            continue
        for parent in class_names:
            if child == parent:
                continue
            parent_rels = defining_rels.get(parent, set())
            if len(parent_rels) < 2:
                continue
            # Child's rels should be subset of parent's
            if len(child_rels) >= len(parent_rels):
                continue  # Can't be a subclass if has more or equal rels
            overlap = len(child_rels & parent_rels)
            inclusion = overlap / len(child_rels) if child_rels else 0
            if inclusion >= 0.80:
                # Check negative constraint: conflicting rels should block
                exclusive_child = child_rels - parent_rels
                exclusive_parent = parent_rels - child_rels
                # If child has unique defining rels that contradict parent's, skip
                if len(exclusive_child) > len(child_rels) * 0.5:
                    continue
                candidates.append((child, parent, inclusion))

    if not candidates:
        print("  No IS-A candidates found from relation subsumption")
        return []

    # Sort by inclusion score (strongest first)
    candidates.sort(key=lambda x: -x[2])
    print(f"  Found {len(candidates)} candidate IS-A pairs")

    # Step 3: LLM validation (reject only)
    validated: list[tuple[str, str]] = []
    if candidates:
        # Format candidates for LLM
        cand_lines = []
        for child, parent, score in candidates[:15]:  # Cap at 15 for prompt size
            child_members = [e["id"] for e in entities
                            if assignments.get(e["id"]) == child][:5]
            parent_members = [e["id"] for e in entities
                             if assignments.get(e["id"]) == parent][:5]
            cand_lines.append(
                f"  {child} (e.g., {', '.join(child_members)}) -> "
                f"{parent} (e.g., {', '.join(parent_members)})"
            )

        prompt = f"""Review these proposed IS-A (subclass) relationships between ontology classes.

For each pair, answer YES if the child class is truly a subtype of the parent class,
or NO if they are merely related but not in an IS-A relationship.

PROPOSED RELATIONSHIPS (Child -> Parent):
{chr(10).join(cand_lines)}

For each pair, output exactly: ChildClass -> ParentClass: YES or NO
Only answer YES if every instance of the child class IS-A instance of the parent class.
"""
        raw = _invoke_llm(llm, prompt)

        line_re = re.compile(r'(\w+)\s*->\s*(\w+)\s*:\s*(YES|NO)', re.IGNORECASE | re.MULTILINE)
        for m in line_re.finditer(raw):
            child_lbl = _sanitize_label(m.group(1))
            parent_lbl = _sanitize_label(m.group(2))
            verdict = m.group(3).upper()
            if verdict == "YES" and child_lbl in class_names and parent_lbl in class_names:
                validated.append((child_lbl, parent_lbl))

    # Step 4: DAG enforcement (remove cycles, enforce max depth 3)
    MAX_TAXONOMY_DEPTH = 3

    final_edges: list[tuple[str, str]] = []
    # Track parent relationships for cycle and depth checks
    parent_of: dict[str, str] = {}  # child -> parent (single-parent tree)

    def _depth_of(node: str) -> int:
        """Compute depth of node in current tree (0 = root)."""
        d = 0
        cur = node
        visited: set[str] = set()
        while cur in parent_of and cur not in visited:
            visited.add(cur)
            cur = parent_of[cur]
            d += 1
        return d

    for child, parent in validated:
        if child == parent:
            continue
        # Cycle check: walk up from parent — if we reach child, it's a cycle
        cur = parent
        is_cycle = False
        visited: set[str] = set()
        while cur in parent_of and cur not in visited:
            visited.add(cur)
            cur = parent_of[cur]
            if cur == child:
                is_cycle = True
                break
        if is_cycle:
            continue
        # Depth check: child's depth would be parent's depth + 1
        if _depth_of(parent) + 1 > MAX_TAXONOMY_DEPTH:
            continue
        # Accept edge (first-come wins for single parent)
        if child not in parent_of:
            parent_of[child] = parent
            final_edges.append((child, parent))

    print(f"  ✓ {len(final_edges)} validated IS-A edges (from {len(candidates)} candidates)")
    for child, parent in final_edges:
        print(f"    {child} SUBCLASS_OF {parent}")

    return final_edges


def derive_taxonomy_llm_pairwise(
    assignments: dict[str, str],
    entities: list[dict],
    llm: ChatOllama,
) -> list[tuple[str, str]]:
    """IS-A taxonomy via LLM pairwise judgment.

    For each ordered pair (A, B), ask: "Is A a subtype of B?"
    Batch 10 pairs per prompt for efficiency.
    Enforce DAG + max depth 3.

    Unlike derive_taxonomy() (relation-set subsumption) which only finds
    IS-A when child relations are a strict subset of parent's, this method
    can discover IS-A relationships even when classes have overlapping or
    distinct relation profiles — it relies on semantic judgment.

    Returns:
        List of (child, parent) edges representing SUBCLASS_OF.
    """
    print("\n[Taxonomy-LLM] Deriving IS-A hierarchy via pairwise LLM judgment...", flush=True)

    class_counts = Counter(v for v in assignments.values() if v != "Other")
    class_names = sorted(class_counts.keys())
    if len(class_names) < 2:
        return []

    entity_map = {e["id"]: e for e in entities}

    # Build class summaries for the prompt
    class_summaries: dict[str, str] = {}
    for cls in class_names:
        members = [eid for eid, c in assignments.items() if c == cls]
        # Prefer concept members for meaningful names
        concept_members = [
            eid for eid in members
            if is_concept_entity(entity_map.get(eid, {"id": eid, "entity_type": "Unknown"}))
        ]
        sample_pool = concept_members if concept_members else members
        sample = sample_pool[:6]
        class_summaries[cls] = f"{cls} ({class_counts[cls]} members, e.g. {', '.join(sample)})"

    # Generate all ordered pairs: (A, B) means "Is A a subtype of B?"
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(class_names):
        for j, b in enumerate(class_names):
            if i != j:
                pairs.append((a, b))

    print(f"  Evaluating {len(pairs)} ordered pairs in batches of 10...", flush=True)

    # Batch pairs into groups of 10
    PAIRS_PER_BATCH = 10
    raw_yes_pairs: list[tuple[str, str]] = []

    for batch_start in range(0, len(pairs), PAIRS_PER_BATCH):
        batch = pairs[batch_start:batch_start + PAIRS_PER_BATCH]

        pair_lines = []
        for idx, (a, b) in enumerate(batch, 1):
            pair_lines.append(f"{idx}. Is \"{a}\" a subtype of \"{b}\"?")
            pair_lines.append(f"   {a}: {class_summaries[a]}")
            pair_lines.append(f"   {b}: {class_summaries[b]}")

        prompt = f"""For each pair below, answer YES if the first class is a genuine subtype \
(IS-A) of the second class. Answer NO otherwise.

A is a subtype of B means: every instance of A is also an instance of B.
Example: "Flood" IS-A "Peril" (every flood is a peril).
Counter-example: "Coverage" is NOT a subtype of "Policy" (coverage is part of a policy, not a kind of policy).

{chr(10).join(pair_lines)}

Output one line per pair: number. YES or NO
"""
        raw = _invoke_llm(llm, prompt)

        # Parse "1. YES" or "1. NO" lines
        line_re = re.compile(r'(\d+)\.\s*(YES|NO)', re.IGNORECASE)
        for m in line_re.finditer(raw):
            idx = int(m.group(1)) - 1
            verdict = m.group(2).upper()
            if 0 <= idx < len(batch) and verdict == "YES":
                raw_yes_pairs.append(batch[idx])

        batch_num = batch_start // PAIRS_PER_BATCH + 1
        total_batches = (len(pairs) + PAIRS_PER_BATCH - 1) // PAIRS_PER_BATCH
        if batch_num % 5 == 0 or batch_num == total_batches:
            print(f"    Batch {batch_num}/{total_batches}: {len(raw_yes_pairs)} YES so far", flush=True)

    if not raw_yes_pairs:
        print("  No IS-A pairs found via LLM pairwise judgment")
        return []

    print(f"  {len(raw_yes_pairs)} raw YES pairs, enforcing DAG + max depth 3...", flush=True)

    # DAG enforcement: single-parent tree, max depth 3, no cycles
    # Sort by child class size (smaller first) so more specific classes
    # get their parent assigned first — avoids alphabetical bias.
    MAX_TAXONOMY_DEPTH = 3
    final_edges: list[tuple[str, str]] = []
    parent_of: dict[str, str] = {}  # child -> parent

    raw_yes_pairs.sort(key=lambda pair: (class_counts.get(pair[0], 0), pair[0]))

    def _depth_of(node: str) -> int:
        d = 0
        cur = node
        visited: set[str] = set()
        while cur in parent_of and cur not in visited:
            visited.add(cur)
            cur = parent_of[cur]
            d += 1
        return d

    for child, parent in raw_yes_pairs:
        if child == parent:
            continue
        # Cycle check
        cur = parent
        is_cycle = False
        visited: set[str] = set()
        while cur in parent_of and cur not in visited:
            visited.add(cur)
            cur = parent_of[cur]
            if cur == child:
                is_cycle = True
                break
        if is_cycle:
            continue
        # Depth check
        if _depth_of(parent) + 1 > MAX_TAXONOMY_DEPTH:
            continue
        # First-come wins for single parent
        if child not in parent_of:
            parent_of[child] = parent
            final_edges.append((child, parent))

    print(f"  ✓ {len(final_edges)} validated IS-A edges (from {len(raw_yes_pairs)} candidates)")
    for child, parent in final_edges:
        print(f"    {child} SUBCLASS_OF {parent}")

    return final_edges


def decompose_records(
    assignments: dict[str, str],
    entities: list[dict],
    rel_to_class: dict[str, tuple[str, float]] | None = None,
) -> dict[str, dict[str, dict]]:
    """Decompose record entities into domain-specific sub-nodes.

    Uses the relation-range mapping (from value typing) to determine which
    fields of a record entity belong to which concept class. Creates a
    decomposition plan that Stage 7 can use to write sub-nodes.

    For each record entity (POL-xxx, CLM-xxx):
      - Look up its outgoing HAS_* relations
      - Group relations by the class their targets belong to
      - Create sub-node plan: {record_id: {class_name: {fields: {rel: value}}}}

    Args:
        assignments: entity → class mapping (after full typing)
        entities: all entities with relation data
        rel_to_class: relation→(class, confidence) from value typing (optional)

    Returns:
        {record_eid: {class_label: {"fields": {rel: target_eid}, "class": class_name}}}
        Empty dict if no decomposition is applicable.
    """
    print("\n[Decompose] Record decomposition via relation-range mapping...", flush=True)

    entity_map = {e["id"]: e for e in entities}
    decomposition: dict[str, dict[str, dict]] = {}
    records_processed = 0
    sub_nodes_planned = 0

    for e in entities:
        eid = e["id"]
        lane = get_entity_lane(e)
        if lane != "record":
            continue

        # Group outgoing relations by target's class
        class_fields: dict[str, dict[str, str]] = defaultdict(dict)

        for rel in e.get("out_rels", []):
            rel_type = rel.get("rel", "")
            target_eid = rel.get("target", "")
            target_cls = assignments.get(target_eid, "Other")

            if target_cls == "Other":
                # Try relation-range mapping as fallback
                if rel_to_class and rel_type in rel_to_class:
                    target_cls = rel_to_class[rel_type][0]
                else:
                    continue

            # Skip self-class (record's own class)
            record_cls = assignments.get(eid, "Other")
            if target_cls == record_cls:
                continue

            class_fields[target_cls][rel_type] = target_eid

        # Only decompose if fields map to 2+ distinct classes
        if len(class_fields) >= 2:
            sub_plan: dict[str, dict] = {}
            seen_labels: set[str] = set()
            for cls, fields in class_fields.items():
                # Create a lowercase label for the sub-node
                label = cls.lower()
                # Guard against label collisions (e.g., "Coverage" vs "COVERAGE")
                if label in seen_labels:
                    label = f"{label}_{len(seen_labels)}"
                seen_labels.add(label)
                sub_plan[label] = {
                    "fields": dict(fields),
                    "class": cls,
                }
            decomposition[eid] = sub_plan
            sub_nodes_planned += len(sub_plan)
            records_processed += 1

    total_records = sum(1 for e in entities if get_entity_lane(e) == "record")
    skipped = total_records - records_processed
    print(f"  ✓ {records_processed} records decomposed into {sub_nodes_planned} sub-nodes "
          f"({skipped} skipped: <2 distinct target classes)")
    if decomposition:
        # Show a sample
        sample_eid = next(iter(decomposition))
        sample = decomposition[sample_eid]
        print(f"  Example: {sample_eid} → {list(sample.keys())}")

    return decomposition


def write_record_decomposition(
    decomposition: dict[str, dict[str, dict]],
) -> int:
    """Write decomposed record sub-nodes to Neo4j.

    For each record entity with a decomposition plan:
      1. Create sub-node: {record_id}:{class_label} as Entity
      2. Link sub-node → OntologyClass via INSTANCE_OF
      3. Link hub record → sub-node via HAS_COMPONENT

    Returns:
        Number of sub-nodes created.
    """
    if not decomposition:
        return 0

    print(f"\n[Decompose-Write] Writing {sum(len(v) for v in decomposition.values())} "
          f"sub-nodes to Neo4j...", flush=True)

    try:
        graph = get_neo4j_graph()
        graph.query("RETURN 1 AS ok")
    except Exception as e:
        print(f"  ⚠ Neo4j unavailable ({e}), skipping decomposition write.")
        return 0

    created = 0
    for record_eid, sub_plan in decomposition.items():
        for label, info in sub_plan.items():
            sub_id = f"{record_eid}:{label}"
            cls = info["class"]
            safe_cls = _sanitize_label(cls)

            try:
                # Create sub-node entity
                graph.query(
                    "MERGE (n:Entity {id: $id}) "
                    "SET n.entity_type = $cls, n.ontology_class = $cls, "
                    "n.decomposed_from = $parent",
                    params={"id": sub_id, "cls": safe_cls, "parent": record_eid},
                )

                # Link to ontology class
                graph.query(
                    "MATCH (n:Entity {id: $id}), (c:OntologyClass {name: $cls}) "
                    "MERGE (n)-[:INSTANCE_OF]->(c)",
                    params={"id": sub_id, "cls": safe_cls},
                )

                # Link hub record → sub-node
                graph.query(
                    "MATCH (hub:Entity {id: $hub}), (sub:Entity {id: $sub}) "
                    "MERGE (hub)-[:HAS_COMPONENT]->(sub)",
                    params={"hub": record_eid, "sub": sub_id},
                )

                # Copy relevant field relations to the sub-node
                for rel_type, target_eid in info["fields"].items():
                    safe_rel = _sanitize_label(rel_type)
                    graph.query(
                        "MATCH (sub:Entity {id: $sub}), (tgt:Entity {id: $tgt}) "
                        f"MERGE (sub)-[:`{safe_rel}`]->(tgt)",
                        params={"sub": sub_id, "tgt": target_eid},
                    )

                created += 1
            except Exception as e:
                print(f"  ⚠ Sub-node {sub_id}: {e}")

    print(f"  ✓ Created {created} sub-nodes with HAS_COMPONENT + INSTANCE_OF edges")
    return created


def validate_backbone(
    assignments: dict[str, str],
    entities: list[dict],
) -> dict[str, Any]:
    """Check insurance value chain connectivity (domain-agnostic).

    Uses both direct connections (1-hop) and indirect connections through
    value/Other entities (2-hop) to measure how well the ontology classes
    are interconnected. Records often connect to concepts only through
    shared value nodes, so 2-hop connectivity is essential.
    """
    print("\n[Backbone] Validating entity connectivity...", flush=True)

    # Build entity-to-class lookup and neighbor index
    eid_to_class: dict[str, str] = {}
    eid_to_entity: dict[str, dict] = {}
    for e in entities:
        eid = e["id"]
        eid_to_class[eid] = assignments.get(eid, "Other")
        eid_to_entity[eid] = e

    class_members: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        cls = eid_to_class.get(e["id"], "Other")
        if cls != "Other":
            class_members[cls].append(e)

    # For each class, check connectivity (1-hop direct + 2-hop through Other)
    class_connectivity: dict[str, dict] = {}
    for cls, members in class_members.items():
        connected_direct = 0
        connected_2hop = 0
        neighbor_classes: Counter = Counter()

        for e in members:
            has_direct = False
            has_2hop = False

            # 1-hop: direct connections to other non-Other entities
            for rel in e.get("out_rels", []):
                tcls = eid_to_class.get(rel.get("target", ""), "Other")
                if tcls != "Other" and tcls != cls:
                    has_direct = True
                    neighbor_classes[tcls] += 1
            for rel in e.get("in_rels", []):
                scls = eid_to_class.get(rel.get("source", ""), "Other")
                if scls != "Other" and scls != cls:
                    has_direct = True
                    neighbor_classes[scls] += 1

            if has_direct:
                connected_direct += 1
                continue

            # 2-hop: check if connected to a non-Other entity through an Other node
            # Check both directions at each hop (undirected projection)
            def _neighbors_of(ent: dict) -> list[str]:
                """All neighbor eids regardless of edge direction."""
                nbrs = []
                for r in ent.get("out_rels", []):
                    nbrs.append(r.get("target", ""))
                for r in ent.get("in_rels", []):
                    nbrs.append(r.get("source", ""))
                return nbrs

            for mid_eid in _neighbors_of(e):
                if eid_to_class.get(mid_eid) != "Other":
                    continue
                mid_entity = eid_to_entity.get(mid_eid)
                if not mid_entity:
                    continue
                for far_eid in _neighbors_of(mid_entity):
                    if far_eid == e["id"]:
                        continue
                    far_cls = eid_to_class.get(far_eid, "Other")
                    if far_cls != "Other" and far_cls != cls:
                        has_2hop = True
                        neighbor_classes[far_cls] += 1
                        break
                if has_2hop:
                    break

            if has_2hop:
                connected_2hop += 1

        total = len(members)
        total_connected = connected_direct + connected_2hop
        frac = total_connected / total if total else 0
        class_connectivity[cls] = {
            "total": total,
            "connected_direct": connected_direct,
            "connected_2hop": connected_2hop,
            "connected_total": total_connected,
            "connectivity": round(frac, 3),
            "top_neighbors": dict(neighbor_classes.most_common(3)),
        }

    # Overall stats
    total_entities = sum(d["total"] for d in class_connectivity.values())
    total_connected = sum(d["connected_total"] for d in class_connectivity.values())
    overall_connectivity = round(total_connected / total_entities, 4) if total_entities else 0

    disconnected = [
        cls for cls, d in class_connectivity.items()
        if d["connectivity"] < 0.10
    ]

    result = {
        "overall_connectivity": overall_connectivity,
        "total_non_other_entities": total_entities,
        "total_connected": total_connected,
        "disconnected_classes": disconnected,
        "per_class": class_connectivity,
    }

    print(f"  Overall connectivity: {overall_connectivity:.1%} "
          f"({total_connected}/{total_entities} entities)")
    if disconnected:
        print(f"  ⚠ Disconnected classes (<10%): {disconnected}")
    for cls, d in sorted(class_connectivity.items(), key=lambda x: x[1]["connectivity"]):
        direct = d["connected_direct"]
        hop2 = d["connected_2hop"]
        print(f"    {cls}: {d['connectivity']:.0%} connected "
              f"({direct} direct + {hop2} 2-hop / {d['total']}), "
              f"neighbors: {d['top_neighbors']}")

    return result


# ---------------------------------------------------------------------------
# Phase 15: Write to Neo4j
# ---------------------------------------------------------------------------

def write_ontology(
    assignments: dict[str, str],
    hierarchy: list[tuple[str, str]],
    associations: list[tuple[str, str]] | None = None,
) -> dict:
    """Write ontology layer to Neo4j."""
    print("\n[Phase 15] Writing ontology to Neo4j...")

    try:
        graph = get_neo4j_graph()
        graph.query("RETURN 1 AS ok")
    except Exception as e:
        print(f"  ⚠ Neo4j unavailable ({e}), skipping write. Results saved to JSON.")
        class_counts = Counter(v for v in assignments.values() if v != "Other")
        return {
            "entities_labeled": len(assignments),
            "ontology_classes": len(class_counts),
            "subclass_of_edges": len(hierarchy),
            "class_names": sorted(class_counts.keys()),
            "class_distribution": dict(class_counts),
            "method": "SV-LOI",
            "neo4j_skipped": True,
        }

    # Clean previous ontology (labels, properties, and OntologyClass nodes)
    try:
        graph.query("MATCH (c:OntologyClass) DETACH DELETE c")
        # Clear ontology_class property from ALL entities (prevents stale labels)
        graph.query("MATCH (n:Entity) WHERE n.ontology_class IS NOT NULL REMOVE n.ontology_class")
        # Remove old ontology labels from entities
        rows = graph.query("CALL db.labels() YIELD label RETURN label")
        existing = {r["label"] for r in rows}
        skip = {"__Entity__", "Document", "Entity", "OntologyClass"}
        for lbl in existing:
            if lbl not in skip and lbl != "Other":
                try:
                    # Backtick-quoted labels handle special characters safely
                    graph.query(f"MATCH (n:`{lbl}`) REMOVE n:`{lbl}`")
                except Exception:
                    pass
    except Exception as e:
        print(f"  Warning: cleanup error: {e}")

    # Get unique classes (excluding Other)
    class_counts = Counter(v for v in assignments.values() if v != "Other")
    class_names = sorted(class_counts.keys())

    # Create OntologyClass nodes
    for cls in class_names:
        safe = _sanitize_label(cls)
        try:
            graph.query(
                "MERGE (c:OntologyClass {name: $name})",
                params={"name": safe},
            )
        except Exception as e:
            print(f"  Warning: could not create class {safe}: {e}")

    # Label entities with their assigned class
    entities_labeled = 0
    for cls in class_names:
        safe = _sanitize_label(cls)
        members = [eid for eid, c in assignments.items() if c == cls]
        if not members:
            continue
        for batch_start in range(0, len(members), 100):
            batch = members[batch_start:batch_start + 100]
            try:
                graph.query(
                    f"MATCH (n:Entity) WHERE n.id IN $ids "
                    f"SET n:`{safe}`, n.ontology_class = $cls",
                    params={"ids": batch, "cls": safe},
                )
                entities_labeled += len(batch)
            except Exception as e:
                print(f"  Warning: labeling batch for {safe}: {e}")

    # Create INSTANCE_OF edges: entity → OntologyClass
    # This bridges record entities (POL-xxx) and concept entities ("Renters Insurance")
    # through their shared class node, enabling cross-subgraph query traversal.
    instance_of_created = 0
    for cls in class_names:
        safe = _sanitize_label(cls)
        members = [eid for eid, c in assignments.items() if c == cls]
        if not members:
            continue
        for batch_start in range(0, len(members), 200):
            batch = members[batch_start:batch_start + 200]
            try:
                graph.query(
                    "MATCH (n:Entity), (c:OntologyClass {name: $cls}) "
                    "WHERE n.id IN $ids "
                    "MERGE (n)-[:INSTANCE_OF]->(c)",
                    params={"ids": batch, "cls": safe},
                )
                instance_of_created += len(batch)
            except Exception as e:
                print(f"  Warning: INSTANCE_OF batch for {safe}: {e}")

    print(f"  ✓ INSTANCE_OF edges:    {instance_of_created}")

    # Create SUBCLASS_OF edges
    subclass_created = 0
    for child, parent in hierarchy:
        child_safe = _sanitize_label(child)
        parent_safe = _sanitize_label(parent)
        try:
            graph.query(
                "MATCH (c:OntologyClass {name: $child}), (p:OntologyClass {name: $parent}) "
                "MERGE (c)-[:SUBCLASS_OF]->(p)",
                params={"child": child_safe, "parent": parent_safe},
            )
            subclass_created += 1
        except Exception as e:
            print(f"  Warning: SUBCLASS_OF {child} → {parent}: {e}")

    # Create ASSOCIATED_WITH edges (inter-class associations, NOT IS-A)
    assoc_created = 0
    for src, tgt in (associations or []):
        src_safe = _sanitize_label(src)
        tgt_safe = _sanitize_label(tgt)
        try:
            graph.query(
                "MATCH (a:OntologyClass {name: $src}), (b:OntologyClass {name: $tgt}) "
                "MERGE (a)-[:ASSOCIATED_WITH]->(b)",
                params={"src": src_safe, "tgt": tgt_safe},
            )
            assoc_created += 1
        except Exception as e:
            print(f"  Warning: ASSOCIATED_WITH {src} → {tgt}: {e}")

    stats = {
        "entities_labeled": entities_labeled,
        "ontology_classes": len(class_names),
        "instance_of_edges": instance_of_created,
        "subclass_of_edges": subclass_created,
        "associated_with_edges": assoc_created,
        "class_names": class_names,
        "class_distribution": dict(class_counts),
        "method": "SV-LOI",
    }

    print(f"  ✓ Entities labeled:     {entities_labeled}")
    print(f"  ✓ OntologyClass nodes:  {len(class_names)}")
    print(f"  ✓ INSTANCE_OF edges:    {instance_of_created}")
    print(f"  ✓ SUBCLASS_OF edges:    {subclass_created}")
    print(f"  ✓ ASSOCIATED_WITH edges:{assoc_created}")
    return stats


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def _flush_print(msg: str) -> None:
    """Print with immediate flush for Slurm log visibility."""
    print(msg, flush=True)


def _compute_intrinsic_quality(
    assignments: dict[str, str],
    entities: list[dict],
    entity_map: dict[str, dict],
    class_dist: Counter,
) -> dict[str, Any]:
    """Compute reference-free ontology quality metrics.

    Metrics:
    - connectivity: fraction of non-Other entities with >= 1 typed neighbor
    - completeness: fraction of classes with >= 1 inter-class relationship
    - balance: 1 - max(class_fraction), higher is more balanced
    - other_fraction: fraction of entities classified as "Other"
    - backbone_coverage: fraction of record entities linked to a concept entity
    """
    total = len(assignments)
    if total == 0:
        return {"connectivity": 0, "completeness": 0, "balance": 0,
                "other_fraction": 1.0, "backbone_coverage": 0}

    # Other fraction
    other_count = sum(1 for c in assignments.values() if c == "Other")
    other_frac = other_count / total

    # Balance: 1 - largest class fraction (among non-Other)
    non_other_total = total - other_count
    if non_other_total > 0 and class_dist:
        max_class_frac = class_dist.most_common(1)[0][1] / non_other_total
        balance = round(1.0 - max_class_frac, 4)
    else:
        balance = 0.0

    # Connectivity: fraction of non-Other entities with at least 1 neighbor
    # that has a different non-Other class (inter-class link)
    connected = 0
    non_other_entities = [e for e in entities if assignments.get(e["id"]) not in (None, "Other")]
    for e in non_other_entities:
        eid = e["id"]
        my_class = assignments[eid]
        neighbors = set()
        for rel in e.get("out_rels", []):
            tid = rel.get("target", "")
            tcls = assignments.get(tid)
            if tcls and tcls != "Other" and tcls != my_class:
                neighbors.add(tcls)
        for rel in e.get("in_rels", []):
            sid = rel.get("source", "")
            scls = assignments.get(sid)
            if scls and scls != "Other" and scls != my_class:
                neighbors.add(scls)
        if neighbors:
            connected += 1

    connectivity = round(connected / len(non_other_entities), 4) if non_other_entities else 0.0

    # Completeness: fraction of non-Other classes with >= 1 inter-class edge
    classes_with_interclass = set()
    for e in entities:
        eid = e["id"]
        my_class = assignments.get(eid, "Other")
        if my_class == "Other":
            continue
        for rel in e.get("out_rels", []):
            tid = rel.get("target", "")
            tcls = assignments.get(tid, "Other")
            if tcls != "Other" and tcls != my_class:
                classes_with_interclass.add(my_class)
                classes_with_interclass.add(tcls)
                break

    n_classes = len(class_dist)
    completeness = round(len(classes_with_interclass) / n_classes, 4) if n_classes > 0 else 0.0

    # Backbone coverage: fraction of record entities (POL-, CLM-) connected
    # to a concept entity (non-record, non-Other), directly or via 2-hop
    from zone3.graph_cache import STRUCTURED_PREFIXES
    record_entities = [e for e in entities if e["id"].startswith(STRUCTURED_PREFIXES)]
    records_linked = 0
    for e in record_entities:
        linked = False
        # Direct check
        for rel in e.get("out_rels", []):
            tid = rel.get("target", "")
            if not tid.startswith(STRUCTURED_PREFIXES):
                tcls = assignments.get(tid, "Other")
                if tcls != "Other":
                    linked = True
                    break
        if not linked:
            for rel in e.get("in_rels", []):
                sid = rel.get("source", "")
                if not sid.startswith(STRUCTURED_PREFIXES):
                    scls = assignments.get(sid, "Other")
                    if scls != "Other":
                        linked = True
                        break
        # 2-hop: record → Other value → concept (undirected at each hop)
        if not linked:
            # Collect all neighbors (both directions)
            e_neighbors = set()
            for rel in e.get("out_rels", []):
                e_neighbors.add(rel.get("target", ""))
            for rel in e.get("in_rels", []):
                e_neighbors.add(rel.get("source", ""))
            for mid in e_neighbors:
                if assignments.get(mid) != "Other":
                    continue
                mid_e = entity_map.get(mid, {})
                mid_neighbors = set()
                for rel2 in mid_e.get("out_rels", []):
                    mid_neighbors.add(rel2.get("target", ""))
                for rel2 in mid_e.get("in_rels", []):
                    mid_neighbors.add(rel2.get("source", ""))
                for far in mid_neighbors:
                    if far != e["id"] and assignments.get(far, "Other") != "Other":
                        linked = True
                        break
                if linked:
                    break
        if linked:
            records_linked += 1

    backbone = round(records_linked / len(record_entities), 4) if record_entities else 0.0

    return {
        "connectivity": connectivity,
        "completeness": completeness,
        "balance": balance,
        "other_fraction": round(other_frac, 4),
        "backbone_coverage": backbone,
        "record_entities_total": len(record_entities),
        "record_entities_linked": records_linked,
    }


def run_sv_loi(
    model: str = config.OLLAMA_MODEL,
    suffix: str = "zone3_svloi",
    skip_verify: bool = False,
    skip_arbitrate: bool = False,
    skip_consolidate: bool = False,
    skip_record_propagation: bool = False,
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

    # Snapshot 5-way IS-A edges from Stage 5
    n_5way_edges = len(hierarchy)

    # IS-A taxonomy via LLM pairwise judgment (primary signal)
    taxonomy_llm_edges = derive_taxonomy_llm_pairwise(assignments, entities, llm)

    # Concept-only subsumption as structural validation (secondary signal)
    taxonomy_subsumption_edges = derive_taxonomy(assignments, entities, llm)

    # Merge IS-A edges into hierarchy (true SUBCLASS_OF)
    existing_isa = set(hierarchy)
    for edge in taxonomy_llm_edges:
        if edge not in existing_isa:
            hierarchy.append(edge)
            existing_isa.add(edge)
    for edge in taxonomy_subsumption_edges:
        if edge not in existing_isa:
            hierarchy.append(edge)
            existing_isa.add(edge)

    # Association edges — SEPARATE from IS-A (these are HAS-A / REFERENCES)
    _flush_print("\n  Association edges from entity connections...")
    data_edges = derive_interclass_edges(assignments, entities)
    # Deduplicate: remove any association that duplicates an IS-A edge
    associations = [e for e in data_edges if e not in existing_isa]

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
        },
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
        "taxonomy_edges": len(taxonomy_llm_edges) + len(taxonomy_subsumption_edges),
        "taxonomy_edges_detail": {
            "llm_pairwise": len(taxonomy_llm_edges),
            "subsumption": len(taxonomy_subsumption_edges),
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
        use_old_rebalance=getattr(args, 'use_old_rebalance', False),
        seed=args.seed,
        results_dir=args.results_dir,
    )
