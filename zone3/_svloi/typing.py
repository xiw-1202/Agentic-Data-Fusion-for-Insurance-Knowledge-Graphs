"""SV-LOI entity typing: evidence analysis, class vocabulary discovery, batched LLM assignment."""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Optional

from langchain_ollama import ChatOllama

import config
from zone3._svloi.constants import (
    BATCH_SIZE,
    MAX_CLASS_FRACTION,
    TARGET_CLASSES_MIN,
    TARGET_CLASSES_MAX,
    FORBIDDEN_CLASS_NAMES,
    ZONE2_TYPE_NORMALIZATION,
)
from zone3._svloi.utils import (
    _invoke_llm,
    _parse_json_safely,
    _sanitize_label,
)
from zone3.graph_cache import (
    get_entity_lane,
    is_concept_entity,
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

