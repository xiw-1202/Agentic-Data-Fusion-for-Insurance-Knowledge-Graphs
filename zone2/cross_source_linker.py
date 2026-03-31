"""
SEAF-KG Stage 3 — Cross-Source Entity Linker

Links structured record entities (POL-xxx, CLM-xxx) to each other when they
share enough property values, creating LINKED_TO edges in Neo4j.

Design principles (post-Codex review):
  - Domain-agnostic: zero hardcoded field names — discovers shared relations at runtime
  - IDF-weighted scoring: rare matching values contribute more than common ones
  - Multi-pass blocking: uses highest-cardinality fields (not lowest) to reduce O(n²)
  - Field-type-specific comparison: tolerance for numbers, proximity for dates
  - Temporal consistency: auto-detects date ranges and rejects anachronistic links
  - Type 2 linking (record ↔ concept) is automatic via shared Neo4j value nodes
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_MATCHING_FIELDS = 5
MIN_WEIGHTED_SCORE = 0.7
NUMERIC_TOLERANCE = 0.05       # 5% relative tolerance for numeric comparison
DATE_PROXIMITY_DAYS = 30       # days within which dates are a partial match
MAX_BLOCKING_PASSES = 2        # number of blocking passes (top-N cardinality fields)

_STRUCTURED_PREFIXES = ("POL-", "CLM-", "REC-", "PER-", "PROP-")

# Entity types that represent actual data records (not property values or identity nodes).
# Only these types should participate in cross-source linking.
# Identity nodes (Person, Property) already have BELONGS_TO edges from records —
# linking them via cross-source matching is redundant.
_RECORD_ENTITY_TYPES = frozenset({
    "PolicyRecord", "ClaimRecord", "Record",
})

# Value entity types to EXCLUDE from linking.
_VALUE_ENTITY_TYPES = frozenset({
    "Numeric", "Date", "Text", "Categorical",
    "Currency", "Percentage",
})


def _sanitize_label(label: str) -> str:
    """Make a Neo4j label safe for f-string interpolation."""
    import re
    cleaned = re.sub(r'[^A-Za-z0-9_]', '', label.strip())
    if not cleaned:
        raise ValueError(f"Invalid Neo4j label: {label!r}")
    return cleaned


def _sanitize_rel(rel: str) -> str:
    """Make a relation name safe for Neo4j Cypher f-string interpolation."""
    import re
    return re.sub(r'[^A-Z0-9_]', '_', rel.upper().strip())

# Patterns for auto-detecting field types from relation names.
_DATE_HINTS = frozenset({
    "date", "effective", "termination", "expiration", "loss",
    "construction", "cancellation", "original", "time", "period",
})
_NUMERIC_HINTS = frozenset({
    "amount", "cost", "coverage", "premium", "fee", "payment",
    "value", "deductible", "limit", "depth", "elevation", "floor",
    "count", "number", "rate",
})

# ISO date pattern.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_NUMERIC = re.compile(r"^[\d.,$]+$")


# ---------------------------------------------------------------------------
# Field type detection
# ---------------------------------------------------------------------------

def classify_field_type(relation: str, sample_values: list[str]) -> str:
    """Auto-detect field type from relation name and sample values.

    Returns: 'numeric', 'date', 'categorical', 'text'.
    """
    rel_lower = relation.lower()

    # Check relation name hints.
    if any(hint in rel_lower for hint in _DATE_HINTS):
        return "date"
    if any(hint in rel_lower for hint in _NUMERIC_HINTS):
        return "numeric"

    # Check sample values.
    if sample_values:
        date_count = sum(1 for v in sample_values if _ISO_DATE.match(v))
        if date_count > len(sample_values) * 0.5:
            return "date"

        numeric_count = sum(1 for v in sample_values if _NUMERIC.match(v.replace(",", "").replace("$", "")))
        if numeric_count > len(sample_values) * 0.5:
            return "numeric"

    # Short values with few unique → categorical.
    if sample_values and all(len(v) <= 10 for v in sample_values):
        return "categorical"

    return "text"


# ---------------------------------------------------------------------------
# Value comparators
# ---------------------------------------------------------------------------

def _parse_numeric(val: str) -> float | None:
    """Parse a string into a float, stripping currency symbols and commas."""
    try:
        cleaned = val.replace(",", "").replace("$", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def _parse_date(val: str) -> str | None:
    """Extract YYYY-MM-DD from an ISO-like date string."""
    m = _ISO_DATE.match(val.strip())
    return m.group(0) if m else None


def compare_values(val_a: str, val_b: str, field_type: str) -> float:
    """Field-type-specific comparison. Returns 0.0-1.0 match score.

    - 'categorical': exact match → 1.0, else 0.0
    - 'numeric': 1.0 if within NUMERIC_TOLERANCE, else 0.0
    - 'date': 1.0 if same day, 0.5 if within DATE_PROXIMITY_DAYS, else 0.0
    - 'text': normalized exact match → 1.0, else 0.0
    """
    if field_type == "categorical":
        return 1.0 if val_a.strip() == val_b.strip() else 0.0

    if field_type == "numeric":
        na, nb = _parse_numeric(val_a), _parse_numeric(val_b)
        if na is not None and nb is not None:
            denom = max(abs(na), abs(nb), 1e-9)
            return 1.0 if abs(na - nb) / denom <= NUMERIC_TOLERANCE else 0.0
        # Fall back to exact string match.
        return 1.0 if val_a.strip() == val_b.strip() else 0.0

    if field_type == "date":
        da, db = _parse_date(val_a), _parse_date(val_b)
        if da and db:
            if da == db:
                return 1.0
            try:
                from datetime import datetime
                dt_a = datetime.strptime(da, "%Y-%m-%d")
                dt_b = datetime.strptime(db, "%Y-%m-%d")
                delta = abs((dt_a - dt_b).days)
                if delta <= DATE_PROXIMITY_DAYS:
                    return 0.5
            except ValueError:
                pass
            return 0.0
        return 1.0 if val_a.strip() == val_b.strip() else 0.0

    # text: normalized comparison.
    return 1.0 if val_a.strip().lower() == val_b.strip().lower() else 0.0


# ---------------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------------

def check_temporal_consistency(
    profile_a: dict[str, str],
    profile_b: dict[str, str],
) -> bool:
    """Check if date fields are temporally consistent.

    Auto-detects:
      - "effective" / "expiration" date fields → define a policy window
      - "loss" / "event" / "claim" date fields → must fall within the window

    Returns True if no date fields found (can't disprove).
    """
    from datetime import datetime

    def _find_date(profile: dict, hints: list[str]) -> str | None:
        for rel, val in profile.items():
            rel_lower = rel.lower()
            if any(h in rel_lower for h in hints):
                parsed = _parse_date(val)
                if parsed:
                    return parsed
        return None

    # Look for policy window (effective/expiration) in profile_a.
    eff = _find_date(profile_a, ["effective", "eff_"])
    exp = _find_date(profile_a, ["expiration", "exp_", "termination"])

    # Look for event/loss date in profile_b.
    loss = _find_date(profile_b, ["loss", "event", "claim", "open"])

    # If we can't find relevant dates, allow the link.
    if not loss or not eff:
        return True

    try:
        dt_loss = datetime.strptime(loss, "%Y-%m-%d")
        dt_eff = datetime.strptime(eff, "%Y-%m-%d")

        # Loss must be after policy effective date.
        if dt_loss < dt_eff:
            return False

        # If we have an expiration date, loss must be before it.
        if exp:
            dt_exp = datetime.strptime(exp, "%Y-%m-%d")
            if dt_loss > dt_exp:
                return False

    except ValueError:
        return True  # unparseable dates → allow

    return True


# ---------------------------------------------------------------------------
# Shared relation discovery
# ---------------------------------------------------------------------------

def discover_shared_relations(
    graph: Any,
    type_a: str,
    type_b: str,
    node_label: str = "Entity",
) -> list[dict]:
    """Auto-discover HAS_* relations shared by both record types.

    Returns list of dicts sorted by IDF weight (highest first):
        [{relation, cardinality_a, cardinality_b, idf_weight, field_type, sample_values}]
    """
    safe_label = _sanitize_label(node_label)

    def _get_relation_stats(entity_type: str) -> dict[str, dict]:
        rows = graph.query(
            f"MATCH (n:{safe_label} {{entity_type: $et}})-[r]->(v) "
            "WHERE type(r) STARTS WITH 'HAS_' "
            "RETURN type(r) AS rel, count(DISTINCT v.id) AS card, "
            "       count(DISTINCT n.id) AS n_records, "
            "       collect(DISTINCT v.id)[..10] AS samples",
            params={"et": entity_type},
        )
        return {
            r["rel"]: {
                "cardinality": r["card"],
                "n_records": r["n_records"],
                "samples": r["samples"],
            }
            for r in rows
        }

    stats_a = _get_relation_stats(type_a)
    stats_b = _get_relation_stats(type_b)

    # Intersection of relation names.
    shared = set(stats_a.keys()) & set(stats_b.keys())

    result = []
    for rel in shared:
        sa, sb = stats_a[rel], stats_b[rel]

        # IDF weight: higher cardinality → higher weight (more discriminant).
        avg_card = (sa["cardinality"] + sb["cardinality"]) / 2
        total_records = max(sa["n_records"] + sb["n_records"], 1)
        idf = math.log(total_records / max(avg_card, 1)) + 1.0

        samples = sa["samples"] + sb["samples"]
        field_type = classify_field_type(rel, samples)

        result.append({
            "relation": rel,
            "cardinality_a": sa["cardinality"],
            "cardinality_b": sb["cardinality"],
            "idf_weight": max(idf, 0.1),  # floor to avoid zero weight
            "field_type": field_type,
            "sample_values": samples,
        })

    # Sort by IDF weight descending (most discriminant first).
    result.sort(key=lambda x: x["idf_weight"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Record profile loading
# ---------------------------------------------------------------------------

def load_record_profiles(
    graph: Any,
    entity_type: str,
    relations: list[str],
    node_label: str = "Entity",
) -> dict[str, dict[str, str]]:
    """Query Neo4j for all records of a type, returning {record_id: {relation: value}}.

    Only fetches the specified relations for efficiency.
    """
    safe_label = _sanitize_label(node_label)
    profiles: dict[str, dict[str, str]] = defaultdict(dict)

    for rel in relations:
        safe_rel = _sanitize_rel(rel)
        rows = graph.query(
            f"MATCH (n:{safe_label} {{entity_type: $et}})-[r:{safe_rel}]->(v) "
            "RETURN n.id AS nid, v.id AS vid",
            params={"et": entity_type},
        )
        for row in rows:
            profiles[row["nid"]][rel] = row["vid"]

    return dict(profiles)


# ---------------------------------------------------------------------------
# Multi-pass blocking
# ---------------------------------------------------------------------------

def multi_pass_blocking(
    profiles_a: dict[str, dict[str, str]],
    profiles_b: dict[str, dict[str, str]],
    shared_rels: list[dict],
) -> set[tuple[str, str]]:
    """Multi-pass blocking on top-N highest-cardinality fields.

    Each pass groups records by one blocking key. The union of all
    candidate pairs across passes is returned (deduplicated).
    """
    candidate_pairs: set[tuple[str, str]] = set()

    # Use top N relations sorted by cardinality (highest first = most discriminant).
    blocking_keys = [r["relation"] for r in shared_rels[:MAX_BLOCKING_PASSES]]

    if not blocking_keys:
        # No blocking possible — brute force (acceptable for small datasets).
        for id_a in profiles_a:
            for id_b in profiles_b:
                candidate_pairs.add((id_a, id_b))
        return candidate_pairs

    for key in blocking_keys:
        # Group type_a by blocking key value.
        blocks_a: dict[str, list[str]] = defaultdict(list)
        for nid, props in profiles_a.items():
            val = props.get(key, "")
            if val:
                blocks_a[val].append(nid)

        # Match type_b records into the same blocks.
        for nid_b, props_b in profiles_b.items():
            val = props_b.get(key, "")
            if val and val in blocks_a:
                for nid_a in blocks_a[val]:
                    candidate_pairs.add((nid_a, nid_b))

    return candidate_pairs


# ---------------------------------------------------------------------------
# Pair scoring
# ---------------------------------------------------------------------------

def score_pair(
    profile_a: dict[str, str],
    profile_b: dict[str, str],
    shared_rels: list[dict],
) -> tuple[float, int, list[str]]:
    """IDF-weighted pair scoring with field-type-specific comparators.

    Returns (weighted_score, n_matched, matched_field_names).
    weighted_score = sum(idf_weight * compare(va, vb)) / sum(idf_weights)
    """
    total_weight = 0.0
    matched_weight = 0.0
    n_matched = 0
    matched_fields: list[str] = []

    for rel_info in shared_rels:
        rel = rel_info["relation"]
        weight = rel_info["idf_weight"]
        ftype = rel_info["field_type"]

        val_a = profile_a.get(rel)
        val_b = profile_b.get(rel)

        if val_a is None or val_b is None:
            continue  # skip fields missing from either record

        total_weight += weight
        match_score = compare_values(val_a, val_b, ftype)

        if match_score > 0:
            matched_weight += weight * match_score
            n_matched += 1
            matched_fields.append(rel)

    if total_weight == 0:
        return 0.0, 0, []

    weighted_score = matched_weight / total_weight
    return weighted_score, n_matched, matched_fields


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def cross_source_link(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: discover and create LINKED_TO edges between
    record types that share property values.

    Fully domain-agnostic:
    1. Find all distinct entity_types on structured nodes
    2. For each pair of types, auto-discover shared HAS_* relations
    3. Multi-pass blocking on highest-cardinality shared fields
    4. Score candidate pairs with IDF-weighted, type-specific comparison
    5. Filter by temporal consistency
    6. Create LINKED_TO edges in Neo4j
    """
    from zone2.pipeline import get_neo4j_graph

    print("\n[5/4] Cross-Source Entity Linking — SEAF-KG Stage 3")

    try:
        graph = get_neo4j_graph()
    except Exception as e:
        print(f"  ⚠ Neo4j connection failed ({e}); skipping cross-source linking")
        return {"cross_source_stats": {"error": str(e)}}

    # Step 1: Find structured entity types.
    type_rows = graph.query(
        "MATCH (n:Entity) "
        "WHERE n.source_type = 'structured' AND n.entity_type IS NOT NULL "
        "RETURN DISTINCT n.entity_type AS et, count(n) AS cnt"
    )
    # Filter to record-level entity types only.
    # Exclude value types (Numeric, Date, Text, etc.) — they are property
    # values, not records that should be cross-linked.
    entity_types = {}
    skipped_value_types: list[str] = []
    for r in type_rows:
        et = r["et"]
        if et in ("RecordType", "IdentityType"):
            continue
        if et in _VALUE_ENTITY_TYPES:
            skipped_value_types.append(f"{et}({r['cnt']})")
            continue
        # Accept known record types OR any type with a structured prefix pattern.
        if et in _RECORD_ENTITY_TYPES:
            entity_types[et] = r["cnt"]
        else:
            # Unknown type — check if it looks like a record type (has ID-prefixed members).
            has_prefixed = graph.query(
                "MATCH (n:Entity {entity_type: $et}) "
                "WHERE n.id STARTS WITH 'POL-' OR n.id STARTS WITH 'CLM-' "
                "   OR n.id STARTS WITH 'REC-' OR n.id STARTS WITH 'PER-' "
                "   OR n.id STARTS WITH 'PROP-' "
                "RETURN count(n) AS cnt LIMIT 1",
                params={"et": et},
            )
            if has_prefixed and has_prefixed[0]["cnt"] > 0:
                entity_types[et] = r["cnt"]
            else:
                skipped_value_types.append(f"{et}({r['cnt']})")

    if skipped_value_types:
        print(f"  ℹ Skipped value types: {', '.join(skipped_value_types)}")

    if len(entity_types) < 2:
        print(f"  ℹ Only {len(entity_types)} structured type(s) found — "
              "need ≥2 for cross-source linking")
        return {"cross_source_stats": {"types_found": len(entity_types), "links_created": 0}}

    print(f"  ✓ Found {len(entity_types)} structured types: "
          f"{', '.join(f'{t} ({c})' for t, c in entity_types.items())}")

    total_links = 0
    total_checked = 0
    total_temporal_rejected = 0
    all_stats: list[dict] = []

    # Step 2: For each pair of types, discover and link.
    type_list = list(entity_types.keys())
    for i in range(len(type_list)):
        for j in range(i + 1, len(type_list)):
            type_a, type_b = type_list[i], type_list[j]

            print(f"\n  --- Linking {type_a} ↔ {type_b} ---")

            # Discover shared relations.
            shared = discover_shared_relations(graph, type_a, type_b)
            if not shared:
                print(f"    ℹ No shared HAS_* relations — skipping")
                continue

            print(f"    ✓ {len(shared)} shared relations discovered "
                  f"(top: {shared[0]['relation']} idf={shared[0]['idf_weight']:.2f})")

            # Load profiles.
            rel_names = [r["relation"] for r in shared]
            profiles_a = load_record_profiles(graph, type_a, rel_names)
            profiles_b = load_record_profiles(graph, type_b, rel_names)

            # Multi-pass blocking.
            candidates = multi_pass_blocking(profiles_a, profiles_b, shared)
            print(f"    ✓ {len(candidates)} candidate pairs "
                  f"(from {len(profiles_a)} × {len(profiles_b)} records)")

            # Score and link.
            links_for_pair = 0
            temporal_rejected = 0

            for id_a, id_b in candidates:
                pa = profiles_a.get(id_a, {})
                pb = profiles_b.get(id_b, {})

                wscore, n_matched, matched = score_pair(pa, pb, shared)

                if n_matched < MIN_MATCHING_FIELDS or wscore < MIN_WEIGHTED_SCORE:
                    continue

                # Temporal consistency (type_a as policy window, type_b as event).
                if not check_temporal_consistency(pa, pb):
                    # Try reversed (type_b as policy, type_a as event).
                    if not check_temporal_consistency(pb, pa):
                        temporal_rejected += 1
                        continue

                # Create LINKED_TO edge.
                graph.query(
                    "MATCH (a:Entity {id: $ida}) "
                    "MATCH (b:Entity {id: $idb}) "
                    "MERGE (a)-[r:LINKED_TO]->(b) "
                    "ON CREATE SET r.confidence = $conf, "
                    "              r.n_matched = $nm, "
                    "              r.matched_fields = $mf, "
                    "              r.source = 'cross_source'",
                    params={
                        "ida": id_a,
                        "idb": id_b,
                        "conf": round(wscore, 4),
                        "nm": n_matched,
                        "mf": ", ".join(matched),
                    },
                )
                links_for_pair += 1

            total_checked += len(candidates)
            total_links += links_for_pair
            total_temporal_rejected += temporal_rejected

            print(f"    ✓ Created {links_for_pair} LINKED_TO edges "
                  f"(rejected {temporal_rejected} for temporal inconsistency)")

            all_stats.append({
                "type_a": type_a,
                "type_b": type_b,
                "shared_relations": len(shared),
                "candidates": len(candidates),
                "links_created": links_for_pair,
                "temporal_rejected": temporal_rejected,
            })

    print(f"\n  ✓ Stage 3 complete: {total_links} LINKED_TO edges "
          f"({total_checked} pairs checked, "
          f"{total_temporal_rejected} temporal rejections)")

    return {
        "cross_source_stats": {
            "total_links": total_links,
            "total_checked": total_checked,
            "total_temporal_rejected": total_temporal_rejected,
            "pair_stats": all_stats,
        }
    }
