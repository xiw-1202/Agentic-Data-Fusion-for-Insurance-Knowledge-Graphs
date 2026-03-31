"""
SEAF-KG Stage 3 — Cross-Source Entity Linker (In-Memory)

Links structured record entities (POL-xxx, CLM-xxx) to each other when they
share enough property values, producing LINKED_TO triples that get inserted
to Neo4j alongside all other triples.

Operates entirely in-memory on the triple list — zero Neo4j queries.

Design principles:
  - Domain-agnostic: zero hardcoded field names — discovers shared relations at runtime
  - IDF-weighted scoring: rare matching values contribute more than common ones
  - Multi-pass blocking: uses highest-cardinality fields to reduce O(n²)
  - Field-type-specific comparison: tolerance for numbers, proximity for dates
  - Temporal consistency: auto-detects date ranges and rejects anachronistic links
  - Anchor field requirement: at least 1 high-cardinality match required
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_MATCHING_FIELDS = 3
MIN_WEIGHTED_SCORE = 0.6
HIGH_CARDINALITY_THRESHOLD = 50
REQUIRE_ANCHOR_MATCH = True
NUMERIC_TOLERANCE = 0.05
DATE_PROXIMITY_DAYS = 30
MAX_BLOCKING_PASSES = 2

# Only link data record types — not identity nodes or value types.
_RECORD_PREFIXES = ("POL-", "CLM-", "REC-")
_IDENTITY_PREFIXES = ("PER-", "PROP-")
_VALUE_ENTITY_TYPES = frozenset({
    "Numeric", "Date", "Text", "Categorical", "Currency", "Percentage",
})

# ISO date pattern.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_NUMERIC = re.compile(r"^[\d.,$]+$")

# Date/numeric field name hints.
_DATE_HINTS = frozenset({
    "date", "effective", "termination", "expiration", "loss",
    "construction", "cancellation", "original", "time", "period",
})
_NUMERIC_HINTS = frozenset({
    "amount", "cost", "coverage", "premium", "fee", "payment",
    "value", "deductible", "limit", "depth", "elevation", "floor",
    "count", "number", "rate",
})


# ---------------------------------------------------------------------------
# Field type detection
# ---------------------------------------------------------------------------

def classify_field_type(relation: str, sample_values: list[str]) -> str:
    """Auto-detect field type from relation name and sample values."""
    rel_lower = relation.lower()
    if any(hint in rel_lower for hint in _DATE_HINTS):
        return "date"
    if any(hint in rel_lower for hint in _NUMERIC_HINTS):
        return "numeric"
    if sample_values:
        date_count = sum(1 for v in sample_values if _ISO_DATE.match(v))
        if date_count > len(sample_values) * 0.5:
            return "date"
        numeric_count = sum(
            1 for v in sample_values
            if _NUMERIC.match(v.replace(",", "").replace("$", ""))
        )
        if numeric_count > len(sample_values) * 0.5:
            return "numeric"
    if sample_values and all(len(v) <= 10 for v in sample_values):
        return "categorical"
    return "text"


# ---------------------------------------------------------------------------
# Value comparators
# ---------------------------------------------------------------------------

def _parse_numeric(val: str) -> float | None:
    try:
        return float(val.replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def _parse_date(val: str) -> str | None:
    m = _ISO_DATE.match(val.strip())
    return m.group(0) if m else None


def compare_values(val_a: str, val_b: str, field_type: str) -> float:
    """Field-type-specific comparison. Returns 0.0-1.0 match score."""
    if field_type == "categorical":
        return 1.0 if val_a.strip() == val_b.strip() else 0.0
    if field_type == "numeric":
        na, nb = _parse_numeric(val_a), _parse_numeric(val_b)
        if na is not None and nb is not None:
            denom = max(abs(na), abs(nb), 1e-9)
            return 1.0 if abs(na - nb) / denom <= NUMERIC_TOLERANCE else 0.0
        return 1.0 if val_a.strip() == val_b.strip() else 0.0
    if field_type == "date":
        da, db = _parse_date(val_a), _parse_date(val_b)
        if da and db:
            if da == db:
                return 1.0
            try:
                dt_a = datetime.strptime(da, "%Y-%m-%d")
                dt_b = datetime.strptime(db, "%Y-%m-%d")
                if abs((dt_a - dt_b).days) <= DATE_PROXIMITY_DAYS:
                    return 0.5
            except ValueError:
                pass
            return 0.0
        return 1.0 if val_a.strip() == val_b.strip() else 0.0
    return 1.0 if val_a.strip().lower() == val_b.strip().lower() else 0.0


# ---------------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------------

def check_temporal_consistency(
    profile_a: dict[str, str],
    profile_b: dict[str, str],
) -> bool:
    """Check if date fields are temporally consistent."""
    def _find_date(profile: dict, hints: list[str]) -> str | None:
        for rel, val in profile.items():
            if any(h in rel.lower() for h in hints):
                parsed = _parse_date(val)
                if parsed:
                    return parsed
        return None

    eff = _find_date(profile_a, ["effective", "eff_"])
    exp = _find_date(profile_a, ["expiration", "exp_", "termination"])
    loss = _find_date(profile_b, ["loss", "event", "claim", "open"])

    if not loss or not eff:
        return True
    try:
        dt_loss = datetime.strptime(loss, "%Y-%m-%d")
        dt_eff = datetime.strptime(eff, "%Y-%m-%d")
        if dt_loss < dt_eff:
            return False
        if exp:
            dt_exp = datetime.strptime(exp, "%Y-%m-%d")
            if dt_loss > dt_exp:
                return False
    except ValueError:
        return True
    return True


# ---------------------------------------------------------------------------
# In-memory shared relation discovery (from triple list)
# ---------------------------------------------------------------------------

def discover_shared_relations_from_triples(
    triples: list[dict],
    type_a: str,
    type_b: str,
) -> list[dict]:
    """Discover HAS_* relations shared by both record types from the triple list.

    No Neo4j queries — builds stats directly from triples in memory.
    """
    # Build per-type relation stats: {relation: {values: set, n_records: set}}
    def _build_stats(entity_type: str) -> dict[str, dict]:
        stats: dict[str, dict] = {}
        for t in triples:
            if t.get("source_type") != "structured":
                continue
            # Check if subject is of this type (via IS_A triple or prefix).
            # We need a mapping of entity_id → entity_type.
            pass
        return stats

    # First pass: build entity_id → entity_type map from IS_A triples.
    entity_types: dict[str, str] = {}
    for t in triples:
        if t["relation"] == "IS_A":
            entity_types[t["subject"]] = t["object"]

    # Second pass: collect relation stats per type.
    stats_a: dict[str, dict[str, Any]] = defaultdict(lambda: {"values": set(), "records": set()})
    stats_b: dict[str, dict[str, Any]] = defaultdict(lambda: {"values": set(), "records": set()})

    for t in triples:
        if not t["relation"].startswith("HAS_"):
            continue
        subj = t["subject"]
        et = entity_types.get(subj, "")
        if et == type_a:
            stats_a[t["relation"]]["values"].add(t["object"])
            stats_a[t["relation"]]["records"].add(subj)
        elif et == type_b:
            stats_b[t["relation"]]["values"].add(t["object"])
            stats_b[t["relation"]]["records"].add(subj)

    # Intersection of relations.
    shared = set(stats_a.keys()) & set(stats_b.keys())

    result = []
    for rel in shared:
        sa, sb = stats_a[rel], stats_b[rel]
        card_a = len(sa["values"])
        card_b = len(sb["values"])
        n_records_a = len(sa["records"])
        n_records_b = len(sb["records"])

        avg_card = (card_a + card_b) / 2
        max_records = max(n_records_a, n_records_b, 1)
        selectivity = avg_card / max_records
        idf = math.log(1 + selectivity * 10) + 0.1

        samples = list(sa["values"])[:5] + list(sb["values"])[:5]
        field_type = classify_field_type(rel, samples)

        result.append({
            "relation": rel,
            "cardinality_a": card_a,
            "cardinality_b": card_b,
            "idf_weight": max(idf, 0.1),
            "field_type": field_type,
            "sample_values": samples,
        })

    result.sort(key=lambda x: x["idf_weight"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# In-memory record profile loading (from triple list)
# ---------------------------------------------------------------------------

def load_record_profiles_from_triples(
    triples: list[dict],
    entity_type: str,
    relations: list[str],
) -> dict[str, dict[str, str]]:
    """Build {record_id: {relation: value}} from the triple list. No Neo4j."""
    # Build entity_id → entity_type map.
    entity_types: dict[str, str] = {}
    for t in triples:
        if t["relation"] == "IS_A":
            entity_types[t["subject"]] = t["object"]

    rel_set = set(relations)
    profiles: dict[str, dict[str, str]] = defaultdict(dict)

    for t in triples:
        if t["relation"] not in rel_set:
            continue
        subj = t["subject"]
        if entity_types.get(subj) == entity_type:
            profiles[subj][t["relation"]] = t["object"]

    return dict(profiles)


# ---------------------------------------------------------------------------
# Multi-pass blocking
# ---------------------------------------------------------------------------

def multi_pass_blocking(
    profiles_a: dict[str, dict[str, str]],
    profiles_b: dict[str, dict[str, str]],
    shared_rels: list[dict],
) -> set[tuple[str, str]]:
    """Multi-pass blocking on top-N highest-cardinality fields."""
    candidate_pairs: set[tuple[str, str]] = set()
    blocking_keys = [r["relation"] for r in shared_rels[:MAX_BLOCKING_PASSES]]

    if not blocking_keys:
        for id_a in profiles_a:
            for id_b in profiles_b:
                candidate_pairs.add((id_a, id_b))
        return candidate_pairs

    for key in blocking_keys:
        blocks_a: dict[str, list[str]] = defaultdict(list)
        for nid, props in profiles_a.items():
            val = props.get(key, "")
            if val:
                blocks_a[val].append(nid)
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
    """IDF-weighted pair scoring with field-type-specific comparators."""
    all_weight = sum(r["idf_weight"] for r in shared_rels)
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
            continue

        match_score = compare_values(val_a, val_b, ftype)
        if match_score > 0:
            matched_weight += weight * match_score
            n_matched += 1
            matched_fields.append(rel)

    total_weight = all_weight
    if total_weight == 0:
        return 0.0, 0, []

    weighted_score = matched_weight / total_weight

    if REQUIRE_ANCHOR_MATCH:
        has_anchor = any(
            rel_info["relation"] in matched_fields
            and max(rel_info.get("cardinality_a", 0), rel_info.get("cardinality_b", 0))
            >= HIGH_CARDINALITY_THRESHOLD
            for rel_info in shared_rels
        )
        if not has_anchor:
            return 0.0, 0, []

    return weighted_score, n_matched, matched_fields


# ---------------------------------------------------------------------------
# LangGraph node (in-memory — no Neo4j queries)
# ---------------------------------------------------------------------------

def cross_source_link(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: in-memory cross-source entity linking.

    Reads triples from state, discovers shared fields, scores pairs,
    and appends LINKED_TO triples to the list. Zero Neo4j queries.

    Pipeline order: runs BEFORE insert_to_neo4j.
    """
    print("\n[4.7] Cross-Source Entity Linking — SEAF-KG Stage 3 (in-memory)")

    triples = state.get("triples", [])
    if not triples:
        return {"cross_source_stats": {"links_created": 0}}

    # Step 1: Find record entity types from IS_A triples.
    entity_types: dict[str, str] = {}
    type_counts: dict[str, int] = defaultdict(int)
    for t in triples:
        if t["relation"] == "IS_A" and t.get("source_type") == "structured":
            entity_types[t["subject"]] = t["object"]
            type_counts[t["object"]] += 1

    # Filter to data record types only (POL-/CLM-/REC- prefixed).
    record_types: dict[str, int] = {}
    skipped: list[str] = []
    for et, cnt in type_counts.items():
        if et in _VALUE_ENTITY_TYPES or et in ("RecordType", "IdentityType"):
            skipped.append(f"{et}({cnt})")
            continue
        # Check if any entity of this type has a record prefix.
        has_record_prefix = any(
            eid.startswith(_RECORD_PREFIXES)
            for eid, etype in entity_types.items()
            if etype == et
        )
        if has_record_prefix:
            record_types[et] = cnt
        else:
            skipped.append(f"{et}({cnt})")

    if skipped:
        print(f"  ℹ Skipped non-record types: {', '.join(skipped)}")

    if len(record_types) < 2:
        print(f"  ℹ Only {len(record_types)} record type(s) — need ≥2 for linking")
        return {"cross_source_stats": {"types_found": len(record_types), "links_created": 0}}

    print(f"  ✓ Found {len(record_types)} record types: "
          f"{', '.join(f'{t} ({c})' for t, c in record_types.items())}")

    total_links = 0
    total_checked = 0
    total_temporal_rejected = 0
    new_triples: list[dict] = []

    # Step 2: For each pair of record types, discover and link.
    type_list = list(record_types.keys())
    for i in range(len(type_list)):
        for j in range(i + 1, len(type_list)):
            type_a, type_b = type_list[i], type_list[j]

            print(f"\n  --- Linking {type_a} ↔ {type_b} ---")

            shared = discover_shared_relations_from_triples(triples, type_a, type_b)
            if not shared:
                print(f"    ℹ No shared HAS_* relations — skipping")
                continue

            print(f"    ✓ {len(shared)} shared relations "
                  f"(top: {shared[0]['relation']} idf={shared[0]['idf_weight']:.2f})")

            rel_names = [r["relation"] for r in shared]
            profiles_a = load_record_profiles_from_triples(triples, type_a, rel_names)
            profiles_b = load_record_profiles_from_triples(triples, type_b, rel_names)

            candidates = multi_pass_blocking(profiles_a, profiles_b, shared)
            print(f"    ✓ {len(candidates)} candidate pairs "
                  f"(from {len(profiles_a)} × {len(profiles_b)} records)")

            links_for_pair = 0
            temporal_rejected = 0

            for id_a, id_b in candidates:
                pa = profiles_a.get(id_a, {})
                pb = profiles_b.get(id_b, {})

                wscore, n_matched, matched = score_pair(pa, pb, shared)

                if n_matched < MIN_MATCHING_FIELDS or wscore < MIN_WEIGHTED_SCORE:
                    continue

                if not check_temporal_consistency(pa, pb):
                    if not check_temporal_consistency(pb, pa):
                        temporal_rejected += 1
                        continue

                new_triples.append({
                    "subject": id_a,
                    "subject_type": type_a,
                    "relation": "LINKED_TO",
                    "object": id_b,
                    "object_type": type_b,
                    "span": f"cross-source link ({n_matched} fields, score={wscore:.2f})",
                    "confidence": round(wscore, 4),
                    "chunk_id": "cross_source",
                    "source": "cross_source_linker",
                    "source_type": "cross_source",
                })
                links_for_pair += 1

            total_checked += len(candidates)
            total_links += links_for_pair
            total_temporal_rejected += temporal_rejected

            print(f"    ✓ {links_for_pair} LINKED_TO triples created "
                  f"(rejected {temporal_rejected} temporal)")

    # Append new triples to the existing list.
    updated_triples = triples + new_triples

    print(f"\n  ✓ Stage 3 complete: {total_links} LINKED_TO triples added "
          f"({total_checked} pairs checked, "
          f"{total_temporal_rejected} temporal rejections)")

    return {
        "triples": updated_triples,
        "cross_source_stats": {
            "total_links": total_links,
            "total_checked": total_checked,
            "total_temporal_rejected": total_temporal_rejected,
        },
    }
