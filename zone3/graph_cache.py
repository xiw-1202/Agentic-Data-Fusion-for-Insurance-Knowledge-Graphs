"""Zone 3 — Graph Cache: Load entities/relationships from local JSON.

Eliminates Neo4j round-trips for Zone 3 ontology induction.
Builds entity dicts from zone2_run_summary.json triples, matching the
exact shapes each Zone 3 method expects.

Usage:
    from zone3.graph_cache import load_cached_entities, export_graph_cache

    # Export after Zone 2 (or build from existing zone2_run_summary.json):
    export_graph_cache()

    # Load for any Zone 3 method:
    entities = load_cached_entities(format="sv_loi")  # or "leiden", "rsi_lcr"
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Entity types that represent data values, not domain concepts.
# Used as FALLBACK floor — the dynamic classifier adds to this set.
VALUE_ENTITY_TYPES = frozenset({
    "Numeric", "Date", "Text", "Categorical",
    "RecordType", "IdentityType",
    "FinancialAmount", "TimePeriod",
})

# Structured entity prefixes — records from CSV ingestion.
STRUCTURED_PREFIXES = ("POL-", "CLM-", "REC-", "PER-", "PROP-")

# Keywords in type names that strongly indicate value types
_VALUE_TYPE_KEYWORDS = frozenset({
    "amount", "date", "numeric", "currency", "code", "score",
    "percentage", "number", "count", "rate", "duration", "period",
    "text", "categorical", "boolean", "flag", "status", "field",
    "value", "id", "identifier", "index", "timestamp",
})

# Keywords that protect a type from being classified as value
_CONCEPT_TYPE_KEYWORDS = frozenset({
    "policy", "claim", "person", "organization", "company",
    "property", "building", "coverage", "product", "risk",
    "peril", "hazard", "damage", "exclusion", "structure",
    "agent", "insured", "beneficiary", "location", "address",
})

import re as _re
_LITERAL_PATTERNS = [
    _re.compile(r"^\$[\d,]+\.?\d*$"),      # Currency
    _re.compile(r"^\d{4}-\d{2}-\d{2}$"),    # ISO date
    _re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),  # US date
    _re.compile(r"^-?[\d,]+\.?\d*%?$"),      # Number/percentage
    _re.compile(r"^[A-Z0-9]{1,5}$"),         # Short code
]

# Dynamically computed set — filled by classify_entity_types()
_dynamic_value_types: frozenset[str] = frozenset()
_entity_role_map: dict[str, str] = {}


def classify_entity_types(
    entities: dict[str, dict],
    out_rels: dict[str, list],
    in_rels: dict[str, list],
) -> tuple[frozenset[str], dict[str, str]]:
    """Data-driven classification of entity types into roles.

    Returns:
        (value_types, role_map) where:
        - value_types: frozenset of entity type names classified as values
        - role_map: {entity_type: role} where role is one of:
            "anchor" (high connectivity), "concept" (moderate),
            "descriptor" (low, mostly incoming), "value" (literal patterns),
            "record" (structured prefix)
    """
    global _dynamic_value_types, _entity_role_map

    # Group entities by type
    type_groups: dict[str, list[str]] = defaultdict(list)
    for eid, edata in entities.items():
        etype = edata.get("entity_type", "Unknown")
        type_groups[etype].append(eid)

    value_types: set[str] = set(VALUE_ENTITY_TYPES)  # start with hardcoded floor
    role_map: dict[str, str] = {}

    for etype, members in type_groups.items():
        if etype in VALUE_ENTITY_TYPES:
            role_map[etype] = "value"
            continue

        # Compute per-type statistics
        n = len(members)
        etype_lower = etype.lower().replace("_", "")

        # Name pattern score: fraction of members matching literal patterns
        literal_count = 0
        for eid in members:
            for pat in _LITERAL_PATTERNS:
                if pat.match(eid):
                    literal_count += 1
                    break

        name_pattern_score = literal_count / n if n > 0 else 0

        # Relation diversity: distinct relation types across all members
        rel_types: set[str] = set()
        total_out_degree = 0
        for eid in members:
            for rel in out_rels.get(eid, []):
                rel_types.add(rel["rel"])
                total_out_degree += 1
            for rel in in_rels.get(eid, []):
                rel_types.add(rel["rel"])

        relation_diversity = len(rel_types)
        avg_out_degree = total_out_degree / n if n > 0 else 0

        # Keyword matching
        has_value_keyword = any(kw in etype_lower for kw in _VALUE_TYPE_KEYWORDS)
        has_concept_keyword = any(kw in etype_lower for kw in _CONCEPT_TYPE_KEYWORDS)

        # Classification logic (precision-first to avoid collapsing real entities)
        if has_concept_keyword:
            # Protected concept type — never classify as value
            if avg_out_degree >= 3 and relation_diversity >= 5:
                role_map[etype] = "anchor"
            else:
                role_map[etype] = "concept"
        elif name_pattern_score > 0.6:
            # Majority of members look like literal values
            value_types.add(etype)
            role_map[etype] = "value"
        elif has_value_keyword and relation_diversity < 4:
            # Type name suggests value AND low relation diversity
            value_types.add(etype)
            role_map[etype] = "value"
        elif relation_diversity < 3 and avg_out_degree < 1.0 and name_pattern_score > 0.3:
            # Low connectivity + some literal patterns
            value_types.add(etype)
            role_map[etype] = "value"
        elif avg_out_degree < 0.5 and relation_diversity < 3:
            # Very low connectivity — descriptor at best
            role_map[etype] = "descriptor"
        elif avg_out_degree >= 3 and relation_diversity >= 5:
            role_map[etype] = "anchor"
        else:
            role_map[etype] = "concept"

    result_types = frozenset(value_types)
    _dynamic_value_types = result_types
    _entity_role_map = role_map

    # Log classification
    role_counts: dict[str, int] = defaultdict(int)
    for r in role_map.values():
        role_counts[r] += 1
    print(f"  Entity type classification: {dict(role_counts)}")
    n_dynamic = len(result_types) - len(VALUE_ENTITY_TYPES)
    if n_dynamic > 0:
        new_values = result_types - VALUE_ENTITY_TYPES
        print(f"    +{n_dynamic} dynamic value types: {sorted(new_values)[:10]}{'...' if n_dynamic > 10 else ''}")

    return result_types, role_map

CACHE_FILENAME = "zone3_graph_cache.json"


def _cache_path(results_dir: str | None = None) -> str:
    return os.path.join(results_dir or config.RESULTS_DIR, CACHE_FILENAME)


def _summary_path(results_dir: str | None = None) -> str:
    return os.path.join(results_dir or config.RESULTS_DIR, "zone2_run_summary.json")


# ---------------------------------------------------------------------------
# Build cache from zone2_run_summary.json triples
# ---------------------------------------------------------------------------

def _build_entity_graph(triples: list[dict]) -> dict:
    """Build adjacency structures from raw triples.

    Returns:
        {
            "entities": {eid: {"entity_type": str}},
            "out_rels": {eid: [{"rel": str, "target": str, "target_type": str, "chunk_id": str}]},
            "in_rels":  {eid: [{"rel": str, "source": str, "source_type": str, "chunk_id": str}]},
        }
    """
    entities: dict[str, dict] = {}
    out_rels: dict[str, list[dict]] = defaultdict(list)
    in_rels: dict[str, list[dict]] = defaultdict(list)

    for t in triples:
        subj = t["subject"]
        obj = t["object"]
        rel = t["relation"]
        chunk_id = t.get("chunk_id", "")

        subj_type = t.get("subject_type", "Unknown")
        obj_type = t.get("object_type", "Unknown")

        # Register entities — concept types override value types.
        # Structured triples come first in the list (zone2 prepends them),
        # so without this priority logic, VALUE types (Numeric, Date, Text)
        # would lock in and block concept types from LLM extraction.
        if subj not in entities:
            entities[subj] = {"entity_type": subj_type}
        elif (subj_type not in VALUE_ENTITY_TYPES
              and entities[subj]["entity_type"] in VALUE_ENTITY_TYPES):
            entities[subj]["entity_type"] = subj_type

        if obj not in entities:
            entities[obj] = {"entity_type": obj_type}
        elif (obj_type not in VALUE_ENTITY_TYPES
              and entities[obj]["entity_type"] in VALUE_ENTITY_TYPES):
            entities[obj]["entity_type"] = obj_type

        out_rels[subj].append({
            "rel": rel,
            "target": obj,
            "target_type": obj_type,
            "chunk_id": chunk_id,
        })
        in_rels[obj].append({
            "rel": rel,
            "source": subj,
            "source_type": subj_type,
            "chunk_id": chunk_id,
        })

    return {"entities": entities, "out_rels": dict(out_rels), "in_rels": dict(in_rels)}


def _load_header_expansions(results_dir: str | None = None) -> dict[str, str]:
    """Load all header expansion caches and build relation→human_name mapping.

    Header expansions are generated by Zone 1 ingestion (_expand_csv_headers)
    and saved as header_expansions_*.json in the data processed directory.
    We convert them to relation→human_name for use in entity typing prompts.
    """
    import glob

    rdir = results_dir or config.RESULTS_DIR
    # Look for header expansions in common locations
    search_paths = [
        os.path.join(rdir, "..", "processed", "header_expansions_*.json"),
        os.path.join(rdir, "header_expansions_*.json"),
    ]
    # Also check DATA_DIR patterns
    for data_dir in ["data/flood/processed", "data/Emory_Spring2026/processed"]:
        search_paths.append(os.path.join(data_dir, "header_expansions_*.json"))

    header_map: dict[str, str] = {}
    files_loaded = 0

    for pattern in search_paths:
        for filepath in glob.glob(pattern):
            try:
                with open(filepath) as f:
                    expansions = json.load(f)
                for abbrev, expansion in expansions.items():
                    # Convert field name to relation name format
                    rel_name = f"HAS_{abbrev.upper().replace(' ', '_')}"
                    header_map[rel_name] = expansion
                files_loaded += 1
            except (json.JSONDecodeError, OSError):
                continue

    # Also add common-sense humanizations for standard relation names
    _COMMON_HUMANIZATIONS = {
        "HAS_AGCY_NAME": "agency name",
        "HAS_ACCT_NAME": "account name",
        "HAS_AS_OF_AGCY_NAME": "agency name (as of date)",
        "HAS_AS_OF_ACCT_NAME": "account name (as of date)",
        "HAS_NPS_SCORE": "NPS score",
        "HAS_NPS_CATEGORY": "NPS category",
        "HAS_CSAT": "customer satisfaction",
        "HAS_CES": "customer effort score",
        "HAS_LTR_SCORE": "likelihood to recommend",
        "IS_A": "is a type of",
        "ABOUT": "is about",
        "COVERS": "covers",
        "EXCLUDED_FROM": "excluded from",
    }
    for rel, human in _COMMON_HUMANIZATIONS.items():
        if rel not in header_map:
            header_map[rel] = human

    if header_map:
        print(f"  ✓ Loaded {len(header_map)} header→human mappings "
              f"({files_loaded} expansion files)")

    return header_map


def _humanize_relation(rel: str, header_map: dict[str, str]) -> str:
    """Convert a relation name to human-readable using header map.

    HAS_AGCY_NAME → 'agency name' (if in map)
    HAS_POLICY_EFFECTIVE_DATE → 'policy effective date' (fallback: strip HAS_ + lowercase)
    """
    if rel in header_map:
        return header_map[rel]
    # Fallback: strip HAS_ prefix and convert underscores to spaces
    human = rel.removeprefix("HAS_").lower().replace("_", " ")
    return human


def export_graph_cache(triples: list[dict] | None = None,
                       results_dir: str | None = None) -> str:
    """Build and save the graph cache JSON.

    If triples is None, loads from zone2_run_summary.json.
    Returns the path to the saved cache file.
    """
    if triples is None:
        sp = _summary_path(results_dir)
        if not os.path.exists(sp):
            raise FileNotFoundError(
                f"No zone2_run_summary.json at {sp}. "
                "Run zone2/pipeline.py first or provide triples."
            )
        with open(sp) as f:
            summary = json.load(f)
        triples = summary["triples"]

    # Apply professor's cardinality rule: collapse single-use literal values
    # to node properties BEFORE building the entity graph. This ensures Zone 3
    # never sees ghost entities that were collapsed to properties in Neo4j.
    from zone2.pipeline import collapse_value_to_properties
    triples, node_properties = collapse_value_to_properties(triples)

    # Load header expansions from Zone 1 (humanized column names for typing prompts)
    header_map = _load_header_expansions(results_dir)

    graph = _build_entity_graph(triples)

    # Run data-driven entity type classification
    value_types, role_map = classify_entity_types(
        graph["entities"], graph["out_rels"], graph["in_rels"],
    )

    # Serialize to cache
    rdir = results_dir or config.RESULTS_DIR
    os.makedirs(rdir, exist_ok=True)

    cache = {
        "entity_count": len(graph["entities"]),
        "triple_count": len(triples),
        "normalization_version": 3,  # v3: data-driven measure/dimension/ID collapse
        "built_from_raw_summary": True,
        "collapse_manifest": {
            "triples_collapsed": sum(len(v) for v in node_properties.values()),
            "entities_with_properties": len(node_properties),
        },
        "entities": graph["entities"],
        "out_rels": graph["out_rels"],
        "in_rels": graph["in_rels"],
        "node_properties": node_properties,
        "header_map": header_map,
        "value_entity_types": sorted(value_types),
        "entity_role_map": role_map,
    }

    path = _cache_path(results_dir)
    with open(path, "w") as f:
        json.dump(cache, f)

    print(f"  ✓ Graph cache saved: {path}")
    print(f"    {cache['entity_count']} entities, {cache['triple_count']} triples")
    print(f"    {cache['collapse_manifest']['triples_collapsed']} values collapsed to properties "
          f"on {cache['collapse_manifest']['entities_with_properties']} entities")
    return path


def _load_raw_cache(results_dir: str | None = None) -> dict:
    """Load raw cache dict, building from zone2_run_summary.json if needed.

    Staleness check: if zone2_run_summary.json is newer than the cache,
    or if the cache lacks normalization_version (built by old code),
    rebuild the cache to pick up property collapse and header expansions.
    """
    global _dynamic_value_types, _entity_role_map

    path = _cache_path(results_dir)
    summary_path = _summary_path(results_dir)

    needs_rebuild = False
    if not os.path.exists(path):
        needs_rebuild = True
        print("  [cache] No graph cache found, building from zone2_run_summary.json...")
    elif os.path.exists(summary_path):
        cache_mtime = os.path.getmtime(path)
        summary_mtime = os.path.getmtime(summary_path)
        if summary_mtime > cache_mtime:
            needs_rebuild = True
            print("  [cache] zone2_run_summary.json is newer than cache — rebuilding...")
        else:
            # Also rebuild if cache version doesn't match current code
            CURRENT_VERSION = 3  # must match export_graph_cache()
            with open(path) as f:
                peek = json.load(f)
            cache_version = peek.get("normalization_version", 0)
            if cache_version < CURRENT_VERSION:
                needs_rebuild = True
                print(f"  [cache] Cache version {cache_version} < {CURRENT_VERSION} — rebuilding...")
            else:
                # Cache is fresh and has correct version
                if "value_entity_types" in peek:
                    _dynamic_value_types = frozenset(peek["value_entity_types"])
                    _entity_role_map = peek.get("entity_role_map", {})
                elif not _dynamic_value_types:
                    vt, rm = classify_entity_types(
                        peek["entities"], peek.get("out_rels", {}), peek.get("in_rels", {}),
                    )
                    _dynamic_value_types = vt
                    _entity_role_map = rm
                return peek

    if needs_rebuild:
        export_graph_cache(results_dir=results_dir)

    with open(path) as f:
        cache = json.load(f)
    if "value_entity_types" in cache:
        _dynamic_value_types = frozenset(cache["value_entity_types"])
        _entity_role_map = cache.get("entity_role_map", {})
    return cache


# ---------------------------------------------------------------------------
# Format-specific loaders (match what each Zone 3 method expects)
# ---------------------------------------------------------------------------

def load_cached_entities(
    fmt: Literal["leiden", "rsi_lcr", "sv_loi"] = "leiden",
    results_dir: str | None = None,
) -> list[dict]:
    """Load entities from cache in the format expected by each Zone 3 method.

    Args:
        fmt: Target method format.
            "leiden"  — matches zone3/pipeline.py load_entities() output
            "rsi_lcr" — matches zone3/rsi_lcr.py load_entities() output
            "sv_loi"  — matches zone3/sv_loi.py load_entities() output
        results_dir: Custom results directory. Default: config.RESULTS_DIR.
    """
    cache = _load_raw_cache(results_dir)
    ent_map = cache["entities"]
    out_map = cache["out_rels"]
    in_map = cache["in_rels"]

    entities: list[dict] = []

    for eid, edata in ent_map.items():
        etype = edata.get("entity_type", "Unknown") or "Unknown"
        out_rels = out_map.get(eid, [])
        in_rels = in_map.get(eid, [])

        # Filter null rels
        out_rels = [r for r in out_rels if r.get("rel")]
        in_rels = [r for r in in_rels if r.get("rel")]

        if fmt == "leiden":
            rel_types_out = [r["rel"] for r in out_rels]
            rel_types_in = [r["rel"] for r in in_rels]
            chunks = list(set(
                r.get("chunk_id", "") for r in out_rels + in_rels if r.get("chunk_id")
            ))
            entities.append({
                "id": eid,
                "entity_type": etype,
                "relations_out": rel_types_out,
                "relations_in": rel_types_in,
                "all_relation_types": list(set(rel_types_out + rel_types_in)),
                "neighbors": list(set(
                    [r["target"] for r in out_rels if r.get("target")] +
                    [r["source"] for r in in_rels if r.get("source")]
                )),
                "chunks": chunks,
            })

        elif fmt == "rsi_lcr":
            out_counts: dict[str, int] = defaultdict(int)
            in_counts: dict[str, int] = defaultdict(int)
            for r in out_rels:
                out_counts[r["rel"]] += 1
            for r in in_rels:
                in_counts[r["rel"]] += 1
            entities.append({
                "id": eid,
                "entity_type": etype,
                "out_rel_counts": dict(out_counts),
                "in_rel_counts": dict(in_counts),
                "out_rel_types": list(out_counts.keys()),
                "in_rel_types": list(in_counts.keys()),
                "degree": len(out_rels) + len(in_rels),
            })

        elif fmt == "sv_loi":
            out_counts: dict[str, int] = defaultdict(int)
            in_counts: dict[str, int] = defaultdict(int)
            for r in out_rels:
                out_counts[r["rel"]] += 1
            for r in in_rels:
                in_counts[r["rel"]] += 1

            # Humanize relation names for typing prompts (A3 fix)
            hmap = cache.get("header_map", {})
            out_summary = []
            for r in out_rels[:5]:
                label = _humanize_relation(r["rel"], hmap) if hmap else r["rel"]
                out_summary.append(f"--{label}--> {r.get('target', '?')}")
            in_summary = []
            for r in in_rels[:5]:
                label = _humanize_relation(r["rel"], hmap) if hmap else r["rel"]
                in_summary.append(f"<--{label}-- {r.get('source', '?')}")

            entities.append({
                "id": eid,
                "entity_type": etype,
                "out_rels": out_rels,
                "in_rels": in_rels,
                "out_summary": out_summary,
                "in_summary": in_summary,
                "out_rel_counts": dict(out_counts),
                "in_rel_counts": dict(in_counts),
                "degree": len(out_rels) + len(in_rels),
            })

    typed = sum(1 for e in entities if e["entity_type"] != "Unknown")
    print(f"  ✓ {len(entities)} entities loaded from cache ({typed} typed)")
    return entities


def is_concept_entity(entity: dict) -> bool:
    """Return True if entity is a domain concept (not a value or record).

    Uses the dynamically computed _dynamic_value_types set if available
    (populated by classify_entity_types during cache export), falling back
    to the hardcoded VALUE_ENTITY_TYPES floor.
    """
    eid = entity["id"]
    etype = entity.get("entity_type", "Unknown")

    # Structured records (POL-xxx, CLM-xxx) are not concepts
    if eid.startswith(STRUCTURED_PREFIXES):
        return False

    # Use dynamic value types if available, otherwise hardcoded floor
    active_value_types = _dynamic_value_types if _dynamic_value_types else VALUE_ENTITY_TYPES
    if etype in active_value_types:
        return False

    # Name-pattern filters: dollar amounts, bare numbers, zip codes
    if eid.startswith("$"):
        return False
    stripped = eid.lstrip("-").replace(".", "", 1).replace(",", "")
    if stripped.isdigit():
        return False

    return True


def get_entity_lane(entity: dict) -> str:
    """Classify entity into one of three processing lanes.

    Returns:
        "concept"  — domain concepts from PDF (drive class discovery + typing)
        "record"   — structured CSV records (POL-xxx, CLM-xxx, PROP-xxx)
        "value"    — data values (dates, amounts, codes, categories)
    """
    eid = entity["id"]
    if eid.startswith(STRUCTURED_PREFIXES):
        return "record"
    if not is_concept_entity(entity):
        return "value"
    return "concept"


def get_concept_entities(entities: list[dict]) -> list[dict]:
    """Filter to only domain concept entities (for SV-LOI class discovery)."""
    concepts = [e for e in entities if get_entity_lane(e) == "concept"]
    print(f"  ✓ {len(concepts)} concept entities (of {len(entities)} total)")
    return concepts


# ---------------------------------------------------------------------------
# CLI: build cache standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building Zone 3 graph cache from zone2_run_summary.json...")
    path = export_graph_cache()
    print(f"\nDone. Cache at: {path}")

    # Quick stats
    with open(path) as f:
        cache = json.load(f)
    n_ent = cache["entity_count"]
    n_tri = cache["triple_count"]

    # Count concept vs value entities
    concept_count = sum(
        1 for eid, edata in cache["entities"].items()
        if not eid.startswith(STRUCTURED_PREFIXES)
        and edata.get("entity_type", "Unknown") not in VALUE_ENTITY_TYPES
    )
    print(f"\nEntity breakdown:")
    print(f"  Total: {n_ent}")
    print(f"  Concept entities: {concept_count}")
    print(f"  Value/record entities: {n_ent - concept_count}")
