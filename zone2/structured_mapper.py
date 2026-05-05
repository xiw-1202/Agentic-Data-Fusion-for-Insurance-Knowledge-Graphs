"""
SEAF-KG Stage 1 — Deterministic Structured Mapper

Detects CSV-derived chunks, parses individual records from batched text,
and generates deterministic triples. Each record becomes a first-class
entity node with typed property relations.

Design principles:
  - Domain-agnostic: works on any CSV structure (flood, auto, health, etc.)
  - Deterministic: no LLM calls, confidence=1.0 for all triples
  - Record-preserving: each CSV row → one entity node with composite key
  - Compatible: same triple format as LLM extraction for seamless merging
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Salient fields for composite key generation per record type.
# Order matters — first 4 non-empty fields are used.
# These field names are NFIP-specific (flood LOB). For other LOBs the salient
# fields won't match, which is fine: generate_composite_key() falls back to
# hashing the first N non-empty fields alphabetically, producing stable keys
# for any CSV structure without requiring configuration changes.
_KEY_FIELDS: dict[str, list[str]] = {
    "policies": [
        "policy effective date",
        "total building insurance coverage",
        "rated flood zone",
        "occupancy type",
        "property state",
        "policy cost",
    ],
    "claims": [
        "date of loss",
        "cause of damage",
        "total building insurance coverage",
        "rated flood zone",
        "amount paid on building claim",
        "property state",
    ],
}

# How many salient fields to hash for the composite key.
_KEY_FIELD_COUNT = 4

# Patterns for value type inference.
_DATE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}"  # ISO date prefix
)
_CURRENCY_PATTERN = re.compile(
    r"^\$[\d,]+(\.\d{2})?$"
)
_NUMERIC_PATTERN = re.compile(
    r"^-?[\d,]+(\.\d+)?$"
)
_PERCENTAGE_PATTERN = re.compile(
    r"^[\d.]+\s*%$"
)

# Fields whose values are dates even without ISO format.
_DATE_FIELD_HINTS = frozenset({
    "date", "effective", "termination", "expiration", "loss",
    "construction", "cancellation", "original",
})

# Record block delimiter in chunk text.
_RECORD_DELIM = re.compile(r"(?:^|\n)RECORD:\s*\n", re.MULTILINE)

# Group line: [GroupName] field: value | field: value
_GROUP_LINE = re.compile(
    r"^\s*\[([^\]]+)\]\s*(.+)$"
)


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

def is_structured_chunk(chunk: dict[str, Any]) -> bool:
    """Detect CSV-derived chunks via source path and section_hierarchy.

    CSV chunks have:
      - `.json` or `.csv` in the source path
      - "records" in section_hierarchy (data batches, not schema)
    """
    source = chunk.get("source", "")
    hierarchy = chunk.get("section_hierarchy", [])
    hierarchy_str = " ".join(str(h) for h in hierarchy).lower()

    is_structured_source = (
        ".json" in source.lower() or ".csv" in source.lower()
    )
    is_data_batch = "records" in hierarchy_str  # not "schema"

    return is_structured_source and is_data_batch


def is_schema_chunk(chunk: dict[str, Any]) -> bool:
    """Detect schema description chunks (chunk 0 of each CSV source)."""
    source = chunk.get("source", "")
    hierarchy = chunk.get("section_hierarchy", [])
    hierarchy_str = " ".join(str(h) for h in hierarchy).lower()

    is_structured_source = (
        ".json" in source.lower() or ".csv" in source.lower()
    )
    return is_structured_source and "schema" in hierarchy_str


# Source role: "entity" records are first-class domain objects (policies, claims).
# "observation" records are data ABOUT entities (surveys, emails, chats).
_SOURCE_ROLES: dict[str, str] = {
    "policies": "entity",
    "claims": "entity",
    "survey": "observation",
    "chat": "observation",
    "email": "observation",
}

# Prefix mapping for record key generation (extensible for LLM-detected types).
_RECORD_PREFIX: dict[str, str] = {
    "policies": "POL",
    "claims": "CLM",
    "survey": "SRV",
    "chat": "CHAT",
    "email": "EMAIL",
}

# Entity type mapping for record triples.
_ENTITY_TYPE_MAP: dict[str, str] = {
    "policies": "PolicyRecord",
    "claims": "ClaimRecord",
    "survey": "SurveyRecord",
    "chat": "ChatRecord",
    "email": "EmailRecord",
}

# Root ontology class — top of the IS_A chain.
_ONTOLOGY_ROOT: str = "Record"

# Class-level subject/object marker (Phase 2 hierarchy triples).
_ONTOLOGY_CLASS_TYPE: str = "OntologyClass"


def _lob_entity_type(record_type: str, lob: str) -> str:
    """Compose an LOB-specific entity type name.

    'flood' + 'policies' → 'FloodPolicyRecord'
    'lender_placed' + 'policies' → 'LenderPlacedPolicyRecord'
    'generic' + 'policies' → 'PolicyRecord' (no LOB layer)
    """
    base = _ENTITY_TYPE_MAP.get(record_type, "Record")
    if not lob or lob == "generic":
        return base
    # snake_case → PascalCase  (lender_placed → LenderPlaced)
    prefix = "".join(part.capitalize() for part in lob.split("_") if part)
    return f"{prefix}{base}"


def _build_class_chain_triples(
    lob_combos: set[tuple[str, str]],
    record_types: set[str],
) -> list[dict[str, Any]]:
    """Emit ontology-class IS_A triples once per pipeline run.

    Two layers:
      • LOB-specific → general:  ``<Lob><RecordType> IS_A <RecordType>``
        (skipped for ``lob == 'generic'``)
      • General → root:          ``<RecordType> IS_A Record``

    Class triples are marked ``source_type='structured'`` so they
    round-trip through the dedup + Neo4j MERGE path identically to
    record triples.  Class subjects/objects are typed
    :data:`_ONTOLOGY_CLASS_TYPE` so Zone 3 induction can recognize
    them as ontology nodes.
    """
    triples: list[dict[str, Any]] = []

    # Layer 1: LOB-specific → general.
    for lob, record_type in sorted(lob_combos):
        if lob == "generic":
            continue
        specific = _lob_entity_type(record_type, lob)
        general = _ENTITY_TYPE_MAP.get(record_type, "Record")
        if specific == general:
            continue  # safety: never emit (X IS_A X)
        triples.append({
            "subject": specific,
            "subject_type": _ONTOLOGY_CLASS_TYPE,
            "relation": "IS_A",
            "relation_raw": "IS_A",
            "object": general,
            "object_type": _ONTOLOGY_CLASS_TYPE,
            "span": f"{specific} is a kind of {general}",
            "confidence": 1.0,
            "chunk_id": "ontology",
            "source": "ontology",
            "source_type": "structured",
            "source_role": "ontology",
        })

    # Layer 2: general → root (deduped).
    for record_type in sorted(record_types):
        general = _ENTITY_TYPE_MAP.get(record_type, "Record")
        if general == _ONTOLOGY_ROOT:
            continue
        triples.append({
            "subject": general,
            "subject_type": _ONTOLOGY_CLASS_TYPE,
            "relation": "IS_A",
            "relation_raw": "IS_A",
            "object": _ONTOLOGY_ROOT,
            "object_type": _ONTOLOGY_CLASS_TYPE,
            "span": f"{general} is a kind of {_ONTOLOGY_ROOT}",
            "confidence": 1.0,
            "chunk_id": "ontology",
            "source": "ontology",
            "source_type": "structured",
            "source_role": "ontology",
        })

    return triples

# Foreign key column patterns → what they reference.
_FOREIGN_KEY_PATTERNS: list[tuple[str, str]] = [
    ("policy_number", "policies"),
    ("policy_no", "policies"),
    ("policy_id", "policies"),
    ("claim_number", "claims"),
    ("claim_no", "claims"),
    ("claim_id", "claims"),
]


def detect_record_type(chunk: dict[str, Any], llm=None) -> str:
    """Infer record type from source filename, content, or LLM classification.

    Two-tier detection:
      Tier 1: Keyword heuristics (fast, no LLM)
      Tier 2: LLM fallback for unknown types (future-proof)

    Returns: 'policies', 'claims', 'survey', 'chat', 'email', or LLM-detected type.
    Falls back to 'unknown' if both tiers fail.
    """
    source = chunk.get("source", "").lower()
    content = chunk.get("content", "").lower()

    # --- Tier 1: Keyword heuristics ---

    # Insurance entity records
    if "policies" in source or "policies" in content:
        return "policies"
    if "claims" in source or "claims" in content:
        return "claims"

    # Observation records (surveys, chats, emails)
    if "survey" in source or any(kw in content for kw in (
        "nps_score", "nps score", "csat", "rating", "satisfaction",
        "recommend", "likelihood", "feedback",
    )):
        return "survey"
    if "chat" in source or any(kw in content for kw in (
        "conversation", "transcript", "chat_id", "chat session",
    )):
        return "chat"
    if "email" in source or any(kw in content for kw in (
        "subject_line", "email_body", "sender", "recipient",
    )):
        return "email"

    # Field-based heuristic for insurance records.
    if any(kw in content for kw in (
        "date of loss", "date of accident", "date of incident",
        "cause of damage", "cause of loss", "cause of accident",
    )):
        return "claims"
    if any(kw in content for kw in (
        "policy effective date", "policy cost",
        "effective date", "inception date",
    )):
        return "policies"

    # --- Tier 2: LLM fallback ---
    if llm is not None:
        detected = _detect_record_type_llm(chunk, llm)
        if detected and detected != "unknown":
            return detected

    return "unknown"


def _detect_record_type_llm(chunk: dict[str, Any], llm) -> str:
    """Classify a CSV file's record type via LLM. Called once per file
    when keyword detection returns 'unknown'.

    Uses the schema chunk (column names + sample rows) to determine:
    1. Record type (policy, claim, survey, email, ticket, assessment, etc.)
    2. Source role (entity vs observation)
    3. Foreign key columns linking to other records
    """
    source = chunk.get("source", "unknown")
    content = chunk.get("content", "")[:2000]  # cap for prompt size

    prompt = (
        f"This CSV file '{source}' has these columns and sample data:\n"
        f"{content}\n\n"
        "1. What type of records does this file contain?\n"
        "   (e.g., policy, claim, survey, email, support_ticket, "
        "inspection, assessment, payment, enrollment, inventory...)\n"
        "2. Is each record a real-world entity (like a policy or claim) "
        "or an observation ABOUT an entity (like a survey or email)?\n\n"
        'Answer with just the type name (one word, lowercase). '
        'Examples: "survey", "policy", "claims", "email", "assessment"'
    )
    try:
        from langchain_ollama import ChatOllama
        response = llm.invoke(prompt)
        answer = response.content.strip().lower().strip('"').strip("'")
        # Normalize to known types
        if "polic" in answer:
            return "policies"
        if "claim" in answer:
            return "claims"
        if "survey" in answer or "feedback" in answer or "nps" in answer:
            return "survey"
        if "chat" in answer or "conversation" in answer:
            return "chat"
        if "email" in answer or "message" in answer:
            return "email"
        # Return LLM's answer as-is for novel types
        if answer and len(answer) < 30:
            return answer
    except Exception as e:
        print(f"  ⚠ LLM record type detection failed: {e}")

    return "unknown"


def detect_foreign_keys(
    record: dict[str, str],
    record_type: str,
    all_record_keys: set[str] | None = None,
) -> list[dict[str, str]]:
    """Detect foreign key fields in a record that reference other records.

    Returns list of {column, value, references_type, target_key} dicts.
    """
    fk_triples: list[dict[str, str]] = []

    for field_name, value in record.items():
        if not value or len(value) > 50:
            continue
        field_lower = field_name.lower().replace(" ", "_")

        for pattern, ref_type in _FOREIGN_KEY_PATTERNS:
            if pattern in field_lower:
                # Determine target key prefix
                target_prefix = _RECORD_PREFIX.get(ref_type, "REC")
                target_key = f"{target_prefix}-{value}"

                # Only emit if we can verify the target exists (if key set provided)
                if all_record_keys is None or target_key in all_record_keys:
                    fk_triples.append({
                        "column": field_name,
                        "value": value,
                        "references_type": ref_type,
                        "target_key": target_key,
                    })
                break

    return fk_triples


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------

def parse_records_from_chunk(chunk: dict[str, Any]) -> list[dict[str, str]]:
    """Parse individual RECORD blocks from a batched CSV chunk.

    The chunk text format is:
        DATASET: OpenFEMA NFIP Policies

        RECORD:
          [Policy] field: value | field: value
          [Coverage] field: value | field: value

        RECORD:
          [Policy] field: value | field: value
          ...

    Returns a list of {field_name: value} dicts, one per record.
    """
    content = chunk.get("content", "")
    blocks = _RECORD_DELIM.split(content)

    records: list[dict[str, str]] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        fields = _parse_record_block(block)
        if fields:
            records.append(fields)

    return records


def _parse_record_block(block: str) -> dict[str, str]:
    """Parse a single RECORD block into {field_name: value} dict."""
    fields: dict[str, str] = {}

    for line in block.split("\n"):
        match = _GROUP_LINE.match(line)
        if not match:
            continue

        _group_name = match.group(1)  # e.g., "Policy", "Coverage"
        pairs_text = match.group(2)

        for pair in pairs_text.split("|"):
            pair = pair.strip()
            if ":" not in pair:
                continue

            # Split on first colon only (values may contain colons, e.g., timestamps).
            field_name, _, value = pair.partition(":")
            field_name = field_name.strip()
            value = value.strip()

            if field_name and value:
                fields[field_name] = value

    return fields


# ---------------------------------------------------------------------------
# Composite key generation
# ---------------------------------------------------------------------------

def generate_composite_key(
    record: dict[str, str],
    record_type: str,
) -> str:
    """Generate a stable entity key for a structured record.

    Priority:
      1. Use the original source ID if present (e.g., OpenFEMA "id" field)
         — preserves queryability by the real identifier.
      2. Fall back to SHA-256 hash of salient fields when no ID exists
         — keeps the pipeline domain-agnostic.
    """
    # --- Priority 1: original source ID ---
    original_id = record.get("id", "")
    prefix = _RECORD_PREFIX.get(record_type, "REC")

    if original_id:
        return f"{prefix}-{original_id}"

    # --- Priority 2: composite hash of salient fields ---
    key_fields = _KEY_FIELDS.get(record_type, [])

    # Collect values from salient fields.
    parts: list[str] = []
    for field in key_fields:
        value = record.get(field, "")
        if value:
            parts.append(value)
        if len(parts) >= _KEY_FIELD_COUNT:
            break

    # Fallback: use first N non-empty fields (alphabetical for stability).
    if len(parts) < _KEY_FIELD_COUNT:
        for field in sorted(record.keys()):
            if field not in key_fields and record[field]:
                parts.append(record[field])
            if len(parts) >= _KEY_FIELD_COUNT:
                break

    # Hash to 12 hex chars.
    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    return f"{prefix}-{digest}"


# ---------------------------------------------------------------------------
# Value type inference
# ---------------------------------------------------------------------------

def infer_value_type(field_name: str, value: str) -> str:
    """Heuristic type inference from field name and value.

    Returns one of: 'Date', 'Currency', 'Numeric', 'Percentage',
    'Categorical', 'Text'.
    """
    if _DATE_PATTERN.match(value):
        return "Date"

    if _CURRENCY_PATTERN.match(value):
        return "Currency"

    if _PERCENTAGE_PATTERN.match(value):
        return "Percentage"

    # Check field name hints for dates (e.g., "original construction date").
    field_lower = field_name.lower()
    if any(hint in field_lower for hint in _DATE_FIELD_HINTS):
        return "Date"

    if _NUMERIC_PATTERN.match(value):
        return "Numeric"

    # Short values with limited unique chars → likely categorical codes.
    if len(value) <= 5 and value.replace(".", "").replace("-", "").isalnum():
        return "Categorical"

    return "Text"


# ---------------------------------------------------------------------------
# Identity field detection & cross-record linking
# ---------------------------------------------------------------------------

# Heuristic patterns for identity-bearing fields (domain-agnostic).
_IDENTITY_PATTERNS: dict[str, list[str]] = {
    "person": [
        "name", "first name", "last name", "full name",
        "policyholder", "insured name", "claimant",
        "email", "e-mail", "email address",
        "date of birth", "birth date", "dob", "birthday",
        "phone", "telephone", "mobile",
        "ssn", "social security",
    ],
    "property": [
        "address", "street", "city", "state", "zip",
        "zip code", "postal code", "county",
        "property address", "mailing address",
    ],
}

# Minimum identity fields required to create a shared node.
_MIN_IDENTITY_FIELDS = 2


def detect_identity_fields(
    record: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Detect identity-bearing fields in a record using heuristic matching.

    Returns a dict of identity group → {field_name: value} for each
    group that has ≥ _MIN_IDENTITY_FIELDS matching fields.

    Example return:
        {"person": {"name": "John Smith", "email": "john@email.com"}}
    """
    matches: dict[str, dict[str, str]] = {
        group: {} for group in _IDENTITY_PATTERNS
    }

    for field_name, value in record.items():
        field_lower = field_name.lower()
        for group, patterns in _IDENTITY_PATTERNS.items():
            if any(pat in field_lower for pat in patterns):
                matches[group][field_name] = value
                break  # each field belongs to at most one group

    # Only return groups with enough fields for reliable linking.
    return {
        group: fields
        for group, fields in matches.items()
        if len(fields) >= _MIN_IDENTITY_FIELDS
    }


# Canonical field categories for cross-source identity matching.
# Maps (scope, attribute) to canonical category names.
# Scope-aware: "property state" → ("property", "state"),
#              "mailing address state" → ("mailing", "state") — different keys.
_IDENTITY_CANONICAL_ATTRIBUTES: dict[str, str] = {
    # Attribute keywords → canonical attribute name
    "state": "state",
    "zip": "zip",
    "zip code": "zip",
    "postal code": "zip",
    "city": "city",
    "county": "county",
    "address": "address",
    "street": "street",
    "name": "name",
    "first name": "first_name",
    "last name": "last_name",
    "full name": "name",
    "email": "email",
    "e-mail": "email",
    "phone": "phone",
    "telephone": "phone",
    "mobile": "phone",
}

# Scope keywords — when present, they qualify the attribute.
# "property state" → scope="property"; "mailing address state" → scope="mailing"
_SCOPE_KEYWORDS = frozenset({
    "property", "insured", "mailing", "billing", "loss",
    "reported", "rated", "original", "primary", "secondary",
    "policyholder", "claimant", "agent", "mortgagee",
})


def _canonicalize_identity_field(field_name: str) -> str:
    """Map a field name to a (scope, attribute) canonical key.

    'property state' → 'property:state'   (scope-qualified)
    'reported zip code' → 'reported:zip'   (scope-qualified)
    'state' → 'state'                      (no scope)
    'mailing address state' → 'mailing:state'  (different from property:state)

    Scope prevents collisions: "property state" and "mailing state"
    produce different canonical keys because they have different scopes.
    """
    field_lower = field_name.lower()

    # Extract scope (first matching scope keyword).
    scope = ""
    for kw in _SCOPE_KEYWORDS:
        if kw in field_lower:
            scope = kw
            break

    # Extract attribute (longest matching attribute keyword).
    attribute = field_lower  # fallback
    for pattern in sorted(_IDENTITY_CANONICAL_ATTRIBUTES.keys(),
                          key=len, reverse=True):
        if pattern in field_lower:
            attribute = _IDENTITY_CANONICAL_ATTRIBUTES[pattern]
            break

    return f"{scope}:{attribute}" if scope else attribute


def generate_identity_key(
    group: str,
    fields: dict[str, str],
) -> str:
    """Generate a stable key for an identity node.

    Hashes canonicalized (scope:attribute, value) pairs so the same
    person/property produces the same key regardless of CSV schema.

    Scope-aware: "property state" and "mailing state" are different
    canonical keys, preventing false merges across different entity roles.

    'property state' → 'property:state=fl'
    'reported zip code' → 'reported:zip=33019'

    POL- and CLM- records for the same physical property get the same
    PROP- node even when CSV column headers differ.
    """
    # Canonicalize field names to (scope:attribute, value) pairs.
    canonical_pairs = [
        (_canonicalize_identity_field(field_name), v.strip().lower())
        for field_name, v in fields.items()
        if v.strip()
    ]
    # Sort by canonical key for stability.
    canonical_pairs.sort(key=lambda x: x[0])
    # Include category names in the hash so "property:state=fl|reported:zip=33019" is unique.
    parts = [f"{cat}={val}" for cat, val in canonical_pairs]
    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    prefix = {
        "person": "PER",
        "property": "PROP",
    }.get(group, "ID")

    return f"{prefix}-{digest}"


def identity_triples(
    record_key: str,
    record_type: str,
    identity_groups: dict[str, dict[str, str]],
    chunk_id: str,
    source: str,
) -> list[dict[str, Any]]:
    """Emit identity nodes and BELONGS_TO links for a record.

    For each detected identity group (person, property):
      - 1 IS_A triple for the identity node (idempotent in Neo4j via MERGE)
      - 1 BELONGS_TO triple linking the record to the identity node
      - N property triples on the identity node (name, email, etc.)
    """
    triples: list[dict[str, Any]] = []

    entity_type_map = {
        "policies": "PolicyRecord",
        "claims": "ClaimRecord",
    }
    record_entity_type = entity_type_map.get(record_type, "Record")

    identity_type_map = {
        "person": "Person",
        "property": "Property",
    }

    for group, fields in identity_groups.items():
        id_key = generate_identity_key(group, fields)
        id_type = identity_type_map.get(group, "IdentityEntity")

        # IS_A triple for the identity node.
        triples.append({
            "subject": id_key,
            "subject_type": id_type,
            "relation": "IS_A",
            "relation_raw": "IS_A",
            "object": id_type,
            "object_type": "IdentityType",
            "span": f"{id_type} identity from {source}",
            "confidence": 1.0,
            "chunk_id": chunk_id,
            "source": source,
            "source_type": "structured",
        })

        # BELONGS_TO link: record → identity.
        triples.append({
            "subject": record_key,
            "subject_type": record_entity_type,
            "relation": "BELONGS_TO",
            "relation_raw": "BELONGS_TO",
            "object": id_key,
            "object_type": id_type,
            "span": f"{record_key} belongs to {id_type} {id_key}",
            "confidence": 1.0,
            "chunk_id": chunk_id,
            "source": source,
            "source_type": "structured",
        })

        # Property triples on the identity node.
        for field_name, value in fields.items():
            relation = _field_to_relation(field_name)
            value_type = infer_value_type(field_name, value)

            triples.append({
                "subject": id_key,
                "subject_type": id_type,
                "relation": relation,
                "relation_raw": relation,
                "object": value,
                "object_type": value_type,
                "span": f"{field_name}: {value}",
                "confidence": 1.0,
                "chunk_id": chunk_id,
                "source": source,
                "source_type": "structured",
            })

    return triples


# ---------------------------------------------------------------------------
# Triple generation
# ---------------------------------------------------------------------------

_ARTICLES = frozenset({"a", "an", "the"})


def _normalize_field_name(field_name: str) -> str:
    """Strip only articles (determiners) that cause near-duplicate relation names.

    'number of floors in the insured building' → 'number of floors in insured building'
    'number of floors in insured building'     → 'number of floors in insured building'

    IMPORTANT: Prepositions (of, in, for, by, etc.) are KEPT because they
    carry semantic role information. Only articles are removed — they are
    the sole source of CSV-header duplication (e.g., "in the" vs "in").

    Letter-spacing guard: if the input looks letter-spaced (majority of tokens
    are length-1, e.g. "m a n u f a c t u r e r"), bypass article stripping —
    every "a" would otherwise be removed, mangling the field name. The upstream
    fix is in zone1/ingestion.py header expansion, but this is defense-in-depth.

    Domain-agnostic: safe for any CSV column naming convention.
    """
    tokens = field_name.lower().split()
    if tokens and sum(1 for t in tokens if len(t) == 1) / len(tokens) > 0.5:
        return field_name.lower()
    return " ".join(t for t in tokens if t not in _ARTICLES)


def _field_to_relation(field_name: str) -> str:
    """Convert a human-readable field name to UPPER_SNAKE_CASE relation.

    'policy effective date' → 'HAS_POLICY_EFFECTIVE_DATE'
    'number of floors in the insured building' → 'HAS_NUMBER_OF_FLOORS_IN_INSURED_BUILDING'
    'number of floors in insured building'     → 'HAS_NUMBER_OF_FLOORS_IN_INSURED_BUILDING'

    Only articles (a/an/the) are stripped. Prepositions are preserved.
    """
    cleaned = _normalize_field_name(field_name)
    normalized = re.sub(r"[^a-z0-9\s]", "", cleaned)
    snake = re.sub(r"\s+", "_", normalized.strip())
    return f"HAS_{snake.upper()}"


def record_to_triples(
    record: dict[str, str],
    record_type: str,
    chunk_id: str,
    source: str,
    record_index: int = 0,
    lob: str = "generic",
) -> list[dict[str, Any]]:
    """Convert one parsed record into deterministic triples.

    Produces:
      - 1 type triple:  (composite_key, IS_A, <Lob>PolicyRecord)
                        — instance-to-specific only.  The class chain
                          (<Lob>X IS_A X, X IS_A Record) is emitted once
                          per pipeline run by :func:`extract_structured`.
      - N property triples: (composite_key, HAS_FIELD_NAME, value)

    When ``lob`` is ``"generic"`` (or omitted), the LOB layer collapses
    and the entity type stays as the base ``PolicyRecord`` /
    ``ClaimRecord`` / etc.  All triples have confidence=1.0 and
    source_type='structured'.
    """
    key = generate_composite_key(record, record_type)

    entity_type = _lob_entity_type(record_type, lob)

    triples: list[dict[str, Any]] = []

    source_role = _SOURCE_ROLES.get(record_type, "entity")

    # Type triple.
    triples.append({
        "subject": key,
        "subject_type": entity_type,
        "relation": "IS_A",
        "relation_raw": "IS_A",
        "object": entity_type,
        "object_type": "RecordType",
        "span": f"{entity_type} record from {source}",
        "confidence": 1.0,
        "chunk_id": chunk_id,
        "source": source,
        "source_type": "structured",
        "source_role": source_role,
    })

    # Property triples.  ``relation_raw`` mirrors ``relation`` for
    # structured triples — the field-derived HAS_X is the lossless,
    # source-specific form Zone 3 will cluster bottom-up.
    for field_name, value in record.items():
        value_type = infer_value_type(field_name, value)
        relation = _field_to_relation(field_name)

        triples.append({
            "subject": key,
            "subject_type": entity_type,
            "relation": relation,
            "relation_raw": relation,
            "object": value,
            "object_type": value_type,
            "span": f"{field_name}: {value}",
            "confidence": 1.0,
            "chunk_id": chunk_id,
            "source": source,
            "source_type": "structured",
            "source_role": source_role,
        })

    # Foreign key ABOUT edges (observation → entity linkage).
    if source_role == "observation":
        fk_hits = detect_foreign_keys(record, record_type)
        for fk in fk_hits:
            triples.append({
                "subject": key,
                "subject_type": entity_type,
                "relation": "ABOUT",
                "relation_raw": "ABOUT",
                "object": fk["target_key"],
                "object_type": _ENTITY_TYPE_MAP.get(fk["references_type"], "Record"),
                "span": f"{fk['column']}: {fk['value']}",
                "confidence": 1.0,
                "chunk_id": chunk_id,
                "source": source,
                "source_type": "structured",
                "source_role": source_role,
            })

    return triples


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def extract_structured(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: deterministic extraction for CSV-derived chunks.

    Splits chunks into structured (CSV) and unstructured (PDF).
    Generates triples for structured chunks deterministically.
    Passes only PDF chunks through to LLM extraction.
    """
    all_chunks = state.get("chunks", [])

    structured_triples: list[dict[str, Any]] = []
    pdf_chunks: list[dict[str, Any]] = []
    schema_chunks: list[dict[str, Any]] = []

    n_records = 0
    n_structured_chunks = 0
    n_identity_nodes = 0
    identity_keys_seen: set[str] = set()
    # Track unique (lob, record_type) and unique record_type combos so the
    # class-chain triples can be emitted exactly once each after the loop.
    seen_lob_combos: set[tuple[str, str]] = set()
    seen_record_types: set[str] = set()

    for i, chunk in enumerate(all_chunks):
        if is_schema_chunk(chunk):
            # Keep schema chunks for bootstrap sampling — they describe fields.
            schema_chunks.append(chunk)
            continue

        if is_structured_chunk(chunk):
            n_structured_chunks += 1
            record_type = detect_record_type(chunk)
            records = parse_records_from_chunk(chunk)

            chunk_id = str(chunk.get("chunk_id", i))
            source = chunk.get("source", "unknown")
            chunk_lob = chunk.get("lob", "generic") or "generic"
            seen_lob_combos.add((chunk_lob, record_type))
            seen_record_types.add(record_type)

            for j, record in enumerate(records):
                triples = record_to_triples(
                    record=record,
                    record_type=record_type,
                    chunk_id=chunk_id,
                    source=source,
                    record_index=j,
                    lob=chunk_lob,
                )
                structured_triples.extend(triples)

                # Identity detection & cross-record linking.
                record_key = generate_composite_key(record, record_type)
                id_groups = detect_identity_fields(record)
                if id_groups:
                    id_triples = identity_triples(
                        record_key=record_key,
                        record_type=record_type,
                        identity_groups=id_groups,
                        chunk_id=chunk_id,
                        source=source,
                    )
                    structured_triples.extend(id_triples)
                    for group, fields in id_groups.items():
                        id_key = generate_identity_key(group, fields)
                        if id_key not in identity_keys_seen:
                            identity_keys_seen.add(id_key)
                            n_identity_nodes += 1

                n_records += 1
        else:
            pdf_chunks.append(chunk)

    # --- Class chain triples (Phase 2 hierarchy) ---------------------------
    # For every (lob, record_type) seen, emit:
    #   <Lob><RecordType> IS_A <RecordType>     (skipped when lob == 'generic')
    #   <RecordType>      IS_A Record            (root, deduped)
    class_chain_triples = _build_class_chain_triples(
        seen_lob_combos, seen_record_types
    )
    structured_triples.extend(class_chain_triples)

    # Include schema chunks with PDF chunks so bootstrap_vocab can sample them.
    remaining_chunks = schema_chunks + pdf_chunks

    print(f"\n[1.5/4] Structured mapper — SEAF-KG Stage 1")
    print(f"  ✓ {n_structured_chunks} structured chunks → "
          f"{n_records} records → {len(structured_triples)} triples")
    if n_identity_nodes:
        print(f"  ✓ {n_identity_nodes} unique identity nodes detected "
              f"(cross-record linking enabled)")
    else:
        print(f"  ℹ No identity fields detected — using record-level keys only")
    print(f"  ✓ {len(pdf_chunks)} PDF chunks passed to LLM extraction")
    print(f"  ✓ {len(schema_chunks)} schema chunks kept for bootstrap sampling")

    return {
        "structured_triples": structured_triples,
        "chunks": remaining_chunks,
    }
