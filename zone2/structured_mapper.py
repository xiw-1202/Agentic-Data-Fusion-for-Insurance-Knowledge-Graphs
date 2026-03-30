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
      - `.json` in the source path (OpenFEMA JSON files)
      - "records" in section_hierarchy (data batches, not schema)
    """
    source = chunk.get("source", "")
    hierarchy = chunk.get("section_hierarchy", [])
    hierarchy_str = " ".join(str(h) for h in hierarchy).lower()

    is_json_source = ".json" in source.lower()
    is_data_batch = "records" in hierarchy_str  # not "schema"

    return is_json_source and is_data_batch


def is_schema_chunk(chunk: dict[str, Any]) -> bool:
    """Detect schema description chunks (chunk 0 of each CSV source)."""
    source = chunk.get("source", "")
    hierarchy = chunk.get("section_hierarchy", [])
    hierarchy_str = " ".join(str(h) for h in hierarchy).lower()

    return ".json" in source.lower() and "schema" in hierarchy_str


def detect_record_type(chunk: dict[str, Any]) -> str:
    """Infer record type from source filename or content.

    Returns 'policies', 'claims', or 'unknown'.
    """
    source = chunk.get("source", "").lower()
    content = chunk.get("content", "").lower()

    if "policies" in source or "policies" in content:
        return "policies"
    if "claims" in source or "claims" in content:
        return "claims"

    # Field-based heuristic for unknown sources.
    if "date of loss" in content or "cause of damage" in content:
        return "claims"
    if "policy effective date" in content or "policy cost" in content:
        return "policies"

    return "unknown"


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
    """Generate a 12-char hex composite key from salient fields.

    Uses predefined salient fields per record type. Falls back to
    first N non-empty fields for unknown record types (domain-agnostic).
    """
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

    # Prefix with record type for readability.
    prefix = {
        "policies": "POL",
        "claims": "CLM",
    }.get(record_type, "REC")

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
# Triple generation
# ---------------------------------------------------------------------------

def _field_to_relation(field_name: str) -> str:
    """Convert a human-readable field name to UPPER_SNAKE_CASE relation.

    'policy effective date' → 'HAS_POLICY_EFFECTIVE_DATE'
    'rated flood zone' → 'HAS_RATED_FLOOD_ZONE'
    """
    normalized = re.sub(r"[^a-z0-9\s]", "", field_name.lower())
    snake = re.sub(r"\s+", "_", normalized.strip())
    return f"HAS_{snake.upper()}"


def record_to_triples(
    record: dict[str, str],
    record_type: str,
    chunk_id: str,
    source: str,
    record_index: int = 0,
) -> list[dict[str, Any]]:
    """Convert one parsed record into deterministic triples.

    Produces:
      - 1 type triple:  (composite_key, IS_A, PolicyRecord/ClaimRecord)
      - N property triples: (composite_key, HAS_FIELD_NAME, value)

    All triples have confidence=1.0 and source_type='structured'.
    """
    key = generate_composite_key(record, record_type)

    entity_type_map = {
        "policies": "PolicyRecord",
        "claims": "ClaimRecord",
    }
    entity_type = entity_type_map.get(record_type, "Record")

    triples: list[dict[str, Any]] = []

    # Type triple.
    triples.append({
        "subject": key,
        "subject_type": entity_type,
        "relation": "IS_A",
        "object": entity_type,
        "object_type": "RecordType",
        "span": f"{entity_type} record from {source}",
        "confidence": 1.0,
        "chunk_id": chunk_id,
        "source": source,
        "source_type": "structured",
    })

    # Property triples.
    for field_name, value in record.items():
        value_type = infer_value_type(field_name, value)
        relation = _field_to_relation(field_name)

        triples.append({
            "subject": key,
            "subject_type": entity_type,
            "relation": relation,
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

            for j, record in enumerate(records):
                triples = record_to_triples(
                    record=record,
                    record_type=record_type,
                    chunk_id=chunk_id,
                    source=source,
                    record_index=j,
                )
                structured_triples.extend(triples)
                n_records += 1
        else:
            pdf_chunks.append(chunk)

    # Include schema chunks with PDF chunks so bootstrap_vocab can sample them.
    remaining_chunks = schema_chunks + pdf_chunks

    print(f"\n[1.5/4] Structured mapper — SEAF-KG Stage 1")
    print(f"  ✓ {n_structured_chunks} structured chunks → "
          f"{n_records} records → {len(structured_triples)} triples")
    print(f"  ✓ {len(pdf_chunks)} PDF chunks passed to LLM extraction")
    print(f"  ✓ {len(schema_chunks)} schema chunks kept for bootstrap sampling")

    return {
        "structured_triples": structured_triples,
        "chunks": remaining_chunks,
    }
