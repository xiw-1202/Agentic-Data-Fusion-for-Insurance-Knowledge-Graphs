"""Phase 3: Bridge-column relationship discovery between candidate classes.

1. find_bridge_columns — algorithmic: headers appearing in 2+ classes form bridges
2. build_relationship_naming_prompt — build LLM prompt for naming relationships
3. name_relationships — LLM call to name each bridge relationship
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from zone3.fbi.class_discovery import CandidateClass
from zone3.fbi.llm_utils import llm_call_json


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClassRelationship:
    """A relationship between two candidate classes via a shared bridge column."""

    source_class: str  # class name
    target_class: str  # class name
    relationship_name: str  # LLM-named (e.g., "references")
    bridge_column: str  # the header that bridges them (e.g., "POLICY_NUMBER")
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Phase 3, Step 1 — Bridge column detection (algorithmic)
# ---------------------------------------------------------------------------


def _collect_all_headers(cls: CandidateClass) -> set[str]:
    """Collect headers from a class and all its children recursively."""
    headers: set[str] = set(cls.headers)
    for child in cls.children:
        headers.update(_collect_all_headers(child))
    return headers


def find_bridge_columns(
    classes: list[CandidateClass],
    raw_headers_by_class: dict[str, set[str]] | None = None,
) -> list[ClassRelationship]:
    """Find headers appearing in 2+ different classes and create relationships.

    If ``raw_headers_by_class`` is provided, it maps top-level class name →
    the union of raw headers from that class's source files. Use this when
    classes were built from shared-header intersections, because bridge
    columns (e.g. ``POLICY_NUMBER``) may not appear in any class's shared
    headers but DO appear in the raw source file headers.

    Otherwise, fall back to walking ``cls.headers`` (and children
    recursively) — useful for simpler inputs.

    Builds a column → set[class_name] mapping, then creates one
    ClassRelationship per unique (source, target, bridge) triple.
    Relationship names are left empty for the LLM to fill later.
    """
    if len(classes) < 2:
        return []

    # Map each column to the ordered list of class names it appears in.
    # Using a dict preserves first-seen order to keep output deterministic.
    column_to_classes: dict[str, list[str]] = defaultdict(list)

    if raw_headers_by_class is not None:
        # Use the caller-provided raw headers. Iterate classes in order so
        # that relationship ordering is stable.
        for cls in classes:
            class_name = cls.name or cls.prefix or "(unnamed)"
            headers = raw_headers_by_class.get(class_name, set())
            for header in headers:
                if class_name not in column_to_classes[header]:
                    column_to_classes[header].append(class_name)
    else:
        # Fallback: collect headers from each class (including children).
        for idx, cls in enumerate(classes):
            class_name = cls.name or cls.prefix or f"Class_{idx}"
            all_headers = _collect_all_headers(cls)
            for header in all_headers:
                if class_name not in column_to_classes[header]:
                    column_to_classes[header].append(class_name)

    # For each bridge column, create relationships for every pair
    seen_triples: set[tuple[str, str, str]] = set()
    relationships: list[ClassRelationship] = []

    for column, class_names in column_to_classes.items():
        if len(class_names) < 2:
            continue

        for source_name, target_name in combinations(class_names, 2):
            triple = (source_name, target_name, column)
            if triple in seen_triples:
                continue
            seen_triples.add(triple)

            relationships.append(
                ClassRelationship(
                    source_class=source_name,
                    target_class=target_name,
                    relationship_name="",
                    bridge_column=column,
                )
            )

    return relationships


# ---------------------------------------------------------------------------
# Phase 3, Step 2 — Relationship naming prompt
# ---------------------------------------------------------------------------


def build_relationship_naming_prompt(
    relationships: list[ClassRelationship],
) -> str:
    """Build a prompt asking the LLM to name each bridge relationship.

    Returns a prompt requesting JSON:
    {"relationships": [{"source": "...", "target": "...", "bridge": "...", "name": "..."}]}
    """
    lines = [
        "You are a data ontology expert.",
        "Name each relationship below based on the bridge column that connects two classes.",
        "",
        "Relationships to name:",
    ]

    for rel in relationships:
        lines.append(
            f"  - {rel.source_class} <-> {rel.target_class} "
            f"(bridge column: {rel.bridge_column})"
        )

    lines.extend([
        "",
        "Return ONLY a JSON object in this exact format:",
        '{"relationships": [{"source": "ClassName", "target": "ClassName", '
        '"bridge": "COLUMN_NAME", "name": "relationship_name"}]}',
        "",
        "Rules:",
        "- Use concise lowercase relationship names (e.g., 'covers', 'references', 'belongs_to')",
        "- The name should describe how the source relates to the target",
        "- Use domain vocabulary where appropriate",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 3, Step 3 — LLM naming
# ---------------------------------------------------------------------------


def name_relationships(
    relationships: list[ClassRelationship],
    model: str | None = None,
) -> None:
    """Call LLM to name each relationship in-place.

    Unnamed relationships default to ``RELATES_TO``.
    """
    if not relationships:
        return

    prompt = build_relationship_naming_prompt(relationships)
    result = llm_call_json(prompt, model=model)

    # Build lookup: (source, target, bridge) -> name
    naming_map: dict[tuple[str, str, str], str] = {}
    raw_rels = result.get("relationships", []) if isinstance(result, dict) else []
    for entry in raw_rels:
        if not isinstance(entry, dict):
            continue
        key = (
            entry.get("source", ""),
            entry.get("target", ""),
            entry.get("bridge", ""),
        )
        name = entry.get("name", "")
        if name:
            naming_map[key] = name

    # Apply names
    for rel in relationships:
        key = (rel.source_class, rel.target_class, rel.bridge_column)
        rel.relationship_name = naming_map.get(key, "RELATES_TO")
