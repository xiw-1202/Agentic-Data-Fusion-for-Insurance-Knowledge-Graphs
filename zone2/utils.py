"""Shared Neo4j sanitizers for Zone 2 pipeline.

Single source of truth for label and relation sanitization used across
pipeline.py, entity_resolution.py, and cross_source_linker.py.
"""

from __future__ import annotations

import re


def sanitize_label(label: str) -> str:
    """Make a Neo4j node label safe for Cypher f-string interpolation.

    Strips all characters that are not alphanumeric or underscore.
    Raises ValueError if the result is empty.
    """
    cleaned = re.sub(r'[^A-Za-z0-9_]', '', label.strip())
    if not cleaned:
        raise ValueError(f"Invalid Neo4j label: {label!r}")
    return cleaned


def sanitize_relation(rel: str) -> str:
    """Make a relation type safe for Neo4j Cypher f-string interpolation.

    Uppercases and replaces non-alphanumeric/underscore chars with '_'.
    Raises ValueError if the result is empty or all underscores.
    """
    cleaned = re.sub(r'[^A-Z0-9_]', '_', rel.upper().strip())
    if not cleaned or set(cleaned) == {'_'}:
        raise ValueError(f"Invalid Neo4j relation name: {rel!r}")
    return cleaned
