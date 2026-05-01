"""Per-answer verification: for each KG relation referenced in the Cypher,
report (CSV non-null count, KG edge count, sample CSV values) so the user
can spot extraction gaps in the answer they're looking at.

Pure pandas/csv — no LLM. Runs locally against the Emory CSVs. Cached.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

REL_RE = re.compile(r"-\[\s*(?::\w+\|)?:(\w+)\s*[\]\*]", re.I)


def extract_rel_types(cypher: str) -> list[str]:
    """Pull HAS_FOO relation types from a Cypher query string."""
    return sorted({m.group(1) for m in REL_RE.finditer(cypher) if m.group(1).startswith("HAS_")})


def _column_to_relation(col: str) -> str:
    snake = re.sub(r"[^A-Za-z0-9]+", "_", col.strip().lower()).strip("_")
    return f"HAS_{snake.upper()}"


@lru_cache(maxsize=1)
def _build_csv_index(data_dir: str) -> dict[str, dict[str, Any]]:
    """Map relation_type → {csv_file, column, non_null, total, sample_values}."""
    index: dict[str, dict[str, Any]] = {}
    base = Path(data_dir)
    if not base.exists():
        return index
    for csv_path in sorted(base.glob("*.csv")):
        try:
            with csv_path.open(encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                non_null = {c: 0 for c in cols}
                samples: dict[str, set[str]] = {c: set() for c in cols}
                total = 0
                for row in reader:
                    total += 1
                    for c in cols:
                        v = (row.get(c) or "").strip()
                        if v:
                            non_null[c] += 1
                            if len(samples[c]) < 8:
                                samples[c].add(v[:60])
        except (OSError, UnicodeDecodeError):
            continue

        for col in cols:
            rel = _column_to_relation(col)
            # First CSV that contains this column wins; deeper merges aren't needed
            # for a quick visual check.
            if rel not in index and non_null[col] > 0:
                index[rel] = {
                    "csv_file": csv_path.name,
                    "column": col,
                    "csv_non_null": non_null[col],
                    "csv_rows": total,
                    "samples": sorted(samples[col]),
                }
    return index


def verify_relations(
    cypher: str, data_dir: str = "data/Emory_Spring2026"
) -> list[dict[str, Any]]:
    """For each HAS_* relation in the Cypher, return CSV ground-truth stats.

    Returns a list of dicts. Relations not found in any CSV are omitted.
    """
    index = _build_csv_index(data_dir)
    rels = extract_rel_types(cypher)
    out: list[dict[str, Any]] = []
    for rel in rels:
        info = index.get(rel)
        if info:
            out.append({"relation": rel, **info})
    return out
