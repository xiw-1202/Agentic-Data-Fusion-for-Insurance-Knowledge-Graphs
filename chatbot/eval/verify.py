"""Per-answer verification: for each KG relation referenced in the Cypher,
compute a side-by-side comparison of the source CSV vs the KG so the user
can see exactly what's missing (which categorical values, which numeric
range, how many distinct vs total) — not just "97/100 covered".

Pure pandas/csv — no LLM. Cached per-CSV.
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


def _try_float(v: str) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


@lru_cache(maxsize=1)
def _build_csv_index(data_dir: str) -> dict[str, dict[str, Any]]:
    """relation_type → {csv_file, column, csv_non_null, csv_rows, csv_values, csv_numeric_stats}.

    csv_values: list of distinct string values (all of them — we use it for set diffs).
    csv_numeric_stats: {min, max, mean, n_numeric} if at least 1 value is numeric, else None.
    """
    index: dict[str, dict[str, Any]] = {}
    base = Path(data_dir)
    if not base.exists():
        return index
    for csv_path in sorted(base.glob("*.csv")):
        try:
            with csv_path.open(encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                values: dict[str, set[str]] = {c: set() for c in cols}
                non_null = {c: 0 for c in cols}
                numerics: dict[str, list[float]] = {c: [] for c in cols}
                total = 0
                for row in reader:
                    total += 1
                    for c in cols:
                        v = (row.get(c) or "").strip()
                        if v:
                            non_null[c] += 1
                            values[c].add(v)
                            n = _try_float(v)
                            if n is not None:
                                numerics[c].append(n)
        except (OSError, UnicodeDecodeError):
            continue

        for col in cols:
            rel = _column_to_relation(col)
            if rel in index or non_null[col] == 0:
                continue
            nums = numerics[col]
            stats = None
            if nums and len(nums) >= 0.5 * non_null[col]:  # mostly numeric
                stats = {
                    "n_numeric": len(nums),
                    "min": round(min(nums), 2),
                    "max": round(max(nums), 2),
                    "mean": round(sum(nums) / len(nums), 2),
                }
            index[rel] = {
                "csv_file": csv_path.name,
                "column": col,
                "csv_non_null": non_null[col],
                "csv_rows": total,
                "csv_distinct": len(values[col]),
                "csv_values": values[col],   # set
                "csv_numeric_stats": stats,
            }
    return index


def kg_relation_stats(graph: Any, rel_type: str) -> dict[str, Any]:
    """Pull KG-side stats for one relation: edge count, distinct objects,
    numeric range if applicable."""
    rows = graph.query(
        f"""
        MATCH ()-[r:`{rel_type}`]->(o:Entity)
        WITH count(r) AS edges, collect(DISTINCT o.id) AS vals
        RETURN edges, vals
        """
    )
    if not rows:
        return {"kg_edges": 0, "kg_distinct": 0, "kg_values": set(), "kg_numeric_stats": None}
    edges = rows[0]["edges"]
    vals = set(rows[0]["vals"] or [])
    nums = [n for n in (_try_float(v) for v in vals) if n is not None]
    stats = None
    if nums:
        stats = {
            "n_numeric": len(nums),
            "min": round(min(nums), 2),
            "max": round(max(nums), 2),
            "mean": round(sum(nums) / len(nums), 2),
        }
    return {
        "kg_edges": edges,
        "kg_distinct": len(vals),
        "kg_values": vals,
        "kg_numeric_stats": stats,
    }


def diff_relation(
    rel_type: str, graph: Any, data_dir: str = "data/Emory_Spring2026"
) -> dict[str, Any] | None:
    """Compute full CSV vs KG diff for a single relation.

    Returns None if the relation isn't in any CSV (likely a synthesized rel
    not directly mapped to a column).
    """
    index = _build_csv_index(data_dir)
    csv_info = index.get(rel_type)
    if not csv_info:
        return None
    kg = kg_relation_stats(graph, rel_type)

    csv_vals = csv_info["csv_values"]
    kg_vals = kg["kg_values"]
    missing = sorted(csv_vals - kg_vals)
    extra = sorted(kg_vals - csv_vals)

    # Coverage on rows (KG edges / CSV non-null) and on distinct values
    row_cov = (kg["kg_edges"] / csv_info["csv_non_null"] * 100) if csv_info["csv_non_null"] else 0
    val_cov = (
        len(csv_vals & kg_vals) / len(csv_vals) * 100 if csv_vals else 0
    )

    return {
        "relation": rel_type,
        "csv_file": csv_info["csv_file"],
        "column": csv_info["column"],
        "csv_non_null": csv_info["csv_non_null"],
        "csv_distinct": csv_info["csv_distinct"],
        "kg_edges": kg["kg_edges"],
        "kg_distinct": kg["kg_distinct"],
        "row_coverage_pct": round(row_cov, 1),
        "value_coverage_pct": round(val_cov, 1),
        "missing_in_kg": missing,        # values present in CSV but not KG
        "extra_in_kg": extra,            # values present in KG but not CSV (extraction noise)
        "csv_numeric_stats": csv_info["csv_numeric_stats"],
        "kg_numeric_stats": kg["kg_numeric_stats"],
    }


def diff_relations_in_cypher(
    cypher: str, graph: Any, data_dir: str = "data/Emory_Spring2026"
) -> list[dict[str, Any]]:
    """Diff every HAS_* relation referenced in the Cypher."""
    rels = extract_rel_types(cypher)
    out: list[dict[str, Any]] = []
    for rel in rels:
        d = diff_relation(rel, graph, data_dir)
        if d:
            out.append(d)
    return out
