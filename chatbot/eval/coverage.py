"""Compare CSV ground truth vs the loaded KG to find extraction gaps.

For each CSV column the loader expects (HAS_COLUMN_NAME), reports:
    csv_rows         total rows in the CSV
    csv_non_null     rows with a value in this column
    kg_edges         relationships of type HAS_<column>
    coverage_pct     kg_edges / csv_non_null * 100

Coverage <50% usually means Zone 2 dropped values (numeric fields, long
strings, formats it didn't recognize).

Usage:
    python -m chatbot.eval.coverage
    python -m chatbot.eval.coverage --data-dir data/Emory_Spring2026 --top 30
    python -m chatbot.eval.coverage --json out/coverage.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from langchain_neo4j import Neo4jGraph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def _column_to_relation(col: str) -> str:
    """Mirror zone2/structured_mapper.py: 'TOTAL_CLAIM_TIME' → 'HAS_TOTAL_CLAIM_TIME'."""
    snake = re.sub(r"[^A-Za-z0-9]+", "_", col.strip().lower()).strip("_")
    return f"HAS_{snake.upper()}"


def _csv_columns(path: Path) -> tuple[list[str], dict[str, int], int]:
    """Return (columns, non_null_counts, total_rows)."""
    with path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        non_null = {c: 0 for c in cols}
        total = 0
        for row in reader:
            total += 1
            for c in cols:
                v = (row.get(c) or "").strip()
                if v:
                    non_null[c] += 1
    return cols, non_null, total


def _kg_edge_counts(graph: Neo4jGraph, rel_types: list[str]) -> dict[str, int]:
    if not rel_types:
        return {}
    rows = graph.query(
        """
        UNWIND $rels AS rel_type
        CALL (rel_type) {
          MATCH ()-[r]->()
          WHERE type(r) = rel_type
          RETURN count(r) AS n
        }
        RETURN rel_type, n
        """,
        params={"rels": rel_types},
    )
    return {r["rel_type"]: r["n"] for r in rows}


def coverage_for_csv(
    graph: Neo4jGraph, csv_path: Path
) -> list[dict[str, Any]]:
    cols, non_null, total = _csv_columns(csv_path)
    rels = [_column_to_relation(c) for c in cols]
    kg_counts = _kg_edge_counts(graph, rels)

    out: list[dict[str, Any]] = []
    for col, rel in zip(cols, rels):
        nn = non_null[col]
        kg = kg_counts.get(rel, 0)
        cov = (kg / nn * 100) if nn else 0.0
        out.append(
            {
                "csv_file": csv_path.name,
                "column": col,
                "relation": rel,
                "csv_rows": total,
                "csv_non_null": nn,
                "kg_edges": kg,
                "coverage_pct": round(cov, 1),
            }
        )
    return out


def _format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(empty)"
    header = ("column", "relation", "non_null", "kg_edges", "cov%")
    widths = (32, 36, 9, 9, 7)
    lines = [
        "  ".join(h.ljust(w) for h, w in zip(header, widths)),
        "  ".join("-" * w for w in widths),
    ]
    for r in rows:
        cov = r["coverage_pct"]
        marker = "🔴" if cov < 30 else ("🟡" if cov < 70 else "🟢")
        lines.append(
            "  ".join(
                [
                    str(r["column"])[:32].ljust(widths[0]),
                    str(r["relation"])[:36].ljust(widths[1]),
                    str(r["csv_non_null"]).rjust(widths[2]),
                    str(r["kg_edges"]).rjust(widths[3]),
                    f"{marker} {cov:>5.1f}".ljust(widths[4]),
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/Emory_Spring2026"),
        help="Directory containing CSV files (default: data/Emory_Spring2026)",
    )
    parser.add_argument(
        "--top", type=int, default=30, help="Show top-N rows per CSV (default: 30)"
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Optional: write full JSON report"
    )
    args = parser.parse_args()

    csv_files = sorted(args.data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )

    all_rows: list[dict[str, Any]] = []
    for csv_path in csv_files:
        print(f"\n=== {csv_path.name} ===")
        rows = coverage_for_csv(graph, csv_path)
        all_rows.extend(rows)
        # Sort by worst coverage first, but only if the column has data
        worst = sorted(
            [r for r in rows if r["csv_non_null"] > 0],
            key=lambda r: (r["coverage_pct"], -r["csv_non_null"]),
        )[: args.top]
        print(_format_table(worst))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(all_rows, indent=2))
        print(f"\n✓ Full report: {args.json}")

    # Summary stats
    total = len([r for r in all_rows if r["csv_non_null"] > 0])
    bad = len([r for r in all_rows if r["csv_non_null"] > 0 and r["coverage_pct"] < 30])
    mid = len([r for r in all_rows if 30 <= r["coverage_pct"] < 70])
    good = len([r for r in all_rows if r["coverage_pct"] >= 70])
    print(f"\n📊 Coverage summary across {total} columns:")
    print(f"   🟢 ≥70%:  {good}")
    print(f"   🟡 30-70%: {mid}")
    print(f"   🔴 <30%:   {bad}")


if __name__ == "__main__":
    main()
