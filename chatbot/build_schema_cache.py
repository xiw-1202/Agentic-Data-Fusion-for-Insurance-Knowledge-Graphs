"""Pre-build the schema prefix from the live Neo4j KG and write it to disk.

Run this once after each Zone 4 reload so the chatbot's prompt always reflects
the real entities + categorical values currently in the graph. The chatbot
auto-loads the cached file if present (skipping the runtime queries).

Usage:
    python -m chatbot.build_schema_cache
    python -m chatbot.build_schema_cache --out chatbot/schema_prefix.flood.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from langchain_neo4j import Neo4jGraph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from chatbot.examples import format_for_prompt as examples_block
from chatbot.schema import summarize_schema


DEFAULT_OUT = Path(__file__).parent / "schema_prefix.cache.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output file (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    print(f"Connecting to {config.NEO4J_URI} (db={config.NEO4J_DATABASE})...")
    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )

    print("Querying schema (labels, classes, hierarchy, properties, samples, enums)...")
    schema = summarize_schema(graph)
    examples = examples_block()
    prefix = f"{schema}\n\n{examples}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(prefix, encoding="utf-8")

    n_lines = prefix.count("\n") + 1
    n_chars = len(prefix)
    print(f"\n✓ Wrote {n_lines} lines / {n_chars:,} chars to {args.out}")
    print("\n--- Preview (first 40 lines) ---")
    for line in prefix.splitlines()[:40]:
        print(line)
    print("...")
    print(f"\nOpen the full file to inspect what Claude actually sees: {args.out}")


if __name__ == "__main__":
    main()
