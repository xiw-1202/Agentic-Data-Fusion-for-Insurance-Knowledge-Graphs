"""Zone 4 — load cached pipeline artifacts into the Neo4j configured by .env.

Reads `zone2_run_summary.json` (triples), `zone3_svloi_summary_*.json`
(classes + hierarchy + associations), and `svloi_provenance_*.json`
(entity -> class map) from a results directory and writes them into the
Neo4j instance pointed at by config.NEO4J_URI. No LLM calls, no cluster
access — pure JSON -> Cypher.

Usage:
    python -m zone4.load_to_neo4j --results data/results/flood
    python -m zone4.load_to_neo4j --results data/results/emory --no-wipe
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from langchain_neo4j import Neo4jGraph


BATCH_SIZE = 500


def _sanitize_rel(rel: str) -> str:
    return re.sub(r"[^A-Z0-9_]", "_", rel.upper().strip()) or "RELATED_TO"


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", label.strip())
    return cleaned or "Class"


def _find_one(results_dir: Path, pattern: str) -> Path | None:
    matches = sorted(glob.glob(str(results_dir / pattern)))
    return Path(matches[-1]) if matches else None


def _load_artifacts(results_dir: Path) -> tuple[dict, dict, dict]:
    zone2_path = results_dir / "zone2_run_summary.json"
    if not zone2_path.exists():
        raise FileNotFoundError(f"missing {zone2_path}")

    svloi_path = _find_one(results_dir, "zone3_svloi_summary_*.json")
    prov_path = _find_one(results_dir, "svloi_provenance_*.json")

    zone2 = json.loads(zone2_path.read_text())
    svloi = json.loads(svloi_path.read_text()) if svloi_path else {}
    prov = json.loads(prov_path.read_text()) if prov_path else {}

    print(f"  zone2 triples:      {len(zone2.get('triples', []))}")
    print(f"  svloi classes:      {len(svloi.get('classes_final', []))}")
    print(f"  provenance entries: {len(prov)}")
    return zone2, svloi, prov


def _wipe(graph: Neo4jGraph) -> None:
    print("[1/4] Wiping graph...")
    graph.query("MATCH (n) DETACH DELETE n")
    print("  ✓ cleared")


def _write_triples(graph: Neo4jGraph, triples: list[dict]) -> int:
    print(f"[2/4] Writing {len(triples)} Zone 2 triples...")
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        by_rel[_sanitize_rel(t.get("relation", ""))].append(t)

    total = 0
    for rel_type, rows in by_rel.items():
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            graph.query(
                f"""
                UNWIND $batch AS row
                MERGE (s:Entity {{id: row.subject}})
                    ON CREATE SET s.entity_type = row.subject_type,
                                  s.source_type = row.source_type
                MERGE (o:Entity {{id: row.object}})
                    ON CREATE SET o.entity_type = row.object_type,
                                  o.source_type = row.source_type
                MERGE (s)-[r:{rel_type}]->(o)
                    ON CREATE SET r.span = row.span,
                                  r.confidence = row.confidence,
                                  r.chunk_id = row.chunk_id,
                                  r.source = row.source
                """,
                params={"batch": batch},
            )
            total += len(batch)
    print(f"  ✓ {total} triples across {len(by_rel)} relation types")
    return total


def _wipe_chunks(graph: Neo4jGraph) -> None:
    """Drop existing :Chunk nodes so we can re-key with composite (id, source)."""
    graph.query("MATCH (c:Chunk) DETACH DELETE c")


def _looks_like_flood_kg(graph: Neo4jGraph) -> bool:
    """Heuristic: does this KG actually contain entities sourced from the
    flood PDF? If not, don't load FEMA chunks — they'd be unrelated noise."""
    rows = graph.query(
        """
        MATCH ()-[r]-()
        WHERE r.source CONTAINS 'fema' OR r.source CONTAINS 'SFIP'
        RETURN count(r) AS n LIMIT 1
        """
    )
    return bool(rows and rows[0]["n"] > 0)


def _write_chunks(graph: Neo4jGraph, chunks_path: Path) -> int:
    """Load source chunks as :Chunk {id, source, text} nodes.

    Keyed by composite (id, source) so chunk ids can collide across
    different source files (e.g. PDF chunk #16 vs CSV row #16).
    """
    if not chunks_path.exists():
        print(f"  (skip chunks: {chunks_path} not found)")
        return 0
    if not _looks_like_flood_kg(graph):
        print(f"  (skip {chunks_path.name}: KG has no flood-sourced triples)")
        return 0

    data = json.loads(chunks_path.read_text())
    chunks = data if isinstance(data, list) else data.get("chunks", [])
    rows: list[dict] = []
    for c in chunks:
        cid = c.get("chunk_id")
        if cid is None:
            continue
        text = c.get("content") or c.get("text", "")
        src = c.get("source") or chunks_path.stem
        rows.append({"id": str(cid), "text": text, "source": src})

    print(f"[1.5/5] Writing {len(rows)} :Chunk nodes from {chunks_path.name}...")
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        graph.query(
            """
            UNWIND $batch AS row
            MERGE (c:Chunk {id: row.id, source: row.source})
              ON CREATE SET c.text = row.text
            """,
            params={"batch": batch},
        )
    print(f"  ✓ {len(rows)} chunk nodes")
    return len(rows)


def _write_csv_chunk_stubs(graph: Neo4jGraph) -> int:
    """Create :Chunk stubs for any (chunk_id, source) referenced by a
    relationship that doesn't already have a matching :Chunk node.

    Composite key matches `_write_chunks` so PDF chunk #16 and CSV row #16
    don't collide.
    """
    result = graph.query(
        """
        MATCH ()-[r]-()
        WHERE r.chunk_id IS NOT NULL
        WITH DISTINCT toString(r.chunk_id) AS cid, coalesce(r.source, '') AS src
        WHERE NOT EXISTS {
          MATCH (c:Chunk {id: cid, source: src})
        }
        MERGE (c:Chunk {id: cid, source: src})
          ON CREATE SET c.text = '(structured record — see source file)'
        RETURN count(c) AS n
        """
    )
    n = result[0]["n"] if result else 0
    print(f"  ✓ {n} :Chunk stubs for source-keyed (chunk_id, source) pairs")
    return n


def _write_classes(graph: Neo4jGraph, svloi: dict) -> dict[str, int]:
    classes = svloi.get("classes_final", []) or []
    hierarchy = svloi.get("hierarchy", []) or []
    associations = svloi.get("associations", []) or []

    print(f"[3/4] Writing {len(classes)} :OntologyClass nodes + ontology edges...")
    if classes:
        graph.query(
            "UNWIND $names AS n MERGE (:OntologyClass {name: n})",
            params={"names": [_sanitize_label(c) for c in classes]},
        )

    sub_edges = [
        {"child": _sanitize_label(h["child"]), "parent": _sanitize_label(h["parent"])}
        for h in hierarchy
        if h.get("type") == "SUBCLASS_OF" and h.get("child") and h.get("parent")
    ]
    if sub_edges:
        graph.query(
            """
            UNWIND $edges AS e
            MATCH (c:OntologyClass {name: e.child})
            MATCH (p:OntologyClass {name: e.parent})
            MERGE (c)-[:SUBCLASS_OF]->(p)
            """,
            params={"edges": sub_edges},
        )

    assoc_edges = [
        {"src": _sanitize_label(a["source"]), "tgt": _sanitize_label(a["target"])}
        for a in associations
        if a.get("source") and a.get("target")
    ]
    if assoc_edges:
        graph.query(
            """
            UNWIND $edges AS e
            MATCH (s:OntologyClass {name: e.src})
            MATCH (t:OntologyClass {name: e.tgt})
            MERGE (s)-[:ASSOCIATED_WITH]->(t)
            """,
            params={"edges": assoc_edges},
        )

    return {
        "classes": len(classes),
        "subclass_edges": len(sub_edges),
        "associated_edges": len(assoc_edges),
    }


def _write_instance_of(graph: Neo4jGraph, provenance: dict) -> int:
    rows: list[dict[str, str]] = []
    for entity_id, meta in provenance.items():
        if not isinstance(meta, dict):
            continue
        cls = meta.get("final_type") or meta.get("llm_type")
        if not cls or cls == "Other":
            continue
        rows.append({"id": entity_id, "cls": _sanitize_label(cls)})

    print(f"[4/5] Writing INSTANCE_OF for {len(rows)} entities...")
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        graph.query(
            """
            UNWIND $batch AS row
            MATCH (e:Entity {id: row.id})
            MATCH (c:OntologyClass {name: row.cls})
            MERGE (e)-[:INSTANCE_OF]->(c)
            """,
            params={"batch": batch},
        )

    # Also SET the class as a Neo4j label on each entity so that
    # `CALL db.labels()` (used by riskine_eval) sees the induced classes.
    # Neo4j doesn't support dynamic labels in plain Cypher, so iterate
    # per class — cheap (≤20 classes typical).
    distinct_classes = sorted({r["cls"] for r in rows})
    for cname in distinct_classes:
        graph.query(
            f"MATCH (e:Entity)-[:INSTANCE_OF]->(:OntologyClass {{name:$n}}) SET e:`{cname}`",
            params={"n": cname},
        )
    return len(rows)


def _propagate_to_untyped(graph: Neo4jGraph) -> int:
    """Propagate class labels to :Entity nodes that didn't receive an
    INSTANCE_OF edge from SV-LOI provenance.

    These are typically raw structured-triple objects (distinct date/amount/
    channel literals) that SV-LOI never included in its canonical set. We
    assign each one the class of its most-common connected neighbor that
    already has a class. One hop, undirected, majority vote.
    """
    untyped = graph.query(
        """
        MATCH (e:Entity)
        WHERE NOT (e)-[:INSTANCE_OF]->(:OntologyClass)
        RETURN count(e) AS n
        """
    )[0]["n"]

    if untyped == 0:
        print("[5/5] Propagating INSTANCE_OF to untyped entities... none to type")
        return 0

    result = graph.query(
        """
        MATCH (e:Entity)
        WHERE NOT (e)-[:INSTANCE_OF]->(:OntologyClass)
        MATCH (e)--(neighbor:Entity)-[:INSTANCE_OF]->(c:OntologyClass)
        WITH e, c.name AS cls, count(*) AS votes
        ORDER BY e, votes DESC
        WITH e, head(collect(cls)) AS best
        MATCH (bc:OntologyClass {name: best})
        MERGE (e)-[:INSTANCE_OF]->(bc)
        RETURN count(*) AS labeled
        """
    )
    labeled = result[0]["labeled"] if result else 0
    print(f"[5/5] Propagating INSTANCE_OF to untyped entities... {labeled}/{untyped} typed via neighbor vote")
    return labeled


def load(results_dir: Path, wipe: bool = True) -> dict[str, Any]:
    print(f"Target: {config.NEO4J_URI} (db={config.NEO4J_DATABASE})")
    print(f"Source: {results_dir}")
    zone2, svloi, prov = _load_artifacts(results_dir)

    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )

    if wipe:
        _wipe(graph)

    triples_written = _write_triples(graph, zone2.get("triples", []))
    chunks_path = Path(__file__).resolve().parents[1] / config.ZONE1_CHUNKS_FILE
    _wipe_chunks(graph)  # composite-key migration: drop old single-key chunks
    _write_chunks(graph, chunks_path)
    _write_csv_chunk_stubs(graph)
    class_stats = _write_classes(graph, svloi)
    instance_edges = _write_instance_of(graph, prov)
    propagated_edges = _propagate_to_untyped(graph)

    node_count = graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
    rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]

    stats = {
        "target_uri": config.NEO4J_URI,
        "triples_written": triples_written,
        "classes": class_stats["classes"],
        "subclass_of_edges": class_stats["subclass_edges"],
        "associated_with_edges": class_stats["associated_edges"],
        "instance_of_edges": instance_edges,
        "instance_of_propagated": propagated_edges,
        "total_nodes": node_count,
        "total_relationships": rel_count,
    }
    print("\n✓ Load complete:")
    for k, v in stats.items():
        print(f"  {k:24s} {v}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Results directory containing zone2_run_summary.json and zone3_svloi_summary_*.json",
    )
    ap.add_argument(
        "--no-wipe",
        action="store_true",
        help="Incremental mode — do not clear the target DB first",
    )
    args = ap.parse_args()

    if not args.results.is_dir():
        print(f"error: {args.results} is not a directory", file=sys.stderr)
        return 2

    load(args.results, wipe=not args.no_wipe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
