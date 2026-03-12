"""
Zone 2.5 — Entity Resolution via Embedding Similarity

Groups semantically near-duplicate Zone2Entity nodes and merges them into a
single canonical node, redirecting all relationships.

Algorithm:
  1. Embed all node IDs with sentence-transformers all-MiniLM-L6-v2
  2. Compute pairwise cosine similarity (L2-normalised embeddings → dot product)
  3. Collect pairs with similarity ≥ threshold (default 0.90 — conservative)
  4. Build connected components via union-find
  5. Elect canonical node per component (shortest ID, then alphabetical tiebreak)
  6. Redirect all relationships from duplicates → canonical (pure Cypher, no APOC)
  7. Delete duplicate nodes

Usage (called from zone2/pipeline.py as pipeline node zone25_entity_resolution):
  from zone2.entity_resolution import resolve_entities
"""

import re
from collections import defaultdict
import numpy as np

SIMILARITY_THRESHOLD = 0.90  # conservative: only merge near-identical node names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_rel(rel: str) -> str:
    """Make a relation name safe for Neo4j Cypher f-string interpolation."""
    return re.sub(r'[^A-Z0-9_]', '_', rel.upper().strip())


def _union_find_components(pairs: list[tuple[str, str]]) -> list[list[str]]:
    """Build connected components from (a, b) similar-node pairs via union-find."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict[str, list[str]] = defaultdict(list)
    for node in {n for pair in pairs for n in pair}:
        groups[find(node)].append(node)
    return [grp for grp in groups.values() if len(grp) > 1]


def _merge_node_into_canonical(graph, dup_id: str, canon_id: str, graph_label: str = "Entity") -> int:
    """
    Redirect all relationships from dup → canonical node, then delete dup.
    Uses pure Cypher MERGE (no APOC) so it works on AuraDB free tier.
    Returns number of relationships redirected.
    """
    label = graph_label
    out_rels = graph.query(
        f"MATCH (n:{label} {{id: $id}})-[r]->(m) "
        "RETURN type(r) AS t, m.id AS mid, properties(r) AS p",
        params={"id": dup_id},
    )
    in_rels = graph.query(
        f"MATCH (m)-[r]->(n:{label} {{id: $id}}) "
        "RETURN type(r) AS t, m.id AS mid, properties(r) AS p",
        params={"id": dup_id},
    )
    moved = 0
    for row in out_rels:
        rt = _sanitize_rel(row["t"])
        graph.query(
            f"MATCH (c:{label} {{id: $cid}}) "
            f"MATCH (tgt:{label} {{id: $tid}}) "
            f"MERGE (c)-[r:{rt}]->(tgt) ON CREATE SET r = $p",
            params={"cid": canon_id, "tid": row["mid"], "p": row["p"]},
        )
        moved += 1
    for row in in_rels:
        rt = _sanitize_rel(row["t"])
        graph.query(
            f"MATCH (src:{label} {{id: $sid}}) "
            f"MATCH (c:{label} {{id: $cid}}) "
            f"MERGE (src)-[r:{rt}]->(c) ON CREATE SET r = $p",
            params={"sid": row["mid"], "cid": canon_id, "p": row["p"]},
        )
        moved += 1
    graph.query(
        f"MATCH (n:{label} {{id: $id}}) DETACH DELETE n",
        params={"id": dup_id},
    )
    return moved


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def resolve_entities(graph, threshold: float = SIMILARITY_THRESHOLD, node_label: str = "Entity") -> dict:
    """
    Embed all Zone2Entity node IDs and merge near-duplicates above cosine threshold.

    Returns stats dict: nodes_before, nodes_after, merged, pairs, groups.
    Near-duplicates are groups connected by similarity ≥ threshold; canonical is
    elected as the shortest (most concise) node ID in each group.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ⚠ sentence_transformers not available; skipping entity resolution")
        return {"error": "sentence_transformers not installed", "merged": 0}

    rows = graph.query(f"MATCH (n:{node_label}) RETURN n.id AS id")
    ids = [r["id"] for r in rows]
    n_before = len(ids)
    if n_before < 2:
        return {"merged": 0, "nodes_before": n_before, "nodes_after": n_before}

    print(f"  Embedding {n_before} node IDs with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(ids, normalize_embeddings=True)  # L2-norm → dot = cosine

    sim_pairs: list[tuple[str, str]] = []
    for i in range(n_before):
        for j in range(i + 1, n_before):
            if float(np.dot(embs[i], embs[j])) >= threshold:
                sim_pairs.append((ids[i], ids[j]))

    print(f"  Near-duplicate pairs (cosine ≥ {threshold}): {len(sim_pairs)}")
    if not sim_pairs:
        return {"merged": 0, "nodes_before": n_before, "nodes_after": n_before, "pairs": 0}

    components = _union_find_components(sim_pairs)
    total_merged = 0
    for group in components:
        canonical = min(group, key=lambda x: (len(x), x))  # shortest, then alpha
        for dup in sorted(group):
            if dup != canonical:
                moved = _merge_node_into_canonical(graph, dup, canonical, graph_label=node_label)
                print(f"    Merged '{dup}' → '{canonical}' ({moved} rels)")
                total_merged += 1

    after_rows = graph.query(f"MATCH (n:{node_label}) RETURN count(n) AS c")
    n_after = after_rows[0]["c"] if after_rows else n_before - total_merged
    return {
        "merged":       total_merged,
        "nodes_before": n_before,
        "nodes_after":  n_after,
        "pairs":        len(sim_pairs),
        "groups":       len(components),
    }
