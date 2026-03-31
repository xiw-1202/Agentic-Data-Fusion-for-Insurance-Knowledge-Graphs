"""
Zone 2.5 — Entity Resolution via Embedding Similarity (In-Memory)

Deduplicates near-identical entity names in the triple list BEFORE Neo4j
insertion. All processing happens in Python — zero Neo4j round-trips.

Algorithm:
  1. Collect unique node IDs from triple list (exclude structured prefixes)
  2. Embed with sentence-transformers all-MiniLM-L6-v2
  3. Vectorized pairwise cosine similarity (matrix multiply)
  4. Build connected components via union-find
  5. Elect canonical node per component (shortest ID, then alphabetical)
  6. Replace duplicate IDs in triple list with canonical IDs
  7. Return deduplicated triple list

Usage (called from zone2/pipeline.py):
  from zone2.entity_resolution import resolve_entities_in_memory
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

SIMILARITY_THRESHOLD = 0.90

# Structured node prefixes — never merge these.
STRUCTURED_PREFIXES = ("POL-", "CLM-", "REC-", "PER-", "PROP-")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# In-memory entity resolution (operates on triple list, no Neo4j)
# ---------------------------------------------------------------------------

def resolve_entities_in_memory(
    triples: list[dict[str, Any]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate near-identical entity names in the triple list.

    Operates entirely in Python — no Neo4j queries. Structured triples
    (source_type='structured') and their nodes are excluded from merging.

    Args:
        triples: List of triple dicts with 'subject', 'object', 'source_type' keys.
        threshold: Cosine similarity threshold for merging (default 0.90).

    Returns:
        (deduplicated_triples, stats_dict)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ⚠ sentence_transformers not available; skipping entity resolution")
        return triples, {"error": "sentence_transformers not installed", "merged": 0}

    # Collect all unique node IDs from triples.
    all_ids: set[str] = set()
    for t in triples:
        all_ids.add(t["subject"])
        all_ids.add(t["object"])

    n_total = len(all_ids)

    # Identify structured IDs to exclude.
    structured_ids: set[str] = set()
    for t in triples:
        if t.get("source_type") == "structured":
            structured_ids.add(t["subject"])
            structured_ids.add(t["object"])
        elif t["subject"].startswith(STRUCTURED_PREFIXES):
            structured_ids.add(t["subject"])
        elif t["object"].startswith(STRUCTURED_PREFIXES):
            structured_ids.add(t["object"])

    # Also exclude value nodes connected to structured subjects via HAS_ relations.
    for t in triples:
        if (t["subject"] in structured_ids
                and t["relation"].startswith("HAS_")):
            structured_ids.add(t["object"])

    # Filter to candidates for resolution.
    candidate_ids = sorted(all_ids - structured_ids)
    n_excluded = n_total - len(candidate_ids)

    if len(candidate_ids) < 2:
        print(f"  ℹ {n_excluded} structured nodes excluded, "
              f"{len(candidate_ids)} candidates — skipping resolution")
        return triples, {
            "merged": 0, "nodes_total": n_total,
            "structured_excluded": n_excluded,
        }

    print(f"  Embedding {len(candidate_ids)} node IDs with all-MiniLM-L6-v2... "
          f"({n_excluded} structured nodes excluded)")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(candidate_ids, normalize_embeddings=True)

    # Vectorized pairwise similarity (replaces O(n²) Python loop).
    sim_matrix = embs @ embs.T
    n = len(candidate_ids)
    # Zero out lower triangle + diagonal, keep upper triangle only.
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairs_i, pairs_j = np.where((sim_matrix >= threshold) & mask)

    sim_pairs = [(candidate_ids[i], candidate_ids[j])
                 for i, j in zip(pairs_i, pairs_j)]

    print(f"  Near-duplicate pairs (cosine ≥ {threshold}): {len(sim_pairs)}")

    if not sim_pairs:
        return triples, {
            "merged": 0, "nodes_total": n_total,
            "structured_excluded": n_excluded, "pairs": 0,
        }

    # Build merge map: duplicate_id → canonical_id.
    components = _union_find_components(sim_pairs)
    merge_map: dict[str, str] = {}
    total_merged = 0

    for group in components:
        canonical = min(group, key=lambda x: (len(x), x))
        for dup in sorted(group):
            if dup != canonical:
                merge_map[dup] = canonical
                print(f"    Merged '{dup}' → '{canonical}'")
                total_merged += 1

    # Apply merge map to all triples (immutable — create new list).
    deduplicated = [
        {
            **t,
            "subject": merge_map.get(t["subject"], t["subject"]),
            "object": merge_map.get(t["object"], t["object"]),
        }
        for t in triples
    ]

    # Remove self-referential triples created by merging.
    deduplicated = [t for t in deduplicated if t["subject"] != t["object"]]

    n_after = len(all_ids) - total_merged
    print(f"  ✓ {total_merged} merged: {n_total} → {n_after} unique nodes")

    stats = {
        "merged": total_merged,
        "nodes_total": n_total,
        "nodes_after": n_after,
        "structured_excluded": n_excluded,
        "pairs": len(sim_pairs),
        "groups": len(components),
        "merge_map": {k: v for k, v in list(merge_map.items())[:20]},  # sample for logging
    }

    return deduplicated, stats
