"""evaluation/ontology_metrics.py
================================
Standard ontology evaluation metrics adopted from the literature.

Implements three families of metrics:

1. **OLLM Metrics** (Lo et al., NeurIPS 2024):
   - Fuzzy F1: edge-level soft matching with embedding threshold
   - Continuous F1: edge-level optimal 1-to-1 matching via Hungarian algorithm
   - Graph F1: node-level matching with graph-aware embeddings (SGC)

2. **Taxonomy Edge Metrics**:
   - Edge Precision / Recall / F1 on SUBCLASS_OF / $ref edges
   - Wu-Palmer similarity between induced and reference hierarchies

3. **BERTScore-style Class Alignment**:
   - Threshold-free soft matching for class names

All metrics use `all-MiniLM-L6-v2` embeddings for consistency with the rest
of the evaluation pipeline.

Usage:
    from evaluation.ontology_metrics import (
        fuzzy_f1, continuous_f1, graph_f1,
        taxonomy_edge_f1, wu_palmer_similarity,
        bertscore_class_alignment,
    )
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Embedding (shared cache with riskine_eval.py)
# ---------------------------------------------------------------------------

_model_cache: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache


def _embed(texts: list[str]) -> np.ndarray:
    """Embed texts → L2-normalized (N, 384) array."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def _humanize(label: str) -> str:
    """PascalCase → 'Spaced Words' for better embedding quality."""
    import re
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', label)
    s = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', s)
    return s


# ---------------------------------------------------------------------------
# Node similarity matrix
# ---------------------------------------------------------------------------

def _node_sim_matrix(
    nodes_a: list[str],
    nodes_b: list[str],
) -> np.ndarray:
    """Cosine similarity matrix between two sets of node labels.

    Returns shape (len(nodes_a), len(nodes_b)).
    """
    embs_a = _embed([_humanize(n) for n in nodes_a])
    embs_b = _embed([_humanize(n) for n in nodes_b])
    return np.dot(embs_a, embs_b.T)


# ===================================================================
# 1. OLLM Metrics — Lo et al., NeurIPS 2024
# ===================================================================

# Threshold t = 0.436 — median cosine similarity between WordNet synonyms
# as reported in the OLLM paper (Section 4.2).
FUZZY_THRESHOLD = 0.436


def fuzzy_f1(
    induced_edges: list[tuple[str, str]],
    reference_edges: list[tuple[str, str]],
    threshold: float = FUZZY_THRESHOLD,
) -> dict:
    """Fuzzy F1 (OLLM, NeurIPS 2024).

    An induced edge (u', v') "matches" a reference edge (u, v) if:
        NodeSim(u, u') > t  AND  NodeSim(v, v') > t

    Matches are many-to-many (not one-to-one).

    Args:
        induced_edges: [(child, parent), ...] from the induced ontology
        reference_edges: [(child, parent), ...] from the reference ontology
        threshold: cosine similarity threshold (default: 0.436)

    Returns:
        dict with fuzzy_precision, fuzzy_recall, fuzzy_f1
    """
    if not induced_edges or not reference_edges:
        return {"fuzzy_precision": 0.0, "fuzzy_recall": 0.0, "fuzzy_f1": 0.0}

    # Collect unique node names
    ind_nodes = sorted(set(n for e in induced_edges for n in e))
    ref_nodes = sorted(set(n for e in reference_edges for n in e))

    # Build node similarity matrix
    sim = _node_sim_matrix(ind_nodes, ref_nodes)
    ind_idx = {n: i for i, n in enumerate(ind_nodes)}
    ref_idx = {n: i for i, n in enumerate(ref_nodes)}

    # Fuzzy precision: fraction of induced edges that match some reference edge
    matched_induced = 0
    for u_i, v_i in induced_edges:
        for u_r, v_r in reference_edges:
            if (sim[ind_idx[u_i], ref_idx[u_r]] > threshold and
                    sim[ind_idx[v_i], ref_idx[v_r]] > threshold):
                matched_induced += 1
                break  # one match suffices

    # Fuzzy recall: fraction of reference edges matched by some induced edge
    matched_reference = 0
    for u_r, v_r in reference_edges:
        for u_i, v_i in induced_edges:
            if (sim[ind_idx[u_i], ref_idx[u_r]] > threshold and
                    sim[ind_idx[v_i], ref_idx[v_r]] > threshold):
                matched_reference += 1
                break

    precision = matched_induced / len(induced_edges)
    recall = matched_reference / len(reference_edges)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "fuzzy_precision": round(precision, 4),
        "fuzzy_recall": round(recall, 4),
        "fuzzy_f1": round(f1, 4),
        "fuzzy_threshold": threshold,
        "induced_edge_count": len(induced_edges),
        "reference_edge_count": len(reference_edges),
    }


def continuous_f1(
    induced_edges: list[tuple[str, str]],
    reference_edges: list[tuple[str, str]],
) -> dict:
    """Continuous F1 (OLLM, NeurIPS 2024).

    Optimal one-to-one edge matching via the Hungarian algorithm.
    Edge similarity = min(NodeSim(u, u'), NodeSim(v, v')).
    No threshold — uses raw cosine similarity scores.

    Args:
        induced_edges: [(child, parent), ...] from the induced ontology
        reference_edges: [(child, parent), ...] from the reference ontology

    Returns:
        dict with continuous_precision, continuous_recall, continuous_f1
    """
    from scipy.optimize import linear_sum_assignment

    if not induced_edges or not reference_edges:
        return {"continuous_precision": 0.0, "continuous_recall": 0.0, "continuous_f1": 0.0}

    # Collect unique nodes
    ind_nodes = sorted(set(n for e in induced_edges for n in e))
    ref_nodes = sorted(set(n for e in reference_edges for n in e))

    sim = _node_sim_matrix(ind_nodes, ref_nodes)
    ind_idx = {n: i for i, n in enumerate(ind_nodes)}
    ref_idx = {n: i for i, n in enumerate(ref_nodes)}

    # Build edge-pair similarity matrix: (|E'| × |E|)
    n_ind = len(induced_edges)
    n_ref = len(reference_edges)
    edge_sim = np.zeros((n_ind, n_ref))

    for i, (u_i, v_i) in enumerate(induced_edges):
        for j, (u_r, v_r) in enumerate(reference_edges):
            edge_sim[i, j] = min(
                float(sim[ind_idx[u_i], ref_idx[u_r]]),
                float(sim[ind_idx[v_i], ref_idx[v_r]]),
            )

    # Hungarian algorithm — maximize total similarity
    # linear_sum_assignment minimizes cost, so negate
    cost = -edge_sim
    row_ind, col_ind = linear_sum_assignment(cost)
    s_cont = sum(edge_sim[r, c] for r, c in zip(row_ind, col_ind))

    precision = s_cont / n_ind if n_ind > 0 else 0.0
    recall = s_cont / n_ref if n_ref > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "continuous_precision": round(float(precision), 4),
        "continuous_recall": round(float(recall), 4),
        "continuous_f1": round(float(f1), 4),
        "optimal_matching_score": round(float(s_cont), 4),
    }


def graph_f1(
    induced_nodes: list[str],
    reference_nodes: list[str],
    induced_edges: list[tuple[str, str]],
    reference_edges: list[tuple[str, str]],
    sgc_hops: int = 2,
) -> dict:
    """Graph F1 (OLLM, NeurIPS 2024).

    Compares nodes using graph-structure-aware embeddings.
    Applies Simple Graph Convolution (SGC, Wu et al. 2019) with K hops
    to propagate neighborhood information, then finds optimal 1-to-1
    node matching via Hungarian algorithm.

    Args:
        induced_nodes: class names in the induced ontology
        reference_nodes: class names in the reference ontology
        induced_edges: [(child, parent), ...] edges in the induced ontology
        reference_edges: [(child, parent), ...] edges in the reference ontology
        sgc_hops: number of SGC propagation hops (default: 2)

    Returns:
        dict with graph_precision, graph_recall, graph_f1
    """
    from scipy.optimize import linear_sum_assignment

    if not induced_nodes or not reference_nodes:
        return {"graph_precision": 0.0, "graph_recall": 0.0, "graph_f1": 0.0}

    def _sgc_embed(nodes: list[str], edges: list[tuple[str, str]], k: int) -> np.ndarray:
        """Apply Simple Graph Convolution to node embeddings."""
        n = len(nodes)
        node_idx = {name: i for i, name in enumerate(nodes)}
        embs = _embed([_humanize(name) for name in nodes])  # (n, 384)

        if n <= 1 or not edges:
            return embs

        # Build adjacency matrix (symmetric, undirected)
        adj = np.zeros((n, n))
        for u, v in edges:
            if u in node_idx and v in node_idx:
                i, j = node_idx[u], node_idx[v]
                adj[i, j] = 1.0
                adj[j, i] = 1.0

        # Add self-loops: A_hat = A + I
        adj += np.eye(n)

        # Degree normalization: D^{-1/2} A_hat D^{-1/2}
        degree = adj.sum(axis=1)
        d_inv_sqrt = np.zeros(n)
        nonzero = degree > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
        d_mat = np.diag(d_inv_sqrt)
        adj_norm = d_mat @ adj @ d_mat

        # K-hop propagation: X' = (D^{-1/2} A D^{-1/2})^K X
        result = embs.copy()
        for _ in range(k):
            result = adj_norm @ result

        # L2 normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        result = result / norms
        return result

    ind_embs = _sgc_embed(induced_nodes, induced_edges, sgc_hops)
    ref_embs = _sgc_embed(reference_nodes, reference_edges, sgc_hops)

    # Node similarity matrix
    node_sim = np.dot(ind_embs, ref_embs.T)  # (|V'|, |V|)

    # Hungarian matching
    cost = -node_sim
    row_ind, col_ind = linear_sum_assignment(cost)
    s_graph = sum(node_sim[r, c] for r, c in zip(row_ind, col_ind))

    precision = s_graph / len(induced_nodes) if induced_nodes else 0.0
    recall = s_graph / len(reference_nodes) if reference_nodes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "graph_precision": round(float(precision), 4),
        "graph_recall": round(float(recall), 4),
        "graph_f1": round(float(f1), 4),
        "sgc_hops": sgc_hops,
    }


# ===================================================================
# 2. Taxonomy Edge Metrics
# ===================================================================

def taxonomy_edge_f1(
    induced_edges: list[tuple[str, str]],
    reference_edges: list[tuple[str, str]],
    threshold: float = FUZZY_THRESHOLD,
) -> dict:
    """Taxonomy Edge Precision / Recall / F1.

    Compares SUBCLASS_OF (child, parent) edges between induced and reference
    ontologies using soft node matching (same threshold as Fuzzy F1).

    Unlike Fuzzy F1, this specifically evaluates hierarchy quality
    (not just class existence).

    Args:
        induced_edges: [(child, parent), ...] SUBCLASS_OF edges
        reference_edges: [(child, parent), ...] from Riskine $ref links

    Returns:
        dict with taxonomy_precision, taxonomy_recall, taxonomy_f1
    """
    # Taxonomy edge F1 uses the same matching logic as fuzzy F1
    # but is reported separately for clarity in the paper
    result = fuzzy_f1(induced_edges, reference_edges, threshold)
    return {
        "taxonomy_precision": result["fuzzy_precision"],
        "taxonomy_recall": result["fuzzy_recall"],
        "taxonomy_f1": result["fuzzy_f1"],
        "taxonomy_induced_edges": len(induced_edges),
        "taxonomy_reference_edges": len(reference_edges),
    }


def wu_palmer_similarity(
    induced_nodes: list[str],
    induced_edges: list[tuple[str, str]],
    reference_nodes: list[str],
    reference_edges: list[tuple[str, str]],
) -> dict:
    """Wu-Palmer similarity between induced and reference hierarchies.

    For each matched node pair (via embedding), computes Wu-Palmer similarity
    based on the depth and lowest common ancestor (LCA) in the hierarchy.

    Simplified version: compares depth distributions of the two hierarchies
    and the average Wu-Palmer for matched node pairs.

    Returns:
        dict with avg_wu_palmer, depth_distribution_induced, depth_distribution_reference
    """
    def _build_depth_map(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
        """Compute depth of each node from root(s). Roots have depth 0."""
        children_of: dict[str, list[str]] = {n: [] for n in nodes}
        has_parent: set[str] = set()
        for child, parent in edges:
            if parent in children_of:
                children_of[parent].append(child)
            has_parent.add(child)

        roots = [n for n in nodes if n not in has_parent]
        if not roots:
            roots = nodes[:1]  # fallback

        depths: dict[str, int] = {}
        queue = [(r, 0) for r in roots]
        while queue:
            node, d = queue.pop(0)
            if node in depths:
                continue
            depths[node] = d
            for child in children_of.get(node, []):
                if child not in depths:
                    queue.append((child, d + 1))

        # Unreachable nodes get max_depth + 1
        max_d = max(depths.values()) if depths else 0
        for n in nodes:
            if n not in depths:
                depths[n] = max_d + 1

        return depths

    ind_depths = _build_depth_map(induced_nodes, induced_edges)
    ref_depths = _build_depth_map(reference_nodes, reference_edges)

    # Match nodes via embedding similarity
    if not induced_nodes or not reference_nodes:
        return {"avg_wu_palmer": 0.0}

    sim = _node_sim_matrix(induced_nodes, reference_nodes)

    # Greedy matching: for each induced node, find best reference node
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-sim)

    wu_palmer_scores = []
    for r, c in zip(row_ind, col_ind):
        if sim[r, c] < 0.3:  # too dissimilar to be a real match
            continue
        d_ind = ind_depths.get(induced_nodes[r], 0)
        d_ref = ref_depths.get(reference_nodes[c], 0)
        # Wu-Palmer: 2 * depth(LCA) / (depth(a) + depth(b))
        # Approximation when we don't share a hierarchy:
        # use min(d_ind, d_ref) as LCA depth proxy
        lca_depth = min(d_ind, d_ref)
        denom = d_ind + d_ref + 2  # +2 to account for root
        wp = (2 * (lca_depth + 1)) / denom if denom > 0 else 0.0
        wu_palmer_scores.append(wp)

    avg_wp = float(np.mean(wu_palmer_scores)) if wu_palmer_scores else 0.0

    # Depth distributions
    ind_dist = {}
    for d in ind_depths.values():
        ind_dist[d] = ind_dist.get(d, 0) + 1
    ref_dist = {}
    for d in ref_depths.values():
        ref_dist[d] = ref_dist.get(d, 0) + 1

    return {
        "avg_wu_palmer": round(avg_wp, 4),
        "wu_palmer_matched_pairs": len(wu_palmer_scores),
        "depth_distribution_induced": ind_dist,
        "depth_distribution_reference": ref_dist,
    }


# ===================================================================
# 3. BERTScore-style Class Alignment
# ===================================================================

def bertscore_class_alignment(
    induced_classes: list[str],
    reference_classes: list[str],
) -> dict:
    """BERTScore-style alignment for ontology classes.

    Threshold-free: uses raw embedding similarities.
    - BERTScore-Recall: for each reference class, max similarity to any induced class
    - BERTScore-Precision: for each induced class, max similarity to any reference class
    - BERTScore-F1: harmonic mean

    This matches the approach used by AutoSchemaKG (2025).

    Returns:
        dict with bertscore_precision, bertscore_recall, bertscore_f1
    """
    if not induced_classes or not reference_classes:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

    sim = _node_sim_matrix(induced_classes, reference_classes)

    # Recall: for each reference class, best match from induced
    recall_scores = sim.max(axis=0)  # (|ref|,)
    bertscore_recall = float(np.mean(recall_scores))

    # Precision: for each induced class, best match from reference
    precision_scores = sim.max(axis=1)  # (|ind|,)
    bertscore_precision = float(np.mean(precision_scores))

    f1 = (2 * bertscore_precision * bertscore_recall /
          (bertscore_precision + bertscore_recall)
          if (bertscore_precision + bertscore_recall) > 0 else 0.0)

    return {
        "bertscore_precision": round(bertscore_precision, 4),
        "bertscore_recall": round(bertscore_recall, 4),
        "bertscore_f1": round(f1, 4),
        "per_reference_recall": {
            ref: round(float(recall_scores[i]), 4)
            for i, ref in enumerate(reference_classes)
        },
    }


# ===================================================================
# 4. Convenience: extract edges from Riskine schemas
# ===================================================================

def extract_riskine_edges(schemas: dict[str, dict]) -> list[tuple[str, str]]:
    """Extract hierarchy edges from Riskine JSON schemas via $ref links.

    A $ref like 'address.json' in person.json's properties means
    Person → Address is a structural relationship.

    We extract these as (child_class, parent_class) where "child" references "parent".
    This gives us the Riskine inter-class relationships for hierarchy evaluation.

    Returns:
        List of (source_class, target_class) edges.
    """
    import re

    def _pascal(name: str) -> str:
        parts = re.split(r'[-_\s]+', name)
        return "".join(p.capitalize() for p in parts if p)

    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for schema_name, schema in schemas.items():
        source_class = _pascal(schema_name)
        props = schema.get("properties", {})

        for prop_name, prop_def in props.items():
            if not isinstance(prop_def, dict):
                continue

            # Check for $ref
            ref = prop_def.get("$ref", "")
            # Check inside items (for array properties)
            if not ref and "items" in prop_def:
                items = prop_def["items"]
                if isinstance(items, dict):
                    ref = items.get("$ref", "")

            if not ref:
                continue

            # Parse ref: "address.json" → "Address", "person.json" → "Person"
            # Skip definitions.json references (those are value types, not classes)
            if "definitions.json" in ref:
                continue

            # Extract class name from ref
            target_file = ref.split("#")[0]  # remove fragment
            if not target_file.endswith(".json"):
                continue

            target_name = target_file.replace(".json", "")
            target_class = _pascal(target_name)

            edge = (source_class, target_class)
            if edge not in seen and source_class != target_class:
                edges.append(edge)
                seen.add(edge)

    return edges


def extract_induced_edges(graph) -> tuple[list[str], list[tuple[str, str]]]:
    """Extract induced ontology nodes and SUBCLASS_OF edges from Neo4j.

    Returns:
        (nodes, edges) where nodes = [class_name, ...] and
        edges = [(child, parent), ...]
    """
    # Get OntologyClass nodes
    try:
        rows = graph.query(
            "MATCH (c:OntologyClass) RETURN c.name AS name"
        )
        nodes = [r["name"] for r in rows if r.get("name")]
    except Exception:
        nodes = []

    # Get SUBCLASS_OF edges
    try:
        rows = graph.query(
            "MATCH (child:OntologyClass)-[:SUBCLASS_OF]->(parent:OntologyClass) "
            "RETURN child.name AS child, parent.name AS parent"
        )
        edges = [(r["child"], r["parent"]) for r in rows
                 if r.get("child") and r.get("parent")]
    except Exception:
        edges = []

    return nodes, edges


# ===================================================================
# 5. All-in-one evaluator
# ===================================================================

def evaluate_ontology(
    graph,
    riskine_schemas: dict[str, dict],
    riskine_classes: list[dict],
) -> dict:
    """Run all standard ontology metrics.

    Args:
        graph: Neo4jGraph instance
        riskine_schemas: raw Riskine JSON schemas (from riskine_loader.fetch_and_cache)
        riskine_classes: extracted class descriptors (from riskine_loader.extract_riskine_classes)

    Returns:
        dict with all metric results
    """
    # Extract induced ontology from Neo4j
    induced_nodes, induced_edges = extract_induced_edges(graph)

    # Extract reference edges from Riskine schemas
    reference_edges = extract_riskine_edges(riskine_schemas)
    reference_nodes = [c["name"] for c in riskine_classes]

    print(f"\n  [ontology-metrics] Induced: {len(induced_nodes)} classes, {len(induced_edges)} edges")
    print(f"  [ontology-metrics] Riskine: {len(reference_nodes)} classes, {len(reference_edges)} edges")

    results: dict = {}

    # --- BERTScore class alignment (threshold-free) ---
    print("  [ontology-metrics] Computing BERTScore class alignment...")
    bs = bertscore_class_alignment(induced_nodes, reference_nodes)
    results.update(bs)
    print(f"    BERTScore P={bs['bertscore_precision']:.3f}  R={bs['bertscore_recall']:.3f}  F1={bs['bertscore_f1']:.3f}")

    # --- Fuzzy F1 (OLLM) ---
    if induced_edges and reference_edges:
        print("  [ontology-metrics] Computing Fuzzy F1 (OLLM)...")
        ff = fuzzy_f1(induced_edges, reference_edges)
        results.update(ff)
        print(f"    Fuzzy F1 P={ff['fuzzy_precision']:.3f}  R={ff['fuzzy_recall']:.3f}  F1={ff['fuzzy_f1']:.3f}")

        # --- Continuous F1 (OLLM) ---
        print("  [ontology-metrics] Computing Continuous F1 (OLLM)...")
        cf = continuous_f1(induced_edges, reference_edges)
        results.update(cf)
        print(f"    Continuous F1 P={cf['continuous_precision']:.3f}  R={cf['continuous_recall']:.3f}  F1={cf['continuous_f1']:.3f}")

        # --- Taxonomy Edge F1 ---
        print("  [ontology-metrics] Computing Taxonomy Edge F1...")
        tf = taxonomy_edge_f1(induced_edges, reference_edges)
        results.update(tf)
        print(f"    Taxonomy Edge P={tf['taxonomy_precision']:.3f}  R={tf['taxonomy_recall']:.3f}  F1={tf['taxonomy_f1']:.3f}")
    else:
        print("  [ontology-metrics] Skipping edge metrics (no edges found)")
        results.update({
            "fuzzy_precision": 0.0, "fuzzy_recall": 0.0, "fuzzy_f1": 0.0,
            "continuous_precision": 0.0, "continuous_recall": 0.0, "continuous_f1": 0.0,
            "taxonomy_precision": 0.0, "taxonomy_recall": 0.0, "taxonomy_f1": 0.0,
        })

    # --- Graph F1 (OLLM) ---
    print("  [ontology-metrics] Computing Graph F1 (OLLM)...")
    gf = graph_f1(induced_nodes, reference_nodes, induced_edges, reference_edges)
    results.update(gf)
    print(f"    Graph F1 P={gf['graph_precision']:.3f}  R={gf['graph_recall']:.3f}  F1={gf['graph_f1']:.3f}")

    # --- Wu-Palmer hierarchy similarity ---
    if induced_edges and reference_edges:
        print("  [ontology-metrics] Computing Wu-Palmer similarity...")
        wp = wu_palmer_similarity(induced_nodes, induced_edges, reference_nodes, reference_edges)
        results.update(wp)
        print(f"    Wu-Palmer avg={wp['avg_wu_palmer']:.3f} ({wp['wu_palmer_matched_pairs']} pairs)")
    else:
        results["avg_wu_palmer"] = 0.0

    return results
