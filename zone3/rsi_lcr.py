"""Zone 3 — RSI-LCR: Relation-Signature Induction with LLM Coherence Refinement

Novel ontology induction method that bridges regular equivalence from social
network analysis with LLM coherence judgment for iterative cluster refinement.

Key insight: Entities of the same ontological class participate in the same
types of relations. "Coverage" entities have `covers`, `has_limit`, `excludes`;
"Person" entities have `has_name`, `resides_at`. Clustering by relation
patterns is more discriminative than embedding similarity.

Algorithm:
  1. Load entities + relation context from Neo4j
  2. Build relation signature vectors (relation_type × direction + entity type)
  3. Hierarchical Agglomerative Clustering (Ward linkage, cosine distance)
  4. LLM coherence refinement loop (2 rounds: judge → split/merge)
  5. LLM naming with convention calibration
  6. Hierarchy from dendrogram + LLM validation
  7. Write ontology to Neo4j

Usage:
  python3 zone3/rsi_lcr.py
  python3 zone3/rsi_lcr.py --model qwen2.5:72b
  python3 zone3/rsi_lcr.py --model qwen2.5:72b --suffix zone3_rsi
  python3 zone3/rsi_lcr.py --refinement-rounds 3
  python3 zone3/rsi_lcr.py --k 15  # force cluster count

Pre-requisite: Zone 2 must have run first (Neo4j populated with :Entity nodes).

Evaluation (run AFTER this pipeline):
  python3 baseline/eval.py --suffix zone3_rsi --riskine
"""

from __future__ import annotations

import json
import re
import time
import os
import sys
import argparse
from typing import Optional, Union
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cluster count search range (silhouette-guided)
K_MIN = 5
K_MAX = 40
K_DEFAULT = 15  # fallback if silhouette is flat

# LLM coherence refinement
COHERENCE_SPLIT_THRESHOLD = 3   # score < this → split cluster
COHERENCE_KEEP_THRESHOLD = 4    # score >= this → keep as-is
MERGE_SIMILARITY_THRESHOLD = 0.85  # cosine between centroids → merge
MAX_REFINEMENT_ROUNDS = 2

# Minimum cluster size (filter noise)
MIN_CLUSTER_SIZE = 2

# Max members to show LLM per cluster (prompt budget)
MAX_MEMBERS_IN_PROMPT = 15


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_llm(model: str) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
    )


def get_neo4j_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )


def _sanitize_label(name: str) -> str:
    """Make a PascalCase name safe for Neo4j label."""
    cleaned = re.sub(r'[^A-Za-z0-9]', '', name.strip())
    if not cleaned:
        return "UnknownClass"
    if cleaned[0].isdigit():
        cleaned = "Class" + cleaned
    return cleaned


def _parse_json_safely(text: str) -> Union[dict, list]:
    """Try to parse JSON from LLM output with fallbacks."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL) or re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Step 1: Load Entities from Neo4j
# ---------------------------------------------------------------------------

def load_entities() -> list[dict]:
    """Load all Entity nodes with their structural context."""
    print("\n[1/7] Loading entities from Neo4j...")
    graph = get_neo4j_graph()

    rows = graph.query("""
        MATCH (n:Entity)
        OPTIONAL MATCH (n)-[r]->(m:Entity)
        OPTIONAL MATCH (p:Entity)-[s]->(n)
        RETURN n.id AS id,
               n.entity_type AS entity_type,
               collect(DISTINCT {rel: type(r), target: m.id}) AS out_rels,
               collect(DISTINCT {rel: type(s), source: p.id}) AS in_rels
    """)
    if not rows:
        print("  ✗ No Entity nodes found. Run zone2/pipeline.py first.")
        return []

    entities: list[dict] = []
    for row in rows:
        eid = row["id"]
        etype = row.get("entity_type") or "Unknown"

        out_rels = [r for r in row.get("out_rels", []) if r.get("rel")]
        in_rels = [r for r in row.get("in_rels", []) if r.get("rel")]

        # Relation type counts (the core of RSI)
        out_counts: dict[str, int] = defaultdict(int)
        in_counts: dict[str, int] = defaultdict(int)
        for r in out_rels:
            out_counts[r["rel"]] += 1
        for r in in_rels:
            in_counts[r["rel"]] += 1

        entities.append({
            "id": eid,
            "entity_type": etype,
            "out_rel_counts": dict(out_counts),
            "in_rel_counts": dict(in_counts),
            "out_rel_types": list(out_counts.keys()),
            "in_rel_types": list(in_counts.keys()),
            "degree": len(out_rels) + len(in_rels),
        })

    typed = sum(1 for e in entities if e["entity_type"] != "Unknown")
    print(f"  ✓ {len(entities)} entities loaded ({typed} typed)")
    return entities


# ---------------------------------------------------------------------------
# Step 2: Build Relation Signature Vectors
# ---------------------------------------------------------------------------

def build_relation_signatures(entities: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build relation signature feature matrix.

    For each entity, the vector is:
      [count(r1_OUT), count(r1_IN), count(r2_OUT), count(r2_IN), ..., onehot(entity_type)]

    Returns:
      features: (n_entities, n_features) L2-normalized matrix
      entity_ids: list of entity IDs (row order)
      feature_names: list of feature names (column order)
    """
    print("\n[2/7] Building relation signature vectors...")

    # Collect all relation types and entity types across the graph
    all_rel_types: set[str] = set()
    all_entity_types: set[str] = set()
    for e in entities:
        all_rel_types.update(e["out_rel_types"])
        all_rel_types.update(e["in_rel_types"])
        if e["entity_type"] != "Unknown":
            all_entity_types.add(e["entity_type"])

    rel_types_sorted = sorted(all_rel_types)
    entity_types_sorted = sorted(all_entity_types)

    # Feature names
    feature_names: list[str] = []
    for rt in rel_types_sorted:
        feature_names.append(f"{rt}_OUT")
        feature_names.append(f"{rt}_IN")
    for et in entity_types_sorted:
        feature_names.append(f"type_{et}")

    n_rel_features = len(rel_types_sorted) * 2
    n_type_features = len(entity_types_sorted)
    n_features = n_rel_features + n_type_features

    print(f"  Relation types: {len(rel_types_sorted)}")
    print(f"  Entity types: {len(entity_types_sorted)}")
    print(f"  Feature dimensions: {n_features} "
          f"({n_rel_features} relation + {n_type_features} type)")

    # Build feature matrix
    entity_ids: list[str] = []
    features = np.zeros((len(entities), n_features), dtype=np.float32)

    rt_to_idx = {rt: i for i, rt in enumerate(rel_types_sorted)}
    et_to_idx = {et: i for i, et in enumerate(entity_types_sorted)}

    for row_idx, e in enumerate(entities):
        entity_ids.append(e["id"])

        # Relation counts
        for rt, count in e["out_rel_counts"].items():
            col = rt_to_idx[rt] * 2  # OUT column
            features[row_idx, col] = count
        for rt, count in e["in_rel_counts"].items():
            col = rt_to_idx[rt] * 2 + 1  # IN column
            features[row_idx, col] = count

        # Entity type one-hot
        if e["entity_type"] != "Unknown" and e["entity_type"] in et_to_idx:
            col = n_rel_features + et_to_idx[e["entity_type"]]
            features[row_idx, col] = 1.0

    # L2 normalize (so cosine distance = 1 - dot product)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    features = features / norms

    # Stats
    nonzero_rows = np.sum(np.any(features != 0, axis=1))
    avg_nonzero = np.mean(np.sum(features != 0, axis=1))
    print(f"  ✓ Feature matrix: {features.shape[0]} × {features.shape[1]}")
    print(f"    Entities with ≥1 relation: {nonzero_rows}/{len(entities)}")
    print(f"    Avg non-zero features per entity: {avg_nonzero:.1f}")

    return features, entity_ids, feature_names


# ---------------------------------------------------------------------------
# Step 3: Hierarchical Agglomerative Clustering
# ---------------------------------------------------------------------------

def hierarchical_cluster(
    features: np.ndarray,
    entity_ids: list[str],
    k_override: Optional[int] = None,
) -> tuple[list[dict], np.ndarray]:
    """
    HAC with Ward linkage on relation signature vectors.

    Returns:
      clusters: list of {cluster_id, members: [entity_ids]}
      linkage_matrix: scipy linkage matrix (for dendrogram hierarchy)
    """
    print("\n[3/7] Hierarchical Agglomerative Clustering...")

    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import silhouette_score

    n = len(entity_ids)
    if n < 2:
        print("  ✗ Too few entities for clustering")
        return [{"cluster_id": 0, "members": entity_ids}], np.array([])

    # HAC with Ward linkage on raw features (Euclidean distance).
    # Ward linkage requires Euclidean space — passing cosine distances
    # to Ward is mathematically invalid.  The L2-normalized features from
    # step 2 make Euclidean distance proportional to cosine distance, so
    # Ward on the raw matrix gives correct results.
    print(f"  Running Ward linkage HAC on {n} entities...")
    Z = linkage(features, method='ward', metric='euclidean')

    if k_override:
        best_k = k_override
        print(f"  Using forced k={best_k}")
    else:
        # Silhouette score sweep to find optimal k
        print(f"  Sweeping k=[{K_MIN}..{K_MAX}] for best silhouette score...")
        best_score = -1.0
        best_k = K_DEFAULT
        for k in range(K_MIN, min(K_MAX + 1, n)):
            labels = fcluster(Z, t=k, criterion='maxclust')
            n_unique = len(set(labels))
            if n_unique < 2 or n_unique >= n:
                continue
            try:
                score = silhouette_score(features, labels, metric='euclidean')
                if score > best_score:
                    best_score = score
                    best_k = k
            except ValueError:
                continue
        print(f"  Best k={best_k} (silhouette={best_score:.4f})")

    # Cut dendrogram at best_k
    labels = fcluster(Z, t=best_k, criterion='maxclust')

    # Build cluster dicts
    cluster_map: dict[int, list[str]] = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_map[label].append(entity_ids[i])

    clusters = [
        {"cluster_id": cid, "members": sorted(members)}
        for cid, members in sorted(cluster_map.items())
    ]

    # Filter singletons
    before = len(clusters)
    clusters = [c for c in clusters if len(c["members"]) >= MIN_CLUSTER_SIZE]
    filtered = before - len(clusters)

    # Renumber
    for i, c in enumerate(clusters):
        c["cluster_id"] = i

    sizes = [len(c["members"]) for c in clusters]
    print(f"  ✓ {len(clusters)} clusters (removed {filtered} singletons)")
    print(f"    Sizes: min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.1f}")

    return clusters, Z


# ---------------------------------------------------------------------------
# Step 4: LLM Coherence Refinement
# ---------------------------------------------------------------------------

def _build_coherence_prompt(cluster: dict, entities_by_id: dict) -> str:
    """Build prompt for LLM coherence judgment."""
    members = cluster["members"][:MAX_MEMBERS_IN_PROMPT]
    lines: list[str] = []

    for mid in members:
        ent = entities_by_id.get(mid, {})
        etype = ent.get("entity_type", "Unknown")
        out_rels = ent.get("out_rel_counts", {})
        in_rels = ent.get("in_rel_counts", {})

        out_str = ", ".join(f"{r}({c})" for r, c in sorted(out_rels.items()))
        in_str = ", ".join(f"{r}({c})" for r, c in sorted(in_rels.items()))

        lines.append(f'- "{mid}" (type: {etype})')
        if out_str:
            lines.append(f'    OUT: {out_str}')
        if in_str:
            lines.append(f'    IN: {in_str}')

    entity_block = "\n".join(lines)
    total = len(cluster["members"])
    shown = len(members)
    note = f" (showing {shown}/{total})" if total > shown else ""

    return (
        "You are an ontology expert. Below is a group of entities from a knowledge graph,\n"
        "along with their relation patterns (what relations they participate in).\n\n"
        f"Entities{note}:\n{entity_block}\n\n"
        "Question: Do these entities form a coherent ontological class (entities that\n"
        "share the same conceptual type)?\n\n"
        "Rate coherence 1-5:\n"
        "1 = clearly mixed (multiple unrelated types)\n"
        "2 = mostly mixed with some overlap\n"
        "3 = somewhat coherent but has outliers\n"
        "4 = coherent with minor noise\n"
        "5 = perfectly coherent single type\n\n"
        "If score < 3, identify which entities don't belong and name the RELATION TYPE\n"
        "that best separates the majority group from the outliers.\n\n"
        'Respond as JSON: {"score": N, "reasoning": "...", "split_relation": "relation_type" | null}'
    )


def llm_coherence_refine(
    clusters: list[dict],
    entities: list[dict],
    features: np.ndarray,
    entity_ids: list[str],
    llm: ChatOllama,
    rounds: int = MAX_REFINEMENT_ROUNDS,
) -> list[dict]:
    """
    Iterative LLM coherence refinement: judge each cluster, split incoherent ones.
    """
    print(f"\n[4/7] LLM coherence refinement ({rounds} rounds)...")

    entities_by_id = {e["id"]: e for e in entities}
    id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

    current_clusters = [dict(c) for c in clusters]

    for round_num in range(1, rounds + 1):
        print(f"\n  --- Round {round_num}/{rounds} ---")
        splits_done = 0
        next_clusters: list[dict] = []

        for cluster in current_clusters:
            if len(cluster["members"]) < 3:
                # Too small to split meaningfully
                next_clusters.append(cluster)
                continue

            prompt = _build_coherence_prompt(cluster, entities_by_id)
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                result = _parse_json_safely(response.content)
                score = int(result.get("score", 4)) if isinstance(result, dict) else 4
                split_rel = result.get("split_relation") if isinstance(result, dict) else None
                reasoning = result.get("reasoning", "") if isinstance(result, dict) else ""
            except Exception as e:
                print(f"    ⚠ LLM error for cluster {cluster['cluster_id']}: {e}")
                next_clusters.append(cluster)
                continue

            cluster["coherence_score"] = score
            cluster["coherence_reasoning"] = reasoning

            if score >= COHERENCE_KEEP_THRESHOLD:
                next_clusters.append(cluster)
                continue

            if score < COHERENCE_SPLIT_THRESHOLD and split_rel:
                # Split by the discriminative relation
                group_a: list[str] = []  # has the relation
                group_b: list[str] = []  # doesn't have it
                for mid in cluster["members"]:
                    ent = entities_by_id.get(mid, {})
                    all_rels = set(ent.get("out_rel_types", []) +
                                   ent.get("in_rel_types", []))
                    if split_rel in all_rels:
                        group_a.append(mid)
                    else:
                        group_b.append(mid)

                if len(group_a) >= MIN_CLUSTER_SIZE and len(group_b) >= MIN_CLUSTER_SIZE:
                    next_clusters.append({
                        "cluster_id": -1,
                        "members": group_a,
                        "split_from": cluster["cluster_id"],
                        "split_relation": split_rel,
                    })
                    next_clusters.append({
                        "cluster_id": -1,
                        "members": group_b,
                        "split_from": cluster["cluster_id"],
                        "split_relation": f"NOT {split_rel}",
                    })
                    splits_done += 1
                    print(f"    Split cluster {cluster['cluster_id']} "
                          f"(score={score}) by '{split_rel}': "
                          f"{len(group_a)} + {len(group_b)}")
                else:
                    # Split would create singletons — keep original
                    next_clusters.append(cluster)
            else:
                # Score is 3 (borderline) or no split suggestion — keep
                next_clusters.append(cluster)

        # Renumber clusters
        for i, c in enumerate(next_clusters):
            c["cluster_id"] = i
        current_clusters = next_clusters

        print(f"  Round {round_num}: {splits_done} splits → {len(current_clusters)} clusters")

        if splits_done == 0:
            print("  No splits needed — stopping early")
            break

    # --- Merge pass: merge clusters with very similar centroids ---
    print("\n  Merge pass: checking for over-fragmented clusters...")
    if len(current_clusters) >= 2:
        centroids: list[np.ndarray] = []
        for c in current_clusters:
            idxs = [id_to_idx[m] for m in c["members"] if m in id_to_idx]
            if idxs:
                centroid = features[idxs].mean(axis=0)
                norm = np.linalg.norm(centroid)
                centroids.append(centroid / (norm + 1e-9))
            else:
                centroids.append(np.zeros(features.shape[1]))

        merged_into: dict[int, int] = {}
        merge_pairs: list[tuple[int, int, float]] = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                sim = float(np.dot(centroids[i], centroids[j]))
                if sim >= MERGE_SIMILARITY_THRESHOLD:
                    merge_pairs.append((i, j, sim))
        merge_pairs.sort(key=lambda x: -x[2])

        def find_root(idx: int) -> int:
            while idx in merged_into:
                idx = merged_into[idx]
            return idx

        for i, j, sim in merge_pairs:
            ci, cj = find_root(i), find_root(j)
            if ci != cj:
                if len(current_clusters[ci]["members"]) >= len(current_clusters[cj]["members"]):
                    keeper, absorbed = ci, cj
                else:
                    keeper, absorbed = cj, ci
                merged_into[absorbed] = keeper
                new_members = list(dict.fromkeys(
                    current_clusters[keeper]["members"] +
                    current_clusters[absorbed]["members"]
                ))
                current_clusters[keeper] = {
                    **current_clusters[keeper],
                    "members": new_members,
                }
                print(f"    Merged cluster {absorbed} into {keeper} (sim={sim:.3f})")

        absorbed_set = set(merged_into.keys())
        current_clusters = [c for i, c in enumerate(current_clusters)
                            if i not in absorbed_set]
        if absorbed_set:
            print(f"  Merged {len(absorbed_set)} clusters")

    # Final renumber
    for i, c in enumerate(current_clusters):
        c["cluster_id"] = i

    print(f"\n  ✓ Refinement complete: {len(current_clusters)} clusters")
    return current_clusters


# ---------------------------------------------------------------------------
# Step 5: LLM Naming with Convention Calibration
# ---------------------------------------------------------------------------

def llm_name_clusters(
    clusters: list[dict],
    entities: list[dict],
    llm: ChatOllama,
) -> list[dict]:
    """Name each cluster with a short, abstract PascalCase ontology class name."""
    print("\n[5/7] Naming clusters with convention-calibrated LLM...")
    entities_by_id = {e["id"]: e for e in entities}
    named: list[dict] = []

    for cluster in clusters:
        members = cluster["members"]
        if len(members) == 1:
            named.append({**cluster, "class_name": _sanitize_label(members[0])})
            continue

        # Build context
        sample = members[:MAX_MEMBERS_IN_PROMPT]
        lines: list[str] = []
        rel_agg: dict[str, int] = defaultdict(int)
        type_agg: dict[str, int] = defaultdict(int)

        for mid in sample:
            ent = entities_by_id.get(mid, {})
            lines.append(f'  "{mid}" (type: {ent.get("entity_type", "?")})')
            for r, c in ent.get("out_rel_counts", {}).items():
                rel_agg[f"{r}_OUT"] += c
            for r, c in ent.get("in_rel_counts", {}).items():
                rel_agg[f"{r}_IN"] += c
            et = ent.get("entity_type", "Unknown")
            if et != "Unknown":
                type_agg[et] += 1

        member_block = "\n".join(lines)
        top_rels = sorted(rel_agg.items(), key=lambda x: -x[1])[:8]
        rel_str = ", ".join(f"{r}({c})" for r, c in top_rels)
        type_str = ", ".join(f"{t}({c})" for t, c in sorted(type_agg.items(), key=lambda x: -x[1])[:5])

        prompt = (
            "You are naming ontology classes for a knowledge graph.\n\n"
            f"Cluster members ({len(members)} entities):\n{member_block}\n\n"
            f"Most common relations: {rel_str}\n"
            f"Most common entity types: {type_str}\n\n"
            "Name this class with a SINGLE PascalCase word.\n\n"
            "RULES:\n"
            "- ONE word maximum (e.g., Coverage, Person, Risk, Structure, Property)\n"
            "- Be ABSTRACT — the broadest category that fits ALL members\n"
            "- BAD: InsuranceCoverage, FloodRisk, PropertyDamage (too specific/compound)\n"
            "- GOOD: Coverage, Risk, Damage, Person, Product, Organization, Address, Object\n\n"
            "Respond with ONLY the class name (one PascalCase word):"
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip().split("\n")[0].strip()
            raw = re.sub(r'["`\']', '', raw).split()[0] if raw.split() else "UnknownClass"
            class_name = _sanitize_label(raw)
        except Exception as e:
            print(f"    ⚠ Naming error for cluster {cluster['cluster_id']}: {e}")
            class_name = _sanitize_label(members[0])

        named.append({**cluster, "class_name": class_name})
        print(f"    Cluster {cluster['cluster_id']}: "
              f"{members[:3]}{'...' if len(members) > 3 else ''} → {class_name}")

    # Deduplicate names (add numeric suffix if needed)
    name_counts: dict[str, int] = defaultdict(int)
    for c in named:
        name_counts[c["class_name"]] += 1
    dupe_counter: dict[str, int] = defaultdict(int)
    for c in named:
        if name_counts[c["class_name"]] > 1:
            dupe_counter[c["class_name"]] += 1
            if dupe_counter[c["class_name"]] > 1:
                c["class_name"] = f"{c['class_name']}{dupe_counter[c['class_name']]}"

    print(f"\n  ✓ {len(named)} clusters named")
    return named


# ---------------------------------------------------------------------------
# Step 6: Derive Hierarchy from Dendrogram
# ---------------------------------------------------------------------------

def derive_hierarchy_from_dendrogram(
    linkage_matrix: np.ndarray,
    clusters: list[dict],
    entity_ids: list[str],
    features: np.ndarray,
) -> list[dict]:
    """
    Derive SUBCLASS_OF edges from the HAC dendrogram.

    For each pair of clusters, check if one is a subset of the other at
    a higher cut level. If cluster A's members are a subset of cluster B's
    members at a coarser cut, then A SUBCLASS_OF B.
    """
    print("\n[6/7] Deriving hierarchy from dendrogram...")

    if len(linkage_matrix) == 0 or len(clusters) < 2:
        print("  ⚠ Cannot derive hierarchy (too few clusters)")
        return []

    from scipy.cluster.hierarchy import fcluster

    # Get class names for current clusters
    id_to_class: dict[str, str] = {}
    for c in clusters:
        for mid in c["members"]:
            id_to_class[mid] = c["class_name"]

    # Cut at a coarser level (fewer clusters) to find parent classes
    current_k = len(clusters)
    coarse_k = max(2, current_k // 2)  # half as many clusters

    coarse_labels = fcluster(linkage_matrix, t=coarse_k, criterion='maxclust')

    # Map each coarse cluster to the fine-grained classes it contains
    coarse_to_fine: dict[int, set[str]] = defaultdict(set)
    for i, eid in enumerate(entity_ids):
        if eid in id_to_class:
            coarse_to_fine[coarse_labels[i]].add(id_to_class[eid])

    hierarchy: list[dict] = []
    for coarse_id, fine_classes in coarse_to_fine.items():
        if len(fine_classes) > 1:
            # Multiple fine classes map to one coarse cluster
            # The largest fine class becomes the parent, others are children
            class_sizes = {}
            for cls in fine_classes:
                class_sizes[cls] = sum(
                    1 for c in clusters if c["class_name"] == cls
                    for _ in c["members"]
                )
            parent = max(class_sizes, key=class_sizes.get)
            for child in fine_classes:
                if child != parent:
                    hierarchy.append({"child": child, "parent": parent})
                    print(f"    {child} SUBCLASS_OF {parent}")

    # Remove cycles
    hierarchy = _remove_cycles(hierarchy)

    print(f"  ✓ {len(hierarchy)} SUBCLASS_OF edges derived")
    return hierarchy


def _remove_cycles(pairs: list[dict]) -> list[dict]:
    """Remove SUBCLASS_OF pairs that create directed cycles."""
    children: dict[str, set] = defaultdict(set)
    result = []
    for rel in pairs:
        child, parent = rel["child"], rel["parent"]
        visited: set[str] = set()
        stack = [parent]
        creates_cycle = False
        while stack:
            node = stack.pop()
            if node == child:
                creates_cycle = True
                break
            if node not in visited:
                visited.add(node)
                stack.extend(children[node])
        if creates_cycle:
            print(f"    ⚠ Cycle removed: {child} → {parent}")
        else:
            children[child].add(parent)
            result.append(rel)
    return result


# ---------------------------------------------------------------------------
# Step 7: Write Ontology to Neo4j
# ---------------------------------------------------------------------------

def write_ontology(clusters: list[dict], hierarchy: list[dict]) -> dict:
    """Write the RSI-LCR induced ontology to Neo4j."""
    print("\n[7/7] Writing ontology to Neo4j...")

    if not clusters:
        print("  ✗ No clusters to write")
        return {"error": "no clusters"}

    graph = get_neo4j_graph()

    # Clear previous ontology
    existing_oc = graph.query("MATCH (c:OntologyClass) RETURN c.name AS name")
    for row in existing_oc:
        old_cls = _sanitize_label(row["name"]) if row["name"] else None
        if old_cls:
            try:
                graph.query(f"MATCH (n:Entity) WHERE n:{old_cls} REMOVE n:{old_cls}")
            except Exception:
                pass
    graph.query("MATCH (c:OntologyClass) DETACH DELETE c")
    print(f"  Cleared {len(existing_oc)} previous OntologyClass nodes")

    # Label entities
    entities_labeled = 0
    class_names: set[str] = set()
    for cluster in clusters:
        class_name = _sanitize_label(cluster["class_name"])
        members = cluster["members"]
        if not class_name or not members:
            continue
        class_names.add(class_name)
        graph.query(
            f"UNWIND $members AS mid "
            f"MATCH (n:Entity {{id: mid}}) "
            f"SET n:{class_name} "
            f"SET n.ontology_class = $cls",
            params={"members": members, "cls": class_name}
        )
        entities_labeled += len(members)

    # Create OntologyClass nodes
    for cluster in clusters:
        class_name = _sanitize_label(cluster["class_name"])
        graph.query(
            "MERGE (c:OntologyClass {name: $name}) "
            "SET c.member_count = $count, "
            "    c.example_members = $examples, "
            "    c.method = 'RSI-LCR'",
            params={
                "name": class_name,
                "count": len(cluster["members"]),
                "examples": cluster["members"][:10],
            }
        )

    # Create SUBCLASS_OF edges
    subclass_created = 0
    for rel in hierarchy:
        child = _sanitize_label(rel["child"])
        parent = _sanitize_label(rel["parent"])
        if child in class_names and parent in class_names:
            graph.query(
                "MATCH (child:OntologyClass {name: $child}) "
                "MATCH (parent:OntologyClass {name: $parent}) "
                "MERGE (child)-[:SUBCLASS_OF]->(parent)",
                params={"child": child, "parent": parent}
            )
            subclass_created += 1

    stats = {
        "entities_labeled": entities_labeled,
        "ontology_classes": len(class_names),
        "subclass_of_edges": subclass_created,
        "class_names": sorted(class_names),
        "method": "RSI-LCR",
    }

    print(f"  ✓ Entities labeled:     {entities_labeled}")
    print(f"  ✓ OntologyClass nodes:  {len(class_names)}")
    print(f"  ✓ SUBCLASS_OF edges:    {subclass_created}")
    print(f"  ✓ Classes:              {sorted(class_names)}")
    return stats


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_rsi_lcr(
    model: str = config.OLLAMA_MODEL,
    suffix: str = "zone3_rsi",
    k_override: Optional[int] = None,
    refinement_rounds: int = MAX_REFINEMENT_ROUNDS,
) -> dict:
    """Run the full RSI-LCR pipeline."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("CS584 Capstone — Zone 3: RSI-LCR")
    print("Relation-Signature Induction with LLM Coherence Refinement")
    print(f"Model: {model} | Suffix: {suffix} | Refinement rounds: {refinement_rounds}")
    print("=" * 70)

    start = time.time()

    # Step 1: Load entities
    entities = load_entities()
    if not entities:
        return {"error": "no entities"}

    # Step 2: Build relation signatures
    features, entity_ids, feature_names = build_relation_signatures(entities)

    # Step 3: HAC clustering
    clusters, linkage_matrix = hierarchical_cluster(
        features, entity_ids, k_override=k_override,
    )

    # Step 4: LLM coherence refinement
    llm = get_llm(model)
    clusters = llm_coherence_refine(
        clusters, entities, features, entity_ids, llm,
        rounds=refinement_rounds,
    )

    # Step 5: LLM naming
    named_clusters = llm_name_clusters(clusters, entities, llm)

    # Step 6: Derive hierarchy
    hierarchy = derive_hierarchy_from_dendrogram(
        linkage_matrix, named_clusters, entity_ids, features,
    )

    # Step 7: Write to Neo4j
    neo4j_stats = write_ontology(named_clusters, hierarchy)

    elapsed = time.time() - start

    # Summary
    print(f"\n{'=' * 70}")
    print(f"RSI-LCR pipeline complete in {elapsed:.1f}s")
    print(f"  Method:            RSI-LCR (Relation-Signature Induction)")
    print(f"  Entities:          {len(entities)}")
    print(f"  Features:          {len(feature_names)} dimensions")
    print(f"  Clusters:          {len(named_clusters)}")
    print(f"  SUBCLASS_OF:       {len(hierarchy)}")
    print(f"  Classes:           {[c['class_name'] for c in named_clusters]}")

    # Save summary
    summary = {
        "mode": "zone3_rsi_lcr",
        "model": model,
        "suffix": suffix,
        "elapsed_seconds": round(elapsed, 2),
        "entity_count": len(entities),
        "feature_dimensions": len(feature_names),
        "feature_names": feature_names,
        "clusters": len(named_clusters),
        "refinement_rounds": refinement_rounds,
        "named_clusters": [
            {
                "cluster_id": c["cluster_id"],
                "class_name": c["class_name"],
                "member_count": len(c["members"]),
                "members": c["members"][:20],
                "coherence_score": c.get("coherence_score"),
            }
            for c in named_clusters
        ],
        "hierarchy": hierarchy,
        "neo4j_stats": neo4j_stats,
    }
    out_path = os.path.join(config.RESULTS_DIR, f"zone3_rsi_lcr_summary_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✓ Summary saved to {out_path}")
    print(f"\nNext steps:")
    print(f"  python3 baseline/eval.py --suffix {suffix}")
    print(f"  python3 baseline/eval.py --suffix {suffix} --riskine")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zone 3: RSI-LCR — Relation-Signature Induction with LLM Coherence Refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Novel ontology induction method: cluster entities by their relation patterns
(what relations they participate in), then refine with LLM coherence judgment.

Pre-requisite: Run zone2/pipeline.py first to populate Neo4j with Entity nodes.

Examples:
  python3 zone3/rsi_lcr.py
  python3 zone3/rsi_lcr.py --model qwen2.5:72b
  python3 zone3/rsi_lcr.py --model qwen2.5:72b --suffix zone3_rsi
  python3 zone3/rsi_lcr.py --refinement-rounds 3
  python3 zone3/rsi_lcr.py --k 15

After running, evaluate with:
  python3 baseline/eval.py --suffix zone3_rsi --riskine
        """,
    )
    parser.add_argument(
        "--model", default=config.OLLAMA_MODEL,
        help=f"Ollama model name (default: {config.OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--suffix", default="zone3_rsi",
        help="Suffix for result files (default: zone3_rsi)"
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Force cluster count (default: silhouette-guided)"
    )
    parser.add_argument(
        "--refinement-rounds", type=int, default=MAX_REFINEMENT_ROUNDS,
        help=f"LLM coherence refinement rounds (default: {MAX_REFINEMENT_ROUNDS})"
    )
    args = parser.parse_args()

    run_rsi_lcr(
        model=args.model,
        suffix=args.suffix,
        k_override=args.k,
        refinement_rounds=args.refinement_rounds,
    )
