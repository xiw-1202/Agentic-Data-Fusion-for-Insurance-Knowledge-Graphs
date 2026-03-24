"""Zone 3 Pipeline — Bottom-Up Ontology Induction via Leiden Community Detection

Core research contribution: automatically discovers ontology classes from the
raw entity graph produced by Zone 2, without any reference ontology.

Architecture (LangGraph pipeline):
  build_similarity_graph → leiden_cluster → name_clusters → propose_hierarchy → write_ontology → END

Algorithm:
  1. Load all Entity nodes and their relationships from Neo4j (Zone 2 output)
  2. Build an entity-similarity graph:
     - Embedding similarity (what entities are called)
     - Structural similarity (shared relation types via Jaccard)
     - Co-occurrence (appear in same chunk)
  3. Leiden community detection → 10-30 clusters
  4. LLM names each cluster as a canonical ontology class (PascalCase)
  5. LLM proposes SUBCLASS_OF relationships between clusters
  6. Write ontology layer to Neo4j:
     - Each entity gets its cluster's class as a Neo4j label
     - OntologyClass nodes created
     - SUBCLASS_OF edges between classes

NO reference ontology is used. The LLM invents class names from scratch.
Riskine alignment is measured ONLY in evaluation/ (separate step).

Usage:
  python3 zone3/pipeline.py                    # default model
  python3 zone3/pipeline.py --model qwen2.5:7b

Pre-requisite: Zone 2 must have run first (Neo4j populated with :Entity nodes).

Evaluation (run AFTER this pipeline):
  python3 baseline/eval.py --suffix zone3
  python3 baseline/eval.py --suffix zone3 --riskine
"""

from __future__ import annotations

import json
import re
import time
import os
import sys
import argparse
from typing import TypedDict, Annotated, Optional, Union
from collections import defaultdict
import operator

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

import config


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class Zone3State(TypedDict):
    entities:          list[dict]          # {id, relations_out, relations_in, chunks}
    entity_map:        dict                # {entity_id: entity_dict} — for quick lookup
    similarity_edges:  list[dict]          # {source, target, weight}
    entity_embeddings: dict                # {ids: [...], embs: [[...]]} — L2-normalized
    clusters:          list[dict]          # {cluster_id, members: [entity_ids], coherence}
    cluster_levels:    dict                # {res_str: list[cluster_dict]} — all resolutions
    named_clusters:    list[dict]          # {cluster_id, members, class_name, coherence}
    named_levels:      dict                # {res_str: list[named_cluster_dict]} — all levels
    hierarchy:         list[dict]          # {child, parent} SUBCLASS_OF pairs
    neo4j_stats:       dict
    errors:            Annotated[list, operator.add]
    model:             str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Composite similarity weights (4 dimensions — entity type added in Phase A)
W_EMBEDDING  = 0.40
W_STRUCTURAL = 0.25
W_COOCCUR    = 0.15
W_TYPE       = 0.20    # NEW: entity type agreement (Jaccard on bootstrapped types)

# Minimum composite similarity to create an edge in the similarity graph
SIMILARITY_THRESHOLD = 0.50  # raised from 0.40 — reduces noisy edges

# Leiden resolution parameter (higher = more clusters) — kept for backward compat
LEIDEN_RESOLUTION = 0.6

# Multi-resolution Leiden: coarse → medium → fine
# Higher resolution → more / smaller clusters (finer granularity)
LEIDEN_RESOLUTIONS       = [0.3, 0.6, 1.2]   # [coarse, medium, fine]
HIERARCHY_OVERLAP_THRESH = 0.6                # min fraction of fine members inside parent

# Post-clustering: filter singletons and merge similar clusters
MIN_CLUSTER_SIZE         = 2                  # drop singletons (noise entities)
CLUSTER_MERGE_THRESHOLD  = 0.70               # merge clusters whose centroids are this similar


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
    # Neo4j labels cannot start with a digit
    if cleaned[0].isdigit():
        cleaned = "Class" + cleaned
    return cleaned


def _parse_json_safely(text: str) -> list | dict:
    """Try to parse JSON from LLM output with fallbacks."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return []


# ---------------------------------------------------------------------------
# Pipeline Nodes
# ---------------------------------------------------------------------------

def load_entities(state: Zone3State) -> dict:
    """Load all Entity nodes from Neo4j with their structural context."""
    print("\n[1/5] Loading entities from Neo4j...")
    graph = get_neo4j_graph()

    # Get all entities with their types (single batch query — was O(n) round-trips)
    rows = graph.query("""
        MATCH (n:Entity)
        OPTIONAL MATCH (n)-[r]->(m:Entity)
        OPTIONAL MATCH (p:Entity)-[s]->(n)
        RETURN n.id AS id,
               n.entity_type AS entity_type,
               collect(DISTINCT {rel: type(r), target: m.id, chunk: r.chunk_id}) AS out_rels,
               collect(DISTINCT {rel: type(s), source: p.id, chunk: s.chunk_id}) AS in_rels
    """)
    if not rows:
        print("  ✗ No Entity nodes found. Run zone2/pipeline.py first.")
        return {"entities": []}

    entities: list[dict] = []
    type_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        eid = row["id"]
        etype = row.get("entity_type") or "Unknown"
        type_counts[etype] += 1

        # Filter out null rels from OPTIONAL MATCH
        out_rels = [r for r in row.get("out_rels", []) if r.get("rel")]
        in_rels  = [r for r in row.get("in_rels", []) if r.get("rel")]

        rel_types_out = [r["rel"] for r in out_rels]
        rel_types_in  = [r["rel"] for r in in_rels]
        chunks = list(set(
            r.get("chunk", "") for r in out_rels + in_rels if r.get("chunk")
        ))

        entities.append({
            "id": eid,
            "entity_type": etype,
            "relations_out": rel_types_out,
            "relations_in": rel_types_in,
            "all_relation_types": list(set(rel_types_out + rel_types_in)),
            "neighbors": list(set(
                [r["target"] for r in out_rels if r.get("target")] +
                [r["source"] for r in in_rels if r.get("source")]
            )),
            "chunks": chunks,
        })

    entity_map = {e["id"]: e for e in entities}
    typed = sum(1 for e in entities if e["entity_type"] != "Unknown")
    print(f"  ✓ {len(entities)} entities loaded ({typed} typed, "
          f"{len(type_counts)} distinct types)")
    return {"entities": entities, "entity_map": entity_map}


def build_similarity_graph(state: Zone3State) -> dict:
    """Build weighted similarity graph between entities."""
    print("\n[2/5] Building entity similarity graph...")
    entities = state.get("entities", [])
    if len(entities) < 2:
        print("  ✗ Too few entities to build similarity graph")
        return {"similarity_edges": []}

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ✗ sentence_transformers not installed")
        return {"similarity_edges": []}

    ids = [e["id"] for e in entities]
    entity_map = {e["id"]: e for e in entities}

    # Embedding similarity
    print(f"  Computing embedding similarity for {len(ids)} entities...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(ids, normalize_embeddings=True)

    # Build edges
    edges: list[dict] = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            ei, ej = entities[i], entities[j]

            # Component 1: Embedding similarity (cosine via dot product)
            sem_sim = float(np.dot(embs[i], embs[j]))

            # Component 2: Structural similarity (Jaccard on relation types)
            rels_i = set(ei["all_relation_types"])
            rels_j = set(ej["all_relation_types"])
            if rels_i or rels_j:
                struct_sim = len(rels_i & rels_j) / len(rels_i | rels_j)
            else:
                struct_sim = 0.0

            # Component 3: Co-occurrence (shared chunks)
            chunks_i = set(ei["chunks"])
            chunks_j = set(ej["chunks"])
            if chunks_i or chunks_j:
                cooccur = len(chunks_i & chunks_j) / len(chunks_i | chunks_j)
            else:
                cooccur = 0.0

            # Component 4: Entity type agreement (Phase A — bootstrapped types)
            type_i = ei.get("entity_type", "Unknown")
            type_j = ej.get("entity_type", "Unknown")
            if type_i != "Unknown" and type_j != "Unknown":
                type_sim = 1.0 if type_i == type_j else 0.0
            else:
                type_sim = 0.0  # no signal if either is untyped

            # Composite score (4 dimensions)
            weight = (W_EMBEDDING * sem_sim +
                      W_STRUCTURAL * struct_sim +
                      W_COOCCUR * cooccur +
                      W_TYPE * type_sim)

            if weight >= SIMILARITY_THRESHOLD:
                edges.append({
                    "source": ei["id"],
                    "target": ej["id"],
                    "weight": round(weight, 4),
                    "sem": round(sem_sim, 3),
                    "struct": round(struct_sim, 3),
                    "cooccur": round(cooccur, 3),
                    "type": round(type_sim, 3),
                })

    print(f"  ✓ {len(edges)} similarity edges (threshold ≥ {SIMILARITY_THRESHOLD})")
    if edges:
        avg_w = sum(e["weight"] for e in edges) / len(edges)
        print(f"    Average edge weight: {avg_w:.3f}")
    return {
        "similarity_edges":  edges,
        "entity_embeddings": {"ids": ids, "embs": embs.tolist()},
    }


def _build_clusters_from_partition(partition, node_list: list[str],
                                    entity_embeddings: dict, res: float) -> list[dict]:
    """Convert a leidenalg partition object to our cluster list format."""
    community_map: dict[int, list[str]] = defaultdict(list)
    for vertex, community_id in enumerate(partition.membership):
        community_map[community_id].append(node_list[vertex])

    clusters = [
        {"cluster_id": cid, "members": sorted(members)}
        for cid, members in sorted(community_map.items())
    ]
    clusters = _add_cluster_coherence(clusters, entity_embeddings)
    return clusters


def leiden_cluster(state: Zone3State) -> dict:
    """Run Leiden community detection at multiple resolutions for multi-level hierarchy."""
    print("\n[3/5] Running multi-resolution Leiden community detection...")
    edges    = state.get("similarity_edges", [])
    entities = state.get("entities", [])
    all_ids  = {e["id"] for e in entities}
    embs     = state.get("entity_embeddings", {})

    if not edges:
        print("  ⚠ No similarity edges — each entity becomes its own cluster")
        clusters = [{"cluster_id": i, "members": [e["id"]]}
                    for i, e in enumerate(entities)]
        clusters = _add_cluster_coherence(clusters, embs)
        # No multi-level data when there are no edges
        return {"clusters": clusters, "cluster_levels": {str(LEIDEN_RESOLUTION): clusters}}

    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print("    Install: pip install leidenalg python-igraph")
        clusters = [{"cluster_id": 0, "members": [e["id"] for e in entities]}]
        return {"clusters": clusters, "cluster_levels": {}}

    # Build igraph graph (leidenalg requires igraph, not networkx)
    node_list  = sorted(all_ids)
    node_index = {n: i for i, n in enumerate(node_list)}
    G = ig.Graph()
    G.add_vertices(len(node_list))
    G.vs["name"] = node_list
    edge_tuples = [(node_index[e["source"]], node_index[e["target"]]) for e in edges
                   if e["source"] in node_index and e["target"] in node_index]
    weights = [e["weight"] for e in edges
               if e["source"] in node_index and e["target"] in node_index]
    G.add_edges(edge_tuples)

    print(f"  Graph: {G.vcount()} nodes, {G.ecount()} edges")

    # Run Leiden at each resolution and collect results
    cluster_levels: dict[str, list[dict]] = {}
    for res in LEIDEN_RESOLUTIONS:
        partition = leidenalg.find_partition(
            G,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights if weights else None,
            resolution_parameter=res,
            n_iterations=10,
            seed=42,
        )
        clusters_at_res = _build_clusters_from_partition(partition, node_list, embs, res)
        cluster_levels[str(res)] = clusters_at_res
        ns_coh = [c["coherence"] for c in clusters_at_res
                  if len(c["members"]) > 1 and "coherence" in c]
        avg = f"  avg_coherence={np.mean(ns_coh):.3f}" if ns_coh else ""
        print(f"    resolution={res}: {len(clusters_at_res)} communities{avg}")

    # Primary level = medium resolution (0.6) for downstream backward compat
    primary_key = str(LEIDEN_RESOLUTION)   # "0.6"
    primary = cluster_levels.get(primary_key, list(cluster_levels.values())[-1])

    print(f"  ✓ Primary level (res={LEIDEN_RESOLUTION}): {len(primary)} communities")
    for c in primary:
        coh_str = (f"  coherence={c['coherence']:.3f}"
                   if "coherence" in c and len(c["members"]) > 1 else "")
        print(f"    Cluster {c['cluster_id']}: {len(c['members'])} members "
              f"→ {c['members'][:5]}{'...' if len(c['members']) > 5 else ''}{coh_str}")

    ns_coh_primary = [c["coherence"] for c in primary
                      if len(c["members"]) > 1 and "coherence" in c]
    if ns_coh_primary:
        print(f"    Avg coherence non-singleton (primary): {np.mean(ns_coh_primary):.4f}")

    return {"clusters": primary, "cluster_levels": cluster_levels}


def _build_cluster_context(members: list[str], entity_map: dict) -> str:
    """Build a rich context string for cluster naming: members + their relationships."""
    lines: list[str] = []
    rel_summary: dict[str, int] = defaultdict(int)
    neighbor_sample: set[str] = set()
    members_set = set(members)

    for mid in members[:10]:  # limit to 10 members for prompt size
        ent = entity_map.get(mid, {})
        for r in ent.get("relations_out", []):
            rel_summary[r] += 1
        for r in ent.get("relations_in", []):
            rel_summary[f"(incoming) {r}"] += 1
        for n in ent.get("neighbors", [])[:3]:
            if n not in members_set:
                neighbor_sample.add(n)

    lines.append(f"Members ({len(members)}): {json.dumps(members[:15])}")
    if rel_summary:
        top_rels = sorted(rel_summary.items(), key=lambda x: -x[1])[:8]
        rel_str = ", ".join(f"{r}({c})" for r, c in top_rels)
        lines.append(f"Relationship types: {rel_str}")
    if neighbor_sample:
        lines.append(f"Connected to: {list(neighbor_sample)[:8]}")
    return "\n".join(lines)


def _name_cluster_list(clusters: list[dict], llm, entity_map: dict | None = None) -> list[dict]:
    """Name a single list of clusters using the LLM with rich relationship context.

    Improved naming strategy (Round 1 fix):
    - Provides relationship context so LLM understands cluster semantics
    - Instructs LLM to prefer SHORT, ABSTRACT ontology class names (1-2 words)
    - Avoids compound/descriptive names that embed poorly against reference ontology
    """
    if entity_map is None:
        entity_map = {}

    named: list[dict] = []
    for cluster in clusters:
        members = cluster["members"]
        if len(members) == 1:
            name = _sanitize_label(members[0])
            named.append({**cluster, "class_name": name})
            continue

        # Build rich context including relationships
        context = _build_cluster_context(members, entity_map)

        prompt = (
            "You are an ontology engineer building an insurance domain ontology.\n"
            "The following entities were grouped into one semantic cluster:\n\n"
            f"{context}\n\n"
            "Propose ONE canonical PascalCase ontology class name.\n\n"
            "RULES:\n"
            "- Use SHORT names: 1-2 words maximum\n"
            "- Prefer ABSTRACT concepts over specific compound names\n"
            "- Think: what broad ontology category do ALL these entities belong to?\n"
            "- BAD examples: InsuranceLossEvent, FloodInsuranceConcepts, PropertyCoverageComponent, RiskReductionMechanism\n"
            "- GOOD examples: Agent, Event, Artifact, Obligation, Location, Temporal, Process, Agreement, Asset, Hazard\n\n"
            "Respond with ONLY the class name (PascalCase, 1-2 words max):"
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip().split("\n")[0].strip()
            raw = re.sub(r'["`\']', '', raw).split()[0] if raw.split() else "UnknownClass"
            class_name = _sanitize_label(raw)
            named.append({**cluster, "class_name": class_name})
        except Exception as e:
            fallback = _sanitize_label(members[0])
            named.append({**cluster, "class_name": fallback})

    return named


def name_clusters(state: Zone3State) -> dict:
    """LLM names each cluster at all resolution levels as canonical ontology classes."""
    print("\n[4/5] Naming clusters with LLM (all resolution levels)...")
    cluster_levels = state.get("cluster_levels", {})
    model_name     = state.get("model", config.OLLAMA_MODEL)
    entity_map     = state.get("entity_map", {})
    llm            = get_llm(model_name)

    # If no multi-level data (e.g. no edges edge-case), fall back to primary clusters
    if not cluster_levels:
        cluster_levels = {str(LEIDEN_RESOLUTION): state.get("clusters", [])}

    named_levels: dict[str, list[dict]] = {}
    errors: list[dict] = []

    for res_str, clusters in cluster_levels.items():
        print(f"  Naming resolution={res_str} ({len(clusters)} clusters)...")
        named_at_res = _name_cluster_list(clusters, llm, entity_map=entity_map)
        named_levels[res_str] = named_at_res
        for nc in named_at_res:
            members = nc["members"]
            print(f"    Cluster {nc['cluster_id']}: "
                  f"{members[:3]}{'...' if len(members) > 3 else ''} "
                  f"→ {nc['class_name']}")

    # Primary (medium resolution = 0.6) for backward compat with write_ontology
    primary_named = named_levels.get(str(LEIDEN_RESOLUTION),
                                     list(named_levels.values())[-1] if named_levels else [])

    print(f"\n  ✓ {sum(len(v) for v in named_levels.values())} clusters named across "
          f"{len(named_levels)} resolution levels")
    return {"named_clusters": primary_named, "named_levels": named_levels, "errors": errors}


def filter_and_merge_clusters(state: Zone3State) -> dict:
    """Post-naming cleanup: filter singletons and merge semantically similar clusters.

    Round 1 fixes:
    - Fix 2: Remove singleton clusters (size < MIN_CLUSTER_SIZE) — these are noise
      entities like "You", "12:01 a.m." that dilute precision.
    - Fix 3: Merge clusters whose member centroids are very similar (> CLUSTER_MERGE_THRESHOLD).
      Reduces fragmentation (e.g., "InsuranceLossEvent" + "FloodLoss" → single merged cluster).
    """
    print("\n[4.5/5] Filtering singletons and merging similar clusters...")
    named_clusters = state.get("named_clusters", [])
    named_levels   = state.get("named_levels", {})
    entity_embs    = state.get("entity_embeddings", {})

    if not named_clusters:
        return {"named_clusters": [], "named_levels": {}}

    # --- Fix 2: Filter singletons from primary level ---
    before_count = len(named_clusters)
    filtered = [c for c in named_clusters if len(c["members"]) >= MIN_CLUSTER_SIZE]
    singleton_removed = before_count - len(filtered)
    if singleton_removed:
        removed_names = [c["class_name"] for c in named_clusters
                         if len(c["members"]) < MIN_CLUSTER_SIZE]
        print(f"  Removed {singleton_removed} singleton clusters: {removed_names[:10]}")

    # --- Fix 3: Merge clusters with highly similar member centroids ---
    merged_into: dict[int, int] = {}   # defined before if-block (code review fix)
    name_remapping: dict[str, str] = {}  # absorbed_name → keeper_name

    emb_ids  = entity_embs.get("ids", [])
    emb_data = entity_embs.get("embs", [])
    if emb_ids and emb_data and len(filtered) >= 2:
        emb_matrix = np.array(emb_data, dtype=np.float32)
        id_to_idx  = {eid: i for i, eid in enumerate(emb_ids)}

        # Compute cluster centroids
        centroids: list[np.ndarray] = []
        for c in filtered:
            idxs = [id_to_idx[m] for m in c["members"] if m in id_to_idx]
            if idxs:
                centroid = emb_matrix[idxs].mean(axis=0)
                norm = np.linalg.norm(centroid)
                centroids.append(centroid / (norm + 1e-9))
            else:
                centroids.append(np.zeros(emb_matrix.shape[1]))

        # Find pairs to merge (greedy: highest similarity first)
        merge_pairs: list[tuple[int, int, float]] = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                sim = float(np.dot(centroids[i], centroids[j]))
                if sim >= CLUSTER_MERGE_THRESHOLD:
                    merge_pairs.append((i, j, sim))
        merge_pairs.sort(key=lambda x: -x[2])

        # Apply merges (union-find style)
        def find_canonical(idx: int) -> int:
            while idx in merged_into:
                idx = merged_into[idx]
            return idx

        for i, j, sim in merge_pairs:
            ci, cj = find_canonical(i), find_canonical(j)
            if ci != cj:
                # Merge smaller into larger
                if len(filtered[ci]["members"]) >= len(filtered[cj]["members"]):
                    keeper, absorbed = ci, cj
                else:
                    keeper, absorbed = cj, ci
                merged_into[absorbed] = keeper
                # Track name remapping for propagation to named_levels
                name_remapping[filtered[absorbed]["class_name"]] = filtered[keeper]["class_name"]
                # Merge members
                new_members = list(dict.fromkeys(
                    filtered[keeper]["members"] + filtered[absorbed]["members"]
                ))
                filtered[keeper] = {
                    **filtered[keeper],
                    "members": new_members,
                    "merged_from": filtered[keeper].get("merged_from", [filtered[keeper]["class_name"]])
                                   + [filtered[absorbed]["class_name"]],
                }
                print(f"    Merged '{filtered[absorbed]['class_name']}' into "
                      f"'{filtered[keeper]['class_name']}' (sim={sim:.3f})")

                # Recompute centroid for keeper after absorbing new members
                # (code review fix: stale centroid bug)
                new_idxs = [id_to_idx[m] for m in new_members if m in id_to_idx]
                if new_idxs:
                    new_centroid = emb_matrix[new_idxs].mean(axis=0)
                    norm = np.linalg.norm(new_centroid)
                    centroids[keeper] = new_centroid / (norm + 1e-9)

        # Remove absorbed clusters
        absorbed_indices = set(merged_into.keys())
        filtered = [c for i, c in enumerate(filtered) if i not in absorbed_indices]

    # Filter singletons from other resolution levels AND propagate merge remapping
    # (code review fix: derive_hierarchy sees orphaned class names without this)
    updated_levels: dict[str, list[dict]] = {}
    for res_str, clusters in named_levels.items():
        remapped: list[dict] = []
        for c in clusters:
            if len(c["members"]) < MIN_CLUSTER_SIZE:
                continue
            new_name = name_remapping.get(c["class_name"], c["class_name"])
            remapped.append({**c, "class_name": new_name})
        updated_levels[res_str] = remapped

    # Renumber cluster IDs
    for i, c in enumerate(filtered):
        c["cluster_id"] = i

    print(f"  ✓ {before_count} → {len(filtered)} clusters "
          f"(removed {singleton_removed} singletons, "
          f"merged {len(merged_into)} similar)")
    return {
        "named_clusters": filtered,
        "named_levels": updated_levels,
    }


def derive_hierarchy(state: Zone3State) -> dict:
    """Derive SUBCLASS_OF edges algorithmically from multi-resolution cluster membership.

    Instead of asking an LLM to guess is-a relationships (fragile, causes type
    inconsistency), we use the structural fact that Leiden clusters at a coarser
    resolution are supersets of finer-resolution clusters.

    For each adjacent (coarse, fine) resolution pair:
      - For each fine cluster, find the coarse cluster that contains the highest
        fraction of its members.
      - If overlap >= HIERARCHY_OVERLAP_THRESH, emit fine SUBCLASS_OF coarse.
    """
    print("\n[5a/5] Deriving ontology hierarchy algorithmically (multi-resolution)...")
    named_levels = state.get("named_levels", {})

    if len(named_levels) < 2:
        # Fallback: nothing to derive from — return empty hierarchy
        print("  ⚠ Need ≥2 resolution levels to derive hierarchy; returning empty")
        return {"hierarchy": []}

    # Sort resolutions numerically: smallest (coarsest) first
    sorted_res = sorted(float(r) for r in named_levels.keys())   # e.g. [0.3, 0.6, 1.2]
    print(f"  Resolutions available: {sorted_res}")

    hierarchy: list[dict] = []

    # Process adjacent pairs: (coarse=0.3, fine=0.6) then (coarse=0.6, fine=1.2)
    for i in range(len(sorted_res) - 1):
        coarse_res = str(sorted_res[i])       # e.g. "0.3"
        fine_res   = str(sorted_res[i + 1])   # e.g. "0.6"
        coarse_clusters = named_levels[coarse_res]
        fine_clusters   = named_levels[fine_res]

        pair_edges = 0
        for fine in fine_clusters:
            fine_members = set(fine["members"])
            if not fine_members:
                continue

            # Find coarse cluster with maximum member overlap
            best_overlap, best_parent = 0.0, None
            for coarse in coarse_clusters:
                overlap = len(fine_members & set(coarse["members"])) / len(fine_members)
                if overlap > best_overlap:
                    best_overlap, best_parent = overlap, coarse

            if best_parent and best_overlap >= HIERARCHY_OVERLAP_THRESH:
                child_name  = fine["class_name"]
                parent_name = best_parent["class_name"]
                if child_name != parent_name:
                    hierarchy.append({"child": child_name, "parent": parent_name})
                    pair_edges += 1
                    print(f"    {child_name} → {parent_name}  "
                          f"(overlap={best_overlap:.2f}, res {fine_res}→{coarse_res})")

        print(f"  Pair res={coarse_res}→{fine_res}: {pair_edges} SUBCLASS_OF edges")

    # Cycles are still theoretically possible (e.g. same class name at multiple levels)
    hierarchy = _remove_cycles(hierarchy)

    print(f"  ✓ {len(hierarchy)} SUBCLASS_OF edges (algorithmic, {len(sorted_res)}-level)")
    return {"hierarchy": hierarchy}


def _remove_cycles(pairs: list[dict]) -> list[dict]:
    """Remove SUBCLASS_OF pairs that create directed cycles (keep first-seen edges)."""
    from collections import defaultdict
    children: dict[str, set] = defaultdict(set)
    result = []
    removed = 0
    for rel in pairs:
        child, parent = rel["child"], rel["parent"]
        # Check if parent is already reachable from child (would create cycle)
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
            removed += 1
        else:
            children[child].add(parent)
            result.append(rel)
    if removed:
        print(f"  Removed {removed} cyclic edge(s)")
    return result


def _add_cluster_coherence(clusters: list[dict], entity_embeddings: dict) -> list[dict]:
    """
    Compute mean pairwise cosine similarity of member embeddings within each cluster.

    This is a *reference-free* quality metric for ontology induction:
    - High coherence (≥ 0.60): cluster members are semantically tight — good class candidate
    - Low coherence (< 0.30):  heterogeneous cluster — may need splitting or re-naming
    - Singletons receive coherence = 1.0 (maximally coherent by definition)

    Embeddings are L2-normalised (from SentenceTransformer with normalize_embeddings=True),
    so cosine similarity = dot product (no division needed).
    """
    emb_ids  = entity_embeddings.get("ids", [])
    emb_data = entity_embeddings.get("embs", [])
    if not emb_ids or not emb_data:
        return clusters

    emb_matrix = np.array(emb_data, dtype=np.float32)   # shape (N, 384)
    id_to_idx  = {eid: i for i, eid in enumerate(emb_ids)}

    for cluster in clusters:
        members = cluster["members"]
        idxs    = [id_to_idx[m] for m in members if m in id_to_idx]
        if len(idxs) >= 2:
            sub  = emb_matrix[idxs]                      # shape (k, 384)
            # Mean of all upper-triangle pairwise dot products
            sims = [float(np.dot(sub[i], sub[j]))
                    for i in range(len(idxs))
                    for j in range(i + 1, len(idxs))]
            cluster["coherence"] = round(float(np.mean(sims)), 4)
        else:
            cluster["coherence"] = 1.0    # singleton — maximally coherent

    return clusters


def write_ontology(state: Zone3State) -> dict:
    """Write the induced ontology to Neo4j (all resolution levels + SUBCLASS_OF edges)."""
    print("\n[5b/5] Writing ontology to Neo4j...")
    named      = state.get("named_clusters", [])    # primary (medium res) level
    named_levels = state.get("named_levels", {})    # all resolution levels
    hierarchy  = state.get("hierarchy", [])

    if not named:
        print("  ✗ No clusters to write")
        return {"neo4j_stats": {"error": "no clusters"}}

    try:
        graph = get_neo4j_graph()

        # Step 0: Clear any previous Zone 3 ontology labels from Entity nodes.
        existing_oc = graph.query(
            "MATCH (c:OntologyClass) RETURN c.name AS name"
        )
        for row in existing_oc:
            old_cls = _sanitize_label(row["name"]) if row["name"] else None
            if old_cls:
                try:
                    graph.query(
                        f"MATCH (n:Entity) WHERE n:{old_cls} REMOVE n:{old_cls}"
                    )
                except Exception as e:
                    print(f"  ⚠ Could not remove stale label {old_cls!r}: {e}")
        graph.query("MATCH (c:OntologyClass) DETACH DELETE c")
        print(f"  Cleared {len(existing_oc)} previous OntologyClass nodes")

        # Step 1: Label Entity nodes using the PRIMARY (medium-res) clusters.
        # Each entity gets the class label from its medium-res cluster membership.
        entities_labeled = 0
        primary_class_names: set[str] = set()
        for cluster in named:
            class_name = _sanitize_label(cluster["class_name"])
            members    = cluster["members"]
            if not class_name or not members:
                continue
            primary_class_names.add(class_name)
            graph.query(
                f"UNWIND $members AS mid "
                f"MATCH (n:Entity {{id: mid}}) "
                f"SET n:{class_name} "
                f"SET n.ontology_class = $cls",
                params={"members": members, "cls": class_name}
            )
            entities_labeled += len(members)

        # Step 2: Create OntologyClass nodes for ALL resolution levels.
        # We need nodes for every class referenced in the SUBCLASS_OF edges,
        # which span multiple resolution levels.
        all_clusters_across_levels: list[dict] = []
        if named_levels:
            for level_clusters in named_levels.values():
                all_clusters_across_levels.extend(level_clusters)
        else:
            all_clusters_across_levels = named   # fallback: primary only

        all_class_names_used: set[str] = set()
        for cluster in all_clusters_across_levels:
            class_name   = _sanitize_label(cluster["class_name"])
            member_count = len(cluster["members"])
            all_class_names_used.add(class_name)
            graph.query(
                "MERGE (c:OntologyClass {name: $name}) "
                "SET c.member_count = $count, "
                "    c.example_members = $examples",
                params={
                    "name": class_name,
                    "count": member_count,
                    "examples": cluster["members"][:10],
                }
            )

        # Step 3: Create SUBCLASS_OF edges (cross-level edges fully supported now).
        subclass_created = 0
        for rel in hierarchy:
            child  = _sanitize_label(rel["child"])
            parent = _sanitize_label(rel["parent"])
            if child in all_class_names_used and parent in all_class_names_used:
                graph.query(
                    "MATCH (child:OntologyClass {name: $child}) "
                    "MATCH (parent:OntologyClass {name: $parent}) "
                    "MERGE (child)-[:SUBCLASS_OF]->(parent)",
                    params={"child": child, "parent": parent}
                )
                subclass_created += 1

        # Final counts
        _nc = graph.query("MATCH (n:Entity) RETURN count(n) AS c")
        _oc = graph.query("MATCH (n:OntologyClass) RETURN count(n) AS c")
        _sc = graph.query("MATCH ()-[r:SUBCLASS_OF]->() RETURN count(r) AS c")
        all_labels = [r["label"] for r in graph.query(
            "CALL db.labels() YIELD label RETURN label"
        )]

        stats = {
            "entities_labeled": entities_labeled,
            "ontology_classes": _oc[0]["c"] if _oc else 0,
            "subclass_of_edges": _sc[0]["c"] if _sc else 0,
            "class_names": sorted(primary_class_names),
            "all_class_names": sorted(all_class_names_used),
            "all_labels": all_labels,
            "hierarchy": hierarchy,
        }

        print(f"  ✓ Entities labeled:     {entities_labeled}")
        print(f"  ✓ OntologyClass nodes:  {stats['ontology_classes']} "
              f"(primary: {len(primary_class_names)}, all levels: {len(all_class_names_used)})")
        print(f"  ✓ SUBCLASS_OF edges:    {subclass_created}")
        print(f"  ✓ Primary classes:      {sorted(primary_class_names)}")
        return {"neo4j_stats": stats}

    except Exception as e:
        print(f"  ✗ Neo4j error: {e}")
        return {"neo4j_stats": {"error": str(e)}}


# ---------------------------------------------------------------------------
# LangGraph Build
# ---------------------------------------------------------------------------

def build_pipeline():
    builder = StateGraph(Zone3State)
    builder.add_node("load_entities",              load_entities)
    builder.add_node("build_similarity_graph",     build_similarity_graph)
    builder.add_node("leiden_cluster",             leiden_cluster)
    builder.add_node("name_clusters",              name_clusters)
    builder.add_node("filter_and_merge_clusters",  filter_and_merge_clusters)
    builder.add_node("derive_hierarchy",           derive_hierarchy)
    builder.add_node("write_ontology",             write_ontology)
    builder.set_entry_point("load_entities")
    builder.add_edge("load_entities",              "build_similarity_graph")
    builder.add_edge("build_similarity_graph",     "leiden_cluster")
    builder.add_edge("leiden_cluster",             "name_clusters")
    builder.add_edge("name_clusters",              "filter_and_merge_clusters")
    builder.add_edge("filter_and_merge_clusters",  "derive_hierarchy")
    builder.add_edge("derive_hierarchy",           "write_ontology")
    builder.add_edge("write_ontology",             END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_zone3(model: str = config.OLLAMA_MODEL):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("CS584 Capstone — Zone 3 Pipeline [Ontology Induction via Leiden]")
    print(f"Bottom-up class discovery + hierarchy  (model: {model})")
    print("=" * 60)

    pipeline = build_pipeline()
    start    = time.time()
    result   = pipeline.invoke({
        "entities":          [],
        "entity_map":        {},
        "similarity_edges":  [],
        "entity_embeddings": {},
        "clusters":          [],
        "cluster_levels":    {},
        "named_clusters":    [],
        "named_levels":      {},
        "hierarchy":         [],
        "neo4j_stats":       {},
        "errors":            [],
        "model":             model,
    })
    elapsed = time.time() - start

    neo4j_stats = result.get("neo4j_stats", {})
    named       = result.get("named_clusters", [])
    hierarchy   = result.get("hierarchy", [])

    # Coherence statistics (reference-free quality metric for cluster tightness)
    all_coh = [c.get("coherence") for c in named if c.get("coherence") is not None]
    ns_coh  = [c.get("coherence") for c in named
               if len(c.get("members", [])) > 1 and c.get("coherence") is not None]
    avg_coh    = round(float(np.mean(all_coh)), 4) if all_coh else None
    avg_ns_coh = round(float(np.mean(ns_coh)),  4) if ns_coh  else None

    print(f"\n{'=' * 60}")
    print(f"Zone 3 pipeline complete in {elapsed:.1f}s")
    print(f"  Ontology classes:  {len(named)}")
    print(f"  SUBCLASS_OF:       {len(hierarchy)}")
    if avg_coh is not None:
        print(f"  Avg cluster coherence (all):           {avg_coh:.4f}")
    if avg_ns_coh is not None:
        print(f"  Avg cluster coherence (non-singleton): {avg_ns_coh:.4f}")
    print(f"  Neo4j stats:       {neo4j_stats}")

    # Save summary
    summary = {
        "mode": "zone3",
        "model": model,
        "elapsed_seconds": round(elapsed, 2),
        "entity_count": len(result.get("entities", [])),
        "similarity_edges": len(result.get("similarity_edges", [])),
        "clusters": len(named),
        "avg_cluster_coherence": avg_coh,
        "avg_cluster_coherence_non_singleton": avg_ns_coh,
        "named_clusters": named,
        "hierarchy": hierarchy,
        "neo4j_stats": neo4j_stats,
        "errors": result.get("errors", []),
    }
    out_path = os.path.join(config.RESULTS_DIR, "zone3_run_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✓ Summary saved to {out_path}")
    print("\nNext steps:")
    print("  python3 baseline/eval.py --suffix zone3")
    print("  python3 baseline/eval.py --suffix zone3 --riskine")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zone 3 Ontology Induction via Leiden Community Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pre-requisite: Run zone2/pipeline.py first to populate Neo4j with Entity nodes.

Examples:
  python3 zone3/pipeline.py
  python3 zone3/pipeline.py --model qwen2.5:7b
  # After running, evaluate with:
  python3 baseline/eval.py --suffix zone3 --riskine
        """,
    )
    parser.add_argument(
        "--model", default=config.OLLAMA_MODEL,
        help=f"Ollama model name (default: {config.OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--method", choices=["leiden", "rsi"], default="leiden",
        help="Induction method: leiden (baseline) or rsi (RSI-LCR novel method)"
    )
    parser.add_argument(
        "--suffix", default=None,
        help="Suffix for result files (default: zone3 for leiden, zone3_rsi for rsi)"
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Force cluster count for RSI method (default: silhouette-guided)"
    )
    parser.add_argument(
        "--refinement-rounds", type=int, default=2,
        help="LLM coherence refinement rounds for RSI method (default: 2)"
    )
    args = parser.parse_args()

    if args.method == "rsi":
        from zone3.rsi_lcr import run_rsi_lcr
        suffix = args.suffix or "zone3_rsi"
        run_rsi_lcr(
            model=args.model,
            suffix=suffix,
            k_override=args.k,
            refinement_rounds=args.refinement_rounds,
        )
    else:
        run_zone3(model=args.model)
