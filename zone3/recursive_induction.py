"""Zone 3 — Recursive Divisive Ontology Induction

Builds multi-level ontology hierarchies via recursive top-down splitting,
using 5 fused signals: entity name semantics, relation profiles, Zone 2
entity types, source provenance, and LLM semantic judgment.

Architecture:
  Stage A: Build multi-signal entity signatures
  Stage B: Discover 3-6 macro-classes (top level)
  Stage C: Recursive divisive splitting (core contribution)
  Stage D: Local sibling merge
  Stage E: Write ontology to Neo4j

Key design: hierarchy emerges DURING induction, not as a post-hoc
recovery step. Each split creates parent + children simultaneously.

Usage:
  python3 zone3/recursive_induction.py
  python3 zone3/recursive_induction.py --model qwen2.5:72b --suffix recursive_v1
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import random
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from zone3.graph_cache import (
    load_cached_entities,
    get_concept_entities,
    get_entity_lane,
    is_concept_entity,
)
from zone3.sv_loi import (
    get_llm,
    get_neo4j_graph,
    _invoke_llm,
    _sanitize_label,
    _parse_json_safely,
    write_ontology,
    derive_interclass_edges,
    propagate_to_records,
    type_value_entities,
)

from langchain_ollama import ChatOllama


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DEPTH = 4
MAX_CLASSES = 25
MIN_SPLIT_SIZE = 20          # minimum concept members to attempt split
MIN_CHILD_SIZE = 8           # absolute min for a child
MIN_CHILD_FRACTION = 0.12    # min fraction of parent
SILHOUETTE_THRESHOLD = 0.12
NAME_SIL_THRESHOLD = 0.15
JS_THRESHOLD = 0.08
TYPE_MIN_SUPPORT = 8         # min entities for entity_type split
MACRO_K_RANGE = (3, 7)       # range for macro-class count
LLM_CALL_BUDGET = 120        # hard cap on total LLM calls

# Signals that should NOT drive splits (data artifacts)
VALUE_ENTITY_TYPES = frozenset({
    "Numeric", "Text", "Date", "Unknown", "Categorical", "Field", "Score",
    "TimePeriod",
})


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TaxonomyNode:
    name: str
    entity_ids: list[str]
    depth: int
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    split_method: str = ""        # "entity_type" | "name_semantic" | "structural" | "macro"
    split_score: float = 0.0
    metrics: dict = field(default_factory=dict)


@dataclass
class Taxonomy:
    nodes: dict[str, TaxonomyNode] = field(default_factory=dict)

    def add_root(self, node: TaxonomyNode) -> None:
        self.nodes[node.name] = node

    def add_child(self, parent_name: str, child: TaxonomyNode) -> None:
        child.parent = parent_name
        self.nodes[child.name] = child
        if parent_name in self.nodes:
            self.nodes[parent_name].children.append(child.name)

    def get_edges(self) -> list[tuple[str, str]]:
        """Return (child, parent) tuples for SUBCLASS_OF."""
        return [
            (name, node.parent)
            for name, node in self.nodes.items()
            if node.parent is not None
        ]

    def get_leaves(self) -> list[TaxonomyNode]:
        return [n for n in self.nodes.values() if not n.children]

    def max_depth(self) -> int:
        return max((n.depth for n in self.nodes.values()), default=0)

    def class_count(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_llm_call_count = 0


def _counted_llm(llm: ChatOllama, prompt: str) -> str:
    """LLM call with global budget tracking."""
    global _llm_call_count
    if _llm_call_count >= LLM_CALL_BUDGET:
        print(f"  [LLM] Budget exhausted ({LLM_CALL_BUDGET} calls)", flush=True)
        return ""
    _llm_call_count += 1
    return _invoke_llm(llm, prompt)


def _flush(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Provenance: chunk_id → source file
# ---------------------------------------------------------------------------

def load_chunk_source_map(data_dir: str | None = None) -> dict[str, str]:
    """Map chunk_id → source filename from zone1_chunks.json.

    Returns dict like {"0": "Auto_Service_form_masked.pdf", "5": "geico_renters_claims.csv"}
    """
    # Try multiple locations
    candidates = []
    if data_dir:
        candidates.append(os.path.join(data_dir, "processed", "zone1_chunks.json"))
    # Try common data directories
    for d in ["data/Emory_Spring2026", "data/flood"]:
        candidates.append(os.path.join(d, "processed", "zone1_chunks.json"))

    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                chunks = json.load(f)
            chunk_map = {}
            for c in chunks:
                cid = str(c.get("chunk_id", ""))
                source = c.get("source", "")
                if cid and source:
                    # Normalize: extract just the filename
                    fname = os.path.basename(source)
                    chunk_map[cid] = fname
            _flush(f"  ✓ Loaded {len(chunk_map)} chunk→source mappings from {path}")
            return chunk_map

    _flush("  ⚠ No zone1_chunks.json found — provenance signal unavailable")
    return {}


def _derive_lob_from_source(filename: str) -> str:
    """Derive a short LOB label from source filename."""
    fn = filename.lower()
    if "renter" in fn or "geico" in fn:
        return "renters"
    elif "tmobile" in fn or "mobile" in fn:
        return "mobile"
    elif "auto" in fn or "vehicle" in fn or "service" in fn:
        return "auto"
    else:
        return "unknown"


# ---------------------------------------------------------------------------
# Stage A: Build Multi-Signal Entity Signatures
# ---------------------------------------------------------------------------

def build_entity_signatures(
    entities: list[dict],
    chunk_source_map: dict[str, str],
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, dict]:
    """Build multi-signal signature for each entity.

    Returns dict: eid -> {name_emb, rel_profile, type_vec, source_vec, combined,
                          entity_type, source_file, lob}
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer

    _flush("\n[Stage A] Building entity signatures...")

    concept_ents = [e for e in entities if is_concept_entity(e)]
    _flush(f"  {len(concept_ents)} concept entities")

    # --- Name embeddings ---
    _flush("  Computing name embeddings...")
    st_model = SentenceTransformer(model_name)
    names = [e["id"] for e in concept_ents]
    name_embs = st_model.encode(names, show_progress_bar=False, normalize_embeddings=True)
    name_dim = name_embs.shape[1]  # typically 384

    # --- Relation profiles (binary participation → TF-IDF → SVD) ---
    _flush("  Computing relation profiles...")
    all_rels: set[str] = set()
    for e in concept_ents:
        all_rels.update(e.get("out_rel_counts", {}).keys())
        all_rels.update(f"~{r}" for r in e.get("in_rel_counts", {}).keys())
    rel_list = sorted(all_rels)
    rel_idx = {r: i for i, r in enumerate(rel_list)}

    rel_mat = np.zeros((len(concept_ents), len(rel_list)))
    for i, e in enumerate(concept_ents):
        for r in e.get("out_rel_counts", {}):
            if r in rel_idx:
                rel_mat[i, rel_idx[r]] = 1.0
        for r in e.get("in_rel_counts", {}):
            key = f"~{r}"
            if key in rel_idx:
                rel_mat[i, rel_idx[key]] = 1.0

    # TF-IDF + SVD to reduce dimensionality
    rel_dim = min(64, len(rel_list) - 1, len(concept_ents) - 1)
    if rel_dim > 1 and rel_mat.sum() > 0:
        tfidf = TfidfTransformer()
        rel_tfidf = tfidf.fit_transform(rel_mat)
        svd = TruncatedSVD(n_components=rel_dim, random_state=42)
        rel_svd = svd.fit_transform(rel_tfidf)
        # L2 normalize
        norms = np.linalg.norm(rel_svd, axis=1, keepdims=True)
        rel_svd = rel_svd / np.maximum(norms, 1e-9)
    else:
        rel_svd = np.zeros((len(concept_ents), max(rel_dim, 1)))

    # --- Zone 2 entity_type (one-hot) ---
    _flush("  Computing entity type vectors...")
    all_types = sorted(set(e.get("entity_type", "Unknown") for e in concept_ents))
    type_idx = {t: i for i, t in enumerate(all_types)}
    type_dim = len(all_types)
    type_mat = np.zeros((len(concept_ents), type_dim))
    for i, e in enumerate(concept_ents):
        et = e.get("entity_type", "Unknown")
        if et in type_idx:
            type_mat[i, type_idx[et]] = 1.0

    # --- Source provenance ---
    _flush("  Computing provenance vectors...")
    all_sources = sorted(set(chunk_source_map.values())) if chunk_source_map else []
    source_idx = {s: i for i, s in enumerate(all_sources)}
    source_dim = len(all_sources)

    source_mat = np.zeros((len(concept_ents), max(source_dim, 1)))
    entity_sources: dict[str, str] = {}
    entity_lobs: dict[str, str] = {}
    for i, e in enumerate(concept_ents):
        source_counts: Counter = Counter()
        for r in e.get("out_rels", []):
            cid = str(r.get("chunk_id", ""))
            if cid in chunk_source_map:
                source_counts[chunk_source_map[cid]] += 1
        for r in e.get("in_rels", []):
            cid = str(r.get("chunk_id", ""))
            if cid in chunk_source_map:
                source_counts[chunk_source_map[cid]] += 1
        if source_counts:
            top_source = source_counts.most_common(1)[0][0]
            entity_sources[e["id"]] = top_source
            entity_lobs[e["id"]] = _derive_lob_from_source(top_source)
            for src, cnt in source_counts.items():
                if src in source_idx:
                    source_mat[i, source_idx[src]] = cnt

    # L2 normalize source vectors
    norms = np.linalg.norm(source_mat, axis=1, keepdims=True)
    source_mat = source_mat / np.maximum(norms, 1e-9)

    # SVD if too many sources
    if source_dim > 16:
        svd_src = TruncatedSVD(n_components=min(16, source_dim - 1), random_state=42)
        source_mat = svd_src.fit_transform(source_mat)
        norms = np.linalg.norm(source_mat, axis=1, keepdims=True)
        source_mat = source_mat / np.maximum(norms, 1e-9)

    # --- Combine with weights ---
    _flush("  Combining signatures (name=0.45, rel=0.30, type=0.15, source=0.10)...")
    signatures = {}
    for i, e in enumerate(concept_ents):
        combined = np.concatenate([
            0.45 * name_embs[i],
            0.30 * rel_svd[i],
            0.15 * type_mat[i],
            0.10 * source_mat[i],
        ])
        signatures[e["id"]] = {
            "name_emb": name_embs[i],
            "rel_profile": rel_svd[i],
            "type_vec": type_mat[i],
            "source_vec": source_mat[i],
            "combined": combined,
            "entity_type": e.get("entity_type", "Unknown"),
            "source_file": entity_sources.get(e["id"], ""),
            "lob": entity_lobs.get(e["id"], ""),
        }

    _flush(f"  ✓ Signatures built: {len(signatures)} entities, "
           f"dim={name_dim}+{rel_dim}+{type_dim}+{source_mat.shape[1]}")
    return signatures


# ---------------------------------------------------------------------------
# Stage A.5: LLM-Learned Type Normalization
# ---------------------------------------------------------------------------

def learn_type_normalization(
    entity_types: list[str],
    type_counts: dict[str, int],
    llm: ChatOllama,
) -> dict[str, str | None]:
    """Ask LLM to group/normalize Zone 2 entity types into ontology classes.

    Returns map: entity_type → normalized class name (or None to drop).
    Fully domain-agnostic — no hardcoded knowledge.
    """
    _flush("\n[Stage A.5] Learning entity type normalization via LLM...")

    # Filter to meaningful types
    meaningful = [(t, c) for t, c in type_counts.items() if t not in VALUE_ENTITY_TYPES and c >= 3]
    if not meaningful:
        return {}

    type_desc = "\n".join(f"  {t} ({c} entities)" for t, c in sorted(meaningful, key=lambda x: -x[1]))

    prompt = f"""These are entity types extracted from a knowledge graph:

{type_desc}

Group these types into ontology classes. For each type, decide:
1. Keep as its own class (if it's a distinct real-world role)
2. Merge into a broader class (if it's a specific instance of a general concept)
3. Drop (if it's a data attribute, not a domain concept)

Output JSON array:
[{{"type": "OriginalType", "class": "NormalizedClassName", "action": "keep|merge|drop"}}]

Rules:
- Class names should be real-world domain roles (e.g., Organization, Coverage, Risk)
- Use PascalCase, 1-2 words
- Types that represent the same concept should map to the same class
- Types that are data attributes (amounts, dates, statuses) should be dropped"""

    raw = _counted_llm(llm, prompt)
    parsed = _parse_json_safely(raw)

    norm_map: dict[str, str | None] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            orig = item.get("type", "")
            action = item.get("action", "keep")
            cls = item.get("class", orig)
            if action == "drop":
                norm_map[orig] = None
            else:
                norm_map[orig] = _sanitize_label(cls) if cls else orig

    _flush(f"  ✓ Normalization learned for {len(norm_map)} types:")
    for orig, norm in sorted(norm_map.items()):
        if norm:
            _flush(f"    {orig} → {norm}")
        else:
            _flush(f"    {orig} → (dropped)")

    return norm_map


# ---------------------------------------------------------------------------
# Stage B: Discover Macro-Classes
# ---------------------------------------------------------------------------

def discover_macro_classes(
    concept_entities: list[dict],
    signatures: dict[str, dict],
    llm: ChatOllama,
    seed: int = 42,
) -> list[TaxonomyNode]:
    """Discover 3-6 top-level macro-classes via clustering + LLM naming."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    _flush("\n[Stage B] Discovering macro-classes...")

    # Build combined signature matrix for concept entities
    eids = [e["id"] for e in concept_entities if e["id"] in signatures]
    if not eids:
        _flush("  ERROR: No concept entities with signatures")
        return []

    sig_matrix = np.array([signatures[eid]["combined"] for eid in eids])

    # Try k = 3..6, pick best silhouette
    best_k, best_sil, best_labels = 3, -1.0, None
    k_lo, k_hi = MACRO_K_RANGE
    for k in range(k_lo, min(k_hi, len(eids))):
        try:
            clust = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            labels = clust.fit_predict(sig_matrix)
            sil = silhouette_score(sig_matrix, labels, metric="cosine", random_state=seed)
            _flush(f"  k={k}: silhouette={sil:.3f}")
            # Prefer higher k if silhouette is within 0.05 of best (favor more classes)
            if sil > best_sil - 0.05:
                if sil > best_sil or k > best_k:
                    best_sil = sil
                    best_k = k
                    best_labels = labels
        except Exception as exc:
            _flush(f"  k={k}: failed ({exc})")

    if best_labels is None:
        _flush("  ERROR: All clustering attempts failed")
        return []

    _flush(f"  ✓ Best k={best_k}, silhouette={best_sil:.3f}")

    # Build macro-class nodes
    cluster_eids: dict[int, list[str]] = defaultdict(list)
    for eid, label in zip(eids, best_labels):
        cluster_eids[label].append(eid)

    entity_map = {e["id"]: e for e in concept_entities}
    roots: list[TaxonomyNode] = []

    for label, member_eids in sorted(cluster_eids.items()):
        # Summarize cluster for LLM naming
        type_counts = Counter(
            signatures[eid]["entity_type"] for eid in member_eids
            if signatures[eid]["entity_type"] not in VALUE_ENTITY_TYPES
        )
        sample_names = random.sample(member_eids, min(10, len(member_eids)))
        top_types = type_counts.most_common(5)

        # Top relations
        rel_counts: Counter = Counter()
        for eid in member_eids[:50]:
            e = entity_map.get(eid, {})
            for r in e.get("out_rel_counts", {}):
                rel_counts[r] += 1
        top_rels = rel_counts.most_common(5)

        prompt = f"""Name this ontology class. It contains {len(member_eids)} entities.

Example entities: {', '.join(sample_names[:8])}

Dominant entity types: {', '.join(f'{t}({c})' for t, c in top_types)}

Top relations: {', '.join(f'{r}({c})' for r, c in top_rels)}

What single PascalCase word best describes the REAL-WORLD ROLE of these entities?
(e.g., Policy, Coverage, Claim, Person, Organization, Property, Risk, Process)

Answer with ONLY the class name:"""

        raw = _counted_llm(llm, prompt).strip()
        name_match = re.match(r'^([A-Z][A-Za-z]{1,25})$', raw.split('\n')[0].strip())
        name = _sanitize_label(name_match.group(1)) if name_match else f"Class{label}"

        node = TaxonomyNode(
            name=name,
            entity_ids=member_eids,
            depth=0,
            split_method="macro",
            metrics={"size": len(member_eids), "silhouette": round(best_sil, 3)},
        )
        roots.append(node)
        _flush(f"  Macro-class {label}: {name} ({len(member_eids)} entities, "
               f"types: {', '.join(t for t, _ in top_types[:3])})")

    return roots


# ---------------------------------------------------------------------------
# Stage C: Recursive Divisive Splitting
# ---------------------------------------------------------------------------

def _try_entity_type_split(
    node: TaxonomyNode,
    entity_map: dict[str, dict],
    signatures: dict[str, dict],
    norm_map: dict[str, str | None],
    llm: ChatOllama,
) -> list[TaxonomyNode] | None:
    """Signal 1: Split by Zone 2 entity_type distribution."""
    type_groups: dict[str, list[str]] = defaultdict(list)
    for eid in node.entity_ids:
        if eid not in signatures:
            continue
        raw_type = signatures[eid]["entity_type"]
        if raw_type in VALUE_ENTITY_TYPES:
            type_groups["_value"].append(eid)
            continue
        norm = norm_map.get(raw_type, raw_type)
        if norm is None:
            type_groups["_dropped"].append(eid)
        else:
            type_groups[norm].append(eid)

    # Filter to significant types
    n = len(node.entity_ids)
    min_child = max(MIN_CHILD_SIZE, int(n * MIN_CHILD_FRACTION))
    significant = {t: eids for t, eids in type_groups.items()
                   if not t.startswith("_") and len(eids) >= min_child}

    if len(significant) < 2:
        return None

    # Don't split if one type dominates >90%
    largest = max(len(eids) for eids in significant.values())
    if largest > n * 0.90:
        return None

    _flush(f"    [type-split] {len(significant)} significant types found")

    # Keep largest as parent, validate others as subclasses
    sorted_types = sorted(significant.items(), key=lambda x: -len(x[1]))
    children: list[TaxonomyNode] = []

    for type_name, type_eids in sorted_types[1:]:  # skip largest (stays as parent)
        sanitized = _sanitize_label(type_name)
        if sanitized.lower() == node.name.lower():
            continue

        # LLM validation
        sample = random.sample(type_eids, min(5, len(type_eids)))
        prompt = (
            f'In a domain ontology, is "{type_name}" a valid subclass (IS-A) of "{node.name}"?\n'
            f'"{type_name}" entities: {", ".join(sample)}\n'
            f'Answer YES, NO, or SIBLING (if they should be separate top-level classes).'
        )
        verdict = _counted_llm(llm, prompt).strip().upper()

        if "NO" in verdict and "YES" not in verdict:
            _flush(f"    LLM rejected: {type_name} is not a subclass of {node.name}")
            continue
        if "SIBLING" in verdict:
            _flush(f"    LLM says sibling: {type_name} (will become separate root)")
            # TODO: handle sibling promotion in future version
            continue

        children.append(TaxonomyNode(
            name=sanitized,
            entity_ids=type_eids,
            depth=node.depth + 1,
            split_method="entity_type",
            metrics={"size": len(type_eids), "parent_type": sorted_types[0][0]},
        ))
        _flush(f"    → {sanitized} SUBCLASS_OF {node.name} ({len(type_eids)} entities)")

    return children if children else None


def _try_name_semantic_split(
    node: TaxonomyNode,
    signatures: dict[str, dict],
    llm: ChatOllama,
    seed: int = 42,
) -> list[TaxonomyNode] | None:
    """Signal 2: Split by entity name embedding clustering."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    eids = [eid for eid in node.entity_ids if eid in signatures]
    if len(eids) < MIN_SPLIT_SIZE:
        return None

    name_embs = np.array([signatures[eid]["name_emb"] for eid in eids])

    best_k, best_sil, best_labels = 2, -1.0, None
    k_max = min(4, len(eids) // MIN_CHILD_SIZE)
    for k in range(2, k_max + 1):
        try:
            clust = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            labels = clust.fit_predict(name_embs)
            sizes = Counter(labels)
            if min(sizes.values()) < MIN_CHILD_SIZE:
                continue
            sil = silhouette_score(name_embs, labels, metric="cosine", random_state=seed)
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels
        except Exception:
            continue

    if best_labels is None or best_sil < NAME_SIL_THRESHOLD:
        return None

    _flush(f"    [name-split] k={best_k}, silhouette={best_sil:.3f}")

    # Group and name clusters
    cluster_eids: dict[int, list[str]] = defaultdict(list)
    for eid, label in zip(eids, best_labels):
        cluster_eids[label].append(eid)

    # Keep largest as parent
    sorted_clusters = sorted(cluster_eids.items(), key=lambda x: -len(x[1]))
    n = len(eids)
    min_child = max(MIN_CHILD_SIZE, int(n * MIN_CHILD_FRACTION))

    children: list[TaxonomyNode] = []
    for label, child_eids in sorted_clusters[1:]:
        if len(child_eids) < min_child:
            continue

        sample = random.sample(child_eids, min(8, len(child_eids)))
        prompt = (
            f'These entities are a subgroup of "{node.name}":\n'
            f'{", ".join(sample)}\n\n'
            f'What is a good SUBCLASS NAME for this group? It must be a genuine '
            f'ontological subtype of {node.name} (IS-A relationship).\n'
            f'Use PascalCase, 1-3 words. If not a real subtype, say SKIP.\n'
            f'Answer with ONLY the name (or SKIP):'
        )
        raw = _counted_llm(llm, prompt).strip()
        match = re.match(r'^([A-Z][A-Za-z0-9]{2,30})$', raw.split('\n')[0].strip())
        if not match or match.group(1).upper() == "SKIP":
            _flush(f"    cluster {label}: LLM rejected naming")
            continue

        name = _sanitize_label(match.group(1))
        if name.lower() == node.name.lower():
            name = f"{node.name}Sub{label}"

        children.append(TaxonomyNode(
            name=name,
            entity_ids=child_eids,
            depth=node.depth + 1,
            split_method="name_semantic",
            metrics={"size": len(child_eids), "silhouette": round(best_sil, 3)},
        ))
        _flush(f"    → {name} SUBCLASS_OF {node.name} ({len(child_eids)} entities)")

    return children if children else None


def _try_structural_split(
    node: TaxonomyNode,
    entity_map: dict[str, dict],
    signatures: dict[str, dict],
    llm: ChatOllama,
    seed: int = 42,
) -> list[TaxonomyNode] | None:
    """Signal 3: Split by relation profile clustering (SOHD-style)."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_distances
    from zone3.sv_loi import (
        _build_class_relation_profiles,
        _js_divergence,
        _top_distinguishing_relations,
    )

    members = [entity_map[eid] for eid in node.entity_ids
               if eid in entity_map and is_concept_entity(entity_map[eid])]
    if len(members) < MIN_SPLIT_SIZE:
        return None

    raw_profiles, rel_names = _build_class_relation_profiles(members)
    if raw_profiles.shape[1] < 2:
        return None

    # Filter zero vectors
    nonzero = raw_profiles.sum(axis=1) > 0
    members = [members[i] for i in range(len(members)) if nonzero[i]]
    raw_profiles = raw_profiles[nonzero]
    n = len(members)
    if n < MIN_SPLIT_SIZE:
        return None

    cos_dist = cosine_distances(raw_profiles)
    np.fill_diagonal(cos_dist, 0)

    k_max = min(4, n // MIN_CHILD_SIZE)
    if k_max < 2:
        return None

    best_k, best_sil, best_labels = 2, -1.0, None
    for k in range(2, k_max + 1):
        try:
            clust = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average",
            )
            labels = clust.fit_predict(cos_dist)
            sizes = Counter(labels)
            if min(sizes.values()) < 2:
                continue
            sil = silhouette_score(cos_dist, labels, metric="precomputed", random_state=seed)
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels
        except Exception:
            continue

    if best_labels is None or best_sil < SILHOUETTE_THRESHOLD:
        return None

    _flush(f"    [struct-split] k={best_k}, silhouette={best_sil:.3f}")

    cluster_map: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(best_labels):
        cluster_map[label].append(idx)

    sorted_clusters = sorted(cluster_map.items(), key=lambda x: -len(x[1]))
    min_child = max(MIN_CHILD_SIZE, int(n * MIN_CHILD_FRACTION))
    children: list[TaxonomyNode] = []

    for label, indices in sorted_clusters[1:]:
        if len(indices) < min_child:
            continue

        sub_mean = raw_profiles[indices].mean(axis=0)
        comp_idx = [i for i in range(n) if i not in set(indices)]
        if not comp_idx:
            continue
        comp_mean = raw_profiles[comp_idx].mean(axis=0)

        js = _js_divergence(sub_mean, comp_mean)
        if js < JS_THRESHOLD:
            continue

        dist_rels = _top_distinguishing_relations(sub_mean, comp_mean, rel_names)
        child_eids = [members[i]["id"] for i in indices]
        sample = [members[i]["id"] for i in indices[:8]]

        # LLM naming
        rel_desc = ", ".join(f"{r}({ratio:.1f}x)" for r, ratio in dist_rels[:3])
        prompt = (
            f'A subgroup of "{node.name}" has distinct relation patterns:\n'
            f'Enriched relations: {rel_desc}\n'
            f'Example entities: {", ".join(sample)}\n'
            f'Name this subclass (PascalCase, 1-3 words). Say SKIP if not a real subtype.\n'
            f'Answer with ONLY the name (or SKIP):'
        )
        raw = _counted_llm(llm, prompt).strip()
        match = re.match(r'^([A-Z][A-Za-z0-9]{2,30})$', raw.split('\n')[0].strip())
        if not match or match.group(1).upper() == "SKIP":
            continue

        name = _sanitize_label(match.group(1))
        if name.lower() == node.name.lower():
            name = f"{node.name}Sub{label}"

        children.append(TaxonomyNode(
            name=name,
            entity_ids=child_eids,
            depth=node.depth + 1,
            split_method="structural",
            metrics={"size": len(child_eids), "js": round(js, 3), "silhouette": round(best_sil, 3)},
        ))
        _flush(f"    → {name} SUBCLASS_OF {node.name} ({len(child_eids)} entities, JS={js:.3f})")

    return children if children else None


def induce_recursive_taxonomy(
    roots: list[TaxonomyNode],
    signatures: dict[str, dict],
    entity_map: dict[str, dict],
    norm_map: dict[str, str | None],
    llm: ChatOllama,
    seed: int = 42,
) -> Taxonomy:
    """Recursively split macro-classes into subclasses using 3 signals."""
    _flush("\n[Stage C] Recursive divisive splitting...")

    taxonomy = Taxonomy()
    for root in roots:
        taxonomy.add_root(root)

    def _recurse(node: TaxonomyNode) -> None:
        n = len(node.entity_ids)
        if n < MIN_SPLIT_SIZE or node.depth >= MAX_DEPTH:
            return
        if taxonomy.class_count() >= MAX_CLASSES:
            _flush(f"  {node.name}: max classes ({MAX_CLASSES}) reached, stop")
            return

        _flush(f"\n  Splitting {node.name} (n={n}, depth={node.depth})...")

        # Signal 1: Entity_type heterogeneity (most direct)
        children = _try_entity_type_split(node, entity_map, signatures, norm_map, llm)
        if children:
            _apply_split(node, children)
            return

        # Signal 2: Name-semantic clustering
        children = _try_name_semantic_split(node, signatures, llm, seed)
        if children:
            _apply_split(node, children)
            return

        # Signal 3: Structural relation-profile clustering
        children = _try_structural_split(node, entity_map, signatures, llm, seed)
        if children:
            _apply_split(node, children)
            return

        _flush(f"  {node.name}: no valid split found")

    def _apply_split(parent: TaxonomyNode, children: list[TaxonomyNode]) -> None:
        # Remove children's eids from parent
        child_eids = set()
        for child in children:
            child_eids.update(child.entity_ids)
            taxonomy.add_child(parent.name, child)

        # Parent keeps remaining entities
        parent.entity_ids = [eid for eid in parent.entity_ids if eid not in child_eids]

        # Recurse on children
        for child in children:
            _recurse(child)

    for root in roots:
        _recurse(root)

    _flush(f"\n  ✓ Taxonomy: {taxonomy.class_count()} classes, "
           f"depth={taxonomy.max_depth()}, "
           f"{len(taxonomy.get_edges())} IS-A edges")
    return taxonomy


# ---------------------------------------------------------------------------
# Stage D: Local Sibling Merge
# ---------------------------------------------------------------------------

def merge_local_siblings(
    taxonomy: Taxonomy,
    signatures: dict[str, dict],
    llm: ChatOllama,
) -> Taxonomy:
    """Merge near-duplicate sibling classes under the same parent."""
    _flush("\n[Stage D] Local sibling merge...")
    merges = 0

    for node in list(taxonomy.nodes.values()):
        if len(node.children) < 2:
            continue

        # Compute centroids for each child
        child_centroids: dict[str, np.ndarray] = {}
        for child_name in node.children:
            child_node = taxonomy.nodes.get(child_name)
            if not child_node:
                continue
            embs = [signatures[eid]["combined"] for eid in child_node.entity_ids
                    if eid in signatures]
            if embs:
                child_centroids[child_name] = np.mean(embs, axis=0)

        # Check all pairs
        children_list = list(child_centroids.keys())
        merged_into: dict[str, str] = {}
        for i in range(len(children_list)):
            a = children_list[i]
            if a in merged_into:
                continue
            for j in range(i + 1, len(children_list)):
                b = children_list[j]
                if b in merged_into:
                    continue
                # Cosine similarity
                cos = np.dot(child_centroids[a], child_centroids[b]) / (
                    np.linalg.norm(child_centroids[a]) * np.linalg.norm(child_centroids[b]) + 1e-9
                )
                if cos >= 0.95:
                    # Auto-merge
                    _flush(f"  Auto-merge: {b} → {a} (cosine={cos:.3f})")
                    merged_into[b] = a
                    merges += 1

        # Apply merges
        for src, dst in merged_into.items():
            src_node = taxonomy.nodes.pop(src, None)
            if src_node:
                taxonomy.nodes[dst].entity_ids.extend(src_node.entity_ids)
                node.children.remove(src)

    _flush(f"  ✓ {merges} merges applied")
    return taxonomy


# ---------------------------------------------------------------------------
# Stage E: Assignment + Write
# ---------------------------------------------------------------------------

def assign_all_entities(
    taxonomy: Taxonomy,
    entities: list[dict],
    signatures: dict[str, dict],
) -> dict[str, str]:
    """Assign all entities (concepts + records + values) to taxonomy leaves."""
    _flush("\n[Stage E] Assigning all entities...")

    # Concept entities: assigned during recursion
    assignments: dict[str, str] = {}

    # From taxonomy nodes (concepts)
    for node in taxonomy.nodes.values():
        for eid in node.entity_ids:
            assignments[eid] = node.name

    # Non-concept entities → "Other" for now (will be handled by propagation)
    for e in entities:
        if e["id"] not in assignments:
            assignments[e["id"]] = "Other"

    concept_assigned = sum(1 for eid, cls in assignments.items()
                           if cls != "Other" and eid in signatures)
    _flush(f"  ✓ {concept_assigned} concept entities assigned to taxonomy classes")
    _flush(f"  ✓ {sum(1 for c in assignments.values() if c == 'Other')} entities as Other (pending propagation)")

    return assignments


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_recursive_induction(
    model: str = config.OLLAMA_MODEL,
    suffix: str = "recursive_v1",
    seed: int = 42,
    results_dir: str | None = None,
    data_dir: str | None = None,
    skip_name_signal: bool = False,
    skip_provenance: bool = False,
    skip_type_signal: bool = False,
    skip_llm_validation: bool = False,
    max_depth: int = MAX_DEPTH,
) -> dict:
    """Run the full recursive induction pipeline."""
    global _llm_call_count, MAX_DEPTH
    _llm_call_count = 0
    MAX_DEPTH = max_depth

    random.seed(seed)
    np.random.seed(seed)

    rdir = results_dir or config.RESULTS_DIR
    os.makedirs(rdir, exist_ok=True)

    _flush("=" * 70)
    _flush("CS584 Capstone — Zone 3: Recursive Divisive Ontology Induction")
    _flush(f"Model: {model} | Suffix: {suffix} | Seed: {seed}")
    _flush(f"Max depth: {max_depth} | Results: {rdir}")
    _flush("=" * 70)

    start = time.time()
    llm = get_llm(model)

    # --- Load entities ---
    _flush("\n[Load] Loading entities from cache...")
    entities = load_cached_entities(fmt="sv_loi", results_dir=rdir)
    if not entities:
        return {"error": "no entities"}
    entity_map = {e["id"]: e for e in entities}
    concept_entities = [e for e in entities if is_concept_entity(e)]
    _flush(f"  ✓ {len(entities)} entities ({len(concept_entities)} concepts)")

    # --- Load provenance ---
    chunk_map = {}
    if not skip_provenance:
        chunk_map = load_chunk_source_map(data_dir)

    # --- Stage A: Signatures ---
    signatures = build_entity_signatures(entities, chunk_map)

    # --- Stage A.5: Learn type normalization ---
    type_counts = Counter(
        e.get("entity_type", "Unknown") for e in concept_entities
        if e.get("entity_type", "Unknown") not in VALUE_ENTITY_TYPES
    )
    norm_map: dict[str, str | None] = {}
    if not skip_type_signal:
        meaningful_types = [t for t, c in type_counts.items() if c >= 3]
        norm_map = learn_type_normalization(meaningful_types, dict(type_counts), llm)

    # --- Stage B: Macro-classes ---
    macro_roots = discover_macro_classes(concept_entities, signatures, llm, seed)
    if not macro_roots:
        _flush("  ERROR: No macro-classes discovered")
        return {"error": "no macro-classes"}

    # --- Stage C: Recursive splitting ---
    taxonomy = induce_recursive_taxonomy(
        macro_roots, signatures, entity_map, norm_map, llm, seed,
    )

    # --- Stage D: Sibling merge ---
    taxonomy = merge_local_siblings(taxonomy, signatures, llm)

    # --- Stage E: Assignment ---
    assignments = assign_all_entities(taxonomy, entities, signatures)

    # Record propagation (reuse from sv_loi)
    _flush("\n[Propagation] Mapping records and values to taxonomy classes...")
    try:
        # Build features for propagation
        from zone3.sv_loi import build_structural_signatures
        features, entity_ids_feat, feature_names = build_structural_signatures(entities)
        assignments = propagate_to_records(assignments, entities, llm=llm)
    except Exception as exc:
        _flush(f"  ⚠ Record propagation failed: {exc}")

    # Inter-class associations
    _flush("\n[Associations] Deriving inter-class edges...")
    associations = derive_interclass_edges(assignments, entities)
    hierarchy = taxonomy.get_edges()

    # --- Write to Neo4j ---
    _flush("\n[Write] Writing ontology to Neo4j...")
    neo4j_stats = write_ontology(assignments, hierarchy, associations)

    # --- Summary ---
    elapsed = time.time() - start
    class_dist = Counter(v for v in assignments.values() if v != "Other")
    other_count = sum(1 for v in assignments.values() if v == "Other")

    _flush("\n" + "=" * 70)
    _flush(f"Recursive Induction complete in {elapsed:.1f}s")
    _flush(f"  Classes:        {taxonomy.class_count()}")
    _flush(f"  SUBCLASS_OF:    {len(hierarchy)}")
    _flush(f"  ASSOCIATED_WITH:{len(associations)}")
    _flush(f"  Max depth:      {taxonomy.max_depth()}")
    _flush(f"  LLM calls:      {_llm_call_count}")
    _flush(f"  Other:          {other_count} ({100*other_count/len(assignments):.1f}%)")
    _flush(f"  Distribution:")
    for cls, cnt in class_dist.most_common():
        _flush(f"    {cls}: {cnt}")

    # Save summary
    summary = {
        "mode": "recursive_induction",
        "model": model,
        "suffix": suffix,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
        "entity_count": len(entities),
        "concept_count": len(concept_entities),
        "classes_final": sorted(taxonomy.nodes.keys()),
        "class_count": taxonomy.class_count(),
        "max_depth": taxonomy.max_depth(),
        "class_distribution": dict(class_dist.most_common()),
        "other_count": other_count,
        "other_fraction": round(other_count / len(assignments), 4),
        "hierarchy": [{"child": c, "parent": p} for c, p in hierarchy],
        "associations_count": len(associations),
        "llm_calls": _llm_call_count,
        "neo4j_stats": neo4j_stats,
        "taxonomy_detail": {
            name: {
                "depth": node.depth,
                "size": len(node.entity_ids),
                "parent": node.parent,
                "children": node.children,
                "split_method": node.split_method,
                "metrics": node.metrics,
            }
            for name, node in taxonomy.nodes.items()
        },
        "normalization_map": {k: v for k, v in norm_map.items() if v is not None},
        "ablation": {
            "skip_name_signal": skip_name_signal,
            "skip_provenance": skip_provenance,
            "skip_type_signal": skip_type_signal,
            "skip_llm_validation": skip_llm_validation,
            "max_depth": max_depth,
        },
    }

    summary_path = os.path.join(rdir, f"zone3_recursive_summary_{suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _flush(f"\n✓ Summary saved to {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zone 3: Recursive Divisive Ontology Induction")
    parser.add_argument("--model", default=config.OLLAMA_MODEL)
    parser.add_argument("--suffix", default="recursive_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    # Ablation flags
    parser.add_argument("--skip-name-signal", action="store_true")
    parser.add_argument("--skip-provenance", action="store_true")
    parser.add_argument("--skip-type-signal", action="store_true")
    parser.add_argument("--skip-llm-validation", action="store_true")

    args = parser.parse_args()
    run_recursive_induction(
        model=args.model,
        suffix=args.suffix,
        seed=args.seed,
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        max_depth=args.max_depth,
        skip_name_signal=args.skip_name_signal,
        skip_provenance=args.skip_provenance,
        skip_type_signal=args.skip_type_signal,
        skip_llm_validation=args.skip_llm_validation,
    )
