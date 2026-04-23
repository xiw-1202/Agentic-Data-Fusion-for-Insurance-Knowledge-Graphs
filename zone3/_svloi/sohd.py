"""Structural Ontological Heterogeneity Detection — splits semantically impure classes."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from langchain_ollama import ChatOllama

from zone3._svloi.constants import FORBIDDEN_CLASS_NAMES
from zone3._svloi.utils import _invoke_llm, _parse_json_safely, _sanitize_label
from zone3.graph_cache import is_concept_entity

# ---------------------------------------------------------------------------
# SOHD: Structural Ontological Heterogeneity Detection
# ---------------------------------------------------------------------------

# Generic relations excluded from profile analysis (too common to be discriminative)
_SOHD_GENERIC_RELS = frozenset({
    "HAS_VALUE", "HAS_NAME", "HAS_DATE", "HAS_TYPE", "HAS_ID",
    "IS_A", "~IS_A", "~HAS_VALUE", "~HAS_NAME", "~HAS_DATE",
})


def _build_class_relation_profiles(
    class_members: list[dict],
) -> tuple[np.ndarray, list[str]]:
    """Build BINARY relation participation matrix for a set of entities.

    Each cell is 1 if the entity participates in that relation type, 0 otherwise.
    Binary profiles capture schema patterns (which relations an entity has)
    rather than degree patterns (how many times), which is the right signal
    for ontological subclass detection.

    Returns:
        mat: (n_entities, n_relation_types) binary matrix
        rel_list: ordered list of relation type names (column labels)
    """
    # Collect all non-generic relation types across these entities
    rel_types: set[str] = set()
    for e in class_members:
        for rt in e.get("out_rel_counts", {}):
            if rt not in _SOHD_GENERIC_RELS:
                rel_types.add(f"{rt}_OUT")
        for rt in e.get("in_rel_counts", {}):
            if rt not in _SOHD_GENERIC_RELS:
                rel_types.add(f"{rt}_IN")
    rel_list = sorted(rel_types)
    if not rel_list:
        return np.zeros((len(class_members), 0)), rel_list

    feat_idx = {name: i for i, name in enumerate(rel_list)}
    mat = np.zeros((len(class_members), len(rel_list)))
    for i, e in enumerate(class_members):
        for rt in e.get("out_rel_counts", {}):
            key = f"{rt}_OUT"
            if key in feat_idx:
                mat[i, feat_idx[key]] = 1.0  # binary: participates or not
        for rt in e.get("in_rel_counts", {}):
            key = f"{rt}_IN"
            if key in feat_idx:
                mat[i, feat_idx[key]] = 1.0  # binary: participates or not
    return mat, rel_list


def _cosine_similarity_matrix(mat: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for rows of mat."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    normed = mat / norms
    return normed @ normed.T


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two nonneg vectors (L1-normalized internally)."""
    p = np.asarray(p, dtype=float).clip(min=0)
    q = np.asarray(q, dtype=float).clip(min=0)
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum < 1e-12 or q_sum < 1e-12:
        return 0.0
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    # KL(p||m) and KL(q||m) with 0*log(0)=0
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log2(p / np.where(m > 0, m, 1)), 0)
        kl_qm = np.where(q > 0, q * np.log2(q / np.where(m > 0, m, 1)), 0)
    return float(0.5 * kl_pm.sum() + 0.5 * kl_qm.sum())


def _top_distinguishing_relations(
    subcluster_profile: np.ndarray,
    complement_profile: np.ndarray,
    rel_names: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Find relations where subcluster differs most from complement.

    Returns list of (rel_name, enrichment_ratio) sorted by ratio descending.
    """
    results = []
    for i, rn in enumerate(rel_names):
        sub_val = subcluster_profile[i]
        comp_val = complement_profile[i]
        # Enrichment: how much more prevalent in subcluster vs complement
        ratio = (sub_val + 1e-6) / (comp_val + 1e-6)
        if ratio > 1.5 or ratio < 0.67:  # at least 50% enriched or depleted
            results.append((rn, ratio))
    results.sort(key=lambda x: -abs(x[1] - 1.0))
    return results[:top_k]


def _name_subclass_llm(
    parent_class: str,
    distinguishing_rels: list[tuple[str, float]],
    example_entities: list[str],
    llm: ChatOllama,
) -> str:
    """Ask LLM to name a subclass based on its distinguishing characteristics."""
    rel_desc = "\n".join(
        f"  - {rn}: {ratio:.1f}x enriched" if ratio > 1 else f"  - {rn}: {1/ratio:.1f}x depleted"
        for rn, ratio in distinguishing_rels[:5]
    )
    examples = ", ".join(example_entities[:8])
    prompt = f"""A subgroup of the ontology class "{parent_class}" has been detected with distinct structural characteristics.

Distinguishing relations (compared to other {parent_class} entities):
{rel_desc}

Example entities in this subgroup: {examples}

What is a good SUBCLASS NAME for this subgroup? Requirements:
- Must be a genuine ontological subtype of {parent_class} (IS-A relationship)
- Use PascalCase, 1-3 words (e.g., "BuildingCoverage", "FloodDamage", "CommercialProperty")
- Do NOT use generic attribute names (like "HighValue", "Recent", "Large")
- If this subgroup does not represent a real ontological subtype, respond with SKIP

Respond with ONLY the subclass name (or SKIP):"""

    raw = _invoke_llm(llm, prompt).strip()
    # Extract first word-like token
    m = re.match(r'^([A-Za-z][A-Za-z0-9]{2,30})$', raw.split('\n')[0].strip())
    if m and m.group(1).upper() != "SKIP":
        return _sanitize_label(m.group(1))
    return ""


def _auto_name_subclass(
    parent_class: str,
    members: list[dict],
    distinguishing_rels: list[tuple[str, float]],
) -> str:
    """Generate a subclass name without LLM from entity_type or top relation."""
    # Try dominant entity_type
    type_counts = Counter(e.get("entity_type", "Unknown") for e in members)
    top_type, top_count = type_counts.most_common(1)[0]
    if top_count >= len(members) * 0.5 and top_type not in ("Unknown", "Text", "Numeric"):
        name = _sanitize_label(top_type)
        if name.lower() != parent_class.lower():
            return name
    # Fall back to top distinguishing relation
    if distinguishing_rels:
        rel_name = distinguishing_rels[0][0]
        # e.g. "COVERS_OUT" -> "Covers"
        clean = rel_name.replace("_OUT", "").replace("_IN", "").replace("~", "Incoming")
        clean = clean.replace("_", " ").title().replace(" ", "")
        name = f"{parent_class}{clean}"
        return _sanitize_label(name)
    return ""


def detect_and_split_heterogeneous_classes(
    assignments: dict[str, str],
    entities: list[dict],
    llm: ChatOllama | None = None,
    min_class_size: int = 20,
    min_subclass_size: int = 5,
    min_subclass_fraction: float = 0.15,
    silhouette_threshold: float = 0.25,
    js_threshold: float = 0.10,
    max_subclusters: int = 5,
    max_depth: int = 2,
    seed: int = 42,
) -> tuple[dict[str, str], list[tuple[str, str]], dict]:
    """Detect structurally heterogeneous classes and split into subclasses.

    SOHD (Structural Ontological Heterogeneity Detection):
    For each class with sufficient members, test whether it contains
    structurally distinct subgroups by clustering on relation profiles.
    Subgroups that are statistically distinct (JS-divergence + silhouette)
    are promoted to subclasses with IS-A edges.

    Only operates on CONCEPT entities (not records/values) to avoid
    creating structural-lane subclasses.

    Args:
        assignments: entity_id -> class_name mapping (MUTATED with new subclasses)
        entities: full entity list with relation data
        llm: optional LLM for subclass naming (falls back to auto-naming)
        min_class_size: minimum concept members to attempt splitting
        min_subclass_size: absolute minimum members for a valid subclass
        min_subclass_fraction: minimum fraction of parent class for a valid subclass
        silhouette_threshold: minimum silhouette score to accept a split
        js_threshold: minimum JS-divergence to consider a subcluster distinct
        max_subclusters: maximum k to try in clustering
        max_depth: maximum recursion depth for nested splits
        seed: random seed for reproducibility

    Returns:
        (updated_assignments, new_isa_edges, sohd_stats)
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_distances

    print("\n[SOHD] Structural Ontological Heterogeneity Detection", flush=True)

    entity_map = {e["id"]: e for e in entities}
    new_isa_edges: list[tuple[str, str]] = []
    stats: dict[str, Any] = {
        "classes_tested": 0,
        "classes_split": 0,
        "new_subclasses": 0,
        "new_isa_edges": 0,
        "heterogeneity_scores": {},
    }
    existing_class_names = set(assignments.values()) | set(FORBIDDEN_CLASS_NAMES)

    def _split_class(
        class_name: str,
        member_eids: list[str],
        depth: int,
    ) -> None:
        """Recursively test and split a single class."""
        if depth > max_depth:
            return

        # Filter to concept entities only
        concept_members = [
            entity_map[eid] for eid in member_eids
            if eid in entity_map and is_concept_entity(entity_map[eid])
        ]
        if len(concept_members) < min_class_size:
            return

        stats["classes_tested"] += 1
        n = len(concept_members)

        # Build raw relation profile matrix (NOT L2-normalized)
        raw_profiles, rel_names = _build_class_relation_profiles(concept_members)
        if raw_profiles.shape[1] < 2:
            print(f"  {class_name}: <2 non-generic relation types, skip", flush=True)
            return

        # Filter out entities with zero relation profiles (no non-generic relations)
        nonzero_mask = raw_profiles.sum(axis=1) > 0
        n_zero = int((~nonzero_mask).sum())
        if n_zero > 0:
            print(f"  {class_name}: {n_zero}/{n} entities have empty relation profiles, excluding", flush=True)
            raw_profiles = raw_profiles[nonzero_mask]
            concept_members = [concept_members[i] for i in range(n) if nonzero_mask[i]]
            n = len(concept_members)
            if n < min_class_size:
                return

        # Pre-split heterogeneity check: average pairwise cosine distance
        cos_sim = _cosine_similarity_matrix(raw_profiles)
        # Exclude diagonal (self-similarity = 1.0) from average
        np.fill_diagonal(cos_sim, 0)
        avg_pairwise = 1.0 - cos_sim.sum() / max(n * (n - 1), 1)
        if avg_pairwise < 0.05:
            print(f"  {class_name} (n={n}): homogeneous (avg cosine dist={avg_pairwise:.3f}), skip", flush=True)
            return

        # Compute cosine distance matrix for clustering
        cos_dist = cosine_distances(raw_profiles)
        np.fill_diagonal(cos_dist, 0)  # clean numerical noise

        # Try agglomerative clustering with cosine distance, k=2..k_max
        k_max = min(max_subclusters, n // max(min_subclass_size, 1))
        if k_max < 2:
            return

        best_k, best_sil, best_labels = 2, -1.0, None
        for k in range(2, k_max + 1):
            try:
                clust = AgglomerativeClustering(
                    n_clusters=k,
                    metric="precomputed",
                    linkage="average",
                )
                labels = clust.fit_predict(cos_dist)
                # Check no empty / singleton clusters
                cluster_sizes = Counter(labels)
                if min(cluster_sizes.values()) < 2:
                    continue
                sil = silhouette_score(cos_dist, labels, metric="precomputed",
                                       random_state=seed)
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_labels = labels
            except Exception as exc:
                print(f"    clustering k={k} failed: {exc}", flush=True)
                continue

        if best_labels is None or best_sil < silhouette_threshold:
            print(f"  {class_name} (n={n}): best silhouette={best_sil:.3f} < {silhouette_threshold}, skip", flush=True)
            stats["heterogeneity_scores"][class_name] = {
                "silhouette": round(best_sil, 3), "split": False, "n": n,
            }
            return

        print(f"  {class_name} (n={n}): heterogeneous! k={best_k}, silhouette={best_sil:.3f}", flush=True)

        # Evaluate each subcluster
        cluster_ids = defaultdict(list)
        for idx, label in enumerate(best_labels):
            cluster_ids[label].append(idx)

        # Compute parent class defining relations (binary: does any member use this rel?)
        parent_active_rels = set()
        for i in range(n):
            for j, rn in enumerate(rel_names):
                if raw_profiles[i, j] > 0:
                    parent_active_rels.add(rn)

        min_sub_size = max(min_subclass_size, int(n * min_subclass_fraction))
        accepted_subclusters = []

        for label, indices in sorted(cluster_ids.items()):
            sub_size = len(indices)
            if sub_size < min_sub_size:
                continue

            sub_mean = raw_profiles[indices].mean(axis=0)
            # Complement = everything NOT in this subcluster
            complement_idx = [i for i in range(n) if i not in set(indices)]
            if not complement_idx:
                continue
            complement_mean = raw_profiles[complement_idx].mean(axis=0)

            js = _js_divergence(sub_mean, complement_mean)
            if js < js_threshold:
                print(f"    cluster {label} (n={sub_size}): JS={js:.3f} < {js_threshold}, not distinct enough", flush=True)
                continue

            # IS-A validation: subgroup's active relations should overlap with
            # parent's (shared heritage) AND have distinguishing relations (specialization).
            sub_active_rels = set()
            for i in indices:
                for j, rn in enumerate(rel_names):
                    if raw_profiles[i, j] > 0:
                        sub_active_rels.add(rn)
            shared_rels = sub_active_rels & parent_active_rels
            exclusive_rels = sub_active_rels - parent_active_rels
            # Must share at least some relations with parent (inheritance)
            if parent_active_rels and len(shared_rels) < len(parent_active_rels) * 0.3:
                print(f"    cluster {label} (n={sub_size}): insufficient relation overlap "
                      f"({len(shared_rels)}/{len(parent_active_rels)}), not a valid subclass", flush=True)
                continue

            # Find distinguishing relations
            distinguishing = _top_distinguishing_relations(sub_mean, complement_mean, rel_names)
            sub_entities = [concept_members[i] for i in indices]
            sub_eids = [e["id"] for e in sub_entities]
            example_names = [e["id"] for e in sub_entities[:8]]

            # Name the subclass (two-stage: LLM can reject with SKIP)
            name = ""
            if llm is not None:
                name = _name_subclass_llm(class_name, distinguishing, example_names, llm)
                if not name:
                    # LLM said SKIP — this subgroup is not a real ontological subtype
                    print(f"    cluster {label} (n={sub_size}): LLM rejected as non-subtype, skip", flush=True)
                    continue
            if not name:
                name = _auto_name_subclass(class_name, sub_entities, distinguishing)
            if not name or name.lower() == class_name.lower():
                name = f"{class_name}Sub{label}"

            # Deduplicate name against existing classes
            orig_name = name
            dedup_counter = 2
            lower_existing = {n.lower() for n in existing_class_names}
            while name.lower() in lower_existing:
                name = f"{orig_name}{dedup_counter}"
                dedup_counter += 1

            accepted_subclusters.append({
                "label": label,
                "name": name,
                "size": sub_size,
                "js": js,
                "eids": sub_eids,
                "distinguishing": distinguishing,
            })
            print(f"    cluster {label} → {name} (n={sub_size}, JS={js:.3f})", flush=True)
            if distinguishing:
                top_rel = distinguishing[0]
                print(f"      top relation: {top_rel[0]} ({top_rel[1]:.1f}x)", flush=True)

        if not accepted_subclusters:
            stats["heterogeneity_scores"][class_name] = {
                "silhouette": round(best_sil, 3), "split": False, "n": n,
                "reason": "no subclusters passed JS threshold",
            }
            return

        # If ALL members end up in accepted subclusters, keep the largest
        # as the parent class (don't orphan the parent)
        total_assigned = sum(sc["size"] for sc in accepted_subclusters)
        if total_assigned == n and len(accepted_subclusters) > 1:
            # Keep the largest subcluster as the parent class itself
            accepted_subclusters.sort(key=lambda x: -x["size"])
            kept_as_parent = accepted_subclusters.pop(0)
            print(f"    keeping largest cluster ({kept_as_parent['name']}, n={kept_as_parent['size']}) as {class_name}", flush=True)

        # Apply splits
        stats["classes_split"] += 1
        for sc in accepted_subclusters:
            subclass_name = sc["name"]
            existing_class_names.add(subclass_name)
            # Update assignments
            for eid in sc["eids"]:
                assignments[eid] = subclass_name
            # Create IS-A edge
            new_isa_edges.append((subclass_name, class_name))
            stats["new_subclasses"] += 1
            stats["new_isa_edges"] += 1

        stats["heterogeneity_scores"][class_name] = {
            "silhouette": round(best_sil, 3),
            "k": best_k,
            "split": True,
            "subclusters": [
                {"name": sc["name"], "size": sc["size"], "js": round(sc["js"], 3)}
                for sc in accepted_subclusters
            ],
            "n": n,
        }

        # Recurse on new subclasses
        if depth + 1 <= max_depth:
            for sc in accepted_subclusters:
                _split_class(sc["name"], sc["eids"], depth + 1)

    # Identify classes eligible for splitting
    class_to_eids: dict[str, list[str]] = defaultdict(list)
    for eid, cls in assignments.items():
        if cls != "Other":
            class_to_eids[cls].append(eid)

    eligible = sorted(
        [(cls, eids) for cls, eids in class_to_eids.items()],
        key=lambda x: -len(x[1]),  # largest classes first
    )

    for cls, eids in eligible:
        _split_class(cls, eids, depth=0)

    print(f"\n  ✓ SOHD complete: {stats['classes_split']} classes split, "
          f"{stats['new_subclasses']} new subclasses, "
          f"{stats['new_isa_edges']} new IS-A edges", flush=True)

    return assignments, new_isa_edges, stats
