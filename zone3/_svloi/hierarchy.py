"""IS-A hierarchy derivation: class merging, inter-class edges, taxonomy building."""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import numpy as np
from langchain_ollama import ChatOllama

from zone3._svloi.constants import MIN_CLASS_SIZE, PROTECTED_CLASS_NAMES
from zone3._svloi.utils import (
    _invoke_llm,
    _parse_json_safely,
    _sanitize_label,
)
from zone3.graph_cache import is_concept_entity


# ---------------------------------------------------------------------------
# Phase 11-14: Post-processing — consolidate, merge small, derive hierarchy
# ---------------------------------------------------------------------------

def infer_class_relations(
    assignments: dict[str, str],
    entities: list[dict],
    llm: ChatOllama,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """5-way typed relation inference between classes (Changes C+D+E unified).

    Replaces separate consolidate_classes() + derive_hierarchy() with a single
    pass that infers typed relations: equivalent / parent / child / overlap / distinct.

    Only 'equivalent' triggers a merge. 'parent/child' become SUBCLASS_OF edges.
    'overlap' and 'distinct' keep both classes separate.

    Protected classes (standard ontology terms) can be merged INTO but never
    renamed away — prevents destroying exact matches with reference ontologies.

    Returns:
        (updated_assignments, hierarchy_edges)
    """
    print("\n[Phase 11] Class relation inference (5-way)...", flush=True)

    dist = Counter(v for v in assignments.values() if v != "Other")
    class_names = sorted(dist.keys())

    if len(class_names) < 2:
        print("  ✓ Too few classes for relation inference")
        return assignments, []

    # Build class descriptions with CONCEPT member examples only
    entity_map = {e["id"]: e for e in entities}
    class_descs: dict[str, str] = {}
    for cls in class_names:
        concept_members = [
            eid for eid, c in assignments.items()
            if c == cls and is_concept_entity(entity_map.get(eid, {"id": eid, "entity_type": "Unknown"}))
        ]
        # If no concept members, use first few record IDs as fallback context
        if not concept_members:
            all_members = [eid for eid, c in assignments.items() if c == cls]
            sample = all_members[:5]
        else:
            sample = concept_members[:8]
        # Include typical relations for context
        rel_types: set[str] = set()
        for eid in (concept_members or [eid for eid, c in assignments.items() if c == cls])[:20]:
            e = entity_map.get(eid, {})
            rel_types.update(e.get("out_rel_counts", {}).keys())
            rel_types.update(e.get("in_rel_counts", {}).keys())
        top_rels = sorted(rel_types)[:5]

        protected = "  [PROTECTED — standard term]" if cls.lower() in PROTECTED_CLASS_NAMES else ""
        class_descs[cls] = (
            f"  {cls} ({dist[cls]} members{protected}):\n"
            f"    Examples: {', '.join(sample)}\n"
            f"    Relations: {', '.join(top_rels) if top_rels else 'none'}"
        )

    # --- Pairwise relation inference ---
    # Only compare plausible pairs (blocking: skip obvious non-matches)
    from sentence_transformers import SentenceTransformer
    st_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embs = st_encoder.encode([c for c in class_names], normalize_embeddings=True)
    sim_matrix = np.dot(embs, embs.T)

    # Block: only compare pairs with cosine > 0.25 (skip clearly unrelated)
    BLOCK_THRESHOLD = 0.25
    pairs_to_compare = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            if sim_matrix[i, j] > BLOCK_THRESHOLD:
                pairs_to_compare.append((class_names[i], class_names[j]))

    print(f"  Comparing {len(pairs_to_compare)} class pairs (blocked from {len(class_names) * (len(class_names)-1) // 2})...", flush=True)

    # Batch pairs into groups for efficient LLM calls
    PAIRS_PER_BATCH = 5
    all_relations: list[dict] = []

    for batch_start in range(0, len(pairs_to_compare), PAIRS_PER_BATCH):
        batch = pairs_to_compare[batch_start:batch_start + PAIRS_PER_BATCH]

        pair_sections = []
        for ci, cj in batch:
            pair_sections.append(
                f"PAIR: {ci} vs {cj}\n"
                f"{class_descs[ci]}\n"
                f"{class_descs[cj]}"
            )

        prompt = f"""For each pair of ontology classes, determine their relationship.

Choose EXACTLY ONE for each pair:
- "equivalent": same concept, different names → MERGE (keep the more standard name)
- "parent_of": first class is a broader category containing the second → SUBCLASS_OF
- "child_of": first class is a subtype of the second → SUBCLASS_OF
- "overlapping": partial intersection, neither fully contains the other → keep both
- "distinct": unrelated or sibling concepts → keep both

IMPORTANT RULES:
- Classes marked [PROTECTED] are standard ontology terms. They can ABSORB other classes
  but must NEVER be renamed or merged INTO a non-standard name.
- "equivalent" should be rare — only for true synonyms (e.g., "Policy" and "Product")
- When unsure, choose "distinct" — it's safer to keep classes separate

{chr(10).join(pair_sections)}

Output JSON array, one entry per pair:
[{{"class_a": "...", "class_b": "...", "relation": "...", "merge_into": "...", "reason": "..."}}]
The "merge_into" field is only needed for "equivalent" — which name to keep."""

        raw = _invoke_llm(llm, prompt)
        parsed = _parse_json_safely(raw)

        if isinstance(parsed, list):
            all_relations.extend(parsed)
        elif isinstance(parsed, dict):
            all_relations.append(parsed)

        if (batch_start // PAIRS_PER_BATCH + 1) % 5 == 0:
            print(f"    Batch {batch_start // PAIRS_PER_BATCH + 1}/{(len(pairs_to_compare) + PAIRS_PER_BATCH - 1) // PAIRS_PER_BATCH}", flush=True)

    # --- Process relations ---
    updated = dict(assignments)
    hierarchy: list[tuple[str, str]] = []
    merges_applied: list[str] = []
    merge_count = 0

    for rel_entry in all_relations:
        if not isinstance(rel_entry, dict):
            continue
        ca = rel_entry.get("class_a", "")
        cb = rel_entry.get("class_b", "")
        relation = rel_entry.get("relation", "distinct").lower()
        merge_into = rel_entry.get("merge_into", "")
        reason = rel_entry.get("reason", "")

        if relation == "equivalent":
            # Determine which name to keep (with validation)
            valid_names = {ca, cb}
            if merge_into:
                sanitized_merge = _sanitize_label(merge_into)
                # Validate: merge_into must match one of the two classes
                if sanitized_merge.lower() in {ca.lower(), cb.lower()}:
                    keep = ca if sanitized_merge.lower() == ca.lower() else cb
                    absorb = cb if keep == ca else ca
                else:
                    # LLM proposed a third name — fall through to protected heuristic
                    merge_into = ""

            if not merge_into:
                # Prefer protected names (standard ontology terms)
                if ca.lower() in PROTECTED_CLASS_NAMES and cb.lower() not in PROTECTED_CLASS_NAMES:
                    keep, absorb = ca, cb
                elif cb.lower() in PROTECTED_CLASS_NAMES and ca.lower() not in PROTECTED_CLASS_NAMES:
                    keep, absorb = cb, ca
                elif ca.lower() in PROTECTED_CLASS_NAMES and cb.lower() in PROTECTED_CLASS_NAMES:
                    # Both protected — skip merge, treat as distinct
                    print(f"  ✗ Both {ca} and {cb} are protected — keeping separate", flush=True)
                    continue
                else:
                    keep, absorb = ca, cb  # neither protected, keep first

            # NEVER rename a protected class into a non-protected one
            if absorb.lower() in PROTECTED_CLASS_NAMES and keep.lower() not in PROTECTED_CLASS_NAMES:
                print(f"  ✗ Blocked: {absorb} is protected, cannot merge into {keep}", flush=True)
                continue

            # Apply merge
            count = 0
            for eid in list(updated.keys()):
                if updated[eid] == absorb:
                    updated[eid] = keep
                    count += 1
            if count > 0:
                print(f"  ✓ EQUIVALENT: {absorb} → {keep} ({count} entities) — {reason}", flush=True)
                merge_count += count
                merges_applied.append(f"{absorb}→{keep}")

        elif relation == "parent_of":
            # ca is parent of cb
            edge = (cb, ca)
            if edge not in hierarchy:
                hierarchy.append(edge)
                print(f"  → PARENT: {ca} ⊃ {cb} — {reason}", flush=True)

        elif relation == "child_of":
            # ca is child of cb
            edge = (ca, cb)
            if edge not in hierarchy:
                hierarchy.append(edge)
                print(f"  → CHILD: {ca} ⊂ {cb} — {reason}", flush=True)

        elif relation == "overlapping":
            print(f"  ~ OVERLAP: {ca} ∩ {cb} — {reason}", flush=True)

        # "distinct" → no action

    # Also mark pure data-artifact classes as Other
    for cls in list(set(updated.values())):
        if cls == "Other":
            continue
        members = [eid for eid, c in updated.items() if c == cls]
        concept_members = [
            eid for eid in members
            if is_concept_entity(entity_map.get(eid, {"id": eid, "entity_type": "Unknown"}))
        ]
        if not concept_members and len(members) > 0:
            # Class has zero concept members — all records/values
            # Only mark as Other if it's not a protected name
            if cls.lower() not in PROTECTED_CLASS_NAMES:
                for eid in members:
                    updated[eid] = "Other"
                print(f"  ✓ {cls} → Other ({len(members)} entities, no concept members)", flush=True)
                merge_count += len(members)

    new_dist = Counter(v for v in updated.values() if v != "Other")
    print(f"\n  ✓ Relation inference complete:", flush=True)
    print(f"    Merges: {len(merges_applied)} ({merge_count} entities moved)", flush=True)
    print(f"    Hierarchy edges: {len(hierarchy)}", flush=True)
    print(f"    Classes remaining: {len(new_dist)}", flush=True)
    for cls, cnt in new_dist.most_common():
        protected = " [P]" if cls.lower() in PROTECTED_CLASS_NAMES else ""
        print(f"      {cls}{protected}: {cnt}", flush=True)

    return updated, hierarchy


def merge_small_classes(
    assignments: dict[str, str],
    features: np.ndarray,
    entity_ids: list[str],
    min_size: int = MIN_CLASS_SIZE,
) -> dict[str, str]:
    """Merge classes with fewer than min_size members into nearest.

    min_size is a fixed floor (default: 10). This is intentionally NOT
    percentage-based: a class with 999 members is valid at any dataset
    size. The LLM validation pass (merge_leaf_classes) handles the
    semantic question; this pass only cleans up degenerate noise clusters.
    """
    print(f"\n  Merging small classes (min_size={min_size})...")

    eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

    # Count class sizes
    class_counts = Counter(assignments.values())
    small_classes = [c for c, n in class_counts.items()
                     if n < min_size and c != "Other" and c.lower() not in PROTECTED_CLASS_NAMES]

    if not small_classes:
        print("  ✓ No small classes to merge")
        return assignments

    # Compute centroids for non-small classes
    centroids: dict[str, np.ndarray] = {}
    for cls, cnt in class_counts.items():
        if cls in small_classes or cls == "Other":
            continue
        indices = [eid_to_idx[eid] for eid, c in assignments.items() if c == cls and eid in eid_to_idx]
        if not indices:
            continue
        centroid = features[indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[cls] = centroid

    updated = dict(assignments)
    merge_map: dict[str, str] = {}

    for small_cls in small_classes:
        # Find nearest large class by centroid similarity
        indices = [eid_to_idx[eid] for eid, c in assignments.items() if c == small_cls and eid in eid_to_idx]
        if not indices:
            continue
        small_centroid = features[indices].mean(axis=0)
        norm = np.linalg.norm(small_centroid)
        if norm > 0:
            small_centroid = small_centroid / norm

        best_cls = None
        best_sim = -1.0
        for cls, centroid in centroids.items():
            sim = float(small_centroid @ centroid)
            if sim > best_sim:
                best_sim = sim
                best_cls = cls

        if best_cls:
            merge_map[small_cls] = best_cls
            for eid in list(updated.keys()):
                if updated[eid] == small_cls:
                    updated[eid] = best_cls

    if merge_map:
        print(f"  ✓ Merged {len(merge_map)} small classes:")
        for s, t in merge_map.items():
            print(f"    {s} ({class_counts[s]} members) → {t}")
    else:
        print("  ✓ No merges needed")

    return updated


# ---------------------------------------------------------------------------
# Phase 13: LLM-guided class validation
# ---------------------------------------------------------------------------

def merge_leaf_classes(
    assignments: dict[str, str],
    entities: list[dict],
    llm: Optional[ChatOllama] = None,
) -> dict[str, str]:
    """LLM-guided class validation with structural evidence.

    After entity typing, some induced classes are genuine ontology classes
    (Risk, Damage, Person) while others are property values masquerading
    as classes (Deductible, Limit, Notification). This function presents
    the LLM with structural evidence for each small class and asks it to
    decide: keep as a real class, or merge into a parent?

    The structural evidence (member count, leaf fraction, relational
    diversity, sample members, top parent) helps the LLM make informed
    decisions. This scales with data — more data means richer evidence,
    better decisions. No hardcoded thresholds that break with scale.

    Algorithm:
        1. Compute structural profile for each small class (<50 members)
        2. Present all small classes + evidence to LLM in one prompt
        3. LLM returns: keep or merge into a target class
        4. Apply merge decisions
        5. Heuristic fallback for any class LLM didn't decide:
           merge if leaf% > 70% AND distinct_rel_types < 4

    Domain-agnostic — uses graph structure + LLM reasoning, no hardcoded names.
    """
    print("\n[Phase 13] LLM-guided class validation...", flush=True)

    entity_map = {e["id"]: e for e in entities}
    class_counts = Counter(assignments.values())

    # Step 1: Compute structural profile for ALL non-Other classes.
    # Present every class to the LLM with its evidence — the LLM decides
    # which are real classes (KEEP) vs property values (MERGE).
    # No artificial large/small split — a class with 17 members (Risk)
    # is a valid merge target for a class with 3 members (Classification).
    all_classes: list[str] = []
    class_profiles: dict[str, dict] = {}

    for cls, count in class_counts.items():
        if cls == "Other":
            continue

        members = [eid for eid, c in assignments.items() if c == cls]
        leaf_count = 0
        all_rel_types: set[str] = set()
        total_degree = 0.0
        sample_members: list[str] = []
        # Prefer concept members for samples (semantic names, not POL-xxx)
        concept_members = [eid for eid in members
                           if not eid.startswith(("POL-", "CLM-", "PROP-", "REC-", "PER-"))]
        sample_pool = concept_members or members

        for eid in members:
            e = entity_map.get(eid)
            if not e:
                continue
            out_count = sum(e.get("out_rel_counts", {}).values())
            if out_count <= 1:
                leaf_count += 1
            total_degree += e.get("degree", 0)
            all_rel_types.update(e.get("out_rel_counts", {}).keys())
            all_rel_types.update(e.get("in_rel_counts", {}).keys())
            if len(sample_members) < 5 and eid in sample_pool:
                sample_members.append(eid)

        leaf_frac = leaf_count / len(members) if members else 0
        avg_degree = total_degree / len(members) if members else 0

        # Find which large classes point to this class's members
        member_set = set(members)
        parent_votes: Counter = Counter()
        for e in entities:
            src_cls = assignments.get(e["id"], "Other")
            if src_cls == "Other" or src_cls == cls or class_counts.get(src_cls, 0) < count:
                continue
            for rel in e.get("out_rels", []):
                if rel.get("target", "") in member_set:
                    parent_votes[src_cls] += 1

        class_profiles[cls] = {
            "count": count,
            "concept_count": len(concept_members),
            "leaf_frac": leaf_frac,
            "avg_degree": avg_degree,
            "rel_types": sorted(all_rel_types)[:8],
            "n_rel_types": len(all_rel_types),
            "sample_members": sample_members,
            "top_parent": parent_votes.most_common(1)[0] if parent_votes else None,
        }
        all_classes.append(cls)

    if not all_classes:
        print("  ✓ No classes to validate")
        return assignments

    # Identify classes backed by database records (distinct data schemas)
    classes_with_records: set[str] = set()
    for cls in all_classes:
        has_records = any(
            eid.startswith(("POL-", "CLM-", "REC-", "PER-", "PROP-"))
            for eid, c in assignments.items() if c == cls
        )
        if has_records:
            classes_with_records.add(cls)

    print(f"  {len(all_classes)} classes to validate "
          f"({len(classes_with_records)} record-backed)", flush=True)

    # Step 2: Ask LLM to validate — present ALL classes with evidence,
    # let LLM decide which are real vs which should merge.
    merge_map: dict[str, str] = {}

    if llm:
        profile_lines = []
        for cls in sorted(all_classes, key=lambda c: -class_counts[c]):
            p = class_profiles[cls]
            parent_info = ""
            if p["top_parent"]:
                parent_info = f", most referenced by: {p['top_parent'][0]}"
            members_str = ', '.join(p['sample_members'][:4]) if p['sample_members'] else '(database records only)'
            record_note = " [RECORDS]" if cls in classes_with_records else ""
            profile_lines.append(
                f"  {cls}{record_note} ({p['count']} total, {p['concept_count']} concepts): "
                f"examples=[{members_str}], "
                f"{p['n_rel_types']} relation types"
                f"{parent_info}"
            )

        prompt = (
            "Ontology class consolidation task.\n\n"
            "Below are ALL induced ontology classes with their evidence.\n"
            "Decide which classes are genuine ontological concepts and which "
            "are really properties/attributes of other classes.\n\n"
            "Classes:\n" + "\n".join(profile_lines) + "\n\n"
            "Rules:\n"
            "- KEEP if the members are real-world THINGS (people, places, risks, "
            "products, organizations, processes, damages, documents). These exist "
            "independently — 'Base Flood' IS a risk, 'FEMA' IS an organization.\n"
            "- MERGE only if the members are MEASUREMENTS or ATTRIBUTES that "
            "describe another class — amounts, limits, codes, dates, thresholds. "
            "Example: 'Separate Deductible' is an ATTRIBUTE of Coverage.\n"
            "- Class SIZE does not matter. A class with 5 members can be real "
            "(e.g., Product with 'Flood Insurance') while a class with 20 "
            "members can be an attribute (e.g., Requirement).\n"
            "- Classes marked [RECORDS] contain database records from distinct "
            "data sources. NEVER merge two [RECORDS] classes together — they "
            "represent different data schemas (e.g., claims vs policies).\n"
            "- When in doubt, KEEP. Over-merging destroys ontology structure.\n\n"
            "Return JSON: {\"ClassName\": \"keep\" or \"merge:TargetClass\"}\n"
            "Include ALL classes listed above."
        )

        try:
            response = llm.invoke(prompt)
            decisions = _parse_json_safely(response.content)
            if isinstance(decisions, dict):
                for cls_key, decision in decisions.items():
                    matched_cls = None
                    for c in all_classes:
                        if c.lower() == cls_key.lower():
                            matched_cls = c
                            break
                    if not matched_cls:
                        continue

                    decision = str(decision).strip()
                    if decision.lower() == "keep":
                        print(f"    LLM: {matched_cls} → KEEP", flush=True)
                    elif decision.lower().startswith("merge:"):
                        target = decision.split(":", 1)[1].strip()
                        matched_target = None
                        for c in all_classes:
                            if c.lower() == target.lower():
                                matched_target = c
                                break
                        if matched_target and matched_target != matched_cls:
                            # Guard: never merge record-backed classes with >100 members
                            member_count = class_counts.get(matched_cls, 0)
                            if (member_count > 100
                                    and matched_cls in classes_with_records):
                                print(f"    LLM: {matched_cls} → merge into "
                                      f"{matched_target} [BLOCKED: {member_count} "
                                      f"members, record-backed]", flush=True)
                            else:
                                merge_map[matched_cls] = matched_target
                                print(f"    LLM: {matched_cls} → merge into "
                                      f"{matched_target}", flush=True)
                        else:
                            print(f"    LLM: {matched_cls} → merge:{target} "
                                  f"(target not found, keeping)", flush=True)
            else:
                print("    LLM returned non-dict, using heuristic", flush=True)
        except Exception as exc:
            print(f"    LLM error: {exc}, using heuristic", flush=True)

    # Step 3: Heuristic fallback for undecided classes
    # Uses structural criterion: leaf% > 70% AND distinct_rels < 4
    # Respects PROTECTED_CLASS_NAMES (never merge protected classes via heuristic)
    for cls in all_classes:
        if cls in merge_map:
            continue
        if cls.lower() in PROTECTED_CLASS_NAMES:
            continue
        p = class_profiles[cls]
        if (p["count"] < 50
                and p["leaf_frac"] > 0.70
                and p["n_rel_types"] < 4
                and p["top_parent"]):
            target = p["top_parent"][0]
            merge_map[cls] = target
            print(f"    Heuristic: {cls} → {target} "
                  f"({p['leaf_frac']:.0%} leaf, {p['n_rel_types']} rels)",
                  flush=True)

    # Step 4: Structural veto — prevent merging classes that represent
    # distinct data schemas (both have record entities from different sources).
    # SV-LOI principle: when semantic (LLM) and structural signals disagree,
    # keep separate. Two record-backed classes = two distinct data schemas.
    vetoed: list[str] = []
    for src_cls, tgt_cls in list(merge_map.items()):
        # Veto: never merge two record-backed classes (conflates data schemas)
        if src_cls in classes_with_records and tgt_cls in classes_with_records:
            vetoed.append(src_cls)
            del merge_map[src_cls]
            print(f"    VETO: {src_cls} → {tgt_cls} "
                  f"(both have record entities — distinct data schemas)",
                  flush=True)

    # Step 5: Apply merges
    updated = dict(assignments)
    for src_cls, tgt_cls in merge_map.items():
        for eid in list(updated.keys()):
            if updated[eid] == src_cls:
                updated[eid] = tgt_cls

    if merge_map:
        print(f"  ✓ Merged {len(merge_map)} classes:", flush=True)
        for src, tgt in merge_map.items():
            print(f"    {src} ({class_counts[src]}) → {tgt}", flush=True)
    if vetoed:
        print(f"  ✓ Vetoed {len(vetoed)} merges: {vetoed}", flush=True)
    if not merge_map and not vetoed:
        print("  ✓ No merges needed")

    return updated


def derive_interclass_edges(
    assignments: dict[str, str],
    entities: list[dict],
    min_edge_count: int = 3,
    min_class_frac: float = 0.05,
) -> list[tuple[str, str]]:
    """Derive inter-class edges from actual entity-level connections (data-driven).

    Instead of asking the LLM to guess SUBCLASS_OF relationships (which produces
    wrong IS-A edges), this function looks at actual entity-to-entity connections
    in the KG and aggregates them into class-to-class edges.

    If entities of class A frequently connect to entities of class B via
    relations, that creates an (A, B) inter-class edge. This matches how
    reference ontologies like Riskine define inter-class relationships
    (via $ref links = association/composition, NOT is-a).

    Threshold: max(min_edge_count, 5% of the SMALLER class in the pair).
    This scales correctly because it's relative to the classes involved,
    not total entities. A small but tightly connected class (100 members,
    30 connections to another class = 30%) will always produce an edge.

    Args:
        assignments: entity → class mapping
        entities: all entities with relation data
        min_edge_count: absolute floor (default: 3)
        min_class_frac: fraction of smaller class needed for edge (default: 5%)

    Returns:
        List of (source_class, target_class) edges (stored as SUBCLASS_OF in Neo4j
        for compatibility with evaluation metrics, but semantically these are
        inter-class associations).
    """
    # Count class sizes for adaptive threshold
    class_sizes = Counter(v for v in assignments.values() if v != "Other")
    print(f"\n  Deriving inter-class edges (min_floor={min_edge_count}, "
          f"min_class_frac={min_class_frac:.0%})...", flush=True)

    # Count entity-level connections between classes
    class_connections: Counter = Counter()  # (src_class, tgt_class) → count

    for e in entities:
        eid = e["id"]
        src_cls = assignments.get(eid, "Other")
        if src_cls == "Other":
            continue

        for rel in e.get("out_rels", []):
            tgt_eid = rel.get("target", "")
            tgt_cls = assignments.get(tgt_eid, "Other")
            if tgt_cls == "Other" or tgt_cls == src_cls:
                continue
            class_connections[(src_cls, tgt_cls)] += 1

    # Filter: only keep edges with enough evidence
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for (src, tgt), count in class_connections.most_common():
        # Adaptive threshold: max(floor, 5% of the smaller class)
        smaller_size = min(class_sizes.get(src, 0), class_sizes.get(tgt, 0))
        pair_min = max(min_edge_count, int(smaller_size * min_class_frac))
        if count < pair_min:
            continue
        # Avoid circular edges: if (A,B) and (B,A) both exist, keep the stronger one
        if (tgt, src) in seen:
            continue
        edge = (src, tgt)
        if edge not in seen:
            edges.append(edge)
            seen.add(edge)

    print(f"  ✓ {len(edges)} inter-class edges (floor={min_edge_count}, "
          f"{min_class_frac:.0%} of smaller class)", flush=True)
    for src, tgt in edges[:20]:
        count = class_connections[(src, tgt)]
        print(f"    {src} → {tgt} ({count} connections)", flush=True)
    if len(edges) > 20:
        print(f"    ... and {len(edges) - 20} more", flush=True)

    return edges


def derive_hierarchy(
    assignments: dict[str, str],
    llm: ChatOllama,
) -> list[tuple[str, str]]:
    """Legacy LLM-based hierarchy (kept for ablation comparison)."""
    print("\n[Phase 11-legacy] LLM hierarchy derivation...", flush=True)

    class_counts = Counter(v for v in assignments.values() if v != "Other")
    class_names = sorted(class_counts.keys())

    if len(class_names) < 2:
        return []

    classes_desc = "\n".join(f"  {c} ({class_counts[c]} members)" for c in class_names)

    prompt = f"""Given these ontology classes, propose SUBCLASS_OF relationships to form a hierarchy.

CLASSES:
{classes_desc}

Rules:
- Only propose relationships where one class is truly a subtype of another
- Use format: ChildClass -> ParentClass (meaning Child SUBCLASS_OF Parent)
- Prefer shallow hierarchies (max depth 3)
- Not every class needs a parent — top-level classes are fine
- Only output valid relationships, one per line
"""
    raw = _invoke_llm(llm, prompt)

    hierarchy: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    line_re = re.compile(r'^[- ]*(\w+)\s*->\s*(\w+)\s*$', re.MULTILINE)
    for m in line_re.finditer(raw):
        child = _sanitize_label(m.group(1))
        parent = _sanitize_label(m.group(2))
        if child != parent and child in class_names and parent in class_names:
            edge = (child, parent)
            if edge not in seen:
                hierarchy.append(edge)
                seen.add(edge)

    print(f"  ✓ {len(hierarchy)} SUBCLASS_OF edges")
    return hierarchy


def derive_taxonomy(
    assignments: dict[str, str],
    entities: list[dict],
    llm: ChatOllama,
) -> list[tuple[str, str]]:
    """Derive IS-A taxonomy using defining relations + LLM validation.

    Unlike derive_hierarchy (pure LLM guessing) or derive_interclass_edges
    (association edges), this produces genuine IS-A relationships validated
    by both structural evidence and LLM judgment.

    Algorithm:
    1. Compute defining relations per class (frequent + discriminative)
    2. Find candidate IS-A pairs via relation-set subsumption
    3. LLM validates candidates (reject-only, not invent)
    4. Enforce DAG (remove cycles, max depth 3)
    """
    print("\n[Taxonomy] Deriving IS-A hierarchy from defining relations...", flush=True)

    class_counts = Counter(v for v in assignments.values() if v != "Other")
    class_names = sorted(class_counts.keys())
    if len(class_names) < 2:
        return []

    # Step 1: Compute defining relations per class
    # A "defining" relation is one that >= 30% of class members participate in
    class_rel_counts: dict[str, Counter] = {c: Counter() for c in class_names}
    for e in entities:
        eid = e["id"]
        cls = assignments.get(eid, "Other")
        if cls == "Other" or cls not in class_rel_counts:
            continue
        for rel in e.get("out_rels", []):
            class_rel_counts[cls][rel["rel"]] += 1
        for rel in e.get("in_rels", []):
            class_rel_counts[cls][f"~{rel['rel']}"] += 1  # ~REL = incoming

    # Filter to defining relations (>= 30% of class members participate)
    _GENERIC_RELS = {"HAS_VALUE", "HAS_NAME", "HAS_DATE", "HAS_TYPE", "HAS_ID",
                     "IS_A", "~IS_A", "~HAS_VALUE", "~HAS_NAME", "~HAS_DATE"}
    defining_rels: dict[str, set[str]] = {}
    for cls in class_names:
        n = class_counts[cls]
        threshold = max(2, int(n * 0.3))
        defining_rels[cls] = {
            rel for rel, count in class_rel_counts[cls].items()
            if count >= threshold and rel not in _GENERIC_RELS
        }

    # Step 2: Find candidate IS-A via relation-set subsumption
    # If A's defining rels are a strict subset of B's (>= 80% inclusion), A may be subclass of B
    candidates: list[tuple[str, str, float]] = []
    for child in class_names:
        child_rels = defining_rels.get(child, set())
        if len(child_rels) < 2:
            continue
        for parent in class_names:
            if child == parent:
                continue
            parent_rels = defining_rels.get(parent, set())
            if len(parent_rels) < 2:
                continue
            # Child's rels should be subset of parent's
            if len(child_rels) >= len(parent_rels):
                continue  # Can't be a subclass if has more or equal rels
            overlap = len(child_rels & parent_rels)
            inclusion = overlap / len(child_rels) if child_rels else 0
            if inclusion >= 0.80:
                # Check negative constraint: conflicting rels should block
                exclusive_child = child_rels - parent_rels
                exclusive_parent = parent_rels - child_rels
                # If child has unique defining rels that contradict parent's, skip
                if len(exclusive_child) > len(child_rels) * 0.5:
                    continue
                candidates.append((child, parent, inclusion))

    if not candidates:
        print("  No IS-A candidates found from relation subsumption")
        return []

    # Sort by inclusion score (strongest first)
    candidates.sort(key=lambda x: -x[2])
    print(f"  Found {len(candidates)} candidate IS-A pairs")

    # Step 3: LLM validation (reject only)
    validated: list[tuple[str, str]] = []
    if candidates:
        # Format candidates for LLM
        cand_lines = []
        for child, parent, score in candidates[:15]:  # Cap at 15 for prompt size
            child_members = [e["id"] for e in entities
                            if assignments.get(e["id"]) == child][:5]
            parent_members = [e["id"] for e in entities
                             if assignments.get(e["id"]) == parent][:5]
            cand_lines.append(
                f"  {child} (e.g., {', '.join(child_members)}) -> "
                f"{parent} (e.g., {', '.join(parent_members)})"
            )

        prompt = f"""Review these proposed IS-A (subclass) relationships between ontology classes.

For each pair, answer YES if the child class is truly a subtype of the parent class,
or NO if they are merely related but not in an IS-A relationship.

PROPOSED RELATIONSHIPS (Child -> Parent):
{chr(10).join(cand_lines)}

For each pair, output exactly: ChildClass -> ParentClass: YES or NO
Only answer YES if every instance of the child class IS-A instance of the parent class.
"""
        raw = _invoke_llm(llm, prompt)

        line_re = re.compile(r'(\w+)\s*->\s*(\w+)\s*:\s*(YES|NO)', re.IGNORECASE | re.MULTILINE)
        for m in line_re.finditer(raw):
            child_lbl = _sanitize_label(m.group(1))
            parent_lbl = _sanitize_label(m.group(2))
            verdict = m.group(3).upper()
            if verdict == "YES" and child_lbl in class_names and parent_lbl in class_names:
                validated.append((child_lbl, parent_lbl))

    # Step 4: DAG enforcement (remove cycles, enforce max depth 3)
    MAX_TAXONOMY_DEPTH = 3

    final_edges: list[tuple[str, str]] = []
    # Track parent relationships for cycle and depth checks
    parent_of: dict[str, str] = {}  # child -> parent (single-parent tree)

    def _depth_of(node: str) -> int:
        """Compute depth of node in current tree (0 = root)."""
        d = 0
        cur = node
        visited: set[str] = set()
        while cur in parent_of and cur not in visited:
            visited.add(cur)
            cur = parent_of[cur]
            d += 1
        return d

    for child, parent in validated:
        if child == parent:
            continue
        # Cycle check: walk up from parent — if we reach child, it's a cycle
        cur = parent
        is_cycle = False
        visited: set[str] = set()
        while cur in parent_of and cur not in visited:
            visited.add(cur)
            cur = parent_of[cur]
            if cur == child:
                is_cycle = True
                break
        if is_cycle:
            continue
        # Depth check: child's depth would be parent's depth + 1
        if _depth_of(parent) + 1 > MAX_TAXONOMY_DEPTH:
            continue
        # Accept edge (first-come wins for single parent)
        if child not in parent_of:
            parent_of[child] = parent
            final_edges.append((child, parent))

    print(f"  ✓ {len(final_edges)} validated IS-A edges (from {len(candidates)} candidates)")
    for child, parent in final_edges:
        print(f"    {child} SUBCLASS_OF {parent}")

    return final_edges


def derive_taxonomy_llm_pairwise(
    assignments: dict[str, str],
    entities: list[dict],
    llm: ChatOllama,
) -> list[tuple[str, str]]:
    """IS-A taxonomy via LLM pairwise judgment.

    For each ordered pair (A, B), ask: "Is A a subtype of B?"
    Batch 10 pairs per prompt for efficiency.
    Enforce DAG + max depth 3.

    Unlike derive_taxonomy() (relation-set subsumption) which only finds
    IS-A when child relations are a strict subset of parent's, this method
    can discover IS-A relationships even when classes have overlapping or
    distinct relation profiles — it relies on semantic judgment.

    Returns:
        List of (child, parent) edges representing SUBCLASS_OF.
    """
    print("\n[Taxonomy-LLM] Deriving IS-A hierarchy via pairwise LLM judgment...", flush=True)

    class_counts = Counter(v for v in assignments.values() if v != "Other")
    class_names = sorted(class_counts.keys())
    if len(class_names) < 2:
        return []

    entity_map = {e["id"]: e for e in entities}

    # Build class summaries for the prompt
    class_summaries: dict[str, str] = {}
    for cls in class_names:
        members = [eid for eid, c in assignments.items() if c == cls]
        # Prefer concept members for meaningful names
        concept_members = [
            eid for eid in members
            if is_concept_entity(entity_map.get(eid, {"id": eid, "entity_type": "Unknown"}))
        ]
        sample_pool = concept_members if concept_members else members
        sample = sample_pool[:6]
        class_summaries[cls] = f"{cls} ({class_counts[cls]} members, e.g. {', '.join(sample)})"

    # Generate all ordered pairs: (A, B) means "Is A a subtype of B?"
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(class_names):
        for j, b in enumerate(class_names):
            if i != j:
                pairs.append((a, b))

    print(f"  Evaluating {len(pairs)} ordered pairs in batches of 10...", flush=True)

    # Batch pairs into groups of 10
    PAIRS_PER_BATCH = 10
    raw_yes_pairs: list[tuple[str, str]] = []

    for batch_start in range(0, len(pairs), PAIRS_PER_BATCH):
        batch = pairs[batch_start:batch_start + PAIRS_PER_BATCH]

        pair_lines = []
        for idx, (a, b) in enumerate(batch, 1):
            pair_lines.append(f"{idx}. Is \"{a}\" a subtype of \"{b}\"?")
            pair_lines.append(f"   {a}: {class_summaries[a]}")
            pair_lines.append(f"   {b}: {class_summaries[b]}")

        prompt = f"""For each pair below, answer YES if the first class is a genuine subtype \
(IS-A) of the second class. Answer NO otherwise.

A is a subtype of B means: every instance of A is also an instance of B.
Example: "Flood" IS-A "Peril" (every flood is a peril).
Counter-example: "Coverage" is NOT a subtype of "Policy" (coverage is part of a policy, not a kind of policy).

{chr(10).join(pair_lines)}

Output one line per pair: number. YES or NO
"""
        raw = _invoke_llm(llm, prompt)

        # Parse "1. YES" or "1. NO" lines
        line_re = re.compile(r'(\d+)\.\s*(YES|NO)', re.IGNORECASE)
        for m in line_re.finditer(raw):
            idx = int(m.group(1)) - 1
            verdict = m.group(2).upper()
            if 0 <= idx < len(batch) and verdict == "YES":
                raw_yes_pairs.append(batch[idx])

        batch_num = batch_start // PAIRS_PER_BATCH + 1
        total_batches = (len(pairs) + PAIRS_PER_BATCH - 1) // PAIRS_PER_BATCH
        if batch_num % 5 == 0 or batch_num == total_batches:
            print(f"    Batch {batch_num}/{total_batches}: {len(raw_yes_pairs)} YES so far", flush=True)

    if not raw_yes_pairs:
        print("  No IS-A pairs found via LLM pairwise judgment")
        return []

    print(f"  {len(raw_yes_pairs)} raw YES pairs, enforcing DAG + max depth 3...", flush=True)

    # DAG enforcement: single-parent tree, max depth 3, no cycles
    # Sort by child class size (smaller first) so more specific classes
    # get their parent assigned first — avoids alphabetical bias.
    MAX_TAXONOMY_DEPTH = 3
    final_edges: list[tuple[str, str]] = []
    parent_of: dict[str, str] = {}  # child -> parent

    raw_yes_pairs.sort(key=lambda pair: (class_counts.get(pair[0], 0), pair[0]))

    def _depth_of(node: str) -> int:
        d = 0
        cur = node
        visited: set[str] = set()
        while cur in parent_of and cur not in visited:
            visited.add(cur)
            cur = parent_of[cur]
            d += 1
        return d

    for child, parent in raw_yes_pairs:
        if child == parent:
            continue
        # Cycle check
        cur = parent
        is_cycle = False
        visited: set[str] = set()
        while cur in parent_of and cur not in visited:
            visited.add(cur)
            cur = parent_of[cur]
            if cur == child:
                is_cycle = True
                break
        if is_cycle:
            continue
        # Depth check
        if _depth_of(parent) + 1 > MAX_TAXONOMY_DEPTH:
            continue
        # First-come wins for single parent
        if child not in parent_of:
            parent_of[child] = parent
            final_edges.append((child, parent))

    print(f"  ✓ {len(final_edges)} validated IS-A edges (from {len(raw_yes_pairs)} candidates)")
    for child, parent in final_edges:
        print(f"    {child} SUBCLASS_OF {parent}")

    return final_edges
