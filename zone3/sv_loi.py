"""Zone 3 — SV-LOI: Structurally-Verified LLM Ontology Induction

Novel ontology induction method that fuses LLM semantic typing with
graph-structural verification. Neither signal alone is sufficient:
LLM typing is accurate but inconsistent across batches; structural
clustering is consistent but over-fragments.

Key insight: entities of the same ontological class share two independent
signals — (1) the LLM recognizes what they ARE (semantic), and
(2) they participate in the same types of relations (structural).
Fusing both with disagreement arbitration eliminates each signal's
failure mode.

Algorithm:
  Phase 1 — LLM Entity Typing:
    Batch entities (20/prompt) with name + type + top relations.
    LLM assigns ontology class from a discovered class vocabulary.
    ~70 LLM calls for 1,351 entities.

  Phase 2 — Structural Consensus Verification:
    Build relation-signature vectors per entity.
    Compute class centroid for each LLM-assigned class.
    Flag entities whose signature deviates >2σ from their class centroid.

  Phase 3 — Disagreement Arbitration:
    For flagged entities, re-query LLM with enriched structural context:
    "You typed this as X, but it structurally resembles Y. Which is correct?"

  Phase 4 — Hierarchy Derivation:
    LLM proposes SUBCLASS_OF from final class set.
    Write ontology layer to Neo4j.

Usage:
  python3 zone3/sv_loi.py
  python3 zone3/sv_loi.py --model qwen2.5:72b
  python3 zone3/sv_loi.py --model qwen2.5:72b --suffix zone3_svloi

Pre-requisite: Zone 2 must have run first (Neo4j populated with :Entity nodes).
"""

from __future__ import annotations

import json
import random
import re
import time
import os
import sys
import argparse
from typing import Any, Optional, Union
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

import config
from zone3.graph_cache import (
    load_cached_entities,
    get_concept_entities,
    get_entity_lane,
    is_concept_entity,
    STRUCTURED_PREFIXES,
)
from zone3._svloi.constants import (
    BATCH_SIZE,
    MAX_MEMBERS_IN_PROMPT,
    MIN_CLASS_SIZE,
    DEVIATION_THRESHOLD,
    MAX_ARBITRATION_BATCH,
    MAX_CLASS_FRACTION,
    _STRUCTURED_PREFIXES,
    TARGET_CLASSES_MIN,
    TARGET_CLASSES_MAX,
    PROTECTED_CLASS_NAMES,
    FORBIDDEN_CLASS_NAMES,
    ZONE2_TYPE_NORMALIZATION,
)
from zone3._svloi.utils import (
    get_llm,
    get_neo4j_graph,
    _sanitize_label,
    _sanitize_rel_type,
    _parse_json_safely,
    _invoke_llm,
    load_entities,
)
from zone3._svloi.sohd import detect_and_split_heterogeneous_classes
from zone3._svloi.records import decompose_records, write_record_decomposition
from zone3._svloi.writer import (
    validate_backbone,
    write_ontology,
    _flush_print,
    _compute_intrinsic_quality,
)
from zone3._svloi.structural import (
    build_structural_signatures,
    structural_consensus_check,
    arbitrate_disagreements,
)
from zone3._svloi.hierarchy import (
    infer_class_relations,
    merge_small_classes,
    merge_leaf_classes,
    derive_interclass_edges,
    derive_hierarchy,
    derive_taxonomy,
    derive_taxonomy_llm_pairwise,
)
from zone3._svloi.typing import (
    analyze_record_evidence,
    discover_class_vocabulary,
    batch_type_entities,
    rescue_other_entities,
    type_value_entities,
    rebalance_mega_classes,
    propagate_to_records,
)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_sv_loi(
    model: str = config.OLLAMA_MODEL,
    suffix: str = "zone3_svloi",
    skip_verify: bool = False,
    skip_arbitrate: bool = False,
    skip_consolidate: bool = False,
    skip_record_propagation: bool = False,
    skip_sohd: bool = False,
    use_old_rebalance: bool = False,
    seed: int = 42,
    results_dir: str | None = None,
) -> dict:
    """Run the full SV-LOI pipeline (7 stages).

    Stage 1: Load + Prepare     (load cache, record evidence, structural sigs)
    Stage 2: Discover Classes   (domain detection, class vocabulary proposal)
    Stage 3: Classify Entities  (batch typing, rebalance, rescue)
    Stage 4: Verify + Propagate (concept verify, record propagation, value typing, full verify)
    Stage 5: Consolidate        (5-way class relations, LLM validation, structural merge)
    Stage 6: Build Structure    (LLM pairwise taxonomy, subsumption, associations, decomposition)
    Stage 7: Write + Report     (confidence scoring, quality metrics, Neo4j write)

    Ablation flags:
        skip_verify:              Skip structural consensus verification
        skip_arbitrate:           Skip disagreement arbitration
        skip_consolidate:         Skip LLM-guided class consolidation
        skip_record_propagation:  Skip record propagation (use Zone 2 entity_type)
        use_old_rebalance:        Use old rebalance (total entities, 25% threshold)
        seed:                     Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    rdir = results_dir or config.RESULTS_DIR
    os.makedirs(rdir, exist_ok=True)

    ablation_flags = []
    if skip_verify:
        ablation_flags.append("no-verify")
    if skip_arbitrate:
        ablation_flags.append("no-arbitrate")
    if skip_consolidate:
        ablation_flags.append("no-consolidate")
    if skip_record_propagation:
        ablation_flags.append("no-record-propagation")
    if skip_sohd:
        ablation_flags.append("no-sohd")
    if use_old_rebalance:
        ablation_flags.append("old-rebalance")
    ablation_str = f" [ABLATION: {', '.join(ablation_flags)}]" if ablation_flags else ""

    _flush_print("=" * 70)
    _flush_print("CS584 Capstone — Zone 3: SV-LOI (7-Stage Pipeline)")
    _flush_print(f"Structurally-Verified LLM Ontology Induction{ablation_str}")
    _flush_print(f"Model: {model} | Suffix: {suffix} | Seed: {seed}")
    _flush_print("=" * 70)

    start = time.time()

    # ===================================================================
    # Stage 1: Load + Prepare
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 1: Load + Prepare")
    _flush_print("─" * 50)

    entities = load_cached_entities(fmt="sv_loi", results_dir=rdir)
    if not entities:
        return {"error": "no entities"}

    llm = get_llm(model)
    entity_map_all = {e["id"]: e for e in entities}

    # Record evidence analysis (informs class discovery)
    record_evidence = analyze_record_evidence(entities)
    if record_evidence:
        _flush_print(f"\n  Record evidence:\n{record_evidence}")

    # Concept entities for class discovery
    concept_entities = get_concept_entities(entities)

    # Structural signatures (used in verify + consolidate stages)
    features, entity_ids, feature_names = build_structural_signatures(entities)

    _flush_print(f"  ✓ Loaded {len(entities)} entities "
                 f"({len(concept_entities)} concepts), "
                 f"{len(feature_names)} features")

    # ===================================================================
    # Stage 2: Discover Classes
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 2: Discover Classes")
    _flush_print("─" * 50)

    class_vocab, detected_domain = discover_class_vocabulary(
        concept_entities, llm, record_evidence=record_evidence,
        all_entities=entities,
    )

    # ===================================================================
    # Stage 3: Classify Entities (batch typing + rebalance + rescue)
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 3: Classify Entities")
    _flush_print("─" * 50)

    # Pass 1: LLM batch typing
    assignments = batch_type_entities(entities, class_vocab, llm, results_dir=rdir)

    # Pass 2: Rebalance mega-classes + rescue Other (combined post-pass)
    assignments, class_vocab = rebalance_mega_classes(
        assignments, entities, class_vocab, llm,
        use_old_rebalance=use_old_rebalance,
    )
    assignments = rescue_other_entities(assignments, entities, llm)

    # ===================================================================
    # Stage 4: Verify + Propagate
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 4: Verify + Propagate")
    _flush_print("─" * 50)

    # --- Decision provenance tracking ---
    provenance: dict[str, dict] = {}
    for eid, cls in assignments.items():
        provenance[eid] = {"llm_type": cls, "flagged": False, "arbitrated": False}

    total_flagged = 0

    # Sub-pass A: Concept-only verification (clean centroids, no record pollution)
    if skip_verify:
        _flush_print("\n  [ABLATION] Skipping structural verification")
    else:
        _flush_print("\n  Sub-pass A: Concept-only structural verification")
        concept_only_assignments = {
            eid: cls for eid, cls in assignments.items()
            if get_entity_lane(entity_map_all.get(eid, {"id": eid, "entity_type": "Unknown"})) == "concept"
        }
        class_stats, flagged = structural_consensus_check(
            concept_only_assignments, features, entity_ids,
        )
        total_flagged += len(flagged)

        for f_entry in flagged:
            eid = f_entry["entity_id"]
            if eid in provenance:
                provenance[eid]["flagged"] = True
                provenance[eid]["outlier_score"] = f_entry["similarity"]
                provenance[eid]["class_mean_sim"] = f_entry["class_mean_sim"]
                provenance[eid]["nearest_alt"] = f_entry["nearest_class"]

        if flagged and not skip_arbitrate:
            pre_arb = dict(assignments)
            assignments = arbitrate_disagreements(flagged, entities, assignments, class_vocab, llm)
            for eid in assignments:
                if eid in pre_arb and assignments[eid] != pre_arb[eid]:
                    if eid in provenance:
                        provenance[eid]["arbitrated"] = True
                        provenance[eid]["pre_arb_class"] = pre_arb[eid]
        elif flagged and skip_arbitrate:
            _flush_print(f"  [ABLATION] Skipping arbitration — {len(flagged)} flagged")

    # Sub-pass B: Record propagation + value typing
    if skip_record_propagation:
        _flush_print("\n  [ABLATION] Skipping record propagation (using Zone 2 types)")
    else:
        _flush_print("\n  Sub-pass B: Record propagation + value typing")
        concept_assignments_verified = {
            eid: cls for eid, cls in assignments.items()
            if get_entity_lane(entity_map_all.get(eid, {"id": eid, "entity_type": "Unknown"})) == "concept"
        }
        record_assignments, redirects = propagate_to_records(
            concept_assignments_verified, entities, entity_map_all,
            class_vocab=class_vocab, llm=llm,
        )
        for eid, cls in record_assignments.items():
            assignments[eid] = cls
        # Apply redirects: fix entities typed with record-type class names
        if redirects:
            redirected = 0
            for eid in list(assignments.keys()):
                old_cls = assignments[eid]
                if old_cls in redirects:
                    assignments[eid] = redirects[old_cls]
                    redirected += 1
            for old_cls, new_cls in redirects.items():
                if old_cls in class_vocab:
                    class_vocab.remove(old_cls)
                if new_cls not in class_vocab:
                    class_vocab.append(new_cls)
            _flush_print(f"  Redirected {redirected} entities, "
                         f"cleaned vocab: removed {list(redirects.keys())}")

    # Value typing AFTER record propagation (needs record neighbor classes)
    assignments, rel_to_class = type_value_entities(assignments, entities, class_vocab)

    # Sub-pass C: Full verification (all entities now typed)
    if not skip_verify:
        _flush_print("\n  Sub-pass C: Full structural verification")
        class_stats, flagged = structural_consensus_check(assignments, features, entity_ids)
        total_flagged += len(flagged)

        for f_entry in flagged:
            eid = f_entry["entity_id"]
            if eid in provenance:
                provenance[eid]["flagged"] = True
                provenance[eid]["outlier_score"] = f_entry["similarity"]

        if flagged and not skip_arbitrate:
            pre_arb = dict(assignments)
            assignments = arbitrate_disagreements(flagged, entities, assignments, class_vocab, llm)
            for eid in assignments:
                if eid in pre_arb and assignments[eid] != pre_arb[eid]:
                    if eid in provenance:
                        provenance[eid]["arbitrated"] = True
                        provenance[eid]["pre_arb_class"] = pre_arb[eid]

    # ===================================================================
    # Stage 5: Consolidate Classes (5-way inference + LLM validation + structural merge)
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 5: Consolidate Classes")
    _flush_print("─" * 50)

    # Two-lane: concept entities drive consolidation
    concept_assignments = {
        eid: cls for eid, cls in assignments.items()
        if get_entity_lane(entity_map_all.get(eid, {"id": eid, "entity_type": "Unknown"})) == "concept"
    }
    _flush_print(f"  {len(concept_assignments)} concepts drive consolidation")

    pre_consolidate = dict(assignments)
    if skip_consolidate:
        _flush_print("\n  [ABLATION] Skipping class relation inference")
        hierarchy = derive_hierarchy(assignments, llm)
    else:
        # 5-way class relation inference (concept entities only)
        concept_assignments, hierarchy = infer_class_relations(
            concept_assignments, entities, llm,
        )
        # Propagate concept consolidation to ALL entities
        remap_candidates: dict[str, set[str]] = defaultdict(set)
        for eid, new_cls in concept_assignments.items():
            old_cls = pre_consolidate.get(eid, "Other")
            if old_cls != new_cls and old_cls != "Other":
                remap_candidates[old_cls].add(new_cls)

        class_remap = {
            old: next(iter(news))
            for old, news in remap_candidates.items()
            if len(news) == 1
        }
        ambiguous = {old: news for old, news in remap_candidates.items() if len(news) > 1}
        if ambiguous:
            _flush_print(f"  ⚠ Ambiguous remaps skipped: {dict(ambiguous)}")

        if class_remap:
            _flush_print(f"  Propagating {len(class_remap)} class remaps to all entities...")
            for eid in list(assignments.keys()):
                old = assignments[eid]
                if old in class_remap:
                    assignments[eid] = class_remap[old]
        for eid, cls in concept_assignments.items():
            assignments[eid] = cls

    # Record consolidation changes in provenance
    for eid in assignments:
        if eid in pre_consolidate and assignments[eid] != pre_consolidate[eid]:
            if eid in provenance:
                provenance[eid]["consolidated_from"] = pre_consolidate[eid]

    # LLM class validation BEFORE structural merge (reordered from old pipeline)
    # This prevents merge_small_classes from merging classes that LLM would keep
    assignments = merge_leaf_classes(assignments, entities, llm=llm)

    # Structural merge of small classes AFTER LLM validation
    assignments = merge_small_classes(assignments, features, entity_ids)

    # ===================================================================
    # Stage 6: Build Ontology Structure (taxonomy + associations + decomposition)
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 6: Build Ontology Structure")
    _flush_print("─" * 50)

    # SOHD: Structural Ontological Heterogeneity Detection
    # Runs FIRST so taxonomy signals operate on the post-split class universe
    sohd_stats: dict = {}
    sohd_edges: list[tuple[str, str]] = []
    if skip_sohd:
        _flush_print("\n  [ABLATION] Skipping SOHD hierarchy deepening")
    else:
        _flush_print("\n  SOHD: Detecting structurally heterogeneous classes...")
        assignments, sohd_edges, sohd_stats = detect_and_split_heterogeneous_classes(
            assignments, entities, llm=llm, seed=seed,
        )
        _flush_print(f"  ✓ SOHD: {len(sohd_edges)} new IS-A edges from "
                     f"{len(set(c for c, _ in sohd_edges))} new subclasses")

    # Snapshot 5-way IS-A edges from Stage 5
    n_5way_edges = len(hierarchy)

    # IS-A taxonomy via LLM pairwise judgment (primary signal)
    # Now operates on post-SOHD assignments (includes new subclasses)
    taxonomy_llm_edges = derive_taxonomy_llm_pairwise(assignments, entities, llm)

    # Concept-only subsumption as structural validation (secondary signal)
    taxonomy_subsumption_edges = derive_taxonomy(assignments, entities, llm)

    # Collect all candidate IS-A edges from all sources
    all_candidate_isa: list[tuple[str, str]] = list(hierarchy)  # 5-way edges
    all_candidate_isa.extend(taxonomy_llm_edges)
    all_candidate_isa.extend(taxonomy_subsumption_edges)
    all_candidate_isa.extend(sohd_edges)

    # Global DAG enforcement across ALL sources:
    # 1. Filter out edges referencing unknown classes
    # 2. Remove cycles (keep first edge, reject contradicting edge)
    # 3. Enforce max depth 3
    final_classes = set(c for c in assignments.values() if c != "Other")
    MAX_DEPTH = 4

    # Filter unknown classes
    valid_edges = [(c, p) for c, p in all_candidate_isa
                   if c in final_classes and p in final_classes and c != p]
    n_unknown = len(all_candidate_isa) - len(valid_edges)
    if n_unknown > 0:
        _flush_print(f"  Filtered {n_unknown} edges referencing unknown classes")

    # Deduplicate
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for edge in valid_edges:
        if edge not in seen:
            deduped.append(edge)
            seen.add(edge)

    # DAG enforcement: no cycles, max depth, single parent per child
    parent_of: dict[str, str] = {}

    def _depth(node: str) -> int:
        d, cur, visited = 0, node, set()
        while cur in parent_of and cur not in visited:
            visited.add(cur)
            cur = parent_of[cur]
            d += 1
        return d

    hierarchy = []
    rejected_cycles = []
    for child, parent in deduped:
        # Cycle check: walk up from parent — if we reach child, it's a cycle
        cur, is_cycle, visited = parent, False, set()
        while cur in parent_of and cur not in visited:
            visited.add(cur)
            cur = parent_of[cur]
            if cur == child:
                is_cycle = True
                break
        if is_cycle:
            rejected_cycles.append((child, parent))
            continue
        # Depth check
        if _depth(parent) + 1 > MAX_DEPTH:
            continue
        # Single parent (first-come wins)
        if child not in parent_of:
            parent_of[child] = parent
            hierarchy.append((child, parent))

    if rejected_cycles:
        _flush_print(f"  Rejected {len(rejected_cycles)} cyclic edges:")
        for c, p in rejected_cycles:
            _flush_print(f"    {c} → {p} (would create cycle)")

    # Association edges — SEPARATE from IS-A (these are HAS-A / REFERENCES)
    _flush_print("\n  Association edges from entity connections...")
    data_edges = derive_interclass_edges(assignments, entities)
    existing_isa = set(hierarchy)
    # Deduplicate: remove any association that duplicates an IS-A edge
    associations = [e for e in data_edges
                    if e not in existing_isa
                    and e[0] in final_classes and e[1] in final_classes]

    n_llm_tax = len(taxonomy_llm_edges)
    n_sub_tax = len(taxonomy_subsumption_edges)
    n_assoc = len(associations)
    _flush_print(f"  IS-A edges: {len(hierarchy)} "
                 f"({n_llm_tax} LLM-pairwise + {n_sub_tax} subsumption + {n_5way_edges} from 5-way)")
    _flush_print(f"  Association edges: {n_assoc} (ASSOCIATED_WITH, not SUBCLASS_OF)")

    # Record decomposition (Q5 bridge fix)
    decomposition = decompose_records(assignments, entities, rel_to_class=rel_to_class)

    # Backbone validation
    backbone_report = validate_backbone(assignments, entities)

    # ===================================================================
    # Stage 7: Write + Report
    # ===================================================================
    _flush_print("\n" + "─" * 50)
    _flush_print("STAGE 7: Write + Report")
    _flush_print("─" * 50)

    # Confidence scoring
    for eid, cls in assignments.items():
        if eid in provenance:
            provenance[eid]["final_type"] = cls
            p = provenance[eid]
            if cls == "Other":
                p["confidence"] = 0.3
            elif p.get("arbitrated"):
                p["confidence"] = 0.5
            elif p.get("flagged"):
                p["confidence"] = 0.7
            elif p.get("consolidated_from"):
                p["confidence"] = 0.8
            else:
                p["confidence"] = 1.0

    # Intrinsic quality metrics
    final_dist = Counter(v for v in assignments.values() if v != "Other")
    quality_metrics = _compute_intrinsic_quality(
        assignments, entities, entity_map_all, final_dist,
    )

    # Write to Neo4j
    neo4j_stats = write_ontology(assignments, hierarchy, associations=associations)

    # Write decomposed record sub-nodes
    decomp_count = 0
    if decomposition:
        decomp_count = write_record_decomposition(decomposition)

    elapsed = time.time() - start

    # Summary
    _flush_print(f"\n{'=' * 70}")
    _flush_print(f"SV-LOI pipeline complete in {elapsed:.1f}s")
    _flush_print(f"  Method:            SV-LOI (Structurally-Verified LLM Ontology Induction)")
    _flush_print(f"  Entities:          {len(entities)}")
    _flush_print(f"  Classes:           {len(final_dist)}")
    _flush_print(f"  SUBCLASS_OF:       {len(hierarchy)} (IS-A)")
    _flush_print(f"  ASSOCIATED_WITH:   {len(associations)} (HAS-A)")
    _flush_print(f"  Flagged/Arbitrated:{total_flagged}")
    _flush_print(f"  Decomposed:        {decomp_count} sub-nodes")
    _flush_print(f"  Distribution:")
    for cls, cnt in final_dist.most_common():
        _flush_print(f"    {cls}: {cnt}")

    # Low-confidence entities
    low_conf = {
        eid: p for eid, p in provenance.items()
        if p.get("confidence", 1.0) < 0.5
    }

    # Save summary
    summary = {
        "mode": "zone3_sv_loi",
        "model": model,
        "suffix": suffix,
        "seed": seed,
        "domain": detected_domain,
        "elapsed_seconds": round(elapsed, 2),
        "entity_count": len(entities),
        "class_vocab_discovered": class_vocab,
        "classes_final": sorted(final_dist.keys()),
        "class_distribution": dict(final_dist),
        "flagged_count": total_flagged,
        "hierarchy": [{"child": c, "parent": p, "type": "SUBCLASS_OF"} for c, p in hierarchy],
        "associations": [{"source": s, "target": t, "type": "ASSOCIATED_WITH"} for s, t in associations],
        "neo4j_stats": neo4j_stats,
        "decomposition_count": decomp_count,
        "ablation": {
            "skip_verify": skip_verify,
            "skip_arbitrate": skip_arbitrate,
            "skip_consolidate": skip_consolidate,
            "skip_sohd": skip_sohd,
        },
        "sohd_stats": sohd_stats,
        "provenance_stats": {
            "total_entities": len(provenance),
            "flagged": sum(1 for p in provenance.values() if p.get("flagged")),
            "arbitrated": sum(1 for p in provenance.values() if p.get("arbitrated")),
            "consolidated": sum(1 for p in provenance.values() if p.get("consolidated_from")),
            "type_changed_by_verification": sum(
                1 for eid, p in provenance.items()
                if p.get("final_type") != p.get("llm_type")
            ),
            "low_confidence_count": len(low_conf),
            "confidence_distribution": {
                "high_1.0": sum(1 for p in provenance.values() if p.get("confidence", 0) == 1.0),
                "good_0.8": sum(1 for p in provenance.values() if p.get("confidence", 0) == 0.8),
                "medium_0.7": sum(1 for p in provenance.values() if p.get("confidence", 0) == 0.7),
                "low_0.5": sum(1 for p in provenance.values() if p.get("confidence", 0) == 0.5),
                "very_low_0.3": sum(1 for p in provenance.values() if p.get("confidence", 0) == 0.3),
            },
        },
        "intrinsic_quality": quality_metrics,
        "backbone": {
            "overall_connectivity": backbone_report.get("overall_connectivity", 0),
            "disconnected_classes": backbone_report.get("disconnected_classes", []),
        },
        "taxonomy_edges": len(taxonomy_llm_edges) + len(taxonomy_subsumption_edges) + len(sohd_edges),
        "taxonomy_edges_detail": {
            "llm_pairwise": len(taxonomy_llm_edges),
            "subsumption": len(taxonomy_subsumption_edges),
            "sohd": len(sohd_edges),
        },
    }

    # Save provenance
    prov_path = os.path.join(rdir, f"svloi_provenance_{suffix}.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    _flush_print(f"  ✓ Decision provenance saved → {prov_path}")

    # Save low-confidence entities
    if low_conf:
        lc_path = os.path.join(rdir, f"zone3_low_confidence_entities_{suffix}.json")
        with open(lc_path, "w") as f:
            json.dump(low_conf, f, indent=2, default=str)
        _flush_print(f"  ✓ {len(low_conf)} low-confidence entities saved → {lc_path}")

    # Print intrinsic quality metrics
    _flush_print(f"  Intrinsic quality:")
    for k, v in quality_metrics.items():
        _flush_print(f"    {k}: {v}")

    out_path = os.path.join(rdir, f"zone3_svloi_summary_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _flush_print(f"\n✓ Summary saved to {out_path}")
    _flush_print(f"\nNext steps:")
    _flush_print(f"  python3 baseline/eval.py --suffix {suffix} --riskine --model {model}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zone 3: SV-LOI — Structurally-Verified LLM Ontology Induction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Novel ontology induction: LLM assigns classes to entities, structural
signatures verify assignments, disagreements are arbitrated by LLM
with enriched context. Produces 10-20 clean ontology classes.

Pre-requisite: Run zone2/pipeline.py first to populate Neo4j with Entity nodes.

Examples:
  python3 zone3/sv_loi.py
  python3 zone3/sv_loi.py --model qwen2.5:72b
  python3 zone3/sv_loi.py --model qwen2.5:72b --suffix zone3_svloi

After running, evaluate with:
  python3 baseline/eval.py --suffix zone3_svloi --riskine
        """,
    )
    parser.add_argument(
        "--model", default=config.OLLAMA_MODEL,
        help=f"Ollama model name (default: {config.OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--suffix", default="zone3_svloi",
        help="Suffix for result files (default: zone3_svloi)"
    )
    parser.add_argument("--skip-verify", action="store_true",
                        help="[ABLATION] Skip Phase 2 structural verification")
    parser.add_argument("--skip-arbitrate", action="store_true",
                        help="[ABLATION] Skip Phase 3 disagreement arbitration")
    parser.add_argument("--skip-consolidate", action="store_true",
                        help="[ABLATION] Skip Phase 4 LLM-guided consolidation")
    parser.add_argument("--skip-record-propagation", action="store_true",
                        help="[ABLATION] Skip Phase 8 record propagation (use Zone 2 types)")
    parser.add_argument("--skip-sohd", action="store_true",
                        help="[ABLATION] Skip SOHD hierarchy deepening")
    parser.add_argument("--use-old-rebalance", action="store_true",
                        help="[ABLATION] Use old rebalance (total entities, 25% threshold)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--results-dir", default=None,
                        help="Results directory (default: config.RESULTS_DIR)")
    args = parser.parse_args()

    run_sv_loi(
        model=args.model,
        suffix=args.suffix,
        skip_verify=args.skip_verify,
        skip_arbitrate=args.skip_arbitrate,
        skip_consolidate=args.skip_consolidate,
        skip_record_propagation=getattr(args, 'skip_record_propagation', False),
        skip_sohd=getattr(args, 'skip_sohd', False),
        use_old_rebalance=getattr(args, 'use_old_rebalance', False),
        seed=args.seed,
        results_dir=args.results_dir,
    )
