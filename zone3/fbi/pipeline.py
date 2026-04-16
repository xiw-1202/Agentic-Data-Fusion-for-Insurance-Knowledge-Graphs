"""FBI pipeline: Fingerprint-Based Ontology Induction orchestrator.

Runs 4 phases:
  1. Fingerprinting  – extract headers, expand via LLM, parse filenames
  2. Class discovery  – prefix groups, sibling patterns, semantic grouping, merge, name
  3. Relationships    – bridge columns, LLM naming
  4. Entity assignment – map Zone 2 triples to classes, write to Neo4j

Usage:
    python -m zone3.fbi.pipeline --data-dir data/Emory_Spring2026 --model qwen2.5:72b
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import config
from zone3.fbi.class_discovery import (
    CandidateClass,
    detect_sibling_patterns,
    find_prefix_groups,
    get_ungrouped_headers,
    merge_cross_file_classes,
    name_classes,
    semantic_group_headers,
)
from zone3.fbi.entity_assign import (
    assign_all_entities,
    write_ontology_to_neo4j,
)
from zone3.fbi.fingerprint import (
    FileFingerprint,
    expand_all_headers,
    extract_fingerprints,
    parse_filename_tokens,
)
from zone3.fbi.relationships import (
    ClassRelationship,
    find_bridge_columns,
    name_relationships,
)


# ---------------------------------------------------------------------------
# Phase 1: Fingerprinting
# ---------------------------------------------------------------------------


def run_phase1(
    data_dir: str,
    model: str | None = None,
) -> list[FileFingerprint]:
    """Phase 1: Extract headers, expand via LLM, parse filenames."""
    print("\n" + "=" * 60)
    print("PHASE 1: Fingerprinting")
    print("=" * 60)

    # Step 1a: extract raw fingerprints
    print("\n[1a] Extracting fingerprints from data files ...")
    fingerprints = extract_fingerprints(data_dir)
    csv_count = sum(1 for fp in fingerprints if fp.file_type == "csv")
    pdf_count = sum(1 for fp in fingerprints if fp.file_type == "pdf")
    txt_count = sum(1 for fp in fingerprints if fp.file_type == "txt")
    print(f"     Found {len(fingerprints)} files: {csv_count} CSV, {pdf_count} PDF, {txt_count} TXT")

    # Step 1b: expand abbreviated headers via LLM
    print("\n[1b] Expanding headers via LLM ...")
    expand_all_headers(fingerprints, model=model)
    total_expanded = sum(len(fp.headers_expanded) for fp in fingerprints)
    print(f"     Expanded {total_expanded} header abbreviations")

    # Step 1c: parse filename tokens via LLM
    print("\n[1c] Parsing filename tokens via LLM ...")
    parse_filename_tokens(fingerprints, model=model)
    total_tokens = sum(len(fp.filename_tokens) for fp in fingerprints)
    print(f"     Extracted {total_tokens} filename tokens")

    return fingerprints


# ---------------------------------------------------------------------------
# Phase 2: Class Discovery
# ---------------------------------------------------------------------------


def run_phase2(
    fingerprints: list[FileFingerprint],
    model: str | None = None,
) -> list[CandidateClass]:
    """Phase 2: Multi-iteration class discovery."""
    print("\n" + "=" * 60)
    print("PHASE 2: Class Discovery")
    print("=" * 60)

    all_classes: list[CandidateClass] = []

    # --- Iter 1: prefix groups + sibling patterns per CSV file ---
    print("\n[Iter 1] Prefix grouping + sibling detection per CSV file ...")
    for fp in fingerprints:
        if fp.file_type != "csv":
            continue

        groups = find_prefix_groups(fp.headers_raw)
        for g in groups:
            g.source_file = fp.basename
            g.source_files = [fp.file_path]

        siblings = detect_sibling_patterns(groups)
        # Sibling groups create parent-child hierarchies
        for sib in siblings:
            parent = CandidateClass(
                prefix=sib.common_prefix,
                headers=[],
                children=list(sib.children),
                source_file=fp.basename,
                source_files=[fp.file_path],
                level=1,
            )
            for child in parent.children:
                child.parent = parent
                child.level = 2
                child.source_files = [fp.file_path]
            all_classes.append(parent)

        # Non-sibling groups are standalone classes
        sibling_group_indices: set[int] = set()
        for sib in siblings:
            for child in sib.children:
                for i, g in enumerate(groups):
                    if g is child:
                        sibling_group_indices.add(i)

        for i, g in enumerate(groups):
            if i not in sibling_group_indices:
                g.level = 1
                g.source_files = [fp.file_path]
                all_classes.append(g)

    print(f"     Found {len(all_classes)} prefix-based classes")

    # --- Iter 2: semantic grouping for ungrouped headers ---
    print("\n[Iter 2] Semantic grouping of ungrouped headers ...")
    semantic_count = 0
    for fp in fingerprints:
        if fp.file_type == "csv":
            # Use expanded names for better semantic grouping
            headers = list(fp.headers_expanded.values()) if fp.headers_expanded else fp.headers_raw
            ungrouped = get_ungrouped_headers(headers, all_classes)
            if ungrouped:
                sem_groups = semantic_group_headers(ungrouped, model=model)
                for sg in sem_groups:
                    sg.source_file = fp.basename
                    sg.source_files = [fp.file_path]
                all_classes.extend(sem_groups)
                semantic_count += len(sem_groups)

        elif fp.file_type in ("pdf", "txt"):
            # One class per document — sections are attributes, not classes
            if fp.sections or fp.defined_terms:
                all_headers = fp.sections + fp.defined_terms
                cls = CandidateClass(
                    prefix="",
                    headers=all_headers,
                    source_file=fp.basename,
                    source_files=[fp.file_path],
                    level=1,
                )
                all_classes.append(cls)
                semantic_count += 1

    print(f"     Added {semantic_count} semantically-grouped classes")

    # --- Iter 3: merge cross-file classes ---
    print("\n[Iter 3] Merging cross-file classes ...")
    pre_merge = len(all_classes)
    all_classes = merge_cross_file_classes(all_classes)
    print(f"     {pre_merge} classes -> {len(all_classes)} after merging")

    # --- Iter 4: name classes via LLM ---
    print("\n[Iter 4] Naming classes via LLM ...")
    name_classes(all_classes, model=model)
    for cls in all_classes:
        children_str = ""
        if cls.children:
            child_names = [c.name or c.prefix for c in cls.children]
            children_str = f" -> [{', '.join(child_names)}]"
        print(f"     - {cls.name or cls.prefix}{children_str}")

    return all_classes


# ---------------------------------------------------------------------------
# Phase 3: Relationship Discovery
# ---------------------------------------------------------------------------


def run_phase3(
    classes: list[CandidateClass],
    model: str | None = None,
) -> list[ClassRelationship]:
    """Phase 3: Relationship discovery via bridge columns."""
    print("\n" + "=" * 60)
    print("PHASE 3: Relationship Discovery")
    print("=" * 60)

    print("\n[3a] Finding bridge columns ...")
    bridges = find_bridge_columns(classes)
    print(f"     Found {len(bridges)} bridge relationships")

    print("\n[3b] Naming relationships via LLM ...")
    name_relationships(bridges, model=model)

    for rel in bridges:
        print(f"     - {rel.source_class} --[{rel.relationship_name}]--> {rel.target_class}  (via {rel.bridge_column})")

    return bridges


# ---------------------------------------------------------------------------
# Phase 4: Entity Assignment + Neo4j
# ---------------------------------------------------------------------------


def run_phase4(
    classes: list[CandidateClass],
    relationships: list[ClassRelationship],
    fingerprints: list[FileFingerprint],
    results_dir: str | None = None,
) -> dict:
    """Phase 4: Assign Zone 2 entities to classes, write to Neo4j."""
    print("\n" + "=" * 60)
    print("PHASE 4: Entity Assignment + Neo4j")
    print("=" * 60)

    if results_dir is None:
        results_dir = config.RESULTS_DIR

    # Load Zone 2 triples
    summary_path = os.path.join(results_dir, "zone2_run_summary.json")
    if not os.path.exists(summary_path):
        print(f"\n  WARNING: {summary_path} not found — skipping Phase 4")
        return {}

    print(f"\n[4a] Loading Zone 2 triples from {summary_path} ...")
    with open(summary_path, "r") as f:
        zone2_summary = json.load(f)

    # Build entity list with source_files from triple's "source" field
    entity_sources: dict[str, set[str]] = defaultdict(set)
    triples = zone2_summary.get("triples", [])
    if not triples:
        # Try alternate key names
        triples = zone2_summary.get("relationships", [])

    for triple in triples:
        source_file = triple.get("source", "")
        subj = triple.get("subject", triple.get("head", ""))
        obj = triple.get("object", triple.get("tail", ""))
        if subj and source_file:
            entity_sources[subj].add(source_file)
        if obj and source_file:
            entity_sources[obj].add(source_file)

    entities = [
        {"id": eid, "source_files": list(files)}
        for eid, files in entity_sources.items()
    ]
    print(f"     Found {len(entities)} unique entities from {len(triples)} triples")

    # Assign entities to classes
    print("\n[4b] Assigning entities to classes ...")
    assignments = assign_all_entities(entities, classes)

    # Count per class
    class_counts: dict[str, int] = defaultdict(int)
    for cls_name in assignments.values():
        class_counts[cls_name] += 1

    print(f"     Assigned {len(assignments)} entities:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"       {cls_name}: {count}")

    # Write to Neo4j
    print("\n[4c] Writing ontology to Neo4j ...")
    filename_tokens = {
        fp.basename: fp.filename_tokens for fp in fingerprints
    }
    neo4j_summary = write_ontology_to_neo4j(
        classes=classes,
        relationships=relationships,
        entity_assignments=assignments,
        filename_tokens=filename_tokens,
    )
    print(f"     Neo4j write complete: {neo4j_summary}")

    return {
        "entity_count": len(entities),
        "assignment_count": len(assignments),
        "class_counts": dict(class_counts),
        "neo4j_summary": neo4j_summary,
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def _class_to_dict(cls: CandidateClass) -> dict:
    """Recursively convert a CandidateClass to a plain dict."""
    return {
        "prefix": cls.prefix,
        "name": cls.name,
        "headers": cls.headers,
        "suffixes": cls.suffixes,
        "source_file": cls.source_file,
        "source_files": cls.source_files,
        "unique_headers": cls.unique_headers,
        "shared_headers": cls.shared_headers,
        "level": cls.level,
        "children": [_class_to_dict(c) for c in cls.children],
    }


def save_results(
    fingerprints: list[FileFingerprint],
    classes: list[CandidateClass],
    relationships: list[ClassRelationship],
    output_dir: str,
) -> None:
    """Save intermediate results as JSON."""
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Fingerprints
    fp_path = os.path.join(output_dir, "fbi_fingerprints.json")
    fp_data = [
        {
            "file_path": fp.file_path,
            "file_type": fp.file_type,
            "filename_tokens": fp.filename_tokens,
            "headers_raw": fp.headers_raw,
            "headers_expanded": fp.headers_expanded,
            "sections": fp.sections,
            "defined_terms": fp.defined_terms,
            "record_count": fp.record_count,
        }
        for fp in fingerprints
    ]
    with open(fp_path, "w") as f:
        json.dump(fp_data, f, indent=2)
    print(f"  -> {fp_path} ({len(fp_data)} fingerprints)")

    # Classes (recursive)
    cls_path = os.path.join(output_dir, "fbi_classes.json")
    cls_data = [_class_to_dict(c) for c in classes]
    with open(cls_path, "w") as f:
        json.dump(cls_data, f, indent=2)
    print(f"  -> {cls_path} ({len(cls_data)} top-level classes)")

    # Relationships
    rel_path = os.path.join(output_dir, "fbi_relationships.json")
    rel_data = [
        {
            "source_class": r.source_class,
            "target_class": r.target_class,
            "relationship_name": r.relationship_name,
            "bridge_column": r.bridge_column,
            "confidence": r.confidence,
        }
        for r in relationships
    ]
    with open(rel_path, "w") as f:
        json.dump(rel_data, f, indent=2)
    print(f"  -> {rel_path} ({len(rel_data)} relationships)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for FBI pipeline."""
    parser = argparse.ArgumentParser(
        description="Fingerprint-Based Ontology Induction (FBI) pipeline",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing CSV/PDF/TXT data files",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name (default: from config.py)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output JSON files (default: RESULTS_DIR/fbi)",
    )
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip Phase 4 (entity assignment + Neo4j write)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(config.RESULTS_DIR, "fbi")

    print("=" * 60)
    print("FBI Pipeline: Fingerprint-Based Ontology Induction")
    print("=" * 60)
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Model:      {args.model or config.OLLAMA_MODEL}")
    print(f"  Output dir: {output_dir}")
    print(f"  Skip Neo4j: {args.skip_neo4j}")

    t0 = time.time()

    # Phase 1: Fingerprinting
    fingerprints = run_phase1(args.data_dir, model=args.model)

    # Phase 2: Class Discovery
    classes = run_phase2(fingerprints, model=args.model)

    # Phase 3: Relationship Discovery
    relationships = run_phase3(classes, model=args.model)

    # Save intermediate results
    save_results(fingerprints, classes, relationships, output_dir)

    # Phase 4: Entity Assignment + Neo4j (optional)
    if not args.skip_neo4j:
        run_phase4(classes, relationships, fingerprints)
    else:
        print("\n[Skipping Phase 4: --skip-neo4j]")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"FBI Pipeline complete in {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
