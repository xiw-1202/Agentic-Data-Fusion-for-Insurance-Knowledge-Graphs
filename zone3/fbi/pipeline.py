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

import config
from zone3.fbi.class_discovery import (
    CandidateClass,
    name_classes,
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
from zone3.fbi.function_grouping import group_files_by_function
from zone3.fbi.functional_classes import build_functional_classes
from zone3.fbi.relationships import (
    ClassRelationship,
    find_bridge_columns,
    name_relationships,
)
from zone3.fbi.token_classifier import classify_tokens


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
    """Phase 2: Function-first class discovery.

    Pipeline:
      Step 1: classify_tokens → separate LOB/function tokens
      Step 2: group_files_by_function → list[FunctionGroup]
      Step 3: build_functional_classes → list[CandidateClass] with sibling sub-hierarchy
      Step 4: name_classes via LLM
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Function-First Class Discovery")
    print("=" * 60)

    # Step 1: classify tokens
    print("\n[Step 1] Classifying filename tokens (LOB vs function) ...")
    token_classification = classify_tokens(fingerprints)
    print(f"     LOB tokens:      {sorted(token_classification.lob_tokens)}")
    print(f"     Function tokens: {sorted(token_classification.function_tokens)}")
    print(f"     Modifier tokens: {sorted(token_classification.modifier_tokens)}")
    print(f"     LOB groups:      {len(token_classification.lob_groups)}")

    # Step 2: group files by function
    print("\n[Step 2] Grouping files by function across LOBs ...")
    groups = group_files_by_function(fingerprints, token_classification)
    print(f"     Produced {len(groups)} functional groups:")
    for g in groups:
        sizes = f"{len(g.files)} file(s)"
        fns = ",".join(sorted(g.function_tokens)) or "(no function tokens)"
        print(f"       - {fns} [{sizes}]")

    # Step 3: build class hierarchy
    print("\n[Step 3] Building class hierarchy with sibling sub-classes ...")
    classes = build_functional_classes(groups)

    def count_all(cs: list[CandidateClass]) -> int:
        total = len(cs)
        for c in cs:
            total += count_all(c.children)
        return total

    print(
        f"     {count_all(classes)} total classes across "
        f"{len(classes)} top-level groups"
    )

    # Step 4: name classes via LLM (recursively)
    print("\n[Step 4] Naming classes via LLM ...")
    name_classes(classes, model=model)

    def name_recursively(cs: list[CandidateClass], model: str | None) -> None:
        for c in cs:
            if c.children:
                name_classes(c.children, model=model)
                name_recursively(c.children, model)

    name_recursively(classes, model)

    # Print hierarchy summary
    def print_tree(cs: list[CandidateClass], indent: int = 0) -> None:
        for c in cs:
            name = c.name or c.prefix or "(unnamed)"
            n_headers = len(c.headers)
            n_children = len(c.children)
            print(
                f"     {'  ' * indent}- {name} "
                f"({n_headers}h, {n_children} children)"
            )
            print_tree(c.children, indent + 1)

    print_tree(classes)

    return classes


# ---------------------------------------------------------------------------
# Phase 3: Relationship Discovery
# ---------------------------------------------------------------------------


def run_phase3(
    classes: list[CandidateClass],
    fingerprints: list[FileFingerprint],
    model: str | None = None,
) -> list[ClassRelationship]:
    """Phase 3: Relationship discovery using raw file headers.

    Builds a map from top-level class name → union of raw headers across
    all source files (including children recursively), then uses that
    mapping in :func:`find_bridge_columns`. This is essential when classes
    were built from shared-header intersections because bridge columns
    may not appear in any class's shared headers but DO appear in the
    raw file headers.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Relationship Discovery")
    print("=" * 60)

    # Build lookup maps for fingerprints: by full path and by basename
    fp_by_path: dict[str, FileFingerprint] = {fp.file_path: fp for fp in fingerprints}
    fp_by_name: dict[str, FileFingerprint] = {
        fp.file_path.split("/")[-1]: fp for fp in fingerprints
    }

    def collect_source_files(cls: CandidateClass, acc: set[str]) -> None:
        acc.update(cls.source_files)
        if cls.source_file:
            acc.add(cls.source_file)
        for child in cls.children:
            collect_source_files(child, acc)

    raw_by_class: dict[str, set[str]] = {}
    for top_class in classes:
        source_paths: set[str] = set()
        collect_source_files(top_class, source_paths)
        all_raw: set[str] = set()
        for path in source_paths:
            fp = fp_by_path.get(path) or fp_by_name.get(path.split("/")[-1])
            if fp is None:
                continue
            all_raw.update(fp.headers_raw)
            all_raw.update(fp.sections)
        class_name = top_class.name or top_class.prefix or "(unnamed)"
        raw_by_class[class_name] = all_raw

    # Step 1: find bridges using raw headers
    print("\n[Step 1] Finding bridge columns via raw file headers ...")
    bridges = find_bridge_columns(classes, raw_headers_by_class=raw_by_class)
    print(f"     Found {len(bridges)} bridge relationships")

    # Step 2: name via LLM
    if bridges:
        print("\n[Step 2] Naming relationships via LLM ...")
        name_relationships(bridges, model=model)

    for rel in bridges[:20]:
        print(
            f"     - {rel.source_class} --[{rel.relationship_name}]--> "
            f"{rel.target_class}  (via {rel.bridge_column})"
        )

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
    relationships = run_phase3(classes, fingerprints, model=args.model)

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
