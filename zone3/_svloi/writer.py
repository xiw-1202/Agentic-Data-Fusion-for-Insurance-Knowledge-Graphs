"""SV-LOI output: validate ontology backbone and write to Neo4j."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from zone3._svloi.utils import _sanitize_label, get_neo4j_graph


def validate_backbone(
    assignments: dict[str, str],
    entities: list[dict],
) -> dict[str, Any]:
    """Check insurance value chain connectivity (domain-agnostic).

    Uses both direct connections (1-hop) and indirect connections through
    value/Other entities (2-hop) to measure how well the ontology classes
    are interconnected. Records often connect to concepts only through
    shared value nodes, so 2-hop connectivity is essential.
    """
    print("\n[Backbone] Validating entity connectivity...", flush=True)

    # Build entity-to-class lookup and neighbor index
    eid_to_class: dict[str, str] = {}
    eid_to_entity: dict[str, dict] = {}
    for e in entities:
        eid = e["id"]
        eid_to_class[eid] = assignments.get(eid, "Other")
        eid_to_entity[eid] = e

    class_members: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        cls = eid_to_class.get(e["id"], "Other")
        if cls != "Other":
            class_members[cls].append(e)

    # For each class, check connectivity (1-hop direct + 2-hop through Other)
    class_connectivity: dict[str, dict] = {}
    for cls, members in class_members.items():
        connected_direct = 0
        connected_2hop = 0
        neighbor_classes: Counter = Counter()

        for e in members:
            has_direct = False
            has_2hop = False

            # 1-hop: direct connections to other non-Other entities
            for rel in e.get("out_rels", []):
                tcls = eid_to_class.get(rel.get("target", ""), "Other")
                if tcls != "Other" and tcls != cls:
                    has_direct = True
                    neighbor_classes[tcls] += 1
            for rel in e.get("in_rels", []):
                scls = eid_to_class.get(rel.get("source", ""), "Other")
                if scls != "Other" and scls != cls:
                    has_direct = True
                    neighbor_classes[scls] += 1

            if has_direct:
                connected_direct += 1
                continue

            # 2-hop: check if connected to a non-Other entity through an Other node
            # Check both directions at each hop (undirected projection)
            def _neighbors_of(ent: dict) -> list[str]:
                """All neighbor eids regardless of edge direction."""
                nbrs = []
                for r in ent.get("out_rels", []):
                    nbrs.append(r.get("target", ""))
                for r in ent.get("in_rels", []):
                    nbrs.append(r.get("source", ""))
                return nbrs

            for mid_eid in _neighbors_of(e):
                if eid_to_class.get(mid_eid) != "Other":
                    continue
                mid_entity = eid_to_entity.get(mid_eid)
                if not mid_entity:
                    continue
                for far_eid in _neighbors_of(mid_entity):
                    if far_eid == e["id"]:
                        continue
                    far_cls = eid_to_class.get(far_eid, "Other")
                    if far_cls != "Other" and far_cls != cls:
                        has_2hop = True
                        neighbor_classes[far_cls] += 1
                        break
                if has_2hop:
                    break

            if has_2hop:
                connected_2hop += 1

        total = len(members)
        total_connected = connected_direct + connected_2hop
        frac = total_connected / total if total else 0
        class_connectivity[cls] = {
            "total": total,
            "connected_direct": connected_direct,
            "connected_2hop": connected_2hop,
            "connected_total": total_connected,
            "connectivity": round(frac, 3),
            "top_neighbors": dict(neighbor_classes.most_common(3)),
        }

    # Overall stats
    total_entities = sum(d["total"] for d in class_connectivity.values())
    total_connected = sum(d["connected_total"] for d in class_connectivity.values())
    overall_connectivity = round(total_connected / total_entities, 4) if total_entities else 0

    disconnected = [
        cls for cls, d in class_connectivity.items()
        if d["connectivity"] < 0.10
    ]

    result = {
        "overall_connectivity": overall_connectivity,
        "total_non_other_entities": total_entities,
        "total_connected": total_connected,
        "disconnected_classes": disconnected,
        "per_class": class_connectivity,
    }

    print(f"  Overall connectivity: {overall_connectivity:.1%} "
          f"({total_connected}/{total_entities} entities)")
    if disconnected:
        print(f"  ⚠ Disconnected classes (<10%): {disconnected}")
    for cls, d in sorted(class_connectivity.items(), key=lambda x: x[1]["connectivity"]):
        direct = d["connected_direct"]
        hop2 = d["connected_2hop"]
        print(f"    {cls}: {d['connectivity']:.0%} connected "
              f"({direct} direct + {hop2} 2-hop / {d['total']}), "
              f"neighbors: {d['top_neighbors']}")

    return result


# ---------------------------------------------------------------------------
# Phase 15: Write to Neo4j
# ---------------------------------------------------------------------------

def write_ontology(
    assignments: dict[str, str],
    hierarchy: list[tuple[str, str]],
    associations: list[tuple[str, str]] | None = None,
) -> dict:
    """Write ontology layer to Neo4j."""
    print("\n[Phase 15] Writing ontology to Neo4j...")

    try:
        graph = get_neo4j_graph()
        graph.query("RETURN 1 AS ok")
    except Exception as e:
        print(f"  ⚠ Neo4j unavailable ({e}), skipping write. Results saved to JSON.")
        class_counts = Counter(v for v in assignments.values() if v != "Other")
        return {
            "entities_labeled": len(assignments),
            "ontology_classes": len(class_counts),
            "subclass_of_edges": len(hierarchy),
            "class_names": sorted(class_counts.keys()),
            "class_distribution": dict(class_counts),
            "method": "SV-LOI",
            "neo4j_skipped": True,
        }

    # Clean previous ontology (labels, properties, and OntologyClass nodes)
    try:
        graph.query("MATCH (c:OntologyClass) DETACH DELETE c")
        # Clear ontology_class property from ALL entities (prevents stale labels)
        graph.query("MATCH (n:Entity) WHERE n.ontology_class IS NOT NULL REMOVE n.ontology_class")
        # Remove old ontology labels from entities
        rows = graph.query("CALL db.labels() YIELD label RETURN label")
        existing = {r["label"] for r in rows}
        skip = {"__Entity__", "Document", "Entity", "OntologyClass"}
        for lbl in existing:
            if lbl not in skip and lbl != "Other":
                try:
                    # Backtick-quoted labels handle special characters safely
                    graph.query(f"MATCH (n:`{lbl}`) REMOVE n:`{lbl}`")
                except Exception:
                    pass
    except Exception as e:
        print(f"  Warning: cleanup error: {e}")

    # Get unique classes (excluding Other)
    class_counts = Counter(v for v in assignments.values() if v != "Other")
    class_names = sorted(class_counts.keys())

    # Create OntologyClass nodes
    for cls in class_names:
        safe = _sanitize_label(cls)
        try:
            graph.query(
                "MERGE (c:OntologyClass {name: $name})",
                params={"name": safe},
            )
        except Exception as e:
            print(f"  Warning: could not create class {safe}: {e}")

    # Label entities with their assigned class
    entities_labeled = 0
    for cls in class_names:
        safe = _sanitize_label(cls)
        members = [eid for eid, c in assignments.items() if c == cls]
        if not members:
            continue
        for batch_start in range(0, len(members), 100):
            batch = members[batch_start:batch_start + 100]
            try:
                graph.query(
                    f"MATCH (n:Entity) WHERE n.id IN $ids "
                    f"SET n:`{safe}`, n.ontology_class = $cls",
                    params={"ids": batch, "cls": safe},
                )
                entities_labeled += len(batch)
            except Exception as e:
                print(f"  Warning: labeling batch for {safe}: {e}")

    # Create INSTANCE_OF edges: entity → OntologyClass
    # This bridges record entities (POL-xxx) and concept entities ("Renters Insurance")
    # through their shared class node, enabling cross-subgraph query traversal.
    instance_of_created = 0
    for cls in class_names:
        safe = _sanitize_label(cls)
        members = [eid for eid, c in assignments.items() if c == cls]
        if not members:
            continue
        for batch_start in range(0, len(members), 200):
            batch = members[batch_start:batch_start + 200]
            try:
                graph.query(
                    "MATCH (n:Entity), (c:OntologyClass {name: $cls}) "
                    "WHERE n.id IN $ids "
                    "MERGE (n)-[:INSTANCE_OF]->(c)",
                    params={"ids": batch, "cls": safe},
                )
                instance_of_created += len(batch)
            except Exception as e:
                print(f"  Warning: INSTANCE_OF batch for {safe}: {e}")

    print(f"  ✓ INSTANCE_OF edges:    {instance_of_created}")

    # Create SUBCLASS_OF edges
    subclass_created = 0
    for child, parent in hierarchy:
        child_safe = _sanitize_label(child)
        parent_safe = _sanitize_label(parent)
        try:
            graph.query(
                "MATCH (c:OntologyClass {name: $child}), (p:OntologyClass {name: $parent}) "
                "MERGE (c)-[:SUBCLASS_OF]->(p)",
                params={"child": child_safe, "parent": parent_safe},
            )
            subclass_created += 1
        except Exception as e:
            print(f"  Warning: SUBCLASS_OF {child} → {parent}: {e}")

    # Create ASSOCIATED_WITH edges (inter-class associations, NOT IS-A)
    assoc_created = 0
    for src, tgt in (associations or []):
        src_safe = _sanitize_label(src)
        tgt_safe = _sanitize_label(tgt)
        try:
            graph.query(
                "MATCH (a:OntologyClass {name: $src}), (b:OntologyClass {name: $tgt}) "
                "MERGE (a)-[:ASSOCIATED_WITH]->(b)",
                params={"src": src_safe, "tgt": tgt_safe},
            )
            assoc_created += 1
        except Exception as e:
            print(f"  Warning: ASSOCIATED_WITH {src} → {tgt}: {e}")

    stats = {
        "entities_labeled": entities_labeled,
        "ontology_classes": len(class_names),
        "instance_of_edges": instance_of_created,
        "subclass_of_edges": subclass_created,
        "associated_with_edges": assoc_created,
        "class_names": class_names,
        "class_distribution": dict(class_counts),
        "method": "SV-LOI",
    }

    print(f"  ✓ Entities labeled:     {entities_labeled}")
    print(f"  ✓ OntologyClass nodes:  {len(class_names)}")
    print(f"  ✓ INSTANCE_OF edges:    {instance_of_created}")
    print(f"  ✓ SUBCLASS_OF edges:    {subclass_created}")
    print(f"  ✓ ASSOCIATED_WITH edges:{assoc_created}")
    return stats


def _flush_print(msg: str) -> None:
    """Print with immediate flush for Slurm log visibility."""
    print(msg, flush=True)


def _compute_intrinsic_quality(
    assignments: dict[str, str],
    entities: list[dict],
    entity_map: dict[str, dict],
    class_dist: Counter,
) -> dict[str, Any]:
    """Compute reference-free ontology quality metrics.

    Metrics:
    - connectivity: fraction of non-Other entities with >= 1 typed neighbor
    - completeness: fraction of classes with >= 1 inter-class relationship
    - balance: 1 - max(class_fraction), higher is more balanced
    - other_fraction: fraction of entities classified as "Other"
    - backbone_coverage: fraction of record entities linked to a concept entity
    """
    total = len(assignments)
    if total == 0:
        return {"connectivity": 0, "completeness": 0, "balance": 0,
                "other_fraction": 1.0, "backbone_coverage": 0}

    # Other fraction
    other_count = sum(1 for c in assignments.values() if c == "Other")
    other_frac = other_count / total

    # Balance: 1 - largest class fraction (among non-Other)
    non_other_total = total - other_count
    if non_other_total > 0 and class_dist:
        max_class_frac = class_dist.most_common(1)[0][1] / non_other_total
        balance = round(1.0 - max_class_frac, 4)
    else:
        balance = 0.0

    # Connectivity: fraction of non-Other entities with at least 1 neighbor
    # that has a different non-Other class (inter-class link)
    connected = 0
    non_other_entities = [e for e in entities if assignments.get(e["id"]) not in (None, "Other")]
    for e in non_other_entities:
        eid = e["id"]
        my_class = assignments[eid]
        neighbors = set()
        for rel in e.get("out_rels", []):
            tid = rel.get("target", "")
            tcls = assignments.get(tid)
            if tcls and tcls != "Other" and tcls != my_class:
                neighbors.add(tcls)
        for rel in e.get("in_rels", []):
            sid = rel.get("source", "")
            scls = assignments.get(sid)
            if scls and scls != "Other" and scls != my_class:
                neighbors.add(scls)
        if neighbors:
            connected += 1

    connectivity = round(connected / len(non_other_entities), 4) if non_other_entities else 0.0

    # Completeness: fraction of non-Other classes with >= 1 inter-class edge
    classes_with_interclass = set()
    for e in entities:
        eid = e["id"]
        my_class = assignments.get(eid, "Other")
        if my_class == "Other":
            continue
        for rel in e.get("out_rels", []):
            tid = rel.get("target", "")
            tcls = assignments.get(tid, "Other")
            if tcls != "Other" and tcls != my_class:
                classes_with_interclass.add(my_class)
                classes_with_interclass.add(tcls)
                break

    n_classes = len(class_dist)
    completeness = round(len(classes_with_interclass) / n_classes, 4) if n_classes > 0 else 0.0

    # Backbone coverage: fraction of record entities (POL-, CLM-) connected
    # to a concept entity (non-record, non-Other), directly or via 2-hop
    from zone3.graph_cache import STRUCTURED_PREFIXES
    record_entities = [e for e in entities if e["id"].startswith(STRUCTURED_PREFIXES)]
    records_linked = 0
    for e in record_entities:
        linked = False
        # Direct check
        for rel in e.get("out_rels", []):
            tid = rel.get("target", "")
            if not tid.startswith(STRUCTURED_PREFIXES):
                tcls = assignments.get(tid, "Other")
                if tcls != "Other":
                    linked = True
                    break
        if not linked:
            for rel in e.get("in_rels", []):
                sid = rel.get("source", "")
                if not sid.startswith(STRUCTURED_PREFIXES):
                    scls = assignments.get(sid, "Other")
                    if scls != "Other":
                        linked = True
                        break
        # 2-hop: record → Other value → concept (undirected at each hop)
        if not linked:
            # Collect all neighbors (both directions)
            e_neighbors = set()
            for rel in e.get("out_rels", []):
                e_neighbors.add(rel.get("target", ""))
            for rel in e.get("in_rels", []):
                e_neighbors.add(rel.get("source", ""))
            for mid in e_neighbors:
                if assignments.get(mid) != "Other":
                    continue
                mid_e = entity_map.get(mid, {})
                mid_neighbors = set()
                for rel2 in mid_e.get("out_rels", []):
                    mid_neighbors.add(rel2.get("target", ""))
                for rel2 in mid_e.get("in_rels", []):
                    mid_neighbors.add(rel2.get("source", ""))
                for far in mid_neighbors:
                    if far != e["id"] and assignments.get(far, "Other") != "Other":
                        linked = True
                        break
                if linked:
                    break
        if linked:
            records_linked += 1

    backbone = round(records_linked / len(record_entities), 4) if record_entities else 0.0

    return {
        "connectivity": connectivity,
        "completeness": completeness,
        "balance": balance,
        "other_fraction": round(other_frac, 4),
        "backbone_coverage": backbone,
        "record_entities_total": len(record_entities),
        "record_entities_linked": records_linked,
    }
