"""Record decomposition — split record entities into ontology-class components."""
from __future__ import annotations

from collections import defaultdict

from zone3._svloi.utils import _sanitize_label, _sanitize_rel_type, get_neo4j_graph
from zone3.graph_cache import get_entity_lane


def decompose_records(
    assignments: dict[str, str],
    entities: list[dict],
    rel_to_class: dict[str, tuple[str, float]] | None = None,
) -> dict[str, dict[str, dict]]:
    """Decompose record entities into domain-specific sub-nodes.

    Uses the relation-range mapping (from value typing) to determine which
    fields of a record entity belong to which concept class. Creates a
    decomposition plan that Stage 7 can use to write sub-nodes.

    For each record entity (POL-xxx, CLM-xxx):
      - Look up its outgoing HAS_* relations
      - Group relations by the class their targets belong to
      - Create sub-node plan: {record_id: {class_name: {fields: {rel: value}}}}

    Args:
        assignments: entity → class mapping (after full typing)
        entities: all entities with relation data
        rel_to_class: relation→(class, confidence) from value typing (optional)

    Returns:
        {record_eid: {class_label: {"fields": {rel: target_eid}, "class": class_name}}}
        Empty dict if no decomposition is applicable.
    """
    print("\n[Decompose] Record decomposition via relation-range mapping...", flush=True)

    entity_map = {e["id"]: e for e in entities}
    decomposition: dict[str, dict[str, dict]] = {}
    records_processed = 0
    sub_nodes_planned = 0

    for e in entities:
        eid = e["id"]
        lane = get_entity_lane(e)
        if lane != "record":
            continue

        # Group outgoing relations by target's class
        class_fields: dict[str, dict[str, str]] = defaultdict(dict)

        for rel in e.get("out_rels", []):
            rel_type = rel.get("rel", "")
            target_eid = rel.get("target", "")

            # Only decompose HAS_* relations — these are property fields
            # from CSV columns. Structural relations (BELONGS_TO, IS_A,
            # ABOUT, INSTANCE_OF, COVERS, etc.) are real entity-to-entity
            # links that should stay on the hub record, not be decomposed.
            if not rel_type.startswith("HAS_"):
                continue

            target_cls = assignments.get(target_eid, "Other")

            if target_cls == "Other":
                # Try relation-range mapping as fallback
                if rel_to_class and rel_type in rel_to_class:
                    target_cls = rel_to_class[rel_type][0]
                else:
                    continue

            # Skip self-class (record's own class)
            record_cls = assignments.get(eid, "Other")
            if target_cls == record_cls:
                continue

            class_fields[target_cls][rel_type] = target_eid

        # Only decompose if fields map to 2+ distinct classes
        if len(class_fields) >= 2:
            sub_plan: dict[str, dict] = {}
            seen_labels: set[str] = set()
            for cls, fields in class_fields.items():
                # Create a lowercase label for the sub-node
                label = cls.lower()
                # Guard against label collisions (e.g., "Coverage" vs "COVERAGE")
                if label in seen_labels:
                    label = f"{label}_{len(seen_labels)}"
                seen_labels.add(label)
                sub_plan[label] = {
                    "fields": dict(fields),
                    "class": cls,
                }
            decomposition[eid] = sub_plan
            sub_nodes_planned += len(sub_plan)
            records_processed += 1

    total_records = sum(1 for e in entities if get_entity_lane(e) == "record")
    skipped = total_records - records_processed
    print(f"  ✓ {records_processed} records decomposed into {sub_nodes_planned} sub-nodes "
          f"({skipped} skipped: <2 distinct target classes)")
    if decomposition:
        # Show a sample
        sample_eid = next(iter(decomposition))
        sample = decomposition[sample_eid]
        print(f"  Example: {sample_eid} → {list(sample.keys())}")

    return decomposition


def write_record_decomposition(
    decomposition: dict[str, dict[str, dict]],
) -> int:
    """Write decomposed record sub-nodes to Neo4j.

    For each record entity with a decomposition plan:
      1. Create sub-node: {record_id}:{class_label} as Entity
      2. Link sub-node → OntologyClass via INSTANCE_OF
      3. Link hub record → sub-node via HAS_COMPONENT

    Returns:
        Number of sub-nodes created.
    """
    if not decomposition:
        return 0

    print(f"\n[Decompose-Write] Writing {sum(len(v) for v in decomposition.values())} "
          f"sub-nodes to Neo4j...", flush=True)

    try:
        graph = get_neo4j_graph()
        graph.query("RETURN 1 AS ok")
    except Exception as e:
        print(f"  ⚠ Neo4j unavailable ({e}), skipping decomposition write.")
        return 0

    created = 0
    for record_eid, sub_plan in decomposition.items():
        for label, info in sub_plan.items():
            sub_id = f"{record_eid}:{label}"
            cls = info["class"]
            safe_cls = _sanitize_label(cls)

            try:
                # Create sub-node entity
                graph.query(
                    "MERGE (n:Entity {id: $id}) "
                    "SET n.entity_type = $cls, n.ontology_class = $cls, "
                    "n.decomposed_from = $parent",
                    params={"id": sub_id, "cls": safe_cls, "parent": record_eid},
                )

                # Link to ontology class
                graph.query(
                    "MATCH (n:Entity {id: $id}), (c:OntologyClass {name: $cls}) "
                    "MERGE (n)-[:INSTANCE_OF]->(c)",
                    params={"id": sub_id, "cls": safe_cls},
                )

                # Link hub record → sub-node
                graph.query(
                    "MATCH (hub:Entity {id: $hub}), (sub:Entity {id: $sub}) "
                    "MERGE (hub)-[:HAS_COMPONENT]->(sub)",
                    params={"hub": record_eid, "sub": sub_id},
                )

                # Copy relevant field relations to the sub-node
                for rel_type, target_eid in info["fields"].items():
                    safe_rel = _sanitize_rel_type(rel_type)
                    graph.query(
                        "MATCH (sub:Entity {id: $sub}), (tgt:Entity {id: $tgt}) "
                        f"MERGE (sub)-[:`{safe_rel}`]->(tgt)",
                        params={"sub": sub_id, "tgt": target_eid},
                    )

                created += 1
            except Exception as e:
                print(f"  ⚠ Sub-node {sub_id}: {e}")

    print(f"  ✓ Created {created} sub-nodes with HAS_COMPONENT + INSTANCE_OF edges")
    return created
