"""Phase 4: Entity assignment and Neo4j materialization.

1. build_file_to_class_map — map source filenames to most specific class
2. assign_entity_to_class — assign single entity via source file majority vote
3. assign_all_entities — batch assignment for all entities
4. write_ontology_to_neo4j — materialize ontology + assignments in Neo4j
"""

from __future__ import annotations

import os
from collections import Counter

import config
from zone3.fbi.class_discovery import CandidateClass
from zone3.fbi.relationships import ClassRelationship


# ---------------------------------------------------------------------------
# File-to-class mapping
# ---------------------------------------------------------------------------


def build_file_to_class_map(
    classes: list[CandidateClass],
) -> dict[str, str]:
    """Build mapping from source filename (basename) to most specific class name.

    Children (subclasses) override parents because they are more specific.
    Recurses into nested children.
    """
    mapping: dict[str, str] = {}

    def _map(cls: CandidateClass) -> None:
        name = cls.name or cls.prefix
        for f in cls.source_files:
            basename = os.path.basename(f)
            mapping[basename] = name
        # Children override parent mapping (more specific)
        for child in cls.children:
            _map(child)

    for cls in classes:
        _map(cls)

    return mapping


# ---------------------------------------------------------------------------
# Entity assignment
# ---------------------------------------------------------------------------


def assign_entity_to_class(
    entity: dict,
    file_to_class: dict[str, str],
) -> str:
    """Assign a single entity to a class based on its source files.

    If the entity appears in files mapping to multiple classes,
    picks the most common class (majority vote).

    Args:
        entity: Dict with 'id' and 'source_files' (set or list of filenames).
        file_to_class: Mapping from basename to class name.

    Returns:
        Class name string, or "Unclassified" if no match.
    """
    source_files = entity.get("source_files", set())
    if not source_files:
        return "Unclassified"

    class_counts: Counter = Counter()
    for f in source_files:
        basename = os.path.basename(f)
        cls = file_to_class.get(basename)
        if cls:
            class_counts[cls] += 1

    if not class_counts:
        return "Unclassified"

    return class_counts.most_common(1)[0][0]


def assign_all_entities(
    entities: list[dict],
    classes: list[CandidateClass],
) -> dict[str, str]:
    """Assign all entities to classes.

    Args:
        entities: List of entity dicts, each with 'id' and 'source_files'.
        classes: Discovered class hierarchy.

    Returns:
        Dict mapping entity_id -> class_name.
    """
    file_map = build_file_to_class_map(classes)
    assignments: dict[str, str] = {}

    for entity in entities:
        eid = entity.get("id", "")
        cls = assign_entity_to_class(entity, file_map)
        assignments[eid] = cls

    return assignments


# ---------------------------------------------------------------------------
# Neo4j materialization
# ---------------------------------------------------------------------------


def write_ontology_to_neo4j(
    classes: list[CandidateClass],
    relationships: list[ClassRelationship],
    entity_assignments: dict[str, str],
    filename_tokens: dict[str, list[str]],
    driver=None,
) -> dict:
    """Write the complete ontology to Neo4j.

    Creates OntologyClass nodes, SUBCLASS_OF edges, RELATES_TO edges,
    and sets ontology_class on Entity nodes.

    Args:
        classes: Discovered class hierarchy.
        relationships: Inter-class relationships from bridge columns.
        entity_assignments: Mapping entity_id -> class_name.
        filename_tokens: Mapping filename -> [semantic tokens] for LOB tagging.
        driver: Neo4j driver instance. If None, creates one from config.

    Returns:
        Summary dict with counts of created nodes/edges.
    """
    if driver is None:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
        )

    stats = {
        "classes_created": 0,
        "subclass_edges": 0,
        "relationship_edges": 0,
        "entities_labeled": 0,
    }

    with driver.session(database=config.NEO4J_DATABASE) as session:
        # 1. Create OntologyClass nodes (recursive)
        def _create_class(
            cls: CandidateClass, parent_name: str | None = None
        ) -> None:
            name = cls.name or cls.prefix
            session.run(
                """
                MERGE (c:OntologyClass {name: $name})
                SET c.level = $level,
                    c.source_files = $source_files,
                    c.header_count = $header_count
                """,
                name=name,
                level=cls.level,
                source_files=cls.source_files,
                header_count=len(cls.headers),
            )
            stats["classes_created"] += 1

            if parent_name:
                session.run(
                    """
                    MATCH (child:OntologyClass {name: $child_name})
                    MATCH (parent:OntologyClass {name: $parent_name})
                    MERGE (child)-[:SUBCLASS_OF]->(parent)
                    """,
                    child_name=name,
                    parent_name=parent_name,
                )
                stats["subclass_edges"] += 1

            for child in cls.children:
                _create_class(child, name)

        for cls in classes:
            _create_class(cls)

        # 2. Create inter-class relationship edges
        for rel in relationships:
            if rel.relationship_name:
                session.run(
                    """
                    MATCH (a:OntologyClass {name: $source})
                    MATCH (b:OntologyClass {name: $target})
                    MERGE (a)-[r:RELATES_TO {name: $rel_name}]->(b)
                    SET r.bridge_column = $bridge
                    """,
                    source=rel.source_class,
                    target=rel.target_class,
                    rel_name=rel.relationship_name,
                    bridge=rel.bridge_column,
                )
                stats["relationship_edges"] += 1

        # 3. Label entities with ontology_class
        for entity_id, class_name in entity_assignments.items():
            session.run(
                """
                MATCH (e:Entity {id: $eid})
                SET e.ontology_class = $cls
                """,
                eid=entity_id,
                cls=class_name,
            )
            stats["entities_labeled"] += 1

    return stats
