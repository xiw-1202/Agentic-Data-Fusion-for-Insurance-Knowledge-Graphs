"""Introspect the Emory KG schema for the Cypher-generation prompt."""
from __future__ import annotations

from langchain_neo4j import Neo4jGraph


SCHEMA_QUERY = """
CALL {
    MATCH (n) UNWIND labels(n) AS label
    RETURN 'labels' AS kind, label AS value, count(*) AS n
    UNION
    MATCH ()-[r]->() RETURN 'rels' AS kind, type(r) AS value, count(*) AS n
}
RETURN kind, value, n ORDER BY kind, n DESC
"""


def summarize_schema(graph: Neo4jGraph, top_rels: int = 60) -> str:
    """Return a compact schema summary for the LLM prompt.

    Lists all node labels and the top-N relation types by frequency, plus
    the :Class ontology (names + SUBCLASS_OF edges).
    """
    rows = graph.query(SCHEMA_QUERY)
    labels = [r for r in rows if r["kind"] == "labels"]
    rels = [r for r in rows if r["kind"] == "rels"]

    classes = graph.query("MATCH (c:Class) RETURN c.name AS name ORDER BY c.name")
    class_names = [c["name"] for c in classes]

    hierarchy = graph.query(
        """
        MATCH (c:Class)-[:SUBCLASS_OF]->(p:Class)
        RETURN c.name AS child, p.name AS parent
        """
    )

    entity_types = graph.query(
        """
        MATCH (e:Entity)
        WITH e.entity_type AS t, count(*) AS n
        WHERE t IS NOT NULL
        RETURN t, n ORDER BY n DESC LIMIT 30
        """
    )

    lines: list[str] = [
        "# Neo4j Knowledge Graph schema",
        "",
        "## Node labels",
    ]
    for r in labels:
        lines.append(f"  :{r['value']} ({r['n']} nodes)")

    lines.append("")
    lines.append("## Top entity_type values on :Entity nodes")
    for et in entity_types:
        lines.append(f"  {et['t']} ({et['n']})")

    lines.append("")
    lines.append("## Ontology classes (:Class)")
    for cn in class_names:
        lines.append(f"  {cn}")
    if hierarchy:
        lines.append("")
        lines.append("## SUBCLASS_OF hierarchy")
        for h in hierarchy:
            lines.append(f"  {h['child']} -> {h['parent']}")

    lines.append("")
    lines.append(f"## Relation types (top {top_rels} by frequency)")
    for r in rels[:top_rels]:
        lines.append(f"  -[:{r['value']}]-> ({r['n']})")

    lines.append("")
    lines.append("## Special edges")
    lines.append("  (:Entity)-[:INSTANCE_OF]->(:Class)")
    lines.append("  (:Class)-[:SUBCLASS_OF]->(:Class)")
    lines.append("  (:Class)-[:ASSOCIATED_WITH]->(:Class)")

    return "\n".join(lines)
