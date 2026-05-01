"""Introspect the Emory KG schema for the Cypher-generation prompt."""
from __future__ import annotations

from langchain_neo4j import Neo4jGraph


SCHEMA_QUERY = """
CALL () {
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
    the :OntologyClass ontology (names + SUBCLASS_OF edges).
    """
    rows = graph.query(SCHEMA_QUERY)
    labels = [r for r in rows if r["kind"] == "labels"]
    rels = [r for r in rows if r["kind"] == "rels"]

    classes = graph.query("MATCH (c:OntologyClass) RETURN c.name AS name ORDER BY c.name")
    class_names = [c["name"] for c in classes]

    hierarchy = graph.query(
        """
        MATCH (c:OntologyClass)-[:SUBCLASS_OF]->(p:OntologyClass)
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

    associations = graph.query(
        """
        MATCH (s:OntologyClass)-[:ASSOCIATED_WITH]->(t:OntologyClass)
        RETURN s.name AS src, t.name AS tgt
        """
    )

    class_props = graph.query(
        """
        MATCH (e:Entity)-[:INSTANCE_OF]->(c:OntologyClass)
        WITH c.name AS cls, e LIMIT 5000
        UNWIND keys(e) AS k
        WITH cls, k, count(*) AS n
        WHERE k <> 'id' AND n >= 3
        RETURN cls, collect(DISTINCT k)[..6] AS props
        ORDER BY cls
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
    lines.append("## Ontology classes (:OntologyClass)")
    for cn in class_names:
        lines.append(f"  {cn}")
    if hierarchy:
        lines.append("")
        lines.append("## SUBCLASS_OF hierarchy")
        for h in hierarchy:
            lines.append(f"  {h['child']} -> {h['parent']}")

    if associations:
        lines.append("")
        lines.append("## ASSOCIATED_WITH edges (class-level wiring)")
        for a in associations:
            lines.append(f"  {a['src']} ASSOCIATED_WITH {a['tgt']}")

    if class_props:
        lines.append("")
        lines.append("## Sample property keys per class")
        for cp in class_props:
            lines.append(f"  {cp['cls']} {{{', '.join(cp['props'])}}}")

    lines.append("")
    lines.append(f"## Relation types (top {top_rels} by frequency)")
    for r in rels[:top_rels]:
        lines.append(f"  -[:{r['value']}]-> ({r['n']})")

    lines.append("")
    lines.append("## Special edges")
    lines.append("  (:Entity)-[:INSTANCE_OF]->(:OntologyClass)")
    lines.append("  (:OntologyClass)-[:SUBCLASS_OF]->(:OntologyClass)")
    lines.append("  (:OntologyClass)-[:ASSOCIATED_WITH]->(:OntologyClass)")

    return "\n".join(lines)
