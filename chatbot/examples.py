"""Hand-curated few-shot English -> Cypher pairs for the Emory KG.

Based on the Slide 9 showcase queries plus a few coverage questions.
Every query here was verified against the Emory schema
(data/results/emory/zone2_run_summary.json relation types).
"""
from __future__ import annotations

EXAMPLES: list[dict[str, str]] = [
    {
        "question": "Which claims had the longest resolution time?",
        "cypher": """
MATCH (c:Entity)-[:HAS_TIME_TO_RESOLVE_HR]->(h:Entity)
WHERE c.entity_type = 'ClaimRecord'
WITH c, toFloat(h.id) AS hours
ORDER BY hours DESC
LIMIT 10
OPTIONAL MATCH (c)-[:HAS_LOSS_TYPE]->(lt:Entity)
OPTIONAL MATCH (c)-[:HAS_DEVICE_TYPE]->(dt:Entity)
RETURN c.id AS claim, hours, lt.id AS loss_type, dt.id AS device_type
""".strip(),
    },
    {
        "question": "What device types have the most claims?",
        "cypher": """
MATCH (c:Entity)-[:HAS_DEVICE_TYPE]->(d:Entity)
WHERE c.entity_type = 'ClaimRecord'
RETURN d.id AS device_type, count(c) AS claim_count
ORDER BY claim_count DESC
LIMIT 10
""".strip(),
    },
    {
        "question": "What loss types are most common?",
        "cypher": """
MATCH (c:Entity)-[:HAS_LOSS_TYPE]->(l:Entity)
WHERE c.entity_type = 'ClaimRecord'
RETURN l.id AS loss_type, count(c) AS claim_count
ORDER BY claim_count DESC
""".strip(),
    },
    {
        "question": "What's the average NPS score by claim channel?",
        "cypher": """
MATCH (c:Entity)-[:HAS_CLAIM_CHANNEL]->(ch:Entity),
      (c)-[:HAS_NPS_SCORE]->(n:Entity)
WHERE c.entity_type = 'ClaimRecord'
RETURN ch.id AS channel,
       avg(toFloat(n.id)) AS avg_nps,
       count(c) AS claims
ORDER BY avg_nps DESC
""".strip(),
    },
    {
        "question": "How many entities are classified into each ontology class?",
        "cypher": """
MATCH (e:Entity)-[:INSTANCE_OF]->(c:OntologyClass)
RETURN c.name AS class_name, count(e) AS entity_count
ORDER BY entity_count DESC
""".strip(),
    },
    {
        "question": "Show the ontology hierarchy.",
        "cypher": """
MATCH (child:OntologyClass)-[:SUBCLASS_OF]->(parent:OntologyClass)
RETURN child.name AS child, parent.name AS parent
""".strip(),
    },
    {
        "question": "Which organizations appear in the claims data?",
        "cypher": """
MATCH (c:Entity)-[:HAS_ORGANIZATION_NAME]->(o:Entity)
WHERE c.entity_type = 'ClaimRecord'
RETURN o.id AS organization, count(c) AS claim_count
ORDER BY claim_count DESC
""".strip(),
    },
    {
        "question": "What are the top cause-of-loss reasons?",
        "cypher": """
MATCH (c:Entity)-[:HAS_CAUSE_OF_LOSS]->(cause:Entity)
WHERE c.entity_type = 'ClaimRecord'
RETURN cause.id AS cause_of_loss, count(c) AS claim_count
ORDER BY claim_count DESC
LIMIT 10
""".strip(),
    },
]


def format_for_prompt() -> str:
    parts = ["# Few-shot examples (English question -> Cypher)"]
    for i, ex in enumerate(EXAMPLES, 1):
        parts.append(f"\n## Example {i}")
        parts.append(f"Question: {ex['question']}")
        parts.append("Cypher:")
        parts.append("```cypher")
        parts.append(ex["cypher"])
        parts.append("```")
    return "\n".join(parts)
