"""Hand-curated few-shot English -> Cypher pairs for the Emory KG.

Doubles as the 'Try one of these' demo deck in the Streamlit sidebar.
The order tells a story; ask them top-to-bottom during a presentation:

  1. Ontology exists  -> SV-LOI induced classes
  2. Ontology is populated  -> entities map to classes
  3. Categorical filter  -> CSV columns become queryable enums
  4. Aggregation & chart  -> KG drives bar charts
  5. Multi-relation join  -> NPS x channel
  6. Multi-source date  -> mold claim dates (T-Mobile vs GEICO date schema)
  7. Cross-type retrieval  -> claim org vs survey org are *different*
  8. Policy text grounding  -> hybrid GraphRAG quotes actual policy language

Every Cypher here was verified against the live Emory KG.
"""
from __future__ import annotations

EXAMPLES: list[dict[str, str]] = [
    # 1. The KG has structure: SV-LOI induced an ontology.
    {
        "question": "Show the ontology hierarchy.",
        "cypher": """
MATCH (child:OntologyClass)-[:SUBCLASS_OF]->(parent:OntologyClass)
RETURN child.name AS child, parent.name AS parent
""".strip(),
    },
    # 2. The classes are populated: every entity has a class assignment.
    {
        "question": "How many entities are in each ontology class?",
        "cypher": """
MATCH (e:Entity)-[:INSTANCE_OF]->(c:OntologyClass)
RETURN c.name AS class_name, count(e) AS entity_count
ORDER BY entity_count DESC
""".strip(),
    },
    # 3. CSV columns become queryable categorical enums.
    {
        "question": "What cause-of-loss values appear in GEICO renters claims?",
        "cypher": """
MATCH (c:Entity)-[:HAS_CAUSE_OF_LOSS]->(cause:Entity)
WHERE c.entity_type = 'ClaimRecord'
RETURN cause.id AS cause_of_loss, count(c) AS claim_count
ORDER BY claim_count DESC
""".strip(),
    },
    # 4. Aggregation + chart — the chatbot picks bar viz automatically.
    {
        "question": "Which device types have the most T-Mobile claims?",
        "cypher": """
MATCH (c:Entity)-[:HAS_DEVICE_TYPE]->(d:Entity)
WHERE c.entity_type = 'ClaimRecord'
RETURN d.id AS device_type, count(c) AS claim_count
ORDER BY claim_count DESC
LIMIT 10
""".strip(),
    },
    # 5. Multi-relation join + numeric aggregation.
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
    # 6. Multi-source date handling — GEICO and T-Mobile use different
    #    date relations even though both are :ClaimRecord.  COALESCE
    #    surfaces whichever exists per claim.
    {
        "question": "When did mold damage claims happen?",
        "cypher": """
MATCH (c:Entity)-[:HAS_CAUSE_OF_LOSS]->(:Entity {id: 'MOLD'})
OPTIONAL MATCH (c)-[:HAS_CLAIM_LOSS_DATE]->(d1)
OPTIONAL MATCH (c)-[:HAS_FISCAL_PMS_ACCOUNT_DATE]->(d2)
OPTIONAL MATCH (c)-[:HAS_CLAIM_OPEN_DATE]->(d3)
RETURN c.id AS claim_id,
       COALESCE(d1.id, d2.id, d3.id) AS date
""".strip(),
    },
    # 7. Cross-type retrieval — claim records and survey records use
    #    DIFFERENT organization relations (HAS_MASTER_NAME vs
    #    HAS_ORGANIZATION_NAME) and reference different orgs.  The
    #    class-aware retrieval pulls both sides.
    {
        "question": "Do claim records and survey records reference the same organization?",
        "cypher": """
MATCH (c:Entity)-[:HAS_MASTER_NAME]->(claim_org:Entity)
WHERE c.entity_type = 'ClaimRecord'
WITH collect(DISTINCT claim_org.id) AS claim_orgs
MATCH (s:Entity)-[:HAS_ORGANIZATION_NAME]->(survey_org:Entity)
WHERE s.entity_type IN ['SurveyRecord', 'ClientJourneySurvey']
RETURN claim_orgs,
       collect(DISTINCT survey_org.id) AS survey_orgs
""".strip(),
    },
    # 8. Policy text grounding — the open-interpretive path quotes
    #    actual policy paragraphs (hybrid GraphRAG).  This question
    #    can be answered structurally too, via :EXCLUDED_FROM edges.
    {
        "question": "What is excluded from Personal Liability Coverage?",
        "cypher": """
MATCH (c:Entity)-[:EXCLUDED_FROM|HAS_EXCLUSION]->(x:Entity)
WHERE toLower(c.id) CONTAINS 'personal liability'
   OR toLower(c.id) CONTAINS 'liability coverage'
RETURN c.id AS coverage, x.id AS exclusion
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
