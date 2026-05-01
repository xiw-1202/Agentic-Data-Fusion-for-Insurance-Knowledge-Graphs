from chatbot.eval.verify import extract_rel_types


def test_extract_rel_types_finds_has_relations():
    cypher = """
    MATCH (c:Entity)-[:HAS_TOTAL_CLAIM_TIME]->(t:Entity)
    OPTIONAL MATCH (c)-[:HAS_LOSS_TYPE]->(lt:Entity)
    RETURN c.id, t.id
    """
    rels = extract_rel_types(cypher)
    assert rels == ["HAS_LOSS_TYPE", "HAS_TOTAL_CLAIM_TIME"]


def test_extract_rel_types_ignores_non_has_relations():
    cypher = "MATCH (e)-[:INSTANCE_OF]->(c) RETURN e"
    assert extract_rel_types(cypher) == []
