from unittest.mock import MagicMock
from chatbot.schema import summarize_schema

def test_schema_includes_associated_with_edges():
    graph = MagicMock()
    graph.query.side_effect = [
        [{"kind": "labels", "value": "Entity", "n": 1351},
         {"kind": "rels", "value": "HAS_COVERAGE", "n": 200}],
        [{"name": "Policy"}, {"name": "Coverage"}],
        [{"child": "Coverage", "parent": "Policy"}],
        [{"t": "ClaimRecord", "n": 50}],
        [{"src": "Policy", "tgt": "Coverage"}],
        [{"cls": "Policy", "props": ["id", "policy_number", "effective_date"]}],
        [{"cls": "Organization", "samples": ["T-Mobile", "Verizon", "AT&T"], "total": 3}],
        [{"rel_type": "HAS_LINE_OF_BUSINESS", "values": ["Auto", "Mobile", "Home"]}],
    ]
    out = summarize_schema(graph)
    assert "Policy ASSOCIATED_WITH Coverage" in out
    assert "Policy {id, policy_number, effective_date}" in out
    # Sample entity values per class
    assert "Organization (3 total): T-Mobile, Verizon, AT&T" in out
    # Categorical enum values
    assert "-[:HAS_LINE_OF_BUSINESS]-> {Auto, Mobile, Home}" in out
