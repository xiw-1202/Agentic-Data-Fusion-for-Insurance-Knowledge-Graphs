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
        [{"src": "Policy", "tgt": "Coverage"}],  # NEW: associations query
        [{"cls": "Policy", "props": ["id", "policy_number", "effective_date"]}],  # NEW
    ]
    out = summarize_schema(graph)
    assert "Policy ASSOCIATED_WITH Coverage" in out
    assert "Policy {id, policy_number, effective_date}" in out
