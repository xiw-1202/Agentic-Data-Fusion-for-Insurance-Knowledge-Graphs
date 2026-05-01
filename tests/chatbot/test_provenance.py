from unittest.mock import MagicMock
from chatbot.qa_chain import fetch_provenance

def test_fetch_provenance_returns_chunks_for_entity_ids_in_rows():
    graph = MagicMock()
    graph.query.return_value = [
        {"entity_id": "claim_42", "chunk_id": "c7", "source": "claims.csv"},
        {"entity_id": "device_12", "chunk_id": "c7", "source": "claims.csv"},
        {"entity_id": "claim_42", "chunk_id": "c9", "source": "policy.csv"},
    ]
    rows = [{"claim": "claim_42", "device_type": "device_12", "n": 5}]
    sources = fetch_provenance(graph, rows)
    assert {"chunk_id": "c7", "source": "claims.csv"} in sources
    assert {"chunk_id": "c9", "source": "policy.csv"} in sources
    assert len(sources) == 2  # deduplicated
