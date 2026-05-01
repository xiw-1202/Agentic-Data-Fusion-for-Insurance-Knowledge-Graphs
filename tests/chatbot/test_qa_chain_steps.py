# tests/chatbot/test_qa_chain_steps.py
from unittest.mock import MagicMock, patch
from chatbot.qa_chain import ask_stream, Step

@patch("chatbot.qa_chain._client")
@patch("chatbot.qa_chain.classify_question")
def test_ask_stream_yields_ordered_steps(mock_classify, mock_client):
    from chatbot.classifier import Classification, QuestionKind
    mock_classify.return_value = Classification(QuestionKind.FACTUAL_KG, "counts", 0.95)

    cypher_msg = MagicMock(); cypher_msg.content = [MagicMock(text="```cypher\nMATCH (n) RETURN count(n) AS n\n```")]
    plan_msg = MagicMock(); plan_msg.content = [MagicMock(text='{"intent":"count","approach":"count all","expected_columns":["n"]}')]
    interp_msg = MagicMock(); interp_msg.content = [MagicMock(text='{"intent":"count","summary":"5 nodes","key_insight":"","viz":{"type":"scalar","value":"n"}}')]
    mock_client.return_value.messages.create.side_effect = [plan_msg, cypher_msg, interp_msg]

    graph = MagicMock(); graph.query.return_value = [{"n": 5}]
    steps = list(ask_stream("how many?", graph=graph, schema_prefix="..."))

    names = [s.name for s in steps]
    assert names == ["classify", "plan", "cypher", "execute", "interpret", "cite"]
    assert steps[0].payload["kind"] == "factual_kg"
