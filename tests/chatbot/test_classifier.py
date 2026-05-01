# tests/chatbot/test_classifier.py
from unittest.mock import MagicMock, patch
from chatbot.classifier import classify_question, QuestionKind

@patch("chatbot.classifier._client")
def test_classify_returns_known_kind(mock_client):
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text='{"kind":"factual_kg","reason":"counts entities","confidence":0.9}')]
    mock_client.return_value.messages.create.return_value = mock_resp

    result = classify_question("How many flood policies are there?", schema_prefix="...")
    assert result.kind == QuestionKind.FACTUAL_KG
    assert result.confidence == 0.9
    assert result.reason
