# tests/chatbot/test_feedback.py
import json
from pathlib import Path
from chatbot.feedback import log_feedback

def test_log_feedback_appends_jsonl(tmp_path):
    p = tmp_path / "fb.jsonl"
    log_feedback(p, question="q", verdict="up", comment="great", trace=[{"name": "classify"}])
    log_feedback(p, question="q2", verdict="down", comment="wrong", trace=[])
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["question"] == "q" and rec["verdict"] == "up"
    assert "ts" in rec
