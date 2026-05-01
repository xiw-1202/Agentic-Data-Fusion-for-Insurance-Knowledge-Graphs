# chatbot/feedback.py
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_feedback(
    path: Path,
    *,
    question: str,
    verdict: str,  # "up" | "down"
    comment: str,
    trace: list[dict[str, Any]],
) -> None:
    """Append a feedback record as one JSON line."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "verdict": verdict,
        "comment": comment,
        "trace": trace,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
