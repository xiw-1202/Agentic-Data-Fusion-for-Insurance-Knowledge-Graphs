# chatbot/classifier.py
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from enum import Enum

import anthropic

CLASSIFIER_MODEL = "claude-sonnet-4-6"

CLASSIFIER_SYSTEM = """You triage user questions for an insurance knowledge-graph chatbot.

Given the question and a compact schema, classify into ONE kind:
- factual_kg          : answerable by a Cypher query (counts, lookups, aggregations)
- open_interpretive   : needs reasoning over KG content (why, recommend, compare, summarize)
- out_of_scope        : KG cannot answer (general knowledge, opinion, unrelated domain)
- needs_clarification : ambiguous, missing key constraint

Return JSON only:
{"kind":"<one of above>","reason":"<1 sentence>","confidence":<0.0-1.0>}
"""


class QuestionKind(str, Enum):
    FACTUAL_KG = "factual_kg"
    OPEN_INTERPRETIVE = "open_interpretive"
    OUT_OF_SCOPE = "out_of_scope"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass(frozen=True)
class Classification:
    kind: QuestionKind
    reason: str
    confidence: float


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def classify_question(question: str, schema_prefix: str) -> Classification:
    resp = _client().messages.create(
        model=CLASSIFIER_MODEL,
        max_tokens=200,
        system=[
            {"type": "text", "text": CLASSIFIER_SYSTEM},
            {"type": "text", "text": schema_prefix, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": question}],
    )
    text = resp.content[0].text
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return Classification(QuestionKind.NEEDS_CLARIFICATION, "classifier returned no JSON", 0.0)
    raw = json.loads(m.group(0))
    return Classification(
        kind=QuestionKind(raw["kind"]),
        reason=raw.get("reason", ""),
        confidence=float(raw.get("confidence", 0.5)),
    )
