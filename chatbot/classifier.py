# chatbot/classifier.py
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from enum import Enum

import anthropic

CLASSIFIER_MODEL = "claude-sonnet-4-6"

CLASSIFIER_SYSTEM = """You triage user questions for an insurance knowledge-graph chatbot.

Classify into ONE kind:
- factual_kg          : answerable by a Cypher query (counts, lookups, aggregations)
- open_interpretive   : needs reasoning over KG content (why, recommend, compare, summarize)
- out_of_scope        : KG cannot answer (general knowledge, opinion, unrelated domain)
- needs_clarification : truly ambiguous; no actionable filter AND no actionable noun

THE NUMBER ONE RULE — read this twice:
A question with a domain noun (claim, policy, coverage, deductible,
survey, premium, county, state, …) AND a category filter (cause-of-loss,
state, dataset, date, status, type, amount, …) is ALWAYS factual_kg.
Multiple matching rows is the EXPECTED outcome, not a problem.
"Multiple claims could match" is NOT a reason to ask for clarification —
it's a reason to RETURN ALL OF THEM as a list. Do NOT classify as
needs_clarification just because the user used a singular noun like
"the claim" or "the policy". Singular = "all matching" in this app.

Definitive examples (do NOT second-guess these):
  factual_kg →
    • "what time was mold damage claim happened?"   (filter: cause=MOLD)
    • "when did the mold claim happen?"             (filter: cause=MOLD)
    • "what time was the claim filed?"              (return all claim dates)
    • "show me the policy expiration date"          (all policy exp dates)
    • "list policies in OH"                         (filter: risk_st=OH)
    • "show large coverage amounts"                 (filter: amt > threshold)

  needs_clarification → ONLY when no domain noun AND no filter:
    • "tell me about it"             (no anchor at all)
    • "show me the data"             (no filter, no specific entity)
    • "what should I look at?"       (meta, no target)

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


_DOMAIN_NOUNS = frozenset({
    "claim", "claims", "policy", "policies", "coverage", "coverages",
    "deductible", "premium", "survey", "surveys", "endorsement",
    "loss", "damage", "peril", "exclusion", "warranty", "contract",
    "service", "device", "vehicle",
})

_FILTER_HINTS = frozenset({
    # Filter-like keywords: temporal, spatial, categorical, comparative
    "when", "what", "where", "which", "list", "show", "give", "find",
    "how", "many", "much", "with", "in", "for", "by", "time",
    # Domain category words from the schema
    "mold", "fire", "water", "theft", "flood", "wind", "tornado",
    "geico", "tmobile", "auto", "renters", "device",
    "ohio", "florida", "texas", "open", "closed", "active",
    "approved", "denied", "paid", "pending",
})


def _looks_factually_answerable(question: str) -> bool:
    """Cheap deterministic check: does the question contain both a
    domain noun and at least one filter-like keyword?  If yes, the
    chatbot can almost certainly run a Cypher filter and return rows —
    even when the LLM classifier hesitates.
    """
    words = {w.strip(".,?!:;()'\"").lower()
             for w in question.split() if w}
    return bool(words & _DOMAIN_NOUNS) and bool(words & _FILTER_HINTS)


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
    kind = QuestionKind(raw["kind"])
    reason = raw.get("reason", "")
    confidence = float(raw.get("confidence", 0.5))

    # Safety net: the LLM classifier sometimes over-flags
    # needs_clarification on questions like "what time was mold damage
    # claim happened?" — singular grammar trips it into demanding a
    # specific row ID.  If the question contains a domain noun + a
    # filter-like keyword, override to factual_kg and let the Cypher
    # generator try.  Worst case the query returns no rows and the UI
    # shows an empty result — strictly better than refusing to answer.
    if kind == QuestionKind.NEEDS_CLARIFICATION and _looks_factually_answerable(question):
        return Classification(
            kind=QuestionKind.FACTUAL_KG,
            reason=f"override: domain noun + filter hint detected (LLM said: {reason})",
            confidence=min(confidence, 0.6),
        )

    return Classification(kind=kind, reason=reason, confidence=confidence)
