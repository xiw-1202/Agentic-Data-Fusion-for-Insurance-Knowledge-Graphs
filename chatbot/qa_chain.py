"""Text-to-Cypher QA chain backed by Claude + the Emory Neo4j KG.

Two-stage: (1) Claude generates Cypher from the question using a cached
schema+examples prefix; (2) Cypher executes read-only against Neo4j;
(3) Claude formats the rows into a natural-language answer.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import anthropic
from langchain_neo4j import Neo4jGraph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from chatbot.examples import format_for_prompt as examples_block
from chatbot.guardrails import clamp_limit, is_read_only
from chatbot.schema import summarize_schema


CYPHER_SYSTEM = """You are a Neo4j Cypher expert for an insurance knowledge graph.
Given a user question and the schema/examples below, emit exactly one Cypher query.

Rules:
- Use the exact relation types and labels shown in the schema — do not invent names.
- Return only a Cypher query inside a ```cypher code fence. No prose outside the fence.
- ClaimRecord entities have entity_type = 'ClaimRecord'. Other record types include SurveyRecord, PolicyRecord.
- Numeric values are stored as :Entity nodes; cast with toFloat(n.id) or toInteger(n.id) for aggregation.
- Prefer MATCH + RETURN only. Never write CREATE/MERGE/DELETE/SET/REMOVE/CALL.
- Add LIMIT where results could be large.
- If a question cannot be answered from the schema, return the single line: -- UNANSWERABLE
"""

ANSWER_SYSTEM = """You are an analyst explaining a Neo4j query result to a business user.
Given the user's question, the Cypher query that was run, and the result rows, write
a concise natural-language answer (2-4 sentences). Cite specific numbers from the rows.
If rows are empty, say so plainly and suggest what the KG does contain.
"""

CYPHER_MODEL = "claude-sonnet-4-6"
ANSWER_MODEL = "claude-sonnet-4-6"


@dataclass
class QAResult:
    question: str
    cypher: str
    rows: list[dict[str, Any]]
    answer: str
    guardrail_ok: bool
    guardrail_reason: str


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _extract_cypher(text: str) -> str:
    m = re.search(r"```(?:cypher)?\s*(.*?)```", text, flags=re.S | re.I)
    return (m.group(1) if m else text).strip()


def build_schema_prefix(graph: Neo4jGraph) -> str:
    """The cacheable prefix: schema + few-shot examples. ~3-6K tokens."""
    return f"{summarize_schema(graph)}\n\n{examples_block()}"


def generate_cypher(
    client: anthropic.Anthropic,
    schema_prefix: str,
    question: str,
) -> str:
    resp = client.messages.create(
        model=CYPHER_MODEL,
        max_tokens=600,
        system=[
            {"type": "text", "text": CYPHER_SYSTEM},
            {
                "type": "text",
                "text": schema_prefix,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": f"Question: {question}"}],
    )
    return _extract_cypher(resp.content[0].text)


def format_answer(
    client: anthropic.Anthropic,
    question: str,
    cypher: str,
    rows: list[dict[str, Any]],
) -> str:
    rows_snippet = rows[:30]
    resp = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=500,
        system=ANSWER_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Cypher:\n{cypher}\n\n"
                    f"Rows (showing up to 30 of {len(rows)}):\n{rows_snippet}"
                ),
            }
        ],
    )
    return resp.content[0].text.strip()


def ask(
    question: str,
    graph: Neo4jGraph | None = None,
    schema_prefix: str | None = None,
) -> QAResult:
    graph = graph or Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )
    prefix = schema_prefix or build_schema_prefix(graph)
    client = _client()

    raw_cypher = generate_cypher(client, prefix, question)
    if raw_cypher.strip() == "-- UNANSWERABLE":
        return QAResult(
            question=question,
            cypher=raw_cypher,
            rows=[],
            answer="This question can't be answered from the current knowledge graph.",
            guardrail_ok=True,
            guardrail_reason="llm-marked-unanswerable",
        )

    ok, reason = is_read_only(raw_cypher)
    if not ok:
        return QAResult(
            question=question,
            cypher=raw_cypher,
            rows=[],
            answer=f"Refused to execute: {reason}.",
            guardrail_ok=False,
            guardrail_reason=reason,
        )

    safe_cypher = clamp_limit(raw_cypher)
    rows = graph.query(safe_cypher)
    answer = format_answer(client, question, safe_cypher, rows)
    return QAResult(
        question=question,
        cypher=safe_cypher,
        rows=rows,
        answer=answer,
        guardrail_ok=True,
        guardrail_reason="ok",
    )
