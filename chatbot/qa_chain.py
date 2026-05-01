"""Text-to-Cypher QA chain backed by Claude + the Emory Neo4j KG.

Pipeline:
  1. Claude generates Cypher from the question using a cached schema+examples prefix.
  2. Cypher executes read-only against Neo4j (with guardrails).
  3. Claude interprets the user's intent, summarizes the rows, and chooses a
     visualization (table / bar / line / pie / graph / scalar / text).
  4. The UI renders whatever Claude picked.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Iterator

import anthropic
from langchain_neo4j import Neo4jGraph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from chatbot.classifier import Classification, QuestionKind, classify_question
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
- Return results in a shape useful for charting: when aggregating, return a
  category column and a numeric column so the UI can plot a bar chart.
- Add LIMIT where results could be large.
- If a question cannot be answered from the schema, return the single line: -- UNANSWERABLE
"""

ANSWER_SYSTEM = """You are an analyst who interprets Neo4j query results for a business user.

You receive:
  - The user's original question
  - The Cypher that was run
  - The result rows (JSON)

You must respond with a single JSON object (no prose around it) matching this schema:

{
  "intent": "<1 sentence on what the user is really trying to learn>",
  "summary": "<2-4 sentences explaining the answer, citing specific numbers>",
  "key_insight": "<1 sentence highlighting the most business-relevant takeaway>",
  "viz": {
    "type": "table" | "bar" | "line" | "pie" | "scalar" | "graph" | "text",
    "title": "<chart title, if applicable>",
    "x": "<column name for x-axis (bar/line/pie)>",
    "y": "<column name for y-axis (bar/line)>",
    "label": "<column name for slice label (pie)>",
    "value": "<column name for slice value (pie) or scalar value>",
    "source": "<column name for edge source (graph)>",
    "target": "<column name for edge target (graph)>"
  }
}

How to pick the viz.type:
- "scalar"  : exactly one row with one numeric cell (e.g. a total count).
- "bar"     : one categorical column + one numeric column, <50 rows.
- "line"    : rows ordered by a date/time or sequential numeric x-axis.
- "pie"     : proportions across <=7 categories that sum to a whole.
- "graph"   : rows describe edges between named nodes (parent/child, source/target).
- "table"   : multi-column detailed rows, or >50 rows, or the user asked to "list" / "show".
- "text"    : no rows, or the question is conceptual and the summary alone answers it.

Only fill fields relevant to the chosen viz.type. Return JSON only — no markdown fences.
"""

CYPHER_MODEL = "claude-sonnet-4-6"
ANSWER_MODEL = "claude-sonnet-4-6"

VALID_VIZ = {"table", "bar", "line", "pie", "scalar", "graph", "text"}


@dataclass
class Viz:
    type: str = "table"
    title: str = ""
    x: str = ""
    y: str = ""
    label: str = ""
    value: str = ""
    source: str = ""
    target: str = ""


@dataclass
class QAResult:
    question: str
    cypher: str
    rows: list[dict[str, Any]]
    intent: str = ""
    summary: str = ""
    key_insight: str = ""
    viz: Viz = field(default_factory=Viz)
    guardrail_ok: bool = True
    guardrail_reason: str = "ok"


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _extract_cypher(text: str) -> str:
    m = re.search(r"```(?:cypher)?\s*(.*?)```", text, flags=re.S | re.I)
    return (m.group(1) if m else text).strip()


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"no JSON object found in response: {text[:200]}")
    return json.loads(m.group(0))


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


def interpret_result(
    client: anthropic.Anthropic,
    question: str,
    cypher: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_snippet = rows[:30]
    columns = list(rows[0].keys()) if rows else []
    user_payload = (
        f"Question: {question}\n\n"
        f"Cypher:\n{cypher}\n\n"
        f"Columns: {columns}\n"
        f"Row count: {len(rows)}\n"
        f"Rows (first 30):\n{json.dumps(rows_snippet, default=str)}"
    )
    resp = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=800,
        system=ANSWER_SYSTEM,
        messages=[{"role": "user", "content": user_payload}],
    )
    return _extract_json(resp.content[0].text)


def _coerce_viz(raw: dict[str, Any], rows: list[dict[str, Any]]) -> Viz:
    v = raw.get("viz") or {}
    vtype = v.get("type", "table")
    if vtype not in VALID_VIZ:
        vtype = "table"

    columns = list(rows[0].keys()) if rows else []

    def pick(name: str) -> str:
        val = v.get(name, "")
        return val if val in columns else ""

    viz = Viz(
        type=vtype,
        title=v.get("title", "") or "",
        x=pick("x"),
        y=pick("y"),
        label=pick("label"),
        value=pick("value"),
        source=pick("source"),
        target=pick("target"),
    )

    # Fallbacks if the LLM forgot fields
    if viz.type == "bar" and not (viz.x and viz.y) and len(columns) >= 2:
        viz.x, viz.y = columns[0], columns[1]
    if viz.type == "pie" and not (viz.label and viz.value) and len(columns) >= 2:
        viz.label, viz.value = columns[0], columns[1]
    if viz.type == "graph" and not (viz.source and viz.target) and len(columns) >= 2:
        viz.source, viz.target = columns[0], columns[1]
    if viz.type == "scalar" and not viz.value and len(columns) >= 1:
        viz.value = columns[0]

    return viz


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
            summary="This question can't be answered from the current knowledge graph.",
            viz=Viz(type="text"),
            guardrail_ok=True,
            guardrail_reason="llm-marked-unanswerable",
        )

    ok, reason = is_read_only(raw_cypher)
    if not ok:
        return QAResult(
            question=question,
            cypher=raw_cypher,
            rows=[],
            summary=f"Refused to execute: {reason}.",
            viz=Viz(type="text"),
            guardrail_ok=False,
            guardrail_reason=reason,
        )

    safe_cypher = clamp_limit(raw_cypher)
    rows = graph.query(safe_cypher)

    if not rows:
        return QAResult(
            question=question,
            cypher=safe_cypher,
            rows=[],
            intent="",
            summary="The query returned no rows. The KG may not contain data for that question.",
            viz=Viz(type="text"),
        )

    try:
        interpretation = interpret_result(client, question, safe_cypher, rows)
    except (ValueError, json.JSONDecodeError) as e:
        return QAResult(
            question=question,
            cypher=safe_cypher,
            rows=rows,
            summary=f"Query ran but answer interpretation failed: {e}. See raw rows below.",
            viz=Viz(type="table"),
        )

    return QAResult(
        question=question,
        cypher=safe_cypher,
        rows=rows,
        intent=interpretation.get("intent", ""),
        summary=interpretation.get("summary", ""),
        key_insight=interpretation.get("key_insight", ""),
        viz=_coerce_viz(interpretation, rows),
    )


@dataclass
class Step:
    name: str  # classify | plan | cypher | execute | interpret | cite
    title: str
    payload: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""


PLAN_SYSTEM = """You plan a Cypher query for an insurance KG question.
Return JSON only:
{"intent":"<1 sentence>","sub_questions":["..."],"approach":"<how you'll query>","expected_columns":["..."],"ontology_classes_used":["..."]}
"""


def plan_query(client: anthropic.Anthropic, schema_prefix: str, question: str) -> dict[str, Any]:
    resp = client.messages.create(
        model=CYPHER_MODEL,
        max_tokens=400,
        system=[
            {"type": "text", "text": PLAN_SYSTEM},
            {"type": "text", "text": schema_prefix, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": question}],
    )
    return _extract_json(resp.content[0].text)


def ask_stream(
    question: str,
    graph: Neo4jGraph,
    schema_prefix: str,
) -> Iterator[Step]:
    client = _client()

    cls = classify_question(question, schema_prefix)
    yield Step(
        name="classify",
        title=f"Classified as {cls.kind.value} (conf {cls.confidence:.2f})",
        payload={"kind": cls.kind.value, "reason": cls.reason, "confidence": cls.confidence},
    )

    if cls.kind == QuestionKind.OUT_OF_SCOPE:
        yield Step(name="cite", title="Out of scope", payload={
            "summary": f"This KG can't answer that. {cls.reason}",
            "rows": [], "sources": [],
        })
        return

    if cls.kind == QuestionKind.NEEDS_CLARIFICATION:
        yield Step(name="cite", title="Needs clarification", payload={
            "summary": cls.reason, "rows": [], "sources": [],
        })
        return

    plan = plan_query(client, schema_prefix, question)
    yield Step(name="plan", title="Query plan", payload=plan)

    raw_cypher = generate_cypher(client, schema_prefix, question)
    ok, reason = is_read_only(raw_cypher)
    if not ok:
        yield Step(name="cypher", title="Cypher rejected", payload={"cypher": raw_cypher},
                   ok=False, error=reason)
        return
    safe_cypher = clamp_limit(raw_cypher)
    yield Step(name="cypher", title="Generated Cypher", payload={"cypher": safe_cypher})

    rows = graph.query(safe_cypher)
    yield Step(name="execute", title=f"Got {len(rows)} rows", payload={"rows": rows})

    if not rows:
        yield Step(name="cite", title="No results", payload={
            "summary": "Query returned no rows.", "rows": [], "sources": [],
        })
        return

    interp = interpret_result(client, question, safe_cypher, rows)
    yield Step(name="interpret", title="Interpretation", payload=interp)

    sources = fetch_provenance(graph, rows)
    yield Step(name="cite", title=f"{len(sources)} source chunks", payload={
        "summary": interp.get("summary", ""),
        "key_insight": interp.get("key_insight", ""),
        "rows": rows,
        "sources": sources,
        "viz": interp.get("viz", {}),
    })


def _extract_sources(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Pull out _chunk_id / _source columns if present."""
    seen = set()
    out: list[dict[str, str]] = []
    for r in rows:
        cid = r.get("_chunk_id") or r.get("chunk_id")
        src = r.get("_source") or r.get("source")
        if cid and (cid, src) not in seen:
            seen.add((cid, src))
            out.append({"chunk_id": cid, "source": src or ""})
    return out


def fetch_provenance(graph: Neo4jGraph, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Look up chunk_id/source for any entity ids referenced in rows.

    Aggregate queries (count/avg/sum) get provenance for the entity ids
    that appear in their non-numeric columns. Pure scalar results get
    no sources, which is correct.
    """
    candidate_ids: set[str] = set()
    for row in rows:
        for v in row.values():
            if isinstance(v, str) and v and not v.replace(".", "").replace("-", "").isdigit():
                candidate_ids.add(v)
    if not candidate_ids:
        return []

    prov_rows = graph.query(
        """
        MATCH (e:Entity)-[r]-()
        WHERE e.id IN $ids AND r.chunk_id IS NOT NULL
        RETURN DISTINCT r.chunk_id AS chunk_id, r.source AS source
        LIMIT 50
        """,
        params={"ids": list(candidate_ids)},
    )
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for r in prov_rows:
        key = (r["chunk_id"], r.get("source") or "")
        if r["chunk_id"] and key not in seen:
            seen.add(key)
            out.append({"chunk_id": r["chunk_id"], "source": r.get("source") or ""})
    return out
