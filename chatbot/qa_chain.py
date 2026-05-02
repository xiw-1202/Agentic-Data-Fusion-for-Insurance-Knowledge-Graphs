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
- When the schema lists "Sample entity ids per class" or "Categorical relation values",
  ALWAYS prefer exact-match equality (`= 'Auto'`) over fuzzy matching (`CONTAINS 'auto'`).
  The listed values are the canonical strings actually present in the KG.
- Only use `CONTAINS` / `toLower` when filtering free-text fields (UUIDs, claim numbers,
  addresses, descriptions). Categorical enums and named entities are exact.
- The "Relation types (top N by frequency)" list shows edge counts. When multiple relations
  could answer the question (e.g. HAS_TOTAL_CLAIM_TIME vs HAS_TIME_TO_RESOLVE_HR for
  "resolution time"), prefer the one with higher count — sparse relations often reflect
  extraction gaps, not real answers.

CRITICAL — multi-source date relations:
  Different source datasets (T-Mobile vs GEICO renters) record dates under
  DIFFERENT relation names even on the same :ClaimRecord type:
    • T-Mobile claims:        HAS_CLAIM_LOSS_DATE, HAS_CLAIM_OPEN_DATE,
                              HAS_CLAIM_AUTHORIZED_DATE, HAS_REPORT_DATE
    • GEICO renters claims:   HAS_FISCAL_PMS_ACCOUNT_DATE, HAS_CLAIM_MONTH_ID,
                              HAS_CLAIM_DENIED_DATE
  For "when did X happen?" questions where the filter spans both datasets
  (e.g. cause-of-loss = MOLD lives in GEICO; device repairs live in T-Mobile),
  use OPTIONAL MATCH on EVERY candidate date relation and return the first
  non-null one via COALESCE.  Example skeleton:
    MATCH (c:Entity)-[:HAS_CAUSE_OF_LOSS]->(:Entity {id:'MOLD'})
    OPTIONAL MATCH (c)-[:HAS_CLAIM_LOSS_DATE]->(d1)
    OPTIONAL MATCH (c)-[:HAS_FISCAL_PMS_ACCOUNT_DATE]->(d2)
    OPTIONAL MATCH (c)-[:HAS_CLAIM_OPEN_DATE]->(d3)
    RETURN c.id AS claim, COALESCE(d1.id, d2.id, d3.id) AS date

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
    """Extract a JSON object from a model response.

    Tolerates: code fences (```json … ```), wrapping text, and (most
    importantly) responses truncated mid-JSON because the model hit a
    token cap.  When the closing brace is missing, walk backwards from
    the cut to the last complete (key, value) pair, append a closing
    brace, and parse the salvageable prefix.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Find the start of the JSON object (first balanced-or-truncated brace).
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in response: {text[:200]}")

    body = text[start:]

    # Try a clean parse first.
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass

    # Try greedy regex (the original behavior — handles trailing prose).
    m = re.search(r"\{.*\}", body, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Salvage path: response truncated mid-string.  An unclosed string
    # leaves an odd number of (unescaped) double-quotes — close it and
    # the object, then retry.  Preserves the partial trailing field
    # rather than dropping it.
    n_quotes = body.count('"') - body.count('\\"')
    if n_quotes % 2 == 1:
        for closer in ('"}', '"]}', '",}'):
            try:
                return json.loads(body + closer)
            except json.JSONDecodeError:
                continue

    # Last resort: walk backwards to the last `,` that follows a
    # closed string or numeric value and close the object there.
    for i in range(len(body) - 1, 0, -1):
        if body[i] == "," and i > 0 and body[i - 1] in '"0123456789]}':
            try:
                return json.loads(body[:i] + "}")
            except json.JSONDecodeError:
                continue

    raise ValueError(f"no JSON object found in response: {text[:200]}")


_SCHEMA_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "schema_prefix.cache.txt"
)


def build_schema_prefix(graph: Neo4jGraph) -> str:
    """The cacheable prefix: schema + few-shot examples. ~3-6K tokens.

    Loads from a pre-built file at chatbot/schema_prefix.cache.txt if it exists
    (built by `python -m chatbot.build_schema_cache` after each KG reload).
    Otherwise queries Neo4j live.
    """
    if os.path.exists(_SCHEMA_CACHE_PATH):
        with open(_SCHEMA_CACHE_PATH, encoding="utf-8") as f:
            return f.read()
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

    if cls.kind == QuestionKind.OPEN_INTERPRETIVE:
        kg_ctx = retrieve_kg_context(graph, question)
        n_triples = len(kg_ctx.get("triples", []))
        n_chunks = len(kg_ctx.get("chunks", []))
        yield Step(
            name="execute",
            title=f"Retrieved {n_triples} triples + {n_chunks} source chunks",
            payload={"rows": kg_ctx.get("triples", []),
                     "chunks": kg_ctx.get("chunks", [])},
        )
        reasoning = reason_open(client, schema_prefix, question, kg_ctx)
        yield Step(name="interpret", title="Open-question reasoning", payload=reasoning)

        # Provenance: ONLY the chunks we actually fed into the LLM —
        # those are the chunks the answer is grounded in.  Triples that
        # matched on a tangential keyword (e.g. a tmobile field called
        # DEVICE_DAMAGE matching a "mold damage" question) are still
        # part of ``triples`` for the LLM's edge-level reasoning, but
        # their chunks were never read by the model and shouldn't be
        # surfaced as sources of the answer.
        sources: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for c in kg_ctx.get("chunks", []):
            cid = str(c.get("chunk_id") or "")
            src = c.get("source") or ""
            key = (cid, src)
            if cid and key not in seen:
                seen.add(key)
                sources.append({"chunk_id": cid, "source": src})

        yield Step(name="cite", title="Open answer", payload={
            "summary": reasoning.get("summary", ""),
            "key_insight": f"Confidence {reasoning.get('confidence', 0.0):.0%}. {reasoning.get('caveats', '')}",
            "rows": [],
            "sources": sources,
            "viz": {"type": "text"},
        })
        return

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

    # Demo fast-path: if the question matches a curated example exactly,
    # use the example's hand-tuned Cypher instead of regenerating it.
    # Saves an LLM round-trip and gives presentations predictable
    # behavior — Claude otherwise tunnel-visions on individual words
    # in the question (e.g. "hierarchy" → SUBCLASS_OF only) and
    # produces narrower queries than the example intends.
    from chatbot.examples import EXAMPLES
    canonical_q = question.strip().rstrip("?").lower()
    matched = next(
        (e for e in EXAMPLES
         if e["question"].strip().rstrip("?").lower() == canonical_q),
        None,
    )
    if matched:
        yield Step(name="plan", title="Query plan",
                   payload={"intent": "Saved demo query — using curated Cypher",
                            "approach": "Bypassing LLM generation; running the "
                                        "verified example query directly.",
                            "classes_used": []})
        raw_cypher = matched["cypher"]
    else:
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


CHUNK_HARD_CAP = 12  # absolute ceiling regardless of ties / max_chunks

# Map plain-English class words in a question to the actual class names
# / entity_type values used in the KG.  Domain-specific to insurance
# but easy to extend.
_CLASS_KEYWORDS: dict[str, list[str]] = {
    "claim":     ["Claim", "ClaimRecord"],
    "claims":    ["Claim", "ClaimRecord"],
    "policy":    ["Policy", "PolicyRecord"],
    "policies":  ["Policy", "PolicyRecord"],
    "survey":    ["Survey", "SurveyRecord", "ClientJourneySurvey"],
    "surveys":   ["Survey", "SurveyRecord", "ClientJourneySurvey"],
    "coverage":  ["Coverage", "CoverageType"],
    "coverages": ["Coverage", "CoverageType"],
    "person":    ["Person"],
    "organization": ["Organization"],
    "procedure": ["Procedure", "WarrantyServiceProcedure", "ServiceProcedure"],
    "device":    ["Device"],
    "property":  ["Property", "InsuredProperty"],
}


def _retrieve_by_class(
    graph: Neo4jGraph,
    class_or_type_names: list[str],
    per_class_limit: int = 8,
) -> list[dict[str, Any]]:
    """Pull a small balanced sample of outgoing relations from entities
    that belong to *any* of the given classes / entity_types.

    Catches the case where a question explicitly names a type
    (e.g. "claim records") but the underlying entities have hex IDs
    (CLM-d0beac0c11fd) that won't textually match the keyword.

    Per-class CALL subquery ensures each class contributes its own
    quota — without it, the first class would consume the global
    LIMIT and later classes get nothing.
    """
    if not class_or_type_names:
        return []
    return graph.query(
        """
        UNWIND $names AS name
        CALL (name) {
            MATCH (e:Entity)
            WHERE EXISTS {
                    MATCH (e)-[:INSTANCE_OF]->(c:OntologyClass {name: name})
                  }
               OR e.entity_type = name
            RETURN e
            ORDER BY e.id
            LIMIT $per_class
        }
        MATCH (e)-[r]->(o:Entity)
        WHERE type(r) <> 'INSTANCE_OF'
        RETURN DISTINCT e.id AS subject, type(r) AS rel, o.id AS object,
               r.chunk_id AS chunk_id, r.source AS source
        """,
        params={"names": class_or_type_names, "per_class": per_class_limit},
    )


def retrieve_kg_context(
    graph: Neo4jGraph,
    question: str,
    limit: int = 30,
    max_chunks: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Hybrid GraphRAG retrieval for open-interpretive questions.

    Returns a payload with both:
      * ``triples`` — keyword-matched (subject, rel, object, chunk_id, source)
        rows for grounding edge-level claims.
      * ``chunks`` — source chunks (with their full text) ranked by how many
        keyword-matched triples cite each one, so the LLM can quote actual
        policy/contract language and reason about clauses that aren't fully
        expressed as edges.

    Chunk selection policy:
      * Take the top ``max_chunks`` chunks by hit count (default 5 ≈ 3–5K
        tokens of grounding text — comfortable for Sonnet 4.6).
      * Then **extend ties at the threshold**: any additional chunk with
        the same hit count as the K-th chunk is included.  Avoids silently
        dropping equally-relevant evidence.
      * A hard ceiling of ``CHUNK_HARD_CAP`` (=12) prevents pathological
        questions from blowing up the LLM payload.
    """
    tokens = [t.lower() for t in re.findall(r"[A-Za-z]{4,}", question)]
    stop = {"what", "which", "when", "where", "show", "list", "tell",
            "have", "with", "from", "this", "that", "they", "would",
            "could", "about", "into", "many", "most", "more"}
    keywords = [t for t in tokens if t not in stop][:6]
    if not keywords:
        return {"triples": [], "chunks": []}

    keyword_triples = graph.query(
        """
        UNWIND $kws AS kw
        MATCH (e:Entity)-[r]-(o:Entity)
        WHERE toLower(e.id) CONTAINS kw
        RETURN DISTINCT e.id AS subject, type(r) AS rel, o.id AS object,
               r.chunk_id AS chunk_id, r.source AS source
        LIMIT $limit
        """,
        params={"kws": keywords, "limit": limit},
    )

    # Also retrieve a balanced sample of triples from any class names
    # mentioned in the question.  Bridges the gap when entity IDs are
    # hex hashes (CLM-d0beac0c11fd) that won't substring-match the
    # natural-language keyword ("claim").
    class_targets: list[str] = []
    for kw in keywords:
        class_targets.extend(_CLASS_KEYWORDS.get(kw, []))
    class_targets = list(dict.fromkeys(class_targets))  # dedup, keep order
    class_triples = _retrieve_by_class(graph, class_targets) if class_targets else []

    # Merge; dedupe on (subject, rel, object).
    seen: set[tuple[str, str, str]] = set()
    triples: list[dict[str, Any]] = []
    for t in keyword_triples + class_triples:
        key = (t["subject"], t["rel"], t["object"])
        if key in seen:
            continue
        seen.add(key)
        triples.append(t)

    if not triples:
        return {"triples": [], "chunks": []}

    # Rank chunks by how many of the matched triples cite them — a chunk
    # cited by many keyword-matched triples is the best candidate to feed
    # to the LLM for source-grounded reasoning.
    #
    # The Zone 4 loader stores ``:Chunk.id`` as ``toString(r.chunk_id)``,
    # but ``r.chunk_id`` itself can be either int (PDF chunks) or string
    # (CSV chunks).  Cast both to string here so PDF and CSV chunks join
    # consistently and aren't silently dropped.
    chunk_hits: dict[tuple[str, str], int] = {}
    for r in triples:
        cid = r.get("chunk_id")
        if cid is None:
            continue
        src = r.get("source") or ""
        key = (str(cid), src)
        chunk_hits[key] = chunk_hits.get(key, 0) + 1

    if not chunk_hits:
        return {"triples": triples, "chunks": []}

    # Sort by hit count desc, then take top K with bounded tie-extension.
    # E.g. with max_chunks=5 and hits [4,4,3,3,3,3,2]:
    #   - top 5 = [4,4,3,3,3], threshold = 3
    #   - tie-extend: also include the 4th '3' → 6 chunks total
    #   - extension is bounded to max_chunks + TIE_EXTENSION_SLACK so
    #     a small max_chunks (e.g. 2) doesn't balloon when many chunks
    #     share the same boundary hit count.
    #   - CHUNK_HARD_CAP is the absolute ceiling.
    TIE_EXTENSION_SLACK = 2
    sorted_hits = sorted(
        chunk_hits.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),  # tiebreak deterministic
    )
    if len(sorted_hits) <= max_chunks:
        kept_keys = sorted_hits
    else:
        threshold = sorted_hits[max_chunks - 1][1]
        tied_extras = [
            (k, h) for k, h in sorted_hits[max_chunks:] if h >= threshold
        ]
        if len(tied_extras) <= TIE_EXTENSION_SLACK:
            kept_keys = sorted_hits[:max_chunks] + tied_extras
        else:
            # Too many ties at threshold — take only top max_chunks
            # deterministically rather than ballooning the prompt.
            kept_keys = sorted_hits[:max_chunks]
        kept_keys = kept_keys[:CHUNK_HARD_CAP]

    top_ids = [k[0] for k, _ in kept_keys]
    top_sources = [k[1] for k, _ in kept_keys]

    chunks = graph.query(
        """
        UNWIND range(0, size($ids) - 1) AS i
        MATCH (c:Chunk {id: $ids[i], source: $sources[i]})
        RETURN c.id AS chunk_id, c.source AS source, c.text AS text
        """,
        params={"ids": top_ids, "sources": top_sources},
    )

    return {"triples": triples, "chunks": chunks[:CHUNK_HARD_CAP]}


OPEN_SYSTEM = """You answer interpretive questions about an insurance knowledge graph.

You receive:
- The user's question
- A small set of retrieved triples from the KG (edges and entities)
- The full text of the top source chunks those triples came from
  (the actual policy / contract / survey paragraphs)

Reasoning rules:
- Quote or paraphrase the source chunks when the answer depends on
  policy language (exclusions, conditions, deductibles, eligibility).
- Cross-check the chunk text against the triples — if they disagree,
  trust the chunk text and note the discrepancy in caveats.
- If neither triples nor chunks support an answer, say "the KG doesn't
  contain evidence for this" and describe what data would answer it.
- Never invent statistics, clauses, or fact patterns not in the inputs.

Return JSON only:
{"summary":"<2-4 sentences>",
 "reasoning":"<how you arrived at it, citing chunk text where relevant>",
 "evidence_used":["<entity_id or rel or chunk_id>", "..."],
 "confidence":<0.0-1.0>,
 "caveats":"<what's missing or uncertain>"}
"""


def _format_kg_context(kg_context: dict[str, list[dict[str, Any]]]) -> str:
    """Render the hybrid retrieval payload for the LLM prompt.

    Triples are shown as compact JSON (cheap, structural).  Chunk text is
    shown as labeled blocks so the LLM can quote them and ``evidence_used``
    can cite ``chunk_id``s.  Each chunk is truncated to keep the prompt
    bounded; the full text is still on disk + visible in the chatbot UI.
    """
    triples = kg_context.get("triples", [])[:30]
    chunks = kg_context.get("chunks", [])

    parts: list[str] = []
    parts.append(f"Retrieved triples ({len(triples)}):")
    parts.append(json.dumps(triples, default=str))
    if chunks:
        parts.append(f"\nSource chunks ({len(chunks)}):")
        for c in chunks:
            cid = c.get("chunk_id", "?")
            src = c.get("source", "?")
            text = (c.get("text") or "").strip()
            if len(text) > 1500:
                text = text[:1500] + " […]"
            parts.append(f"\n[chunk_id={cid} | source={src}]\n{text}")
    return "\n".join(parts)


def reason_open(
    client: anthropic.Anthropic,
    schema_prefix: str,
    question: str,
    kg_context: dict[str, list[dict[str, Any]]] | list[dict[str, Any]],
) -> dict[str, Any]:
    # Back-compat: accept the old list-of-triples shape from any caller
    # that hasn't been migrated yet.
    if isinstance(kg_context, list):
        kg_context = {"triples": kg_context, "chunks": []}

    body = _format_kg_context(kg_context)
    payload = f"Question: {question}\n\n{body}"

    resp = client.messages.create(
        model=ANSWER_MODEL,
        # Bumped from 700 to 1500 because hybrid GraphRAG payloads
        # (triples + chunk-text) push the model toward longer answers
        # that quote source language.  At 700 the JSON could be cut
        # mid-string and crash _extract_json.
        max_tokens=1500,
        system=[
            {"type": "text", "text": OPEN_SYSTEM},
            {"type": "text", "text": schema_prefix, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": payload}],
    )
    return _extract_json(resp.content[0].text)
