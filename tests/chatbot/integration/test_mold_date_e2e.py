"""End-to-end live test: 'when did mold damage claims happen?' must
return non-null dates.

This is the regression test for the multi-source date schema bug:
GEICO renters claims (the source of MOLD claims) record dates under
``HAS_FISCAL_PMS_ACCOUNT_DATE`` / ``HAS_CLAIM_MONTH_ID`` while
T-Mobile claims use ``HAS_CLAIM_LOSS_DATE`` / ``HAS_CLAIM_OPEN_DATE``.
Claude must generate a query that uses OPTIONAL MATCH + COALESCE
across all candidate date relations, otherwise it picks the
T-Mobile-only relations and returns null for every mold claim.

Test calls the real Anthropic API and the real local Neo4j — no mocks.
Auto-skips when either is unavailable.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module", autouse=True)
def _load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        pytest.skip("python-dotenv not installed")
    here = Path(__file__).resolve()
    for env_path in (
        here.parents[3] / ".env",
        (here.parents[3] / ".." / ".." / ".." / ".env").resolve(),
    ):
        if env_path.exists():
            load_dotenv(env_path, override=True)
            break
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


@pytest.fixture(scope="module")
def graph():
    try:
        import config  # type: ignore
        from langchain_neo4j import Neo4jGraph
    except ImportError:
        pytest.skip("config or langchain_neo4j not available")
    try:
        g = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
            database=config.NEO4J_DATABASE,
        )
        rows = g.query(
            "MATCH (c:Entity)-[:HAS_CAUSE_OF_LOSS]->(:Entity {id:'MOLD'}) "
            "RETURN count(c) AS n"
        )
        if not rows or rows[0]["n"] == 0:
            pytest.skip("no MOLD-cause claims in this DB; skip")
        return g
    except Exception as e:
        pytest.skip(f"Neo4j unavailable: {e}")


@pytest.fixture(scope="module")
def schema_prefix() -> str:
    here = Path(__file__).resolve()
    for cache in (
        here.parents[3] / "chatbot" / "schema_prefix.cache.txt",
        (here.parents[3] / ".." / ".." / ".." / "chatbot"
         / "schema_prefix.cache.txt").resolve(),
    ):
        if cache.exists():
            return cache.read_text(encoding="utf-8")
    pytest.skip("schema_prefix.cache.txt not built")


def test_mold_claim_dates_returned_non_null(graph, schema_prefix):
    """The chatbot must return at least one non-null date when asked
    when the mold damage claims happened."""
    import anthropic
    from chatbot.qa_chain import generate_cypher
    from chatbot.guardrails import is_read_only

    client = anthropic.Anthropic()
    cypher = generate_cypher(client, schema_prefix, "When did mold damage claims happen?")

    ok, reason = is_read_only(cypher)
    assert ok, f"generated Cypher rejected by guardrails: {reason}\n{cypher}"

    rows = graph.query(cypher)
    assert rows, "generated Cypher returned zero rows for MOLD claims"

    # At least one row must have a non-null date-like value.  We don't
    # know the exact column name (Claude picks it), so look at every
    # value in the row and accept any string that looks date/time-ish.
    found_date = False
    for r in rows:
        for v in r.values():
            if v is None:
                continue
            s = str(v)
            if any(tok in s for tok in ("2021", "2022", "2023", "2024", "2025",
                                         "20210", "20220", "20230", "20240", "20250",
                                         "T0", "T1", "T2",
                                         "242", "243")):
                found_date = True
                break
        if found_date:
            break

    assert found_date, (
        "every column was null for the mold claims — multi-source date "
        f"COALESCE missing from generated Cypher:\n{cypher}\n\nrows: {rows[:3]}"
    )
