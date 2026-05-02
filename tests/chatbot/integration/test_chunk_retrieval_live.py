"""Integration test: ``retrieve_kg_context`` against the live local Neo4j.

This test hits the real Neo4j instance configured via ``config.NEO4J_URI``
(loaded from ``.env``) — the same DB Zone 4 wrote into and the chatbot
queries in production.  No mocks.

The test is auto-skipped when:
  * The Neo4j connection can't be opened (no .env, server down, wrong creds).
  * The graph has zero ``:Chunk`` nodes (DB hasn't been populated yet).

Run locally with:
    pytest tests/chatbot/integration/test_chunk_retrieval_live.py -v

The test deliberately does NOT mutate the graph and uses only read queries.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def graph():
    try:
        import config  # type: ignore
        from langchain_neo4j import Neo4jGraph
    except ImportError as e:
        pytest.skip(f"langchain_neo4j not available: {e}")

    try:
        g = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
            database=config.NEO4J_DATABASE,
        )
        # Probe: must have :Chunk nodes for the test to be meaningful.
        rows = g.query("MATCH (c:Chunk) RETURN count(c) AS n")
        if not rows or rows[0]["n"] == 0:
            pytest.skip("Live Neo4j has 0 :Chunk nodes — load Zone 4 first.")
        return g
    except Exception as e:  # connection refused, auth fail, etc.
        pytest.skip(f"Live Neo4j unavailable: {e}")


class TestRetrieveKgContextLive:
    def test_returns_triples_and_chunks_for_policy_question(self, graph):
        from chatbot.qa_chain import retrieve_kg_context

        out = retrieve_kg_context(graph, "what does the policy cover?")

        # Basic shape — same contract as the unit test
        assert isinstance(out, dict)
        assert "triples" in out and "chunks" in out

        # Live data should produce real grounding for an obvious policy
        # question, otherwise the chatbot's reasoning path is broken.
        assert out["triples"], "no triples returned for a policy question"
        assert out["chunks"], "no chunks returned for a policy question"

        # Schema sanity: chunk rows must carry id, source, and non-empty text.
        for c in out["chunks"]:
            assert c.get("chunk_id"), f"chunk missing chunk_id: {c}"
            assert c.get("source"), f"chunk missing source: {c}"
            text = c.get("text") or ""
            assert text and len(text) > 20, (
                f"chunk text suspiciously short ({len(text)} chars): {c}"
            )

        # Every chunk must be among those cited by the matched triples —
        # otherwise the join is querying the wrong key.  Cast chunk_id to
        # str on both sides because PDF and CSV chunks store the id with
        # different python types in the relationship vs the :Chunk node.
        triple_chunk_keys = {
            (str(r["chunk_id"]), r.get("source") or "")
            for r in out["triples"] if r.get("chunk_id") is not None
        }
        for c in out["chunks"]:
            key = (str(c["chunk_id"]), c.get("source") or "")
            assert key in triple_chunk_keys, (
                f"chunk {c['chunk_id']} not cited by any matched triple"
            )

    def test_chunk_count_capped_at_default(self, graph):
        from chatbot.qa_chain import retrieve_kg_context

        out = retrieve_kg_context(graph, "what does the policy cover?")
        assert len(out["chunks"]) <= 5, "default max_chunks=5 not enforced"

    def test_explicit_max_chunks_respected(self, graph):
        from chatbot.qa_chain import retrieve_kg_context

        out = retrieve_kg_context(
            graph, "what does the policy cover?", max_chunks=2,
        )
        assert len(out["chunks"]) <= 2

    def test_empty_question_returns_empty_payload(self, graph):
        from chatbot.qa_chain import retrieve_kg_context

        out = retrieve_kg_context(graph, "is it ok?")
        assert out == {"triples": [], "chunks": []}

    def test_top_chunks_are_highest_triple_density(self, graph):
        """The chunks returned should be the ones cited by the most matched
        triples — that's how the function ranks for source-grounded QA."""
        from chatbot.qa_chain import retrieve_kg_context

        out = retrieve_kg_context(graph, "deductible coverage exclusion")
        if not out["chunks"]:
            pytest.skip("no chunks for this question on this DB")

        # Recompute hit counts and confirm the returned chunks are among
        # the top-K by hit count (allowing ties).  Cast chunk_id to str
        # because the live graph stores PDF chunk_ids as ints and CSV
        # chunk_ids as strings — the production code normalizes both to
        # str before the :Chunk join, so we do the same here.
        hits: dict[tuple[str, str], int] = {}
        for r in out["triples"]:
            cid = r.get("chunk_id")
            src = r.get("source") or ""
            if cid is not None:
                hits[(str(cid), src)] = hits.get((str(cid), src), 0) + 1
        if not hits:
            pytest.skip("triples carry no chunk_id on this DB")

        sorted_hits = sorted(hits.values(), reverse=True)
        threshold = sorted_hits[min(len(out["chunks"]), len(sorted_hits)) - 1]
        for c in out["chunks"]:
            key = (str(c["chunk_id"]), c.get("source") or "")
            assert hits.get(key, 0) >= threshold, (
                f"chunk {key} (hits={hits.get(key, 0)}) ranked below "
                f"threshold {threshold}"
            )
