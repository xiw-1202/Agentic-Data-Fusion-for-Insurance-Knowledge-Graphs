"""Tests for LOB threading through Zone 2 LLM extraction (Phase 3).

Every triple produced by any extraction path (Pass 1 LLM, decompose,
regex numeric fallback, structured CSV) must carry a ``lob`` field
matching the originating chunk's tag.  This is what lets Zone 3 induce
LOB-aware ontology hierarchies and the chatbot filter by LOB.

The Neo4j edge MERGE must also write ``r.lob = row.lob`` so the LOB
survives onto the relationship and not just the endpoint nodes.
"""

from __future__ import annotations

from zone2.pipeline import (
    _parse_chunk_triples,
    _extract_numeric_from_text,
    _batch_merge_triples,
)


# ---------------------------------------------------------------------------
# _parse_chunk_triples — accepts lob, stamps every triple
# ---------------------------------------------------------------------------

class TestParseChunkTriplesLOB:
    def test_stamps_lob_on_every_triple(self):
        parsed = [
            {"subject": "Coverage A", "relation": "COVERS",
             "object": "Building", "subject_type": "CoverageType",
             "object_type": "InsuredProperty",
             "span": "Coverage A covers building", "confidence": 0.9},
            {"subject": "Policy", "relation": "HAS_DEDUCTIBLE",
             "object": "$500", "subject_type": "InsurancePolicy",
             "object_type": "FinancialAmount",
             "span": "deductible $500", "confidence": 0.95},
        ]
        triples = _parse_chunk_triples(parsed, "c1", "auto.csv", lob="auto")
        assert len(triples) == 2
        for t in triples:
            assert t["lob"] == "auto"

    def test_default_lob_is_generic_for_back_compat(self):
        parsed = [{"subject": "X", "relation": "COVERS",
                   "object": "Y", "span": "x covers y", "confidence": 0.9}]
        # Old call signature without lob still works.
        triples = _parse_chunk_triples(parsed, "c1", "x.csv")
        assert triples[0]["lob"] == "generic"


# ---------------------------------------------------------------------------
# _extract_numeric_from_text — regex fallback path
# ---------------------------------------------------------------------------

class TestExtractNumericLOB:
    def test_regex_numeric_triples_carry_lob(self):
        text = "The deductible is $500 per occurrence under this auto policy."
        triples = _extract_numeric_from_text(text, "c1", "auto.csv", lob="auto")
        # The regex needs to find a deductible-related triple.
        assert any(t["relation"] == "HAS_DEDUCTIBLE" for t in triples)
        for t in triples:
            assert t["lob"] == "auto"

    def test_regex_numeric_default_lob_is_generic(self):
        text = "The deductible is $500 per occurrence."
        # Old call signature without lob still works.
        triples = _extract_numeric_from_text(text, "c1", "x.csv")
        for t in triples:
            assert t["lob"] == "generic"


# ---------------------------------------------------------------------------
# _batch_merge_triples — Neo4j MERGE writes r.lob onto edges
# ---------------------------------------------------------------------------

class _FakeNeo4jGraph:
    """Captures Cypher queries + params so we can assert against them."""

    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    def query(self, cypher: str, params: dict | None = None):
        self.queries.append((cypher, params or {}))


class TestBatchMergeIncludesLOBOnEdge:
    def test_merge_query_sets_lob_property_on_relationship(self):
        graph = _FakeNeo4jGraph()
        batch = [{
            "subject": "POL-1", "subject_type": "FloodPolicyRecord",
            "object": "$500", "object_type": "Currency",
            "span": "deductible 500", "confidence": 1.0,
            "chunk_id": "c1", "source": "policies.json",
            "source_type": "structured", "lob": "flood",
        }]
        by_relation = {"HAS_DEDUCTIBLE": batch}

        n = _batch_merge_triples(graph, by_relation)

        assert n == 1
        assert len(graph.queries) == 1
        cypher, params = graph.queries[0]
        # The Cypher must SET r.lob from the row payload.
        assert "r.lob" in cypher, f"Cypher missing r.lob:\n{cypher}"
        # And the row must carry the lob through.
        assert params["batch"][0]["lob"] == "flood"
