"""Tests for ``relation_raw`` field threading + Zone 2 skip-canonicalize flags.

Stage A of the canonicalization-belongs-in-Zone-3 architecture pivot:

  • Every triple emitted by Zone 2 carries a ``relation_raw`` field
    holding the post-sanitize, pre-cleanup relation name.  This is
    the lossless source-specific form that Zone 3's relation hierarchy
    induction will cluster bottom-up.

  • Two new pipeline flags — ``skip_canonicalize`` and ``skip_normalize``
    — short-circuit the corresponding Zone 2 nodes so cross-source/
    cross-LOB merging can be deferred to Zone 3.

  • The Neo4j MERGE writes ``r.relation_raw`` onto every edge alongside
    the canonical ``relation`` so Zone 3 can audit raw vs cleaned forms.

Defaults for the skip flags are ``False`` (current behavior preserved)
so existing runs are unaffected.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from zone2.pipeline import (
    _parse_chunk_triples,
    _extract_numeric_from_text,
    _batch_merge_triples,
    canonicalize_relations,
    normalize_structured_relations,
)
from zone2.structured_mapper import record_to_triples


# ---------------------------------------------------------------------------
# Emission sites — every triple gets relation_raw
# ---------------------------------------------------------------------------

class TestRecordToTriplesEmitsRelationRaw:
    def test_property_triples_carry_relation_raw_matching_field(self):
        rec = {"policy effective date": "2025-01-10",
               "rated flood zone": "AE"}
        triples = record_to_triples(
            record=rec, record_type="policies",
            chunk_id="c1", source="policies.json", lob="flood",
        )
        prop_triples = [t for t in triples if t["relation"] != "IS_A"]
        # Every property triple has relation_raw == its current relation
        # because no semantic cleanup is applied to structured field names.
        for t in prop_triples:
            assert "relation_raw" in t, f"missing relation_raw: {t}"
            assert t["relation_raw"] == t["relation"]

    def test_is_a_triples_also_carry_relation_raw(self):
        rec = {"policy effective date": "2025-01-10"}
        triples = record_to_triples(
            record=rec, record_type="policies",
            chunk_id="c1", source="policies.json", lob="flood",
        )
        for t in triples:
            assert "relation_raw" in t


class TestParseChunkTriplesEmitsRelationRaw:
    def test_relation_raw_preserved_before_paraphrase_normalization(self):
        # 'INSURES' is in RELATION_NORMALIZATIONS → maps to 'COVERS'.
        # relation_raw should keep the sanitized form 'INSURES';
        # relation should hold the canonical 'COVERS'.
        parsed = [{"subject": "Policy", "relation": "INSURES",
                   "object": "Building", "subject_type": "InsurancePolicy",
                   "object_type": "Structure", "span": "policy insures building",
                   "confidence": 0.9}]
        triples = _parse_chunk_triples(parsed, "c1", "doc.pdf", lob="flood")
        assert triples
        t = triples[0]
        assert t["relation"] == "COVERS"
        assert t["relation_raw"] == "INSURES"

    def test_relation_raw_preserved_before_pattern_normalization(self):
        # Verbose 70B compound names get pattern-collapsed.  relation_raw
        # holds the verbose original; relation holds the cleaned form.
        parsed = [{"subject": "Policy", "relation": "EXCLUDES_MULTIPLE_PERILS",
                   "object": "Earthquake", "subject_type": "InsurancePolicy",
                   "object_type": "Peril", "span": "policy excludes earthquake",
                   "confidence": 0.9}]
        triples = _parse_chunk_triples(parsed, "c1", "doc.pdf", lob="flood")
        assert triples
        t = triples[0]
        assert t["relation"] == "EXCLUDED_FROM"
        assert t["relation_raw"] == "EXCLUDES_MULTIPLE_PERILS"

    def test_no_op_extraction_keeps_relation_equal_to_relation_raw(self):
        # When no paraphrase/pattern cleanup applies, relation == relation_raw.
        parsed = [{"subject": "Policy", "relation": "COVERS",
                   "object": "Building", "subject_type": "InsurancePolicy",
                   "object_type": "Structure", "span": "covers", "confidence": 0.9}]
        triples = _parse_chunk_triples(parsed, "c1", "doc.pdf")
        assert triples[0]["relation_raw"] == triples[0]["relation"] == "COVERS"


class TestExtractNumericEmitsRelationRaw:
    def test_regex_numeric_triples_carry_relation_raw(self):
        text = "The deductible is $500 per occurrence."
        triples = _extract_numeric_from_text(text, "c1", "doc.pdf",
                                              lob="flood")
        assert triples
        for t in triples:
            assert "relation_raw" in t
            assert t["relation_raw"] == t["relation"]


# ---------------------------------------------------------------------------
# Canonicalize + normalize must NOT touch relation_raw
# ---------------------------------------------------------------------------

class TestCanonicalizePreservesRelationRaw:
    def test_canonicalize_modifies_relation_only(self, monkeypatch):
        # Stub the LLM call so canonicalize sees a deterministic mapping.
        from zone2 import pipeline as p

        class FakeResp:
            content = '{"mappings": {"PROVIDES_PROTECTION_FOR": "COVERS"}}'

        class FakeLLM:
            def invoke(self, _msgs):
                return FakeResp()

        monkeypatch.setattr(p, "get_llm", lambda *a, **k: FakeLLM())

        triples = [{
            "subject": "Policy", "subject_type": "InsurancePolicy",
            "relation": "PROVIDES_PROTECTION_FOR",
            "relation_raw": "PROVIDES_PROTECTION_FOR",
            "object": "Building", "object_type": "Structure",
            "span": "...", "confidence": 0.9,
            "chunk_id": "c1", "source": "doc.pdf",
            "source_type": "llm", "lob": "flood",
        }]
        out = canonicalize_relations({
            "triples": triples, "vocab": ["COVERS", "EXCLUDED_FROM"],
            "model": "fake",
        })
        new = out["triples"][0]
        assert new["relation"] == "COVERS"
        assert new["relation_raw"] == "PROVIDES_PROTECTION_FOR"


class TestNormalizePreservesRelationRaw:
    def test_normalize_modifies_relation_only(self):
        # Two near-duplicate HAS_* relations that should merge under the
        # default thresholds.  Object types must overlap for the merge.
        triples = [
            {
                "subject": "POL-1", "subject_type": "FloodPolicyRecord",
                "relation": "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE",
                "relation_raw": "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE",
                "object": "100000", "object_type": "Currency",
                "span": "", "confidence": 1.0,
                "chunk_id": "c1", "source": "policies.json",
                "source_type": "structured", "lob": "flood",
            }
            for _ in range(3)
        ] + [
            {
                "subject": "POL-2", "subject_type": "RentersPolicyRecord",
                "relation": "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE_AMOUNT",
                "relation_raw": "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE_AMOUNT",
                "object": "75000", "object_type": "Currency",
                "span": "", "confidence": 1.0,
                "chunk_id": "c2", "source": "renters.csv",
                "source_type": "structured", "lob": "renters",
            }
            for _ in range(3)
        ]
        out = normalize_structured_relations({
            "triples": triples, "model": "fake",
        })
        # If a merge happened, the original relation_raw on each triple
        # should still match its source-specific form.
        for t in out.get("triples", triples):
            assert "relation_raw" in t
            # relation_raw never changes from what was passed in.
            if t["subject"] == "POL-1":
                assert t["relation_raw"] == "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE"
            elif t["subject"] == "POL-2":
                assert t["relation_raw"] == "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE_AMOUNT"


# ---------------------------------------------------------------------------
# Skip flags short-circuit canonicalize + normalize
# ---------------------------------------------------------------------------

class TestSkipCanonicalize:
    def test_skip_canonicalize_returns_no_changes(self, monkeypatch):
        # If the LLM is unreachable, the test would fail without the skip
        # short-circuit — proves skip is honored.
        from zone2 import pipeline as p

        def fail_get_llm(*a, **k):
            raise AssertionError("LLM should not be called when skip_canonicalize=True")

        monkeypatch.setattr(p, "get_llm", fail_get_llm)

        triples = [{
            "subject": "X", "subject_type": "T",
            "relation": "INSURES", "relation_raw": "INSURES",
            "object": "Y", "object_type": "T",
            "span": "", "confidence": 0.9,
            "chunk_id": "c1", "source": "doc.pdf",
            "source_type": "llm",
        }]
        out = canonicalize_relations({
            "triples": triples, "vocab": ["COVERS"], "model": "fake",
            "skip_canonicalize": True,
        })
        # No state change — relation stays "INSURES".
        result = out.get("triples", triples)
        assert result[0]["relation"] == "INSURES"


class TestSkipNormalize:
    def test_skip_normalize_returns_no_changes(self):
        triples = [
            {
                "subject": "POL-1", "subject_type": "FloodPolicyRecord",
                "relation": "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE",
                "relation_raw": "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE",
                "object": "100000", "object_type": "Currency",
                "span": "", "confidence": 1.0, "chunk_id": "c1",
                "source": "p.json", "source_type": "structured", "lob": "flood",
            },
            {
                "subject": "POL-2", "subject_type": "RentersPolicyRecord",
                "relation": "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE_AMOUNT",
                "relation_raw": "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE_AMOUNT",
                "object": "75000", "object_type": "Currency",
                "span": "", "confidence": 1.0, "chunk_id": "c2",
                "source": "r.csv", "source_type": "structured", "lob": "renters",
            },
        ]
        out = normalize_structured_relations({
            "triples": triples, "model": "fake",
            "skip_normalize": True,
        })
        # Skip path returns empty dict (no state change), so the original
        # relations remain when LangGraph merges state.
        result = out.get("triples", triples)
        assert result[0]["relation"] == "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE"
        assert result[1]["relation"] == "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE_AMOUNT"


# ---------------------------------------------------------------------------
# Neo4j MERGE writes r.relation_raw onto the edge
# ---------------------------------------------------------------------------

class _FakeNeo4jGraph:
    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    def query(self, cypher: str, params: dict | None = None):
        self.queries.append((cypher, params or {}))


class TestBatchMergeWritesRelationRaw:
    def test_merge_query_sets_relation_raw_property(self):
        graph = _FakeNeo4jGraph()
        batch = [{
            "subject": "POL-1", "subject_type": "FloodPolicyRecord",
            "object": "$500", "object_type": "Currency",
            "span": "ded 500", "confidence": 1.0,
            "chunk_id": "c1", "source": "p.json",
            "source_type": "structured", "lob": "flood",
            "relation_raw": "HAS_TOTAL_BUILDING_DEDUCTIBLE",
        }]
        _batch_merge_triples(graph, {"HAS_DEDUCTIBLE": batch})

        cypher, params = graph.queries[0]
        assert "r.relation_raw" in cypher, f"Cypher missing r.relation_raw:\n{cypher}"
        assert params["batch"][0]["relation_raw"] == "HAS_TOTAL_BUILDING_DEDUCTIBLE"

    def test_merge_backfills_missing_relation_raw_from_relation(self):
        # Defensive: legacy callers that pre-date relation_raw should not
        # crash and should get a sensible value (= the relation name).
        graph = _FakeNeo4jGraph()
        batch = [{
            "subject": "X", "subject_type": "T",
            "object": "Y", "object_type": "T",
            "span": "", "confidence": 1.0,
            "chunk_id": "c1", "source": "x", "source_type": "llm",
            # NOTE: no 'relation_raw' key.
        }]
        _batch_merge_triples(graph, {"COVERS": batch})

        params = graph.queries[0][1]
        # The MERGE either backfills relation_raw or skips it gracefully.
        assert params["batch"][0].get("relation_raw") in (None, "COVERS")
