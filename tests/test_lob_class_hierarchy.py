"""Tests for LOB-aware class hierarchy in zone2.structured_mapper (Phase 2).

The deterministic CSV path emits three levels per record::

    (POL-X)              -[:IS_A]-> (FloodPolicyRecord)   instance → LOB-specific
    (FloodPolicyRecord)  -[:IS_A]-> (PolicyRecord)         LOB-specific → general
    (PolicyRecord)       -[:IS_A]-> (Record)               general → root

Class-chain triples are emitted ONCE per pipeline run regardless of how
many records share the same LOB.  Back-compat: when ``lob`` is omitted
or ``"generic"``, the LOB level is collapsed and the chain is just
``PolicyRecord IS_A Record``.
"""

from __future__ import annotations

from zone2.structured_mapper import (
    record_to_triples,
    extract_structured,
)


# ---------------------------------------------------------------------------
# record_to_triples — instance-to-specific subject typing
# ---------------------------------------------------------------------------

class TestRecordToTriplesLOBTyping:
    """When a record carries lob, its entity_type and IS_A target use the
    LOB-prefixed name (e.g. FloodPolicyRecord).  Otherwise back-compat."""

    def test_flood_policy_record_uses_lob_prefixed_entity_type(self):
        rec = {"policy effective date": "2025-01-10", "policy cost": "500"}
        triples = record_to_triples(
            record=rec, record_type="policies",
            chunk_id="c1", source="policies_sample.json",
            lob="flood",
        )
        # IS_A target should be FloodPolicyRecord, not PolicyRecord.
        is_a = [t for t in triples if t["relation"] == "IS_A"]
        assert len(is_a) == 1
        assert is_a[0]["object"] == "FloodPolicyRecord"
        # subject_type stamped on every triple should be FloodPolicyRecord.
        for t in triples:
            assert t["subject_type"] == "FloodPolicyRecord"

    def test_auto_claim_record_uses_lob_prefixed_entity_type(self):
        rec = {"vin": "1HGBH41JXMN109186", "mileage": "45000"}
        triples = record_to_triples(
            record=rec, record_type="claims",
            chunk_id="c1", source="auto_claims.csv",
            lob="auto",
        )
        is_a = [t for t in triples if t["relation"] == "IS_A"]
        assert is_a[0]["object"] == "AutoClaimRecord"

    def test_generic_lob_falls_back_to_unprefixed_entity_type(self):
        rec = {"name": "X", "value": "Y"}
        triples = record_to_triples(
            record=rec, record_type="policies",
            chunk_id="c1", source="x.csv",
            lob="generic",
        )
        is_a = [t for t in triples if t["relation"] == "IS_A"]
        assert is_a[0]["object"] == "PolicyRecord"

    def test_lob_omitted_preserves_legacy_behavior(self):
        # No lob argument → behaves exactly like the pre-Phase-2 version.
        rec = {"name": "X"}
        triples = record_to_triples(
            record=rec, record_type="claims",
            chunk_id="c1", source="x.csv",
        )
        is_a = [t for t in triples if t["relation"] == "IS_A"]
        assert is_a[0]["object"] == "ClaimRecord"

    def test_renters_lob_for_renters_record(self):
        rec = {"name": "X"}
        triples = record_to_triples(
            record=rec, record_type="policies",
            chunk_id="c1", source="renters.csv",
            lob="renters",
        )
        is_a = [t for t in triples if t["relation"] == "IS_A"]
        assert is_a[0]["object"] == "RentersPolicyRecord"

    def test_lender_placed_compound_lob_capitalizes_correctly(self):
        # "lender_placed" → "LenderPlacedPolicyRecord" (snake → PascalCase).
        rec = {"name": "X"}
        triples = record_to_triples(
            record=rec, record_type="policies",
            chunk_id="c1", source="lender_placed.csv",
            lob="lender_placed",
        )
        is_a = [t for t in triples if t["relation"] == "IS_A"]
        assert is_a[0]["object"] == "LenderPlacedPolicyRecord"


# ---------------------------------------------------------------------------
# extract_structured — class-chain triples
# ---------------------------------------------------------------------------

CHUNK_FLOOD_POLICIES = {
    "content": (
        "DATASET: OpenFEMA NFIP Policies\n\n"
        "RECORD 0:\n"
        "  [Policy] policy effective date: 2025-01-10 | policy cost: 500\n"
    ),
    "source": "policies_sample.json",
    "section_hierarchy": ["FimaNfipPolicies", "records 0–0"],
    "pages": [],
    "chunk_type": "text",
    "lob": "flood",
}

CHUNK_FLOOD_POLICIES_2 = {
    "content": (
        "DATASET: OpenFEMA NFIP Policies\n\n"
        "RECORD 0:\n"
        "  [Policy] policy effective date: 2026-02-15 | policy cost: 700\n"
    ),
    "source": "policies_sample.json",
    "section_hierarchy": ["FimaNfipPolicies", "records 0–0"],
    "pages": [],
    "chunk_type": "text",
    "lob": "flood",
}

CHUNK_AUTO_CLAIMS = {
    "content": (
        "DATASET: Auto Claims\n\n"
        "RECORD 0:\n"
        "  [Vehicle] vin: 1HGBH41JXMN | mileage: 45000\n"
    ),
    "source": "auto_claims.csv",
    "section_hierarchy": ["auto_claims.csv", "records 0–0"],
    "pages": [],
    "chunk_type": "text",
    "lob": "auto",
}

CHUNK_GENERIC_SURVEY = {
    "content": (
        "DATASET: Survey\n\n"
        "RECORD 0:\n"
        "  [Identity] survey id: SRV-1 | client name: ACME\n"
    ),
    "source": "synthetic_data_sample_survey_responses.csv",
    "section_hierarchy": ["survey_responses.csv", "records 0–0"],
    "pages": [],
    "chunk_type": "text",
    "lob": "generic",
}


class TestClassChainEmission:
    """extract_structured emits class chain triples once per unique
    (lob, record_type) combo plus a single root triple per record_type.
    """

    @staticmethod
    def _class_triples(triples: list[dict]) -> list[tuple[str, str]]:
        """Return [(subject, object), ...] for IS_A triples whose subject
        is a class name (not a per-record key)."""
        return [
            (t["subject"], t["object"])
            for t in triples
            if t["relation"] == "IS_A"
            and not t["subject"].startswith(("POL-", "CLM-", "SRV-",
                                              "CHAT-", "EMAIL-", "REC-"))
        ]

    def test_flood_chain_emitted(self):
        out = extract_structured({"chunks": [CHUNK_FLOOD_POLICIES]})
        triples = out["structured_triples"]
        chain = self._class_triples(triples)
        assert ("FloodPolicyRecord", "PolicyRecord") in chain
        assert ("PolicyRecord", "Record") in chain

    def test_class_chain_deduplicated_across_records(self):
        # Two flood-policy chunks → only one (FloodPolicyRecord, IS_A,
        # PolicyRecord) and one (PolicyRecord, IS_A, Record).
        out = extract_structured({
            "chunks": [CHUNK_FLOOD_POLICIES, CHUNK_FLOOD_POLICIES_2]
        })
        chain = self._class_triples(out["structured_triples"])
        assert chain.count(("FloodPolicyRecord", "PolicyRecord")) == 1
        assert chain.count(("PolicyRecord", "Record")) == 1

    def test_multiple_lobs_share_root_but_have_distinct_lob_layers(self):
        out = extract_structured({
            "chunks": [CHUNK_FLOOD_POLICIES, CHUNK_AUTO_CLAIMS]
        })
        chain = self._class_triples(out["structured_triples"])
        # Distinct LOB → general edges
        assert ("FloodPolicyRecord", "PolicyRecord") in chain
        assert ("AutoClaimRecord", "ClaimRecord") in chain
        # Each general type → Record root, exactly once
        assert chain.count(("PolicyRecord", "Record")) == 1
        assert chain.count(("ClaimRecord", "Record")) == 1

    def test_generic_lob_skips_lob_specific_layer(self):
        out = extract_structured({"chunks": [CHUNK_GENERIC_SURVEY]})
        chain = self._class_triples(out["structured_triples"])
        # No "GenericSurveyRecord IS_A SurveyRecord" — the LOB layer is collapsed.
        assert ("SurveyRecord", "Record") in chain
        for s, _o in chain:
            assert "Generic" not in s

    def test_class_chain_triples_marked_as_structured(self):
        out = extract_structured({"chunks": [CHUNK_FLOOD_POLICIES]})
        # All class triples should be marked source_type=structured so they
        # round-trip through dedup + Neo4j MERGE the same way as record triples.
        for t in out["structured_triples"]:
            if t["relation"] == "IS_A" and not t["subject"].startswith(("POL-", "CLM-")):
                assert t["source_type"] == "structured"
                assert t["confidence"] == 1.0

    def test_class_chain_triples_have_ontology_object_type(self):
        # Class-level subjects/objects should carry an ontology-style type
        # so Zone 3 induction can recognize them as ontology nodes.
        out = extract_structured({"chunks": [CHUNK_FLOOD_POLICIES]})
        for t in out["structured_triples"]:
            if t["relation"] == "IS_A" and t["subject"] == "FloodPolicyRecord":
                assert t["object_type"] in ("RecordType", "OntologyClass")
                assert t["subject_type"] in ("RecordType", "OntologyClass")
