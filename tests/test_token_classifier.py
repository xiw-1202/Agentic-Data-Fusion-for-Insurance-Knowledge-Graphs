"""Tests for zone3.fbi.token_classifier — LOB vs function token separation."""

from __future__ import annotations

from zone3.fbi.fingerprint import FileFingerprint
from zone3.fbi.token_classifier import TokenClassification, classify_tokens


def make_fp(name: str, tokens: list[str]) -> FileFingerprint:
    return FileFingerprint(
        file_path=f"/data/{name}",
        file_type="csv",
        filename_tokens=tokens,
    )


class TestClassifyTokens:
    def test_basic_lob_vs_function(self) -> None:
        """geico+renters appear together -> LOB; claim/claims appear across groups -> function."""
        fps = [
            make_fp("f1.csv", ["geico", "renters", "claims"]),
            make_fp("f2.csv", ["geico", "renters", "survey"]),
            make_fp("f3.csv", ["geico", "renters", "policies"]),
            make_fp("f4.csv", ["tmobile", "claim"]),
            make_fp("f5.csv", ["tmobile", "survey"]),
        ]
        result = classify_tokens(fps)
        assert "geico" in result.lob_tokens
        # "renters" normalizes to "renter"
        assert "renter" in result.lob_tokens
        assert "tmobile" in result.lob_tokens
        # survey appears in both GEICO and TMobile groups -> function
        assert "survey" in result.function_tokens
        # "claim" and "claims" both normalize to "claim" before classification
        assert "claim" in result.function_tokens

    def test_modifier_tokens(self) -> None:
        """Tokens appearing in only 1 file next to another non-LOB token are modifiers."""
        fps = [
            make_fp("f1.csv", ["geico", "renters", "claims"]),
            make_fp("f2.csv", ["geico", "renters", "cancel", "survey"]),
            make_fp("f3.csv", ["tmobile", "survey"]),
        ]
        result = classify_tokens(fps)
        # "cancel" appears in only 1 file and coexists with "survey" -> modifier
        assert "cancel" in result.modifier_tokens

    def test_all_unique_files(self) -> None:
        """When no LOB groups form, shared tokens are still function tokens."""
        fps = [
            make_fp("f1.csv", ["apple", "claim"]),
            make_fp("f2.csv", ["banana", "claim"]),
            make_fp("f3.csv", ["cherry", "survey"]),
        ]
        result = classify_tokens(fps)
        # "claim" appears in 2 files but they don't share LOB tokens
        assert "claim" in result.function_tokens

    def test_lob_group_clustering(self) -> None:
        """Files with shared LOB tokens cluster together."""
        fps = [
            make_fp("f1.csv", ["geico", "renters", "a"]),
            make_fp("f2.csv", ["geico", "renters", "b"]),
            make_fp("f3.csv", ["tmobile", "x"]),
            make_fp("f4.csv", ["tmobile", "y"]),
        ]
        result = classify_tokens(fps)
        # Should form 2 LOB groups (geico+renters, tmobile)
        assert len(result.lob_groups) == 2

    def test_empty_input(self) -> None:
        result = classify_tokens([])
        assert result.lob_tokens == set()
        assert result.function_tokens == set()
        assert result.modifier_tokens == set()
        assert result.lob_groups == []

    def test_real_emory_data(self) -> None:
        """Test against the actual filenames from Emory_Spring2026 data."""
        fps = [
            make_fp("geicorenterssurveysample.csv", ["geico", "renters", "survey"]),
            make_fp("geicorenterscancelsurvey.csv", ["geico", "renters", "cancel", "survey"]),
            make_fp("geicorentersclaims.csv", ["geico", "renters", "claims"]),
            make_fp("geicorenterspoliciesdetails.csv", ["geico", "renters", "policies", "details"]),
            make_fp("geicorenterssurvey.csv", ["geico", "renters", "survey"]),
            make_fp("tmobilechatsurveysample.csv", ["tmobile", "chat", "survey"]),
            make_fp("tmobileclaimsample.csv", ["tmobile", "claim"]),
            make_fp("tmobilesurveysample.csv", ["tmobile", "survey"]),
            make_fp("auto_service_form.pdf", ["auto", "service", "form"]),
        ]
        result = classify_tokens(fps)
        # Expected LOB tokens
        assert "geico" in result.lob_tokens
        # "renters" normalizes to "renter"
        assert "renter" in result.lob_tokens
        assert "tmobile" in result.lob_tokens
        # Expected function tokens (cross-LOB)
        assert "survey" in result.function_tokens
        # "claim" (normalized from both "claim" and "claims") must be function
        assert "claim" in result.function_tokens

    def test_claim_claims_normalize_together(self) -> None:
        """After normalization, 'claim' and 'claims' should both map to 'claim'."""
        # Two files per LOB so each LOB forms a real cluster (>= 2 shared tokens).
        fps = [
            make_fp("geicorentersclaims.csv", ["geico", "renters", "claims"]),
            make_fp("geicorenterspolicy.csv", ["geico", "renters", "policy"]),
            make_fp("tmobileclaim.csv", ["tmobile", "wireless", "claim"]),
            make_fp("tmobilesurvey.csv", ["tmobile", "wireless", "survey"]),
        ]
        result = classify_tokens(fps)
        # "claim" (normalized from "claims") should be function token — it
        # appears in both GEICO and TMobile clusters after normalization.
        assert "claim" in result.function_tokens, (
            f"Expected 'claim' in function_tokens, got {result.function_tokens}"
        )
        # "geico" and "tmobile" should be LOB tokens
        assert "geico" in result.lob_tokens
        assert "tmobile" in result.lob_tokens


class TestHelpers:
    def test_token_index_basic(self) -> None:
        from zone3.fbi.token_classifier import _build_token_index

        fps = [
            make_fp("a.csv", ["x", "y"]),
            make_fp("b.csv", ["y", "z"]),
        ]
        idx = _build_token_index(fps)
        assert idx["x"] == {"a.csv"}
        assert idx["y"] == {"a.csv", "b.csv"}
        assert idx["z"] == {"b.csv"}

    def test_cluster_files_strict(self) -> None:
        from zone3.fbi.token_classifier import _cluster_files_by_shared_tokens

        fps = [
            make_fp("a.csv", ["x", "y", "z"]),
            make_fp("b.csv", ["x", "y", "w"]),
            make_fp("c.csv", ["q"]),
        ]
        clusters = _cluster_files_by_shared_tokens(fps, min_shared=2)
        # a and b share {x,y} -> cluster together; c alone
        cluster_sets = sorted([tuple(sorted(c)) for c in clusters])
        assert ("a.csv", "b.csv") in cluster_sets
        assert ("c.csv",) in cluster_sets
