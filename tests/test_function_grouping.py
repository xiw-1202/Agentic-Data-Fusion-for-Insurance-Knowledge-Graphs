"""Tests for zone3.fbi.function_grouping."""

from __future__ import annotations

from zone3.fbi.fingerprint import FileFingerprint
from zone3.fbi.function_grouping import (
    FunctionGroup,
    _compute_dominant_prefixes,
    _compute_merge_score,
    _normalize_token,
    group_files_by_function,
)
from zone3.fbi.token_classifier import TokenClassification, classify_tokens


def make_fp(name: str, tokens: list[str], headers: list[str] | None = None) -> FileFingerprint:
    return FileFingerprint(
        file_path=f"/data/{name}",
        file_type="csv",
        filename_tokens=tokens,
        headers_raw=headers or [],
    )


class TestNormalizeToken:
    def test_strips_trailing_s(self):
        assert _normalize_token("claims") == "claim"
        assert _normalize_token("surveys") == "survey"

    def test_short_tokens_kept(self):
        # Don't strip 's' from short tokens (would lose meaning)
        assert _normalize_token("cds") == "cds"

    def test_lowercase(self):
        assert _normalize_token("CLAIM") == "claim"


class TestComputeDominantPrefixes:
    def test_finds_major_prefix(self):
        headers = ["CLAIM_A", "CLAIM_B", "CLAIM_C", "CLAIM_D", "OTHER"]
        result = _compute_dominant_prefixes(headers, min_size=3)
        assert "CLAIM" in result

    def test_ignores_small_prefixes(self):
        headers = ["A_X", "A_Y", "B_Z"]  # only 2 A_, 1 B_
        result = _compute_dominant_prefixes(headers, min_size=3)
        assert result == set()

    def test_multiple_prefixes(self):
        headers = ["A_1", "A_2", "A_3", "B_1", "B_2", "B_3"]
        result = _compute_dominant_prefixes(headers, min_size=3)
        assert "A" in result and "B" in result


class TestGroupFilesByFunction:
    def test_cross_lob_claim_grouping(self):
        """GEICO claims + TMobile claims should group."""
        fps = [
            make_fp(
                "f1.csv",
                ["geico", "renters", "claims"],
                headers=["CLAIM_NUMBER", "CLAIM_STATUS", "CAUSE_OF_LOSS"],
            ),
            make_fp(
                "f2.csv",
                ["tmobile", "claim"],
                headers=["CLAIM_NUMBER", "CLAIM_STATUS", "DEVICE_DAMAGE_TYPE"],
            ),
            make_fp(
                "f3.csv",
                ["geico", "renters", "policies"],
                headers=["POLICY_NUMBER", "COVERAGE_A"],
            ),
        ]
        tc = classify_tokens(fps)
        groups = group_files_by_function(fps, tc)
        # Find the group containing the claim files
        claim_group = next(
            (g for g in groups if any("f1.csv" in f.file_path for f in g.files)),
            None,
        )
        assert claim_group is not None
        claim_names = {f.file_path.split("/")[-1] for f in claim_group.files}
        assert claim_names == {"f1.csv", "f2.csv"}
        assert "CLAIM" in claim_group.dominant_prefixes

    def test_same_lob_files_dont_merge(self):
        """Files from the same LOB group should be separate functions."""
        fps = [
            make_fp(
                "policy.csv",
                ["geico", "renters", "policies"],
                headers=["POLICY_NUMBER"],
            ),
            make_fp(
                "claim.csv",
                ["geico", "renters", "claims"],
                headers=["CLAIM_NUMBER", "POLICY_NUMBER"],
            ),
        ]
        tc = classify_tokens(fps)
        groups = group_files_by_function(fps, tc)
        # Both are GEICO — they serve different functions, should NOT merge
        assert len(groups) == 2

    def test_singleton_group(self):
        """A file with no cross-LOB match forms its own group."""
        fps = [
            make_fp("unique.csv", ["acme", "widget"], headers=["WIDGET_ID"]),
            make_fp("other.csv", ["zeta", "gadget"], headers=["GADGET_ID"]),
        ]
        tc = classify_tokens(fps)
        groups = group_files_by_function(fps, tc)
        assert len(groups) == 2
        assert all(g.is_singleton for g in groups)

    def test_shared_headers_computed(self):
        fps = [
            make_fp("f1.csv", ["geico", "claim"], headers=["A", "B", "C"]),
            make_fp("f2.csv", ["tmobile", "claim"], headers=["A", "B", "D"]),
        ]
        tc = classify_tokens(fps)
        groups = group_files_by_function(fps, tc)
        merged = [g for g in groups if len(g.files) == 2]
        assert len(merged) == 1
        assert merged[0].shared_headers == {"A", "B"}

    def test_fuzzy_claim_claims_match(self):
        """'claim' and 'claims' should match via normalization."""
        fps = [
            make_fp("f1.csv", ["geico", "claims"], headers=["X"]),
            make_fp("f2.csv", ["tmobile", "claim"], headers=["Y"]),
        ]
        tc = classify_tokens(fps)
        groups = group_files_by_function(fps, tc)
        # Should merge despite "claim" vs "claims" difference
        merged = [g for g in groups if len(g.files) == 2]
        assert len(merged) == 1

    def test_empty_input(self):
        tc = TokenClassification()
        assert group_files_by_function([], tc) == []

    def test_real_emory_data(self):
        """Smoke test with real filename patterns."""
        fps = [
            make_fp(
                "geicorentersclaims.csv",
                ["geico", "renters", "claims"],
                headers=["CLAIM_NUMBER", "CLAIM_STATUS", "POLICY_NUMBER"],
            ),
            make_fp(
                "tmobileclaimsample.csv",
                ["tmobile", "claim"],
                headers=["CLAIM_NUMBER", "CLAIM_STATUS", "DEVICE_TYPE"],
            ),
            make_fp(
                "geicorenterssurvey.csv",
                ["geico", "renters", "survey"],
                headers=["SURVEY_ID", "NPS"],
            ),
            make_fp(
                "tmobilesurveysample.csv",
                ["tmobile", "survey"],
                headers=["SURVEY_ID", "NPS", "CSAT"],
            ),
            make_fp(
                "geicorenterspoliciesdetails.csv",
                ["geico", "renters", "policies", "details"],
                headers=["POLICY_NUMBER", "COVAMT_PERS"],
            ),
        ]
        tc = classify_tokens(fps)
        groups = group_files_by_function(fps, tc)
        # Expect 3 groups: Claim (2 files), Survey (2 files), Policy (1 file)
        sizes = sorted(len(g.files) for g in groups)
        assert sizes == [1, 2, 2]
