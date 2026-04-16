"""Tests for zone3.fbi.class_discovery — algorithmic functions only (no LLM)."""

from __future__ import annotations

import pytest

from zone3.fbi.class_discovery import (
    CandidateClass,
    SiblingGroup,
    _longest_common_prefix,
    build_naming_prompt,
    build_semantic_grouping_prompt,
    detect_sibling_patterns,
    find_prefix_groups,
    get_ungrouped_headers,
    merge_cross_file_classes,
)


# ---------------------------------------------------------------------------
# TestLongestCommonPrefix
# ---------------------------------------------------------------------------


class TestLongestCommonPrefix:
    def test_common_prefix(self) -> None:
        assert _longest_common_prefix(["abc", "abd", "abe"]) == "ab"

    def test_identical(self) -> None:
        assert _longest_common_prefix(["foo", "foo", "foo"]) == "foo"

    def test_empty_list(self) -> None:
        assert _longest_common_prefix([]) == ""

    def test_no_common(self) -> None:
        assert _longest_common_prefix(["abc", "xyz"]) == ""


# ---------------------------------------------------------------------------
# TestFindPrefixGroups
# ---------------------------------------------------------------------------


class TestFindPrefixGroups:
    def test_endorsement_prefixes(self) -> None:
        headers = [
            "ENDORSEMENT_EARTHQUAKE_CODE",
            "ENDORSEMENT_EARTHQUAKE_NWP",
            "ENDORSEMENT_EARTHQUAKE_DED_CODE",
            "ENDORSEMENT_EARTHQUAKE_EXPOSURE",
            "ENDORSEMENT_JEWELRY_CODE",
            "ENDORSEMENT_JEWELRY_NWP",
            "ENDORSEMENT_JEWELRY_DED_CODE",
            "ENDORSEMENT_JEWELRY_EXPOSURE",
            "POLICY_NUMBER",
            "POLICY_EFF_DATE",
        ]
        groups = find_prefix_groups(headers, min_group_size=3)

        # Should find ENDORSEMENT_EARTHQUAKE and ENDORSEMENT_JEWELRY
        prefixes = {g.prefix for g in groups}
        assert "ENDORSEMENT_EARTHQUAKE" in prefixes
        assert "ENDORSEMENT_JEWELRY" in prefixes

        # Each group should have 4 headers
        for g in groups:
            if g.prefix.startswith("ENDORSEMENT_"):
                assert len(g.headers) == 4

    def test_no_groups_when_all_unique(self) -> None:
        headers = ["ALPHA", "BETA", "GAMMA"]
        groups = find_prefix_groups(headers, min_group_size=3)
        assert groups == []

    def test_cov_prefixes(self) -> None:
        headers = [
            "COVPROP_LIMIT",
            "COVPROP_DEDUCTIBLE",
            "COVPROP_PREMIUM",
            "COVLIAB_LIMIT",
            "COVLIAB_DEDUCTIBLE",
            "COVLIAB_PREMIUM",
        ]
        groups = find_prefix_groups(headers, min_group_size=3)
        prefixes = {g.prefix for g in groups}
        assert "COVPROP" in prefixes
        assert "COVLIAB" in prefixes
        assert len(groups) == 2

    def test_longer_prefix_takes_priority(self) -> None:
        """More specific prefixes should claim headers before broader ones."""
        headers = [
            "A_B_X",
            "A_B_Y",
            "A_B_Z",
            "A_C_X",
            "A_C_Y",
            "A_C_Z",
        ]
        groups = find_prefix_groups(headers, min_group_size=3)
        prefixes = {g.prefix for g in groups}
        # A_B and A_C should be found (length 3 each),
        # not A (length 1) which would grab all 6
        assert "A_B" in prefixes
        assert "A_C" in prefixes

    def test_empty_headers(self) -> None:
        assert find_prefix_groups([], min_group_size=3) == []


# ---------------------------------------------------------------------------
# TestDetectSiblingPatterns
# ---------------------------------------------------------------------------


class TestDetectSiblingPatterns:
    def test_endorsement_siblings(self) -> None:
        """Two groups with the same suffix pattern → 1 SiblingGroup."""
        group_eq = CandidateClass(
            prefix="ENDORSEMENT_EARTHQUAKE",
            headers=[
                "ENDORSEMENT_EARTHQUAKE_CODE",
                "ENDORSEMENT_EARTHQUAKE_NWP",
                "ENDORSEMENT_EARTHQUAKE_DED_CODE",
                "ENDORSEMENT_EARTHQUAKE_EXPOSURE",
            ],
            suffixes=["CODE", "NWP", "DED_CODE", "EXPOSURE"],
            level=1,
        )
        group_jw = CandidateClass(
            prefix="ENDORSEMENT_JEWELRY",
            headers=[
                "ENDORSEMENT_JEWELRY_CODE",
                "ENDORSEMENT_JEWELRY_NWP",
                "ENDORSEMENT_JEWELRY_DED_CODE",
                "ENDORSEMENT_JEWELRY_EXPOSURE",
            ],
            suffixes=["CODE", "NWP", "DED_CODE", "EXPOSURE"],
            level=1,
        )

        siblings = detect_sibling_patterns([group_eq, group_jw])
        assert len(siblings) == 1
        assert siblings[0].common_prefix == "ENDORSEMENT"
        assert len(siblings[0].children) == 2
        assert set(siblings[0].suffix_pattern) == {"CODE", "NWP", "DED_CODE", "EXPOSURE"}

    def test_no_siblings_when_different_suffixes(self) -> None:
        group_a = CandidateClass(
            prefix="FOO",
            headers=["FOO_X", "FOO_Y"],
            suffixes=["X", "Y"],
            level=1,
        )
        group_b = CandidateClass(
            prefix="BAR",
            headers=["BAR_A", "BAR_B"],
            suffixes=["A", "B"],
            level=1,
        )
        siblings = detect_sibling_patterns([group_a, group_b])
        assert len(siblings) == 0

    def test_single_group_no_siblings(self) -> None:
        group = CandidateClass(
            prefix="FOO",
            headers=["FOO_X"],
            suffixes=["X"],
            level=1,
        )
        assert detect_sibling_patterns([group]) == []


# ---------------------------------------------------------------------------
# TestGetUngroupedHeaders
# ---------------------------------------------------------------------------


class TestGetUngroupedHeaders:
    def test_returns_ungrouped(self) -> None:
        all_headers = ["A_X", "A_Y", "A_Z", "B_SOLO", "C_LONE"]
        groups = [
            CandidateClass(
                prefix="A",
                headers=["A_X", "A_Y", "A_Z"],
                level=1,
            )
        ]
        ungrouped = get_ungrouped_headers(all_headers, groups)
        assert ungrouped == ["B_SOLO", "C_LONE"]

    def test_all_grouped(self) -> None:
        all_headers = ["A_X", "A_Y"]
        groups = [
            CandidateClass(prefix="A", headers=["A_X", "A_Y"], level=1)
        ]
        assert get_ungrouped_headers(all_headers, groups) == []

    def test_none_grouped(self) -> None:
        all_headers = ["X", "Y", "Z"]
        assert get_ungrouped_headers(all_headers, []) == ["X", "Y", "Z"]


# ---------------------------------------------------------------------------
# TestMergeCrossFileClasses
# ---------------------------------------------------------------------------


class TestMergeCrossFileClasses:
    def test_claim_classes_merge(self) -> None:
        """Two CLAIM classes from different files → 1 parent with 2 children."""
        cls_a = CandidateClass(
            prefix="CLAIM",
            headers=["CLAIM_ID", "CLAIM_DATE", "CLAIM_STATUS", "CLAIM_ADJUSTER"],
            source_file="claims_flood.csv",
            level=1,
        )
        cls_b = CandidateClass(
            prefix="CLAIM",
            headers=["CLAIM_ID", "CLAIM_DATE", "CLAIM_STATUS", "CLAIM_REGION"],
            source_file="claims_auto.csv",
            level=1,
        )

        merged = merge_cross_file_classes([cls_a, cls_b])
        assert len(merged) == 1

        parent = merged[0]
        assert len(parent.children) == 2

        # Shared headers should be the intersection
        shared = set(parent.shared_headers)
        assert shared == {"CLAIM_ID", "CLAIM_DATE", "CLAIM_STATUS"}

        # Children should have unique headers
        child_uniques = [set(c.unique_headers) for c in parent.children]
        assert {"CLAIM_ADJUSTER"} in child_uniques
        assert {"CLAIM_REGION"} in child_uniques

    def test_unrelated_classes_stay_separate(self) -> None:
        """CLAIM + SURVEY with no overlap → stays as 2 classes."""
        cls_a = CandidateClass(
            prefix="CLAIM",
            headers=["CLAIM_ID", "CLAIM_DATE", "CLAIM_STATUS"],
            source_file="claims.csv",
            level=1,
        )
        cls_b = CandidateClass(
            prefix="SURVEY",
            headers=["SURVEY_ID", "SURVEY_DATE", "SURVEY_SCORE"],
            source_file="surveys.csv",
            level=1,
        )

        merged = merge_cross_file_classes([cls_a, cls_b])
        assert len(merged) == 2

    def test_single_class_passthrough(self) -> None:
        cls = CandidateClass(
            prefix="POLICY",
            headers=["POLICY_ID"],
            level=1,
        )
        merged = merge_cross_file_classes([cls])
        assert len(merged) == 1
        assert merged[0].prefix == "POLICY"

    def test_jaccard_overlap_merge(self) -> None:
        """Classes with different prefixes but high header overlap should merge."""
        cls_a = CandidateClass(
            prefix="",
            headers=["ID", "NAME", "DATE", "STATUS", "AMOUNT"],
            source_file="file_a.csv",
            level=1,
        )
        cls_b = CandidateClass(
            prefix="",
            headers=["ID", "NAME", "DATE", "STATUS", "REGION"],
            source_file="file_b.csv",
            level=1,
        )

        # Jaccard = 4/6 = 0.667 > 0.3
        merged = merge_cross_file_classes([cls_a, cls_b])
        assert len(merged) == 1
        assert len(merged[0].children) == 2


# ---------------------------------------------------------------------------
# TestBuildNamingPrompt
# ---------------------------------------------------------------------------


class TestBuildNamingPrompt:
    def test_prompt_contains_evidence(self) -> None:
        child_a = CandidateClass(
            prefix="CLAIM",
            headers=["CLAIM_ID", "CLAIM_DATE", "CLAIM_ADJUSTER"],
            unique_headers=["CLAIM_ADJUSTER"],
            shared_headers=["CLAIM_ID", "CLAIM_DATE"],
            source_file="claims_flood.csv",
            level=2,
        )
        child_b = CandidateClass(
            prefix="CLAIM",
            headers=["CLAIM_ID", "CLAIM_DATE", "CLAIM_REGION"],
            unique_headers=["CLAIM_REGION"],
            shared_headers=["CLAIM_ID", "CLAIM_DATE"],
            source_file="claims_auto.csv",
            level=2,
        )
        parent = CandidateClass(
            prefix="CLAIM",
            headers=["CLAIM_ID", "CLAIM_DATE", "CLAIM_ADJUSTER", "CLAIM_REGION"],
            shared_headers=["CLAIM_ID", "CLAIM_DATE"],
            children=[child_a, child_b],
            source_files=["claims_flood.csv", "claims_auto.csv"],
            level=1,
        )

        prompt = build_naming_prompt(parent)

        # Shared headers present
        assert "CLAIM_ID" in prompt
        assert "CLAIM_DATE" in prompt

        # Unique headers present
        assert "CLAIM_ADJUSTER" in prompt
        assert "CLAIM_REGION" in prompt

        # Source files present
        assert "claims_flood.csv" in prompt
        assert "claims_auto.csv" in prompt

        # Contains JSON format instruction
        assert "parent_name" in prompt
        assert "children" in prompt


# ---------------------------------------------------------------------------
# TestBuildSemanticGroupingPrompt
# ---------------------------------------------------------------------------


class TestBuildSemanticGroupingPrompt:
    def test_prompt_contains_headers(self) -> None:
        headers = ["POLICY_NUMBER", "CLAIM_DATE", "DEDUCTIBLE"]
        prompt = build_semantic_grouping_prompt(headers)
        for h in headers:
            assert h in prompt
        assert "groups" in prompt
        assert "JSON" in prompt
