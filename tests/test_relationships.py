"""Tests for zone3.fbi.relationships — algorithmic functions only (no LLM)."""

from __future__ import annotations

import pytest

from zone3.fbi.class_discovery import CandidateClass
from zone3.fbi.relationships import (
    ClassRelationship,
    build_relationship_naming_prompt,
    find_bridge_columns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_class(
    name: str,
    headers: list[str],
    children: list[CandidateClass] | None = None,
) -> CandidateClass:
    return CandidateClass(
        prefix="",
        headers=headers,
        name=name,
        children=children or [],
    )


# ---------------------------------------------------------------------------
# TestFindBridgeColumns
# ---------------------------------------------------------------------------


class TestFindBridgeColumns:
    def test_policy_number_bridges_classes(self) -> None:
        """POLICY_NUMBER in both classes creates a bridge relationship."""
        cls_a = _make_class("Policy", ["POLICY_NUMBER", "POLICY_TYPE", "EFFECTIVE_DATE"])
        cls_b = _make_class("Claim", ["CLAIM_NUMBER", "POLICY_NUMBER", "LOSS_DATE"])

        rels = find_bridge_columns([cls_a, cls_b])

        assert len(rels) == 1
        rel = rels[0]
        assert rel.source_class == "Policy"
        assert rel.target_class == "Claim"
        assert rel.bridge_column == "POLICY_NUMBER"
        assert rel.relationship_name == ""
        assert rel.confidence == 1.0

    def test_no_bridges_with_no_overlap(self) -> None:
        """Two classes with completely different headers produce no bridges."""
        cls_a = _make_class("Policy", ["POLICY_NUMBER", "POLICY_TYPE"])
        cls_b = _make_class("Address", ["STREET", "CITY", "STATE"])

        rels = find_bridge_columns([cls_a, cls_b])

        assert rels == []

    def test_multiple_bridges(self) -> None:
        """Three classes with POLICY_NUMBER and CLAIM_NUMBER bridges."""
        cls_policy = _make_class("Policy", ["POLICY_NUMBER", "POLICY_TYPE"])
        cls_claim = _make_class("Claim", ["CLAIM_NUMBER", "POLICY_NUMBER", "LOSS_DATE"])
        cls_payment = _make_class("Payment", ["CLAIM_NUMBER", "AMOUNT"])

        rels = find_bridge_columns([cls_policy, cls_claim, cls_payment])

        bridges = {(r.source_class, r.target_class, r.bridge_column) for r in rels}

        # POLICY_NUMBER bridges Policy <-> Claim
        assert ("Policy", "Claim", "POLICY_NUMBER") in bridges
        # CLAIM_NUMBER bridges Claim <-> Payment
        assert ("Claim", "Payment", "CLAIM_NUMBER") in bridges
        assert len(rels) == 2

    def test_deduplicates_pairs(self) -> None:
        """Same pair found via parent and child headers yields only 1 relationship."""
        child = _make_class("SubClaim", ["POLICY_NUMBER", "SUB_TYPE"])
        cls_a = _make_class(
            "Policy",
            ["POLICY_NUMBER", "POLICY_TYPE"],
        )
        cls_b = _make_class(
            "Claim",
            ["CLAIM_NUMBER"],
            children=[child],
        )
        # POLICY_NUMBER appears in cls_a.headers and in cls_b's child headers
        # but should only produce one relationship between Policy and Claim

        rels = find_bridge_columns([cls_a, cls_b])

        policy_bridges = [
            r for r in rels if r.bridge_column == "POLICY_NUMBER"
        ]
        assert len(policy_bridges) == 1
        assert policy_bridges[0].source_class == "Policy"
        assert policy_bridges[0].target_class == "Claim"

    def test_single_class_returns_empty(self) -> None:
        """A single class cannot have bridge relationships."""
        cls_a = _make_class("Policy", ["POLICY_NUMBER"])
        assert find_bridge_columns([cls_a]) == []

    def test_empty_list_returns_empty(self) -> None:
        """No classes means no relationships."""
        assert find_bridge_columns([]) == []


# ---------------------------------------------------------------------------
# TestBuildRelationshipNamingPrompt
# ---------------------------------------------------------------------------


class TestBuildRelationshipNamingPrompt:
    def test_build_prompt(self) -> None:
        """Prompt contains class names and bridge column."""
        rels = [
            ClassRelationship(
                source_class="Policy",
                target_class="Claim",
                relationship_name="",
                bridge_column="POLICY_NUMBER",
            ),
        ]

        prompt = build_relationship_naming_prompt(rels)

        assert "Policy" in prompt
        assert "Claim" in prompt
        assert "POLICY_NUMBER" in prompt
        assert "JSON" in prompt

    def test_prompt_multiple_relationships(self) -> None:
        """Prompt includes all relationships."""
        rels = [
            ClassRelationship(
                source_class="Policy",
                target_class="Claim",
                relationship_name="",
                bridge_column="POLICY_NUMBER",
            ),
            ClassRelationship(
                source_class="Claim",
                target_class="Payment",
                relationship_name="",
                bridge_column="CLAIM_NUMBER",
            ),
        ]

        prompt = build_relationship_naming_prompt(rels)

        assert "POLICY_NUMBER" in prompt
        assert "CLAIM_NUMBER" in prompt
        assert "Payment" in prompt
