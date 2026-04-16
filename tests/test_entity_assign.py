"""Tests for zone3.fbi.entity_assign — algorithmic functions only (no Neo4j)."""

from __future__ import annotations

import pytest

from zone3.fbi.class_discovery import CandidateClass
from zone3.fbi.entity_assign import (
    assign_all_entities,
    assign_entity_to_class,
    build_file_to_class_map,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_parent_with_children() -> CandidateClass:
    """Parent 'Property' from claims.csv + policy.csv; child 'Claim' from claims.csv."""
    child = CandidateClass(
        prefix="claim",
        name="Claim",
        headers=["claim_id", "claim_amount"],
        source_files=["claims.csv"],
        level=2,
    )
    parent = CandidateClass(
        prefix="property",
        name="Property",
        headers=["property_id", "claim_id", "claim_amount", "policy_number"],
        source_files=["claims.csv", "policy.csv"],
        children=[child],
        level=1,
    )
    child.parent = parent
    return parent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildFileToClassMap:
    def test_maps_files_to_classes(self) -> None:
        """Children override parent for the same source file."""
        parent = _make_parent_with_children()
        mapping = build_file_to_class_map([parent])

        # claims.csv mapped by parent first, then overridden by child
        assert mapping["claims.csv"] == "Claim"
        # policy.csv only in parent
        assert mapping["policy.csv"] == "Property"

    def test_empty_classes(self) -> None:
        assert build_file_to_class_map([]) == {}

    def test_uses_prefix_when_no_name(self) -> None:
        cls = CandidateClass(
            prefix="hazard",
            name="",
            headers=["h1"],
            source_files=["hazard_data.csv"],
            level=1,
        )
        mapping = build_file_to_class_map([cls])
        assert mapping["hazard_data.csv"] == "hazard"


class TestAssignEntityToClass:
    def test_single_source_assignment(self) -> None:
        """Entity in 1 file -> direct assignment."""
        file_map = {"claims.csv": "Claim", "policy.csv": "Property"}
        entity = {"id": "e1", "source_files": {"claims.csv"}}
        assert assign_entity_to_class(entity, file_map) == "Claim"

    def test_multi_source_picks_most_common(self) -> None:
        """Entity in 3 files, 2 map to Survey -> Survey wins."""
        file_map = {
            "survey_a.csv": "Survey",
            "survey_b.csv": "Survey",
            "claims.csv": "Claim",
        }
        entity = {
            "id": "e2",
            "source_files": {"survey_a.csv", "survey_b.csv", "claims.csv"},
        }
        assert assign_entity_to_class(entity, file_map) == "Survey"

    def test_unclassified_when_no_match(self) -> None:
        """Entity with unknown source -> Unclassified."""
        file_map = {"claims.csv": "Claim"}
        entity = {"id": "e3", "source_files": {"unknown.csv"}}
        assert assign_entity_to_class(entity, file_map) == "Unclassified"

    def test_unclassified_when_empty_sources(self) -> None:
        file_map = {"claims.csv": "Claim"}
        entity = {"id": "e4", "source_files": set()}
        assert assign_entity_to_class(entity, file_map) == "Unclassified"

    def test_unclassified_when_no_source_files_key(self) -> None:
        file_map = {"claims.csv": "Claim"}
        entity = {"id": "e5"}
        assert assign_entity_to_class(entity, file_map) == "Unclassified"


class TestAssignAllEntities:
    def test_assign_all(self) -> None:
        """Multiple entities -> all get assigned correctly."""
        parent = _make_parent_with_children()
        entities = [
            {"id": "e1", "source_files": {"claims.csv"}},
            {"id": "e2", "source_files": {"policy.csv"}},
            {"id": "e3", "source_files": {"unknown.csv"}},
        ]
        result = assign_all_entities(entities, [parent])

        assert result["e1"] == "Claim"
        assert result["e2"] == "Property"
        assert result["e3"] == "Unclassified"

    def test_assign_all_empty(self) -> None:
        result = assign_all_entities([], [])
        assert result == {}
