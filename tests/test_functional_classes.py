"""Tests for zone3.fbi.functional_classes (Phase 2 Tasks 3+4)."""

from __future__ import annotations

from zone3.fbi.class_discovery import CandidateClass
from zone3.fbi.fingerprint import FileFingerprint
from zone3.fbi.function_grouping import FunctionGroup
from zone3.fbi.functional_classes import (
    _attach_sibling_subclasses,
    _build_multi_file_class,
    _build_singleton_class,
    build_functional_classes,
)


def make_fp(
    name: str,
    headers: list[str] | None,
    file_type: str = "csv",
    sections: list[str] | None = None,
    defined_terms: list[str] | None = None,
) -> FileFingerprint:
    return FileFingerprint(
        file_path=f"/data/{name}",
        file_type=file_type,
        filename_tokens=[],
        headers_raw=headers or [],
        sections=sections or [],
        defined_terms=defined_terms or [],
    )


class TestBuildMultiFileClass:
    def test_shared_and_unique_headers(self):
        group = FunctionGroup(
            function_tokens={"claim"},
            files=[
                make_fp("g.csv", ["CLAIM_NUMBER", "CLAIM_STATUS", "CAUSE_OF_LOSS"]),
                make_fp("t.csv", ["CLAIM_NUMBER", "CLAIM_STATUS", "DEVICE_TYPE"]),
            ],
            dominant_prefixes={"CLAIM"},
            shared_headers={"CLAIM_NUMBER", "CLAIM_STATUS"},
        )
        parent = _build_multi_file_class(group)
        assert set(parent.headers) == {"CLAIM_NUMBER", "CLAIM_STATUS"}
        assert parent.level == 1
        assert len(parent.children) == 2
        # Child 1: GEICO file
        child_g = next(c for c in parent.children if "g.csv" in c.source_file)
        assert "CAUSE_OF_LOSS" in child_g.unique_headers
        assert "CLAIM_NUMBER" not in child_g.unique_headers  # in shared
        assert child_g.level == 2
        assert child_g.parent is parent

    def test_sets_source_files(self):
        group = FunctionGroup(
            function_tokens={"x"},
            files=[make_fp("a.csv", ["A"]), make_fp("b.csv", ["A", "B"])],
            dominant_prefixes=set(),
            shared_headers={"A"},
        )
        parent = _build_multi_file_class(group)
        assert len(parent.source_files) == 2


class TestBuildSingletonClass:
    def test_csv_uses_headers(self):
        group = FunctionGroup(
            function_tokens={"policy"},
            files=[make_fp("p.csv", ["POLICY_NUMBER", "COVAMT_A"])],
            dominant_prefixes=set(),
            shared_headers={"POLICY_NUMBER", "COVAMT_A"},
        )
        cls = _build_singleton_class(group)
        assert set(cls.headers) == {"POLICY_NUMBER", "COVAMT_A"}
        assert cls.level == 1
        assert cls.children == []

    def test_pdf_uses_sections(self):
        group = FunctionGroup(
            function_tokens={"service"},
            files=[make_fp("doc.pdf", [], file_type="pdf",
                           sections=["VEHICLE INFORMATION", "KEY TERMS"],
                           defined_terms=["Breakdown", "Cost"])],
            dominant_prefixes=set(),
            shared_headers=set(),
        )
        cls = _build_singleton_class(group)
        assert "VEHICLE INFORMATION" in cls.headers
        assert "Breakdown" in cls.headers


class TestAttachSiblingSubclasses:
    def test_endorsement_siblings_become_subclasses(self):
        # 3 endorsement types × 4 suffixes, all sharing the same suffix pattern
        headers = []
        for subtype in ["EARTHQUAKE", "JEWELRY", "WATERDMG"]:
            for suffix in ["CODE", "NWP", "DED_CODE", "EXPOSURE"]:
                headers.append(f"ENDORSEMENT_{subtype}_{suffix}")
        cls = CandidateClass(prefix="Policy", headers=headers, level=1)
        _attach_sibling_subclasses(cls)
        # Should detect the sibling pattern and create a sub-class
        assert len(cls.children) >= 1
        sibling_parent = cls.children[0]
        assert sibling_parent.level == 2
        # Siblings as its children
        assert len(sibling_parent.children) == 3  # 3 endorsement types
        for sibling_child in sibling_parent.children:
            assert sibling_child.level == 3
            assert sibling_child.parent is sibling_parent

    def test_non_sibling_prefix_not_promoted(self):
        # Headers with a prefix but no sibling pattern
        headers = ["BILLING_DATE", "BILLING_CODE", "BILLING_PLAN"]
        cls = CandidateClass(prefix="Policy", headers=headers, level=1)
        _attach_sibling_subclasses(cls)
        # No sibling pattern → no children added
        assert cls.children == []

    def test_empty_headers(self):
        cls = CandidateClass(prefix="Empty", headers=[], level=1)
        _attach_sibling_subclasses(cls)
        assert cls.children == []


class TestBuildFunctionalClasses:
    def test_full_pipeline(self):
        groups = [
            # Multi-file Claim group
            FunctionGroup(
                function_tokens={"claim"},
                files=[
                    make_fp("geico_claims.csv",
                            ["CLAIM_NUMBER", "CLAIM_STATUS", "CAUSE_OF_LOSS"]),
                    make_fp("tmobile_claim.csv",
                            ["CLAIM_NUMBER", "CLAIM_STATUS", "DEVICE_TYPE"]),
                ],
                dominant_prefixes={"CLAIM"},
                shared_headers={"CLAIM_NUMBER", "CLAIM_STATUS"},
            ),
            # Singleton Policy group
            FunctionGroup(
                function_tokens={"policy"},
                files=[make_fp("policy.csv",
                               ["POLICY_NUMBER", "COVAMT_PERS", "COVAMT_LIAB"])],
                dominant_prefixes=set(),
                shared_headers={"POLICY_NUMBER", "COVAMT_PERS", "COVAMT_LIAB"},
            ),
        ]
        classes = build_functional_classes(groups)
        assert len(classes) == 2
        # Claim class has 2 children (LOB variants)
        claim = next(c for c in classes if len(c.children) >= 2)
        assert len(claim.children) == 2

    def test_policy_with_endorsement_siblings(self):
        """A singleton class with sibling patterns inside gets sub-hierarchy."""
        headers = ["POLICY_NUMBER", "EFF_DATE"]
        for subtype in ["EARTHQUAKE", "JEWELRY", "WATERDMG"]:
            for suffix in ["CODE", "NWP", "DED_CODE", "EXPOSURE"]:
                headers.append(f"ENDORSEMENT_{subtype}_{suffix}")
        groups = [
            FunctionGroup(
                function_tokens={"policy"},
                files=[make_fp("policy.csv", headers)],
                dominant_prefixes=set(),
                shared_headers=set(headers),
            ),
        ]
        classes = build_functional_classes(groups)
        assert len(classes) == 1
        policy = classes[0]
        # Should have detected Endorsement sibling pattern
        assert len(policy.children) >= 1
        endorsement = policy.children[0]
        assert len(endorsement.children) == 3  # 3 endorsement subtypes

    def test_empty_input(self):
        assert build_functional_classes([]) == []
