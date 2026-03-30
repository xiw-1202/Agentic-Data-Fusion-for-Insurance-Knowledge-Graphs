"""Tests for SEAF-KG Stage 3 — Cross-Source Entity Linker."""

import math
import pytest

from zone2.cross_source_linker import (
    classify_field_type,
    compare_values,
    check_temporal_consistency,
    multi_pass_blocking,
    score_pair,
)


# ── Field Type Detection ───────────────────────────────────────────────


class TestClassifyFieldType:

    def test_date_from_relation_name(self) -> None:
        assert classify_field_type("HAS_POLICY_EFFECTIVE_DATE", []) == "date"
        assert classify_field_type("HAS_DATE_OF_LOSS", []) == "date"
        assert classify_field_type("HAS_EXPIRATION_DATE", []) == "date"

    def test_numeric_from_relation_name(self) -> None:
        assert classify_field_type("HAS_TOTAL_BUILDING_INSURANCE_COVERAGE", []) == "numeric"
        assert classify_field_type("HAS_AMOUNT_PAID_ON_BUILDING_CLAIM", []) == "numeric"
        assert classify_field_type("HAS_DEDUCTIBLE", []) == "numeric"

    def test_date_from_values(self) -> None:
        vals = ["2020-01-15", "2021-03-22", "2019-12-01"]
        assert classify_field_type("HAS_SOME_FIELD", vals) == "date"

    def test_numeric_from_values(self) -> None:
        vals = ["250000", "100000", "75000.50"]
        assert classify_field_type("HAS_SOME_FIELD", vals) == "numeric"

    def test_categorical_short_values(self) -> None:
        vals = ["AE", "X", "A", "VE"]
        assert classify_field_type("HAS_ZONE", vals) == "categorical"

    def test_text_fallback(self) -> None:
        vals = ["HILLSBOROUGH COUNTY", "MIAMI-DADE COUNTY"]
        assert classify_field_type("HAS_COMMUNITY", vals) == "text"


# ── Value Comparators ──────────────────────────────────────────────────


class TestCompareValues:

    def test_categorical_exact_match(self) -> None:
        assert compare_values("AE", "AE", "categorical") == 1.0

    def test_categorical_no_match(self) -> None:
        assert compare_values("AE", "X", "categorical") == 0.0

    def test_numeric_within_tolerance(self) -> None:
        # 100000 vs 100500 → 0.5% difference, within 5% tolerance
        assert compare_values("100000", "100500", "numeric") == 1.0

    def test_numeric_outside_tolerance(self) -> None:
        # 100000 vs 200000 → 50% difference
        assert compare_values("100000", "200000", "numeric") == 0.0

    def test_numeric_with_currency_symbols(self) -> None:
        assert compare_values("$250,000", "$250,000", "numeric") == 1.0

    def test_numeric_zero_handling(self) -> None:
        assert compare_values("0", "0", "numeric") == 1.0

    def test_date_same_day(self) -> None:
        assert compare_values("2020-10-29", "2020-10-29", "date") == 1.0

    def test_date_within_proximity(self) -> None:
        # 15 days apart → 0.5
        assert compare_values("2020-10-29", "2020-11-13", "date") == 0.5

    def test_date_outside_proximity(self) -> None:
        assert compare_values("2020-01-01", "2020-06-01", "date") == 0.0

    def test_date_iso_with_time(self) -> None:
        assert compare_values(
            "2020-10-29T00:00:00.000Z",
            "2020-10-29T12:30:00.000Z",
            "date",
        ) == 1.0

    def test_text_normalized(self) -> None:
        assert compare_values("  Miami  ", "miami", "text") == 1.0

    def test_text_no_match(self) -> None:
        assert compare_values("Miami", "Tampa", "text") == 0.0


# ── Temporal Consistency ───────────────────────────────────────────────


class TestTemporalConsistency:

    def test_claim_within_policy_window(self) -> None:
        policy = {
            "HAS_POLICY_EFFECTIVE_DATE": "2020-01-01",
            "HAS_POLICY_EXPIRATION_DATE": "2021-01-01",
        }
        claim = {"HAS_DATE_OF_LOSS": "2020-06-15"}
        assert check_temporal_consistency(policy, claim) is True

    def test_claim_before_policy(self) -> None:
        policy = {"HAS_POLICY_EFFECTIVE_DATE": "2020-01-01"}
        claim = {"HAS_DATE_OF_LOSS": "2019-06-15"}
        assert check_temporal_consistency(policy, claim) is False

    def test_claim_after_expiration(self) -> None:
        policy = {
            "HAS_POLICY_EFFECTIVE_DATE": "2020-01-01",
            "HAS_POLICY_EXPIRATION_DATE": "2021-01-01",
        }
        claim = {"HAS_DATE_OF_LOSS": "2021-06-15"}
        assert check_temporal_consistency(policy, claim) is False

    def test_no_date_fields_allows_link(self) -> None:
        policy = {"HAS_RATED_FLOOD_ZONE": "AE"}
        claim = {"HAS_RATED_FLOOD_ZONE": "AE"}
        assert check_temporal_consistency(policy, claim) is True

    def test_policy_without_expiration(self) -> None:
        policy = {"HAS_POLICY_EFFECTIVE_DATE": "2020-01-01"}
        claim = {"HAS_DATE_OF_LOSS": "2025-06-15"}
        # No expiration → only check effective date.
        assert check_temporal_consistency(policy, claim) is True


# ── Multi-Pass Blocking ────────────────────────────────────────────────


class TestMultiPassBlocking:

    def test_blocking_by_shared_field(self) -> None:
        profiles_a = {
            "POL-1": {"HAS_ZONE": "AE", "HAS_COVERAGE": "250000"},
            "POL-2": {"HAS_ZONE": "X", "HAS_COVERAGE": "100000"},
        }
        profiles_b = {
            "CLM-1": {"HAS_ZONE": "AE", "HAS_COVERAGE": "250000"},
            "CLM-2": {"HAS_ZONE": "X", "HAS_COVERAGE": "50000"},
        }
        shared = [
            {"relation": "HAS_COVERAGE", "idf_weight": 3.0},
            {"relation": "HAS_ZONE", "idf_weight": 1.0},
        ]

        candidates = multi_pass_blocking(profiles_a, profiles_b, shared)

        # Pass 1 (HAS_COVERAGE): POL-1↔CLM-1 (both 250000)
        # Pass 2 (HAS_ZONE): POL-1↔CLM-1 (AE), POL-2↔CLM-2 (X)
        assert ("POL-1", "CLM-1") in candidates
        assert ("POL-2", "CLM-2") in candidates
        # POL-1 and CLM-2 should NOT be paired (different zone AND coverage)
        assert ("POL-1", "CLM-2") not in candidates

    def test_empty_shared_rels_brute_force(self) -> None:
        profiles_a = {"A": {}}
        profiles_b = {"B": {}}
        candidates = multi_pass_blocking(profiles_a, profiles_b, [])
        assert ("A", "B") in candidates

    def test_missing_blocking_value_skipped(self) -> None:
        profiles_a = {"POL-1": {"HAS_ZONE": "AE"}}
        profiles_b = {"CLM-1": {}}  # no zone value
        shared = [{"relation": "HAS_ZONE", "idf_weight": 1.0}]

        candidates = multi_pass_blocking(profiles_a, profiles_b, shared)
        assert len(candidates) == 0


# ── Pair Scoring ───────────────────────────────────────────────────────


class TestScorePair:

    def _make_rels(self, names: list[str], weights: list[float],
                   types: list[str]) -> list[dict]:
        return [
            {"relation": n, "idf_weight": w, "field_type": t}
            for n, w, t in zip(names, weights, types)
        ]

    def test_all_fields_match(self) -> None:
        shared = self._make_rels(
            ["HAS_A", "HAS_B", "HAS_C"],
            [1.0, 2.0, 3.0],
            ["categorical", "numeric", "text"],
        )
        pa = {"HAS_A": "X", "HAS_B": "100", "HAS_C": "hello"}
        pb = {"HAS_A": "X", "HAS_B": "100", "HAS_C": "hello"}

        score, n, fields = score_pair(pa, pb, shared)
        assert score == pytest.approx(1.0)
        assert n == 3
        assert len(fields) == 3

    def test_no_fields_match(self) -> None:
        shared = self._make_rels(
            ["HAS_A", "HAS_B"],
            [1.0, 1.0],
            ["categorical", "categorical"],
        )
        pa = {"HAS_A": "X", "HAS_B": "Y"}
        pb = {"HAS_A": "Z", "HAS_B": "W"}

        score, n, _ = score_pair(pa, pb, shared)
        assert score == 0.0
        assert n == 0

    def test_weighted_score_favors_high_idf(self) -> None:
        shared = self._make_rels(
            ["HAS_LOW", "HAS_HIGH"],
            [0.5, 5.0],
            ["categorical", "categorical"],
        )
        # Only high-IDF field matches.
        pa = {"HAS_LOW": "A", "HAS_HIGH": "RARE_VALUE"}
        pb = {"HAS_LOW": "B", "HAS_HIGH": "RARE_VALUE"}

        score, n, _ = score_pair(pa, pb, shared)
        # weighted = (0 + 5.0*1.0) / (0.5 + 5.0) = 5.0/5.5 ≈ 0.909
        assert score > 0.9
        assert n == 1

    def test_missing_value_skipped(self) -> None:
        shared = self._make_rels(
            ["HAS_A", "HAS_B"],
            [1.0, 1.0],
            ["categorical", "categorical"],
        )
        pa = {"HAS_A": "X"}  # HAS_B missing
        pb = {"HAS_A": "X", "HAS_B": "Y"}

        score, n, _ = score_pair(pa, pb, shared)
        # Only HAS_A is compared (both present), and it matches.
        assert score == 1.0
        assert n == 1

    def test_numeric_tolerance_in_scoring(self) -> None:
        shared = self._make_rels(
            ["HAS_AMOUNT"],
            [2.0],
            ["numeric"],
        )
        pa = {"HAS_AMOUNT": "100000"}
        pb = {"HAS_AMOUNT": "100500"}  # within 5%

        score, n, _ = score_pair(pa, pb, shared)
        assert score == 1.0
        assert n == 1


# ── Domain-Agnostic ────────────────────────────────────────────────────


class TestDomainAgnostic:

    def test_arbitrary_field_names(self) -> None:
        """Works with non-OpenFEMA field names."""
        shared = [
            {"relation": "HAS_RISK_ST", "idf_weight": 1.5, "field_type": "categorical"},
            {"relation": "HAS_COVAMT_PERS", "idf_weight": 3.0, "field_type": "numeric"},
            {"relation": "HAS_COUNTY", "idf_weight": 2.0, "field_type": "text"},
        ]
        pa = {"HAS_RISK_ST": "OH", "HAS_COVAMT_PERS": "10000", "HAS_COUNTY": "HAMILTON"}
        pb = {"HAS_RISK_ST": "OH", "HAS_COVAMT_PERS": "10000", "HAS_COUNTY": "HAMILTON"}

        score, n, _ = score_pair(pa, pb, shared)
        assert score == pytest.approx(1.0)
        assert n == 3
