"""Regression tests for the Zone 2 punch list (2026-05-01).

Covers extraction-side bugs surfaced by the chatbot CSV ground-truth eval:

* Fix 1 — header-expander letter-spacing on real English words
* Fix 2 — `_normalize_field_name` length guard against letter-spaced labels
* Fix 3 — `_safe_to_merge_rels` rejects over-aggressive short-relation merges
* Fix 4 — empty-value list keeps zeros for numeric/currency columns

Each test asserts behavior on a small, self-contained input — no Neo4j,
no Ollama.  The pytest target is the conda anaconda3 Python 3.13 env.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fix 2 — letter-spacing guard in _normalize_field_name
# ---------------------------------------------------------------------------

class TestNormalizeFieldNameLetterSpacingGuard:
    def test_letter_spaced_label_passes_through_unchanged(self):
        from zone2.structured_mapper import _normalize_field_name
        # Every "a" was previously stripped as an article, mangling MANUFACTURER
        # to M_N_U_F_C_T_U_R_E_R.
        out = _normalize_field_name("m a n u f a c t u r e r")
        assert out == "m a n u f a c t u r e r"

    def test_normal_label_still_strips_articles(self):
        from zone2.structured_mapper import _normalize_field_name
        assert (
            _normalize_field_name("number of floors in the insured building")
            == "number of floors in insured building"
        )

    def test_mostly_short_tokens_treated_as_letter_spaced(self):
        from zone2.structured_mapper import _normalize_field_name
        # Three of four tokens are length 1 → guard fires
        assert _normalize_field_name("a b c word") == "a b c word"

    def test_field_to_relation_preserves_letter_spaced_input(self):
        from zone2.structured_mapper import _field_to_relation
        # Result is still ugly but recognizable — beats the previous mangled
        # form HAS_M_N_U_F_C_T_U_R_E_R that lost characters.
        rel = _field_to_relation("m a n u f a c t u r e r")
        assert "M_A_N_U_F_A_C_T_U_R_E_R" in rel


# ---------------------------------------------------------------------------
# Fix 3 — _safe_to_merge_rels guards short-relation over-merging
# ---------------------------------------------------------------------------

class TestSafeToMergeRels:
    """Three known-bad pairs MUST NOT merge."""

    @pytest.mark.parametrize(
        "rel_i,rel_j,sim",
        [
            ("HAS_CLAIM_STATUS", "HAS_CLAIM_ISSUE", 0.92),
            ("HAS_CLAIM_AUTHORIZED_DATE", "HAS_CLAIM_APPROVED_DATE", 0.92),
            ("HAS_TIME_TO_REPLACE_CSAT", "HAS_TIME_TO_HANDLE_CSAT", 0.95),
            ("HAS_MIN_AGE", "HAS_MAX_AGE", 0.96),
        ],
    )
    def test_known_bad_pairs_do_not_merge(self, rel_i, rel_j, sim):
        from zone2.pipeline import _rel_content_tokens, _safe_to_merge_rels
        t_i = _rel_content_tokens(rel_i)
        t_j = _rel_content_tokens(rel_j)
        assert _safe_to_merge_rels(sim, t_i, t_j) is False, (
            f"{rel_i} should not merge into {rel_j}"
        )

    def test_short_relations_require_higher_threshold(self):
        from zone2.pipeline import _rel_content_tokens, _safe_to_merge_rels
        t_i = _rel_content_tokens("HAS_FOO_BAR")
        t_j = _rel_content_tokens("HAS_FOO_BAR_VALUE")
        # Below 0.92 = no merge for short relations
        assert _safe_to_merge_rels(0.88, t_i, t_j) is False
        # Above 0.92 with high overlap = merge OK
        assert _safe_to_merge_rels(0.94, t_i, t_j) is True

    def test_low_jaccard_blocks_merge_even_at_high_cosine(self):
        from zone2.pipeline import _rel_content_tokens, _safe_to_merge_rels
        # 1 shared token out of 5 union = 0.2 jaccard < 0.6
        t_i = _rel_content_tokens("HAS_CLAIM_STATUS")
        t_j = _rel_content_tokens("HAS_POLICY_STATUS")
        assert _safe_to_merge_rels(0.95, t_i, t_j) is False


# ---------------------------------------------------------------------------
# Fix 4 — zero handling for numeric/currency columns
# ---------------------------------------------------------------------------

class TestEmptyValueZeroHandling:
    def test_zero_kept_for_currency_or_count_columns(self):
        from zone1.ingestion import _is_meaningful_value
        # A zero deductible or zero claim count is real data, not missing.
        assert _is_meaningful_value("0", value_type="currency") is True
        assert _is_meaningful_value("0.00", value_type="currency") is True
        assert _is_meaningful_value("0", value_type="integer") is True
        assert _is_meaningful_value("0", value_type="float") is True

    def test_zero_dropped_for_boolean_columns(self):
        from zone1.ingestion import _is_meaningful_value
        # A "0" in a boolean-encoded flag column is "false" / absent — drop it.
        assert _is_meaningful_value("0", value_type="boolean") is False
        assert _is_meaningful_value("false", value_type="boolean") is False

    def test_truly_empty_values_always_dropped(self):
        from zone1.ingestion import _is_meaningful_value
        for v in ("", "nan", "none", "null", "1900-01-01"):
            assert _is_meaningful_value(v, value_type="currency") is False
            assert _is_meaningful_value(v, value_type="boolean") is False

    def test_unknown_type_falls_back_to_legacy_drop_set(self):
        from zone1.ingestion import _is_meaningful_value
        # Default behavior when type is None: keep current narrative-column
        # behavior (drop zero/false) for safety.
        assert _is_meaningful_value("0", value_type=None) is False
        assert _is_meaningful_value("real value", value_type=None) is True


# ---------------------------------------------------------------------------
# Fix 1 — header-expander filter rejects real English words and letter-spacing
# ---------------------------------------------------------------------------

class TestHeaderExpansionFilter:
    def test_real_words_skipped_by_abbreviation_filter(self):
        from zone1.ingestion import _looks_like_abbreviation
        # These came back letter-spaced from the LLM in the punch-list run.
        for word in (
            "MANUFACTURER", "ACCOUNT", "PLATFORM", "JOURNEY", "PERIOD",
            "COMMENTS", "CLIENT", "INSURED", "CLAIM", "COUNTY", "CREDIT",
            "EARNED", "KIND", "RANK", "TAXES", "TERM", "FEES",
        ):
            assert _looks_like_abbreviation(word) is False, (
                f"{word} is a real English word; should not be expanded"
            )

    def test_genuine_abbreviations_pass_filter(self):
        from zone1.ingestion import _looks_like_abbreviation
        for abbr in ("GWP", "MCO", "NWP", "CSAT", "NB", "RWP", "CAT"):
            assert _looks_like_abbreviation(abbr) is True, (
                f"{abbr} is an abbreviation; should be sent for expansion"
            )

    def test_letter_spaced_llm_response_is_rejected(self):
        from zone1.ingestion import _is_letter_spaced_response
        assert _is_letter_spaced_response("MANUFACTURER", "m a n u f a c t u r e r") is True
        assert _is_letter_spaced_response("MCO", "master company code") is False


class TestHumanizeFieldNameWithDigits:
    """Regression: all-caps column names with digits (LADD1, ADD2, LOC3)
    were being letter-spaced by the camelCase fallback because the
    isalpha() guard excluded digit-bearing strings.  Result was relations
    like HAS_L_A_D_D1 in Neo4j."""

    def test_all_caps_with_trailing_digit_is_lowercased(self):
        from zone1.ingestion import _humanize_field_name
        assert _humanize_field_name("LADD1") == "ladd1"
        assert _humanize_field_name("ADD1") == "add1"
        assert _humanize_field_name("LADD2") == "ladd2"
        assert _humanize_field_name("LOC3") == "loc3"

    def test_camel_case_with_digits_still_handled(self):
        from zone1.ingestion import _humanize_field_name
        # genuine camelCase with embedded digit — keep current behavior
        assert _humanize_field_name("amountPaid1") == "amount paid1"
        assert _humanize_field_name("Item3Cost") == "item3 cost"

    def test_underscore_separated_with_digits_preserved(self):
        from zone1.ingestion import _humanize_field_name
        assert _humanize_field_name("CLAIM_NUMBER_1") == "claim number 1"
        assert _humanize_field_name("ADDR_LINE_2") == "addr line 2"
