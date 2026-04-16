"""Tests for zone3.fbi.fingerprint — algorithmic functions only (no LLM)."""

from __future__ import annotations

import os
import tempfile

import pytest

from zone3.fbi.fingerprint import (
    FileFingerprint,
    apply_expansions,
    build_filename_parse_prompt,
    build_header_expansion_prompt,
    count_csv_rows,
    extract_csv_headers,
    strip_audit_columns,
)


# ---------------------------------------------------------------------------
# TestExtractCsvHeaders
# ---------------------------------------------------------------------------


class TestExtractCsvHeaders:
    """Tests for extract_csv_headers."""

    def test_basic_csv(self, tmp_path: object) -> None:
        p = os.path.join(str(tmp_path), "test.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("name,age,city\n")
            f.write("Alice,30,NYC\n")

        headers = extract_csv_headers(p)
        assert headers == ["NAME", "AGE", "CITY"]

    def test_empty_csv(self, tmp_path: object) -> None:
        p = os.path.join(str(tmp_path), "empty.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("")

        headers = extract_csv_headers(p)
        assert headers == []

    def test_whitespace_in_headers(self, tmp_path: object) -> None:
        p = os.path.join(str(tmp_path), "ws.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("  first name , last name , ZIP Code \n")
            f.write("Alice,Smith,10001\n")

        headers = extract_csv_headers(p)
        assert headers == ["FIRST NAME", "LAST NAME", "ZIP CODE"]

    def test_bom_encoded_csv(self, tmp_path: object) -> None:
        p = os.path.join(str(tmp_path), "bom.csv")
        with open(p, "w", encoding="utf-8-sig") as f:
            f.write("id,value\n")
            f.write("1,100\n")

        headers = extract_csv_headers(p)
        assert headers == ["ID", "VALUE"]

    def test_header_only_csv(self, tmp_path: object) -> None:
        p = os.path.join(str(tmp_path), "header_only.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("col_a,col_b\n")

        headers = extract_csv_headers(p)
        assert headers == ["COL_A", "COL_B"]


# ---------------------------------------------------------------------------
# TestCountCsvRows
# ---------------------------------------------------------------------------


class TestCountCsvRows:
    def test_count_rows(self, tmp_path: object) -> None:
        p = os.path.join(str(tmp_path), "data.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("a,b\n1,2\n3,4\n5,6\n")

        assert count_csv_rows(p) == 3

    def test_empty_file(self, tmp_path: object) -> None:
        p = os.path.join(str(tmp_path), "empty.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("")

        assert count_csv_rows(p) == 0


# ---------------------------------------------------------------------------
# TestStripAuditColumns
# ---------------------------------------------------------------------------


class TestStripAuditColumns:
    """Tests for strip_audit_columns."""

    def test_removes_audit_columns_present_in_all_files(self) -> None:
        headers_by_file = {
            "file1.csv": ["NAME", "AGE", "BI_CREATED_DT", "BI_MODIFIED_BY"],
            "file2.csv": ["CITY", "STATE", "BI_CREATED_DT", "BI_MODIFIED_BY"],
        }

        result = strip_audit_columns(headers_by_file)
        assert result["file1.csv"] == ["NAME", "AGE"]
        assert result["file2.csv"] == ["CITY", "STATE"]

    def test_keeps_audit_column_not_in_all_files(self) -> None:
        headers_by_file = {
            "file1.csv": ["NAME", "BI_CREATED_DT"],
            "file2.csv": ["CITY", "STATE"],
        }

        result = strip_audit_columns(headers_by_file)
        # BI_CREATED_DT is only in file1, not common to all — keep it
        assert result["file1.csv"] == ["NAME", "BI_CREATED_DT"]
        assert result["file2.csv"] == ["CITY", "STATE"]

    def test_keeps_unique_columns(self) -> None:
        headers_by_file = {
            "file1.csv": ["POLICY_ID", "PREMIUM", "ETL_LOAD_DT"],
            "file2.csv": ["CLAIM_ID", "AMOUNT", "ETL_LOAD_DT"],
        }

        result = strip_audit_columns(headers_by_file)
        # ETL_LOAD_DT matches audit pattern and is in all files
        assert "ETL_LOAD_DT" not in result["file1.csv"]
        assert "ETL_LOAD_DT" not in result["file2.csv"]
        assert "POLICY_ID" in result["file1.csv"]
        assert "CLAIM_ID" in result["file2.csv"]

    def test_empty_input(self) -> None:
        assert strip_audit_columns({}) == {}

    def test_non_audit_common_columns_kept(self) -> None:
        headers_by_file = {
            "f1.csv": ["ID", "STATE", "BI_CREATED_DT"],
            "f2.csv": ["ID", "STATE", "BI_CREATED_DT"],
        }

        result = strip_audit_columns(headers_by_file)
        # ID and STATE are common but not audit-pattern — keep them
        assert "ID" in result["f1.csv"]
        assert "STATE" in result["f1.csv"]
        assert "BI_CREATED_DT" not in result["f1.csv"]


# ---------------------------------------------------------------------------
# TestBuildHeaderExpansionPrompt
# ---------------------------------------------------------------------------


class TestBuildHeaderExpansionPrompt:
    """Tests for build_header_expansion_prompt."""

    def test_prompt_contains_all_headers(self) -> None:
        headers = ["POLICYCOUNT", "RATEDFLOODZONE", "OCCUPANCYTYPE"]
        prompt = build_header_expansion_prompt(headers)

        for h in headers:
            assert h in prompt

    def test_prompt_format(self) -> None:
        headers = ["COL_A", "COL_B"]
        prompt = build_header_expansion_prompt(headers)

        assert "ABBREVIATION -> Full Name" in prompt

    def test_batch_size_limits_headers(self) -> None:
        headers = [f"COL_{i}" for i in range(100)]
        prompt = build_header_expansion_prompt(headers, batch_size=10)

        assert "COL_9" in prompt
        assert "COL_10" not in prompt


# ---------------------------------------------------------------------------
# TestBuildFilenameParsePrompt
# ---------------------------------------------------------------------------


class TestBuildFilenameParsePrompt:
    """Tests for build_filename_parse_prompt."""

    def test_prompt_contains_filenames(self) -> None:
        filenames = ["claims_sample.csv", "policies_sample.csv"]
        prompt = build_filename_parse_prompt(filenames)

        for f in filenames:
            assert f in prompt

    def test_prompt_requests_json(self) -> None:
        prompt = build_filename_parse_prompt(["test.csv"])
        assert "JSON" in prompt


# ---------------------------------------------------------------------------
# TestApplyExpansions
# ---------------------------------------------------------------------------


class TestApplyExpansions:
    """Tests for apply_expansions."""

    def test_applies_mapping(self) -> None:
        fp = FileFingerprint(
            file_path="test.csv",
            file_type="csv",
            headers_raw=["COL_A", "COL_B", "COL_C"],
        )
        expansions = {
            "COL_A": "Column Alpha",
            "COL_B": "Column Beta",
            "COL_C": "Column Charlie",
        }

        apply_expansions(fp, expansions)

        assert fp.headers_expanded["COL_A"] == "Column Alpha"
        assert fp.headers_expanded["COL_B"] == "Column Beta"
        assert fp.headers_expanded["COL_C"] == "Column Charlie"

    def test_missing_keys_use_raw_value(self) -> None:
        fp = FileFingerprint(
            file_path="test.csv",
            file_type="csv",
            headers_raw=["KNOWN", "UNKNOWN"],
        )
        expansions = {"KNOWN": "Known Column"}

        apply_expansions(fp, expansions)

        assert fp.headers_expanded["KNOWN"] == "Known Column"
        assert fp.headers_expanded["UNKNOWN"] == "UNKNOWN"

    def test_empty_headers(self) -> None:
        fp = FileFingerprint(
            file_path="test.csv",
            file_type="csv",
            headers_raw=[],
        )

        apply_expansions(fp, {"X": "Y"})
        assert fp.headers_expanded == {}


# ---------------------------------------------------------------------------
# TestFileFingerprint
# ---------------------------------------------------------------------------


class TestFileFingerprint:
    """Tests for FileFingerprint dataclass."""

    def test_basename(self) -> None:
        fp = FileFingerprint(
            file_path="/some/path/to/claims_sample.csv",
            file_type="csv",
        )
        assert fp.basename == "claims_sample.csv"

    def test_basename_simple(self) -> None:
        fp = FileFingerprint(file_path="file.txt", file_type="txt")
        assert fp.basename == "file.txt"
