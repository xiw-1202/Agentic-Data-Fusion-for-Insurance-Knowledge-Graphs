"""Tests for SEAF-KG Stage 1 — Deterministic Structured Mapper."""

import pytest

from zone2.structured_mapper import (
    detect_record_type,
    generate_composite_key,
    infer_value_type,
    is_schema_chunk,
    is_structured_chunk,
    parse_records_from_chunk,
    record_to_triples,
    extract_structured,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CSV_CHUNK = {
    "content": (
        "DATASET: OpenFEMA NFIP Policies\n\n"
        "RECORD:\n"
        "  [Policy] policy effective date: 2025-01-10T00:00:00.000Z | "
        "policy termination date: 2026-01-10T00:00:00.000Z | policy cost: 681\n"
        "  [Coverage] total building insurance coverage: 81000 | "
        "building deductible code: 1\n"
        "  [Location] rated flood zone: A | property state: GA\n"
        "\n"
        "RECORD:\n"
        "  [Policy] policy effective date: 2025-06-17T00:00:00.000Z | "
        "policy cost: 668\n"
        "  [Coverage] total building insurance coverage: 250000\n"
        "  [Location] rated flood zone: X | property state: FL\n"
    ),
    "source": "data/flood/raw/openfema/policies_sample.json",
    "section_hierarchy": ["FimaNfipPolicies", "records 0\u20137"],
    "pages": [],
    "chunk_type": "text",
}

SAMPLE_CLAIMS_CHUNK = {
    "content": (
        "DATASET: OpenFEMA NFIP Claims\n\n"
        "RECORD:\n"
        "  [Loss] date of loss: 2017-08-25T00:00:00.000Z | "
        "cause of damage: Flood | water depth: 3\n"
        "  [Building] total building insurance coverage: 250000 | "
        "amount paid on building claim: 45000\n"
        "  [Location] rated flood zone: A | state: TX\n"
    ),
    "source": "data/flood/raw/openfema/claims_sample.json",
    "section_hierarchy": ["FimaNfipClaims", "records 0\u20134"],
    "pages": [],
    "chunk_type": "text",
}

SAMPLE_PDF_CHUNK = {
    "content": (
        "A. Coverage Under This Policy\n"
        "1. Except as provided in I.A.2, this policy provides\n"
        "coverage for residential buildings."
    ),
    "source": "fema_F-123-general-property-SFIP_2021.pdf",
    "section_hierarchy": ["I. AGREEMENT", "A. Coverage Under This Policy"],
    "pages": [0, 1],
    "chunk_type": "text",
}

SAMPLE_SCHEMA_CHUNK = {
    "content": (
        "DATASET SCHEMA: OpenFEMA NFIP Policies\n"
        "Key fields and their meanings:\n"
        "  total building insurance coverage: maximum building claim payout"
    ),
    "source": "data/flood/raw/openfema/policies_sample.json",
    "section_hierarchy": ["FimaNfipPolicies", "schema"],
    "pages": [],
    "chunk_type": "text",
}


# ---------------------------------------------------------------------------
# Source detection tests
# ---------------------------------------------------------------------------

class TestSourceDetection:
    def test_csv_chunk_detected_as_structured(self) -> None:
        assert is_structured_chunk(SAMPLE_CSV_CHUNK) is True

    def test_pdf_chunk_not_structured(self) -> None:
        assert is_structured_chunk(SAMPLE_PDF_CHUNK) is False

    def test_schema_chunk_not_structured(self) -> None:
        assert is_structured_chunk(SAMPLE_SCHEMA_CHUNK) is False

    def test_schema_chunk_detected(self) -> None:
        assert is_schema_chunk(SAMPLE_SCHEMA_CHUNK) is True

    def test_csv_data_not_schema(self) -> None:
        assert is_schema_chunk(SAMPLE_CSV_CHUNK) is False

    def test_pdf_not_schema(self) -> None:
        assert is_schema_chunk(SAMPLE_PDF_CHUNK) is False


class TestRecordTypeDetection:
    def test_policies_from_source(self) -> None:
        assert detect_record_type(SAMPLE_CSV_CHUNK) == "policies"

    def test_claims_from_source(self) -> None:
        assert detect_record_type(SAMPLE_CLAIMS_CHUNK) == "claims"

    def test_pdf_defaults_to_unknown(self) -> None:
        result = detect_record_type(SAMPLE_PDF_CHUNK)
        assert result == "unknown"


# ---------------------------------------------------------------------------
# Record parsing tests
# ---------------------------------------------------------------------------

class TestRecordParsing:
    def test_parses_two_records(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CSV_CHUNK)
        assert len(records) == 2

    def test_first_record_has_expected_fields(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CSV_CHUNK)
        rec = records[0]
        assert rec["policy effective date"] == "2025-01-10T00:00:00.000Z"
        assert rec["total building insurance coverage"] == "81000"
        assert rec["rated flood zone"] == "A"
        assert rec["property state"] == "GA"
        assert rec["policy cost"] == "681"

    def test_second_record_different_values(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CSV_CHUNK)
        rec = records[1]
        assert rec["rated flood zone"] == "X"
        assert rec["property state"] == "FL"

    def test_claims_record_parsed(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CLAIMS_CHUNK)
        assert len(records) == 1
        rec = records[0]
        assert rec["cause of damage"] == "Flood"
        assert rec["amount paid on building claim"] == "45000"

    def test_empty_chunk_returns_empty(self) -> None:
        chunk = {"content": "DATASET: Something\n\nNo records here."}
        records = parse_records_from_chunk(chunk)
        assert len(records) == 0


# ---------------------------------------------------------------------------
# Composite key tests
# ---------------------------------------------------------------------------

class TestCompositeKey:
    def test_same_record_same_key(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CSV_CHUNK)
        key1 = generate_composite_key(records[0], "policies")
        key2 = generate_composite_key(records[0], "policies")
        assert key1 == key2

    def test_different_records_different_keys(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CSV_CHUNK)
        key1 = generate_composite_key(records[0], "policies")
        key2 = generate_composite_key(records[1], "policies")
        assert key1 != key2

    def test_key_has_prefix(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CSV_CHUNK)
        key = generate_composite_key(records[0], "policies")
        assert key.startswith("POL-")
        assert len(key) == 16  # "POL-" + 12 hex chars

    def test_claims_key_prefix(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CLAIMS_CHUNK)
        key = generate_composite_key(records[0], "claims")
        assert key.startswith("CLM-")

    def test_unknown_type_fallback(self) -> None:
        record = {"field_a": "val1", "field_b": "val2", "field_c": "val3", "field_d": "val4"}
        key = generate_composite_key(record, "unknown")
        assert key.startswith("REC-")
        assert len(key) == 16


# ---------------------------------------------------------------------------
# Value type inference tests
# ---------------------------------------------------------------------------

class TestValueTypeInference:
    def test_iso_date(self) -> None:
        assert infer_value_type("some field", "2025-01-10T00:00:00.000Z") == "Date"

    def test_date_field_hint(self) -> None:
        assert infer_value_type("original construction date", "1990") == "Date"

    def test_numeric(self) -> None:
        assert infer_value_type("coverage amount", "250000") == "Numeric"

    def test_currency(self) -> None:
        assert infer_value_type("cost", "$1,500.00") == "Currency"

    def test_percentage(self) -> None:
        assert infer_value_type("rate", "15.5 %") == "Percentage"

    def test_categorical_code(self) -> None:
        assert infer_value_type("zone code", "A") == "Categorical"

    def test_text(self) -> None:
        assert infer_value_type("community name", "NASHVILLE, CITY OF") == "Text"


# ---------------------------------------------------------------------------
# Triple generation tests
# ---------------------------------------------------------------------------

class TestTripleGeneration:
    def test_generates_type_triple(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CSV_CHUNK)
        triples = record_to_triples(records[0], "policies", "chunk_1", "test.json")
        type_triples = [t for t in triples if t["relation"] == "IS_A"]
        assert len(type_triples) == 1
        assert type_triples[0]["object"] == "PolicyRecord"
        assert type_triples[0]["confidence"] == 1.0

    def test_generates_property_triples(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CSV_CHUNK)
        triples = record_to_triples(records[0], "policies", "chunk_1", "test.json")
        # 1 IS_A + N property triples = total
        prop_triples = [t for t in triples if t["relation"] != "IS_A"]
        assert len(prop_triples) == len(records[0])

    def test_property_relation_format(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CSV_CHUNK)
        triples = record_to_triples(records[0], "policies", "chunk_1", "test.json")
        relations = {t["relation"] for t in triples}
        assert "HAS_POLICY_EFFECTIVE_DATE" in relations
        assert "HAS_RATED_FLOOD_ZONE" in relations
        assert "HAS_TOTAL_BUILDING_INSURANCE_COVERAGE" in relations

    def test_all_triples_have_structured_source_type(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CSV_CHUNK)
        triples = record_to_triples(records[0], "policies", "chunk_1", "test.json")
        for t in triples:
            assert t["source_type"] == "structured"

    def test_subject_is_composite_key(self) -> None:
        records = parse_records_from_chunk(SAMPLE_CSV_CHUNK)
        triples = record_to_triples(records[0], "policies", "chunk_1", "test.json")
        subjects = {t["subject"] for t in triples}
        assert len(subjects) == 1
        subject = subjects.pop()
        assert subject.startswith("POL-")


# ---------------------------------------------------------------------------
# LangGraph node integration test
# ---------------------------------------------------------------------------

class TestExtractStructuredNode:
    def test_splits_chunks_correctly(self) -> None:
        state = {
            "chunks": [
                SAMPLE_CSV_CHUNK,
                SAMPLE_PDF_CHUNK,
                SAMPLE_SCHEMA_CHUNK,
                SAMPLE_CLAIMS_CHUNK,
            ],
        }
        result = extract_structured(state)

        # PDF chunks + schema chunks should remain.
        remaining = result["chunks"]
        remaining_sources = [c["source"] for c in remaining]
        assert any("pdf" in s.lower() or "SFIP" in s for s in remaining_sources)
        assert any("schema" in " ".join(c.get("section_hierarchy", [])).lower()
                    for c in remaining)

        # Structured triples should be generated.
        structured = result["structured_triples"]
        assert len(structured) > 0

        # Should have triples from both policy and claims records.
        subjects = {t["subject"] for t in structured}
        has_pol = any(s.startswith("POL-") for s in subjects)
        has_clm = any(s.startswith("CLM-") for s in subjects)
        assert has_pol
        assert has_clm

    def test_empty_chunks_no_crash(self) -> None:
        result = extract_structured({"chunks": []})
        assert result["structured_triples"] == []
        assert result["chunks"] == []
