"""Tests for zone2.lob_detector — detects Line of Business per chunk.

Phase 1 of the LOB-aware extraction work.  Every Zone 1 chunk gets
tagged with one of Assurant's product LOBs so downstream extraction
can dispatch domain-specific behavior consistently.

LOB enum (Assurant-tailored): device, appliance, auto, credit, flood,
home, renters, condo, manufactured, lender_placed, generic.
"""

from __future__ import annotations

import pytest

from zone2.lob_detector import LOBS, detect_lob, detect_lobs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(source: str = "", content: str = "",
           hierarchy: list[str] | None = None) -> dict:
    """Minimal Zone 1-shaped chunk for testing."""
    return {
        "source": source,
        "content": content,
        "section_hierarchy": hierarchy or [],
    }


# ---------------------------------------------------------------------------
# Filename-driven detection
# ---------------------------------------------------------------------------

class TestFilenameDetection:
    """Filename keyword scan is the highest-priority signal."""

    def test_tmobile_phone_csv_is_device(self):
        c = _chunk(source="data/synthetic_data_sample_tmobileclaimsample.csv")
        assert detect_lob(c) == "device"

    def test_geico_renters_csv_is_renters(self):
        c = _chunk(source="data/synthetic_data_sample_geicorenterspoliciesdetails.csv")
        assert detect_lob(c) == "renters"

    def test_geico_renters_claims_csv_is_renters(self):
        c = _chunk(source="data/synthetic_data_sample_geicorentersclaims.csv")
        assert detect_lob(c) == "renters"

    def test_auto_policies_csv_is_auto(self):
        c = _chunk(source="auto_policies_2026.csv")
        assert detect_lob(c) == "auto"

    def test_vehicle_warranty_is_auto(self):
        c = _chunk(source="vehicle_warranty.csv")
        assert detect_lob(c) == "auto"

    def test_nfip_flood_pdf_is_flood(self):
        c = _chunk(source="NFIP_GeneralPropertyForm_SFIP.pdf")
        assert detect_lob(c) == "flood"

    def test_homeowner_csv_is_home(self):
        c = _chunk(source="homeowner_policy.csv")
        assert detect_lob(c) == "home"

    def test_condo_csv_is_condo(self):
        c = _chunk(source="condominium_policies.csv")
        assert detect_lob(c) == "condo"

    def test_manufactured_housing_csv_is_manufactured(self):
        # "manufactured" must take priority over the substring "mobile" (which
        # might otherwise route to device).  Disambiguation case.
        c = _chunk(source="manufactured_housing_policies.csv")
        assert detect_lob(c) == "manufactured"

    def test_lender_placed_csv_is_lender_placed(self):
        c = _chunk(source="lender_placed_homeowners_2025.csv")
        assert detect_lob(c) == "lender_placed"

    def test_force_placed_alias_is_lender_placed(self):
        c = _chunk(source="force_placed_insurance.csv")
        assert detect_lob(c) == "lender_placed"

    def test_credit_insurance_is_credit(self):
        c = _chunk(source="credit_insurance_policies.csv")
        assert detect_lob(c) == "credit"

    def test_appliance_warranty_is_appliance(self):
        c = _chunk(source="appliance_extended_warranty.csv")
        assert detect_lob(c) == "appliance"

    def test_path_components_ignored_only_basename_matched(self):
        # "Emory" appears in the path but is not an LOB keyword;
        # detection must use filename only.
        c = _chunk(source="/data/Emory_Spring2026/raw/credit_card_protection.csv")
        assert detect_lob(c) == "credit"


# ---------------------------------------------------------------------------
# Structured field signature fallback
# ---------------------------------------------------------------------------

class TestFieldSignatureFallback:
    """When filename gives no signal, scan first RECORD block fields."""

    def test_openfema_policies_sample_routes_to_flood(self):
        # The filename "policies_sample.json" has no LOB keyword.  Detection
        # must fall through to field signature: ratedFloodZone, iccPremium.
        c = _chunk(
            source="policies_sample.json",
            content=(
                "DATASET: OpenFEMA NFIP Policies\n\n"
                "RECORD 0:\n"
                "  [Policy] policy effective date: 2025-01-10\n"
                "  [Location] ratedfloodzone: AE | iccpremium: 50\n"
            ),
        )
        assert detect_lob(c) == "flood"

    def test_record_with_vin_routes_to_auto(self):
        c = _chunk(
            source="claims.csv",
            content=(
                "DATASET: Auto Claims\n\n"
                "RECORD 0:\n"
                "  [Vehicle] vin: 1HGBH41JXMN109186 | mileage: 45000\n"
            ),
        )
        assert detect_lob(c) == "auto"

    def test_record_with_imei_routes_to_device(self):
        c = _chunk(
            source="claims.csv",
            content=(
                "DATASET: Device Claims\n\n"
                "RECORD 0:\n"
                "  [Device] imei: 358240051111110 | devicemodel: iPhone 14\n"
            ),
        )
        assert detect_lob(c) == "device"


# ---------------------------------------------------------------------------
# Content fallback (prose / PDF)
# ---------------------------------------------------------------------------

class TestContentFallback:
    """Last-resort: scan section_hierarchy + first 500 chars."""

    def test_flood_keywords_in_prose_route_to_flood(self):
        c = _chunk(
            source="policy_doc.pdf",
            content=(
                "This Standard Flood Insurance Policy is issued under the "
                "National Flood Insurance Program (NFIP).  Coverage applies to "
                "buildings located in designated flood zones."
            ),
            hierarchy=["I. AGREEMENT"],
        )
        assert detect_lob(c) == "flood"

    def test_single_keyword_below_threshold_returns_generic(self):
        # A single passing reference shouldn't reroute the LOB.
        c = _chunk(
            source="policy_doc.pdf",
            content="This document mentions flood once but is otherwise generic.",
        )
        assert detect_lob(c) == "generic"


# ---------------------------------------------------------------------------
# Default and edge cases
# ---------------------------------------------------------------------------

class TestDefaultAndEdges:
    def test_empty_chunk_returns_generic(self):
        assert detect_lob(_chunk()) == "generic"

    def test_unrecognized_source_returns_generic(self):
        c = _chunk(source="unknown_data.csv", content="random content")
        assert detect_lob(c) == "generic"

    def test_filename_priority_over_field_signature(self):
        # Filename says renters; field signature would say auto.  Filename wins.
        c = _chunk(
            source="geico_renters.csv",
            content="RECORD 0:\n  [Vehicle] vin: 1HGBH41JXMN109186\n",
        )
        assert detect_lob(c) == "renters"

    def test_returns_value_in_lobs_set(self):
        c = _chunk(source="flood_sample.json")
        result = detect_lob(c)
        assert result in LOBS

    def test_lobs_has_expected_assurant_set(self):
        # All eleven Assurant LOBs (10 specific + generic).
        expected = {
            "device", "appliance", "auto", "credit",
            "flood", "home", "renters", "condo",
            "manufactured", "lender_placed",
            "generic",
        }
        assert set(LOBS) == expected


# ---------------------------------------------------------------------------
# detect_lobs LangGraph node — stamps chunk['lob'] on every chunk
# ---------------------------------------------------------------------------

class TestDetectLobsNode:
    def test_stamps_lob_field_on_every_chunk(self):
        chunks = [
            _chunk(source="tmobile_claims.csv"),
            _chunk(source="auto_warranty.csv"),
            _chunk(source="unknown.txt"),
        ]
        state = {"chunks": chunks}

        result = detect_lobs(state)

        assert "chunks" in result
        out = result["chunks"]
        assert len(out) == 3
        assert out[0]["lob"] == "device"
        assert out[1]["lob"] == "auto"
        assert out[2]["lob"] == "generic"

    def test_stamps_lob_on_schema_chunks_too(self):
        # Schema chunks also need the tag (they describe a specific dataset).
        schema_chunk = {
            "source": "policies_sample.json",
            "content": "DATASET SCHEMA: OpenFEMA NFIP Policies\n  ratedFloodZone\n",
            "section_hierarchy": ["FimaNfipPolicies", "schema"],
        }
        state = {"chunks": [schema_chunk]}

        result = detect_lobs(state)

        assert result["chunks"][0]["lob"] == "flood"

    def test_handles_empty_chunks_list(self):
        result = detect_lobs({"chunks": []})
        assert result["chunks"] == []

    def test_does_not_drop_existing_chunk_fields(self):
        chunks = [{
            "source": "flood.pdf",
            "content": "...",
            "section_hierarchy": ["I. AGREEMENT"],
            "chunk_id": 42,
            "token_count": 150,
        }]
        state = {"chunks": chunks}

        result = detect_lobs(state)

        out = result["chunks"][0]
        assert out["chunk_id"] == 42
        assert out["token_count"] == 150
        assert out["lob"] == "flood"
