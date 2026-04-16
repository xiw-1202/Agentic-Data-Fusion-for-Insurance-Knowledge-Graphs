"""Tests for zone3.fbi.llm_utils — parse_arrow_mapping only (no live LLM)."""

import sys
import os

import pytest

# Ensure project root is on the path so zone3 is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zone3.fbi.llm_utils import parse_arrow_mapping


class TestParseArrowMapping:
    """Tests for parse_arrow_mapping."""

    def test_standard_arrow(self) -> None:
        text = "SB → Sub-Business\nMCO → Master Company Organization"
        result = parse_arrow_mapping(text)
        assert result == {
            "SB": "Sub-Business",
            "MCO": "Master Company Organization",
        }

    def test_ascii_arrow(self) -> None:
        text = "SB -> Sub-Business"
        result = parse_arrow_mapping(text)
        assert result == {"SB": "Sub-Business"}

    def test_empty_lines_skipped(self) -> None:
        text = "A → Alpha\n\n\nB → Beta\n"
        result = parse_arrow_mapping(text)
        assert result == {"A": "Alpha", "B": "Beta"}

    def test_no_arrows(self) -> None:
        text = "Just some plain text\nwithout any arrows"
        result = parse_arrow_mapping(text)
        assert result == {}

    def test_multi_word_key(self) -> None:
        text = "COVAMT_PERS → Coverage Amount Personal Property"
        result = parse_arrow_mapping(text)
        assert result == {"COVAMT_PERS": "Coverage Amount Personal Property"}
