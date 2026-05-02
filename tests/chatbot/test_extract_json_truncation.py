"""Regression tests for _extract_json — must tolerate model responses
that get cut off mid-JSON when the token cap is hit."""
from __future__ import annotations

import pytest

from chatbot.qa_chain import _extract_json


class TestExtractJson:
    def test_clean_json_parses(self):
        out = _extract_json('{"summary":"ok","confidence":0.9}')
        assert out["summary"] == "ok"
        assert out["confidence"] == 0.9

    def test_json_with_code_fence(self):
        out = _extract_json('```json\n{"k":"v"}\n```')
        assert out == {"k": "v"}

    def test_json_with_leading_prose(self):
        out = _extract_json("Here you go:\n{\"k\":\"v\"}\nthanks")
        assert out == {"k": "v"}

    def test_truncated_mid_string_recovers_prefix(self):
        """Live failure mode: 'Is mold damage covered?' returned
        a JSON that was cut mid-summary because max_tokens=700 wasn't
        enough.  We should salvage everything before the truncation
        rather than crashing the whole chatbot turn."""
        truncated = (
            '{ "summary": "The claims data confirms that MOLD appears as '
            'a recognized cause-of-loss code in the GEICO Renters claims '
            'dataset, with at least two closed claims (CLM-e11b0de1b0ca '
            'and CLM-f08143cb280'
        )
        out = _extract_json(truncated)
        assert "summary" in out
        assert "MOLD" in out["summary"]

    def test_truncated_after_complete_field_recovers(self):
        truncated = (
            '{"summary":"complete field","reasoning":"another complete",'
            '"confidence":0.8,"caveats":"trailing partial that is cut off mid-'
        )
        out = _extract_json(truncated)
        assert out["summary"] == "complete field"
        assert out["reasoning"] == "another complete"
        assert out["confidence"] == 0.8

    def test_unrecoverable_garbage_still_raises(self):
        with pytest.raises(ValueError, match="no JSON"):
            _extract_json("not even close to json")
