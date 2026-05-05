"""Tests for chunk-context preamble in LLM prompts (Phase 5).

Both LLM extraction paths — :func:`_extract_one_pass` (main few-shot
extraction) and :func:`_decompose_then_extract` (CoDe-KG-style
decomposition) — now prepend a context line to the chunk text:

    Source: <basename>  |  LOB: <lob>  |  Section: <section_hierarchy>

This gives the LLM enough context to disambiguate "not covered" between
an exclusion section and a definition section, and to apply LOB-aware
extraction patterns.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from zone2.pipeline import (
    _build_chunk_preamble,
    _extract_one_pass,
    _decompose_then_extract,
)


# ---------------------------------------------------------------------------
# _build_chunk_preamble — unit
# ---------------------------------------------------------------------------

class TestBuildChunkPreamble:
    def test_full_preamble_contains_source_lob_section(self):
        chunk = {
            "source": "data/flood/raw/openfema/policies_sample.json",
            "lob": "flood",
            "section_hierarchy": ["FimaNfipPolicies", "records 0-49"],
        }
        preamble = _build_chunk_preamble(chunk)
        # Filename only (not the full path).
        assert "policies_sample.json" in preamble
        assert "data/flood" not in preamble
        assert "flood" in preamble.lower()
        assert "FimaNfipPolicies" in preamble or "records 0-49" in preamble

    def test_missing_lob_falls_back_to_generic(self):
        chunk = {"source": "x.csv", "section_hierarchy": []}
        preamble = _build_chunk_preamble(chunk)
        assert "generic" in preamble.lower()

    def test_no_section_hierarchy_omits_section_field(self):
        chunk = {"source": "policy.pdf", "lob": "auto",
                 "section_hierarchy": []}
        preamble = _build_chunk_preamble(chunk)
        # Should still mention source + lob but no Section: line.
        assert "policy.pdf" in preamble
        assert "auto" in preamble.lower()
        assert "Section:" not in preamble

    def test_preamble_is_short_under_300_chars(self):
        # Long hierarchies must still produce a compact preamble.
        chunk = {
            "source": "huge_doc.pdf",
            "lob": "flood",
            "section_hierarchy": ["I. AGREEMENT" * 5,
                                  "A. Coverage" * 5,
                                  "part 1"],
        }
        preamble = _build_chunk_preamble(chunk)
        assert len(preamble) < 300


# ---------------------------------------------------------------------------
# _extract_one_pass uses preamble
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Captures invoked messages so we can assert on their content."""

    def __init__(self, response_content: str = "[]"):
        self.invocations: list = []
        self.response_content = response_content

    def invoke(self, messages):
        self.invocations.append(messages)
        resp = MagicMock()
        resp.content = self.response_content
        return resp


class TestExtractOnePassPreamble:
    def test_human_message_includes_chunk_preamble(self):
        llm = _FakeLLM(response_content="[]")
        chunk = {
            "chunk_id": "c1",
            "content": "Earth movement is excluded from coverage.",
            "source": "fema_F-123-general-property-SFIP_2021.pdf",
            "section_hierarchy": ["IV. PROPERTY NOT INSURED"],
            "lob": "flood",
        }
        _extract_one_pass(
            llm=llm, base_messages=[], chunks=[chunk],
            pass_label="test", errors=[],
        )
        assert len(llm.invocations) == 1
        # The last message in the call is the human message (we may have
        # 0 base messages → only the human message).
        human = llm.invocations[0][-1]
        assert "fema_F-123-general-property-SFIP_2021.pdf" in human.content
        assert "flood" in human.content.lower()
        assert "IV. PROPERTY NOT INSURED" in human.content


# ---------------------------------------------------------------------------
# _decompose_then_extract uses preamble
# ---------------------------------------------------------------------------

class _FakeLLMQueueResponses:
    """Returns canned responses in order — supports both stages of decompose."""

    def __init__(self, responses: list[str]):
        self.invocations: list = []
        self._responses = list(responses)

    def invoke(self, messages):
        self.invocations.append(messages)
        out = MagicMock()
        # Pop next response, or empty list as a safe default.
        out.content = self._responses.pop(0) if self._responses else "[]"
        return out


class TestDecomposeThenExtractPreamble:
    def test_decomposition_prompt_contains_chunk_preamble(self, monkeypatch):
        # Prepare a fake free-form LLM (Stage 1) that records the prompt
        # used; Stage 2 doesn't need to do anything for our assertion.
        stage1 = _FakeLLMQueueResponses(["1. Earth movement is excluded.\n"])
        stage2 = _FakeLLMQueueResponses(["{}"])

        # Patch get_llm to return our fakes for the two stages.
        from zone2 import pipeline as p

        called: list = []

        def fake_get_llm(model, json_mode=False, *args, **kwargs):
            called.append(json_mode)
            return stage2 if json_mode else stage1

        monkeypatch.setattr(p, "get_llm", fake_get_llm)

        chunk = {
            "chunk_id": "c1",
            "content": "Earth movement is excluded from coverage.",
            "source": "fema_F-123-general-property-SFIP_2021.pdf",
            "section_hierarchy": ["IV. PROPERTY NOT INSURED"],
            "lob": "flood",
        }
        # Force decomposition by giving zero pass1 triples for this chunk.
        _decompose_then_extract(
            model="qwen2.5:7b",
            chunks=[chunk],
            pass1_triples=[],
            vocab=["EXCLUDED_FROM"],
            errors=[],
        )

        # Stage 1 must have received a prompt that contains the preamble.
        assert stage1.invocations, "Stage 1 LLM was not invoked"
        msg = stage1.invocations[0][0]
        prompt_text = msg.content if hasattr(msg, "content") else str(msg)
        assert "fema_F-123-general-property-SFIP_2021.pdf" in prompt_text
        assert "flood" in prompt_text.lower()
        assert "IV. PROPERTY NOT INSURED" in prompt_text
