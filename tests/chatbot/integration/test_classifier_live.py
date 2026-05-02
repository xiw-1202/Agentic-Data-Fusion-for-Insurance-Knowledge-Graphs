"""Live integration test: classify_question against the real Anthropic API.

Loads the project ``.env`` for credentials and the cached
``chatbot/schema_prefix.cache.txt`` for context, then asks the
classifier to triage real-world questions.

Auto-skips when the API key is not available.  Marked ``slow`` because
each test makes a real API call.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module", autouse=True)
def load_env():
    """Load .env so ANTHROPIC_API_KEY is available, then verify.

    Searches the worktree root first, then walks up to the main repo
    root (worktrees are at ``<repo>/.claude/worktrees/<name>/``).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        pytest.skip("python-dotenv not installed")

    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / ".env",                # worktree root
        here.parents[3] / ".." / ".." / ".." / ".env",  # repo root from worktree
    ]
    for env_path in candidates:
        env_path = env_path.resolve()
        if env_path.exists():
            # override=True because some shells export an empty
            # ANTHROPIC_API_KEY which would otherwise block the file.
            load_dotenv(env_path, override=True)
            break
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; skipping live classifier test")


@pytest.fixture(scope="module")
def schema_prefix() -> str:
    """Real schema cache from the chatbot directory.

    Looks in the worktree first, then in the main repo (one parent up
    through ``.claude/worktrees/<name>``).
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "chatbot" / "schema_prefix.cache.txt",
        (here.parents[3] / ".." / ".." / ".." / "chatbot"
         / "schema_prefix.cache.txt").resolve(),
    ]
    for cache in candidates:
        if cache.exists():
            return cache.read_text(encoding="utf-8")
    pytest.skip("schema_prefix.cache.txt not built; run build_schema_cache.py")


class TestClassifierLive:
    @pytest.mark.parametrize("question", [
        "what time was mold damage claim happened?",
        "when did mold claims occur?",
        "show me all claims with cause of loss = MOLD",
        "list policies in OH",
        "how many claims are in the GEICO renters dataset?",
    ])
    def test_general_kg_questions_are_factual(self, schema_prefix, question):
        """A general lookup question over a category the KG can filter on
        (mold cause-of-loss, OH policies, claim counts) must NOT be marked
        needs_clarification just because no specific row ID was given."""
        from chatbot.classifier import classify_question, QuestionKind

        result = classify_question(question, schema_prefix=schema_prefix)
        assert result.kind != QuestionKind.NEEDS_CLARIFICATION, (
            f"Question {question!r} was over-flagged as needs_clarification "
            f"(reason: {result.reason})"
        )
        # Should land in factual_kg (preferred) or open_interpretive
        # (acceptable for borderline reasoning questions).
        assert result.kind in (
            QuestionKind.FACTUAL_KG,
            QuestionKind.OPEN_INTERPRETIVE,
        ), f"unexpected kind: {result.kind}"

    @pytest.mark.parametrize("question", [
        "tell me about claim CLM-12345",          # specific ID, factual lookup
        "what is the loss date for policy POL-XYZ",  # specific ID
    ])
    def test_specific_id_lookups_are_factual(self, schema_prefix, question):
        from chatbot.classifier import classify_question, QuestionKind

        result = classify_question(question, schema_prefix=schema_prefix)
        assert result.kind == QuestionKind.FACTUAL_KG, (
            f"specific-ID question {question!r} → {result.kind} "
            f"(reason: {result.reason})"
        )

    @pytest.mark.parametrize("question", [
        "what is the meaning of life?",
        "should I switch insurance providers?",
    ])
    def test_unrelated_questions_are_out_of_scope(self, schema_prefix, question):
        from chatbot.classifier import classify_question, QuestionKind

        result = classify_question(question, schema_prefix=schema_prefix)
        assert result.kind in (
            QuestionKind.OUT_OF_SCOPE,
            QuestionKind.NEEDS_CLARIFICATION,
        ), f"unrelated question {question!r} accepted as {result.kind}"

    def test_genuinely_ambiguous_question_asks_for_clarification(self, schema_prefix):
        """A question with NO actionable filter and a vague verb should
        legitimately be flagged as needs_clarification."""
        from chatbot.classifier import classify_question, QuestionKind

        result = classify_question("tell me about it", schema_prefix=schema_prefix)
        assert result.kind == QuestionKind.NEEDS_CLARIFICATION, (
            f"truly vague question accepted as {result.kind}"
        )
