import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config
from langchain_ollama import ChatOllama


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "given",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}

PLACEHOLDER_ANSWER_PATTERNS = (
    "e.g.",
    "for example",
    "not explicitly stated",
    "reasonable to assume",
    "we can infer",
    "based on industry standards",
    "appears that",
)

GENERIC_GRAPH_QUESTION_PATTERNS = (
    "what type of record is a claim",
    "what type of customer is classified as",
    "what is the likelihood that a customer would recommend",
    "what type of entity is the person",
)

CATEGORY_SOURCE_RULES = {
    "lookup": {"min": 2},
    "reasoning": {"min": 3},
    "global": {"min": 1},
    "1_hop": {"min": 2},
    "2_hop": {"min": 3},
    "cross_source": {"exact": 2},
}


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _squash_alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[a-z0-9$]+", _normalize_text(value))


def _meaningful_tokens(value: str) -> List[str]:
    return [token for token in _tokenize(value) if len(token) > 2 and token not in STOPWORDS]


def _answer_literal_support(expected_answer: str, evidence_text: str) -> bool:
    evidence_norm = _normalize_text(evidence_text)
    literals = (
        re.findall(r"\$\s*\d+(?:\.\d+)?", expected_answer)
        + re.findall(r"\b\d+(?:\.\d+)?\b", expected_answer)
        + re.findall(r"\b[A-Z][A-Z0-9_\-]+\b", expected_answer)
    )
    if not literals:
        return False
    return any(_normalize_text(literal) in evidence_norm for literal in literals)


def _support_ratio(question: str, expected_answer: str, evidence_text: str) -> float:
    evidence_tokens = set(_meaningful_tokens(evidence_text))
    question_tokens = set(_meaningful_tokens(question))
    answer_tokens = [
        token for token in _meaningful_tokens(expected_answer) if token not in question_tokens
    ]
    if not answer_tokens:
        return 1.0
    supported = sum(1 for token in answer_tokens if token in evidence_tokens)
    return supported / len(answer_tokens)


def _value_mentioned(value: str, text: str) -> bool:
    value_norm = _normalize_text(value)
    text_norm = _normalize_text(text)
    if value_norm and value_norm in text_norm:
        return True

    value_compact = _squash_alnum(value)
    text_compact = _squash_alnum(text)
    return bool(value_compact) and value_compact in text_compact


def _clean_json_response(raw_content: str) -> Dict[str, Any]:
    content = raw_content.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif content.startswith("```"):
        content = content.split("```", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(content)


class LLMAutoGoldReviewer:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.OLLAMA_MODEL
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json",
        )

    def review_entry(
        self,
        entry: Dict[str, Any],
        deterministic_findings: List[Dict[str, Any]],
        evidence_context: str,
    ) -> Dict[str, Any]:
        findings_text = json.dumps(deterministic_findings, indent=2) if deterministic_findings else "[]"
        prompt = f"""
You are reviewing a graph-oriented benchmark item for retrieval QA quality.
Decide whether the question, expected answer, and source mapping are good enough to distinguish
real retrieval from guessing.

Benchmark entry:
{json.dumps(entry, indent=2)}

Available evidence context:
{evidence_context}

Deterministic findings:
{findings_text}

Return ONLY valid JSON with this schema:
{{
  "verdict": "pass" | "needs_rewrite" | "unanswerable_from_evidence" | "rewards_guessing" | "source_mapping_problem" | "too_easy",
  "answerable_from_evidence": true,
  "expected_answer_grounded": true,
  "retrieval_discriminative": true,
  "uses_both_sources": true,
  "scores": {{
    "answerability": 1,
    "grounding": 1,
    "retrieval_discriminativeness": 1
  }},
  "rationale": "Short explanation.",
  "recommended_question": "",
  "recommended_expected_answer": ""
}}

Guidance:
- For graph-only items, judge whether the question is specific enough to require the cited entity/path.
- For cross_source items, judge whether both the KG side and document side are actually needed.
- Use "rewards_guessing" when a generic or speculative answer could look correct without the cited evidence.
- Use "source_mapping_problem" when the source_ids do not appear to support the question/answer well.
- If a rewrite is needed, provide both recommended_question and recommended_expected_answer.
"""
        response = self.llm.invoke(prompt)
        review = _clean_json_response(response.content)
        review["model"] = self.model_name
        return review


def _default_dataset_path() -> str:
    return os.path.join(_ROOT, config.GRAPH_PRIMARY_EVAL_FILE)


def _default_chunks_path() -> str:
    return os.path.join(_ROOT, config.CHUNKS_FILE)


def _default_output_path() -> str:
    return os.path.join(_ROOT, config.RESULTS_DIR, "auto_gold_quality_report.json")


def _check_source_shape(entry: Dict[str, Any], chunk_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    category = entry.get("category", "")
    source_ids = entry.get("source_ids", [])
    rule = CATEGORY_SOURCE_RULES.get(category, {})

    exact = rule.get("exact")
    minimum = rule.get("min")
    if exact is not None and len(source_ids) != exact:
        findings.append(
            {
                "severity": "high",
                "type": "unexpected_source_id_count",
                "message": f"Category '{category}' expects exactly {exact} source IDs, found {len(source_ids)}.",
                "suggestion": "Repair the source_ids mapping or regenerate this benchmark item.",
            }
        )
    elif minimum is not None and len(source_ids) < minimum:
        findings.append(
            {
                "severity": "high",
                "type": "insufficient_source_ids",
                "message": f"Category '{category}' expects at least {minimum} source IDs, found {len(source_ids)}.",
                "suggestion": "Repair the source_ids mapping or regenerate this benchmark item.",
            }
        )

    if category == "cross_source" and len(source_ids) >= 2:
        chunk_id = source_ids[1]
        if chunk_id not in chunk_lookup:
            findings.append(
                {
                    "severity": "high",
                    "type": "missing_cross_source_chunk",
                    "message": f"Cross-source chunk '{chunk_id}' was not found in the processed chunk file.",
                    "suggestion": "Repair the chunk reference or regenerate this benchmark item.",
                }
            )

    return findings


def _check_speculative_answer(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    expected_answer = _normalize_text(entry.get("expected_answer", ""))
    if any(pattern in expected_answer for pattern in PLACEHOLDER_ANSWER_PATTERNS):
        findings.append(
            {
                "severity": "high",
                "type": "speculative_or_placeholder_answer",
                "message": "The expected answer contains placeholder or speculative language.",
                "suggestion": "Replace it with a literal grounded answer or rewrite the question to match the available evidence.",
            }
        )
    return findings


def _check_question_specificity(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    category = entry.get("category", "")
    question = entry.get("question", "")
    question_norm = _normalize_text(question)
    source_ids = entry.get("source_ids", [])

    if category in {"1_hop", "2_hop"}:
        mentions_source = any(_value_mentioned(source_id, question) for source_id in source_ids)
        if not mentions_source and any(pattern in question_norm for pattern in GENERIC_GRAPH_QUESTION_PATTERNS):
            findings.append(
                {
                    "severity": "medium",
                    "type": "generic_graph_question_not_tied_to_source",
                    "message": (
                        "The question is generic and does not clearly mention the cited graph record or path, "
                        "so it may be answerable by guessing rather than retrieval."
                    ),
                    "suggestion": "Rewrite the question to include the specific claim, person, or linked value from source_ids.",
                }
            )

    return findings


def _check_cross_source_grounding(
    entry: Dict[str, Any], chunk_lookup: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if entry.get("category") != "cross_source":
        return findings

    source_ids = entry.get("source_ids", [])
    if len(source_ids) < 2:
        return findings

    chunk = chunk_lookup.get(source_ids[1])
    if not chunk:
        return findings

    chunk_text = chunk.get("content", "")
    support_ratio = _support_ratio(entry.get("question", ""), entry.get("expected_answer", ""), chunk_text)
    literal_support = _answer_literal_support(entry.get("expected_answer", ""), chunk_text)

    if support_ratio < 0.35 and not literal_support:
        findings.append(
            {
                "severity": "medium",
                "type": "cross_source_answer_weak_document_support",
                "message": (
                    f"The expected answer has weak lexical support in the paired document chunk "
                    f"(support ratio {support_ratio:.2f})."
                ),
                "suggestion": "Check whether the answer is relying on inference rather than literal evidence from the paired document source.",
            }
        )

    question_mentions_kg = any(_value_mentioned(source_ids[0], entry.get("question", "")) for _ in [0])
    if not question_mentions_kg and "given the" not in _normalize_text(entry.get("question", "")):
        findings.append(
            {
                "severity": "low",
                "type": "cross_source_question_weak_kg_anchor",
                "message": "The question does not clearly expose the KG-side anchor from source_ids.",
                "suggestion": "Consider mentioning the claim or policy anchor more explicitly in the question text.",
            }
        )

    return findings


def _check_duplicate_questions(entry: Dict[str, Any], duplicate_counts: Counter[str]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    normalized_question = _normalize_text(entry.get("question", ""))
    if duplicate_counts[normalized_question] > 1:
        findings.append(
            {
                "severity": "low",
                "type": "duplicate_question_prompt",
                "message": "This question prompt appears multiple times in the dataset.",
                "suggestion": "Diversify repeated prompts so the benchmark covers more distinct retrieval behaviors.",
            }
        )
    return findings


def _build_evidence_context(entry: Dict[str, Any], chunk_lookup: Dict[str, Dict[str, Any]]) -> str:
    source_ids = entry.get("source_ids", [])
    parts = [f"Category: {entry.get('category')}", f"Source IDs: {source_ids}"]
    if entry.get("category") == "cross_source" and len(source_ids) >= 2:
        chunk = chunk_lookup.get(source_ids[1])
        if chunk:
            parts.append(f"Document chunk content:\n{chunk.get('content', '')}")
    else:
        parts.append(
            "Graph evidence is represented only by source_ids in this audit. "
            "No live Neo4j traversal was executed by the deterministic checker."
        )
    return "\n\n".join(parts)


def _should_run_llm_review(entry: Dict[str, Any], findings: List[Dict[str, Any]], llm_scope: str) -> bool:
    if llm_scope == "all":
        return True
    if llm_scope == "flagged":
        return bool(findings)
    if llm_scope == "cross_source":
        return entry.get("category") == "cross_source"
    if llm_scope == "flagged_cross_source":
        return entry.get("category") == "cross_source" and bool(findings)
    return False


def audit_auto_gold_quality(
    dataset_path: str,
    chunks_path: str,
    llm_reviewer: Optional[LLMAutoGoldReviewer] = None,
    llm_scope: str = "flagged",
) -> Dict[str, Any]:
    dataset = _load_json(dataset_path)
    chunks = _load_json(chunks_path)
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    duplicate_counts = Counter(_normalize_text(entry.get("question", "")) for entry in dataset)

    report_entries: List[Dict[str, Any]] = []
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    llm_review_counts: Counter[str] = Counter()
    deterministic_flagged_count = 0

    for entry in dataset:
        findings: List[Dict[str, Any]] = []
        findings.extend(_check_source_shape(entry, chunk_lookup))
        findings.extend(_check_speculative_answer(entry))
        findings.extend(_check_question_specificity(entry))
        findings.extend(_check_cross_source_grounding(entry, chunk_lookup))
        findings.extend(_check_duplicate_questions(entry, duplicate_counts))

        for finding in findings:
            severity_counts[finding["severity"]] += 1
        if findings:
            deterministic_flagged_count += 1

        llm_review = None
        llm_review_error = None
        if llm_reviewer and _should_run_llm_review(entry, findings, llm_scope):
            try:
                llm_review = llm_reviewer.review_entry(
                    entry,
                    findings,
                    _build_evidence_context(entry, chunk_lookup),
                )
                llm_review_counts[llm_review.get("verdict", "unknown")] += 1
            except Exception as exc:
                llm_review_error = str(exc)
                llm_review_counts["error"] += 1

        if findings or llm_review or llm_review_error:
            report_entry = {
                "id": entry.get("id"),
                "category": entry.get("category"),
                "question": entry.get("question"),
                "expected_answer": entry.get("expected_answer"),
                "source_ids": entry.get("source_ids", []),
                "findings": findings,
            }
            if llm_review is not None:
                report_entry["llm_review"] = llm_review
            if llm_review_error is not None:
                report_entry["llm_review_error"] = llm_review_error
            report_entries.append(report_entry)

    return {
        "dataset_path": dataset_path,
        "chunks_path": chunks_path,
        "total_entries": len(dataset),
        "flagged_entries": deterministic_flagged_count,
        "report_entries": len(report_entries),
        "severity_counts": severity_counts,
        "llm_reviewed_entries": sum(llm_review_counts.values()),
        "llm_review_counts": dict(llm_review_counts),
        "entries": report_entries,
    }


def run() -> None:
    parser = argparse.ArgumentParser(description="Audit auto gold-standard question quality.")
    parser.add_argument(
        "-i",
        "--input",
        default=_default_dataset_path(),
        help="Path to the auto gold dataset JSON.",
    )
    parser.add_argument(
        "--chunks",
        default=_default_chunks_path(),
        help="Path to the processed chunk JSON file used for cross-source checks.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=_default_output_path(),
        help="Path to write the JSON audit report.",
    )
    parser.add_argument(
        "--llm-review",
        action="store_true",
        help="Run an LLM-based semantic review in addition to deterministic linting.",
    )
    parser.add_argument(
        "--llm-scope",
        choices=["flagged", "cross_source", "flagged_cross_source", "all"],
        default="flagged",
        help="Which entries to send to the LLM reviewer.",
    )
    parser.add_argument(
        "--model",
        help="Optional Ollama model override for LLM review.",
    )
    args = parser.parse_args()

    dataset_path = args.input if os.path.isabs(args.input) else os.path.join(_ROOT, args.input)
    chunks_path = args.chunks if os.path.isabs(args.chunks) else os.path.join(_ROOT, args.chunks)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(_ROOT, args.output)

    llm_reviewer = LLMAutoGoldReviewer(model_name=args.model) if args.llm_review else None
    report = audit_auto_gold_quality(
        dataset_path,
        chunks_path,
        llm_reviewer=llm_reviewer,
        llm_scope=args.llm_scope,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("AUTO GOLD QUALITY AUDIT")
    print("=" * 72)
    print(f"Dataset: {dataset_path}")
    print(f"Chunks:  {chunks_path}")
    print(f"Flagged entries: {report['flagged_entries']} / {report['total_entries']}")
    print(f"Severity counts: {report['severity_counts']}")
    if args.llm_review:
        print(f"LLM reviewed entries: {report['llm_reviewed_entries']}")
        print(f"LLM verdict counts: {report['llm_review_counts']}")
    print(f"Report entries written: {report['report_entries']}")
    print(f"Report written to: {output_path}")

    for entry in report["entries"]:
        print("-" * 72)
        print(f"{entry['id']} [{entry['category']}]: {entry['question']}")
        for finding in entry["findings"]:
            print(f"  [{finding['severity']}] {finding['type']}: {finding['message']}")
        if "llm_review" in entry:
            review = entry["llm_review"]
            print(f"  [llm] verdict={review.get('verdict')} rationale={review.get('rationale')}")
        if "llm_review_error" in entry:
            print(f"  [llm-error] {entry['llm_review_error']}")


if __name__ == "__main__":
    run()
