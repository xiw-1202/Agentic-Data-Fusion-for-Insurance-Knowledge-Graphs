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
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "under",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}

ENUMERATION_PATTERNS = (
    "what services",
    "which services",
    "what items",
    "which items",
    "what types",
    "which types",
    "what breakdowns",
    "which breakdowns",
)

WEAK_REFERENCE_PATTERNS = (
    "specified services",
    "items listed under",
    "listed under the sections",
    "listed under section",
    "such a defect",
)

LOCATION_ONLY_PATTERNS = (
    "listed under section",
    "listed under the sections",
    "listed under section iv",
    "listed under section v",
)

LIST_EVIDENCE_MARKERS = (
    "•",
    " a. ",
    " b. ",
    " c. ",
    " d. ",
    ": ",
)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[a-z0-9$]+", _normalize_text(value))


def _meaningful_tokens(value: str) -> List[str]:
    return [token for token in _tokenize(value) if len(token) > 2 and token not in STOPWORDS]


def _answer_literal_support(expected_answer: str, chunk_text: str) -> bool:
    chunk_norm = _normalize_text(chunk_text)
    currency_values = re.findall(r"\$\s*\d+(?:\.\d+)?", expected_answer)
    numeric_values = re.findall(r"\b\d+(?:\.\d+)?\b", expected_answer)
    uppercase_terms = re.findall(r"\b[A-Z][A-Z_]+\b", expected_answer)

    literals = currency_values + numeric_values + uppercase_terms
    if not literals:
        return False

    return any(_normalize_text(literal) in chunk_norm for literal in literals)


def _is_atomic_lookup_question(question: str) -> bool:
    lowered = _normalize_text(question)
    patterns = (
        "what is the nps score",
        "what is the current status",
        "what is the total claim time",
        "what is the required transfer fee",
        "what is the minimum notice period",
        "what is the maximum amount",
        "what is the maximum daily rental reimbursement",
    )
    return any(pattern in lowered for pattern in patterns)


def _chunk_has_list_evidence(chunk_text: str) -> bool:
    lowered = _normalize_text(chunk_text)
    return any(marker in lowered for marker in LIST_EVIDENCE_MARKERS)


def _keyword_support(keyword: str, chunk_text: str) -> bool:
    keyword_norm = _normalize_text(keyword)
    chunk_norm = _normalize_text(chunk_text)
    if keyword_norm in chunk_norm:
        return True

    keyword_tokens = [token for token in _meaningful_tokens(keyword) if not token.isdigit()]
    if not keyword_tokens:
        return False

    matched = sum(1 for token in keyword_tokens if token in chunk_norm)
    return matched >= max(1, len(keyword_tokens) - 1)


def _answer_support_ratio(question: str, expected_answer: str, chunk_text: str) -> float:
    chunk_tokens = set(_meaningful_tokens(chunk_text))
    question_tokens = set(_meaningful_tokens(question))

    answer_tokens = [
        token
        for token in _meaningful_tokens(expected_answer)
        if token not in question_tokens
    ]
    if not answer_tokens:
        return 1.0

    supported = sum(1 for token in answer_tokens if token in chunk_tokens)
    return supported / len(answer_tokens)


def _question_requests_enumeration(question: str) -> bool:
    lowered = _normalize_text(question)
    return any(pattern in lowered for pattern in ENUMERATION_PATTERNS)


def _analyze_entry(entry: Dict[str, Any], chunk_text: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    question = entry.get("question", "")
    expected_answer = entry.get("expected_answer", "")
    keywords = entry.get("keywords", [])

    question_norm = _normalize_text(question)
    answer_norm = _normalize_text(expected_answer)
    chunk_norm = _normalize_text(chunk_text)

    if _question_requests_enumeration(question):
        weak_reference = any(pattern in chunk_norm for pattern in WEAK_REFERENCE_PATTERNS)
        if weak_reference and not _chunk_has_list_evidence(chunk_text):
            findings.append(
                {
                    "severity": "high",
                    "type": "under_specified_enumeration",
                    "message": (
                        "The question asks for a concrete list or category, but the chunk only "
                        "contains a weak reference such as 'specified services' or 'items listed under'."
                    ),
                    "suggestion": (
                        "Rewrite the question so it asks about the effect, condition, or location "
                        "explicitly stated in the chunk."
                    ),
                }
            )

    if _question_requests_enumeration(question):
        if any(pattern in chunk_norm for pattern in LOCATION_ONLY_PATTERNS) and "where" not in question_norm:
            findings.append(
                {
                    "severity": "high",
                    "type": "asks_for_content_but_chunk_only_gives_location",
                    "message": (
                        "The chunk appears to point to where information is listed rather than "
                        "enumerating the information itself."
                    ),
                    "suggestion": (
                        "Consider a question shaped like 'where are the excluded items listed?' "
                        "instead of asking for the full excluded item set."
                    ),
                }
            )

    support_ratio = _answer_support_ratio(question, expected_answer, chunk_text)
    literal_support = _answer_literal_support(expected_answer, chunk_text)
    if support_ratio < 0.45 and not (_is_atomic_lookup_question(question) and literal_support):
        findings.append(
            {
                "severity": "medium",
                "type": "expected_answer_low_chunk_support",
                "message": (
                    f"The expected answer has low lexical support in the source chunk "
                    f"(support ratio {support_ratio:.2f})."
                ),
                "suggestion": (
                    "Review whether the answer is paraphrased too aggressively or includes "
                    "details not visible in the chunk."
                ),
            }
        )

    unsupported_keywords = [keyword for keyword in keywords if not _keyword_support(keyword, chunk_text)]
    if unsupported_keywords:
        findings.append(
            {
                "severity": "low",
                "type": "keywords_not_clearly_supported",
                "message": (
                    "Some benchmark keywords are not clearly recoverable from the source chunk."
                ),
                "unsupported_keywords": unsupported_keywords,
                "suggestion": (
                    "Tighten the keyword list so it better matches the literal chunk content, "
                    "especially for PDF-based items."
                ),
            }
        )

    if (
        any(pattern in answer_norm for pattern in LOCATION_ONLY_PATTERNS)
        and "where" not in question_norm
        and "listed" in answer_norm
    ):
        findings.append(
            {
                "severity": "medium",
                "type": "answer_is_location_not_content",
                "message": (
                    "The expected answer mainly points to sections or locations instead of "
                    "answering a content question directly."
                ),
                "suggestion": (
                    "Either rewrite the question to ask where the information is listed, or swap "
                    "in a chunk that actually contains the excluded items."
                ),
            }
        )

    return findings


def _clean_json_response(raw_content: str) -> Dict[str, Any]:
    content = raw_content.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif content.startswith("```"):
        content = content.split("```", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(content)


class LLMGoldReviewer:
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
        chunk_text: str,
        deterministic_findings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        findings_text = json.dumps(deterministic_findings, indent=2) if deterministic_findings else "[]"
        prompt = f"""
You are reviewing a document-grounded benchmark item for retrieval QA quality.
Your goal is to determine whether the question and expected answer are fully supported by the provided chunk,
and whether this benchmark item clearly distinguishes real retrieval from guessing.

Benchmark entry:
{json.dumps(entry, indent=2)}

Source chunk:
{chunk_text}

Deterministic audit findings:
{findings_text}

Return ONLY valid JSON with this schema:
{{
  "verdict": "pass" | "needs_rewrite" | "unanswerable_from_chunk" | "rewards_guessing" | "too_easy",
  "answerable_from_chunk": true,
  "expected_answer_grounded": true,
  "retrieval_discriminative": true,
  "scores": {{
    "answerability": 1,
    "grounding": 1,
    "retrieval_discriminativeness": 1
  }},
  "rationale": "Short explanation.",
  "recommended_question": "",
  "recommended_expected_answer": ""
}}

Scoring guidance:
- answerability: can a faithful model answer from this chunk alone?
- grounding: is the expected answer directly supported by the chunk?
- retrieval_discriminativeness: would a model need the right chunk instead of bluffing?

Use "pass" only when the item is clearly answerable from the chunk, grounded, and retrieval-discriminative.
Use "needs_rewrite" when the item is mostly salvageable with a better question or answer.
Use "unanswerable_from_chunk" when the chunk does not contain enough information.
Use "rewards_guessing" when a model could plausibly sound right without retrieving the exact chunk.
Use "too_easy" when the item is valid but extremely trivial.
If you recommend a rewrite, provide both recommended_question and recommended_expected_answer.
"""

        response = self.llm.invoke(prompt)
        review = _clean_json_response(response.content)
        review["model"] = self.model_name
        return review


def _should_run_llm_review(
    entry: Dict[str, Any],
    findings: List[Dict[str, Any]],
    llm_scope: str,
) -> bool:
    if llm_scope == "all":
        return True
    if llm_scope == "flagged":
        return bool(findings)
    if llm_scope == "pdf_only":
        return entry.get("category") == "pdf_fact"
    if llm_scope == "flagged_pdf":
        return entry.get("category") == "pdf_fact" and bool(findings)
    return False


def audit_doc_gold_quality(
    dataset_path: str,
    chunks_path: str,
    llm_reviewer: Optional[LLMGoldReviewer] = None,
    llm_scope: str = "flagged",
) -> Dict[str, Any]:
    dataset = _load_json(dataset_path)
    chunks = _load_json(chunks_path)
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

    report_entries: List[Dict[str, Any]] = []
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    llm_review_counts: Counter[str] = Counter()
    deterministic_flagged_count = 0

    for entry in dataset:
        source_ids = entry.get("source_ids", [])
        source_id = source_ids[0] if source_ids else None
        chunk = chunk_lookup.get(source_id)
        chunk_text = chunk.get("content", "") if chunk else ""

        if not chunk:
            findings = [
                {
                    "severity": "high",
                    "type": "missing_source_chunk",
                    "message": "The source chunk referenced by this benchmark entry could not be found.",
                    "suggestion": "Repair or regenerate the dataset so each question points to a valid chunk.",
                }
            ]
        else:
            findings = _analyze_entry(entry, chunk_text)

        for finding in findings:
            severity_counts[finding["severity"]] += 1
        if findings:
            deterministic_flagged_count += 1

        llm_review = None
        llm_review_error = None
        if llm_reviewer and chunk and _should_run_llm_review(entry, findings, llm_scope):
            try:
                llm_review = llm_reviewer.review_entry(entry, chunk_text, findings)
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
                "source_id": source_id,
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


def _default_dataset_path() -> str:
    return os.path.join(_ROOT, config.DOC_PRIMARY_EVAL_FILE)


def _default_chunks_path() -> str:
    return os.path.join(_ROOT, config.CHUNKS_FILE)


def _default_output_path() -> str:
    return os.path.join(_ROOT, config.RESULTS_DIR, "doc_gold_quality_report.json")


def run() -> None:
    parser = argparse.ArgumentParser(description="Audit doc-primary gold-standard question quality.")
    parser.add_argument(
        "-i",
        "--input",
        default=_default_dataset_path(),
        help="Path to the evaluation dataset JSON.",
    )
    parser.add_argument(
        "--chunks",
        default=_default_chunks_path(),
        help="Path to the source chunk JSON file.",
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
        choices=["flagged", "pdf_only", "flagged_pdf", "all"],
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

    llm_reviewer = LLMGoldReviewer(model_name=args.model) if args.llm_review else None
    report = audit_doc_gold_quality(
        dataset_path,
        chunks_path,
        llm_reviewer=llm_reviewer,
        llm_scope=args.llm_scope,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("DOC GOLD QUALITY AUDIT")
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
        print(f"{entry['id']}: {entry['question']}")
        for finding in entry["findings"]:
            print(f"  [{finding['severity']}] {finding['type']}: {finding['message']}")
        if "llm_review" in entry:
            review = entry["llm_review"]
            print(f"  [llm] verdict={review.get('verdict')} rationale={review.get('rationale')}")
        if "llm_review_error" in entry:
            print(f"  [llm-error] {entry['llm_review_error']}")


if __name__ == "__main__":
    run()
