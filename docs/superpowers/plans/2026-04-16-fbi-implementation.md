# FBI (Fingerprint-Based Ontology Induction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a new Zone 3 ontology induction pipeline that discovers multi-level class hierarchies from file-level structural metadata (CSV headers, PDF sections, filenames) rather than entity-level LLM classification.

**Architecture:** 4-phase pipeline — Phase 1 extracts and expands headers from raw files, Phase 2 discovers classes via prefix trie + semantic grouping + cross-file merging, Phase 3 discovers inter-class relationships from bridge columns, Phase 4 assigns Zone 2 entities to discovered classes. Phases 1-3 are independent of Zone 2.

**Tech Stack:** Python 3.10+, langchain-ollama (ChatOllama, qwen2.5:72b), pandas (CSV reading), pdfplumber (PDF sections), neo4j (graph storage), pytest (testing)

**Spec:** `docs/superpowers/specs/2026-04-16-fingerprint-ontology-induction-design.md`

**Data directory:** `data/Emory_Spring2026/`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `zone3/fbi/__init__.py` | Package exports |
| `zone3/fbi/fingerprint.py` | Phase 1: extract headers, expand abbreviations, parse filenames |
| `zone3/fbi/class_discovery.py` | Phase 2: prefix trie, semantic grouping, cross-file merging, naming |
| `zone3/fbi/relationships.py` | Phase 3: bridge column detection, relationship naming |
| `zone3/fbi/entity_assign.py` | Phase 4: assign entities to classes, write to Neo4j |
| `zone3/fbi/llm_utils.py` | Shared LLM calling helpers (prompt, parse, retry) |
| `zone3/fbi/pipeline.py` | Main entry point, orchestrates all 4 phases |
| `tests/test_fingerprint.py` | Tests for Phase 1 |
| `tests/test_class_discovery.py` | Tests for Phase 2 |
| `tests/test_relationships.py` | Tests for Phase 3 |
| `tests/test_entity_assign.py` | Tests for Phase 4 |

---

## Task 1: LLM Utilities Module

**Files:**
- Create: `zone3/fbi/__init__.py`
- Create: `zone3/fbi/llm_utils.py`
- Test: `tests/test_llm_utils.py`

- [ ] **Step 1: Create package and LLM utility module**

```python
# zone3/fbi/__init__.py
"""Fingerprint-Based Ontology Induction (FBI) for Zone 3."""
```

```python
# zone3/fbi/llm_utils.py
"""Shared LLM calling utilities for FBI pipeline."""

from __future__ import annotations

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def get_llm(model: str | None = None, json_mode: bool = False,
            num_predict: int = 4096) -> ChatOllama:
    """Return a ChatOllama instance."""
    kwargs: dict = dict(
        model=model or config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
        num_predict=num_predict,
    )
    if json_mode:
        kwargs["format"] = "json"
    return kwargs


def llm_call(prompt: str, model: str | None = None,
             json_mode: bool = False,
             num_predict: int = 4096) -> str:
    """Make a single LLM call and return the response text.

    Args:
        prompt: The prompt string (self-contained, no external references).
        model: Ollama model name. Defaults to config.OLLAMA_MODEL.
        json_mode: If True, request JSON output format.
        num_predict: Max tokens to generate.

    Returns:
        The stripped response text from the LLM.
    """
    llm = ChatOllama(
        model=model or config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
        num_predict=num_predict,
    )
    if json_mode:
        llm = ChatOllama(
            model=model or config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            num_predict=num_predict,
            format="json",
        )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def llm_call_json(prompt: str, model: str | None = None,
                  num_predict: int = 4096) -> dict | list:
    """Make an LLM call expecting JSON output. Parse and return.

    Falls back to regex extraction if direct parsing fails.
    """
    raw = llm_call(prompt, model=model, json_mode=True,
                   num_predict=num_predict)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code blocks
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Try finding first { or [
        for i, ch in enumerate(raw):
            if ch in "{[":
                try:
                    return json.loads(raw[i:])
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:500]}")


def parse_arrow_mapping(text: str) -> dict[str, str]:
    """Parse 'KEY → Value' lines from LLM output.

    Returns dict mapping original key to expanded value.
    """
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match patterns like: KEY → Value  or  KEY -> Value
        match = re.match(r"^(.+?)\s*[→\->]+\s*(.+)$", line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            result[key] = value
    return result
```

- [ ] **Step 2: Write tests for llm_utils**

```python
# tests/test_llm_utils.py
"""Tests for FBI LLM utilities — parsing functions only (no live LLM)."""

import pytest
from zone3.fbi.llm_utils import parse_arrow_mapping


class TestParseArrowMapping:
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
        text = "\nSB → Sub-Business\n\nMCO → Master Company\n"
        result = parse_arrow_mapping(text)
        assert len(result) == 2

    def test_no_arrows(self) -> None:
        text = "This is just text with no arrows"
        result = parse_arrow_mapping(text)
        assert result == {}

    def test_multi_word_key(self) -> None:
        text = "COVAMT_PERS → Coverage Amount Personal Property"
        result = parse_arrow_mapping(text)
        assert result["COVAMT_PERS"] == "Coverage Amount Personal Property"
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/sam/Documents/School/Emory/CS584_AI_Capstone/.claude/worktrees/suspicious-sanderson && python -m pytest tests/test_llm_utils.py -v`

Expected: All 5 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add zone3/fbi/__init__.py zone3/fbi/llm_utils.py tests/test_llm_utils.py
git commit -m "feat(fbi): add LLM utilities module with prompt parsing helpers"
```

---

## Task 2: Phase 1 — Header Extraction (Algorithmic)

**Files:**
- Create: `zone3/fbi/fingerprint.py`
- Test: `tests/test_fingerprint.py`

- [ ] **Step 1: Write failing tests for CSV header extraction**

```python
# tests/test_fingerprint.py
"""Tests for FBI Phase 1: fingerprint extraction."""

import pytest
import os
import tempfile
import csv
from zone3.fbi.fingerprint import (
    extract_csv_headers,
    extract_pdf_sections,
    extract_txt_sections,
    strip_audit_columns,
    FileFingerprint,
)


class TestExtractCsvHeaders:
    def test_basic_csv(self, tmp_path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("NAME,AGE,CITY\nAlice,30,NYC\n")
        headers = extract_csv_headers(str(csv_file))
        assert headers == ["NAME", "AGE", "CITY"]

    def test_empty_csv(self, tmp_path) -> None:
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        headers = extract_csv_headers(str(csv_file))
        assert headers == []

    def test_whitespace_in_headers(self, tmp_path) -> None:
        csv_file = tmp_path / "ws.csv"
        csv_file.write_text(" NAME , AGE , CITY \nAlice,30,NYC\n")
        headers = extract_csv_headers(str(csv_file))
        assert headers == ["NAME", "AGE", "CITY"]


class TestStripAuditColumns:
    def test_removes_bi_columns(self) -> None:
        headers_by_file = {
            "file1.csv": ["NAME", "BI_CREATED_DT", "BI_MODIFIED_BY", "AGE"],
            "file2.csv": ["CITY", "BI_CREATED_DT", "BI_MODIFIED_BY", "ZIP"],
        }
        result = strip_audit_columns(headers_by_file)
        assert "BI_CREATED_DT" not in result["file1.csv"]
        assert "BI_MODIFIED_BY" not in result["file1.csv"]
        assert "NAME" in result["file1.csv"]

    def test_keeps_unique_columns(self) -> None:
        headers_by_file = {
            "file1.csv": ["NAME", "SPECIAL_COL"],
            "file2.csv": ["CITY"],
        }
        result = strip_audit_columns(headers_by_file)
        assert "SPECIAL_COL" in result["file1.csv"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fingerprint.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'zone3.fbi.fingerprint'`

- [ ] **Step 3: Implement fingerprint.py — CSV and audit stripping**

```python
# zone3/fbi/fingerprint.py
"""Phase 1: Extract and normalize headers from raw data files.

Reads CSV column headers, PDF section headings, and TXT section headings.
Strips universal audit columns. Optionally expands cryptic abbreviations
via LLM.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileFingerprint:
    """Metadata record for a single source file."""
    file_path: str
    file_type: str  # "csv", "pdf", "txt"
    filename_tokens: list[str] = field(default_factory=list)
    headers_raw: list[str] = field(default_factory=list)
    headers_expanded: dict[str, str] = field(default_factory=dict)
    sections: list[str] = field(default_factory=list)
    defined_terms: list[str] = field(default_factory=list)
    record_count: int = 0

    @property
    def basename(self) -> str:
        return os.path.basename(self.file_path)

    @property
    def headers(self) -> list[str]:
        """Return expanded headers if available, else raw."""
        if self.headers_expanded:
            return list(self.headers_expanded.values())
        return self.headers_raw


def extract_csv_headers(file_path: str) -> list[str]:
    """Extract column headers from a CSV file.

    Returns list of stripped, uppercase header names.
    Returns empty list if file is empty.
    """
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
        if not first_row:
            return []
        return [h.strip().upper() for h in first_row if h.strip()]
    except (FileNotFoundError, StopIteration):
        return []


def extract_pdf_sections(file_path: str) -> tuple[list[str], list[str]]:
    """Extract section headings and defined terms from a PDF file.

    Returns:
        (section_headings, defined_terms)
    """
    try:
        import pdfplumber
    except ImportError:
        return [], []

    sections: list[str] = []
    defined_terms: list[str] = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Detect section headings: lines in ALL CAPS or with Roman numeral prefix
                if (line.isupper() and len(line) > 3 and len(line) < 80
                        and not line.startswith("•")):
                    sections.append(line)
                # Detect defined terms: "Term" means ...  or  "Term" pattern
                term_match = re.match(
                    r'["\u201c]([A-Z][A-Za-z\s]+)["\u201d]\s*means', line
                )
                if term_match:
                    defined_terms.append(term_match.group(1).strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_sections: list[str] = []
    for s in sections:
        if s not in seen:
            seen.add(s)
            unique_sections.append(s)

    return unique_sections, defined_terms


def extract_txt_sections(file_path: str) -> tuple[list[str], list[str]]:
    """Extract section headings and key terms from a TXT file.

    Looks for lines in ALL CAPS (section headings) and defined terms.
    Returns:
        (section_headings, defined_terms)
    """
    sections: list[str] = []
    defined_terms: list[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Section headings: ALL CAPS lines
        if line.isupper() and len(line) > 3 and len(line) < 80:
            sections.append(line)
        # Defined terms: "Term" means ... or **Term** means ...
        term_match = re.match(
            r'(?:["\u201c]|(?:\*\*))([A-Z][A-Za-z\s]+)(?:["\u201d]|(?:\*\*))\s*means',
            line,
        )
        if term_match:
            defined_terms.append(term_match.group(1).strip())

    seen: set[str] = set()
    unique_sections: list[str] = []
    for s in sections:
        if s not in seen:
            seen.add(s)
            unique_sections.append(s)

    return unique_sections, defined_terms


def strip_audit_columns(
    headers_by_file: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Remove columns that appear in ALL files with identical names.

    These are typically audit/metadata columns (BI_CREATED_DT, etc.)
    that carry no domain meaning.
    """
    if not headers_by_file:
        return headers_by_file

    all_files = list(headers_by_file.keys())
    if len(all_files) < 2:
        return headers_by_file

    # Find headers present in ALL files
    common = set(headers_by_file[all_files[0]])
    for f in all_files[1:]:
        common &= set(headers_by_file[f])

    # Remove common audit-pattern columns
    audit_patterns = re.compile(
        r"^(BI_CREATED|BI_MODIFIED|CREATED_BY|MODIFIED_BY|CREATED_DT|MODIFIED_DT)",
        re.IGNORECASE,
    )
    audit_cols = {h for h in common if audit_patterns.match(h)}

    return {
        f: [h for h in headers if h not in audit_cols]
        for f, headers in headers_by_file.items()
    }


def count_csv_rows(file_path: str) -> int:
    """Count data rows in a CSV (excluding header)."""
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return sum(1 for _ in f) - 1
    except FileNotFoundError:
        return 0


def extract_fingerprints(data_dir: str) -> list[FileFingerprint]:
    """Extract fingerprints from all files in a data directory.

    Scans for CSV, PDF, and TXT files. Returns one FileFingerprint per file.
    Does NOT call the LLM — that is handled by expand_headers() separately.
    """
    fingerprints: list[FileFingerprint] = []
    data_path = Path(data_dir)

    # Collect all files
    files: list[tuple[str, str]] = []  # (path, type)
    for ext, ftype in [("*.csv", "csv"), ("*.pdf", "pdf"), ("*.txt", "txt")]:
        for f in data_path.glob(ext):
            files.append((str(f), ftype))
        # Also check subdirectories (e.g., web_policies/)
        for f in data_path.rglob(ext):
            if str(f) not in [x[0] for x in files]:
                files.append((str(f), ftype))

    for file_path, file_type in files:
        fp = FileFingerprint(file_path=file_path, file_type=file_type)

        if file_type == "csv":
            fp.headers_raw = extract_csv_headers(file_path)
            fp.record_count = count_csv_rows(file_path)
        elif file_type == "pdf":
            fp.sections, fp.defined_terms = extract_pdf_sections(file_path)
        elif file_type == "txt":
            fp.sections, fp.defined_terms = extract_txt_sections(file_path)

        fingerprints.append(fp)

    # Strip audit columns from CSV files
    csv_headers = {
        fp.file_path: fp.headers_raw
        for fp in fingerprints
        if fp.file_type == "csv"
    }
    if csv_headers:
        cleaned = strip_audit_columns(csv_headers)
        for fp in fingerprints:
            if fp.file_path in cleaned:
                fp.headers_raw = cleaned[fp.file_path]

    return fingerprints
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fingerprint.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add zone3/fbi/fingerprint.py tests/test_fingerprint.py
git commit -m "feat(fbi): Phase 1 header extraction — CSV, PDF, TXT parsing"
```

---

## Task 3: Phase 1 — LLM Header Expansion + Filename Parsing

**Files:**
- Modify: `zone3/fbi/fingerprint.py`
- Test: `tests/test_fingerprint.py` (add tests)

- [ ] **Step 1: Write tests for LLM expansion helpers**

Add to `tests/test_fingerprint.py`:

```python
from zone3.fbi.fingerprint import (
    build_header_expansion_prompt,
    build_filename_parse_prompt,
    apply_expansions,
)


class TestBuildHeaderExpansionPrompt:
    def test_prompt_contains_headers(self) -> None:
        headers = ["SB", "MCO", "COVAMT_PERS"]
        prompt = build_header_expansion_prompt(headers)
        assert "SB" in prompt
        assert "MCO" in prompt
        assert "COVAMT_PERS" in prompt
        assert "expand" in prompt.lower() or "full" in prompt.lower()

    def test_prompt_batch_size(self) -> None:
        headers = [f"COL_{i}" for i in range(100)]
        prompts = build_header_expansion_prompt(headers, batch_size=50)
        # Should return a string for a single batch if called with <=batch_size
        assert isinstance(prompts, str)


class TestBuildFilenameParsePrompt:
    def test_prompt_contains_filenames(self) -> None:
        filenames = ["synthetic_data_sample_geicorentersclaims.csv",
                     "Auto_Service_form_masked.pdf"]
        prompt = build_filename_parse_prompt(filenames)
        assert "geicorentersclaims" in prompt
        assert "Auto_Service" in prompt


class TestApplyExpansions:
    def test_applies_mapping(self) -> None:
        fp = FileFingerprint(
            file_path="test.csv",
            file_type="csv",
            headers_raw=["SB", "MCO", "POLICY_NUMBER"],
        )
        expansions = {
            "SB": "Sub-Business",
            "MCO": "Master Company Organization",
            "POLICY_NUMBER": "Policy Number",
        }
        apply_expansions(fp, expansions)
        assert fp.headers_expanded["SB"] == "Sub-Business"
        assert fp.headers_expanded["POLICY_NUMBER"] == "Policy Number"

    def test_missing_expansion_uses_raw(self) -> None:
        fp = FileFingerprint(
            file_path="test.csv",
            file_type="csv",
            headers_raw=["SB", "UNKNOWN_HEADER"],
        )
        expansions = {"SB": "Sub-Business"}
        apply_expansions(fp, expansions)
        assert fp.headers_expanded["UNKNOWN_HEADER"] == "UNKNOWN_HEADER"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fingerprint.py::TestBuildHeaderExpansionPrompt -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement LLM expansion functions**

Add to `zone3/fbi/fingerprint.py`:

```python
from zone3.fbi.llm_utils import llm_call, parse_arrow_mapping


def build_header_expansion_prompt(headers: list[str],
                                   batch_size: int = 50) -> str:
    """Build a prompt asking the LLM to expand abbreviated column headers.

    Args:
        headers: List of raw header names to expand.
        batch_size: Max headers per prompt (for batching externally).

    Returns:
        Prompt string ready to send to LLM.
    """
    header_list = "\n".join(f"- {h}" for h in headers[:batch_size])
    return (
        "Below is a list of abbreviated column headers from a business database.\n"
        "For each one, expand the abbreviation into its full descriptive name.\n"
        "If the meaning is unclear, make your best guess based on the "
        "abbreviation pattern.\n\n"
        "Format your response as:\n"
        "ABBREVIATION → Full Name\n\n"
        f"Headers:\n{header_list}"
    )


def build_filename_parse_prompt(filenames: list[str]) -> str:
    """Build a prompt asking the LLM to extract semantic tokens from filenames.

    Returns:
        Prompt string ready to send to LLM.
    """
    # Strip extensions for cleaner input
    cleaned = [os.path.splitext(f)[0] for f in filenames]
    file_list = "\n".join(f"- {f}" for f in cleaned)
    return (
        "Below are filenames from a data directory.\n"
        "For each filename, extract the meaningful domain tokens.\n"
        "Ignore technical prefixes like 'synthetic_data_sample_' or "
        "suffixes like '_masked'.\n"
        "Ignore file extensions.\n\n"
        "Format: filename → [token1, token2, ...]\n\n"
        f"Filenames:\n{file_list}"
    )


def apply_expansions(fp: FileFingerprint,
                     expansions: dict[str, str]) -> None:
    """Apply header expansions to a fingerprint in-place.

    For any raw header not found in expansions, uses the raw header as-is.
    """
    fp.headers_expanded = {
        h: expansions.get(h, h)
        for h in fp.headers_raw
    }


def expand_all_headers(fingerprints: list[FileFingerprint],
                       model: str | None = None,
                       batch_size: int = 50) -> dict[str, str]:
    """Expand all unique headers across all fingerprints via LLM.

    Makes ceil(n_unique / batch_size) LLM calls.
    Returns the full expansion mapping.
    """
    # Collect unique headers
    all_headers: set[str] = set()
    for fp in fingerprints:
        all_headers.update(fp.headers_raw)

    unique = sorted(all_headers)
    full_mapping: dict[str, str] = {}

    for i in range(0, len(unique), batch_size):
        batch = unique[i : i + batch_size]
        prompt = build_header_expansion_prompt(batch, batch_size=len(batch))
        response = llm_call(prompt, model=model)
        mapping = parse_arrow_mapping(response)
        full_mapping.update(mapping)

    # Apply to all fingerprints
    for fp in fingerprints:
        apply_expansions(fp, full_mapping)

    return full_mapping


def parse_filename_tokens(fingerprints: list[FileFingerprint],
                          model: str | None = None) -> None:
    """Parse filename tokens for all fingerprints via LLM.

    Modifies fingerprints in-place, setting filename_tokens.
    """
    filenames = [fp.basename for fp in fingerprints]
    prompt = build_filename_parse_prompt(filenames)
    response = llm_call(prompt, model=model)

    # Parse response — expect "filename → [tok1, tok2, ...]"
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(.+?)\s*[→\->]+\s*\[(.+)\]$", line)
        if match:
            fname_part = match.group(1).strip().rstrip(".")
            tokens_str = match.group(2).strip()
            tokens = [t.strip().strip("'\"").lower()
                      for t in tokens_str.split(",")]
            # Find matching fingerprint
            for fp in fingerprints:
                base_no_ext = os.path.splitext(fp.basename)[0]
                if (fname_part in base_no_ext
                        or base_no_ext in fname_part
                        or fname_part == base_no_ext):
                    fp.filename_tokens = tokens
                    break
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_fingerprint.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add zone3/fbi/fingerprint.py tests/test_fingerprint.py
git commit -m "feat(fbi): Phase 1 LLM header expansion and filename parsing"
```

---

## Task 4: Phase 2 Iteration 1 — Prefix Trie Grouping

**Files:**
- Create: `zone3/fbi/class_discovery.py`
- Test: `tests/test_class_discovery.py`

- [ ] **Step 1: Write failing tests for prefix grouping**

```python
# tests/test_class_discovery.py
"""Tests for FBI Phase 2: class discovery."""

import pytest
from zone3.fbi.class_discovery import (
    find_prefix_groups,
    detect_sibling_patterns,
    CandidateClass,
)


class TestFindPrefixGroups:
    def test_endorsement_prefixes(self) -> None:
        headers = [
            "ENDORSEMENT_EARTHQUAKE_CODE",
            "ENDORSEMENT_EARTHQUAKE_NWP",
            "ENDORSEMENT_EARTHQUAKE_DED_CODE",
            "ENDORSEMENT_EARTHQUAKE_EXPOSURE",
            "ENDORSEMENT_JEWELRY_CODE",
            "ENDORSEMENT_JEWELRY_NWP",
            "ENDORSEMENT_JEWELRY_DED_CODE",
            "ENDORSEMENT_JEWELRY_EXPOSURE",
            "POLICY_NUMBER",
            "INSURED_NAME",
        ]
        groups = find_prefix_groups(headers, min_group_size=3)
        # Should find ENDORSEMENT_ as a prefix group
        prefixes = [g.prefix for g in groups]
        assert any("ENDORSEMENT" in p for p in prefixes)

    def test_no_groups_when_all_unique(self) -> None:
        headers = ["NAME", "AGE", "CITY"]
        groups = find_prefix_groups(headers, min_group_size=3)
        assert len(groups) == 0

    def test_cov_prefixes(self) -> None:
        headers = [
            "COVPROP_CODE1", "COVPROP_PREM1", "COVPROP_DED1", "COVPROP_AINS1",
            "COVLIAB_CODE2", "COVLIAB_PREM2", "COVLIAB_DED2", "COVLIAB_AINS2",
        ]
        groups = find_prefix_groups(headers, min_group_size=3)
        prefixes = [g.prefix for g in groups]
        assert any("COVPROP" in p for p in prefixes)
        assert any("COVLIAB" in p for p in prefixes)


class TestDetectSiblingPatterns:
    def test_endorsement_siblings(self) -> None:
        groups = [
            CandidateClass(
                prefix="ENDORSEMENT_EARTHQUAKE",
                headers=["ENDORSEMENT_EARTHQUAKE_CODE",
                         "ENDORSEMENT_EARTHQUAKE_NWP",
                         "ENDORSEMENT_EARTHQUAKE_DED_CODE",
                         "ENDORSEMENT_EARTHQUAKE_EXPOSURE"],
                suffixes=["CODE", "NWP", "DED_CODE", "EXPOSURE"],
                source_file="test.csv",
            ),
            CandidateClass(
                prefix="ENDORSEMENT_JEWELRY",
                headers=["ENDORSEMENT_JEWELRY_CODE",
                         "ENDORSEMENT_JEWELRY_NWP",
                         "ENDORSEMENT_JEWELRY_DED_CODE",
                         "ENDORSEMENT_JEWELRY_EXPOSURE"],
                suffixes=["CODE", "NWP", "DED_CODE", "EXPOSURE"],
                source_file="test.csv",
            ),
        ]
        siblings = detect_sibling_patterns(groups)
        # Both groups share suffix pattern → they are siblings
        assert len(siblings) >= 1
        sibling_group = siblings[0]
        assert len(sibling_group.children) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_class_discovery.py -v`

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement prefix trie grouping**

```python
# zone3/fbi/class_discovery.py
"""Phase 2: Multi-iteration class discovery from header structure.

Iteration 1: Prefix-based grouping (algorithmic)
Iteration 2: Semantic grouping of ungrouped headers (LLM)
Iteration 3: Cross-file class merging (algorithmic)
Iteration 4: Hierarchy assembly + LLM naming
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from zone3.fbi.fingerprint import FileFingerprint


@dataclass
class CandidateClass:
    """A candidate ontology class discovered from header analysis."""
    prefix: str
    headers: list[str] = field(default_factory=list)
    suffixes: list[str] = field(default_factory=list)
    source_file: str = ""
    name: str = ""  # LLM-assigned name (filled later)
    children: list[CandidateClass] = field(default_factory=list)
    parent: CandidateClass | None = field(default=None, repr=False)
    level: int = 0  # 0=root, 1=top, 2=sub, 3=leaf
    source_files: list[str] = field(default_factory=list)
    # Headers unique to this class (vs parent/siblings)
    unique_headers: list[str] = field(default_factory=list)
    # Headers shared with parent
    shared_headers: list[str] = field(default_factory=list)


@dataclass
class SiblingGroup:
    """A group of CandidateClasses sharing the same suffix pattern."""
    common_prefix: str
    suffix_pattern: list[str]
    children: list[CandidateClass]


def _extract_suffix(header: str, prefix: str) -> str:
    """Extract the suffix of a header after removing the prefix."""
    if header.startswith(prefix):
        suffix = header[len(prefix):]
        # Strip leading underscores or separators
        return suffix.lstrip("_").lstrip("-")
    return header


def find_prefix_groups(
    headers: list[str],
    min_group_size: int = 3,
    separator: str = "_",
) -> list[CandidateClass]:
    """Find groups of headers sharing a common prefix.

    Builds a prefix tree and identifies groups where ≥min_group_size
    headers share a prefix.

    Args:
        headers: List of header names (uppercase).
        min_group_size: Minimum headers to form a group.
        separator: Character separating prefix parts (default underscore).

    Returns:
        List of CandidateClass objects, one per discovered prefix group.
    """
    # Build prefix → headers mapping at each depth
    # Try progressively longer prefixes
    groups: list[CandidateClass] = []
    used: set[str] = set()

    # Sort headers so we process longer prefixes first
    sorted_headers = sorted(headers)

    # Find all possible prefixes
    prefix_map: dict[str, list[str]] = defaultdict(list)
    for h in sorted_headers:
        parts = h.split(separator)
        # Generate all prefixes of length 1..n-1
        for depth in range(1, len(parts)):
            prefix = separator.join(parts[:depth])
            prefix_map[prefix].append(h)

    # Find the most specific (longest) prefixes with enough members
    # Sort by prefix length descending so we pick the most specific first
    candidates = sorted(
        ((prefix, members) for prefix, members in prefix_map.items()
         if len(members) >= min_group_size),
        key=lambda x: len(x[0]),
        reverse=True,
    )

    for prefix, members in candidates:
        # Skip headers already claimed by a more specific prefix
        unclaimed = [m for m in members if m not in used]
        if len(unclaimed) < min_group_size:
            continue

        suffixes = [_extract_suffix(h, prefix + separator) for h in unclaimed]
        group = CandidateClass(
            prefix=prefix,
            headers=unclaimed,
            suffixes=suffixes,
        )
        groups.append(group)
        used.update(unclaimed)

    return groups


def detect_sibling_patterns(
    groups: list[CandidateClass],
    min_suffix_overlap: float = 0.8,
) -> list[SiblingGroup]:
    """Detect groups that share the same suffix pattern → siblings.

    If ENDORSEMENT_EARTHQUAKE has {CODE, NWP, DED_CODE, EXPOSURE}
    and ENDORSEMENT_JEWELRY has {CODE, NWP, DED_CODE, EXPOSURE},
    they are siblings under a common "ENDORSEMENT" parent.

    Args:
        groups: List of CandidateClass from find_prefix_groups.
        min_suffix_overlap: Minimum Jaccard similarity of suffix sets.

    Returns:
        List of SiblingGroup objects.
    """
    if not groups:
        return []

    # Group by suffix pattern similarity
    sibling_groups: list[SiblingGroup] = []
    assigned: set[int] = set()

    for i, g1 in enumerate(groups):
        if i in assigned:
            continue
        siblings = [g1]
        s1 = set(g1.suffixes)

        for j, g2 in enumerate(groups):
            if j <= i or j in assigned:
                continue
            s2 = set(g2.suffixes)
            if not s1 or not s2:
                continue
            jaccard = len(s1 & s2) / len(s1 | s2)
            if jaccard >= min_suffix_overlap:
                siblings.append(g2)
                assigned.add(j)

        if len(siblings) >= 2:
            assigned.add(i)
            # Find common prefix among sibling prefixes
            prefixes = [s.prefix for s in siblings]
            common = _longest_common_prefix(prefixes)
            sibling_groups.append(SiblingGroup(
                common_prefix=common.rstrip("_"),
                suffix_pattern=sorted(s1 & set(siblings[1].suffixes)),
                children=siblings,
            ))

    return sibling_groups


def _longest_common_prefix(strings: list[str]) -> str:
    """Find the longest common prefix of a list of strings."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def get_ungrouped_headers(
    all_headers: list[str],
    groups: list[CandidateClass],
) -> list[str]:
    """Return headers not assigned to any prefix group."""
    grouped: set[str] = set()
    for g in groups:
        grouped.update(g.headers)
    return [h for h in all_headers if h not in grouped]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_class_discovery.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add zone3/fbi/class_discovery.py tests/test_class_discovery.py
git commit -m "feat(fbi): Phase 2 iter 1 — prefix trie grouping with sibling detection"
```

---

## Task 5: Phase 2 Iterations 2-4 — Semantic Grouping, Cross-File Merging, Naming

**Files:**
- Modify: `zone3/fbi/class_discovery.py`
- Test: `tests/test_class_discovery.py` (add tests)

- [ ] **Step 1: Write tests for semantic grouping prompt builder**

Add to `tests/test_class_discovery.py`:

```python
from zone3.fbi.class_discovery import (
    build_semantic_grouping_prompt,
    build_naming_prompt,
    merge_cross_file_classes,
)


class TestBuildSemanticGroupingPrompt:
    def test_prompt_contains_headers(self) -> None:
        headers = ["Insured Name", "Location Zip", "Risk State",
                    "Policy Number", "Effective Date"]
        prompt = build_semantic_grouping_prompt(headers)
        assert "Insured Name" in prompt
        assert "concept" in prompt.lower() or "group" in prompt.lower()


class TestMergeCrossFileClasses:
    def test_claim_classes_merge(self) -> None:
        c1 = CandidateClass(
            prefix="CLAIM",
            headers=["CLAIM_STATUS", "CLAIM_NUMBER", "CAUSE_OF_LOSS"],
            source_file="geico_claims.csv",
            source_files=["geico_claims.csv"],
        )
        c2 = CandidateClass(
            prefix="CLAIM",
            headers=["CLAIM_STATUS", "CLAIM_NUMBER", "DEVICE_DAMAGE_TYPE"],
            source_file="tmobile_claims.csv",
            source_files=["tmobile_claims.csv"],
        )
        merged = merge_cross_file_classes([c1, c2], overlap_threshold=0.3)
        # Should produce 1 parent with 2 children
        assert len(merged) == 1
        parent = merged[0]
        assert len(parent.children) == 2
        assert "CLAIM_STATUS" in parent.shared_headers

    def test_unrelated_classes_stay_separate(self) -> None:
        c1 = CandidateClass(
            prefix="CLAIM",
            headers=["CLAIM_STATUS", "CLAIM_NUMBER"],
            source_file="f1.csv",
            source_files=["f1.csv"],
        )
        c2 = CandidateClass(
            prefix="SURVEY",
            headers=["SURVEY_ID", "NPS_SCORE"],
            source_file="f2.csv",
            source_files=["f2.csv"],
        )
        merged = merge_cross_file_classes([c1, c2], overlap_threshold=0.3)
        assert len(merged) == 2


class TestBuildNamingPrompt:
    def test_prompt_contains_evidence(self) -> None:
        parent = CandidateClass(
            prefix="CLAIM",
            shared_headers=["CLAIM_STATUS", "CLAIM_NUMBER"],
            children=[
                CandidateClass(
                    prefix="CLAIM_GEICO",
                    unique_headers=["CAUSE_OF_LOSS", "IS_CAT"],
                    source_files=["geico_claims.csv"],
                ),
            ],
        )
        prompt = build_naming_prompt(parent)
        assert "CLAIM_STATUS" in prompt
        assert "CAUSE_OF_LOSS" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_class_discovery.py::TestMergeCrossFileClasses -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement iterations 2-4**

Add to `zone3/fbi/class_discovery.py`:

```python
from zone3.fbi.llm_utils import llm_call, llm_call_json


def build_semantic_grouping_prompt(headers: list[str]) -> str:
    """Build prompt for LLM to group ungrouped headers by concept."""
    header_list = "\n".join(f"- {h}" for h in headers)
    return (
        "Below are column headers from a single data file that don't share "
        "obvious naming prefixes with each other.\n\n"
        "Group them by the concept they describe. Each group should represent "
        "one coherent business concept.\n\n"
        "Format your response as JSON:\n"
        '{"groups": [{"name": "ConceptName", "headers": ["Header1", "Header2"]}]}\n\n'
        f"Headers:\n{header_list}"
    )


def semantic_group_headers(
    headers: list[str],
    model: str | None = None,
) -> list[CandidateClass]:
    """Use LLM to group ungrouped headers into semantic classes.

    Returns list of CandidateClass objects, one per semantic group.
    """
    if not headers:
        return []

    prompt = build_semantic_grouping_prompt(headers)
    result = llm_call_json(prompt, model=model)

    groups: list[CandidateClass] = []
    raw_groups = result if isinstance(result, list) else result.get("groups", [])
    for g in raw_groups:
        name = g.get("name", "Unknown")
        group_headers = g.get("headers", [])
        if group_headers:
            groups.append(CandidateClass(
                prefix=name.upper().replace(" ", "_"),
                headers=group_headers,
                name=name,
            ))

    return groups


def merge_cross_file_classes(
    classes: list[CandidateClass],
    overlap_threshold: float = 0.3,
) -> list[CandidateClass]:
    """Merge candidate classes from different files that overlap.

    Classes with header overlap > threshold become parent-child:
    - Shared headers → parent class attributes
    - Unique headers → child (subclass) attributes

    Args:
        classes: All candidate classes from all files.
        overlap_threshold: Minimum Jaccard similarity to merge.

    Returns:
        List of merged CandidateClass objects (some with children).
    """
    if not classes:
        return []

    merged: list[CandidateClass] = []
    used: set[int] = set()

    for i, c1 in enumerate(classes):
        if i in used:
            continue

        # Find all classes that overlap with c1
        group = [c1]
        h1 = set(c1.headers)

        for j, c2 in enumerate(classes):
            if j <= i or j in used:
                continue
            h2 = set(c2.headers)
            if not h1 or not h2:
                continue

            # Check prefix similarity first
            prefix_match = (
                c1.prefix and c2.prefix
                and (c1.prefix.startswith(c2.prefix)
                     or c2.prefix.startswith(c1.prefix)
                     or c1.prefix == c2.prefix)
            )

            # Check header overlap
            jaccard = len(h1 & h2) / len(h1 | h2) if h1 | h2 else 0

            if prefix_match or jaccard >= overlap_threshold:
                group.append(c2)
                used.add(j)

        if len(group) == 1:
            # No merging needed — standalone class
            c1.source_files = [c1.source_file] if c1.source_file else c1.source_files
            merged.append(c1)
        else:
            # Create parent from shared headers, children from unique
            all_header_sets = [set(c.headers) for c in group]
            shared = set.intersection(*all_header_sets) if all_header_sets else set()

            parent = CandidateClass(
                prefix=c1.prefix,
                headers=sorted(shared),
                shared_headers=sorted(shared),
                source_files=[c.source_file for c in group if c.source_file],
                level=1,
            )

            for child in group:
                child_unique = sorted(set(child.headers) - shared)
                child_class = CandidateClass(
                    prefix=child.prefix + "_" + (child.source_file or "").split(".")[0],
                    headers=child.headers,
                    unique_headers=child_unique,
                    shared_headers=sorted(shared),
                    source_file=child.source_file,
                    source_files=[child.source_file] if child.source_file else child.source_files,
                    parent=parent,
                    level=2,
                )
                parent.children.append(child_class)

            used.add(i)
            merged.append(parent)

    return merged


def build_naming_prompt(parent_class: CandidateClass) -> str:
    """Build prompt for LLM to name a class and its children."""
    shared = ", ".join(parent_class.shared_headers[:15]) if parent_class.shared_headers else "(none)"

    lines = [
        "I have groups of column headers from business data files.",
        "Each group represents a distinct concept in the data.\n",
        "For each group, provide a short name (1-2 words) that describes "
        "what concept these columns represent.\n",
        f"Parent group (shared across files):\n  {{{shared}}}\n",
    ]

    for i, child in enumerate(parent_class.children):
        unique = ", ".join(child.unique_headers[:15])
        src = ", ".join(child.source_files[:3])
        lines.append(
            f"Child group {chr(65 + i)} (from files: [{src}]):\n"
            f"  adds: {{{unique}}}\n"
        )

    lines.append(
        "\nFor each group:\n"
        "1. Name the group (1-2 words)\n"
        "2. Is each child group a more specific type of the parent? (yes/no)\n"
        "\nFormat as JSON:\n"
        '{"parent_name": "...", "children": [{"name": "...", "is_subclass": true}]}'
    )

    return "\n".join(lines)


def name_classes(
    classes: list[CandidateClass],
    model: str | None = None,
) -> None:
    """Use LLM to name all classes and their children. Modifies in-place."""
    for cls in classes:
        if cls.children:
            prompt = build_naming_prompt(cls)
            result = llm_call_json(prompt, model=model)
            if isinstance(result, dict):
                cls.name = result.get("parent_name", cls.prefix)
                children_names = result.get("children", [])
                for i, child in enumerate(cls.children):
                    if i < len(children_names):
                        child.name = children_names[i].get("name", child.prefix)
        elif not cls.name:
            # Single class without children — name from headers
            header_sample = ", ".join(cls.headers[:10])
            prompt = (
                f"These column headers describe one concept:\n"
                f"  {{{header_sample}}}\n\n"
                f"What 1-2 word name describes this concept?\n"
                f"Answer with just the name."
            )
            cls.name = llm_call(prompt, model=model).strip().strip('"')
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_class_discovery.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add zone3/fbi/class_discovery.py tests/test_class_discovery.py
git commit -m "feat(fbi): Phase 2 iters 2-4 — semantic grouping, cross-file merge, naming"
```

---

## Task 6: Phase 3 — Relationship Discovery

**Files:**
- Create: `zone3/fbi/relationships.py`
- Test: `tests/test_relationships.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_relationships.py
"""Tests for FBI Phase 3: relationship discovery."""

import pytest
from zone3.fbi.relationships import find_bridge_columns, ClassRelationship
from zone3.fbi.class_discovery import CandidateClass


class TestFindBridgeColumns:
    def test_policy_number_bridges_classes(self) -> None:
        classes = [
            CandidateClass(
                prefix="POLICY",
                name="Policy",
                headers=["POLICY_NUMBER", "EFF_DATE", "GWP"],
                source_files=["policies.csv"],
            ),
            CandidateClass(
                prefix="CLAIM",
                name="Claim",
                headers=["CLAIM_NUMBER", "POLICY_NUMBER", "CLAIM_STATUS"],
                source_files=["claims.csv"],
            ),
        ]
        bridges = find_bridge_columns(classes)
        assert any(b.bridge_column == "POLICY_NUMBER" for b in bridges)

    def test_no_bridges_with_no_overlap(self) -> None:
        classes = [
            CandidateClass(
                prefix="A", name="A",
                headers=["COL_1", "COL_2"],
                source_files=["a.csv"],
            ),
            CandidateClass(
                prefix="B", name="B",
                headers=["COL_3", "COL_4"],
                source_files=["b.csv"],
            ),
        ]
        bridges = find_bridge_columns(classes)
        assert len(bridges) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_relationships.py -v`

Expected: FAIL

- [ ] **Step 3: Implement relationship discovery**

```python
# zone3/fbi/relationships.py
"""Phase 3: Discover inter-class relationships from bridge columns."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

from zone3.fbi.class_discovery import CandidateClass
from zone3.fbi.llm_utils import llm_call


@dataclass
class ClassRelationship:
    """A discovered relationship between two ontology classes."""
    source_class: str
    target_class: str
    relationship_name: str
    bridge_column: str
    confidence: float = 1.0


def find_bridge_columns(
    classes: list[CandidateClass],
) -> list[ClassRelationship]:
    """Find columns that appear in multiple classes → bridge relationships.

    A bridge column is a header that appears in the header lists of
    two or more different classes (including their children).

    Returns:
        List of ClassRelationship objects (without names yet).
    """
    # Build column → classes mapping
    col_to_classes: dict[str, list[str]] = defaultdict(list)

    def _collect(cls: CandidateClass) -> None:
        name = cls.name or cls.prefix
        for h in cls.headers:
            if name not in col_to_classes[h]:
                col_to_classes[h].append(name)
        for child in cls.children:
            _collect(child)

    for cls in classes:
        _collect(cls)

    # Find columns bridging 2+ classes
    bridges: list[ClassRelationship] = []
    seen_pairs: set[tuple[str, str]] = set()

    for col, class_names in col_to_classes.items():
        if len(class_names) < 2:
            continue
        # Create relationship for each pair
        for i, c1 in enumerate(class_names):
            for c2 in class_names[i + 1:]:
                pair = tuple(sorted([c1, c2]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                bridges.append(ClassRelationship(
                    source_class=c1,
                    target_class=c2,
                    relationship_name="",  # filled by LLM
                    bridge_column=col,
                ))

    return bridges


def build_relationship_naming_prompt(
    relationships: list[ClassRelationship],
) -> str:
    """Build prompt for LLM to name discovered relationships."""
    lines = [
        "I found columns that appear in data about multiple concepts, "
        "indicating a relationship between them.\n",
        "For each pair, suggest a short relationship name (1-3 words) "
        "that describes how the first concept relates to the second.\n",
        "Format as JSON:\n"
        '{"relationships": [{"source": "...", "target": "...", '
        '"bridge": "...", "name": "..."}]}\n',
    ]
    for r in relationships:
        lines.append(
            f"- Column '{r.bridge_column}' appears in both "
            f"'{r.source_class}' and '{r.target_class}'"
        )
    return "\n".join(lines)


def name_relationships(
    relationships: list[ClassRelationship],
    model: str | None = None,
) -> None:
    """Use LLM to name all discovered relationships. Modifies in-place."""
    if not relationships:
        return

    from zone3.fbi.llm_utils import llm_call_json

    prompt = build_relationship_naming_prompt(relationships)
    result = llm_call_json(prompt, model=model)

    named = result if isinstance(result, list) else result.get("relationships", [])
    # Match back to our relationship objects
    for named_r in named:
        src = named_r.get("source", "")
        tgt = named_r.get("target", "")
        name = named_r.get("name", "RELATES_TO")
        for r in relationships:
            if (r.source_class == src and r.target_class == tgt
                    and not r.relationship_name):
                r.relationship_name = name
                break

    # Default any unnamed
    for r in relationships:
        if not r.relationship_name:
            r.relationship_name = "RELATES_TO"
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_relationships.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add zone3/fbi/relationships.py tests/test_relationships.py
git commit -m "feat(fbi): Phase 3 — bridge column detection and relationship naming"
```

---

## Task 7: Phase 4 — Entity Assignment

**Files:**
- Create: `zone3/fbi/entity_assign.py`
- Test: `tests/test_entity_assign.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_entity_assign.py
"""Tests for FBI Phase 4: entity assignment."""

import pytest
from zone3.fbi.entity_assign import (
    build_file_to_class_map,
    assign_entity_to_class,
)
from zone3.fbi.class_discovery import CandidateClass


class TestBuildFileToClassMap:
    def test_maps_files_to_classes(self) -> None:
        classes = [
            CandidateClass(
                prefix="CLAIM", name="Claim",
                source_files=["claims.csv"],
                children=[
                    CandidateClass(
                        prefix="CLAIM_GEICO", name="PropertyClaim",
                        source_files=["geico_claims.csv"],
                    ),
                ],
            ),
        ]
        mapping = build_file_to_class_map(classes)
        assert mapping["claims.csv"] == "Claim"
        assert mapping["geico_claims.csv"] == "PropertyClaim"


class TestAssignEntityToClass:
    def test_single_source_assignment(self) -> None:
        file_map = {"claims.csv": "Claim", "policies.csv": "Policy"}
        entity = {
            "id": "CLM-001",
            "source_files": {"claims.csv"},
        }
        cls = assign_entity_to_class(entity, file_map)
        assert cls == "Claim"

    def test_multi_source_picks_most_common(self) -> None:
        file_map = {
            "claims.csv": "Claim",
            "surveys.csv": "Survey",
            "surveys2.csv": "Survey",
        }
        entity = {
            "id": "ENT-001",
            "source_files": {"claims.csv", "surveys.csv", "surveys2.csv"},
        }
        cls = assign_entity_to_class(entity, file_map)
        assert cls == "Survey"  # appears in 2 Survey files vs 1 Claim
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_entity_assign.py -v`

Expected: FAIL

- [ ] **Step 3: Implement entity assignment**

```python
# zone3/fbi/entity_assign.py
"""Phase 4: Assign Zone 2 entities to discovered classes and write to Neo4j."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from zone3.fbi.class_discovery import CandidateClass
from zone3.fbi.relationships import ClassRelationship


def build_file_to_class_map(
    classes: list[CandidateClass],
) -> dict[str, str]:
    """Build mapping from source filename to most specific class name.

    Children (subclasses) take priority over parents.
    """
    mapping: dict[str, str] = {}

    def _map(cls: CandidateClass) -> None:
        name = cls.name or cls.prefix
        for f in cls.source_files:
            # Normalize to basename
            basename = os.path.basename(f)
            mapping[basename] = name
        # Children override parent mapping (more specific)
        for child in cls.children:
            _map(child)

    for cls in classes:
        _map(cls)

    return mapping


def assign_entity_to_class(
    entity: dict,
    file_to_class: dict[str, str],
) -> str:
    """Assign a single entity to a class based on its source files.

    If entity appears in multiple classes, picks the most common one.

    Args:
        entity: Dict with 'id' and 'source_files' (set of filenames).
        file_to_class: Mapping from filename to class name.

    Returns:
        Class name string.
    """
    source_files = entity.get("source_files", set())
    if not source_files:
        return "Unclassified"

    class_counts: Counter = Counter()
    for f in source_files:
        basename = os.path.basename(f)
        cls = file_to_class.get(basename)
        if cls:
            class_counts[cls] += 1

    if not class_counts:
        return "Unclassified"

    return class_counts.most_common(1)[0][0]


def assign_all_entities(
    entities: list[dict],
    classes: list[CandidateClass],
) -> dict[str, str]:
    """Assign all entities to classes.

    Returns:
        Dict mapping entity_id → class_name.
    """
    file_map = build_file_to_class_map(classes)
    assignments: dict[str, str] = {}

    for entity in entities:
        eid = entity.get("id", "")
        cls = assign_entity_to_class(entity, file_map)
        assignments[eid] = cls

    return assignments


def write_ontology_to_neo4j(
    classes: list[CandidateClass],
    relationships: list[ClassRelationship],
    entity_assignments: dict[str, str],
    filename_tokens: dict[str, list[str]],
    driver=None,
) -> dict:
    """Write the complete ontology to Neo4j.

    Creates OntologyClass nodes, SUBCLASS_OF edges, RELATES_TO edges,
    and labels entities with their assigned class.

    Args:
        classes: Discovered class hierarchy.
        relationships: Inter-class relationships.
        entity_assignments: Mapping entity_id → class_name.
        filename_tokens: Mapping filename → [semantic tokens] for LOB tagging.
        driver: Neo4j driver instance. If None, creates one from config.

    Returns:
        Summary dict with counts.
    """
    if driver is None:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
        )

    stats = {"classes_created": 0, "subclass_edges": 0,
             "relationship_edges": 0, "entities_labeled": 0}

    with driver.session(database=config.NEO4J_DATABASE) as session:
        # 1. Create OntologyClass nodes
        def _create_class(cls: CandidateClass, parent_name: str | None = None):
            name = cls.name or cls.prefix
            session.run(
                """
                MERGE (c:OntologyClass {name: $name})
                SET c.level = $level,
                    c.source_files = $source_files,
                    c.header_count = $header_count
                """,
                name=name,
                level=cls.level,
                source_files=cls.source_files,
                header_count=len(cls.headers),
            )
            stats["classes_created"] += 1

            if parent_name:
                session.run(
                    """
                    MATCH (child:OntologyClass {name: $child_name})
                    MATCH (parent:OntologyClass {name: $parent_name})
                    MERGE (child)-[:SUBCLASS_OF]->(parent)
                    """,
                    child_name=name,
                    parent_name=parent_name,
                )
                stats["subclass_edges"] += 1

            for child in cls.children:
                _create_class(child, name)

        for cls in classes:
            _create_class(cls)

        # 2. Create inter-class relationship edges
        for rel in relationships:
            if rel.relationship_name:
                session.run(
                    """
                    MATCH (a:OntologyClass {name: $source})
                    MATCH (b:OntologyClass {name: $target})
                    MERGE (a)-[r:RELATES_TO {name: $rel_name}]->(b)
                    SET r.bridge_column = $bridge
                    """,
                    source=rel.source_class,
                    target=rel.target_class,
                    rel_name=rel.relationship_name,
                    bridge=rel.bridge_column,
                )
                stats["relationship_edges"] += 1

        # 3. Label entities
        for entity_id, class_name in entity_assignments.items():
            # Determine LOB tag from entity's source file
            session.run(
                """
                MATCH (e:Entity {id: $eid})
                SET e.ontology_class = $cls
                """,
                eid=entity_id,
                cls=class_name,
            )
            stats["entities_labeled"] += 1

    return stats
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_entity_assign.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add zone3/fbi/entity_assign.py tests/test_entity_assign.py
git commit -m "feat(fbi): Phase 4 — entity assignment and Neo4j materialization"
```

---

## Task 8: Main Pipeline Orchestrator

**Files:**
- Create: `zone3/fbi/pipeline.py`

- [ ] **Step 1: Implement the main pipeline**

```python
# zone3/fbi/pipeline.py
"""FBI Pipeline: Fingerprint-Based Ontology Induction.

Orchestrates all 4 phases:
  Phase 1: Header Intelligence (extract + expand + parse)
  Phase 2: Multi-Iteration Class Discovery
  Phase 3: Relationship Discovery
  Phase 4: Entity Assignment + Neo4j Materialization

Usage:
    python -m zone3.fbi.pipeline --data-dir data/Emory_Spring2026 [--model qwen2.5:72b]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from zone3.fbi.fingerprint import (
    extract_fingerprints,
    expand_all_headers,
    parse_filename_tokens,
)
from zone3.fbi.class_discovery import (
    find_prefix_groups,
    detect_sibling_patterns,
    get_ungrouped_headers,
    semantic_group_headers,
    merge_cross_file_classes,
    name_classes,
    CandidateClass,
)
from zone3.fbi.relationships import (
    find_bridge_columns,
    name_relationships,
)
from zone3.fbi.entity_assign import (
    assign_all_entities,
    write_ontology_to_neo4j,
)


def run_phase1(data_dir: str, model: str | None = None) -> list:
    """Phase 1: Extract and expand headers from all source files."""
    print("\n=== PHASE 1: Header Intelligence ===")

    # Step 1a: Extract raw headers
    print("  Step 1a: Extracting raw headers...")
    fingerprints = extract_fingerprints(data_dir)
    print(f"    Found {len(fingerprints)} files:")
    for fp in fingerprints:
        if fp.file_type == "csv":
            print(f"      {fp.basename}: {len(fp.headers_raw)} columns, "
                  f"{fp.record_count} rows")
        else:
            print(f"      {fp.basename}: {len(fp.sections)} sections, "
                  f"{len(fp.defined_terms)} defined terms")

    # Step 1b: Expand cryptic headers via LLM
    print("  Step 1b: Expanding headers via LLM...")
    expansions = expand_all_headers(fingerprints, model=model)
    print(f"    Expanded {len(expansions)} unique headers")

    # Step 1c: Parse filename tokens via LLM
    print("  Step 1c: Parsing filename tokens via LLM...")
    parse_filename_tokens(fingerprints, model=model)
    for fp in fingerprints:
        print(f"    {fp.basename} → {fp.filename_tokens}")

    return fingerprints


def run_phase2(fingerprints: list, model: str | None = None) -> list:
    """Phase 2: Multi-iteration class discovery."""
    print("\n=== PHASE 2: Class Discovery ===")

    all_classes: list[CandidateClass] = []

    # Iteration 1: Prefix-based grouping (per file)
    print("  Iteration 1: Prefix-based grouping...")
    for fp in fingerprints:
        if fp.file_type == "csv" and fp.headers_raw:
            groups = find_prefix_groups(fp.headers_raw)
            for g in groups:
                g.source_file = fp.basename
                g.source_files = [fp.basename]
            all_classes.extend(groups)
            print(f"    {fp.basename}: {len(groups)} prefix groups")

            # Detect sibling patterns
            siblings = detect_sibling_patterns(groups)
            for sg in siblings:
                print(f"      Siblings under '{sg.common_prefix}': "
                      f"{len(sg.children)} types, "
                      f"shared suffixes: {sg.suffix_pattern}")

    # Iteration 2: Semantic grouping of ungrouped headers
    print("  Iteration 2: Semantic grouping of ungrouped headers...")
    for fp in fingerprints:
        if fp.file_type == "csv" and fp.headers_raw:
            ungrouped = get_ungrouped_headers(fp.headers_raw, all_classes)
            if ungrouped:
                # Use expanded names for better LLM understanding
                expanded_ungrouped = [
                    fp.headers_expanded.get(h, h) for h in ungrouped
                ]
                semantic_groups = semantic_group_headers(
                    expanded_ungrouped, model=model
                )
                for g in semantic_groups:
                    g.source_file = fp.basename
                    g.source_files = [fp.basename]
                all_classes.extend(semantic_groups)
                print(f"    {fp.basename}: {len(semantic_groups)} semantic groups "
                      f"from {len(ungrouped)} ungrouped headers")

    # Handle PDF/TXT files
    for fp in fingerprints:
        if fp.file_type in ("pdf", "txt") and fp.sections:
            # Each major section becomes a candidate class
            pdf_class = CandidateClass(
                prefix=fp.basename.split(".")[0].upper(),
                headers=fp.sections,
                source_file=fp.basename,
                source_files=[fp.basename],
            )
            all_classes.append(pdf_class)
            print(f"    {fp.basename}: 1 document class with "
                  f"{len(fp.sections)} sections")

    # Iteration 3: Cross-file merging
    print("  Iteration 3: Cross-file class merging...")
    merged = merge_cross_file_classes(all_classes)
    print(f"    {len(all_classes)} candidate classes → {len(merged)} after merging")
    for cls in merged:
        if cls.children:
            print(f"      '{cls.prefix}': {len(cls.children)} subclasses")

    # Iteration 4: LLM naming
    print("  Iteration 4: Naming classes via LLM...")
    name_classes(merged, model=model)
    for cls in merged:
        print(f"    {cls.name} (level {cls.level})")
        for child in cls.children:
            print(f"      └── {child.name}")

    return merged


def run_phase3(classes: list, model: str | None = None) -> list:
    """Phase 3: Relationship discovery."""
    print("\n=== PHASE 3: Relationship Discovery ===")

    bridges = find_bridge_columns(classes)
    print(f"  Found {len(bridges)} bridge columns")

    if bridges:
        name_relationships(bridges, model=model)
        for b in bridges:
            print(f"    {b.source_class} --{b.relationship_name}--> "
                  f"{b.target_class} (via {b.bridge_column})")

    return bridges


def run_phase4(
    classes: list,
    relationships: list,
    fingerprints: list,
    results_dir: str | None = None,
) -> dict:
    """Phase 4: Entity assignment + Neo4j materialization."""
    print("\n=== PHASE 4: Entity Assignment ===")

    # Load entities from Zone 2 results
    rdir = results_dir or config.RESULTS_DIR
    summary_path = os.path.join(rdir, "zone2_run_summary.json")

    if not os.path.exists(summary_path):
        print(f"  WARNING: Zone 2 results not found at {summary_path}")
        print("  Skipping entity assignment. Run Zone 2 first.")
        return {"entities_labeled": 0}

    with open(summary_path) as f:
        zone2_data = json.load(f)

    triples = zone2_data.get("triples", [])
    print(f"  Loaded {len(triples)} triples from Zone 2")

    # Build entity list with source files
    entity_sources: dict[str, set[str]] = {}
    for t in triples:
        subj = t.get("subject", "")
        source = t.get("source", "")
        if subj:
            entity_sources.setdefault(subj, set()).add(source)
        obj = t.get("object", "")
        if obj:
            entity_sources.setdefault(obj, set()).add(source)

    entities = [
        {"id": eid, "source_files": sources}
        for eid, sources in entity_sources.items()
    ]
    print(f"  Found {len(entities)} unique entities")

    # Assign entities to classes
    assignments = assign_all_entities(entities, classes)

    # Count assignments per class
    from collections import Counter
    counts = Counter(assignments.values())
    print("  Entity assignments:")
    for cls, count in counts.most_common():
        print(f"    {cls}: {count} entities")

    # Build filename tokens map for LOB tagging
    filename_tokens = {
        fp.basename: fp.filename_tokens
        for fp in fingerprints
    }

    # Write to Neo4j
    print("  Writing ontology to Neo4j...")
    stats = write_ontology_to_neo4j(
        classes, relationships, assignments, filename_tokens
    )
    print(f"  Done: {stats}")

    return stats


def save_results(
    fingerprints: list,
    classes: list,
    relationships: list,
    output_dir: str,
) -> None:
    """Save intermediate results as JSON for debugging and evaluation."""
    os.makedirs(output_dir, exist_ok=True)

    # Save fingerprints
    fp_data = []
    for fp in fingerprints:
        fp_data.append({
            "file": fp.basename,
            "file_type": fp.file_type,
            "filename_tokens": fp.filename_tokens,
            "headers_raw": fp.headers_raw,
            "headers_expanded": fp.headers_expanded,
            "sections": fp.sections,
            "defined_terms": fp.defined_terms,
            "record_count": fp.record_count,
        })
    with open(os.path.join(output_dir, "fbi_fingerprints.json"), "w") as f:
        json.dump(fp_data, f, indent=2)

    # Save class hierarchy
    def _class_to_dict(cls: CandidateClass) -> dict:
        return {
            "name": cls.name,
            "prefix": cls.prefix,
            "level": cls.level,
            "headers": cls.headers,
            "shared_headers": cls.shared_headers,
            "unique_headers": cls.unique_headers,
            "source_files": cls.source_files,
            "children": [_class_to_dict(c) for c in cls.children],
        }

    class_data = [_class_to_dict(c) for c in classes]
    with open(os.path.join(output_dir, "fbi_classes.json"), "w") as f:
        json.dump(class_data, f, indent=2)

    # Save relationships
    rel_data = [
        {
            "source": r.source_class,
            "target": r.target_class,
            "name": r.relationship_name,
            "bridge": r.bridge_column,
        }
        for r in relationships
    ]
    with open(os.path.join(output_dir, "fbi_relationships.json"), "w") as f:
        json.dump(rel_data, f, indent=2)

    print(f"\n  Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="FBI: Fingerprint-Based Ontology Induction"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to the data directory containing CSV/PDF/TXT files"
    )
    parser.add_argument(
        "--model", default=None,
        help="Ollama model name (default: config.OLLAMA_MODEL)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save results (default: config.RESULTS_DIR)"
    )
    parser.add_argument(
        "--skip-neo4j", action="store_true",
        help="Skip Phase 4 (Neo4j materialization)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir or config.RESULTS_DIR

    start = time.time()

    # Phase 1
    fingerprints = run_phase1(args.data_dir, model=args.model)

    # Phase 2
    classes = run_phase2(fingerprints, model=args.model)

    # Phase 3
    relationships = run_phase3(classes, model=args.model)

    # Save intermediate results
    save_results(fingerprints, classes, relationships, output_dir)

    # Phase 4
    if not args.skip_neo4j:
        stats = run_phase4(classes, relationships, fingerprints,
                           results_dir=output_dir)

    elapsed = time.time() - start
    print(f"\n=== FBI Pipeline Complete ({elapsed:.1f}s) ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add zone3/fbi/pipeline.py
git commit -m "feat(fbi): main pipeline orchestrating all 4 phases"
```

---

## Task 9: Integration Test — Run on Real Data

- [ ] **Step 1: Run Phase 1-3 on real data (no Neo4j needed)**

```bash
cd /Users/sam/Documents/School/Emory/CS584_AI_Capstone/.claude/worktrees/suspicious-sanderson
python -m zone3.fbi.pipeline \
  --data-dir data/Emory_Spring2026 \
  --model qwen2.5:72b \
  --skip-neo4j \
  --output-dir data/results/emory_fbi
```

- [ ] **Step 2: Inspect results**

```bash
cat data/results/emory_fbi/fbi_classes.json | python -m json.tool | head -100
cat data/results/emory_fbi/fbi_relationships.json | python -m json.tool
```

Verify:
- Classes discovered include Policy, Claim, Survey (or similar)
- Endorsement subtypes are discovered
- Coverage subtypes are discovered
- Hierarchy has 3+ levels
- Relationships include bridge columns like POLICY_NUMBER

- [ ] **Step 3: Fix any issues discovered during integration**

Debug and fix based on actual LLM responses. Common issues:
- LLM response format doesn't match expected parsing
- Prefix groups too granular or too coarse (adjust min_group_size)
- Cross-file merging threshold needs tuning

- [ ] **Step 4: Commit fixes**

```bash
git add -A
git commit -m "fix(fbi): integration test fixes from real data run"
```

---

## Task 10: Run All Tests + Final Commit

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/test_llm_utils.py tests/test_fingerprint.py \
  tests/test_class_discovery.py tests/test_relationships.py \
  tests/test_entity_assign.py -v
```

Expected: All tests PASS.

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "feat(fbi): complete Fingerprint-Based Ontology Induction pipeline"
```

---

Plan complete and saved to `docs/superpowers/plans/2026-04-16-fbi-implementation.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?