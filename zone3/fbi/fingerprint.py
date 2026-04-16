"""Phase 1: File fingerprinting — header extraction, LLM expansion, filename parsing."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from zone3.fbi.llm_utils import llm_call, parse_arrow_mapping

# Audit column patterns — columns matching these appear in every file and carry no domain signal
_AUDIT_PATTERNS: tuple[str, ...] = (
    "BI_CREATED_DT",
    "BI_CREATED_BY",
    "BI_MODIFIED_DT",
    "BI_MODIFIED_BY",
    "BI_CREATED_DATE",
    "BI_MODIFIED_DATE",
    "ETL_",
    "DW_",
    "SRC_SYS_",
    "LOAD_DT",
    "UPDATE_DT",
)


@dataclass
class FileFingerprint:
    """Fingerprint for a single data file."""

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
        """Return just the filename without directory path."""
        return os.path.basename(self.file_path)


# ---------------------------------------------------------------------------
# Algorithmic extraction (no LLM)
# ---------------------------------------------------------------------------


def extract_csv_headers(file_path: str) -> list[str]:
    """Read first row of a CSV, strip whitespace, uppercase.

    Returns an empty list for empty files or files with no columns.
    """
    try:
        with open(file_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
    except (OSError, StopIteration):
        return []

    if first_row is None:
        return []

    return [col.strip().upper() for col in first_row if col.strip()]


def extract_pdf_sections(
    file_path: str,
) -> tuple[list[str], list[str]]:
    """Extract section headings and defined terms from a PDF.

    Section headings: lines that are entirely uppercase (>= 3 chars).
    Defined terms: phrases preceding "means" (e.g. '"Flood" means ...').

    Returns (sections, defined_terms) — both deduplicated, order-preserved.
    """
    try:
        import pdfplumber
    except ImportError:
        return [], []

    sections: list[str] = []
    defined_terms: list[str] = []
    seen_sections: set[str] = set()
    seen_terms: set[str] = set()

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for line in text.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue

                    # Section heading: all-uppercase, >= 3 chars, mostly letters
                    alpha_chars = [c for c in stripped if c.isalpha()]
                    if (
                        len(stripped) >= 3
                        and alpha_chars
                        and all(c.isupper() for c in alpha_chars)
                    ):
                        if stripped not in seen_sections:
                            seen_sections.add(stripped)
                            sections.append(stripped)

                    # Defined terms: "Term" means ... or 'Term' means ...
                    term_matches = re.findall(
                        r"""["\u201c]([^"\u201d]+)["\u201d]\s+means""",
                        stripped,
                        re.IGNORECASE,
                    )
                    for term in term_matches:
                        term_clean = term.strip()
                        if term_clean and term_clean not in seen_terms:
                            seen_terms.add(term_clean)
                            defined_terms.append(term_clean)
    except Exception:
        return [], []

    return sections, defined_terms


def extract_txt_sections(
    file_path: str,
) -> tuple[list[str], list[str]]:
    """Extract section headings and defined terms from a TXT file.

    Same heuristics as extract_pdf_sections but reads plain text.
    """
    sections: list[str] = []
    defined_terms: list[str] = []
    seen_sections: set[str] = set()
    seen_terms: set[str] = set()

    try:
        with open(file_path, encoding="utf-8-sig") as f:
            text = f.read()
    except OSError:
        return [], []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        alpha_chars = [c for c in stripped if c.isalpha()]
        if (
            len(stripped) >= 3
            and alpha_chars
            and all(c.isupper() for c in alpha_chars)
        ):
            if stripped not in seen_sections:
                seen_sections.add(stripped)
                sections.append(stripped)

        term_matches = re.findall(
            r"""["\u201c]([^"\u201d]+)["\u201d]\s+means""",
            stripped,
            re.IGNORECASE,
        )
        for term in term_matches:
            term_clean = term.strip()
            if term_clean and term_clean not in seen_terms:
                seen_terms.add(term_clean)
                defined_terms.append(term_clean)

    return sections, defined_terms


def strip_audit_columns(
    headers_by_file: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Remove audit/ETL columns that appear in ALL files.

    A column is considered an audit column if it matches any prefix in
    ``_AUDIT_PATTERNS`` AND appears in every file in *headers_by_file*.
    """
    if not headers_by_file:
        return {}

    all_files = list(headers_by_file.values())

    def _is_audit(col: str) -> bool:
        upper = col.upper()
        return any(upper.startswith(pat) for pat in _AUDIT_PATTERNS)

    # Candidate audit columns: audit-patterned columns present in ALL files
    sets = [set(h) for h in all_files]
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    audit_to_remove = {c for c in common if _is_audit(c)}

    return {
        fpath: [h for h in headers if h not in audit_to_remove]
        for fpath, headers in headers_by_file.items()
    }


def count_csv_rows(file_path: str) -> int:
    """Count data rows in a CSV file, excluding the header row."""
    try:
        with open(file_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            return sum(1 for _ in reader)
    except OSError:
        return 0


def extract_fingerprints(data_dir: str) -> list[FileFingerprint]:
    """Scan *data_dir* recursively for CSV/PDF/TXT files, returning one fingerprint per file.

    This function does NOT call the LLM.  Headers, sections, and record
    counts are populated algorithmically.
    """
    fingerprints: list[FileFingerprint] = []

    for root, _dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            ext = Path(fname).suffix.lower().lstrip(".")

            if ext == "csv":
                headers = extract_csv_headers(fpath)
                fp = FileFingerprint(
                    file_path=fpath,
                    file_type="csv",
                    headers_raw=headers,
                    record_count=count_csv_rows(fpath),
                )
                fingerprints.append(fp)

            elif ext == "pdf":
                secs, terms = extract_pdf_sections(fpath)
                fp = FileFingerprint(
                    file_path=fpath,
                    file_type="pdf",
                    sections=secs,
                    defined_terms=terms,
                )
                fingerprints.append(fp)

            elif ext == "txt":
                secs, terms = extract_txt_sections(fpath)
                fp = FileFingerprint(
                    file_path=fpath,
                    file_type="txt",
                    sections=secs,
                    defined_terms=terms,
                )
                fingerprints.append(fp)

    return fingerprints


# ---------------------------------------------------------------------------
# LLM-assisted functions
# ---------------------------------------------------------------------------


def build_header_expansion_prompt(
    headers: list[str],
    batch_size: int = 50,
) -> str:
    """Build a prompt asking the LLM to expand abbreviated column headers.

    If *headers* exceeds *batch_size*, only the first *batch_size* are
    included (caller is responsible for batching).

    Expected LLM output format: ``ABBREVIATION -> Full Name`` per line.
    """
    batch = headers[:batch_size]
    header_list = "\n".join(f"- {h}" for h in batch)

    return (
        "You are a data-dictionary expert for business datasets.\n"
        "Expand each abbreviated column header into its full human-readable name.\n"
        "Output EXACTLY one line per header in the format:\n"
        "ABBREVIATION -> Full Name\n\n"
        "Do NOT add explanations, numbering, or extra text.\n\n"
        f"Headers:\n{header_list}"
    )


def build_filename_parse_prompt(filenames: list[str]) -> str:
    """Build a prompt asking the LLM to extract semantic tokens from filenames.

    Expected LLM output format (JSON):
    ``{"filename1": ["token1", "token2"], ...}``
    """
    file_list = "\n".join(f"- {f}" for f in filenames)

    return (
        "You are a data engineer analyzing data files.\n"
        "For each filename below, extract semantic tokens that describe\n"
        "the file's domain content. Remove file extensions, split on\n"
        "underscores/hyphens/camelCase, and normalize to lowercase.\n"
        "Ignore generic tokens like 'sample', 'data', 'raw', 'v1', 'v2'.\n\n"
        "Return a JSON object mapping each filename (with extension) to a list of tokens.\n"
        "Use the EXACT filename (including extension) as the JSON key.\n\n"
        f"Filenames:\n{file_list}"
    )


def apply_expansions(
    fp: FileFingerprint,
    expansions: dict[str, str],
) -> None:
    """Apply header expansion mapping to a fingerprint in-place.

    Missing expansions fall back to the raw header itself.
    """
    fp.headers_expanded = {
        raw: expansions.get(raw, raw) for raw in fp.headers_raw
    }


def expand_all_headers(
    fingerprints: list[FileFingerprint],
    model: str | None = None,
    batch_size: int = 50,
) -> dict[str, str]:
    """Collect all unique headers across fingerprints, expand via LLM, apply.

    Returns the combined expansion mapping.
    """
    all_headers: list[str] = []
    seen: set[str] = set()
    for fp in fingerprints:
        for h in fp.headers_raw:
            if h not in seen:
                seen.add(h)
                all_headers.append(h)

    if not all_headers:
        return {}

    combined: dict[str, str] = {}

    # Process in batches
    for start in range(0, len(all_headers), batch_size):
        batch = all_headers[start : start + batch_size]
        prompt = build_header_expansion_prompt(batch, batch_size=batch_size)
        raw_response = llm_call(prompt, model=model)
        mapping = parse_arrow_mapping(raw_response)
        combined.update(mapping)

    # Apply to every fingerprint
    for fp in fingerprints:
        apply_expansions(fp, combined)

    return combined


def parse_filename_tokens(
    fingerprints: list[FileFingerprint],
    model: str | None = None,
) -> None:
    """Call LLM to parse semantic tokens from filenames, set on each fingerprint."""
    import json

    filenames = [fp.basename for fp in fingerprints]
    if not filenames:
        return

    prompt = build_filename_parse_prompt(filenames)
    raw_response = llm_call(prompt, model=model, json_mode=True)

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from response
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    if not isinstance(parsed, dict):
        parsed = {}

    for fp in fingerprints:
        # Try exact match first
        tokens = parsed.get(fp.basename, None)
        if tokens is None:
            # Try without extension
            base_no_ext = os.path.splitext(fp.basename)[0]
            tokens = parsed.get(base_no_ext, None)
        if tokens is None:
            # Try fuzzy: find any key that is a substring of basename or vice versa
            base_no_ext = os.path.splitext(fp.basename)[0]
            for key, val in parsed.items():
                if (key in fp.basename or fp.basename in key
                        or base_no_ext in key or key in base_no_ext):
                    tokens = val
                    break
        if isinstance(tokens, list):
            fp.filename_tokens = [str(t).lower().strip() for t in tokens if str(t).strip()]
        else:
            fp.filename_tokens = []
