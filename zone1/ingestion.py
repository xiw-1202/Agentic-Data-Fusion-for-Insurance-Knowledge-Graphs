"""Zone 1: Multimodal Ingestion — Novel Pipeline
=============================================
Hybrid chunking strategy (per project plan §3.2):
  1. Title-based splitting at document section boundaries
  2. Sub-chunking: split oversized sections by paragraph (MAX_CHUNK_TOKENS=1200)
  3. Semantic merging: merge adjacent chunks when cosine similarity > τ=0.85
     (with token ceiling — will NOT merge if combined size exceeds MAX_CHUNK_TOKENS)
  4. Table extraction: pdfplumber parallel pass for structured table data
  5. Metadata enrichment: source, section hierarchy, temporal markers, chunk_type

Contrast with baseline: fixed 512-token sliding window with no semantic awareness.

Supports:
  - PDF: SFIP policy documents (section-aware + table extraction)
  - CSV: OpenFEMA policies + claims (token-capped dynamic batching with field grouping)
  - Generic CSV: any standard CSV file (auto-grouped fields, domain-agnostic)
"""

from __future__ import annotations

import json
import re
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

# Allow imports from project root (config.py lives there)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("  [warn] pdfplumber not installed — table extraction disabled. "
          "Run: pip install pdfplumber>=0.10")

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEMANTIC_MERGE_THRESHOLD = 0.85   # τ from plan §3.2
EMBED_MODEL = "all-MiniLM-L6-v2"  # matches plan stack
MAX_CHUNK_TOKENS = 1200            # hard ceiling; sections exceeding this are sub-chunked

# SFIP section header patterns (Roman numerals + lettered sub-sections).
# Major sections use ALL-CAPS titles (e.g. "IV. PROPERTY NOT INSURED").
# Sub-sections use Mixed-Case titles (e.g. "A. Coverage Under This Policy").
# Single "I." is ambiguous — only treat as major when followed by ALL-CAPS.
ROMAN_SECTION = re.compile(
    r'^(I{1,3}|IV|V?I{0,3}|IX|X{1,2})\.\s+[A-Z]{2,}', re.MULTILINE
)
LETTER_SUBSECTION = re.compile(
    r'^([A-Z])\.\s+[A-Z][a-z]'
)
# Page footer/header noise to strip
PAGE_HEADER = re.compile(
    r'NFIP GENERAL PROPERTY FORM SFIP\s+P\s*AG\s*E\s*\d+\s*OF\s*\d+',
    re.IGNORECASE
)
# Date-like patterns for temporal marker extraction
DATE_PATTERN = re.compile(
    r'\b(\d{4}|\w+ \d{4}|October \d{4}|January \d{4})\b'
)

# ---------------------------------------------------------------------------
# CSV field grouping configuration
# ---------------------------------------------------------------------------

# Semantic field groups for human-readable CSV chunking.
# Keys map to "policies" or "claims" record_type values.
CSV_FIELD_GROUPS: dict[str, dict[str, list[str]]] = {
    "policies": {
        "Policy":   ["policyEffectiveDate", "policyTerminationDate", "policyCost",
                     "waitingPeriodType", "cancellationDateOfFloodPolicy", "iccPremium"],
        "Coverage": ["totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage",
                     "buildingDeductibleCode", "contentsDeductibleCode", "buildingReplacementCost"],
        "Property": ["occupancyType", "construction", "numberOfFloorsInInsuredBuilding",
                     "elevatedBuildingIndicator", "originalConstructionDate", "obstructionType"],
        "Location": ["ratedFloodZone", "floodZoneCurrent", "propertyState",
                     "reportedCity", "reportedZipCode", "nfipCommunityName"],
    },
    "claims": {
        "Loss":     ["dateOfLoss", "yearOfLoss", "causeOfDamage", "floodEvent", "waterDepth"],
        "Building": ["totalBuildingInsuranceCoverage", "buildingDamageAmount",
                     "netBuildingPaymentAmount", "amountPaidOnBuildingClaim", "buildingPropertyValue"],
        "Contents": ["totalContentsInsuranceCoverage", "contentsDamageAmount",
                     "netContentsPaymentAmount", "amountPaidOnContentsClaim"],
        "ICC":      ["iccCoverage", "amountPaidOnIncreasedCostOfComplianceClaim", "netIccPaymentAmount"],
        "Property": ["ratedFloodZone", "floodZoneCurrent", "occupancyType",
                     "numberOfFloorsInTheInsuredBuilding", "buildingReplacementCost"],
        "Location": ["state", "reportedCity", "reportedZipCode", "nfipCommunityName"],
    },
}

# Fields to skip: too granular, no ontology value, or personal identifiers
CSV_SKIP_FIELDS: set[str] = {
    "latitude", "longitude", "censusBlockGroupFips", "censusTract",
    "countyCode", "mapPanelNumber", "mapPanelSuffix", "ficoNumber",
}

# Schema description chunks — emitted as chunk_id=0 before data batches,
# giving the LLM the vocabulary to interpret subsequent records correctly.
CSV_SCHEMA_HEADERS: dict[str, str] = {
    "policies": (
        "DATASET SCHEMA: OpenFEMA NFIP Policies\n"
        "Key fields and their meanings:\n"
        "  total building insurance coverage: maximum building claim payout under NFIP policy\n"
        "  total contents insurance coverage: maximum contents claim payout\n"
        "  building deductible code: deductible tier applied to building coverage\n"
        "  rated flood zone: FEMA flood zone designation "
        "(A=high risk, B/C=moderate, V=coastal)\n"
        "  icc premium / icc coverage: Increased Cost of Compliance coverage amount\n"
        "  waiting period type: days before policy takes effect\n"
        "  building replacement cost: estimated cost to replace the insured building\n"
    ),
    "claims": (
        "DATASET SCHEMA: OpenFEMA NFIP Claims\n"
        "Key fields and their meanings:\n"
        "  total building insurance coverage: maximum building claim payout under NFIP policy\n"
        "  total contents insurance coverage: maximum contents claim payout\n"
        "  amount paid on building claim: actual building payment made\n"
        "  amount paid on contents claim: actual contents payment made\n"
        "  cause of damage: flood event type causing the loss\n"
        "  rated flood zone: FEMA flood zone designation\n"
        "  icc coverage: Increased Cost of Compliance coverage amount\n"
        "  water depth: depth of floodwater at insured property\n"
    ),
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HybridChunk:
    chunk_id: int
    content: str
    source: str                       # filename
    section_hierarchy: list[str]      # e.g. ["II. DEFINITIONS", "A. Building"]
    temporal_markers: list[str]       # dates / version strings found in content
    pages: list[int]                  # page numbers spanned
    token_count: int = 0
    merged_from: list[int] = field(default_factory=list)  # pre-merge chunk IDs
    chunk_type: str = "text"          # "text" | "table"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_noise(text: str) -> str:
    """Remove page headers/footers and soft-hyphens from PDF text."""
    text = PAGE_HEADER.sub("", text)
    text = text.replace("\xad", "")   # soft hyphen
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _approx_tokens(text: str) -> int:
    """Fast token estimate: ~0.75 words per token (GPT-style)."""
    return int(len(text.split()) / 0.75)


def _extract_temporal_markers(text: str) -> list[str]:
    return list(dict.fromkeys(DATE_PATTERN.findall(text)))  # dedup, preserve order


def _detect_section_label(line: str) -> Optional[tuple[str, str]]:
    """
    Returns (level, label) if line is a section header, else None.
    level: 'major' (Roman) or 'sub' (letter)
    """
    stripped = line.strip()
    if ROMAN_SECTION.match(stripped):
        return ("major", stripped.split('\n')[0].strip())
    if LETTER_SUBSECTION.match(stripped):
        return ("sub", stripped.split('\n')[0].strip())
    return None


def _humanize_field_name(field_name: str, expansions: dict[str, str] | None = None) -> str:
    """Convert camelCase or UPPER_SNAKE_CASE to human-readable label.

    If *expansions* is provided, uses LLM-generated readable names first.

    Examples:
      'amountPaidOnBuildingClaim' → 'amount paid on building claim'
      'POLICY_NUMBER'             → 'policy number'
      'CLAIM_STATUS'              → 'claim status'
      'POLNO'                     → 'polno'  (or 'policy number' via expansion)
      'GWP'                       → 'gwp'    (or 'gross written premium' via expansion)
    """
    # LLM-expanded readable name takes priority.
    if expansions and field_name in expansions:
        return expansions[field_name]

    if "_" in field_name and field_name == field_name.upper():
        # UPPER_SNAKE_CASE → split on underscores
        return field_name.lower().replace("_", " ").strip()
    # ALL-CAPS without underscores (e.g., POLNO, GWP, MCO) → just lowercase
    if field_name == field_name.upper() and field_name.isalpha():
        return field_name.lower()
    # camelCase → insert spaces before uppercase letters
    s = re.sub(r'([A-Z])', r' \1', field_name)
    return s.lower().strip()


# Heuristic patterns for auto-grouping generic CSV fields.
_FIELD_GROUP_PATTERNS: dict[str, list[str]] = {
    "Identity":   ["id", "name", "insured", "client", "claimant", "master",
                   "polno", "policy_no", "policy_number", "claim_number",
                   "subscriber"],
    "Date":       ["date", "eff_", "exp_", "cxl_", "trm_", "time",
                   "begin_", "period", "month"],
    "Coverage":   ["cov", "coverage", "ded", "deductible", "premium",
                   "endorsement", "limit"],
    "Financial":  ["amount", "paid", "cost", "fee", "gwp", "nwp", "tax",
                   "credit", "loss", "payment", "prem"],
    "Location":   ["state", "city", "zip", "county", "address", "add1",
                   "add2", "add3", "terr", "loc", "risk_st"],
    "Status":     ["status", "indicator", "code", "type", "kind", "reason",
                   "category"],
}


def _auto_group_field(field_name: str) -> str:
    """Assign a field to a semantic group using heuristic pattern matching.

    Uses priority rules to resolve ambiguous fields:
    - ENDORSEMENT_* → Coverage (even if contains "id" or "exp")
    - *_DATE / *_DT suffix → Date (even if contains "subscriber")
    - *_CSAT / *_CES suffix → Status (satisfaction scores, not dates)
    """
    field_lower = field_name.lower()

    # Priority rules (checked before general patterns)
    if field_lower.startswith("endorsement"):
        return "Coverage"
    if field_lower.endswith("_date") or field_lower.endswith("_dt"):
        return "Date"
    if field_lower.endswith("_csat") or field_lower.endswith("_ces"):
        return "Status"

    for group, patterns in _FIELD_GROUP_PATTERNS.items():
        if any(pat in field_lower for pat in patterns):
            return group
    return "Other"


# ---------------------------------------------------------------------------
# PDF ingestion: title-based splitting
# ---------------------------------------------------------------------------

def _split_pdf_by_sections(pdf_path: str) -> list[dict]:
    """
    Load PDF and split at Roman-numeral and letter section boundaries.
    Returns list of raw section dicts with text, hierarchy, pages.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    sections: list[dict] = []
    current_text_lines: list[str] = []
    current_pages: list[int] = []
    current_major = ""
    current_sub = ""

    def flush(next_major: str = "", next_sub: str = "") -> None:
        nonlocal current_text_lines, current_pages, current_major, current_sub
        text = _strip_noise("\n".join(current_text_lines))
        if len(text) > 40:   # skip near-empty sections
            hierarchy = [h for h in [current_major, current_sub] if h]
            sections.append({
                "text": text,
                "section_hierarchy": hierarchy,
                "pages": list(dict.fromkeys(current_pages)),
                "temporal_markers": _extract_temporal_markers(text),
                "merged_from": [],
                "chunk_type": "text",
            })
        current_text_lines = []
        current_pages = []
        current_major = next_major
        current_sub = next_sub

    for page_num, page in enumerate(pages):
        raw = _strip_noise(page.page_content)
        lines = raw.split('\n')

        for line in lines:
            detection = _detect_section_label(line)
            if detection:
                level, label = detection
                if level == "major":
                    flush(next_major=label, next_sub="")
                else:  # sub-section
                    flush(next_major=current_major, next_sub=label)
            current_text_lines.append(line)
            current_pages.append(page_num)

    flush()  # capture last section
    return sections


# ---------------------------------------------------------------------------
# Sub-chunking: split oversized sections by paragraph
# ---------------------------------------------------------------------------

def _sub_chunk_section(section: dict,
                        max_tokens: int = MAX_CHUNK_TOKENS) -> list[dict]:
    """
    Split a single section that exceeds max_tokens by paragraph boundaries.
    Each sub-chunk inherits parent section_hierarchy and appends a 'part N' label.
    The first paragraph (section header) is repeated in every sub-chunk for context.
    Returns [section] unchanged if already within the token limit.
    """
    if _approx_tokens(section["text"]) <= max_tokens:
        return [section]  # already fine — no split needed

    paragraphs = [p.strip() for p in section["text"].split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        # Can't split further (single unsplittable block); return as-is
        return [section]

    header = paragraphs[0]   # section heading — prepended to every sub-chunk
    header_tokens = _approx_tokens(header)
    sub_chunks: list[dict] = []
    current_paras = [header]
    current_tokens = header_tokens
    sub_idx = 0

    for para in paragraphs[1:]:
        para_tokens = _approx_tokens(para)
        if current_tokens + para_tokens > max_tokens and len(current_paras) > 1:
            # Flush current sub-chunk
            sub_chunks.append({
                **section,
                "text": "\n\n".join(current_paras),
                "section_hierarchy": (
                    section["section_hierarchy"] + [f"part {sub_idx + 1}"]
                ),
            })
            sub_idx += 1
            # Next sub-chunk starts with header + this paragraph
            current_paras = [header, para]
            current_tokens = header_tokens + para_tokens
        else:
            current_paras.append(para)
            current_tokens += para_tokens

    # Flush final sub-chunk
    if current_paras:
        sub_chunks.append({
            **section,
            "text": "\n\n".join(current_paras),
            "section_hierarchy": (
                section["section_hierarchy"] + [f"part {sub_idx + 1}"]
            ),
        })

    return sub_chunks if sub_chunks else [section]


# ---------------------------------------------------------------------------
# Semantic merging
# ---------------------------------------------------------------------------

def _semantic_merge(sections: list[dict], model: SentenceTransformer,
                    threshold: float = SEMANTIC_MERGE_THRESHOLD) -> list[dict]:
    """
    Merge adjacent sections whose embedding cosine similarity > threshold.
    Token ceiling guard: refuses to merge if combined size > MAX_CHUNK_TOKENS,
    preserving section boundaries even when similarity is high.
    Implements plan §3.2: 'semantic merging to avoid fragmenting related content'.
    """
    if not sections:
        return []

    texts = [s["text"] for s in sections]
    print(f"  Embedding {len(texts)} sections with {EMBED_MODEL}...")
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    # Precompute all adjacent similarities in one vectorised pass.
    # Embeddings are L2-normalized, so dot product = cosine similarity.
    adjacent_sims = (embeddings[:-1] * embeddings[1:]).sum(axis=1) if len(embeddings) > 1 else np.array([])

    merged: list[dict] = []
    i = 0
    while i < len(sections):
        current = dict(sections[i])
        current["merged_from"] = [i]
        # Anchor embedding: always compare next candidate against the FIRST
        # section in this merge group (not the last-absorbed one).
        anchor_idx = i

        # Try to absorb the next section if similar enough AND within token ceiling
        while i + 1 < len(sections):
            # Use anchor similarity: compare first section of group to candidate.
            # For the first candidate (i == anchor_idx) this equals adjacent_sims[i].
            # For subsequent candidates, recompute against the anchor.
            if i == anchor_idx:
                sim = float(adjacent_sims[i])
            else:
                sim = float(np.dot(embeddings[anchor_idx], embeddings[i + 1]))

            if sim >= threshold:
                next_sec = sections[i + 1]
                # Token ceiling guard: refuse merge if combined exceeds MAX_CHUNK_TOKENS
                combined_tokens = _approx_tokens(
                    current["text"] + "\n\n" + next_sec["text"]
                )
                if combined_tokens > MAX_CHUNK_TOKENS:
                    break  # preserve boundary even when similarity is high

                # Merge: concatenate text, union metadata
                current["text"] += "\n\n" + next_sec["text"]
                current["pages"] = list(dict.fromkeys(
                    current["pages"] + next_sec["pages"]
                ))
                current["temporal_markers"] = list(dict.fromkeys(
                    current["temporal_markers"] + next_sec["temporal_markers"]
                ))
                # Keep the more specific hierarchy (longer list wins)
                if len(next_sec["section_hierarchy"]) > len(current["section_hierarchy"]):
                    current["section_hierarchy"] = next_sec["section_hierarchy"]
                current["merged_from"].append(i + 1)
                i += 1
            else:
                break

        merged.append(current)
        i += 1

    return merged


# ---------------------------------------------------------------------------
# PDF table extraction (pdfplumber)
# ---------------------------------------------------------------------------

def _extract_pdf_tables(pdf_path: str) -> list[dict]:
    """
    Use pdfplumber to extract tables with preserved cell structure.
    Returns list of table section dicts (same shape as text sections but
    with chunk_type='table' and structured TABLE: header | header\\nrow... text).
    Falls back to empty list if pdfplumber is unavailable.
    """
    if not PDFPLUMBER_AVAILABLE:
        return []

    table_sections: list[dict] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if not tables:
                    continue
                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    headers = [str(h or "").strip() for h in table[0]]
                    if not any(headers):   # skip tables with no usable header row
                        continue

                    rows: list[dict] = []
                    for row in table[1:]:
                        cells = [str(c or "").strip() for c in row]
                        if any(cells):    # skip blank rows
                            rows.append(dict(zip(headers, cells)))

                    if not rows:
                        continue

                    # Format as structured text the LLM can parse as triples
                    lines = ["TABLE: " + " | ".join(h for h in headers if h)]
                    for row in rows:
                        row_parts = [f"{k}: {v}" for k, v in row.items() if k and v]
                        if row_parts:
                            lines.append(" | ".join(row_parts))

                    text = "\n".join(lines)
                    table_sections.append({
                        "text": text,
                        "section_hierarchy": [f"[Table p.{page_num + 1}]"],
                        "pages": [page_num],
                        "temporal_markers": _extract_temporal_markers(text),
                        "merged_from": [],
                        "chunk_type": "table",
                    })
    except Exception as e:
        print(f"  [warn] pdfplumber table extraction error: {e}")

    return table_sections


# ---------------------------------------------------------------------------
# CSV ingestion (OpenFEMA)
# ---------------------------------------------------------------------------

def _format_csv_record(rec: dict, record_type: str) -> str:
    """
    Format a single CSV record into grouped, human-readable text.
    Uses CSV_FIELD_GROUPS to organize fields by semantic category.
    Skips null/False/empty values and fields in CSV_SKIP_FIELDS.
    camelCase field names are converted to human-readable labels.
    """
    groups = CSV_FIELD_GROUPS.get(record_type, {})
    lines: list[str] = []
    seen: set[str] = set()

    for group_name, fields in groups.items():
        parts: list[str] = []
        for f in fields:
            if f in CSV_SKIP_FIELDS:
                continue
            v = rec.get(f, "")
            if v is None or str(v).strip() in ("", "False", "0", "0.0", "0.00"):
                continue
            seen.add(f)
            label = _humanize_field_name(f)
            parts.append(f"{label}: {v}")
        if parts:
            lines.append(f"  [{group_name}] " + " | ".join(parts))

    # Up to 5 ungrouped remaining fields to cap token bloat
    extras: list[str] = []
    for f, v in rec.items():
        if f in seen or f in CSV_SKIP_FIELDS:
            continue
        if v is None or str(v).strip() in ("", "False", "0", "0.0", "0.00"):
            continue
        label = _humanize_field_name(f)
        extras.append(f"{label}: {v}")
    if extras:
        lines.append("  [Other] " + " | ".join(extras[:5]))

    return "RECORD:\n" + "\n".join(lines)


def _chunk_csv_records(
    records: list[dict],
    source: str,
    record_type: str = "policies",
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[dict]:
    """
    Convert structured CSV records into token-capped text chunks.
    Uses dynamic batching (replaces old fixed batch_size=50) to respect max_tokens.
    First chunk is always a schema description to give the LLM field vocabulary.
    Each subsequent chunk prepends a brief dataset header for context.
    """
    DATE_FIELDS = {
        "dateOfLoss", "originalNBDate", "originalConstructionDate",
        "policyEffectiveDate", "policyTerminationDate", "asOfDate",
    }
    dataset_title = record_type.title()  # "Policies" | "Claims"
    schema_header_line = f"DATASET: OpenFEMA NFIP {dataset_title}\n"

    chunks: list[dict] = []

    # --- Chunk 0: schema description ---
    schema_text = schema_header_line + CSV_SCHEMA_HEADERS.get(record_type, "")
    chunks.append({
        "text": schema_text,
        "section_hierarchy": [source, "schema"],
        "pages": [],
        "temporal_markers": [],
        "merged_from": [],
        "chunk_type": "text",
    })

    # --- Data chunks: dynamic token-capped batching ---
    current_batch: list[str] = []
    current_tokens: int = 0
    current_temporal: list[str] = []
    batch_start_idx: int = 0

    def flush_batch(end_idx: int) -> None:
        nonlocal current_batch, current_tokens, current_temporal, batch_start_idx
        if not current_batch:
            return
        text = schema_header_line + "\n\n".join(current_batch)
        chunks.append({
            "text": text,
            "section_hierarchy": [source, f"records {batch_start_idx}–{end_idx}"],
            "pages": [],
            "temporal_markers": list(dict.fromkeys(current_temporal)),
            "merged_from": [],
            "chunk_type": "text",
        })
        current_batch = []
        current_tokens = 0
        current_temporal = []
        batch_start_idx = end_idx + 1

    for rec_idx, rec in enumerate(records):
        # Extract temporal markers before formatting
        temporal: list[str] = []
        for df in DATE_FIELDS:
            v = rec.get(df, "")
            if v and isinstance(v, str):
                temporal.append(v[:10])  # YYYY-MM-DD prefix

        formatted = _format_csv_record(rec, record_type)
        tok = _approx_tokens(formatted)

        if current_tokens + tok > max_tokens and current_batch:
            flush_batch(rec_idx - 1)

        current_batch.append(formatted)
        current_tokens += tok
        current_temporal.extend(temporal)

    flush_batch(len(records) - 1)   # flush final batch

    return chunks


# ---------------------------------------------------------------------------
# Generic CSV ingestion (domain-agnostic)
# ---------------------------------------------------------------------------

# Fields to always skip in generic CSVs (internal/audit columns).
_GENERIC_SKIP_FIELDS: set[str] = {
    "bi_created_dt", "bi_created_by", "bi_modified_dt", "bi_modified_by",
}

# Values treated as empty in generic CSVs.
_EMPTY_VALUES: frozenset[str] = frozenset({
    "", "nan", "none", "null", "false", "0", "0.0", "0.00",
    "1900-01-01", "1900-01-01T00:00:00.000Z",
})


def _infer_record_type_from_filename(filename: str) -> str:
    """Infer record type from filename heuristics.

    'synthetic_data_sample_geicorentersclaims.csv' → 'claims'
    'synthetic_data_sample_geicorenterspoliciesdetails.csv' → 'policies'
    'synthetic_data_sample_tmobileclaimsample.csv' → 'claims'
    """
    name = filename.lower()
    if "polic" in name:
        return "policies"
    if "claim" in name:
        return "claims"
    if "survey" in name or "cancel" in name:
        return "survey"
    return "unknown"


def _expand_csv_headers(
    headers: list[str],
    sample_rows: list[dict[str, str]],
    cache_path: str | None = None,
    model: str | None = None,
) -> dict[str, str]:
    """Use an LLM to expand cryptic/abbreviated CSV column headers.

    Sends column names + a few sample rows and asks the LLM to return a
    JSON mapping of abbreviated names → readable English names.  Results
    are cached to *cache_path* so subsequent runs skip the LLM call.

    Only expands fields that look abbreviated (all-caps, short, no spaces).
    Returns a mapping ``{RAW_HEADER: "readable name", ...}``.
    """
    # Load cache if available.
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            print(f"  ✓ Loaded cached header expansions ({len(cached)} entries)")
            return cached
        except (json.JSONDecodeError, OSError):
            pass  # rebuild

    # Identify headers that need expansion (short all-caps abbreviations).
    needs_expansion = [
        h for h in headers
        if h == h.upper() and len(h) <= 12 and h.isalpha()
        and h.lower() not in _GENERIC_SKIP_FIELDS
    ]
    if not needs_expansion:
        return {}

    # Build sample data for context.
    sample_lines: list[str] = []
    for i, row in enumerate(sample_rows[:3]):
        vals = {h: str(row.get(h, ""))[:40] for h in needs_expansion if row.get(h)}
        if vals:
            sample_lines.append(f"Row {i+1}: {json.dumps(vals)}")

    prompt = (
        "You are a data dictionary expert. Given these abbreviated CSV column "
        "names and sample values, expand each abbreviation into a short, "
        "readable English label (lowercase, 1-4 words).\n\n"
        f"Columns to expand: {needs_expansion}\n\n"
        + ("\n".join(sample_lines) + "\n\n" if sample_lines else "")
        + "Return ONLY a JSON object mapping each column name to its "
        "expanded label. Example: {\"GWP\": \"gross written premium\", "
        "\"MCO\": \"master company code\"}\n"
        "If you are unsure about a column, use the original name lowercased."
    )

    # Call LLM (optional — gracefully degrade if unavailable).
    try:
        import requests
    except ImportError:
        print("  ⚠ requests library not available, skipping header expansion")
        return {}

    try:
        import config as cfg
        llm_model = model or cfg.OLLAMA_MODEL
        base_url = cfg.OLLAMA_BASE_URL
    except (ImportError, AttributeError) as e:
        print(f"  ⚠ Config not available ({e}), skipping header expansion")
        return {}

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": llm_model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0, "num_predict": 1024}},
            timeout=120,
        )
        resp.raise_for_status()
        raw_text = resp.json().get("response", "")

        # Extract JSON from response (flat object, no nesting expected).
        json_match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
        if json_match:
            expansions = json.loads(json_match.group())
            # Normalize values to lowercase strings.
            expansions = {
                k: str(v).lower().strip()
                for k, v in expansions.items()
                if k in needs_expansion and isinstance(v, str) and v.strip()
            }
            print(f"  ✓ LLM expanded {len(expansions)}/{len(needs_expansion)} headers")

            # Cache results.
            if cache_path:
                parent = os.path.dirname(cache_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(expansions, f, indent=2)

            return expansions

    except requests.RequestException as e:
        print(f"  ⚠ Header expansion failed ({e}), using raw names")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠ Could not parse LLM response ({e}), using raw names")

    return {}


def _format_generic_csv_record(
    rec: dict[str, str],
    headers: list[str],
    expansions: dict[str, str] | None = None,
) -> str:
    """Format a generic CSV record into grouped, human-readable text.

    Uses auto-grouping by field name heuristics. Skips empty/zero values
    and internal audit columns.
    """
    groups: dict[str, list[str]] = {}

    for field_name in headers:
        if field_name.lower() in _GENERIC_SKIP_FIELDS:
            continue
        value = str(rec.get(field_name, "")).strip()
        if value.lower() in _EMPTY_VALUES:
            continue

        group = _auto_group_field(field_name)
        label = _humanize_field_name(field_name, expansions)
        if group not in groups:
            groups[group] = []
        groups[group].append(f"{label}: {value}")

    lines: list[str] = []
    # Emit groups in a stable order.
    group_order = ["Identity", "Date", "Coverage", "Financial",
                   "Location", "Status", "Other"]
    for g in group_order:
        parts = groups.get(g, [])
        if parts:
            lines.append(f"  [{g}] " + " | ".join(parts))

    return "RECORD:\n" + "\n".join(lines) if lines else ""


def _chunk_generic_csv_records(
    records: list[dict[str, str]],
    headers: list[str],
    source: str,
    dataset_name: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    expansions: dict[str, str] | None = None,
) -> list[dict]:
    """Convert generic CSV records into token-capped text chunks.

    Same batching logic as OpenFEMA but with auto-generated schema
    and auto-grouped fields.
    """
    # Schema chunk: list all column headers with their auto-detected groups.
    schema_lines = [f"DATASET SCHEMA: {dataset_name}"]
    schema_lines.append(f"Total fields: {len(headers)}")
    for g in ["Identity", "Date", "Coverage", "Financial",
              "Location", "Status", "Other"]:
        fields_in_group = [h for h in headers if _auto_group_field(h) == g
                           and h.lower() not in _GENERIC_SKIP_FIELDS]
        if fields_in_group:
            humanized = [_humanize_field_name(f, expansions) for f in fields_in_group]
            schema_lines.append(f"  [{g}] " + ", ".join(humanized))

    schema_text = "\n".join(schema_lines)
    dataset_header = f"DATASET: {dataset_name}\n"

    chunks: list[dict] = []

    # Chunk 0: schema description.
    chunks.append({
        "text": schema_text,
        "section_hierarchy": [source, "schema"],
        "pages": [],
        "temporal_markers": [],
        "merged_from": [],
        "chunk_type": "text",
    })

    # Data chunks: dynamic token-capped batching.
    current_batch: list[str] = []
    current_tokens: int = 0
    batch_start_idx: int = 0

    def flush_batch(end_idx: int) -> None:
        nonlocal current_batch, current_tokens, batch_start_idx
        if not current_batch:
            return
        text = dataset_header + "\n\n".join(current_batch)
        chunks.append({
            "text": text,
            "section_hierarchy": [source, f"records {batch_start_idx}–{end_idx}"],
            "pages": [],
            "temporal_markers": [],
            "merged_from": [],
            "chunk_type": "text",
        })
        current_batch = []
        current_tokens = 0
        batch_start_idx = end_idx + 1

    for rec_idx, rec in enumerate(records):
        formatted = _format_generic_csv_record(rec, headers, expansions)
        if not formatted:
            continue
        tok = _approx_tokens(formatted)

        if current_tokens + tok > max_tokens and current_batch:
            flush_batch(rec_idx - 1)

        current_batch.append(formatted)
        current_tokens += tok

    flush_batch(len(records) - 1)

    return chunks


def ingest_generic_csv(
    csv_path: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    cache_dir: str | None = None,
    model: str | None = None,
) -> list[HybridChunk]:
    """Zone 1 ingestion for standard CSV files (domain-agnostic).

    Auto-detects record type from filename, auto-groups fields by
    heuristic patterns, and produces the same chunk format as OpenFEMA
    ingestion for seamless downstream processing.

    If *cache_dir* is provided, uses LLM-based header expansion to
    convert cryptic abbreviated column names into readable labels.
    """
    import csv as csv_mod

    filename = os.path.basename(csv_path)
    record_type = _infer_record_type_from_filename(filename)
    dataset_name = filename.replace("synthetic_data_sample_", "").replace(
        ".csv", "").replace("_", " ").title()

    print(f"\n[CSV] {filename} (record_type={record_type}, dataset={dataset_name})")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        headers = list(reader.fieldnames or [])
        records = list(reader)

    print(f"  Loaded {len(records)} records, {len(headers)} columns")

    # Expand abbreviated headers via LLM (cached per-file).
    expansions: dict[str, str] = {}
    if cache_dir:
        cache_file = os.path.join(
            cache_dir, f"header_expansions_{filename.replace('.csv', '')}.json"
        )
        expansions = _expand_csv_headers(
            headers, records[:3], cache_path=cache_file, model=model,
        )

    raw = _chunk_generic_csv_records(
        records, headers, filename, dataset_name, max_tokens,
        expansions=expansions,
    )
    print(f"  → {len(raw)} chunks total "
          f"(1 schema + {len(raw) - 1} data, dynamic tok-cap max={max_tokens})")

    chunks: list[HybridChunk] = []
    for i, sec in enumerate(raw):
        chunks.append(HybridChunk(
            chunk_id=i,
            content=sec["text"],
            source=csv_path,
            section_hierarchy=sec["section_hierarchy"],
            temporal_markers=sec["temporal_markers"],
            pages=[],
            token_count=_approx_tokens(sec["text"]),
            merged_from=sec["merged_from"],
            chunk_type=sec.get("chunk_type", "text"),
        ))
    return chunks


# ---------------------------------------------------------------------------
# Main ingestion functions
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: str, model: SentenceTransformer) -> list[HybridChunk]:
    """Full Zone 1 pipeline for a PDF: split → sub-chunk → embed → merge → tables → package."""
    print(f"\n[PDF] {os.path.basename(pdf_path)}")
    source = os.path.basename(pdf_path)

    print("  Step 1: Title-based section splitting...")
    raw_sections = _split_pdf_by_sections(pdf_path)
    print(f"  → {len(raw_sections)} raw sections detected")

    print(f"  Step 1b: Sub-chunking oversized sections "
          f"(ceiling={MAX_CHUNK_TOKENS} tokens)...")
    expanded: list[dict] = []
    oversized_count = 0
    for sec in raw_sections:
        if _approx_tokens(sec["text"]) > MAX_CHUNK_TOKENS:
            oversized_count += 1
        sub = _sub_chunk_section(sec)
        expanded.extend(sub)
    print(f"  → {len(expanded)} sections after sub-chunking "
          f"({oversized_count} oversized section(s) split)")

    print(f"  Step 2: Semantic merging "
          f"(τ={SEMANTIC_MERGE_THRESHOLD}, ceiling={MAX_CHUNK_TOKENS} tok)...")
    merged = _semantic_merge(expanded, model)
    print(f"  → {len(merged)} chunks after merging "
          f"({len(expanded) - len(merged)} merge(s) performed)")

    # Build text chunks
    chunks: list[HybridChunk] = []
    for i, sec in enumerate(merged):
        chunks.append(HybridChunk(
            chunk_id=i,
            content=sec["text"],
            source=source,
            section_hierarchy=sec["section_hierarchy"],
            temporal_markers=sec["temporal_markers"],
            pages=sec["pages"],
            token_count=_approx_tokens(sec["text"]),
            merged_from=sec.get("merged_from", []),
            chunk_type=sec.get("chunk_type", "text"),
        ))

    # --- Table extraction (pdfplumber parallel pass) ---
    print("  Step 3: Table extraction (pdfplumber)...")
    table_sections = _extract_pdf_tables(pdf_path)
    print(f"  → {len(table_sections)} table(s) detected")
    for sec in table_sections:
        chunks.append(HybridChunk(
            chunk_id=len(chunks),
            content=sec["text"],
            source=source,
            section_hierarchy=sec["section_hierarchy"],
            temporal_markers=sec["temporal_markers"],
            pages=sec["pages"],
            token_count=_approx_tokens(sec["text"]),
            merged_from=sec.get("merged_from", []),
            chunk_type="table",
        ))

    return chunks


def ingest_csv(
    json_path: str,
    record_key: str,
    record_type: str = "policies",
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[HybridChunk]:
    """Zone 1 ingestion for OpenFEMA JSON/CSV data with dynamic token-capped chunking."""
    print(f"\n[CSV] {os.path.basename(json_path)} (record_type={record_type})")
    with open(json_path) as f:
        data = json.load(f)

    records = dict(data).get(record_key, [])
    print(f"  Loaded {len(records)} records")

    raw = _chunk_csv_records(records, record_key, record_type, max_tokens)
    # -1 to exclude the schema description chunk from data chunk count
    print(f"  → {len(raw)} chunks total "
          f"(1 schema + {len(raw) - 1} data, dynamic tok-cap max={max_tokens})")

    chunks: list[HybridChunk] = []
    for i, sec in enumerate(raw):
        chunks.append(HybridChunk(
            chunk_id=i,
            content=sec["text"],
            source=json_path,
            section_hierarchy=sec["section_hierarchy"],
            temporal_markers=sec["temporal_markers"],
            pages=[],
            token_count=_approx_tokens(sec["text"]),
            merged_from=sec["merged_from"],
            chunk_type=sec.get("chunk_type", "text"),
        ))
    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_zone1(
    data_dir: str | None = None,
    output_path: str | None = None,
    llm_model: str | None = None,
) -> list[HybridChunk]:
    """Run Zone 1 ingestion on a data directory.

    Auto-discovers PDFs and CSVs in the directory structure:
      - data_dir/raw/pdf/*.pdf        → PDF ingestion
      - data_dir/raw/openfema/*.json  → OpenFEMA JSON ingestion
      - data_dir/*.csv                → Generic CSV ingestion
      - data_dir/*.pdf                → PDF ingestion

    Args:
        data_dir: Root directory containing source files. If None, uses
                  the default flood data directory from config.
        output_path: Where to save chunks JSON. If None, uses config default.
        llm_model: Ollama model for LLM-based header expansion.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("Zone 1: Hybrid Ingestion (Novel Pipeline)")
    print("=" * 60)

    model = SentenceTransformer(EMBED_MODEL)

    all_chunks: list[HybridChunk] = []
    pdf_chunks: list[HybridChunk] = []
    csv_chunk_groups: list[tuple[str, list[HybridChunk]]] = []

    if data_dir is None:
        # --- Default: flood data (OpenFEMA + SFIP PDF) ---
        pdf_path = config.PDF_PATH
        pdf_chunks = ingest_pdf(pdf_path, model)
        all_chunks.extend(pdf_chunks)

        policies_chunks = ingest_csv(
            os.path.join(config.OPENFEMA_DIR, "policies_sample.json"),
            record_key="FimaNfipPolicies",
            record_type="policies",
        )
        all_chunks.extend(policies_chunks)
        csv_chunk_groups.append(("Policies (OpenFEMA)", policies_chunks))

        claims_chunks = ingest_csv(
            os.path.join(config.OPENFEMA_DIR, "claims_sample.json"),
            record_key="FimaNfipClaims",
            record_type="claims",
        )
        all_chunks.extend(claims_chunks)
        csv_chunk_groups.append(("Claims (OpenFEMA)", claims_chunks))
    else:
        # --- Generic: scan directory for PDFs and CSVs ---
        print(f"\n  Data directory: {data_dir}")

        # Discover PDFs (in root and subdirectories).
        for root, _dirs, files in os.walk(data_dir):
            for f in sorted(files):
                fpath = os.path.join(root, f)
                if f.lower().endswith(".pdf"):
                    chunks = ingest_pdf(fpath, model)
                    pdf_chunks.extend(chunks)
                    all_chunks.extend(chunks)

        # Discover CSVs (in root and subdirectories).
        # Cache LLM header expansions alongside the output.
        cache_dir = os.path.join(
            output_path and os.path.dirname(output_path) or data_dir,
            "processed" if not output_path else "",
        ).rstrip("/")
        for root, _dirs, files in os.walk(data_dir):
            for f in sorted(files):
                fpath = os.path.join(root, f)
                if f.lower().endswith(".csv"):
                    chunks = ingest_generic_csv(
                        fpath, cache_dir=cache_dir, model=llm_model,
                    )
                    all_chunks.extend(chunks)
                    csv_chunk_groups.append((f, chunks))

        # Discover OpenFEMA JSONs (in case they're mixed in).
        for root, _dirs, files in os.walk(data_dir):
            for f in sorted(files):
                fpath = os.path.join(root, f)
                if f.lower().endswith(".json") and "sample" in f.lower():
                    # Attempt to load as OpenFEMA format.
                    try:
                        with open(fpath) as fh:
                            data = json.load(fh)
                        for key in data:
                            if isinstance(data[key], list) and len(data[key]) > 0:
                                rtype = "policies" if "polic" in key.lower() else "claims"
                                chunks = ingest_csv(fpath, key, rtype)
                                all_chunks.extend(chunks)
                                csv_chunk_groups.append((f"{f} ({key})", chunks))
                    except (json.JSONDecodeError, Exception):
                        pass

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("ZONE 1 SUMMARY")
    print(f"{'=' * 60}")
    pdf_text_chunks  = [c for c in pdf_chunks if c.chunk_type == "text"]
    pdf_table_chunks = [c for c in pdf_chunks if c.chunk_type == "table"]
    print(f"  PDF text chunks:   {len(pdf_text_chunks):>4}")
    print(f"  PDF table chunks:  {len(pdf_table_chunks):>4}")
    for name, chunks in csv_chunk_groups:
        print(f"  {name}: {len(chunks):>4} chunks "
              f"(1 schema + {len(chunks)-1} data)")
    print(f"  Total chunks:      {len(all_chunks):>4}")

    # Token distribution
    token_counts = [c.token_count for c in all_chunks if c.token_count > 0]
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        max_tok     = max(token_counts)
        over_limit  = sum(1 for t in token_counts if t > MAX_CHUNK_TOKENS)
        print(f"  Avg tokens/chunk:  {avg_tokens:.0f}")
        print(f"  Max tokens/chunk:  {max_tok}")
        if over_limit:
            print(f"  ⚠ Chunks > {MAX_CHUNK_TOKENS} tok: {over_limit} "
                  f"(check for unsplittable blocks)")
        else:
            print(f"  ✓ All chunks within {MAX_CHUNK_TOKENS}-token ceiling")

    # Show a few PDF chunks to verify section detection
    if pdf_chunks:
        print(f"\n  Sample PDF chunks:")
        for c in pdf_chunks[:5]:
            print(f"    [{c.chunk_id:>2}] type={c.chunk_type:<5}  "
                  f"tokens={c.token_count:>4}  "
                  f"hierarchy={c.section_hierarchy}")

    # Save
    out = [asdict(c) for c in all_chunks]
    out_path = output_path or config.ZONE1_CHUNKS_FILE
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ✓ Saved {len(out)} chunks → {out_path}")

    return all_chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zone 1: Hybrid Ingestion")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to data directory with PDFs and CSVs. "
             "Default: OpenFEMA flood data.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for chunks JSON. "
             "Default: data/flood/processed/zone1_chunks.json",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model for LLM-based header expansion. "
             "Default: config.OLLAMA_MODEL",
    )
    args = parser.parse_args()
    run_zone1(data_dir=args.data_dir, output_path=args.output,
              llm_model=args.model)
