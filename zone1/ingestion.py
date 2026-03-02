"""
Zone 1: Multimodal Ingestion — Novel Pipeline
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
"""

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
from sklearn.metrics.pairwise import cosine_similarity

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
PARAGRAPH_OVERLAP_TOKENS = 50      # header repeated in each sub-chunk for context

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
    "id", "latitude", "longitude", "censusBlockGroupFips", "censusTract",
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


def _humanize_field_name(field_name: str) -> str:
    """Convert camelCase to human-readable label.

    Example: 'amountPaidOnBuildingClaim' → 'amount paid on building claim'
    """
    s = re.sub(r'([A-Z])', r' \1', field_name)
    return s.lower().strip()


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

    merged: list[dict] = []
    i = 0
    while i < len(sections):
        current = dict(sections[i])
        current["merged_from"] = [i]

        # Try to absorb the next section if similar enough AND within token ceiling
        while i + 1 < len(sections):
            sim = float(cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0])

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
            batch_start_idx = rec_idx  # reset after flush

        current_batch.append(formatted)
        current_tokens += tok
        current_temporal.extend(temporal)

    flush_batch(len(records) - 1)   # flush final batch

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

def run_zone1() -> list[HybridChunk]:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("Zone 1: Hybrid Ingestion (Novel Pipeline)")
    print("=" * 60)

    model = SentenceTransformer(EMBED_MODEL)

    all_chunks: list[HybridChunk] = []

    # --- PDF ---
    pdf_path = config.PDF_PATH
    pdf_chunks = ingest_pdf(pdf_path, model)
    all_chunks.extend(pdf_chunks)

    # --- OpenFEMA CSVs ---
    policies_chunks = ingest_csv(
        os.path.join(config.OPENFEMA_DIR, "policies_sample.json"),
        record_key="FimaNfipPolicies",
        record_type="policies",
    )
    all_chunks.extend(policies_chunks)

    claims_chunks = ingest_csv(
        os.path.join(config.OPENFEMA_DIR, "claims_sample.json"),
        record_key="FimaNfipClaims",
        record_type="claims",
    )
    all_chunks.extend(claims_chunks)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("ZONE 1 SUMMARY")
    print(f"{'=' * 60}")
    pdf_text_chunks  = [c for c in pdf_chunks if c.chunk_type == "text"]
    pdf_table_chunks = [c for c in pdf_chunks if c.chunk_type == "table"]
    print(f"  PDF text chunks:   {len(pdf_text_chunks):>4}  "
          f"(baseline had 56 fixed-size)")
    print(f"  PDF table chunks:  {len(pdf_table_chunks):>4}")
    print(f"  Policy chunks:     {len(policies_chunks):>4}  "
          f"(1 schema + {len(policies_chunks)-1} data)")
    print(f"  Claims chunks:     {len(claims_chunks):>4}  "
          f"(1 schema + {len(claims_chunks)-1} data)")
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
    print(f"\n  Sample PDF chunks:")
    for c in pdf_chunks[:5]:
        print(f"    [{c.chunk_id:>2}] type={c.chunk_type:<5}  "
              f"tokens={c.token_count:>4}  "
              f"hierarchy={c.section_hierarchy}")

    # Save
    out = [asdict(c) for c in all_chunks]
    out_path = config.ZONE1_CHUNKS_FILE
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ✓ Saved {len(out)} chunks → {out_path}")

    return all_chunks


if __name__ == "__main__":
    run_zone1()
