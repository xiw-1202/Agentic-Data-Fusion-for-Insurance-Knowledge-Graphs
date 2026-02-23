"""
Zone 1: Multimodal Ingestion — Novel Pipeline
=============================================
Hybrid chunking strategy (per project plan §3.2):
  1. Title-based splitting at document section boundaries
  2. Semantic merging: merge adjacent chunks when cosine similarity > τ=0.85
     to avoid fragmenting related content
  3. Metadata enrichment: source, section hierarchy, temporal markers

Contrast with baseline: fixed 512-token sliding window with no semantic awareness.

Supports:
  - PDF: SFIP policy documents (section-aware)
  - CSV: OpenFEMA policies + claims (row-batch chunking with field metadata)
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

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEMANTIC_MERGE_THRESHOLD = 0.85   # τ from plan §3.2
EMBED_MODEL = "all-MiniLM-L6-v2"  # matches plan stack

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

    sections = []
    current_text_lines = []
    current_pages = []
    current_major = ""
    current_sub = ""
    current_start_page = 0

    def flush(next_major="", next_sub=""):
        nonlocal current_text_lines, current_pages, current_major, current_sub
        text = _strip_noise("\n".join(current_text_lines))
        if len(text) > 40:   # skip near-empty sections
            hierarchy = [h for h in [current_major, current_sub] if h]
            sections.append({
                "text": text,
                "section_hierarchy": hierarchy,
                "pages": list(dict.fromkeys(current_pages)),
                "temporal_markers": _extract_temporal_markers(text),
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
# Semantic merging
# ---------------------------------------------------------------------------

def _semantic_merge(sections: list[dict], model: SentenceTransformer,
                    threshold: float = SEMANTIC_MERGE_THRESHOLD) -> list[dict]:
    """
    Merge adjacent sections whose embedding cosine similarity > threshold.
    Implements plan §3.2: 'semantic merging to avoid fragmenting related content'.
    """
    if not sections:
        return []

    texts = [s["text"] for s in sections]
    print(f"  Embedding {len(texts)} sections with {EMBED_MODEL}...")
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    merged = []
    i = 0
    while i < len(sections):
        current = dict(sections[i])
        current["merged_from"] = [i]

        # Try to absorb the next section if similar enough
        while i + 1 < len(sections):
            sim = float(cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0])

            if sim >= threshold:
                next_sec = sections[i + 1]
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
# CSV ingestion (OpenFEMA)
# ---------------------------------------------------------------------------

def _chunk_csv_records(records: list[dict], source: str,
                        batch_size: int = 50) -> list[dict]:
    """
    Convert structured CSV records into text chunks for triple extraction.
    Groups records into batches; each batch becomes one chunk.
    Temporal markers extracted from date fields.
    """
    DATE_FIELDS = {
        "dateOfLoss", "originalNBDate", "originalConstructionDate",
        "policyEffectiveDate", "policyTerminationDate", "asOfDate",
    }

    chunks = []
    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start: batch_start + batch_size]
        lines = []
        temporal = []

        for rec in batch:
            parts = []
            for k, v in rec.items():
                if v is None or v == "":
                    continue
                if k in DATE_FIELDS and isinstance(v, str):
                    temporal.append(v[:10])  # YYYY-MM-DD
                parts.append(f"{k}: {v}")
            lines.append("RECORD: " + " | ".join(parts))

        chunks.append({
            "text": "\n".join(lines),
            "section_hierarchy": [source, f"records {batch_start}–{batch_start+len(batch)-1}"],
            "pages": [],
            "temporal_markers": list(dict.fromkeys(temporal)),
            "merged_from": [],
        })
    return chunks


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: str, model: SentenceTransformer) -> list[HybridChunk]:
    """Full Zone 1 pipeline for a PDF: split → embed → merge → package."""
    print(f"\n[PDF] {os.path.basename(pdf_path)}")
    source = os.path.basename(pdf_path)

    print("  Step 1: Title-based section splitting...")
    raw_sections = _split_pdf_by_sections(pdf_path)
    print(f"  → {len(raw_sections)} raw sections detected")

    print(f"  Step 2: Semantic merging (τ={SEMANTIC_MERGE_THRESHOLD})...")
    merged = _semantic_merge(raw_sections, model)
    print(f"  → {len(merged)} chunks after merging "
          f"({len(raw_sections) - len(merged)} merges performed)")

    chunks = []
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
        ))

    return chunks


def ingest_csv(json_path: str, record_key: str,
               batch_size: int = 50) -> list[HybridChunk]:
    """Zone 1 ingestion for OpenFEMA JSON/CSV data."""
    print(f"\n[CSV] {os.path.basename(json_path)}")
    with open(json_path) as f:
        data = json.load(f)

    records = dict(data).get(record_key, [])
    print(f"  Loaded {len(records)} records")

    raw = _chunk_csv_records(records, record_key, batch_size)
    print(f"  → {len(raw)} chunks (batch_size={batch_size})")

    chunks = []
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
        ))
    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_zone1():
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
        batch_size=50,
    )
    all_chunks.extend(policies_chunks)

    claims_chunks = ingest_csv(
        os.path.join(config.OPENFEMA_DIR, "claims_sample.json"),
        record_key="FimaNfipClaims",
        batch_size=50,
    )
    all_chunks.extend(claims_chunks)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("ZONE 1 SUMMARY")
    print(f"{'=' * 60}")
    print(f"  PDF chunks:      {len(pdf_chunks):>4}  (baseline had 56 fixed-size)")
    print(f"  Policy chunks:   {len(policies_chunks):>4}")
    print(f"  Claims chunks:   {len(claims_chunks):>4}")
    print(f"  Total chunks:    {len(all_chunks):>4}")
    avg_tokens = sum(c.token_count for c in all_chunks) / len(all_chunks)
    print(f"  Avg tokens/chunk: {avg_tokens:.0f}")

    # Show a few PDF chunks to verify section detection
    print(f"\n  Sample PDF chunks:")
    for c in pdf_chunks[:5]:
        print(f"    [{c.chunk_id}] hierarchy={c.section_hierarchy} | "
              f"pages={c.pages} | tokens={c.token_count} | "
              f"merged={len(c.merged_from)} raw sections")

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
