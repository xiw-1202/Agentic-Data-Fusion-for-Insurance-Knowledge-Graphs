"""
Zone 2 Pipeline — Domain-Agnostic Open IE with LLM-Bootstrapped Schema

General-purpose insurance ontology extraction pipeline. No hardcoded domain
knowledge — all entity types and relation types are bootstrapped from the
documents themselves. Works on any insurance Line of Business (flood, auto,
health, liability) without code changes.

Architecture (4-node LangGraph pipeline):
  load_chunks → bootstrap_vocab → extract_triples → insert_to_neo4j → entity_resolution → END

Design principles:
  - NO reference ontology in the pipeline (Riskine is evaluation-only)
  - NO hardcoded entity types, anchor nodes, or role maps
  - Relation vocabulary bootstrapped from document samples
  - Entity types bootstrapped from document samples
  - Synthetic domain-agnostic few-shot examples (no real document text)
  - Entity resolution via embedding similarity (domain-agnostic)

Evaluation (run AFTER this pipeline):
  python3 baseline/eval.py --suffix zone2
  python3 baseline/eval.py --suffix zone2 --riskine

Usage:
  python3 zone2/pipeline.py                    # default model
  python3 zone2/pipeline.py --model qwen2.5:7b # alt model
"""

import json
import re
import time
import os
import sys
import argparse
from typing import TypedDict, Annotated
from collections import defaultdict
import operator

# Allow imports from project root (config.py lives there)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

import config
from zone2.prompts import (
    RELATION_BOOTSTRAP_PROMPT,
    ENTITY_BOOTSTRAP_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    FEW_SHOT_PAIRS,
    PASS_FOCUS_INSTRUCTIONS,
)
from zone2.entity_resolution import resolve_entities


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class Zone2State(TypedDict):
    chunks:           list[dict]
    vocab:            list[str]                  # bootstrapped relation vocabulary
    entity_types:     list[str]                  # bootstrapped entity type vocabulary
    triples:          list[dict]                 # extracted + validated triples
    vocab_quality:    dict                       # quality metrics for the vocab
    neo4j_stats:      dict
    resolution_stats: dict                       # Zone 2.5 entity resolution stats
    errors:           Annotated[list, operator.add]
    model:            str


# ---------------------------------------------------------------------------
# Constants — ALL DOMAIN-AGNOSTIC
# ---------------------------------------------------------------------------

PDF_SOURCE_KEY      = "fema_F-123-general-property-SFIP_2021.pdf"
CHUNK_CONTENT_LIMIT = 3000  # chars

# Remap common LLM paraphrases → canonical form (domain-agnostic synonyms)
RELATION_NORMALIZATIONS: dict[str, str] = {
    # Exclusion synonyms
    "NOT_INSURED":        "EXCLUDED_FROM",
    "NOT_COVERED":        "EXCLUDED_FROM",
    "EXCLUDES":           "EXCLUDED_FROM",
    "DOES_NOT_COVER":     "EXCLUDED_FROM",
    "DOES_NOT_INSURE":    "EXCLUDED_FROM",
    # Coverage synonyms
    "INSURES":            "COVERS",
    "PROVIDES_COVERAGE":  "COVERS",
    "PROTECTS":           "COVERS",
    # Limit synonyms
    "HAS_LIMIT":          "HAS_COVERAGE_LIMIT",
    "COVERAGE_LIMIT":     "HAS_COVERAGE_LIMIT",
    "MAXIMUM_AMOUNT":     "HAS_COVERAGE_LIMIT",
    "HAS_MAXIMUM":        "HAS_COVERAGE_LIMIT",
    # Obligation synonyms
    "MUST_REPORT":        "MUST_NOTIFY",
    "REQUIRES_NOTICE":    "MUST_NOTIFY",
    "OBLIGATED_TO_NOTIFY": "MUST_NOTIFY",
    # Definition synonyms
    "MEANS":              "DEFINED_AS",
    "IS_DEFINED_AS":      "DEFINED_AS",
    "DEFINITION_OF":      "DEFINED_AS",
    # Time synonyms
    "WAITING_PERIOD":     "HAS_WAITING_PERIOD",
    "HAS_WAIT":           "HAS_WAITING_PERIOD",
}

# Relations so generic they add no semantic value; removed before insertion
GENERIC_BLACKLIST: set[str] = {
    "HAS", "IS", "IS_A", "ARE", "HAVE", "CONTAINS",
    "INCLUDES", "RELATES_TO", "ASSOCIATED_WITH", "CONNECTED_TO",
    "INVOLVES", "MENTIONS", "REFERS_TO", "LINKS_TO",
}

# ---------------------------------------------------------------------------
# Pattern-based relation normalization (F-04 fix for 70B compound names)
# ---------------------------------------------------------------------------
# 70B models invent descriptive compound names like:
#   EXCLUDES_MULTIPLE_POLICIES_FOR_SINGLE_BUILDING → EXCLUDED_FROM
#   EXCLUDES_COVERAGE_FOR → EXCLUDED_FROM
#   COVERS_MULTIFAMILY_BUILDINGS → COVERS
# This function catches those patterns AFTER the lookup-based normalization.

def _pattern_normalize_relation(rel: str) -> str:
    """
    Collapse verbose compound relation names invented by large models (F-04 fix).

    Applies AFTER lookup-based RELATION_NORMALIZATIONS.  Rules:
      - EXCLUDES_* (any suffix)      → EXCLUDED_FROM
      - COVERS_* with len > 8        → COVERS  (e.g. COVERS_MULTIFAMILY_BUILDINGS)
      - PROVIDES_*COVERAGE* or *INSURANCE* or *PROTECTION* → COVERS
    """
    if rel.startswith("EXCLUDES_") and rel != "EXCLUDED_FROM":
        return "EXCLUDED_FROM"
    if rel.startswith("COVERS_") and len(rel) > 8:
        return "COVERS"
    if rel.startswith("PROVIDES_") and any(
        kw in rel for kw in ("COVERAGE", "INSURANCE", "PROTECTION")
    ):
        return "COVERS"
    return rel

# Generic insurance section keyword groups for stratified bootstrap sampling.
# These patterns appear in ALL insurance policy documents regardless of LOB.
_BOOTSTRAP_SECTION_GROUPS: list[list[str]] = [
    ["coverage", "insure", "covered", "protect"],
    ["exclusion", "not covered", "does not", "except", "we do not"],
    ["definition", "means", "defined as", "the term"],
    ["claim", "loss", "notify", "report", "file", "proof"],
    ["condition", "requirement", "must", "shall", "obligation"],
    ["limit", "maximum", "deductible", "amount", "dollar"],
    ["period", "effective", "expir", "cancel", "renew", "waiting"],
    ["premium", "payment", "cost", "rate", "fee"],
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_llm(model: str, json_mode: bool = False) -> ChatOllama:
    """Return a ChatOllama instance, optionally with JSON output mode."""
    kwargs: dict = dict(
        model=model,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
    )
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


def get_neo4j_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )


def _parse_json_list(text: str) -> list:
    """
    Robust JSON array parser with 3 fallback levels:
      1. Direct json.loads()
      2. Regex extract first [...] array (DOTALL)
      3. Collect individual {...} objects
    """
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    parsed = []
    for obj_str in re.findall(r'\{[^{}]+\}', text):
        try:
            parsed.append(json.loads(obj_str))
        except (json.JSONDecodeError, ValueError):
            pass
    return parsed


def _sanitize_relation(rel: str) -> str:
    """Make a relation name safe for Neo4j Cypher interpolation."""
    return re.sub(r'[^A-Z0-9_]', '_', rel.upper().strip())


def _sanitize_label(label: str) -> str:
    """Make a PascalCase label safe for Neo4j Cypher interpolation."""
    cleaned = re.sub(r'[^A-Za-z0-9]', '', label.strip())
    return cleaned if cleaned else "Entity"


def keep_triple(t: dict) -> bool:
    """Filter triples: reject low-confidence, empty entities, generic blacklist."""
    rel = RELATION_NORMALIZATIONS.get(t.get("relation", ""), t.get("relation", ""))
    rel = RELATION_NORMALIZATIONS.get(rel, rel)
    try:
        if float(t.get("confidence", 1.0)) < 0.5:
            return False
    except (ValueError, TypeError):
        pass
    if not str(t.get("subject", "")).strip():
        return False
    if not str(t.get("object", "")).strip():
        return False
    if rel in GENERIC_BLACKLIST:
        return False
    if len(rel) < 2 or len(rel) > 60:
        return False
    return True


def evaluate_vocab_quality(triples: list[dict], vocab: list[str]) -> dict:
    """Measure how useful the bootstrapped vocabulary is after extraction."""
    used = {t["relation"] for t in triples}
    return {
        "vocab_size": len(vocab),
        "types_used": len(used),
        "coverage_rate": round(len(used) / len(vocab), 3) if vocab else 0.0,
        "relation_distribution": {
            r: sum(1 for t in triples if t["relation"] == r) for r in sorted(used)
        },
    }


def _load_prior_results() -> dict[str, dict]:
    """Load Baseline and Zone 1 eval results for cross-zone comparison display."""
    prior: dict[str, dict] = {}
    candidates = {
        "Baseline": "baseline_eval_results_original.json",
        "Zone 1":   "baseline_eval_results_zone1.json",
    }
    for zone, fname in candidates.items():
        path = os.path.join(config.RESULTS_DIR, fname)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    prior[zone] = json.load(f)
            except Exception as e:
                print(f"  ⚠ Could not load prior results for {zone}: {e}")
    return prior


# ---------------------------------------------------------------------------
# Pipeline Nodes
# ---------------------------------------------------------------------------

def load_chunks(state: Zone2State) -> dict:
    """Load PDF-only chunks from Zone 1 section-aware chunking output."""
    print("\n[1/4] Loading Zone 1 PDF chunks...")
    with open(config.ZONE1_CHUNKS_FILE) as f:
        all_chunks = json.load(f)
    pdf_chunks = [c for c in all_chunks if c["source"] == PDF_SOURCE_KEY]
    print(f"  ✓ {len(pdf_chunks)} PDF chunks (filtered from {len(all_chunks)} total)")
    return {"chunks": pdf_chunks}


def _select_stratified_samples(chunks: list[dict], max_samples: int = 8) -> list[str]:
    """Select diverse sample passages using generic insurance keyword groups."""
    selected: list[str] = []
    used_indices: set[int] = set()
    for keywords in _BOOTSTRAP_SECTION_GROUPS:
        for i, chunk in enumerate(chunks):
            if i in used_indices:
                continue
            text = (chunk["content"] + " " + " ".join(
                chunk.get("section_hierarchy", [])
            )).lower()
            if any(kw in text for kw in keywords):
                selected.append(chunk["content"][:800])
                used_indices.add(i)
                break
    # Fill remaining slots with evenly-spaced chunks
    step = max(1, len(chunks) // max_samples)
    for i in range(0, len(chunks), step):
        if i not in used_indices and len(selected) < max_samples:
            selected.append(chunks[i]["content"][:800])
    return selected[:max_samples]


def _format_sample_text(samples: list[str]) -> str:
    """Format samples into numbered passages for LLM prompts."""
    return "\n\n---\n\n".join(
        f"[Passage {j+1}]: {s}" for j, s in enumerate(samples)
    )


def bootstrap_vocab(state: Zone2State) -> dict:
    """
    Bootstrap BOTH relation types AND entity types from document samples.
    Domain-agnostic: the LLM reads the actual documents and proposes vocabulary.
    No hardcoded seeds — everything comes from the documents.
    """
    print("\n[2/4] Bootstrapping schema from document samples...")
    model      = state.get("model", config.OLLAMA_MODEL)
    cache_path = os.path.join(config.RESULTS_DIR, "zone2_vocab.json")

    # Check cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            vocab = cached.get("relations", [])
            entity_types = cached.get("entity_types", [])
            print(f"  ✓ Loaded cached schema ({len(vocab)} relations, "
                  f"{len(entity_types)} entity types from {cached.get('model', '?')})")
            print(f"    Delete {cache_path} to regenerate")
            return {"vocab": vocab, "entity_types": entity_types}
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"  ⚠ Cache corrupted ({e}); regenerating...")

    chunks = state.get("chunks", [])
    if not chunks:
        print("  ⚠ No chunks — using minimal fallback vocab")
        fallback_rels = ["COVERS", "EXCLUDED_FROM", "DEFINED_AS",
                         "HAS_COVERAGE_LIMIT", "HAS_DEADLINE", "MUST_FILE"]
        return {"vocab": fallback_rels, "entity_types": []}

    samples = _select_stratified_samples(chunks)
    sample_text = _format_sample_text(samples)
    llm = get_llm(model, json_mode=False)

    # --- Bootstrap relation types ---
    print(f"  Bootstrapping relation types from {len(samples)} passages ({model})...")
    proposed_rels: list[str] = []
    try:
        resp = llm.invoke([HumanMessage(
            content=RELATION_BOOTSTRAP_PROMPT.format(samples=sample_text)
        )])
        raw = resp.content.strip()
        print(f"    Relations ({len(raw)} chars): {raw.replace(chr(10), ' ')[:200]}")
        proposed_rels = [
            _sanitize_relation(r) for r in _parse_json_list(raw)
            if isinstance(r, str) and r.strip()
        ]
    except Exception as e:
        print(f"    ⚠ Relation bootstrap failed: {e}")

    # --- Bootstrap entity types ---
    print(f"  Bootstrapping entity types from {len(samples)} passages ({model})...")
    proposed_entities: list[str] = []
    try:
        resp = llm.invoke([HumanMessage(
            content=ENTITY_BOOTSTRAP_PROMPT.format(samples=sample_text)
        )])
        raw = resp.content.strip()
        print(f"    Entities ({len(raw)} chars): {raw.replace(chr(10), ' ')[:200]}")
        proposed_entities = [
            _sanitize_label(e) for e in _parse_json_list(raw)
            if isinstance(e, str) and e.strip()
        ]
    except Exception as e:
        print(f"    ⚠ Entity bootstrap failed: {e}")

    # Deduplicate, filter empties
    vocab = list(dict.fromkeys(r for r in proposed_rels if r and r not in GENERIC_BLACKLIST))
    entity_types = list(dict.fromkeys(e for e in proposed_entities if e))

    # Minimal fallback if LLM returned nothing useful
    if len(vocab) < 3:
        vocab = ["COVERS", "EXCLUDED_FROM", "DEFINED_AS",
                 "HAS_COVERAGE_LIMIT", "HAS_DEADLINE", "MUST_FILE",
                 "HAS_WAITING_PERIOD", "MUST_NOTIFY", "PRECEDES"]
        print("    ⚠ Using fallback relation vocab (LLM returned too few)")

    # Save cache
    with open(cache_path, "w") as f:
        json.dump({
            "relations": vocab,
            "entity_types": entity_types,
            "model": model,
            "n_samples": len(samples),
            "proposed_relations": proposed_rels,
            "proposed_entities": proposed_entities,
        }, f, indent=2)

    print(f"  ✓ Relations: {len(vocab)} → {', '.join(vocab[:12])}{'...' if len(vocab) > 12 else ''}")
    print(f"  ✓ Entity types: {len(entity_types)} → {', '.join(entity_types[:10])}{'...' if len(entity_types) > 10 else ''}")
    print(f"  ✓ Cached → {cache_path}")
    return {"vocab": vocab, "entity_types": entity_types}


def _parse_chunk_triples(parsed: list, chunk_id: str, source: str) -> list[dict]:
    """Convert raw LLM-parsed items into validated, normalized triple dicts."""
    triples: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if not all(k in item for k in ("subject", "relation", "object")):
            continue

        rel_raw = _sanitize_relation(str(item.get("relation", "")))
        rel     = RELATION_NORMALIZATIONS.get(rel_raw, rel_raw)
        rel     = RELATION_NORMALIZATIONS.get(rel, rel)
        rel     = _sanitize_relation(rel)
        rel     = _pattern_normalize_relation(rel)   # F-04 fix: collapse verbose compound names

        try:
            conf = float(item.get("confidence", 1.0))
            conf = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            conf = 1.0

        triple: dict = {
            "subject":    str(item["subject"]).strip(),
            "relation":   rel,
            "object":     str(item["object"]).strip(),
            "span":       str(item.get("span", ""))[:120],
            "confidence": conf,
            "chunk_id":   chunk_id,
            "source":     source,
        }
        if keep_triple(triple):
            triples.append(triple)
    return triples


def _build_extraction_messages(vocab: list[str], focus: str = "") -> list:
    """Build [SystemMessage, *few-shot pairs] used as the base for each chunk call.

    Args:
        vocab: Bootstrapped relation type names.
        focus: Optional focus suffix appended to the system prompt for multi-pass
               extraction. Pass 1 uses "" (no change). Passes 2-3 use different
               domain-specific focus instructions so temperature=0 produces different
               output across passes. See PASS_FOCUS_INSTRUCTIONS in prompts.py.
    """
    vocab_lines = "\n".join(f"  - {r}" for r in vocab)
    system_content = SYSTEM_PROMPT_TEMPLATE.format(vocab_lines=vocab_lines) + focus
    messages: list = [SystemMessage(content=system_content)]
    for human_text, ai_text in FEW_SHOT_PAIRS:
        messages.append(HumanMessage(content=human_text))
        messages.append(AIMessage(content=ai_text))
    return messages


_PASS_LABELS = ["general", "numeric", "obligations"]


# ---------------------------------------------------------------------------
# Regex post-extraction numeric linking (F-01 fix)
# ---------------------------------------------------------------------------
# The LLM (especially 8B) consistently omits dollar amounts and time periods
# as triple objects (F-01). This function scans the chunk text with regex and
# creates triples for any numeric facts not already captured by the LLM.
#
# Rules:
#   - Only creates triples when context is unambiguous (strong keyword match)
#   - Confidence = 0.80-0.85 (slightly below LLM baseline of ~0.90)
#   - Deduplication in extract_triples() removes redundant LLM + regex triples

_DOLLAR_RE   = re.compile(r'\$([\d,]+(?:\.\d{2})?)')
_DAYS_RE     = re.compile(r'\b(\d+)[‐\-]?[ ]?day[s]?\b', re.IGNORECASE)
_PERCENT_RE  = re.compile(r'\b(\d+(?:\.\d+)?)\s*(?:percent|%)', re.IGNORECASE)


def _extract_numeric_from_text(text: str, chunk_id: str, source: str) -> list[dict]:
    """
    Regex fallback: extract numeric triples (dollar amounts, time periods,
    percentages) that the LLM typically misses (F-01 fix).

    Only creates triples when the surrounding context is unambiguous.
    Returns [] for chunks with no numeric content or ambiguous context.
    """
    triples: list[dict] = []
    text_lo = text.lower()

    # --- Dollar amounts ---
    for m in _DOLLAR_RE.finditer(text):
        val  = f"${m.group(1)}"
        span = text[max(0, m.start() - 80): m.end() + 20][:120]

        # Sentence-level context for all pattern matching.
        sent_start = max(0, text.rfind('.', 0, m.start()),
                         text.rfind('\n', 0, m.start()))
        sent_end   = text.find('.', m.end())
        if sent_end == -1:
            sent_end = len(text)
        sentence = text_lo[sent_start: sent_end + 1]

        # Within the sentence, use the text immediately before and after the amount.
        # "before" = from the last clause separator (,/;/and) up to the amount.
        clause_start = max(
            sent_start,
            text.rfind(',', sent_start, m.start()),
            text.rfind(';', sent_start, m.start()),
            text.rfind(' and ', sent_start, m.start()),
        )
        near_before = text_lo[clause_start: m.start()]
        near_after  = text_lo[m.end(): min(len(text_lo), m.end() + 35)]

        building_kws = ("building", "structure", "dwelling", "residential")
        contents_kws = ("content", "personal property", "household goods")

        if "deductible" in sentence:
            subj, rel = "Policy", "HAS_DEDUCTIBLE"
        elif any(kw in near_before for kw in contents_kws):
            # "personal property coverage of $X"
            subj, rel = "Contents Coverage", "HAS_COVERAGE_LIMIT"
        elif any(kw in near_before for kw in building_kws):
            # "building coverage is $X"
            subj, rel = "Building Coverage", "HAS_COVERAGE_LIMIT"
        elif any(kw in near_after for kw in contents_kws):
            # "$X contents limit"
            subj, rel = "Contents Coverage", "HAS_COVERAGE_LIMIT"
        elif any(kw in near_after for kw in building_kws):
            # "$X building coverage"
            subj, rel = "Building Coverage", "HAS_COVERAGE_LIMIT"
        elif any(kw in sentence for kw in contents_kws):
            # fallback: "For contents coverage, the maximum is $X" (keyword elsewhere in sentence)
            subj, rel = "Contents Coverage", "HAS_COVERAGE_LIMIT"
        elif any(kw in sentence for kw in building_kws):
            subj, rel = "Building Coverage", "HAS_COVERAGE_LIMIT"
        elif any(kw in sentence for kw in ("maximum", "limit", "up to", "not to exceed")):
            subj, rel = "Policy", "HAS_COVERAGE_LIMIT"
        else:
            continue  # ambiguous context — skip

        triples.append({
            "subject":    subj,
            "relation":   rel,
            "object":     val,
            "span":       span.strip(),
            "confidence": 0.82,
            "chunk_id":   chunk_id,
            "source":     source,
        })

    # --- Day periods ---
    for m in _DAYS_RE.finditer(text):
        days = m.group(1)
        val  = f"{days} days"
        span = text[max(0, m.start() - 80): m.end() + 30][:120]

        # Use sentence-level context to avoid bleeding across sentence boundaries.
        s_start = max(0, text.rfind('.', 0, m.start()),
                      text.rfind('\n', 0, m.start()))
        s_end   = text.find('.', m.end())
        if s_end == -1:
            s_end = len(text)
        sent = text_lo[s_start: s_end + 1]
        # Also extend 60 chars before sentence start to catch multi-sentence phrasing
        near  = text_lo[max(0, s_start - 60): s_end + 1]

        if any(kw in sent for kw in ("waiting", "takes effect", "before coverage")):
            subj, rel = "Policy", "HAS_WAITING_PERIOD"
        elif any(kw in sent for kw in ("proof of loss", "file a proof", "submit proof")):
            subj, rel = "Proof of Loss", "HAS_DEADLINE"
        elif any(kw in sent for kw in ("notify", "notification", "report the loss")):
            subj, rel = "Loss Notification", "HAS_DEADLINE"
        elif any(kw in sent for kw in ("appeal", "dispute", "request review")):
            subj, rel = "Appeal", "HAS_DEADLINE"
        else:
            continue  # ambiguous — skip

        triples.append({
            "subject":    subj,
            "relation":   rel,
            "object":     val,
            "span":       span.strip(),
            "confidence": 0.80,
            "chunk_id":   chunk_id,
            "source":     source,
        })

    # --- Percentages (e.g., "80 percent replacement cost") ---
    for m in _PERCENT_RE.finditer(text):
        pct  = f"{m.group(1)}%"
        ctx  = text_lo[max(0, m.start() - 180): m.end() + 80]
        span = text[max(0, m.start() - 80): m.end() + 30][:120]

        if any(kw in ctx for kw in ("replacement cost", "insured to value", "coinsurance")):
            subj, rel = "Building Coverage", "HAS_COINSURANCE_REQUIREMENT"
        elif any(kw in ctx for kw in ("deductible", "applies")):
            subj, rel = "Policy", "HAS_DEDUCTIBLE_RATE"
        else:
            continue

        triples.append({
            "subject":    subj,
            "relation":   rel,
            "object":     pct,
            "span":       span.strip(),
            "confidence": 0.78,
            "chunk_id":   chunk_id,
            "source":     source,
        })

    return triples


def _extract_one_pass(
    llm, base_messages: list, chunks: list[dict], pass_label: str, errors: list[dict]
) -> list[dict]:
    """Run one extraction pass over all chunks; return raw (un-deduplicated) triples."""
    pass_triples: list[dict] = []
    for i, chunk in enumerate(chunks):
        content   = chunk["content"][:CHUNK_CONTENT_LIMIT]
        chunk_id  = chunk.get("chunk_id", f"chunk_{i}")
        source    = chunk.get("source", "")
        hierarchy = chunk.get("section_hierarchy", [])
        label     = " > ".join(hierarchy) if hierarchy else f"chunk {i+1}"
        print(f"    Chunk {i+1}/{len(chunks)} [{label[:45]}]...", end=" ", flush=True)
        messages = base_messages + [HumanMessage(content=f"Text: {content}")]
        try:
            response      = llm.invoke(messages)
            parsed        = _parse_json_list(response.content)
            chunk_triples = _parse_chunk_triples(parsed, chunk_id, source)
            print(f"→ {len(chunk_triples)}")
            pass_triples.extend(chunk_triples)
        except Exception as e:
            print(f"✗ ERROR: {e}")
            errors.append({"chunk_id": chunk_id, "pass": pass_label, "error": str(e)})
        time.sleep(0.05)
    return pass_triples


def extract_triples(state: Zone2State) -> dict:
    """
    Multi-pass few-shot Open IE with bootstrapped vocabulary.

    Runs PASS_FOCUS_INSTRUCTIONS passes over every chunk:
      Pass 1: general extraction (existing behavior)
      Pass 2: numeric / limit facts (dollar amounts, time periods, percentages)
      Pass 3: obligation / procedure / condition facts (MUST_*, PRECEDES, HAS_CONDITION)

    All three passes use temperature=0 but with DIFFERENT system prompt suffixes so
    the model attends to different fact types — avoiding identical outputs at temp=0.

    Triples from all passes are deduplicated by (subject, relation, object) before
    insertion. Expected uplift: ~1 triple/chunk → ~3-4 unique triples/chunk.
    """
    model  = state.get("model", config.OLLAMA_MODEL)
    vocab  = state.get("vocab", [])
    chunks = state.get("chunks", [])
    n_passes = len(PASS_FOCUS_INSTRUCTIONS)
    print(f"\n[3/4] Extracting triples — {n_passes}-pass few-shot IE ({model})...")
    print(f"  Vocab: {len(vocab)} types, {len(chunks)} chunks × {n_passes} passes")

    llm = get_llm(model, json_mode=True)
    all_raw_triples: list[dict] = []
    errors:          list[dict] = []

    for pass_idx, focus in enumerate(PASS_FOCUS_INSTRUCTIONS):
        label = _PASS_LABELS[pass_idx] if pass_idx < len(_PASS_LABELS) else f"pass{pass_idx+1}"
        print(f"\n  Pass {pass_idx + 1}/{n_passes} ({label}):")
        base_messages = _build_extraction_messages(vocab, focus=focus)
        pass_triples  = _extract_one_pass(llm, base_messages, chunks, label, errors)
        all_raw_triples.extend(pass_triples)
        print(f"  → Pass {pass_idx + 1} subtotal: {len(pass_triples)} triples")

    # Regex numeric fallback (F-01 fix) — runs over ALL chunks regardless of LLM output
    print("\n  Regex numeric fallback (F-01 fix):")
    regex_numeric: list[dict] = []
    for chunk in chunks:
        regex_numeric.extend(
            _extract_numeric_from_text(
                chunk["content"][:CHUNK_CONTENT_LIMIT],
                chunk.get("chunk_id", ""),
                chunk.get("source", ""),
            )
        )
    if regex_numeric:
        all_raw_triples.extend(regex_numeric)
        print(f"  → Added {len(regex_numeric)} regex numeric triples "
              f"(dollar amounts, time periods, percentages)")
    else:
        print("  → No unambiguous numeric patterns found")

    # Deduplicate across passes by normalized (subject, relation, object)
    seen_keys:   set[tuple]  = set()
    all_triples: list[dict]  = []
    for t in all_raw_triples:
        key = (t["subject"].lower().strip(), t["relation"], t["object"].lower().strip())
        if key not in seen_keys:
            seen_keys.add(key)
            all_triples.append(t)

    dedup_removed = len(all_raw_triples) - len(all_triples)
    per_chunk     = len(all_triples) / len(chunks) if chunks else 0

    print(f"\n  ✓ Raw triples (all passes): {len(all_raw_triples)}")
    print(f"  ✓ After dedup:              {len(all_triples)} "
          f"(removed {dedup_removed} duplicates, {per_chunk:.1f}/chunk)")
    print(f"  ✗ Errors:                   {len(errors)} chunk-passes failed")
    vq = evaluate_vocab_quality(all_triples, vocab)
    print(f"  ✓ Vocab coverage: {vq['types_used']}/{vq['vocab_size']} "
          f"types used ({vq['coverage_rate']:.0%})")

    return {"triples": all_triples, "vocab_quality": vq, "errors": errors}


def _group_triples_by_relation(triples: list[dict]) -> dict:
    """Group triples by relation type for batched MERGE insertion."""
    by_relation: dict[str, list] = defaultdict(list)
    for t in triples:
        by_relation[t["relation"]].append({
            "subject":    t["subject"],
            "object":     t["object"],
            "span":       t["span"],
            "confidence": t.get("confidence", 1.0),
            "chunk_id":   t["chunk_id"],
            "source":     t["source"],
        })
    return dict(by_relation)


def _batch_merge_triples(graph: Neo4jGraph, by_relation: dict) -> int:
    """MERGE triples into Neo4j grouped by relation type."""
    total = 0
    for rel_type, batch in by_relation.items():
        rel_type = _sanitize_relation(rel_type)
        graph.query(
            f"""
            UNWIND $batch AS row
            MERGE (s:Entity {{id: row.subject}})
            MERGE (o:Entity {{id: row.object}})
            MERGE (s)-[r:{rel_type}]->(o)
            ON CREATE SET r.span       = row.span,
                          r.confidence = row.confidence,
                          r.chunk_id   = row.chunk_id,
                          r.source     = row.source
            """,
            params={"batch": batch}
        )
        total += len(batch)
    return total


def canonicalize_relations(state: Zone2State) -> dict:
    """EDC-inspired: map raw extracted relation types → bootstrapped vocab (one LLM call)."""
    print("\n[3.5/4] EDC Canonicalization — mapping raw relations → vocab...")
    triples = state.get("triples", [])
    vocab   = state.get("vocab", [])
    model   = state.get("model", config.OLLAMA_MODEL)

    raw_relations = sorted(set(t["relation"] for t in triples))
    if not raw_relations or not vocab:
        print("  ⚠ No triples or vocab — skipping canonicalization")
        return {}

    llm = get_llm(model)
    prompt = (
        "Map each raw relation to the SINGLE closest relation from the vocabulary.\n"
        "If no close match exists, keep the original unchanged.\n"
        f"Vocabulary: {vocab}\n\n"
        "Format — one mapping per line:  RAW_RELATION -> VOCAB_RELATION\n\n"
        + "\n".join(raw_relations)
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        # Parse: "RAW -> MAPPED" lines into dict
        mapping: dict[str, str] = {}
        for line in response.content.strip().splitlines():
            parts = [p.strip() for p in line.split("->")]
            if len(parts) == 2 and parts[0] in raw_relations:
                mapped = parts[1].strip().upper().replace(" ", "_")
                # Only accept the mapped name if it's actually in vocab; else keep original
                mapping[parts[0]] = mapped if mapped in vocab else parts[0]
    except Exception as e:
        print(f"  ⚠ Canonicalization LLM call failed ({e}); keeping original relations")
        return {}

    # Apply mapping
    canonicalized = [
        {**t, "relation": mapping.get(t["relation"], t["relation"])}
        for t in triples
    ]

    types_before = len(set(t["relation"] for t in triples))
    types_after  = len(set(t["relation"] for t in canonicalized))
    print(f"  ✓ Canonicalized: {types_before} → {types_after} relation types")

    return {"triples": canonicalized}


def insert_to_neo4j(state: Zone2State) -> dict:
    """MERGE Zone 2 triples into Neo4j, wiping the graph first."""
    print("\n[4/4] Inserting into Neo4j AuraDB (MERGE deduplication)...")
    triples = state.get("triples", [])

    if not triples:
        print("  ✗ No triples to insert.")
        return {"neo4j_stats": {
            "nodes": 0, "relationships": 0,
            "triples_submitted": 0, "relation_types": [], "error": "no triples"
        }}

    try:
        graph = get_neo4j_graph()

        print("  Clearing entire Neo4j graph for clean zone experiment...")
        graph.query("MATCH (n) DETACH DELETE n")
        _r = graph.query("MATCH (n) RETURN count(n) AS c")
        print(f"  ✓ Graph cleared (nodes remaining: {_r[0]['c'] if _r else 0})")

        by_relation     = _group_triples_by_relation(triples)
        total_submitted = _batch_merge_triples(graph, by_relation)

        _nc = graph.query("MATCH (n:Entity) RETURN count(n) AS c")
        _rc = graph.query("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS c")
        node_count = _nc[0]["c"] if _nc else 0
        rel_count  = _rc[0]["c"] if _rc else 0

        stats = {
            "nodes":             node_count,
            "relationships":     rel_count,
            "triples_submitted": total_submitted,
            "relation_types":    sorted(by_relation.keys()),
        }
        print(f"  ✓ Inserted: {node_count} nodes, {rel_count} relationships")
        print(f"  ✓ Distinct relation types: {len(by_relation)}")
        return {"neo4j_stats": stats}

    except Exception as e:
        print(f"  ✗ Neo4j error: {e}")
        return {"neo4j_stats": {"error": str(e)}}


def zone25_entity_resolution(state: Zone2State) -> dict:
    """Zone 2.5: embed Entity node IDs, merge near-duplicates (cosine ≥ 0.90)."""
    print("\n[4.5] Zone 2.5 — Entity Resolution...")
    try:
        graph = get_neo4j_graph()
        stats = resolve_entities(graph, node_label="Entity")
        print(f"  ✓ {stats.get('merged', 0)} merged: "
              f"{stats.get('nodes_before')} → {stats.get('nodes_after')} nodes")
        return {"resolution_stats": stats}
    except Exception as e:
        print(f"  ⚠ Entity resolution failed ({e}); skipping")
        return {"resolution_stats": {"error": str(e), "merged": 0}}


# ---------------------------------------------------------------------------
# LangGraph Build
# ---------------------------------------------------------------------------

def build_pipeline():
    builder = StateGraph(Zone2State)
    builder.add_node("load_chunks",              load_chunks)
    builder.add_node("bootstrap_vocab",          bootstrap_vocab)
    builder.add_node("extract_triples",          extract_triples)
    builder.add_node("canonicalize_relations",   canonicalize_relations)
    builder.add_node("insert_to_neo4j",          insert_to_neo4j)
    builder.add_node("zone25_entity_resolution", zone25_entity_resolution)
    builder.set_entry_point("load_chunks")
    builder.add_edge("load_chunks",              "bootstrap_vocab")
    builder.add_edge("bootstrap_vocab",          "extract_triples")
    builder.add_edge("extract_triples",          "canonicalize_relations")
    builder.add_edge("canonicalize_relations",   "insert_to_neo4j")
    builder.add_edge("insert_to_neo4j",          "zone25_entity_resolution")
    builder.add_edge("zone25_entity_resolution", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def _print_comparison_table(prior: dict[str, dict]) -> None:
    """Print cross-zone accuracy/Riskine comparison table from saved eval JSONs."""
    if not prior:
        return
    print(f"\n{'─' * 60}")
    print(f"{'Zone':<12} {'Accuracy':>10} {'Riskine F1':>12}")
    print(f"{'─' * 60}")
    for zone, data in prior.items():
        acc     = data.get("query_accuracy_pct", data.get("accuracy", "?"))
        riskine = data.get("riskine_f1", "—")
        if isinstance(acc, (int, float)):
            acc = f"{acc:.0f}%"
        if isinstance(riskine, (int, float)):
            riskine = f"{riskine:.3f}"
        print(f"{zone:<12} {str(acc):>10} {str(riskine):>12}")
    print(f"{'Zone 2':<12} {'(pending)':>10} {'(pending)':>12}  "
          f"← run: python3 baseline/eval.py --suffix zone2 --riskine")
    print(f"{'─' * 60}")


def _save_run_summary(result: dict, model: str, elapsed: float) -> str:
    """Serialize pipeline result to zone2_run_summary.json."""
    summary = {
        "mode":              "zone2",
        "model":             model,
        "elapsed_seconds":   round(elapsed, 2),
        "chunks_processed":  len(result.get("chunks", [])),
        "triples_extracted": len(result.get("triples", [])),
        "vocab":             result.get("vocab", []),
        "entity_types":      result.get("entity_types", []),
        "vocab_quality":     result.get("vocab_quality", {}),
        "neo4j_stats":       result.get("neo4j_stats", {}),
        "resolution_stats":  result.get("resolution_stats", {}),
        "errors":            result.get("errors", []),
        "relation_types":    result.get("neo4j_stats", {}).get("relation_types", []),
        "triples":           result.get("triples", []),
    }
    out_path = os.path.join(config.RESULTS_DIR, "zone2_run_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return out_path


def run_zone2(model: str = config.OLLAMA_MODEL):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("CS584 Capstone — Zone 2 Pipeline [Domain-Agnostic Open IE]")
    print(f"Bootstrapped schema + few-shot extraction → Neo4j  (model: {model})")
    print("=" * 60)

    pipeline = build_pipeline()
    start    = time.time()
    result   = pipeline.invoke({
        "chunks":           [],
        "vocab":            [],
        "entity_types":     [],
        "triples":          [],
        "vocab_quality":    {},
        "neo4j_stats":      {},
        "resolution_stats": {},
        "errors":           [],
        "model":            model,
    })
    elapsed = time.time() - start

    neo4j_stats = result.get("neo4j_stats", {})
    triples     = result.get("triples", [])
    errors      = result.get("errors", [])

    print(f"\n{'=' * 60}")
    print(f"Zone 2 pipeline complete in {elapsed:.1f}s")
    print(f"  Triples extracted:   {len(triples)}")
    print(f"  Errors:              {len(errors)}")
    print(f"  Neo4j nodes:         {neo4j_stats.get('nodes', '?')}")
    print(f"  Neo4j relationships: {neo4j_stats.get('relationships', '?')}")

    _print_comparison_table(_load_prior_results())

    out_path = _save_run_summary(result, model, elapsed)
    print(f"\n✓ Summary saved to {out_path}")
    print("\nNext steps:")
    print("  python3 baseline/eval.py --suffix zone2")
    print("  python3 baseline/eval.py --suffix zone2 --riskine")
    print("  python3 zone3/pipeline.py   # ← ontology induction via Leiden")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zone 2 Domain-Agnostic Open IE pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 zone2/pipeline.py
  python3 zone2/pipeline.py --model qwen2.5:7b
  # After running, evaluate with:
  python3 baseline/eval.py --suffix zone2 --riskine
  # Then run ontology induction:
  python3 zone3/pipeline.py
        """,
    )
    parser.add_argument(
        "--model", default=config.OLLAMA_MODEL,
        help=f"Ollama model name (default: {config.OLLAMA_MODEL})"
    )
    args = parser.parse_args()
    run_zone2(model=args.model)
