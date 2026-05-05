"""
Zone 2 Pipeline — Domain-Agnostic Open IE with LLM-Bootstrapped Schema

General-purpose insurance ontology extraction pipeline. No hardcoded domain
knowledge — all entity types and relation types are bootstrapped from the
documents themselves. Works on any insurance Line of Business (flood, auto,
health, liability) without code changes.

Architecture (LangGraph pipeline):
  load_chunks → detect_lobs → extract_structured → bootstrap_vocab →
  extract_triples → canonicalize_relations → normalize_relations →
  zone25_entity_resolution → cross_source_link → insert_to_neo4j → END

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

from __future__ import annotations

import json
import re
import time
import os
import sys
import argparse
from typing import TypedDict, Annotated, Optional
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
    SINGLE_PASS_FOCUS,
    RECALL_PASS_PROMPT,
    DECOMPOSITION_PROMPT,
    SINGLE_FACT_EXTRACTION_PROMPT,
)
from zone2.entity_resolution import resolve_entities_in_memory
from zone2.structured_mapper import extract_structured
from zone2.cross_source_linker import cross_source_link
from zone2.lob_detector import detect_lobs
from zone2.utils import sanitize_label, sanitize_relation


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class Zone2State(TypedDict):
    chunks:              list[dict]
    vocab:               list[str]                  # bootstrapped relation vocabulary
    entity_types:        list[str]                  # bootstrapped entity type vocabulary
    triples:             list[dict]                 # extracted + validated triples
    structured_triples:  list[dict]                 # SEAF-KG: deterministic CSV triples
    vocab_quality:       dict                       # quality metrics for the vocab
    neo4j_stats:         dict
    resolution_stats:    dict                       # Zone 2.5 entity resolution stats
    errors:              Annotated[list, operator.add]
    model:               str
    num_passes:          int
    chunks_file:         str                        # path to zone1_chunks.json
    results_dir:         str                        # output directory for results
    no_wipe:             bool                       # skip Neo4j graph clear


# ---------------------------------------------------------------------------
# Constants — ALL DOMAIN-AGNOSTIC
# ---------------------------------------------------------------------------


def _results_dir(state: Zone2State) -> str:
    """Return the results directory from state, falling back to config."""
    return state.get("results_dir") or config.RESULTS_DIR


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

def get_llm(model: str, json_mode: bool = False,
            num_predict: int = 4096) -> ChatOllama:
    """Return a ChatOllama instance, optionally with JSON output mode.

    Args:
        num_predict: Max output tokens. 4096 ≈ 50 triples max, prevents
                     runaway generation while keeping all important facts.
    """
    kwargs: dict = dict(
        model=model,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
        num_predict=num_predict,
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
    m = re.search(r'\[.*\]', text, re.DOTALL)
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


# ---------------------------------------------------------------------------
# Context-aware placeholder filtering (Priority 1)
# ---------------------------------------------------------------------------
# "Currently Unavailable" appears 2,000× as HAS_REPORTED_CITY target because
# FEMA redacts city names. A global blocklist is wrong — the same string
# could be meaningful in another relation context. Instead we check
# (relation_type, value) heuristically.

# Null-like patterns — ANCHORED matching (exact after normalization).
# Substring matching causes false positives ("Unknown Road", "N/A Holdings").
_NULL_EXACT_PATTERNS = frozenset({
    "unavailable", "unknown", "n/a", "na", "not available",
    "not applicable", "none", "null", "redacted",
    "undetermined", "unspecified", "pending", "missing",
    "not provided", "tbd", "unk", "-", "--", "*", ".",
    "currently unavailable", "not reported",
})

# Sentinel date/numeric patterns that indicate missing data.
_NULL_DATE_PATTERNS = frozenset({"0000-00-00", "9999-99-99", "99/99/9999"})
_NULL_NUMERIC_PATTERNS = frozenset({"0", "000000", "-1", "999999"})

# Relation datatype families — controls which null lexicon applies.
# Each family uses anchored exact matching on normalized values.
_TEXT_BEARING_HINTS = frozenset({
    "city", "county", "state", "zip", "address", "street",
    "country", "province", "region", "municipality", "town",
    "name", "policyholder", "claimant", "insured", "agent",
    "adjuster", "beneficiary", "contact", "description",
    "type", "code", "class", "category", "zone", "status",
    "occupancy", "construction", "valuation", "condition",
})
_DATE_BEARING_HINTS = frozenset({
    "date", "effective", "termination", "expiration",
    "loss", "construction", "cancellation", "original",
})
_AMOUNT_BEARING_HINTS = frozenset({
    "amount", "cost", "coverage", "premium", "deductible",
    "limit", "payment", "fee", "value", "rate",
})

# Prevalence threshold: a (rel, val) pair is suspicious if it accounts
# for more than this fraction of a relation's total triples.
_PREVALENCE_THRESHOLD = 0.20
# Absolute floor — don't flag low-count pairs even if prevalence is high.
_MIN_ABSOLUTE_COUNT = 50


def _normalize_for_null_check(value: str) -> str:
    """Normalize value for null-pattern matching: lowercase, strip, collapse punctuation."""
    return value.strip().lower().strip(".-*/ ")


def _is_placeholder_value(relation: str, value: str) -> bool:
    """Detect placeholder values using (relation, value) context.

    Rules (domain-agnostic, anchored matching):
      1. Empty / whitespace-only → always filter
      2. Short alphanumeric codes (≤3 chars) → never filter (FEMA codes, zips)
      3. Text-bearing relations: exact null-pattern match → filter
      4. Date-bearing relations: sentinel date patterns → filter
      5. Amount-bearing relations: sentinel numeric patterns → filter
    """
    stripped = value.strip()

    # Rule 1: empty or whitespace
    if not stripped:
        return True

    # Rule 2: short codes are often meaningful (FEMA codes, zips)
    # But check for null sentinels before skipping.
    if len(stripped) <= 3:
        norm = _normalize_for_null_check(stripped)
        # Null sentinel tokens: "-", "na", "*", empty-after-strip
        if norm in _NULL_EXACT_PATTERNS or norm == "":
            return True
        # Also check type-specific sentinels for short values.
        rel_lower = relation.lower()
        if any(h in rel_lower for h in _DATE_BEARING_HINTS) and norm in _NULL_DATE_PATTERNS:
            return True
        if any(h in rel_lower for h in _AMOUNT_BEARING_HINTS) and norm in _NULL_NUMERIC_PATTERNS:
            return True
        # Short alphanumeric code — likely meaningful
        if stripped.replace(".", "").replace("-", "").isalnum():
            return False

    normalized = _normalize_for_null_check(stripped)
    rel_lower = relation.lower()

    # Rule 3: text-bearing relation + exact null match
    if any(hint in rel_lower for hint in _TEXT_BEARING_HINTS):
        if normalized in _NULL_EXACT_PATTERNS:
            return True

    # Rule 4: date-bearing relation + sentinel date
    if any(hint in rel_lower for hint in _DATE_BEARING_HINTS):
        if normalized in _NULL_DATE_PATTERNS:
            return True

    # Rule 5: amount-bearing relation + sentinel numeric
    if any(hint in rel_lower for hint in _AMOUNT_BEARING_HINTS):
        if normalized in _NULL_NUMERIC_PATTERNS:
            return True

    return False


def _filter_placeholder_triples(
    triples: list[dict],
) -> tuple[list[dict], dict[str, int]]:
    """Context-aware placeholder filtering on the full triple list.

    Two-pass approach:
      Pass 1: Detect high-prevalence (relation, value) pairs where:
              - The pair accounts for >20% of that relation's total triples
              - AND the pair appears ≥50 times (absolute floor)
              - AND the value matches a null-like pattern (anchored exact)
      Pass 2: Remove those triples + per-triple context checks.

    Returns (filtered_triples, removal_stats).
    """
    # Pass 1: Count per-relation totals and per-(relation, value) frequencies.
    rel_totals: dict[str, int] = defaultdict(int)
    rel_val_counts: dict[tuple[str, str], int] = defaultdict(int)
    for t in triples:
        rel_totals[t["relation"]] += 1
        rel_val_counts[(t["relation"], t["object"])] += 1

    # Identify high-prevalence placeholders.
    high_prevalence_placeholders: set[tuple[str, str]] = set()
    for (rel, val), count in rel_val_counts.items():
        if count < _MIN_ABSOLUTE_COUNT:
            continue
        prevalence = count / max(rel_totals[rel], 1)
        if prevalence >= _PREVALENCE_THRESHOLD:
            normalized = _normalize_for_null_check(val)
            if normalized in _NULL_EXACT_PATTERNS:
                high_prevalence_placeholders.add((rel, val))

    # Pass 2: Filter.
    removal_stats: dict[str, int] = defaultdict(int)
    filtered: list[dict] = []
    for t in triples:
        rel, val = t["relation"], t["object"]

        # Always remove empty/whitespace objects.
        if not val.strip():
            removal_stats["empty_value"] += 1
            continue

        # High-prevalence placeholder.
        if (rel, val) in high_prevalence_placeholders:
            removal_stats["high_prevalence_placeholder"] += 1
            continue

        # Per-triple context check (catches low-frequency placeholders too).
        if _is_placeholder_value(rel, val):
            removal_stats["context_placeholder"] += 1
            continue

        filtered.append(t)

    return filtered, dict(removal_stats)


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


def _load_prior_results(results_dir: str | None = None) -> dict[str, dict]:
    """Load Baseline and Zone 1 eval results for cross-zone comparison display."""
    rdir = results_dir or config.RESULTS_DIR
    prior: dict[str, dict] = {}
    candidates = {
        "Baseline": "baseline_eval_results_original.json",
        "Zone 1":   "baseline_eval_results_zone1.json",
    }
    for zone, fname in candidates.items():
        path = os.path.join(rdir, fname)
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
    """Load all chunks from Zone 1 (PDF + CSV sources)."""
    print("\n[1/4] Loading Zone 1 chunks...")
    chunks_file = state.get("chunks_file") or config.ZONE1_CHUNKS_FILE
    with open(chunks_file) as f:
        all_chunks = json.load(f)
    sources = sorted(set(c.get("source", "unknown") for c in all_chunks))
    print(f"  ✓ {len(all_chunks)} chunks from {len(sources)} sources: {sources}")
    return {"chunks": all_chunks}


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
    cache_path = os.path.join(_results_dir(state), "zone2_vocab.json")

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
            sanitize_label(e) for e in _parse_json_list(raw)
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


_PREAMBLE_SECTION_LIMIT = 100   # cap section-hierarchy section in preamble
_PREAMBLE_TOTAL_LIMIT = 280     # hard cap on total preamble length


def _build_chunk_preamble(chunk: dict) -> str:
    """Build a compact ``Source: ... | LOB: ... | Section: ...`` preamble.

    Used by both LLM extraction paths (main few-shot and decompose) so
    the model gets enough context to disambiguate cross-section meaning
    and apply LOB-aware extraction patterns without enlarging few-shot
    payloads.

    Returns at most :data:`_PREAMBLE_TOTAL_LIMIT` characters; ends with
    a single newline so downstream callers can prepend directly.
    """
    src_full = str(chunk.get("source", "")).strip()
    basename = src_full.replace("\\", "/").rsplit("/", 1)[-1]
    lob = chunk.get("lob") or "generic"
    hierarchy = chunk.get("section_hierarchy") or []

    parts: list[str] = []
    if basename:
        parts.append(f"Source: {basename}")
    parts.append(f"LOB: {lob}")
    if hierarchy:
        section = " > ".join(str(h) for h in hierarchy)
        section = section[:_PREAMBLE_SECTION_LIMIT]
        parts.append(f"Section: {section}")

    line = "  |  ".join(parts)
    if len(line) > _PREAMBLE_TOTAL_LIMIT:
        line = line[: _PREAMBLE_TOTAL_LIMIT - 1] + "…"
    return f"[Context] {line}\n"


def _parse_chunk_triples(
    parsed: list,
    chunk_id: str,
    source: str,
    lob: str = "generic",
) -> list[dict]:
    """Convert raw LLM-parsed items into validated, normalized triple dicts.

    Every emitted triple carries the chunk's ``lob`` so Zone 3 induction
    and chatbot filters can scope by Line of Business.
    """
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

        # Extract entity types if present (Phase A entity type propagation)
        subj_type = str(item.get("subject_type", "")).strip() or "Unknown"
        obj_type  = str(item.get("object_type", "")).strip() or "Unknown"

        triple: dict = {
            "subject":      str(item["subject"]).strip(),
            "subject_type": subj_type,
            "relation":    rel,
            "object":      str(item["object"]).strip(),
            "object_type": obj_type,
            "span":       str(item.get("span", ""))[:120],
            "confidence": conf,
            "chunk_id":   chunk_id,
            "source":     source,
            "lob":        lob or "generic",
        }
        if keep_triple(triple):
            triples.append(triple)
    return triples


def _build_extraction_messages(vocab: list[str], focus: str = "",
                               entity_types: "Optional[list[str]]" = None) -> list:
    """Build [SystemMessage, *few-shot pairs] used as the base for each chunk call.

    Args:
        vocab: Bootstrapped relation type names.
        focus: Optional focus suffix appended to the system prompt for multi-pass
               extraction. Pass 1 uses "" (no change). Passes 2-3 use different
               domain-specific focus instructions so temperature=0 produces different
               output across passes. See PASS_FOCUS_INSTRUCTIONS in prompts.py.
        entity_types: Bootstrapped entity type names (e.g. InsurancePolicy, CoverageType).
    """
    vocab_lines = "\n".join(f"  - {r}" for r in vocab)
    if entity_types:
        entity_type_lines = "\n".join(f"  - {t}" for t in entity_types)
    else:
        entity_type_lines = "  (no entity types available — use your best judgment)"
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        vocab_lines=vocab_lines,
        entity_type_lines=entity_type_lines,
    ) + focus
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


def _extract_subject_from_context(text: str, match_start: int) -> str:
    """Extract the most likely subject noun phrase from text near a numeric value.

    Domain-agnostic: looks for the nearest capitalized noun phrase or
    section header before the match position.
    """
    # Look backwards from match for a likely subject.
    before = text[max(0, match_start - 120):match_start]

    # Try section hierarchy (e.g., "Coverage A—Building" from chunk header).
    # Look for capitalized multi-word phrases.
    caps_phrases = re.findall(r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', before)
    if caps_phrases:
        # Take the last (closest to the number) capitalized phrase.
        return caps_phrases[-1].strip()

    # Try the nearest noun-like word before a colon or "is" or "of".
    for pattern in [r'(\w[\w\s]{2,30}):\s*$', r'(\w[\w\s]{2,30})\s+(?:is|of|for)\s']:
        m = re.search(pattern, before, re.IGNORECASE)
        if m:
            return m.group(1).strip().title()

    return ""


def _extract_numeric_from_text(
    text: str,
    chunk_id: str,
    source: str,
    lob: str = "generic",
) -> list[dict]:
    """
    Regex fallback: extract numeric triples (dollar amounts, time periods,
    percentages) that the LLM typically misses (F-01 fix).

    Domain-agnostic: derives subjects from surrounding text context
    instead of hardcoded entity names. Relations are pattern-based
    (HAS_COVERAGE_LIMIT, HAS_DEADLINE, etc.) using generic keywords.

    Only creates triples when the surrounding context is unambiguous.
    Returns [] for chunks with no numeric content or ambiguous context.

    Every emitted triple carries the chunk's ``lob`` so Zone 3 induction
    and chatbot filters can scope by Line of Business.
    """
    lob = lob or "generic"
    triples: list[dict] = []
    text_lo = text.lower()

    # Generic keyword groups (not tied to any specific insurance LOB).
    _DEDUCTIBLE_KWS = ("deductible",)
    _LIMIT_KWS = ("maximum", "limit", "up to", "not to exceed", "coverage")
    _WAITING_KWS = ("waiting", "takes effect", "before coverage", "effective after")
    _DEADLINE_KWS = ("proof of loss", "file", "submit", "deadline", "within")
    _NOTIFY_KWS = ("notify", "notification", "report", "notice")
    _APPEAL_KWS = ("appeal", "dispute", "request review", "appraisal")
    _COINSURANCE_KWS = ("replacement cost", "insured to value", "coinsurance")

    # --- Dollar amounts ---
    for m in _DOLLAR_RE.finditer(text):
        val = f"${m.group(1)}"
        span = text[max(0, m.start() - 80): m.end() + 20][:120]

        # Sentence-level context.
        sent_start = max(0, text.rfind('.', 0, m.start()),
                         text.rfind('\n', 0, m.start()))
        sent_end = text.find('.', m.end())
        if sent_end == -1:
            sent_end = len(text)
        sentence = text_lo[sent_start: sent_end + 1]

        # Determine relation from context keywords.
        if any(kw in sentence for kw in _DEDUCTIBLE_KWS):
            rel = "HAS_DEDUCTIBLE"
        elif any(kw in sentence for kw in _LIMIT_KWS):
            rel = "HAS_COVERAGE_LIMIT"
        else:
            continue  # ambiguous context — skip

        # Derive subject from surrounding text (domain-agnostic).
        subj = _extract_subject_from_context(text, m.start())
        if not subj:
            continue  # can't determine subject — skip

        triples.append({
            "subject":      subj,
            "subject_type": "Unknown",
            "relation":     rel,
            "object":       val,
            "object_type":  "Currency",
            "span":         span.strip(),
            "confidence":   0.82,
            "chunk_id":     chunk_id,
            "source":       source,
            "source_type":  "regex",
            "lob":          lob,
        })

    # --- Day periods ---
    for m in _DAYS_RE.finditer(text):
        days = m.group(1)
        val = f"{days} days"
        span = text[max(0, m.start() - 80): m.end() + 30][:120]

        s_start = max(0, text.rfind('.', 0, m.start()),
                      text.rfind('\n', 0, m.start()))
        s_end = text.find('.', m.end())
        if s_end == -1:
            s_end = len(text)
        sent = text_lo[s_start: s_end + 1]

        if any(kw in sent for kw in _WAITING_KWS):
            rel = "HAS_WAITING_PERIOD"
        elif any(kw in sent for kw in _DEADLINE_KWS):
            rel = "HAS_DEADLINE"
        elif any(kw in sent for kw in _NOTIFY_KWS):
            rel = "HAS_DEADLINE"
        elif any(kw in sent for kw in _APPEAL_KWS):
            rel = "HAS_DEADLINE"
        else:
            continue  # ambiguous — skip

        subj = _extract_subject_from_context(text, m.start())
        if not subj:
            continue

        triples.append({
            "subject":      subj,
            "subject_type": "Unknown",
            "relation":     rel,
            "object":       val,
            "object_type":  "Duration",
            "span":         span.strip(),
            "confidence":   0.80,
            "chunk_id":     chunk_id,
            "source":       source,
            "source_type":  "regex",
            "lob":          lob,
        })

    # --- Percentages ---
    for m in _PERCENT_RE.finditer(text):
        pct = f"{m.group(1)}%"
        ctx = text_lo[max(0, m.start() - 180): m.end() + 80]
        span = text[max(0, m.start() - 80): m.end() + 30][:120]

        if any(kw in ctx for kw in _COINSURANCE_KWS):
            rel = "HAS_COINSURANCE_REQUIREMENT"
        elif any(kw in ctx for kw in _DEDUCTIBLE_KWS):
            rel = "HAS_DEDUCTIBLE_RATE"
        else:
            continue

        subj = _extract_subject_from_context(text, m.start())
        if not subj:
            continue

        triples.append({
            "subject":      subj,
            "subject_type": "Unknown",
            "relation":     rel,
            "object":       pct,
            "object_type":  "Percentage",
            "span":         span.strip(),
            "confidence":   0.78,
            "chunk_id":     chunk_id,
            "source":       source,
            "source_type":  "regex",
            "lob":          lob,
        })

    return triples



# ---------------------------------------------------------------------------
# Post-extraction semantic validator (Priority 5)
# ---------------------------------------------------------------------------
# Lightweight domain/range checks for known confusion patterns.
# Rejects triples that are structurally valid but semantically wrong.
# Domain-agnostic: checks relation TYPES, not specific entity names.

# Relations that require a numeric/amount object (not an entity).
_AMOUNT_RELATIONS = frozenset({
    "HAS_DEDUCTIBLE", "HAS_COVERAGE_LIMIT", "HAS_MAXIMUM",
    "HAS_PREMIUM", "HAS_COST", "HAS_FEE", "HAS_PAYMENT",
})

# Relations that require an entity subject, not an amount.
_ENTITY_SUBJECT_RELATIONS = frozenset({
    "COVERS", "EXCLUDED_FROM", "ADMINISTERS", "MUST_NOTIFY",
    "MUST_FILE", "DEFINED_AS", "HAS_OPTION",
})

# Known confusion pairs: (wrong_relation, pattern) → should be different.
# These catch the most common LLM extraction errors.
_ASYMMETRIC_CHECKS: list[tuple[str, str, str]] = [
    # (relation, subject_type_pattern, reason)
    # An organization should ADMINISTER, not COVER
    ("COVERS", "Organization", "organizations administer programs, not cover risks"),
]


def _validate_triple_semantics(t: dict) -> tuple[bool, str]:
    """Validate a single triple for semantic correctness.

    Returns (is_valid, reason). Domain-agnostic checks only.
    """
    rel = t.get("relation", "")
    subj = t.get("subject", "")
    obj = t.get("object", "")
    subj_type = t.get("subject_type", "Unknown")
    obj_type = t.get("object_type", "Unknown")

    # Check 1: Amount relations should have numeric-like objects.
    if rel in _AMOUNT_RELATIONS:
        # Object should look like a number, currency, or percentage.
        obj_stripped = obj.strip()
        looks_numeric = bool(
            re.match(r'^[\$]?[\d,]+(\.\d+)?(%)?$', obj_stripped)
            or obj_type in ("Currency", "Numeric", "Percentage", "FinancialAmount")
            or re.match(r'^\d+\s*(days?|months?|years?|hours?)$', obj_stripped, re.I)
        )
        if not looks_numeric and len(obj_stripped) > 5:
            # Object is a long text string, likely an entity name → wrong
            return False, f"{rel} expects numeric object, got '{obj_stripped[:30]}'"

    # Check 2: Asymmetric relation checks.
    for check_rel, type_pattern, reason in _ASYMMETRIC_CHECKS:
        if rel == check_rel and type_pattern.lower() in subj_type.lower():
            return False, reason

    return True, ""


def _filter_semantic_errors(
    triples: list[dict],
) -> tuple[list[dict], dict[str, int]]:
    """Remove triples that fail semantic validation.

    Only filters LLM-extracted triples (not structured or cross-source).
    Returns (filtered_triples, error_stats).
    """
    error_stats: dict[str, int] = defaultdict(int)
    filtered: list[dict] = []

    for t in triples:
        # Don't validate structured/deterministic triples — they're correct by construction.
        if t.get("source_type") in ("structured", "cross_source"):
            filtered.append(t)
            continue

        is_valid, reason = _validate_triple_semantics(t)
        if is_valid:
            filtered.append(t)
        else:
            error_stats[reason] += 1

    return filtered, dict(error_stats)


MAX_TRIPLES_PER_CHUNK = 50   # Safety cap per chunk (num_predict handles most cases)


# ---------------------------------------------------------------------------
# Span-grounding verification (recall pass precision guard)
# ---------------------------------------------------------------------------

def _normalize_for_span_match(text: str) -> str:
    """Normalize text for span matching: lowercase, collapse whitespace, strip punctuation edges."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def _verify_span_grounding(
    triple: dict,
    chunk_text: str,
    min_overlap: float = 0.80,
) -> tuple[bool, float]:
    """Check if a triple's span is grounded in the source chunk.

    Returns (is_grounded, confidence_multiplier).
    - Exact substring match → (True, 1.0)
    - ≥80% character overlap → (True, 0.9)
    - No match → (False, 0.0)
    """
    span = triple.get("span", "").strip()
    if not span or len(span) < 5:
        return False, 0.0

    norm_span = _normalize_for_span_match(span)
    norm_chunk = _normalize_for_span_match(chunk_text)

    # Exact substring match.
    if norm_span in norm_chunk:
        return True, 1.0

    # Fuzzy: token-based overlap. Split span into words and check how many
    # appear in the chunk. More robust than character sliding window when
    # the LLM paraphrases slightly or drops articles.
    span_tokens = set(norm_span.split())
    if len(span_tokens) < 2:
        return False, 0.0

    chunk_tokens = set(norm_chunk.split())
    matched = span_tokens & chunk_tokens
    overlap = len(matched) / len(span_tokens)

    if overlap >= min_overlap:
        return True, 0.9

    return False, 0.0


# ---------------------------------------------------------------------------
# Decompose-then-Extract (CoDe-KG style, replaces failed recall pass)
# ---------------------------------------------------------------------------
# Root cause: json_mode + temp=0 produces 1-item JSON arrays for prose chunks.
# Fix: Stage 1 decomposes chunk into facts (free-form, no json_mode).
#      Stage 2 extracts 1 triple per fact (json_mode OK for single object).
# Only runs on prose chunks where Pass 1 extracted ≤2 triples.

_DECOMPOSE_THRESHOLD = 2  # Chunks with ≤ this many Pass 1 triples get decomposition.
_MIN_FACT_LENGTH = 10     # Reject decomposed facts shorter than this.
_MAX_FACTS_PER_CHUNK = 20 # Safety cap on facts from decomposition.


def _parse_decomposed_facts(text: str) -> list[str]:
    """Parse numbered facts from free-form LLM decomposition output.

    Handles formats like:
      1. Fact one here
      2. Fact two here
    or:
      1) Fact one
      2) Fact two
    """
    facts: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        # Strip numbering: "1. ", "1) ", "- ", "• "
        cleaned = re.sub(r'^(?:\d+[.)]\s*|[-•]\s*)', '', line).strip()
        if len(cleaned) >= _MIN_FACT_LENGTH:
            facts.append(cleaned)
    return facts[:_MAX_FACTS_PER_CHUNK]


def _decompose_then_extract(
    model: str,
    chunks: list[dict],
    pass1_triples: list[dict],
    vocab: list[str],
    errors: list[dict],
) -> list[dict]:
    """Decompose-then-extract for prose chunks that Pass 1 under-extracted.

    Stage 1: LLM (NO json_mode) lists all facts as numbered plain text.
    Stage 2: For each fact, LLM (json_mode) extracts 1 triple.
    Span-grounding verification filters hallucinations.

    Only processes chunks where Pass 1 extracted ≤ _DECOMPOSE_THRESHOLD triples.

    Args:
        model: Ollama model name.
        chunks: PDF chunks to process.
        pass1_triples: All triples from Pass 1 (for counting per-chunk).
        vocab: Bootstrapped relation vocabulary.
        errors: Error accumulator.

    Returns:
        List of new verified triples (source_type='decompose').
    """
    # Count Pass 1 triples per chunk_id.
    from collections import Counter
    p1_counts = Counter(t.get("chunk_id") for t in pass1_triples)

    # Identify prose chunks that need decomposition.
    prose_chunks = [
        c for c in chunks
        if p1_counts.get(c.get("chunk_id", ""), 0) <= _DECOMPOSE_THRESHOLD
    ]

    if not prose_chunks:
        print("    No prose chunks need decomposition (all extracted well)")
        return []

    print(f"    {len(prose_chunks)} prose chunks (≤{_DECOMPOSE_THRESHOLD} triples in Pass 1)")

    # Stage 1 LLM: free-form decomposition (NO json_mode).
    llm_freeform = get_llm(model, json_mode=False)
    # Stage 2 LLM: single-object extraction (json_mode OK).
    llm_json = get_llm(model, json_mode=True)

    vocab_lines = "\n".join(f"  - {r}" for r in vocab)

    all_new_triples: list[dict] = []
    total_facts = 0
    total_triples = 0
    total_rejected = 0

    for i, chunk in enumerate(prose_chunks):
        content = chunk["content"][:CHUNK_CONTENT_LIMIT]
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        source = chunk.get("source", "")
        hierarchy = chunk.get("section_hierarchy", [])
        label = " > ".join(hierarchy) if hierarchy else f"chunk {i+1}"
        p1_count = p1_counts.get(chunk_id, 0)

        print(f"    [{i+1}/{len(prose_chunks)}] {label[:50]} (P1={p1_count})...",
              end=" ", flush=True)

        # --- Stage 1: Decompose into facts ---
        # Prepend the source/LOB/section preamble so the LLM understands
        # what kind of document and section it is decomposing.
        preamble = _build_chunk_preamble(chunk)
        try:
            decomp_prompt = DECOMPOSITION_PROMPT.format(
                chunk_text=f"{preamble}\n{content}",
            )
            response = llm_freeform.invoke([HumanMessage(content=decomp_prompt)])
            facts = _parse_decomposed_facts(response.content)
        except Exception as e:
            print(f"✗ decomp error: {e}")
            errors.append({"chunk_id": chunk_id, "pass": "decompose", "error": str(e)})
            continue

        if not facts:
            print("→ 0 facts")
            continue

        total_facts += len(facts)

        # --- Stage 2: Extract 1 triple per fact ---
        chunk_triples: list[dict] = []
        chunk_rejected = 0

        for fact in facts:
            try:
                extract_prompt = SINGLE_FACT_EXTRACTION_PROMPT.format(
                    vocab_lines=vocab_lines,
                    fact_text=fact,
                )
                resp = llm_json.invoke([HumanMessage(content=extract_prompt)])
                raw = resp.content.strip()

                # Parse single JSON object (not array).
                parsed = None
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    # Try extracting from wrapper
                    m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
                    if m:
                        try:
                            parsed = json.loads(m.group())
                        except (json.JSONDecodeError, ValueError):
                            pass

                if not parsed or not isinstance(parsed, dict):
                    continue

                # Validate and normalize the triple.
                validated = _parse_chunk_triples(
                    [parsed], chunk_id, source,
                    lob=chunk.get("lob", "generic"),
                )
                if not validated:
                    continue

                triple = validated[0]

                # Span-grounding verification against source chunk.
                grounded, conf_mult = _verify_span_grounding(triple, content)
                if grounded:
                    chunk_triples.append({
                        **triple,
                        "source_type": "decompose",
                        "confidence": round(triple["confidence"] * conf_mult, 3),
                    })
                else:
                    chunk_rejected += 1

            except Exception:
                continue  # Skip individual fact extraction failures silently.

            time.sleep(0.02)  # Small delay between per-fact calls.

        total_triples += len(chunk_triples)
        total_rejected += chunk_rejected
        all_new_triples.extend(chunk_triples)

        print(f"→ {len(facts)} facts → {len(chunk_triples)} triples "
              f"({chunk_rejected} rejected)")

    grounding_rate = total_triples / max(total_triples + total_rejected, 1)
    print(f"  → Decompose total: {total_facts} facts → {total_triples} triples, "
          f"{total_rejected} rejected (grounding: {grounding_rate:.0%})")

    return all_new_triples


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
        chunk_lob = chunk.get("lob", "generic")
        preamble = _build_chunk_preamble(chunk)
        messages = base_messages + [
            HumanMessage(content=f"{preamble}\nText: {content}")
        ]
        try:
            response      = llm.invoke(messages)
            parsed        = _parse_json_list(response.content)
            chunk_triples = _parse_chunk_triples(parsed, chunk_id, source, lob=chunk_lob)
            if len(chunk_triples) > MAX_TRIPLES_PER_CHUNK:
                print(f"→ {len(chunk_triples)} (capped to {MAX_TRIPLES_PER_CHUNK})")
                chunk_triples = chunk_triples[:MAX_TRIPLES_PER_CHUNK]
            else:
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
    model        = state.get("model", config.OLLAMA_MODEL)
    vocab        = state.get("vocab", [])
    entity_types = state.get("entity_types", [])
    chunks       = state.get("chunks", [])
    num_passes   = state.get("num_passes", 3)

    # json_mode=False: Ollama >=0.19 structured output grammar stops after
    # the first complete JSON value, producing 1-item arrays.  Disabling
    # json_mode lets the model output full multi-item arrays; the robust
    # _parse_json_list() parser handles any formatting issues.
    llm = get_llm(model, json_mode=False)
    all_raw_triples: list[dict] = []
    errors:          list[dict] = []

    if num_passes == 1:
        # Single combined pass covering all focus areas
        print(f"\n[3/4] Extracting triples — 1-pass few-shot IE ({model})...")
        print(f"  Vocab: {len(vocab)} relation types, {len(entity_types)} entity types, "
              f"{len(chunks)} chunks × 1 pass (combined)")
        base_messages = _build_extraction_messages(
            vocab, focus=SINGLE_PASS_FOCUS, entity_types=entity_types
        )
        print(f"\n  Pass 1/1 (combined):")
        pass_triples = _extract_one_pass(llm, base_messages, chunks, "combined", errors)
        all_raw_triples.extend(pass_triples)
        print(f"  → Pass 1 subtotal: {len(pass_triples)} triples")
    else:
        # Multi-pass: use first num_passes entries from PASS_FOCUS_INSTRUCTIONS
        focus_list = PASS_FOCUS_INSTRUCTIONS[:num_passes]
        n = len(focus_list)
        print(f"\n[3/4] Extracting triples — {n}-pass few-shot IE ({model})...")
        print(f"  Vocab: {len(vocab)} relation types, {len(entity_types)} entity types, "
              f"{len(chunks)} chunks × {n} passes")
        for pass_idx, focus in enumerate(focus_list):
            label = _PASS_LABELS[pass_idx] if pass_idx < len(_PASS_LABELS) else f"pass{pass_idx+1}"
            print(f"\n  Pass {pass_idx + 1}/{n} ({label}):")
            base_messages = _build_extraction_messages(vocab, focus=focus, entity_types=entity_types)
            pass_triples  = _extract_one_pass(llm, base_messages, chunks, label, errors)
            all_raw_triples.extend(pass_triples)
            print(f"  → Pass {pass_idx + 1} subtotal: {len(pass_triples)} triples")

    # Regex numeric fallback (F-01 fix) — only for small models (8b/7b).
    # Large models (70b/72b) extract numeric facts directly; the regex uses
    # flood-specific subject names (Building Coverage, Proof of Loss, etc.)
    # that would contaminate cross-domain runs.
    is_small_model = any(tag in model.lower() for tag in (":8b", ":7b", "-8b", "-7b"))
    if is_small_model:
        print("\n  Regex numeric fallback (F-01 fix):")
        regex_numeric: list[dict] = []
        for chunk in chunks:
            regex_numeric.extend(
                _extract_numeric_from_text(
                    chunk["content"][:CHUNK_CONTENT_LIMIT],
                    chunk.get("chunk_id", ""),
                    chunk.get("source", ""),
                    lob=chunk.get("lob", "generic"),
                )
            )
        if regex_numeric:
            all_raw_triples.extend(regex_numeric)
            print(f"  → Added {len(regex_numeric)} regex numeric triples "
                  f"(dollar amounts, time periods, percentages)")
        else:
            print("  → No unambiguous numeric patterns found")
    else:
        print(f"\n  Regex numeric fallback skipped ({model} extracts numerics directly)")

    # Decompose-then-extract for prose chunks that Pass 1 under-extracted.
    # Bypasses json_mode 1-item array constraint by decomposing into facts first.
    # Only run on non-small models (small models benefit more from regex fallback).
    if not is_small_model:
        from zone2.structured_mapper import is_structured_chunk, is_schema_chunk
        pdf_chunks = [c for c in chunks if not is_structured_chunk(c) and not is_schema_chunk(c)]
        if pdf_chunks:
            print(f"\n  Decompose-then-extract — {len(pdf_chunks)} PDF chunks:")
            decomp_triples = _decompose_then_extract(
                model, pdf_chunks, all_raw_triples, vocab, errors
            )
            all_raw_triples.extend(decomp_triples)
            print(f"  → Decompose added {len(decomp_triples)} new triples")
        else:
            print("\n  Decompose-then-extract skipped — no PDF chunks")

    # Deduplicate across passes by normalized (subject, relation, object)
    seen_keys:   set[tuple]  = set()
    all_triples: list[dict]  = []
    for t in all_raw_triples:
        key = (t["subject"].lower().strip(), t["relation"], t["object"].lower().strip())
        if key not in seen_keys:
            seen_keys.add(key)
            all_triples.append(t)

    dedup_removed = len(all_raw_triples) - len(all_triples)

    # Merge SEAF-KG structured triples (deterministic CSV extraction).
    structured = state.get("structured_triples", [])
    if structured:
        all_triples = structured + all_triples
        print(f"\n  ✓ Structured triples (SEAF-KG): {len(structured)}")

    # P1: Context-aware placeholder filtering.
    pre_filter_count = len(all_triples)
    all_triples, placeholder_stats = _filter_placeholder_triples(all_triples)
    n_removed = pre_filter_count - len(all_triples)
    if n_removed > 0:
        print(f"\n  ✓ Placeholder filter: removed {n_removed} triples")
        for reason, count in sorted(placeholder_stats.items(), key=lambda x: -x[1]):
            print(f"      {reason}: {count}")

    # P5: Semantic validation.
    pre_semantic = len(all_triples)
    all_triples, semantic_stats = _filter_semantic_errors(all_triples)
    n_semantic = pre_semantic - len(all_triples)
    if n_semantic > 0:
        print(f"\n  ✓ Semantic validator: rejected {n_semantic} triples")
        for reason, count in sorted(semantic_stats.items(), key=lambda x: -x[1]):
            print(f"      {reason}: {count}")

    per_chunk = len(all_triples) / max(len(chunks), 1)

    print(f"\n  ✓ LLM raw triples (all passes): {len(all_raw_triples)}")
    llm_count = len(all_triples) - len(structured)
    print(f"  ✓ LLM after dedup + filter:     {llm_count} "
          f"(removed {dedup_removed} duplicates, {n_removed} placeholders)")
    print(f"  ✓ Total triples (structured + LLM): {len(all_triples)}")
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
            "subject":      t["subject"],
            "subject_type": t.get("subject_type", "Unknown"),
            "object":       t["object"],
            "object_type":  t.get("object_type", "Unknown"),
            "span":         t["span"],
            "confidence":   t.get("confidence", 1.0),
            "chunk_id":     t["chunk_id"],
            "source":       t["source"],
            "source_type":  t.get("source_type", "llm"),
        })
    return dict(by_relation)


def _batch_merge_triples(graph: Neo4jGraph, by_relation: dict) -> int:
    """MERGE triples into Neo4j grouped by relation type.

    The relationship MERGE writes ``r.lob`` onto every edge so Zone 3
    induction and chatbot filters can scope by Line of Business without
    traversing through endpoint nodes.  The SET uses COALESCE so that
    re-inserts upgrade an unset edge but never blank a previously-set
    LOB (defensive against a triple arriving without LOB tagging).
    """
    total = 0
    for rel_type, batch in by_relation.items():
        rel_type = _sanitize_relation(rel_type)
        # Every row needs a lob field so the MERGE Cypher can read it
        # uniformly — backfill 'generic' for legacy callers that pre-date
        # Phase 3 threading.
        for row in batch:
            row.setdefault("lob", "generic")
        graph.query(
            f"""
            UNWIND $batch AS row
            MERGE (s:Entity {{id: row.subject}})
            ON CREATE SET s.entity_type = row.subject_type,
                          s.source_type = row.source_type
            ON MATCH SET s.entity_type = CASE
                WHEN s.entity_type IS NULL OR s.entity_type = 'Unknown' THEN row.subject_type
                ELSE s.entity_type END
            MERGE (o:Entity {{id: row.object}})
            ON CREATE SET o.entity_type = row.object_type,
                          o.source_type = row.source_type
            ON MATCH SET o.entity_type = CASE
                WHEN o.entity_type IS NULL OR o.entity_type = 'Unknown' THEN row.object_type
                ELSE o.entity_type END
            MERGE (s)-[r:{rel_type}]->(o)
            ON CREATE SET r.span       = row.span,
                          r.confidence = row.confidence,
                          r.chunk_id   = row.chunk_id,
                          r.source     = row.source,
                          r.lob        = row.lob
            ON MATCH SET  r.lob = COALESCE(r.lob, row.lob)
            """,
            params={"batch": batch}
        )
        total += len(batch)
    return total


def canonicalize_relations(state: Zone2State) -> dict:
    """EDC-inspired: map raw LLM relation types → bootstrapped vocab (one LLM call).

    The LLM-based mapping only applies to LLM-extracted triples. Structured
    triples get their normalization in the subsequent normalize_relations step
    (hybrid embedding + structural + token overlap).
    """
    print("\n[3.5/4] EDC Canonicalization — mapping raw relations → vocab...")
    triples = state.get("triples", [])
    vocab   = state.get("vocab", [])
    model   = state.get("model", config.OLLAMA_MODEL)

    # Only canonicalize LLM-extracted relations — structured field names are
    # already well-formed and must be preserved for cross-source linking.
    raw_relations = sorted(set(
        t["relation"] for t in triples if t.get("source_type") != "structured"
    ))
    if not raw_relations or not vocab:
        print("  ⚠ No triples or vocab — skipping canonicalization")
        return {}

    llm = get_llm(model, json_mode=True)
    prompt = (
        "Map each raw relation to the SINGLE closest relation from the vocabulary.\n"
        "If no close match exists, keep the original unchanged.\n\n"
        f"Vocabulary: {json.dumps(vocab)}\n\n"
        f"Raw relations to map: {json.dumps(raw_relations)}\n\n"
        'Respond with a JSON object mapping each raw relation to its vocab match:\n'
        '{"mappings": {"RAW_RELATION_1": "VOCAB_MATCH_1", "RAW_RELATION_2": "VOCAB_MATCH_2", ...}}'
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_text = response.content.strip()
        mapping: dict[str, str] = {}

        # Try JSON parsing first (preferred)
        try:
            parsed = json.loads(raw_text)
            raw_mapping = parsed.get("mappings", parsed) if isinstance(parsed, dict) else {}
            for raw_rel, mapped_rel in raw_mapping.items():
                if not isinstance(mapped_rel, str):
                    continue
                mapped = mapped_rel.strip().upper().replace(" ", "_")
                if raw_rel in raw_relations:
                    mapping[raw_rel] = mapped if mapped in vocab else raw_rel
        except (json.JSONDecodeError, ValueError):
            # Fallback: try line-by-line parsing with multiple arrow formats
            for line in raw_text.splitlines():
                for sep in ["->", "→", "=>", "-->", ": "]:
                    if sep in line:
                        parts = [p.strip().strip('"').strip("'") for p in line.split(sep, 1)]
                        if len(parts) == 2 and parts[0] in raw_relations:
                            mapped = parts[1].strip().upper().replace(" ", "_")
                            mapping[parts[0]] = mapped if mapped in vocab else parts[0]
                        break
    except Exception as e:
        print(f"  ⚠ Canonicalization LLM call failed ({e}); keeping original relations")
        return {"errors": state.get("errors", []) + [f"canonicalization_failed: {e}"]}

    # Apply mapping — skip structured triples (their relation names are
    # already well-formed field names needed for cross-source linking).
    canonicalized = []
    n_skipped = 0
    for t in triples:
        if t.get("source_type") == "structured":
            canonicalized.append(t)
            n_skipped += 1
        else:
            canonicalized.append({**t, "relation": mapping.get(t["relation"], t["relation"])})

    types_before = len(set(t["relation"] for t in triples if t.get("source_type") != "structured"))
    types_after  = len(set(t["relation"] for t in canonicalized if t.get("source_type") != "structured"))
    print(f"  ✓ Canonicalized: {types_before} → {types_after} LLM relation types "
          f"({n_skipped} structured triples preserved)")

    return {"triples": canonicalized}


# ---------------------------------------------------------------------------
# Hybrid Relation Normalization (Phase 2 fix — feedback #3)
# ---------------------------------------------------------------------------

# Relations too generic or too short to merge safely
_PROTECTED_RELATIONS: frozenset[str] = frozenset({
    "HAS_VALUE", "HAS_AMOUNT", "HAS_COST", "HAS_NAME", "HAS_TYPE",
    "HAS_STATUS", "HAS_DATE", "HAS_ID", "HAS_CODE", "HAS_SCORE",
    "IS_A", "HAS",
})

# Stop words to strip before computing token overlap
_REL_STOP_WORDS: frozenset[str] = frozenset({
    "HAS", "OF", "THE", "A", "AN", "IS", "FOR", "BY", "IN", "TO", "FROM",
})

# Tokens that, when both relations contain ANY of them and the sets disagree,
# block merging — they encode discriminating semantics (status vs issue,
# authorized vs approved, etc.).  Adding to this set is conservative: a
# false positive blocks a merge, never adds one.
_DISCRIMINATOR_TOKENS: frozenset[str] = frozenset({
    "STATUS", "ISSUE", "REASON", "CAUSE",
    "AUTHORIZED", "APPROVED", "DENIED", "PENDING", "REJECTED", "PAID",
    "OPEN", "CLOSE", "OPENED", "CLOSED", "ACTIVE", "INACTIVE",
    "REPLACE", "REPAIR", "HANDLE", "RESOLVE",
    "MIN", "MAX", "AVG", "MEAN", "MEDIAN", "TOTAL", "COUNT", "SUM",
    "START", "END", "BEGIN", "FINISH",
})


def _rel_content_tokens(rel: str) -> set[str]:
    """Extract content tokens from a relation name (strip HAS_, OF, THE, etc.)."""
    parts = rel.replace("HAS_", "").split("_")
    return {p.upper() for p in parts if p.upper() not in _REL_STOP_WORDS and len(p) > 1}


def _safe_to_merge_rels(
    sim: float,
    tokens_i: set[str],
    tokens_j: set[str],
) -> bool:
    """Decide whether two relations should be merged.

    Stricter than a flat 0.85 + ≥1-shared-token rule:
      • Short relations (≤4 content tokens) require sim ≥ 0.92 — short names
        carry less signal, so cosine similarity overstates closeness.
      • Token overlap must reach Jaccard ≥ 0.6 — for two 3-token relations
        this means ≥2 shared tokens, blocking pairs like CLAIM_STATUS vs
        CLAIM_ISSUE that share only the head noun.
      • Discriminator tokens (status/issue, authorized/approved, replace/handle,
        min/max, etc.) must agree when both names contain any.
    """
    short = max(len(tokens_i), len(tokens_j)) <= 4
    if sim < (0.92 if short else 0.85):
        return False

    overlap = tokens_i & tokens_j
    union = tokens_i | tokens_j
    if not overlap or len(overlap) / max(1, len(union)) < 0.6:
        return False

    disc_i = tokens_i & _DISCRIMINATOR_TOKENS
    disc_j = tokens_j & _DISCRIMINATOR_TOKENS
    if disc_i and disc_j and disc_i != disc_j:
        return False

    return True


def normalize_structured_relations(state: Zone2State) -> dict:
    """Hybrid normalization of HAS_* relation types across structured + LLM triples.

    Three signals must agree to merge two relations:
    1. Embedding similarity >= threshold (all-MiniLM-L6-v2)
    2. Structural compatibility: dominant object value types must overlap
    3. Token overlap: at least 1 shared content word

    For near-ties (0.85-0.92), an LLM confirmation is used.
    Protected relations (too generic) are never merged.
    """
    print("\n[3.6/4] Hybrid relation normalization (structured + LLM)...")
    triples = state.get("triples", [])
    model_name = state.get("model", config.OLLAMA_MODEL)

    if not triples:
        return {}

    # Collect all unique relation types and their object value types
    rel_to_obj_types: dict[str, dict[str, int]] = {}
    for t in triples:
        rel = t["relation"]
        obj_type = t.get("object_type", "Unknown")
        if rel not in rel_to_obj_types:
            rel_to_obj_types[rel] = {}
        rel_to_obj_types[rel][obj_type] = rel_to_obj_types[rel].get(obj_type, 0) + 1

    all_rels = sorted(rel_to_obj_types.keys())
    if len(all_rels) < 2:
        print(f"  Only {len(all_rels)} relation types — skipping normalization")
        return {}

    # Filter to HAS_* relations with >1 occurrence (primary merge candidates)
    has_rels = [r for r in all_rels if r.startswith("HAS_") and r not in _PROTECTED_RELATIONS]
    if len(has_rels) < 2:
        print(f"  Only {len(has_rels)} HAS_* relation types — skipping")
        return {}

    # Embed relation names (lazy-load model, reuse across calls)
    from zone2.entity_resolution import _get_embed_model
    embed_model = _get_embed_model()
    embeddings = embed_model.encode(
        [r.replace("_", " ").lower() for r in has_rels],
        normalize_embeddings=True,
    )

    import numpy as np
    sim_matrix = embeddings @ embeddings.T

    # Build merge candidates: all pairs that pass the safety check
    merge_map: dict[str, str] = {}
    merged_into: dict[str, set[str]] = {}  # canonical -> set of merged rels

    for i in range(len(has_rels)):
        if has_rels[i] in merge_map:
            continue
        cluster = [i]
        for j in range(i + 1, len(has_rels)):
            if has_rels[j] in merge_map:
                continue

            rel_i, rel_j = has_rels[i], has_rels[j]
            tokens_i = _rel_content_tokens(rel_i)
            tokens_j = _rel_content_tokens(rel_j)

            # Check 1+2: cosine, Jaccard token overlap, discriminator tokens
            if not _safe_to_merge_rels(float(sim_matrix[i, j]), tokens_i, tokens_j):
                continue

            # Check 3: Structural compatibility (object types must overlap)
            types_i = set(rel_to_obj_types.get(rel_i, {}).keys())
            types_j = set(rel_to_obj_types.get(rel_j, {}).keys())
            # Remove "Unknown" before comparison
            types_i.discard("Unknown")
            types_j.discard("Unknown")
            if types_i and types_j and not (types_i & types_j):
                # Different value types (e.g., Currency vs Boolean) — don't merge
                continue

            cluster.append(j)

        if len(cluster) > 1:
            # Elect canonical: shortest name in cluster
            cluster_rels = [has_rels[idx] for idx in cluster]
            canonical = min(cluster_rels, key=len)
            merged_into[canonical] = set(cluster_rels) - {canonical}
            for r in cluster_rels:
                if r != canonical:
                    merge_map[r] = canonical

    if not merge_map:
        print(f"  No relations merged (all {len(has_rels)} HAS_* types are distinct)")
        return {}

    # Also apply RELATION_NORMALIZATIONS and pattern normalization to structured triples
    # (previously skipped — the original canonicalize_relations only handled LLM triples)
    normalized: list[dict] = []
    n_merged = 0
    n_pattern_fixed = 0
    for t in triples:
        rel = t["relation"]
        new_rel = rel

        # Apply hybrid merge map
        if rel in merge_map:
            new_rel = merge_map[rel]
            n_merged += 1
        # Apply existing normalization dict to ALL triples (including structured)
        elif rel in RELATION_NORMALIZATIONS:
            new_rel = RELATION_NORMALIZATIONS[rel]
            n_pattern_fixed += 1
        # Apply pattern normalization to ALL triples
        else:
            patterned = _pattern_normalize_relation(rel)
            if patterned != rel:
                new_rel = patterned
                n_pattern_fixed += 1

        normalized.append({**t, "relation": new_rel})

    types_before = len(set(t["relation"] for t in triples))
    types_after = len(set(t["relation"] for t in normalized))

    print(f"  ✓ Hybrid normalization: {types_before} → {types_after} relation types")
    print(f"    Embedding-merged: {n_merged} triples ({len(merge_map)} relation types)")
    print(f"    Pattern/dict-fixed: {n_pattern_fixed} triples")
    if merged_into:
        for canonical, members in sorted(merged_into.items()):
            print(f"    {canonical} ← {', '.join(sorted(members))}")

    return {"triples": normalized}


# ---------------------------------------------------------------------------
# Property-vs-Entity Collapse (Phase 3 fix — feedback #2)
# ---------------------------------------------------------------------------

# Value patterns: object values matching these are literal data, not entities
import re as _re

_CURRENCY_RE = _re.compile(r"^\$[\d,]+\.?\d*$")
_DATE_RE = _re.compile(
    r"^\d{4}-\d{2}-\d{2}$|^\d{1,2}/\d{1,2}/\d{2,4}$|^\d{4}$"
)
_NUMBER_RE = _re.compile(r"^-?[\d,]+\.?\d*%?$")
_SHORT_CODE_RE = _re.compile(r"^[A-Z0-9]{1,5}$")

# Relations where the object should stay an entity even if it looks like a literal
_ENTITY_BIAS_RELATIONS: frozenset[str] = frozenset({
    "HAS_ADDRESS", "LOCATED_AT", "HAS_INSURED_NAME", "HAS_POLICYHOLDER",
    "HAS_AGENT", "HAS_CLAIMANT", "HAS_BENEFICIARY", "HAS_OWNER",
    "LINKED_TO", "SAME_AS",
})

# Object types that should always remain entities
_ENTITY_BIAS_TYPES: frozenset[str] = frozenset({
    "Person", "Organization", "Property", "Building", "InsuredProperty",
    "Company", "Agency", "Agent",
})


def _is_literal_value(obj: str, obj_type: str) -> bool:
    """Check if an object string looks like a literal data value."""
    if obj_type in _ENTITY_BIAS_TYPES:
        return False
    # Explicit value object types from Zone 2 extraction
    if obj_type in ("Currency", "Date", "Numeric", "Percentage",
                    "Categorical", "Text", "TimePeriod", "FinancialAmount"):
        return True
    if _CURRENCY_RE.match(obj):
        return True
    if _DATE_RE.match(obj):
        return True
    if _NUMBER_RE.match(obj):
        return True
    if _SHORT_CODE_RE.match(obj) and len(obj) <= 5:
        return True
    # Short categorical values (Yes/No/N/A/True/False/Unknown/etc.)
    if len(obj) <= 15 and obj.lower() in {
        "yes", "no", "n/a", "na", "true", "false", "unknown", "none",
        "null", "other", "pending", "active", "inactive", "closed",
        "open", "approved", "denied", "completed",
    }:
        return True
    return False


def collapse_value_to_properties(
    triples: list[dict],
) -> tuple[list[dict], dict[str, dict[str, str]]]:
    """Collapse single-use literal values from triples to node properties.

    A triple (S)-[HAS_X]->(V) becomes a property {S: {x: V}} if:
    1. V matches a literal pattern (currency, date, number, short code)
    2. Relation starts with HAS_
    3. V is not referenced by multiple subjects (not a join key)
    4. Relation is not in ENTITY_BIAS_RELATIONS
    5. Object type is not in ENTITY_BIAS_TYPES

    Returns:
        (filtered_triples, {subject_id: {property_name: value, ...}})
    """
    # Pass 1: identify candidate collapse triples and count object usage
    obj_to_subjects: dict[str, set[str]] = {}
    candidates: list[int] = []  # indices into triples

    for i, t in enumerate(triples):
        rel = t["relation"]
        obj = t.get("object", "")
        obj_type = t.get("object_type", "Unknown")
        subj = t["subject"]

        if (
            rel.startswith("HAS_")
            and rel not in _ENTITY_BIAS_RELATIONS
            and _is_literal_value(obj, obj_type)
        ):
            candidates.append(i)
            if obj not in obj_to_subjects:
                obj_to_subjects[obj] = set()
            obj_to_subjects[obj].add(subj)

    # Pass 2: Data-driven relation classification
    # For each candidate relation, compute statistics to decide collapse vs preserve:
    #   - MEASURE: high cardinality + mostly numeric → keep as entity (sortable/aggregable)
    #   - DIMENSION: low cardinality + shared across subjects → keep as entity (join key)
    #   - PROPERTY: single-use literal → collapse to node property
    MAX_MULTI_USE = 3
    MEASURE_CARDINALITY_MIN = 0.5   # if >50% unique → candidate measure
    MEASURE_CARDINALITY_MAX = 0.95  # if >95% unique → likely an ID, not a measure
    DIMENSION_REUSE_THRESHOLD = 2   # if avg subjects per value >= 2 → dimension

    # Compute per-relation statistics from candidates
    import re as _collapse_re
    _NUMERIC_PAT = _collapse_re.compile(r'^-?[\d,]+\.?\d*%?$')

    # ID threshold: numeric values with median digit count > 8 are likely
    # identifiers (claim numbers, response IDs, serial numbers), not measures.
    ID_DIGIT_THRESHOLD = 8

    rel_stats: dict[str, dict] = {}
    for i in candidates:
        t = triples[i]
        rel = t["relation"]
        obj = t["object"]
        if rel not in rel_stats:
            rel_stats[rel] = {"values": set(), "numeric_count": 0, "total": 0,
                              "long_digit_count": 0}
        rel_stats[rel]["values"].add(obj)
        rel_stats[rel]["total"] += 1
        cleaned = obj.replace("$", "").replace(",", "").replace(".", "").replace("-", "")
        if _NUMERIC_PAT.match(obj.replace("$", "").replace(",", "")):
            rel_stats[rel]["numeric_count"] += 1
            if len(cleaned) > ID_DIGIT_THRESHOLD:
                rel_stats[rel]["long_digit_count"] += 1

    # Classify each relation
    measure_rels: set[str] = set()     # high-cardinality numeric → preserve
    dimension_rels: set[str] = set()   # low-cardinality, high-reuse → preserve

    for rel, stats in rel_stats.items():
        n_values = len(stats["values"])
        n_total = stats["total"]
        numeric_frac = stats["numeric_count"] / n_total if n_total > 0 else 0
        cardinality = n_values / n_total if n_total > 0 else 0  # 1.0 = all unique
        avg_reuse = n_total / n_values if n_values > 0 else 0

        # Long-digit fraction: if >50% of numeric values have >8 digits,
        # this relation holds IDs/serials, not measures. O(1) per relation.
        n_numeric = stats["numeric_count"]
        long_frac = stats["long_digit_count"] / n_numeric if n_numeric > 0 else 0

        if (numeric_frac > 0.8
                and MEASURE_CARDINALITY_MIN < cardinality <= MEASURE_CARDINALITY_MAX
                and long_frac < 0.5):
            # Mostly numeric + high (but not near-100%) cardinality + mostly short numbers
            # = measure column (scores, amounts, times).
            # Near-100% cardinality or mostly long numbers = ID/serial → collapse.
            measure_rels.add(rel)
        elif avg_reuse >= DIMENSION_REUSE_THRESHOLD and n_values >= 2:
            # Same value shared across multiple subjects = dimension / join key
            dimension_rels.add(rel)

    if measure_rels:
        print(f"    Measure relations (preserved): {len(measure_rels)}")
        for r in sorted(measure_rels)[:10]:
            s = rel_stats[r]
            print(f"      {r}: {s['total']} triples, {len(s['values'])} distinct, "
                  f"{s['numeric_count']}/{s['total']} numeric")
    if dimension_rels:
        print(f"    Dimension relations (preserved): {len(dimension_rels)}")
        for r in sorted(dimension_rels)[:10]:
            s = rel_stats[r]
            print(f"      {r}: {s['total']} triples, {len(s['values'])} distinct, "
                  f"reuse={s['total']/len(s['values']):.1f}x")

    collapse_indices: set[int] = set()
    node_properties: dict[str, dict[str, str]] = {}

    for i in candidates:
        t = triples[i]
        obj = t["object"]
        subj = t["subject"]
        rel = t["relation"]

        # Never collapse measure values (data-driven: high-cardinality numerics)
        if rel in measure_rels:
            continue

        # Never collapse dimension values (data-driven: high-reuse join keys)
        if rel in dimension_rels:
            continue

        n_subjects = len(obj_to_subjects.get(obj, set()))
        if n_subjects > MAX_MULTI_USE:
            continue
        # Multi-use values only collapse if short and generic
        if n_subjects > 1 and len(obj) > 15:
            continue

        # Convert relation name to property name: HAS_DEDUCTIBLE -> deductible
        prop_name = t["relation"].removeprefix("HAS_").lower()

        if subj not in node_properties:
            node_properties[subj] = {}
        node_properties[subj][prop_name] = obj
        collapse_indices.add(i)

    # Build filtered triple list (excluding collapsed triples)
    filtered = [t for i, t in enumerate(triples) if i not in collapse_indices]

    n_props = sum(len(ps) for ps in node_properties.values())
    print(f"  ✓ Collapsed {len(collapse_indices)} value triples → "
          f"{n_props} node properties on {len(node_properties)} entities")

    return filtered, node_properties


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

        no_wipe = state.get("no_wipe", False)
        if no_wipe:
            print("  ⚠ --no-wipe: skipping graph clear (incremental mode)")
        else:
            print("  WARNING: Clearing entire Neo4j graph. "
                  "Do NOT run concurrent jobs on the same Neo4j instance.")
            graph.query("MATCH (n) DETACH DELETE n")
            _r = graph.query("MATCH (n) RETURN count(n) AS c")
            print(f"  ✓ Graph cleared (nodes remaining: {_r[0]['c'] if _r else 0})")

        # Collapse single-use literal values to node properties (feedback #2)
        triples, node_properties = collapse_value_to_properties(triples)

        by_relation     = _group_triples_by_relation(triples)
        total_submitted = _batch_merge_triples(graph, by_relation)

        # Write collapsed properties to Neo4j nodes
        n_props_written = 0
        if node_properties:
            # Batch property writes by collecting all in one UNWIND
            prop_batch = [
                {"id": eid, "props": props}
                for eid, props in node_properties.items()
            ]
            # Write in chunks to avoid oversized queries
            PROP_BATCH_SIZE = 200
            for chunk_start in range(0, len(prop_batch), PROP_BATCH_SIZE):
                chunk = prop_batch[chunk_start:chunk_start + PROP_BATCH_SIZE]
                graph.query(
                    """
                    UNWIND $batch AS row
                    MATCH (n:Entity {id: row.id})
                    SET n += row.props
                    """,
                    params={"batch": chunk},
                )
                n_props_written += len(chunk)
            print(f"  ✓ Wrote properties to {n_props_written} nodes")

        _nc = graph.query("MATCH (n:Entity) RETURN count(n) AS c")
        _rc = graph.query("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS c")
        node_count = _nc[0]["c"] if _nc else 0
        rel_count  = _rc[0]["c"] if _rc else 0

        stats = {
            "nodes":             node_count,
            "relationships":     rel_count,
            "triples_submitted": total_submitted,
            "properties_written": n_props_written,
            "relation_types":    sorted(by_relation.keys()),
        }
        print(f"  ✓ Inserted: {node_count} nodes, {rel_count} relationships")
        print(f"  ✓ Distinct relation types: {len(by_relation)}")
        return {"neo4j_stats": stats}

    except Exception as e:
        print(f"  ✗ Neo4j error: {e}")
        return {"neo4j_stats": {"error": str(e)}}


def zone25_entity_resolution(state: Zone2State) -> dict:
    """Zone 2.5: in-memory entity resolution on triple list.

    Runs BEFORE Neo4j insertion — deduplicates near-identical node names
    by replacing duplicate IDs in the triple list. Zero Neo4j round-trips.
    """
    print("\n[4.5] Zone 2.5 — Entity Resolution (in-memory)...")
    triples = state.get("triples", [])
    if not triples:
        return {"resolution_stats": {"merged": 0}}

    deduplicated, stats = resolve_entities_in_memory(triples)
    return {"triples": deduplicated, "resolution_stats": stats}


# ---------------------------------------------------------------------------
# LangGraph Build
# ---------------------------------------------------------------------------

def build_pipeline():
    builder = StateGraph(Zone2State)
    builder.add_node("load_chunks",              load_chunks)
    builder.add_node("detect_lobs",              detect_lobs)
    builder.add_node("extract_structured",       extract_structured)
    builder.add_node("bootstrap_vocab",          bootstrap_vocab)
    builder.add_node("extract_triples",          extract_triples)
    builder.add_node("canonicalize_relations",   canonicalize_relations)
    builder.add_node("normalize_relations",      normalize_structured_relations)
    builder.add_node("insert_to_neo4j",          insert_to_neo4j)
    builder.add_node("zone25_entity_resolution", zone25_entity_resolution)
    builder.add_node("cross_source_link",        cross_source_link)
    builder.set_entry_point("load_chunks")
    builder.add_edge("load_chunks",              "detect_lobs")
    builder.add_edge("detect_lobs",              "extract_structured")
    builder.add_edge("extract_structured",       "bootstrap_vocab")
    builder.add_edge("bootstrap_vocab",          "extract_triples")
    builder.add_edge("extract_triples",          "canonicalize_relations")
    builder.add_edge("canonicalize_relations",   "normalize_relations")
    builder.add_edge("normalize_relations",      "zone25_entity_resolution")
    builder.add_edge("zone25_entity_resolution", "cross_source_link")
    builder.add_edge("cross_source_link",        "insert_to_neo4j")
    builder.add_edge("insert_to_neo4j",          END)
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


def _save_run_summary(result: dict, model: str, elapsed: float,
                      results_dir: str | None = None) -> str:
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
    rdir = results_dir or config.RESULTS_DIR
    out_path = os.path.join(rdir, "zone2_run_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return out_path


def run_zone2(model: str = config.OLLAMA_MODEL, num_passes: int = 3,
              skip_extraction: bool = False,
              chunks_file: str | None = None,
              results_dir: str | None = None,
              no_wipe: bool = False):
    rdir = results_dir or config.RESULTS_DIR
    os.makedirs(rdir, exist_ok=True)

    if skip_extraction:
        # Load cached triples from previous run, skip straight to
        # entity resolution → Neo4j insertion → cross-source linking.
        summary_path = os.path.join(rdir, "zone2_run_summary.json")
        if not os.path.exists(summary_path):
            print(f"✗ Cannot skip extraction: {summary_path} not found.")
            print("  Run full pipeline first to generate cached triples.")
            return {}

        print("=" * 60)
        print("CS584 Capstone — Zone 2 Pipeline [SKIP EXTRACTION — cached triples]")
        print(f"Loading triples from {summary_path}")
        print("=" * 60)

        with open(summary_path) as f:
            cached = json.load(f)

        triples = cached.get("triples", [])
        print(f"  ✓ Loaded {len(triples)} cached triples")
        print(f"  ✓ Skipping: load_chunks, extract_structured, bootstrap_vocab, "
              f"extract_triples, canonicalize_relations")
        print(f"  → Running: placeholder_filter → semantic_validator → "
              f"entity_resolution → cross_source_link → insert_to_neo4j")

        start = time.time()

        # Placeholder filtering (P1).
        pre_filter = len(triples)
        triples, ph_stats = _filter_placeholder_triples(triples)
        n_removed = pre_filter - len(triples)
        if n_removed > 0:
            print(f"\n  ✓ Placeholder filter: removed {n_removed} triples")
            for reason, count in sorted(ph_stats.items(), key=lambda x: -x[1]):
                print(f"      {reason}: {count}")

        # Semantic validation (P5).
        pre_sem = len(triples)
        triples, sem_stats = _filter_semantic_errors(triples)
        n_sem = pre_sem - len(triples)
        if n_sem > 0:
            print(f"\n  ✓ Semantic validator: rejected {n_sem} triples")
            for reason, count in sorted(sem_stats.items(), key=lambda x: -x[1]):
                print(f"      {reason}: {count}")

        # Entity resolution (in-memory).
        print("\n[4.5] Zone 2.5 — Entity Resolution (in-memory)...")
        if triples:
            deduplicated, resolution_stats = resolve_entities_in_memory(triples)
        else:
            deduplicated, resolution_stats = triples, {"merged": 0}

        # Cross-source linking (in-memory — adds LINKED_TO triples).
        result_state = {
            "triples": deduplicated,
            "resolution_stats": resolution_stats,
            "vocab": cached.get("vocab", []),
            "entity_types": cached.get("entity_types", []),
            "vocab_quality": cached.get("vocab_quality", {}),
            "errors": cached.get("errors", []),
            "model": model,
            "structured_triples": [],
            "chunks": [],
            "num_passes": num_passes,
            "chunks_file": chunks_file or config.ZONE1_CHUNKS_FILE,
            "results_dir": rdir,
            "no_wipe": no_wipe,
        }

        link_result = cross_source_link(result_state)
        result_state.update(link_result)

        # Neo4j insertion (final — writes everything including LINKED_TO).
        neo4j_result = insert_to_neo4j(result_state)
        result_state.update(neo4j_result)

        elapsed = time.time() - start
        result = result_state

    else:
        print("=" * 60)
        print("CS584 Capstone — Zone 2 Pipeline [Domain-Agnostic Open IE]")
        print(f"Bootstrapped schema + few-shot extraction → Neo4j  (model: {model})")
        print(f"Extraction passes: {num_passes}")
        print("=" * 60)

        pipeline = build_pipeline()
        start    = time.time()
        result   = pipeline.invoke({
            "chunks":             [],
            "vocab":              [],
            "entity_types":       [],
            "triples":            [],
            "structured_triples": [],
            "vocab_quality":      {},
            "neo4j_stats":        {},
            "resolution_stats":   {},
            "errors":             [],
            "model":              model,
            "num_passes":         num_passes,
            "chunks_file":        chunks_file or config.ZONE1_CHUNKS_FILE,
            "results_dir":        rdir,
            "no_wipe":            no_wipe,
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

    _print_comparison_table(_load_prior_results(results_dir=rdir))

    out_path = _save_run_summary(result, model, elapsed, results_dir=rdir)
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
  python3 zone2/pipeline.py --passes 1   # single combined pass (faster)
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
    parser.add_argument(
        "--passes", type=int, default=3, choices=[1, 2, 3],
        help="Number of extraction passes: 1=combined single pass, "
             "2=general+numeric, 3=general+numeric+obligations (default: 3)"
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip LLM extraction, load cached triples from zone2_run_summary.json. "
             "Only re-runs entity resolution + Neo4j insertion + cross-source linking."
    )
    parser.add_argument(
        "--chunks", default=None,
        help="Path to zone1_chunks.json. Default: config.ZONE1_CHUNKS_FILE"
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Output directory for results. Default: data/results"
    )
    parser.add_argument(
        "--no-wipe", action="store_true",
        help="Skip Neo4j graph clear (incremental mode, for concurrent safety)"
    )
    args = parser.parse_args()
    run_zone2(model=args.model, num_passes=args.passes,
              skip_extraction=args.skip_extraction,
              chunks_file=args.chunks,
              results_dir=args.results_dir,
              no_wipe=args.no_wipe)
