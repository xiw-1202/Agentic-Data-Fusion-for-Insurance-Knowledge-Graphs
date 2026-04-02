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
)
from zone2.entity_resolution import resolve_entities_in_memory
from zone2.structured_mapper import extract_structured
from zone2.cross_source_linker import cross_source_link
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


# ---------------------------------------------------------------------------
# Constants — ALL DOMAIN-AGNOSTIC
# ---------------------------------------------------------------------------

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
    """Load all chunks from Zone 1 (PDF + CSV sources)."""
    print("\n[1/4] Loading Zone 1 chunks...")
    with open(config.ZONE1_CHUNKS_FILE) as f:
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


def _extract_numeric_from_text(text: str, chunk_id: str, source: str) -> list[dict]:
    """
    Regex fallback: extract numeric triples (dollar amounts, time periods,
    percentages) that the LLM typically misses (F-01 fix).

    Domain-agnostic: derives subjects from surrounding text context
    instead of hardcoded entity names. Relations are pattern-based
    (HAS_COVERAGE_LIMIT, HAS_DEADLINE, etc.) using generic keywords.

    Only creates triples when the surrounding context is unambiguous.
    Returns [] for chunks with no numeric content or ambiguous context.
    """
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
# Recall-oriented Pass 2 (two-pass exhaustive extraction)
# ---------------------------------------------------------------------------

def _run_recall_pass(
    llm,
    chunks: list[dict],
    pass1_triples: list[dict],
    errors: list[dict],
) -> list[dict]:
    """Run recall-oriented Pass 2 on PDF chunks only.

    For each chunk, shows the LLM what was already extracted in Pass 1,
    then asks it to find additional explicit facts it missed.
    Verifies each new triple via span-grounding before accepting.

    Returns only verified new triples (not Pass 1 triples).
    """
    # Index Pass 1 triples by chunk_id for fast lookup.
    triples_by_chunk: dict[str, list[dict]] = defaultdict(list)
    for t in pass1_triples:
        triples_by_chunk[t.get("chunk_id", "")].append(t)

    recall_triples: list[dict] = []
    n_rejected = 0
    n_verified = 0

    for i, chunk in enumerate(chunks):
        content = chunk["content"][:CHUNK_CONTENT_LIMIT]
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        source = chunk.get("source", "")
        hierarchy = chunk.get("section_hierarchy", [])
        label = " > ".join(hierarchy) if hierarchy else f"chunk {i+1}"

        # Format existing triples for this chunk.
        existing = triples_by_chunk.get(chunk_id, [])
        if not existing:
            # No Pass 1 triples for this chunk — skip recall (nothing to build on)
            continue

        existing_str = "\n".join(
            f"  - ({t['subject']}) --[{t['relation']}]--> ({t['object']})"
            for t in existing
        )

        prompt = RECALL_PASS_PROMPT.format(
            existing_triples=existing_str,
            chunk_text=content,
        )

        print(f"    Chunk {i+1}/{len(chunks)} [{label[:45]}]...", end=" ", flush=True)

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            parsed = _parse_json_list(response.content)
            new_triples = _parse_chunk_triples(parsed, chunk_id, source)

            # Tag as recall pass and verify grounding.
            verified: list[dict] = []
            for t in new_triples:
                grounded, conf_mult = _verify_span_grounding(t, content)
                if grounded:
                    verified.append({
                        **t,
                        "source_type": "recall_pass",
                        "confidence": round(t["confidence"] * conf_mult, 3),
                    })
                    n_verified += 1
                else:
                    n_rejected += 1

            print(f"→ +{len(verified)} verified ({len(new_triples) - len(verified)} rejected)")
            recall_triples.extend(verified)

        except Exception as e:
            print(f"✗ ERROR: {e}")
            errors.append({"chunk_id": chunk_id, "pass": "recall", "error": str(e)})

        time.sleep(0.05)

    print(f"  → Recall pass total: {len(recall_triples)} verified, {n_rejected} rejected "
          f"(grounding rate: {n_verified / max(n_verified + n_rejected, 1):.0%})")

    return recall_triples


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

    llm = get_llm(model, json_mode=True)
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

    # Recall-oriented Pass 2: find what Pass 1 missed (PDF chunks only).
    # Only run on non-small models (small models benefit more from regex fallback).
    if not is_small_model:
        # Filter to PDF-only chunks for recall pass.
        from zone2.structured_mapper import is_structured_chunk, is_schema_chunk
        pdf_chunks = [c for c in chunks if not is_structured_chunk(c) and not is_schema_chunk(c)]
        if pdf_chunks:
            print(f"\n  Recall pass (Pass 2) — {len(pdf_chunks)} PDF chunks:")
            recall_triples = _run_recall_pass(llm, pdf_chunks, all_raw_triples, errors)
            all_raw_triples.extend(recall_triples)
            print(f"  → Recall pass added {len(recall_triples)} new triples")
        else:
            print("\n  Recall pass skipped — no PDF chunks")

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
    """MERGE triples into Neo4j grouped by relation type."""
    total = 0
    for rel_type, batch in by_relation.items():
        rel_type = _sanitize_relation(rel_type)
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
                          r.source     = row.source
            """,
            params={"batch": batch}
        )
        total += len(batch)
    return total


def canonicalize_relations(state: Zone2State) -> dict:
    """EDC-inspired: map raw LLM relation types → bootstrapped vocab (one LLM call).

    Structured triples (source_type='structured') are passed through unchanged —
    their relation names are field-derived (HAS_{FIELD_NAME}) and needed as-is
    for cross-source entity linking in Stage 3.
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
    builder.add_node("extract_structured",       extract_structured)
    builder.add_node("bootstrap_vocab",          bootstrap_vocab)
    builder.add_node("extract_triples",          extract_triples)
    builder.add_node("canonicalize_relations",   canonicalize_relations)
    builder.add_node("insert_to_neo4j",          insert_to_neo4j)
    builder.add_node("zone25_entity_resolution", zone25_entity_resolution)
    builder.add_node("cross_source_link",        cross_source_link)
    builder.set_entry_point("load_chunks")
    builder.add_edge("load_chunks",              "extract_structured")
    builder.add_edge("extract_structured",       "bootstrap_vocab")
    builder.add_edge("bootstrap_vocab",          "extract_triples")
    builder.add_edge("extract_triples",          "canonicalize_relations")
    builder.add_edge("canonicalize_relations",   "zone25_entity_resolution")
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


def run_zone2(model: str = config.OLLAMA_MODEL, num_passes: int = 3,
              skip_extraction: bool = False):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if skip_extraction:
        # Load cached triples from previous run, skip straight to
        # entity resolution → Neo4j insertion → cross-source linking.
        summary_path = os.path.join(config.RESULTS_DIR, "zone2_run_summary.json")
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
    args = parser.parse_args()
    run_zone2(model=args.model, num_passes=args.passes,
              skip_extraction=args.skip_extraction)
