"""Extraction Quality Evaluation (Domain-Agnostic)
===================================================
Evaluates Zone 2 extraction quality using 5 metrics, following
KGGen (NeurIPS 2025), AutoSchemaKG (2025), and GraphJudge (EMNLP 2025).

Metrics:
  1. Vocabulary Coverage — what % of source document vocabulary is captured?
  2. Triple Precision — LLM-as-judge on sampled triples (AutoSchemaKG approach)
  3. Fact Recall — MINE-1 style: LLM extracts facts, embedding match to KG
  4. Source Grounding — are triples traceable to source text?
  5. Graph Statistics — density, connectivity, degree distribution

All metrics are domain-agnostic — no hardcoded concept lists.
Neither metric uses the reference ontology (Riskine).

References:
  KGGen (NeurIPS 2025): MINE benchmark, 15 facts × 100 articles, LLM judge
  AutoSchemaKG (2025): LLM-judged triple P/R/F1, counting FP/FN
  GraphJudge (EMNLP 2025): G-BERTScore on linearized triples

Usage:
  python3 evaluation/extraction_quality.py --suffix zone2_seaf --model llama3.1:8b
  python3 evaluation/extraction_quality.py --suffix zone2_seaf --sample-size 100
"""

from __future__ import annotations

import json
import os
import sys
import argparse
import random
import re
from collections import Counter
from typing import Any

random.seed(42)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stopwords to exclude from vocabulary coverage (common English + formatting).
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "it", "its", "not", "no", "if", "then",
    "than", "when", "where", "which", "who", "whom", "how", "what", "we",
    "you", "your", "our", "they", "their", "he", "she", "his", "her",
    "i", "me", "my", "us", "them", "any", "all", "each", "every", "both",
    "such", "more", "most", "other", "some", "only", "also", "about",
    "up", "out", "so", "very", "just", "into", "over", "after", "before",
    "under", "between", "through", "during", "without", "within", "upon",
    "must", "see", "one", "two", "per", "part", "section", "page",
    "following", "above", "below", "whether", "because", "same",
    # Formatting/structural tokens
    "record", "dataset", "schema", "field", "value", "type", "code",
    "true", "false", "null", "none", "nan", "0", "1",
})

# Minimum word length for vocabulary coverage.
_MIN_WORD_LEN = 3

# Default sample sizes (literature-calibrated).
DEFAULT_PRECISION_SAMPLES = 100   # KGGen uses 100
DEFAULT_RECALL_CHUNKS = 30        # KGGen uses 100 articles; 30 is practical
DEFAULT_GROUNDING_SAMPLES = 50    # Reasonable for LLM-as-judge cost
FACTS_PER_CHUNK = 8               # KGGen uses 15; 8 balances cost vs coverage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )


def _get_llm(model: str) -> ChatOllama:
    return ChatOllama(model=model, base_url=config.OLLAMA_BASE_URL, temperature=0)


def _invoke_llm(llm: ChatOllama, prompt: str) -> str:
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        print(f"    [llm] Error: {e}", flush=True)
        return ""


def _get_all_entity_ids(graph: Neo4jGraph) -> list[str]:
    rows = graph.query("MATCH (n:Entity) RETURN n.id AS id")
    return [r["id"] for r in rows if r.get("id")]


def _get_all_triples(graph: Neo4jGraph) -> list[dict]:
    rows = graph.query("""
        MATCH (s:Entity)-[r]->(o:Entity)
        RETURN s.id AS subject, type(r) AS relation, o.id AS object,
               s.source_type AS s_source, o.source_type AS o_source
    """)
    return [
        {
            "subject": r["subject"],
            "relation": r["relation"],
            "object": r["object"],
            "source_type": r.get("s_source") or "llm",
        }
        for r in rows if r.get("subject") and r.get("object")
    ]


def _load_chunks() -> list[dict]:
    """Load Zone 1 chunks from disk."""
    chunks_path = config.ZONE1_CHUNKS_FILE
    if not os.path.exists(chunks_path):
        print(f"  Warning: chunks not found at {chunks_path}", flush=True)
        return []
    with open(chunks_path) as f:
        return json.load(f)


def _extract_vocabulary(chunks: list[dict]) -> set[str]:
    """Extract meaningful vocabulary from source chunks.

    Tokenizes chunk text, removes stopwords and short tokens,
    returns set of lowercase terms that represent source content.
    """
    vocab: set[str] = set()
    word_re = re.compile(r'[a-zA-Z]{3,}')

    for chunk in chunks:
        text = chunk.get("content", chunk.get("text", ""))
        words = word_re.findall(text.lower())
        for w in words:
            if w not in _STOPWORDS and len(w) >= _MIN_WORD_LEN:
                vocab.add(w)

    return vocab


# ---------------------------------------------------------------------------
# Metric 1: Vocabulary Coverage (domain-agnostic)
# ---------------------------------------------------------------------------

def measure_vocabulary_coverage(graph: Neo4jGraph, chunks: list[dict]) -> dict:
    """Measure what % of source document vocabulary is captured in the KG.

    Domain-agnostic: extracts vocabulary from actual source chunks,
    not from a hardcoded concept list.

    Sub-metrics:
      - Token coverage: % of source vocab tokens found in entity names
      - Entity diversity: unique entities, relation types, entity types
      - Structured vs LLM: breakdown by source type
    """
    print("\n[Metric 1] Vocabulary Coverage (domain-agnostic)...", flush=True)

    entity_ids = _get_all_entity_ids(graph)
    entity_ids_lower = {eid.lower() for eid in entity_ids}

    # Extract source vocabulary.
    source_vocab = _extract_vocabulary(chunks)
    if not source_vocab:
        print("  No source vocabulary extracted.", flush=True)
        return {"token_coverage": 0.0, "source_vocab_size": 0}

    # Check which source terms appear in any entity name.
    covered_terms: set[str] = set()
    for term in source_vocab:
        if any(term in eid for eid in entity_ids_lower):
            covered_terms.add(term)

    token_coverage = len(covered_terms) / len(source_vocab)

    # Entity diversity metrics.
    rel_rows = graph.query(
        "MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel"
    )
    n_relation_types = len(rel_rows)

    type_rows = graph.query(
        "MATCH (n:Entity) WHERE n.entity_type IS NOT NULL "
        "RETURN DISTINCT n.entity_type AS et"
    )
    n_entity_types = len([r for r in type_rows if r["et"] and r["et"] != "Unknown"])

    # Structured vs LLM breakdown.
    source_rows = graph.query(
        "MATCH (n:Entity) RETURN n.source_type AS st, count(n) AS cnt"
    )
    source_breakdown = {r["st"] or "llm": r["cnt"] for r in source_rows}

    print(f"  Source vocabulary: {len(source_vocab)} terms", flush=True)
    print(f"  Covered in KG:    {len(covered_terms)} ({token_coverage:.0%})", flush=True)
    print(f"  Entity diversity: {len(entity_ids)} entities, "
          f"{n_relation_types} relation types, {n_entity_types} entity types", flush=True)
    print(f"  Source breakdown: {source_breakdown}", flush=True)

    return {
        "token_coverage": round(token_coverage, 4),
        "source_vocab_size": len(source_vocab),
        "covered_terms": len(covered_terms),
        "uncovered_sample": sorted(list(source_vocab - covered_terms))[:20],
        "total_entities": len(entity_ids),
        "relation_types": n_relation_types,
        "entity_types": n_entity_types,
        "source_breakdown": source_breakdown,
    }


# ---------------------------------------------------------------------------
# Metric 2: Triple Precision (LLM-as-judge, AutoSchemaKG approach)
# ---------------------------------------------------------------------------

def measure_triple_precision(
    graph: Neo4jGraph,
    llm: ChatOllama,
    sample_size: int = DEFAULT_PRECISION_SAMPLES,
) -> dict:
    """Sample N triples, LLM judges correctness.

    Following AutoSchemaKG: LLM judges CORRECT / INCORRECT / UNCERTAIN.
    Precision = CORRECT / (CORRECT + INCORRECT)  [excludes uncertain]
    Accuracy = CORRECT / total                    [includes uncertain]

    Separates structured (confidence=1.0) from LLM-extracted triples
    since structured triples are deterministic and shouldn't need judging.
    """
    print(f"\n[Metric 2] Triple Precision (n={sample_size})...", flush=True)

    all_triples = _get_all_triples(graph)
    if not all_triples:
        print("  No triples found.", flush=True)
        return {"precision": 0.0, "accuracy": 0.0, "sample_size": 0}

    # Separate structured vs LLM triples for fair evaluation.
    llm_triples = [t for t in all_triples if t.get("source_type") != "structured"]
    structured_triples = [t for t in all_triples if t.get("source_type") == "structured"]

    # Only judge LLM-extracted triples (structured are deterministic).
    sample_pool = llm_triples if llm_triples else all_triples
    sample = random.sample(sample_pool, min(sample_size, len(sample_pool)))

    batch_size = 10
    results: list[dict] = []

    for batch_start in range(0, len(sample), batch_size):
        batch = sample[batch_start:batch_start + batch_size]

        triple_lines = [
            f"{i+1}. ({t['subject']}) --[{t['relation']}]--> ({t['object']})"
            for i, t in enumerate(batch)
        ]

        prompt = f"""You are a knowledge graph quality verifier for an insurance domain.
Judge whether each triple represents a factually plausible relationship.

TRIPLES:
{chr(10).join(triple_lines)}

For each triple, respond with EXACTLY one verdict:
  CORRECT — the relationship is factually plausible and semantically meaningful
  INCORRECT — the relationship is factually wrong or semantically nonsensical
  UNCERTAIN — can't determine without more context

Format: one line per triple, e.g. "1. CORRECT"
"""
        raw = _invoke_llm(llm, prompt)

        for i, t in enumerate(batch):
            verdict = "UNCERTAIN"
            pattern = rf'{i+1}[.)]\s*(CORRECT|INCORRECT|UNCERTAIN)'
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                verdict = m.group(1).upper()
            results.append({**t, "verdict": verdict})

        done = min(batch_start + batch_size, len(sample))
        if done % 20 == 0 or done == len(sample):
            print(f"    Judged {done}/{len(sample)}", flush=True)

    verdicts = Counter(r["verdict"] for r in results)
    correct = verdicts.get("CORRECT", 0)
    incorrect = verdicts.get("INCORRECT", 0)
    uncertain = verdicts.get("UNCERTAIN", 0)
    total = len(results)

    judged = correct + incorrect
    precision = correct / judged if judged > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    print(f"  Correct: {correct}, Incorrect: {incorrect}, Uncertain: {uncertain}", flush=True)
    print(f"  Precision: {precision:.1%} (excl. uncertain) | "
          f"Accuracy: {accuracy:.1%} (incl. uncertain)", flush=True)
    print(f"  Note: {len(structured_triples)} structured triples excluded "
          f"(deterministic, confidence=1.0)", flush=True)

    return {
        "precision": round(precision, 4),
        "accuracy": round(accuracy, 4),
        "sample_size": total,
        "correct": correct,
        "incorrect": incorrect,
        "uncertain": uncertain,
        "llm_triples_total": len(llm_triples),
        "structured_triples_total": len(structured_triples),
        "sampled_triples": results[:20],  # save first 20 for inspection
    }


# ---------------------------------------------------------------------------
# Metric 3: Fact Recall (MINE-1 style, embedding-based matching)
# ---------------------------------------------------------------------------

def _linearize_triple(t: dict) -> str:
    """Linearize a KG triple as a natural language sentence for G-BERTScore.

    (Policy, COVERS, Building) → "Policy covers Building"
    (Insured, MUST_NOTIFY, Insurer) → "Insured must notify Insurer"
    """
    subj = t.get("subject", "")
    rel = t.get("relation", "").replace("_", " ").lower()
    obj = t.get("object", "")
    return f"{subj} {rel} {obj}"


def measure_fact_recall(
    graph: Neo4jGraph,
    llm: ChatOllama,
    n_chunks: int = DEFAULT_RECALL_CHUNKS,
) -> dict:
    """Fact recall via G-BERTScore (Ghanem & Cruz KGSWC 2024, PiVe 2023).

    For each source chunk:
      1. LLM extracts key facts as short sentences
      2. All KG triples are linearized as sentences
      3. For each expected fact, find the best-matching KG triple via
         embedding cosine similarity (sentence-level, not entity-level)
      4. A fact is "recalled" if its best KG match exceeds the threshold

    This is G-BERTScore-style matching: compares fact sentences against
    triple sentences, handling paraphrases ("You" = "Insured",
    "Coverage A—Building Property" = "building coverage") naturally.

    Unlike our old approach (match S and O separately at ≥0.75),
    this matches the FULL fact against the FULL linearized triple.
    """
    print(f"\n[Metric 3] Fact Recall — G-BERTScore style (n_chunks={n_chunks})...",
          flush=True)

    chunks = _load_chunks()
    if not chunks:
        return {"fact_recall": 0.0, "note": "chunks not found"}

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("  sentence_transformers not available; falling back to substring match")
        return _measure_fact_recall_substring(graph, llm, n_chunks)

    # Sample diverse chunks (prefer content-rich, skip schema chunks).
    content_chunks = [
        c for c in chunks
        if c.get("token_count", 0) > 100
        and "schema" not in " ".join(str(h) for h in c.get("section_hierarchy", [])).lower()
    ]
    if not content_chunks:
        content_chunks = chunks
    sample = random.sample(content_chunks, min(n_chunks, len(content_chunks)))

    # Build linearized triple embedding index from KG.
    all_triples = _get_all_triples(graph)
    linearized = [_linearize_triple(t) for t in all_triples]
    print(f"  Linearizing {len(linearized)} KG triples...", flush=True)

    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed all KG triples as sentences (batch for speed).
    BATCH_SIZE = 512
    triple_embs_list = []
    for i in range(0, len(linearized), BATCH_SIZE):
        batch = linearized[i:i + BATCH_SIZE]
        embs = emb_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        triple_embs_list.append(embs)
    triple_embs = np.vstack(triple_embs_list)
    print(f"  Embedded {triple_embs.shape[0]} KG triples", flush=True)

    # Threshold: a fact is "recalled" if its best-matching KG triple
    # has cosine similarity ≥ this value. 0.65 is more forgiving than
    # the old 0.75-per-entity approach, but still requires strong semantic match.
    MATCH_THRESHOLD = 0.65

    total_facts = 0
    found_facts = 0
    chunk_results: list[dict] = []

    for chunk in sample:
        text = chunk.get("content", chunk.get("text", ""))[:1500]
        chunk_id = chunk.get("chunk_id", "?")

        # Step 1: LLM extracts key facts as short sentences.
        extract_prompt = f"""Read this text and list the {FACTS_PER_CHUNK} most important factual statements.
Each fact should be a single short sentence describing a specific relationship.
Do NOT include facts not directly stated in the text.

TEXT:
{text}

Output one fact per line as a short sentence (no numbering, no bullets).
"""
        raw_facts = _invoke_llm(llm, extract_prompt)

        facts: list[str] = []
        for line in raw_facts.strip().splitlines():
            line = line.strip().lstrip("0123456789.-) •")
            if len(line) > 15:
                facts.append(line)
        facts = facts[:FACTS_PER_CHUNK]

        if not facts:
            chunk_results.append({
                "chunk_id": chunk_id, "facts_extracted": 0,
                "facts_found_in_kg": 0, "recall": 0.0,
            })
            continue

        # Step 2: Embed expected facts.
        fact_embs = emb_model.encode(facts, normalize_embeddings=True, show_progress_bar=False)

        # Step 3: For each fact, find best-matching KG triple.
        # (fact_embs @ triple_embs.T) → [n_facts, n_triples] similarity matrix.
        sim_matrix = fact_embs @ triple_embs.T
        best_sims = np.max(sim_matrix, axis=1)  # best match per fact

        # Step 4: Count recalled facts.
        chunk_found = int(np.sum(best_sims >= MATCH_THRESHOLD))

        total_facts += len(facts)
        found_facts += chunk_found

        recall = chunk_found / len(facts) if facts else 0.0
        chunk_results.append({
            "chunk_id": chunk_id,
            "facts_extracted": len(facts),
            "facts_found_in_kg": chunk_found,
            "recall": round(recall, 3),
            "best_match_sims": [round(float(s), 3) for s in best_sims],
        })

        if len(chunk_results) % 10 == 0 or len(chunk_results) == len(sample):
            print(f"    Processed {len(chunk_results)}/{len(sample)} chunks "
                  f"(running recall: {found_facts}/{total_facts})", flush=True)

    overall_recall = found_facts / total_facts if total_facts > 0 else 0.0
    print(f"  Overall Fact Recall: {found_facts}/{total_facts} ({overall_recall:.0%})",
          flush=True)

    return {
        "fact_recall": round(overall_recall, 4),
        "total_facts": total_facts,
        "found_facts": found_facts,
        "chunks_sampled": len(sample),
        "method": "G-BERTScore (fact-vs-linearized-triple, cosine)",
        "match_threshold": MATCH_THRESHOLD,
        "chunk_results": chunk_results,
    }


def _measure_fact_recall_substring(
    graph: Neo4jGraph,
    llm: ChatOllama,
    n_chunks: int,
) -> dict:
    """Fallback fact recall using substring matching (no sentence-transformers)."""
    chunks = _load_chunks()
    content_chunks = [c for c in chunks if c.get("token_count", 0) > 100]
    if not content_chunks:
        content_chunks = chunks
    sample = random.sample(content_chunks, min(n_chunks, len(content_chunks)))

    entity_ids = _get_all_entity_ids(graph)
    entity_ids_lower = {eid.lower() for eid in entity_ids}

    total_facts = 0
    found_facts = 0
    chunk_results: list[dict] = []

    for chunk in sample:
        text = chunk.get("content", chunk.get("text", ""))[:1500]
        chunk_id = chunk.get("chunk_id", "?")

        extract_prompt = f"""Read this text and list the {FACTS_PER_CHUNK} most important factual statements as simple (subject, relation, object) triples.

TEXT:
{text}

Output one triple per line in format: subject | relation | object
"""
        raw_facts = _invoke_llm(llm, extract_prompt)

        facts = []
        for line in raw_facts.strip().splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3 and all(p for p in parts):
                facts.append({"subject": parts[0], "relation": parts[1], "object": parts[2]})

        chunk_found = 0
        for fact in facts:
            subj_lower = fact["subject"].lower()
            obj_lower = fact["object"].lower()
            subj_match = any(subj_lower in eid or eid in subj_lower for eid in entity_ids_lower)
            obj_match = any(obj_lower in eid or eid in obj_lower for eid in entity_ids_lower)
            if subj_match and obj_match:
                chunk_found += 1

        total_facts += len(facts)
        found_facts += chunk_found
        chunk_results.append({
            "chunk_id": chunk_id,
            "facts_extracted": len(facts),
            "facts_found_in_kg": chunk_found,
            "recall": round(chunk_found / len(facts), 3) if facts else 0.0,
        })

    overall_recall = found_facts / total_facts if total_facts > 0 else 0.0
    print(f"  Overall Fact Recall (substring): {found_facts}/{total_facts} "
          f"({overall_recall:.0%})", flush=True)

    return {
        "fact_recall": round(overall_recall, 4),
        "total_facts": total_facts,
        "found_facts": found_facts,
        "chunks_sampled": len(sample),
        "match_method": "substring",
        "chunk_results": chunk_results,
    }


# ---------------------------------------------------------------------------
# Metric 4: Source Grounding (triple-to-source traceability)
# ---------------------------------------------------------------------------

def measure_source_grounding(
    graph: Neo4jGraph,
    llm: ChatOllama,
    sample_size: int = DEFAULT_GROUNDING_SAMPLES,
) -> dict:
    """Check if extracted triples can be traced back to source text.

    Sample N triples, find their likely source chunk, ask LLM:
    "Does this chunk support this triple?"

    Only evaluates LLM-extracted triples (structured triples are
    deterministic and always grounded by definition).
    """
    print(f"\n[Metric 4] Source Grounding (n={sample_size})...", flush=True)

    chunks = _load_chunks()
    if not chunks:
        return {"grounding_rate": 0.0, "note": "chunks not found"}

    all_triples = _get_all_triples(graph)
    # Only evaluate LLM-extracted triples.
    llm_triples = [t for t in all_triples if t.get("source_type") != "structured"]
    if not llm_triples:
        print("  No LLM-extracted triples to evaluate.", flush=True)
        return {"grounding_rate": 1.0, "note": "all triples are structured"}

    sample = random.sample(llm_triples, min(sample_size, len(llm_triples)))

    # Build chunk index.
    chunk_texts = [
        (c.get("chunk_id", "?"), (c.get("content", "") or c.get("text", "")).lower())
        for c in chunks
    ]

    supported = 0
    partially = 0
    not_supported = 0
    no_source = 0
    results: list[dict] = []

    for i, triple in enumerate(sample):
        subj = triple["subject"].lower()
        obj = triple["object"].lower()

        # Find best matching chunk.
        best_chunk_id = None
        best_chunk_text = None

        # Priority 1: chunk containing both subject and object.
        for cid, ctext in chunk_texts:
            if subj[:20] in ctext and obj[:20] in ctext:
                best_chunk_id = cid
                best_chunk_text = ctext[:800]
                break

        # Priority 2: chunk containing subject only.
        if best_chunk_text is None:
            for cid, ctext in chunk_texts:
                if subj[:20] in ctext:
                    best_chunk_id = cid
                    best_chunk_text = ctext[:800]
                    break

        if best_chunk_text is None:
            results.append({**triple, "grounded": "NO_SOURCE", "chunk_id": None})
            no_source += 1
            continue

        prompt = f"""Does this source text support the following knowledge graph triple?

TRIPLE: ({triple['subject']}) --[{triple['relation']}]--> ({triple['object']})

SOURCE TEXT:
{best_chunk_text}

Answer EXACTLY one of:
  SUPPORTED — the text clearly states or implies this relationship
  NOT_SUPPORTED — the text does not contain this information
  PARTIALLY — the text partially supports this (e.g., similar but not exact)
"""
        raw = _invoke_llm(llm, prompt)
        verdict = "NOT_SUPPORTED"
        for v in ["SUPPORTED", "PARTIALLY", "NOT_SUPPORTED"]:
            if v in raw.upper():
                verdict = v
                break

        results.append({**triple, "grounded": verdict, "chunk_id": best_chunk_id})
        if verdict == "SUPPORTED":
            supported += 1
        elif verdict == "PARTIALLY":
            partially += 1
        else:
            not_supported += 1

        if (i + 1) % 10 == 0 or i + 1 == len(sample):
            print(f"    Checked {i+1}/{len(sample)}", flush=True)

    total = len(results)
    # Score: SUPPORTED=1.0, PARTIALLY=0.5, NOT_SUPPORTED/NO_SOURCE=0.0
    grounding_score = (supported + partially * 0.5) / total if total > 0 else 0.0

    print(f"  Supported: {supported}, Partially: {partially}, "
          f"Not supported: {not_supported}, No source: {no_source}", flush=True)
    print(f"  Grounding rate: {grounding_score:.1%}", flush=True)

    return {
        "grounding_rate": round(grounding_score, 4),
        "total_checked": total,
        "supported": supported,
        "partially_supported": partially,
        "not_supported": not_supported,
        "no_source_found": no_source,
        "sampled_results": results[:20],
    }


# ---------------------------------------------------------------------------
# Metric 5: Graph Statistics
# ---------------------------------------------------------------------------

def measure_graph_statistics(graph: Neo4jGraph) -> dict:
    """Compute structural statistics of the extracted KG. No LLM needed."""
    print("\n[Metric 5] Graph Statistics...", flush=True)

    import numpy as np

    node_count = graph.query("MATCH (n:Entity) RETURN count(n) AS c")[0]["c"]
    edge_count = graph.query("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS c")[0]["c"]

    # Degree distribution.
    degrees = graph.query(
        "MATCH (n:Entity) OPTIONAL MATCH (n)-[r]-() "
        "RETURN n.id AS id, count(r) AS degree"
    )
    degree_values = [r["degree"] for r in degrees]

    if degree_values:
        arr = np.array(degree_values)
        avg_degree = float(np.mean(arr))
        median_degree = float(np.median(arr))
        max_degree = int(np.max(arr))
        isolated = int(np.sum(arr == 0))
    else:
        avg_degree = median_degree = max_degree = isolated = 0

    # Relation type distribution.
    rel_types = graph.query(
        "MATCH (:Entity)-[r]->(:Entity) "
        "RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC"
    )
    n_rel_types = len(rel_types)
    top_5_rels = [(r["rel"], r["cnt"]) for r in rel_types[:5]]

    # Entity type distribution.
    entity_types = graph.query(
        "MATCH (n:Entity) RETURN n.entity_type AS etype, count(n) AS cnt "
        "ORDER BY cnt DESC"
    )
    n_entity_types = len([e for e in entity_types if e.get("etype") and e["etype"] != "Unknown"])
    top_5_types = [(e["etype"], e["cnt"]) for e in entity_types[:5] if e.get("etype")]

    # Density.
    density = edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0.0

    stats = {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 2),
        "median_degree": round(median_degree, 1),
        "max_degree": max_degree,
        "isolated_nodes": isolated,
        "relation_types": n_rel_types,
        "entity_types": n_entity_types,
        "top_5_relations": top_5_rels,
        "top_5_entity_types": top_5_types,
        "triples_per_entity": round(edge_count / node_count, 2) if node_count > 0 else 0,
    }

    print(f"  Nodes: {node_count} | Edges: {edge_count} | Density: {density:.6f}", flush=True)
    print(f"  Avg degree: {avg_degree:.1f} | Median: {median_degree:.0f} | Max: {max_degree}", flush=True)
    print(f"  Relation types: {n_rel_types} | Entity types: {n_entity_types}", flush=True)
    print(f"  Top relations: {', '.join(f'{r}({c})' for r, c in top_5_rels)}", flush=True)
    print(f"  Top entity types: {', '.join(f'{t}({c})' for t, c in top_5_types)}", flush=True)

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_extraction_quality(
    model: str = config.OLLAMA_MODEL,
    suffix: str = "zone2_seaf",
    sample_size: int = DEFAULT_PRECISION_SAMPLES,
) -> dict:
    """Run all 5 extraction quality metrics."""
    print("=" * 60)
    print("EXTRACTION QUALITY EVALUATION (5 metrics, domain-agnostic)")
    print(f"Model: {model} | Suffix: {suffix} | Sample: {sample_size}")
    print("=" * 60)

    graph = _get_graph()
    llm = _get_llm(model)
    chunks = _load_chunks()

    # Metric 1: Vocabulary coverage (domain-agnostic).
    coverage = measure_vocabulary_coverage(graph, chunks)

    # Metric 2: Triple precision (LLM-as-judge).
    precision = measure_triple_precision(graph, llm, sample_size)

    # Metric 3: Fact recall (MINE-1 style, embedding match).
    fact_recall = measure_fact_recall(graph, llm, n_chunks=DEFAULT_RECALL_CHUNKS)

    # Metric 4: Source grounding.
    grounding = measure_source_grounding(graph, llm, sample_size=DEFAULT_GROUNDING_SAMPLES)

    # Metric 5: Graph statistics.
    stats = measure_graph_statistics(graph)

    result = {
        "suffix": suffix,
        "model": model,
        "vocabulary_coverage": coverage,
        "triple_precision": precision,
        "fact_recall": fact_recall,
        "source_grounding": grounding,
        "graph_statistics": stats,
    }

    # Save.
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"extraction_quality_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n✓ Results saved to {out_path}")

    # Summary.
    print(f"\n{'=' * 60}")
    print(f"EXTRACTION QUALITY SUMMARY [{suffix}]")
    print(f"{'=' * 60}")
    print(f"  Vocab Coverage:     {coverage.get('token_coverage', 0):.0%} "
          f"({coverage.get('covered_terms', 0)}/{coverage.get('source_vocab_size', 0)} terms)")
    print(f"  Triple Precision:   {precision['precision']:.0%} "
          f"({precision['correct']}/{precision['correct']+precision['incorrect']} judged correct)")
    print(f"  Fact Recall:        {fact_recall['fact_recall']:.0%} "
          f"({fact_recall['found_facts']}/{fact_recall['total_facts']} facts in KG)")
    print(f"  Source Grounding:   {grounding['grounding_rate']:.0%} "
          f"({grounding.get('supported', 0)}/{grounding['total_checked']} grounded)")
    print(f"  Graph: {stats['node_count']} nodes, {stats['edge_count']} edges, "
          f"{stats['triples_per_entity']} triples/entity")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction Quality Evaluation — 5 domain-agnostic metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics (following KGGen, AutoSchemaKG, GraphJudge):
  1. Vocabulary Coverage: source vocab tokens found in entity names
  2. Triple Precision: LLM judges sample triples (AutoSchemaKG approach)
  3. Fact Recall: LLM extracts facts, embedding match to KG (MINE-1 style)
  4. Source Grounding: traces triples back to source text
  5. Graph Statistics: density, degree distribution, relation types
        """,
    )
    parser.add_argument("--suffix", default="zone2_seaf", help="Result file suffix")
    parser.add_argument("--model", default=config.OLLAMA_MODEL, help="Ollama model")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_PRECISION_SAMPLES,
                        help="Triples to sample for precision")
    args = parser.parse_args()

    run_extraction_quality(model=args.model, suffix=args.suffix, sample_size=args.sample_size)
