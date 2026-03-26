"""Extraction Quality Evaluation
================================
Comprehensive evaluation of Zone 2 extraction quality, following
best practices from KGGen (NeurIPS 2025) and AutoSchemaKG (2025).

Five metrics:
  1. Entity Coverage — what % of key source concepts appear in the KG?
  2. Triple Precision — sample N triples, LLM judges correctness
  3. Fact Recall — given source chunks, what % of facts were captured?
  4. Source Grounding — are triples traceable to source text?
  5. Graph Statistics — density, connectivity, degree distribution

Neither metric uses the reference ontology (Riskine) — they evaluate
raw extraction quality independent of ontology induction.

References:
  KGGen (NeurIPS 2025): MINE benchmark, fact recall, entity extraction accuracy
  AutoSchemaKG (2025): LLM-judged triple P/R/F1, BertScore-Coverage

Usage:
  python3 evaluation/extraction_quality.py --suffix zone3_svloi --model qwen2.5:72b
  python3 evaluation/extraction_quality.py --suffix zone3_svloi --sample-size 50
"""

from __future__ import annotations

import json
import os
import sys
import argparse
import random
import re
from collections import Counter
from typing import Optional

random.seed(42)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

import config


# ---------------------------------------------------------------------------
# Key concepts from SFIP + OpenFEMA (ground truth for entity coverage)
# ---------------------------------------------------------------------------

EXPECTED_CONCEPTS = {
    "coverage_types": [
        "Coverage A", "Coverage B", "Coverage C", "Coverage D",
        "building coverage", "contents coverage", "other coverages",
        "debris removal", "loss avoidance",
    ],
    "policy_terms": [
        "policy", "SFIP", "Standard Flood Insurance Policy",
        "declarations page", "deductible", "premium",
        "waiting period", "policy term",
    ],
    "perils_and_exclusions": [
        "flood", "mudflow", "erosion", "earth movement",
        "water damage", "sewer backup", "gradual damage",
        "land subsidence", "sinkholes",
    ],
    "structures_and_property": [
        "building", "foundation", "basement", "crawlspace",
        "contents", "personal property", "residential",
        "elevated building", "enclosure",
    ],
    "geographic_and_risk": [
        "flood zone", "base flood elevation", "floodplain",
        "community", "county", "state",
        "special flood hazard area",
    ],
    "financial_terms": [
        "replacement cost", "actual cash value", "depreciation",
        "claim", "proof of loss", "maximum coverage",
        "building claim", "contents claim",
    ],
    "obligations_and_procedures": [
        "notice", "inspection", "appraisal",
        "compliance", "elevation certificate",
        "relocation", "demolition",
    ],
}


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
        RETURN s.id AS subject, type(r) AS relation, o.id AS object
    """)
    return [{"subject": r["subject"], "relation": r["relation"], "object": r["object"]}
            for r in rows if r.get("subject") and r.get("object")]


def _load_chunks() -> list[dict]:
    """Load Zone 1 chunks from disk."""
    chunks_path = config.ZONE1_CHUNKS_FILE
    if not os.path.exists(chunks_path):
        print(f"  Warning: chunks not found at {chunks_path}", flush=True)
        return []
    with open(chunks_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric 1: Entity Coverage (same as before but with semantic matching)
# ---------------------------------------------------------------------------

def measure_entity_coverage(graph: Neo4jGraph) -> dict:
    """Check what % of expected concepts appear in the KG.

    Uses case-insensitive substring matching against entity names.
    """
    print("\n[Metric 1] Entity Coverage...", flush=True)

    entity_ids = _get_all_entity_ids(graph)
    entity_ids_lower = [eid.lower() for eid in entity_ids]

    category_results = {}
    total_found = 0
    total_expected = 0

    for category, terms in EXPECTED_CONCEPTS.items():
        found = []
        missing = []
        for term in terms:
            term_lower = term.lower()
            matched = any(term_lower in eid for eid in entity_ids_lower)
            if matched:
                found.append(term)
            else:
                missing.append(term)

        rate = len(found) / len(terms) if terms else 0.0
        category_results[category] = {
            "found": len(found),
            "total": len(terms),
            "rate": round(rate, 3),
            "found_terms": found,
            "missing_terms": missing,
        }
        total_found += len(found)
        total_expected += len(terms)

        status = "OK" if rate >= 0.7 else ("PARTIAL" if rate >= 0.4 else "LOW")
        print(f"  {category}: {len(found)}/{len(terms)} ({rate:.0%}) [{status}]", flush=True)
        if missing:
            print(f"    Missing: {', '.join(missing[:5])}", flush=True)

    overall = total_found / total_expected if total_expected > 0 else 0.0
    print(f"  Overall: {total_found}/{total_expected} ({overall:.0%})", flush=True)

    return {
        "overall_coverage": round(overall, 4),
        "total_found": total_found,
        "total_expected": total_expected,
        "total_entities_in_graph": len(entity_ids),
        "category_coverage": category_results,
    }


# ---------------------------------------------------------------------------
# Metric 2: Triple Precision (LLM-judged, following AutoSchemaKG approach)
# ---------------------------------------------------------------------------

def measure_triple_precision(
    graph: Neo4jGraph,
    llm: ChatOllama,
    sample_size: int = 50,
) -> dict:
    """Sample N triples, LLM judges correctness with structured verification.

    Following AutoSchemaKG's approach: LLM acts as structured verifier,
    judging each triple as CORRECT / INCORRECT / UNCERTAIN.

    Precision = CORRECT / (CORRECT + INCORRECT)
    Accuracy = CORRECT / total
    """
    print(f"\n[Metric 2] Triple Precision (n={sample_size})...", flush=True)

    all_triples = _get_all_triples(graph)
    if not all_triples:
        print("  No triples found.", flush=True)
        return {"precision": 0.0, "accuracy": 0.0, "sample_size": 0}

    sample = random.sample(all_triples, min(sample_size, len(all_triples)))

    batch_size = 10
    results = []

    for batch_start in range(0, len(sample), batch_size):
        batch = sample[batch_start:batch_start + batch_size]

        triple_lines = []
        for i, t in enumerate(batch):
            triple_lines.append(f"{i+1}. ({t['subject']}) --[{t['relation']}]--> ({t['object']})")

        prompt = f"""You are a knowledge graph quality verifier. Judge whether each triple represents a factually plausible relationship.

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

    # Precision excludes UNCERTAIN (following AutoSchemaKG)
    judged = correct + incorrect
    precision = correct / judged if judged > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    print(f"  Correct: {correct}, Incorrect: {incorrect}, Uncertain: {uncertain}", flush=True)
    print(f"  Precision: {precision:.1%} (excl. uncertain) | Accuracy: {accuracy:.1%} (incl. uncertain)", flush=True)

    return {
        "precision": round(precision, 4),
        "accuracy": round(accuracy, 4),
        "sample_size": total,
        "correct": correct,
        "incorrect": incorrect,
        "uncertain": uncertain,
        "sampled_triples": results,
    }


# ---------------------------------------------------------------------------
# Metric 3: Fact Recall (following KGGen MINE approach)
# ---------------------------------------------------------------------------

def measure_fact_recall(
    graph: Neo4jGraph,
    llm: ChatOllama,
    n_chunks: int = 10,
) -> dict:
    """Given source chunks, check what % of key facts were extracted.

    Following KGGen's approach: for each source chunk, ask LLM to list
    the key facts. Then check which facts appear in the KG.

    This measures extraction COMPLETENESS — did we miss important information?
    """
    print(f"\n[Metric 3] Fact Recall (n_chunks={n_chunks})...", flush=True)

    chunks = _load_chunks()
    if not chunks:
        return {"fact_recall": 0.0, "note": "chunks not found"}

    # Sample diverse chunks (prefer longer, content-rich chunks)
    content_chunks = [c for c in chunks if c.get("token_count", 0) > 100]
    if not content_chunks:
        content_chunks = chunks
    sample = random.sample(content_chunks, min(n_chunks, len(content_chunks)))

    # Get all triples for matching
    all_triples = _get_all_triples(graph)
    entity_ids = _get_all_entity_ids(graph)
    entity_ids_lower = {eid.lower() for eid in entity_ids}

    total_facts = 0
    found_facts = 0
    chunk_results = []

    for chunk in sample:
        text = chunk.get("text", "")[:1500]  # limit for prompt budget
        chunk_id = chunk.get("chunk_id", "?")

        # Step 1: LLM extracts key facts from the chunk
        extract_prompt = f"""Read this text and list the 5-10 most important factual statements as simple (subject, relation, object) triples.

TEXT:
{text}

Output one triple per line in format: subject | relation | object
Only include clear factual relationships, not vague statements.
"""
        raw_facts = _invoke_llm(llm, extract_prompt)

        # Parse facts
        facts = []
        for line in raw_facts.strip().splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3 and all(p for p in parts):
                facts.append({"subject": parts[0], "relation": parts[1], "object": parts[2]})

        # Step 2: Check which facts appear in the KG (fuzzy matching)
        chunk_found = 0
        for fact in facts:
            subj_lower = fact["subject"].lower()
            obj_lower = fact["object"].lower()

            # Check: does any entity name contain the subject or object?
            subj_match = any(subj_lower in eid for eid in entity_ids_lower) or any(eid in subj_lower for eid in entity_ids_lower)
            obj_match = any(obj_lower in eid for eid in entity_ids_lower) or any(eid in obj_lower for eid in entity_ids_lower)

            if subj_match and obj_match:
                chunk_found += 1

        total_facts += len(facts)
        found_facts += chunk_found

        recall = chunk_found / len(facts) if facts else 0.0
        chunk_results.append({
            "chunk_id": chunk_id,
            "facts_extracted": len(facts),
            "facts_found_in_kg": chunk_found,
            "recall": round(recall, 3),
        })
        print(f"    Chunk {chunk_id}: {chunk_found}/{len(facts)} facts found ({recall:.0%})", flush=True)

    overall_recall = found_facts / total_facts if total_facts > 0 else 0.0
    print(f"  Overall Fact Recall: {found_facts}/{total_facts} ({overall_recall:.0%})", flush=True)

    return {
        "fact_recall": round(overall_recall, 4),
        "total_facts": total_facts,
        "found_facts": found_facts,
        "chunks_sampled": len(sample),
        "chunk_results": chunk_results,
    }


# ---------------------------------------------------------------------------
# Metric 4: Source Grounding (triple-to-source traceability)
# ---------------------------------------------------------------------------

def measure_source_grounding(
    graph: Neo4jGraph,
    llm: ChatOllama,
    sample_size: int = 30,
) -> dict:
    """Check if extracted triples can be traced back to source text.

    Sample N triples, find their likely source chunk, ask LLM:
    "Does this chunk support this triple?"

    Measures whether extraction is grounded in evidence vs hallucinated.
    """
    print(f"\n[Metric 4] Source Grounding (n={sample_size})...", flush=True)

    chunks = _load_chunks()
    if not chunks:
        return {"grounding_rate": 0.0, "note": "chunks not found"}

    all_triples = _get_all_triples(graph)
    if not all_triples:
        return {"grounding_rate": 0.0, "note": "no triples"}

    sample = random.sample(all_triples, min(sample_size, len(all_triples)))

    # Build simple chunk index for matching
    chunk_texts = [(c.get("chunk_id", "?"), c.get("text", "").lower()) for c in chunks]

    grounded = 0
    not_grounded = 0
    uncertain = 0
    results = []

    batch_size = 5
    for batch_start in range(0, len(sample), batch_size):
        batch = sample[batch_start:batch_start + batch_size]

        for triple in batch:
            subj = triple["subject"].lower()
            obj = triple["object"].lower()

            # Find best matching chunk (contains both subject and object)
            best_chunk_id = None
            best_chunk_text = None
            for cid, ctext in chunk_texts:
                if subj[:20] in ctext and obj[:20] in ctext:
                    best_chunk_id = cid
                    best_chunk_text = ctext[:800]
                    break
            if best_chunk_text is None:
                # Try subject only
                for cid, ctext in chunk_texts:
                    if subj[:20] in ctext:
                        best_chunk_id = cid
                        best_chunk_text = ctext[:800]
                        break

            if best_chunk_text is None:
                results.append({**triple, "grounded": "NO_SOURCE", "chunk_id": None})
                not_grounded += 1
                continue

            # Ask LLM if the chunk supports the triple
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
                grounded += 1
            elif verdict == "PARTIALLY":
                grounded += 0.5
                uncertain += 1
            else:
                not_grounded += 1

        done = min(batch_start + batch_size, len(sample))
        if done % 10 == 0 or done == len(sample):
            print(f"    Checked {done}/{len(sample)}", flush=True)

    total = len(results)
    grounding_rate = grounded / total if total > 0 else 0.0
    print(f"  Grounding rate: {grounding_rate:.1%} ({int(grounded)}/{total})", flush=True)

    return {
        "grounding_rate": round(grounding_rate, 4),
        "total_checked": total,
        "supported": sum(1 for r in results if r["grounded"] == "SUPPORTED"),
        "partially_supported": sum(1 for r in results if r["grounded"] == "PARTIALLY"),
        "not_supported": sum(1 for r in results if r["grounded"] == "NOT_SUPPORTED"),
        "no_source_found": sum(1 for r in results if r["grounded"] == "NO_SOURCE"),
        "sampled_results": results[:20],  # save first 20 for inspection
    }


# ---------------------------------------------------------------------------
# Metric 5: Graph Statistics
# ---------------------------------------------------------------------------

def measure_graph_statistics(graph: Neo4jGraph) -> dict:
    """Compute structural statistics of the extracted KG.

    No LLM needed — pure graph analysis.
    """
    print("\n[Metric 5] Graph Statistics...", flush=True)

    # Node and edge counts
    node_count = graph.query("MATCH (n:Entity) RETURN count(n) AS c")[0]["c"]
    edge_count = graph.query("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS c")[0]["c"]

    # Degree distribution
    degrees = graph.query("""
        MATCH (n:Entity)
        OPTIONAL MATCH (n)-[r]-()
        RETURN n.id AS id, count(r) AS degree
    """)
    degree_values = [r["degree"] for r in degrees]

    if degree_values:
        import numpy as np
        arr = np.array(degree_values)
        avg_degree = float(np.mean(arr))
        median_degree = float(np.median(arr))
        max_degree = int(np.max(arr))
        isolated = int(np.sum(arr == 0))
    else:
        avg_degree = median_degree = max_degree = isolated = 0

    # Relation type distribution
    rel_types = graph.query("""
        MATCH (:Entity)-[r]->(:Entity)
        RETURN type(r) AS rel, count(r) AS cnt
        ORDER BY cnt DESC
    """)
    n_rel_types = len(rel_types)
    top_5_rels = [(r["rel"], r["cnt"]) for r in rel_types[:5]]

    # Entity type distribution
    entity_types = graph.query("""
        MATCH (n:Entity)
        RETURN n.entity_type AS etype, count(n) AS cnt
        ORDER BY cnt DESC
    """)
    n_entity_types = len([e for e in entity_types if e.get("etype") and e["etype"] != "Unknown"])

    # Density = edges / (nodes * (nodes-1))
    density = edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0.0

    # Connected components (approximate via largest cluster)
    try:
        components = graph.query("""
            MATCH (n:Entity)
            WHERE NOT exists((n)--(:Entity))
            RETURN count(n) AS isolated_nodes
        """)
        isolated_from_query = components[0]["isolated_nodes"] if components else 0
    except Exception:
        isolated_from_query = isolated

    stats = {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 2),
        "median_degree": round(median_degree, 1),
        "max_degree": max_degree,
        "isolated_nodes": isolated_from_query,
        "relation_types": n_rel_types,
        "entity_types": n_entity_types,
        "top_5_relations": top_5_rels,
        "triples_per_entity": round(edge_count / node_count, 2) if node_count > 0 else 0,
    }

    print(f"  Nodes: {node_count} | Edges: {edge_count} | Density: {density:.6f}", flush=True)
    print(f"  Avg degree: {avg_degree:.1f} | Median: {median_degree:.0f} | Max: {max_degree}", flush=True)
    print(f"  Relation types: {n_rel_types} | Entity types: {n_entity_types}", flush=True)
    print(f"  Top relations: {', '.join(f'{r}({c})' for r, c in top_5_rels)}", flush=True)

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_extraction_quality(
    model: str = config.OLLAMA_MODEL,
    suffix: str = "zone3_svloi",
    sample_size: int = 50,
) -> dict:
    """Run all 5 extraction quality metrics."""
    print("=" * 60)
    print("EXTRACTION QUALITY EVALUATION (5 metrics)")
    print(f"Model: {model} | Suffix: {suffix} | Sample: {sample_size}")
    print("=" * 60)

    graph = _get_graph()
    llm = _get_llm(model)

    # Metric 1: Entity coverage
    coverage = measure_entity_coverage(graph)

    # Metric 2: Triple precision
    precision = measure_triple_precision(graph, llm, sample_size)

    # Metric 3: Fact recall
    fact_recall = measure_fact_recall(graph, llm, n_chunks=10)

    # Metric 4: Source grounding
    grounding = measure_source_grounding(graph, llm, sample_size=30)

    # Metric 5: Graph statistics
    stats = measure_graph_statistics(graph)

    # Combined result
    result = {
        "suffix": suffix,
        "model": model,
        "entity_coverage": coverage,
        "triple_precision": precision,
        "fact_recall": fact_recall,
        "source_grounding": grounding,
        "graph_statistics": stats,
    }

    # Save
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"extraction_quality_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n✓ Results saved to {out_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"EXTRACTION QUALITY SUMMARY [{suffix}]")
    print(f"{'=' * 60}")
    print(f"  Entity Coverage:    {coverage['overall_coverage']:.0%} ({coverage['total_found']}/{coverage['total_expected']} concepts)")
    print(f"  Triple Precision:   {precision['precision']:.0%} ({precision['correct']}/{precision['correct']+precision['incorrect']} judged correct)")
    print(f"  Fact Recall:        {fact_recall['fact_recall']:.0%} ({fact_recall['found_facts']}/{fact_recall['total_facts']} facts in KG)")
    print(f"  Source Grounding:   {grounding['grounding_rate']:.0%} ({grounding['supported']}/{grounding['total_checked']} grounded)")
    print(f"  Graph Density:      {stats['density']:.6f} ({stats['node_count']} nodes, {stats['edge_count']} edges)")
    print(f"  Triples/Entity:     {stats['triples_per_entity']}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction Quality Evaluation — 5 metrics following KGGen/AutoSchemaKG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics:
  1. Entity Coverage: substring match against expected concepts
  2. Triple Precision: LLM judges sample triples (AutoSchemaKG approach)
  3. Fact Recall: LLM extracts facts from source, checks if in KG (KGGen approach)
  4. Source Grounding: traces triples back to source text
  5. Graph Statistics: density, degree distribution, relation types
        """,
    )
    parser.add_argument("--suffix", default="zone3_svloi", help="Result file suffix")
    parser.add_argument("--model", default=config.OLLAMA_MODEL, help="Ollama model")
    parser.add_argument("--sample-size", type=int, default=50, help="Triples to sample")
    args = parser.parse_args()

    run_extraction_quality(model=args.model, suffix=args.suffix, sample_size=args.sample_size)
