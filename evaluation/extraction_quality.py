"""Extraction Quality Evaluation
================================
Measures how well Zone 2 extraction captures the source document content.

Two complementary metrics:
  1. Entity Coverage — what % of key source concepts appear in the KG?
  2. Relation Accuracy — sample N triples, LLM judges correctness.

Neither metric uses the reference ontology (Riskine) — they evaluate the
raw extraction quality independent of ontology induction.

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
from collections import Counter

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
# These are domain-agnostic concept CATEGORIES, not Riskine classes.
# Each category has example terms that should appear in ANY insurance KG.
# The pipeline is expected to extract entities matching these terms.

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


def _get_all_entity_ids(graph: Neo4jGraph) -> list[str]:
    """Fetch all Entity node IDs from Neo4j."""
    rows = graph.query("MATCH (n:Entity) RETURN n.id AS id")
    return [r["id"] for r in rows if r.get("id")]


def _get_sample_triples(graph: Neo4jGraph, n: int = 50) -> list[dict]:
    """Sample N random triples from the graph."""
    rows = graph.query("""
        MATCH (s:Entity)-[r]->(o:Entity)
        RETURN s.id AS subject, type(r) AS relation, o.id AS object
    """)
    if not rows:
        return []
    sample = random.sample(rows, min(n, len(rows)))
    return [{"subject": r["subject"], "relation": r["relation"], "object": r["object"]} for r in sample]


# ---------------------------------------------------------------------------
# Metric 1: Entity Coverage
# ---------------------------------------------------------------------------

def measure_entity_coverage(graph: Neo4jGraph) -> dict:
    """Check what % of expected concepts appear in the KG.

    For each concept term, check if ANY entity name contains it
    (case-insensitive substring match).

    Returns:
        overall_coverage: float (0-1)
        category_coverage: {category: {found: int, total: int, rate: float, missing: [...]}}
        total_found: int
        total_expected: int
    """
    print("\n[Extraction Quality] Measuring entity coverage...", flush=True)

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
            # Check if any entity name contains this term
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
# Metric 2: Relation Accuracy (LLM-judged)
# ---------------------------------------------------------------------------

def measure_relation_accuracy(
    graph: Neo4jGraph,
    llm: ChatOllama,
    sample_size: int = 50,
) -> dict:
    """Sample N triples and ask LLM to judge if each is factually correct.

    The LLM sees the triple (subject, relation, object) and judges:
      CORRECT — the relationship is factually plausible
      INCORRECT — the relationship is factually wrong
      UNCERTAIN — can't determine without more context

    Returns:
        accuracy: float (0-1, CORRECT / total)
        sample_size: int
        correct: int
        incorrect: int
        uncertain: int
        sampled_triples: [{subject, relation, object, verdict}]
    """
    print(f"\n[Extraction Quality] Measuring relation accuracy (n={sample_size})...", flush=True)

    triples = _get_sample_triples(graph, sample_size)
    if not triples:
        print("  No triples found in graph.", flush=True)
        return {"accuracy": 0.0, "sample_size": 0, "correct": 0, "incorrect": 0, "uncertain": 0, "sampled_triples": []}

    # Batch triples for LLM judgment (10 per prompt)
    batch_size = 10
    results = []

    for batch_start in range(0, len(triples), batch_size):
        batch = triples[batch_start:batch_start + batch_size]

        triple_lines = []
        for i, t in enumerate(batch):
            triple_lines.append(f"{i+1}. ({t['subject']}) --[{t['relation']}]--> ({t['object']})")

        prompt = f"""Judge whether each knowledge graph triple is factually plausible in an insurance context.

TRIPLES:
{chr(10).join(triple_lines)}

For each triple, respond with the number and one of:
  CORRECT — the relationship makes sense (e.g., "Policy COVERS Building" is correct)
  INCORRECT — the relationship is wrong (e.g., "Flood Zone COVERS Person" is wrong)
  UNCERTAIN — can't determine without more context

Output format (one per line):
1. CORRECT
2. INCORRECT
...
"""
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            print(f"    LLM error: {e}", flush=True)
            raw = ""

        # Parse verdicts
        import re
        for i, t in enumerate(batch):
            verdict = "UNCERTAIN"
            pattern = rf'{i+1}\.\s*(CORRECT|INCORRECT|UNCERTAIN)'
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                verdict = m.group(1).upper()
            results.append({**t, "verdict": verdict})

        done = min(batch_start + batch_size, len(triples))
        print(f"    Judged {done}/{len(triples)} triples", flush=True)

    # Tally
    verdicts = Counter(r["verdict"] for r in results)
    correct = verdicts.get("CORRECT", 0)
    incorrect = verdicts.get("INCORRECT", 0)
    uncertain = verdicts.get("UNCERTAIN", 0)
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    print(f"  Results: {correct} correct, {incorrect} incorrect, {uncertain} uncertain", flush=True)
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})", flush=True)

    return {
        "accuracy": round(accuracy, 4),
        "sample_size": total,
        "correct": correct,
        "incorrect": incorrect,
        "uncertain": uncertain,
        "sampled_triples": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_extraction_quality(
    model: str = config.OLLAMA_MODEL,
    suffix: str = "zone3_svloi",
    sample_size: int = 50,
) -> dict:
    """Run both extraction quality metrics."""
    print("=" * 60)
    print("EXTRACTION QUALITY EVALUATION")
    print(f"Model: {model} | Suffix: {suffix} | Sample: {sample_size}")
    print("=" * 60)

    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )
    llm = ChatOllama(model=model, base_url=config.OLLAMA_BASE_URL, temperature=0)

    # Metric 1: Entity coverage
    coverage = measure_entity_coverage(graph)

    # Metric 2: Relation accuracy
    rel_accuracy = measure_relation_accuracy(graph, llm, sample_size)

    # Combined result
    result = {
        "suffix": suffix,
        "model": model,
        "entity_coverage": coverage,
        "relation_accuracy": rel_accuracy,
    }

    # Save
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"extraction_quality_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"EXTRACTION QUALITY SUMMARY [{suffix}]")
    print(f"{'=' * 60}")
    print(f"  Entity Coverage:    {coverage['overall_coverage']:.0%} ({coverage['total_found']}/{coverage['total_expected']} concepts found)")
    print(f"  Relation Accuracy:  {rel_accuracy['accuracy']:.0%} ({rel_accuracy['correct']}/{rel_accuracy['sample_size']} triples correct)")
    print(f"  Total entities:     {coverage['total_entities_in_graph']}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction Quality Evaluation")
    parser.add_argument("--suffix", default="zone3_svloi", help="Result file suffix")
    parser.add_argument("--model", default=config.OLLAMA_MODEL, help="Ollama model")
    parser.add_argument("--sample-size", type=int, default=50, help="Triples to sample for accuracy")
    args = parser.parse_args()

    run_extraction_quality(model=args.model, suffix=args.suffix, sample_size=args.sample_size)
