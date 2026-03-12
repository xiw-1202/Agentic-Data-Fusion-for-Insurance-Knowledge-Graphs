"""
Baseline Evaluation — measures quality of the raw LLMGraphTransformer graph.

Metrics (per project plan Section 4 / Appendix D):
  1. Entity duplication rate  — duplicate nodes for same real-world entity
  2. Type consistency         — unique label count per concept
  3. Schema coherence         — detected via label proliferation
  4. Query accuracy           — 20 tasks against the NFIP SFIP policy

Run AFTER baseline/pipeline.py has populated Neo4j.
"""

import json
import os
import sys
import re
from dataclasses import dataclass, asdict

# Allow imports from project root (config.py lives there)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# Allow imports from evaluation/ directory (riskine_loader, riskine_eval)
_EVAL_DIR = os.path.join(_ROOT, "evaluation")
sys.path.insert(0, _EVAL_DIR)

from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

import config


# ---------------------------------------------------------------------------
# 20 Evaluation Tasks
# Each task has:
#   question   — natural-language question about NFIP policy
#   cypher     — Cypher query to attempt on the graph
#   keywords   — expected keywords/concepts in a correct answer
# ---------------------------------------------------------------------------

EVAL_TASKS = [
    {
        "id": 1,
        "category": "coverage",
        "question": "What types of property does NFIP cover under the General Property Form?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'property' OR toLower(n.id) CONTAINS 'coverage' RETURN DISTINCT n.id, labels(n) LIMIT 20",
        "keywords": ["building", "contents", "property"],
    },
    {
        "id": 2,
        "category": "coverage",
        # NOTE: $250,000 building limit is NOT in the 49 SFIP PDF chunks (it's on the
        # Declarations Page). Q2 now asks about the ICC Coverage D limit ($30,000),
        # which IS present in chunk 13.
        "question": "What is the maximum limit for Increased Cost of Compliance (ICC / Coverage D) under NFIP?",
        "cypher": "MATCH (n)-[r]->(m) WHERE toLower(n.id) CONTAINS 'icc' OR toLower(n.id) CONTAINS 'compliance' OR toLower(n.id) CONTAINS 'coverage d' RETURN n.id, type(r), m.id LIMIT 10",
        "keywords": ["30,000", "30000", "icc", "compliance"],
    },
    {
        "id": 3,
        "category": "coverage",
        # NOTE: $100,000 contents limit is NOT in the 49 SFIP PDF chunks. Q3 now asks
        # about Coverage B (Personal Property) scope, grounded in chunk 11.
        "question": "What personal property is insured under Coverage B of the NFIP SFIP?",
        "cypher": "MATCH (n)-[r]->(m) WHERE toLower(n.id) CONTAINS 'coverage b' OR (toLower(n.id) CONTAINS 'personal property' AND toLower(type(r)) CONTAINS 'cover') RETURN n.id, type(r), m.id LIMIT 10",
        "keywords": ["personal property", "contents", "furniture", "clothing"],
    },
    {
        "id": 4,
        "category": "exclusions",
        "question": "What types of damage are excluded from NFIP coverage?",
        "cypher": "MATCH (n)-[r]->(m) WHERE toLower(type(r)) CONTAINS 'exclud' OR toLower(n.id) CONTAINS 'exclud' RETURN n.id, type(r), m.id LIMIT 20",
        "keywords": ["earth movement", "sewer", "moisture", "mold"],
    },
    {
        "id": 5,
        "category": "exclusions",
        "question": "Is flood damage from earth movement covered?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'earth' OR toLower(n.id) CONTAINS 'sinkhole' RETURN n.id, labels(n) LIMIT 10",
        "keywords": ["earth movement", "excluded", "not covered"],
    },
    {
        "id": 6,
        "category": "policy_terms",
        # NOTE: The 30-day new-policy waiting period is in the NFIA statute, not in the
        # SFIP policy form text (0 SFIP chunk hits). Q6 now asks about the 90-day
        # continuous lake flooding provision, which IS in chunk 39.
        "question": "Under NFIP, how many continuous days must a lake flood an insured building before a pre-loss claim may be filed?",
        "cypher": "MATCH (n)-[r]->(m) WHERE toLower(type(r)) CONTAINS 'wait' OR toLower(n.id) CONTAINS 'lake' OR toLower(m.id) CONTAINS 'lake' OR toLower(m.id) CONTAINS '90' RETURN n.id, type(r), m.id LIMIT 10",
        "keywords": ["90", "days", "lake", "continuous"],
    },
    {
        "id": 7,
        "category": "policy_terms",
        "question": "What flood zones are mentioned in the SFIP?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'zone' OR n.id =~ '.*[AV][0-9]*.*' RETURN DISTINCT n.id, labels(n) LIMIT 20",
        "keywords": ["zone", "flood zone", "A", "V"],
    },
    {
        "id": 8,
        "category": "policy_terms",
        "question": "What is Increased Cost of Compliance (ICC) coverage?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'icc' OR toLower(n.id) CONTAINS 'compliance' RETURN n.id, labels(n) LIMIT 10",
        "keywords": ["ICC", "compliance", "increased cost"],
    },
    {
        "id": 9,
        "category": "claims",
        "question": "What must a policyholder do after a flood loss?",
        "cypher": "MATCH (n)-[r]->(m) WHERE toLower(n.id) CONTAINS 'loss' OR toLower(type(r)) CONTAINS 'notif' OR toLower(type(r)) CONTAINS 'report' RETURN n.id, type(r), m.id LIMIT 20",
        "keywords": ["notify", "report", "proof of loss", "60 days"],
    },
    {
        "id": 10,
        "category": "claims",
        "question": "What is the deadline for filing a proof of loss?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'proof' OR toLower(n.id) CONTAINS 'deadline' RETURN n.id LIMIT 10",
        "keywords": ["60 days", "proof of loss"],
    },
    {
        "id": 11,
        "category": "definitions",
        "question": "How does NFIP define 'flood'?",
        # FIX: SFIP definitions are triples: Flood -[DEFINED_AS]-> "description".
        # Keywords (overflow, inundation, surface water) are in the OBJECT node, not
        # the subject. Must traverse the definition edge to find the description text.
        "cypher": "MATCH (n)-[r]->(m) WHERE (n.id = 'Flood' OR toLower(n.id) CONTAINS 'flood') AND (toLower(type(r)) CONTAINS 'defin' OR toLower(type(r)) CONTAINS 'mean') RETURN n.id, type(r), m.id LIMIT 10",
        "keywords": ["overflow", "surface water", "inundation"],
    },
    {
        "id": 12,
        "category": "definitions",
        "question": "What is the definition of a 'building' under NFIP?",
        # FIX: Same structural fix as Q11 — keywords (walled, roofed, structure) are
        # in the object node of the DEFINED_AS triple, not the Building subject node.
        "cypher": "MATCH (n)-[r]->(m) WHERE (n.id = 'Building' OR toLower(n.id) CONTAINS 'building') AND (toLower(type(r)) CONTAINS 'defin' OR toLower(type(r)) CONTAINS 'mean') RETURN n.id, type(r), m.id LIMIT 10",
        "keywords": ["walled", "roofed", "structure"],
    },
    {
        "id": 13,
        "category": "coverage",
        "question": "Does NFIP cover basement contents?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'basement' RETURN n.id, labels(n) LIMIT 10",
        "keywords": ["basement", "limited", "contents"],
    },
    {
        "id": 14,
        "category": "coverage",
        "question": "What personal property is covered under contents coverage?",
        # FIX: SFIP calls contents coverage "Coverage B". Current query only matched
        # nodes whose ID contains 'content' — misses nodes named "Coverage B" or
        # "Personal Property" that are the actual COVERS subjects/objects.
        "cypher": "MATCH (n)-[r]->(m) WHERE (toLower(n.id) CONTAINS 'content' OR toLower(n.id) CONTAINS 'coverage b' OR toLower(n.id) CONTAINS 'personal property') AND toLower(type(r)) CONTAINS 'cover' RETURN n.id, type(r), m.id LIMIT 20",
        "keywords": ["furniture", "clothing", "appliances", "personal property"],
    },
    {
        "id": 15,
        "category": "policy_terms",
        "question": "What happens if there is a coinsurance or other insurance?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'coinsuranc' OR toLower(n.id) CONTAINS 'other insurance' RETURN n.id LIMIT 10",
        "keywords": ["coinsurance", "other insurance", "pro rata"],
    },
    {
        "id": 16,
        "category": "claims",
        # NOTE: SFIP uses "Appraisal" (not "appeal/arbitration") as its dispute mechanism.
        # "appraisal" appears in SFIP chunks 32 and 35; "appeal" has 0 hits.
        "question": "What is the appraisal process for resolving disputes about the amount of an NFIP loss?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'appraisal' OR toLower(n.id) CONTAINS 'dispute' OR toLower(n.id) CONTAINS 'umpire' RETURN n.id, labels(n) LIMIT 10",
        "keywords": ["appraisal", "appraiser", "umpire", "dispute"],
    },
    {
        "id": 17,
        "category": "coverage",
        "question": "What improvements and betterments are covered?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'improvement' OR toLower(n.id) CONTAINS 'betterment' RETURN n.id LIMIT 10",
        "keywords": ["improvements", "betterments", "tenant"],
    },
    {
        "id": 18,
        "category": "exclusions",
        "question": "What financial losses are excluded from NFIP (e.g., business interruption)?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'business' OR toLower(n.id) CONTAINS 'interruption' OR toLower(n.id) CONTAINS 'profit' RETURN n.id LIMIT 10",
        "keywords": ["business interruption", "loss of use", "excluded"],
    },
    {
        "id": 19,
        "category": "policy_terms",
        "question": "How is the replacement cost value (RCV) determined for building coverage?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'replacement' OR toLower(n.id) CONTAINS 'rcv' OR toLower(n.id) CONTAINS 'actual cash' RETURN n.id LIMIT 10",
        "keywords": ["replacement cost", "actual cash value", "ACV"],
    },
    {
        "id": 20,
        "category": "policy_terms",
        "question": "What is the Liberalization Clause in NFIP?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'liberaliz' RETURN n.id LIMIT 10",
        "keywords": ["liberalization", "broadens", "coverage"],
    },
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class BaselineMetrics:
    total_nodes: int = 0
    total_relationships: int = 0
    unique_labels: int = 0
    # Duplication
    duplicate_node_count: int = 0
    duplication_rate: float = 0.0
    # Type consistency
    concepts_with_multiple_labels: int = 0
    type_inconsistency_rate: float = 0.0
    # Query accuracy
    tasks_total: int = 0
    tasks_with_results: int = 0
    tasks_keyword_match: int = 0
    query_accuracy: float = 0.0
    # Model
    model: str = config.OLLAMA_MODEL
    # Riskine ontology alignment (optional — only when --riskine is passed)
    riskine_precision: float = 0.0
    riskine_recall: float = 0.0
    riskine_f1: float = 0.0
    riskine_available: bool = False


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def measure_duplication(graph: Neo4jGraph) -> dict:
    """
    Count nodes that share the same lowercased id — these are duplicates.
    The baseline creates raw nodes with no entity resolution, so e.g.
    'Policy 5123' and 'policy #5123' become separate nodes.
    """
    result = graph.query("""
        MATCH (n:__Entity__)
        WITH toLower(trim(n.id)) AS normalized_id, count(n) AS cnt
        WHERE cnt > 1
        RETURN count(normalized_id) AS duplicate_groups,
               sum(cnt) AS total_duplicated_nodes,
               sum(cnt - 1) AS excess_nodes
    """)
    return result[0] if result else {"duplicate_groups": 0, "total_duplicated_nodes": 0, "excess_nodes": 0}


def measure_type_consistency(graph: Neo4jGraph) -> dict:
    """
    Count how many different labels exist for similar concepts.
    High label proliferation = low type consistency (baseline weakness).
    """
    all_labels = graph.query("""
        CALL db.labels() YIELD label
        WHERE label <> '__Entity__' AND label <> 'Document'
        RETURN label
    """)
    labels = [r["label"] for r in all_labels]

    # Group labels by their first meaningful token as a proxy for concept.
    # Split on underscore, space, or camelCase boundary (but strip leading empty strings).
    from collections import defaultdict
    root_groups: dict = defaultdict(list)
    for label in labels:
        tokens = [t for t in re.split(r'_| |(?<=[a-z])(?=[A-Z])', label) if t]
        root = tokens[0].lower() if tokens else label.lower()
        root_groups[root].append(label)

    multi_label_concepts = {k: v for k, v in root_groups.items() if len(v) > 1}
    return {
        "total_labels": len(labels),
        "unique_root_concepts": len(root_groups),
        "concepts_with_multiple_labels": len(multi_label_concepts),
        "examples": {k: v for k, v in list(multi_label_concepts.items())[:5]},
    }


def run_query_tasks(graph: Neo4jGraph) -> list[dict]:
    """Run all 20 evaluation tasks and record results."""
    results = []
    for task in EVAL_TASKS:
        try:
            rows = graph.query(task["cypher"])
            result_text = " ".join(str(r) for r in rows).lower()
            keyword_hit = any(kw.lower() in result_text for kw in task["keywords"])
            results.append({
                "id": task["id"],
                "category": task["category"],
                "question": task["question"],
                "rows_returned": len(rows),
                "has_results": len(rows) > 0,
                "keyword_match": keyword_hit,
                "sample": rows[:3],
            })
        except Exception as e:
            results.append({
                "id": task["id"],
                "category": task["category"],
                "question": task["question"],
                "rows_returned": 0,
                "has_results": False,
                "keyword_match": False,
                "error": str(e),
            })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(suffix: str = "original", run_riskine: bool = False, model: str = config.OLLAMA_MODEL):
    """
    Run evaluation against the current Neo4j graph.

    Args:
        suffix:       'original' | 'zone1' | 'zone1_qwen' — output filename label
        run_riskine:  if True, run step [5/5] Riskine alignment (slow, needs Ollama)
        model:        Ollama model name for LLM judge in Riskine step
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    label = "Zone1 Chunks (Ablation)" if suffix.startswith("zone1") else "Original 512-Token Chunks"
    print("=" * 60)
    print(f"CS584 Capstone — Baseline Evaluation [{label}]")
    print(f"  suffix={suffix}  model={model}  riskine={'yes' if run_riskine else 'no'}")
    print("=" * 60)

    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )

    metrics = BaselineMetrics(model=model)

    # --- Graph size ---
    print("\n[1/4] Graph size...")
    counts = graph.query("MATCH (n) RETURN count(n) AS nodes")
    rel_counts = graph.query("MATCH ()-[r]->() RETURN count(r) AS rels")
    label_counts = graph.query("CALL db.labels() YIELD label RETURN count(label) AS cnt")
    metrics.total_nodes = counts[0]["nodes"]
    metrics.total_relationships = rel_counts[0]["rels"]
    metrics.unique_labels = label_counts[0]["cnt"]
    print(f"  Nodes: {metrics.total_nodes}  Rels: {metrics.total_relationships}  Labels: {metrics.unique_labels}")

    # --- Duplication ---
    print("\n[2/4] Measuring entity duplication...")
    dup = measure_duplication(graph)
    total_entities = graph.query("MATCH (n:__Entity__) RETURN count(n) AS cnt")[0]["cnt"]
    metrics.duplicate_node_count = dup.get("excess_nodes", 0)
    metrics.duplication_rate = (
        dup.get("excess_nodes", 0) / total_entities if total_entities > 0 else 0.0
    )
    print(f"  Duplicate groups: {dup.get('duplicate_groups', 0)}")
    print(f"  Excess (duplicate) nodes: {dup.get('excess_nodes', 0)}")
    print(f"  Duplication rate: {metrics.duplication_rate:.1%}  (target for novel: <5%)")

    # --- Type consistency ---
    print("\n[3/4] Measuring type consistency...")
    tc = measure_type_consistency(graph)
    metrics.concepts_with_multiple_labels = tc["concepts_with_multiple_labels"]
    metrics.type_inconsistency_rate = (
        tc["concepts_with_multiple_labels"] / tc["unique_root_concepts"]
        if tc["unique_root_concepts"] > 0 else 0.0
    )
    print(f"  Total labels: {tc['total_labels']}")
    print(f"  Root concepts: {tc['unique_root_concepts']}")
    print(f"  Concepts with multiple labels: {tc['concepts_with_multiple_labels']}")
    print(f"  Type inconsistency rate: {metrics.type_inconsistency_rate:.1%}")
    if tc["examples"]:
        print(f"  Examples: {tc['examples']}")

    # --- Query accuracy (20 tasks) ---
    print("\n[4/4] Running 20 evaluation tasks...")
    task_results = run_query_tasks(graph)
    metrics.tasks_total = len(task_results)
    metrics.tasks_with_results = sum(1 for t in task_results if t["has_results"])
    metrics.tasks_keyword_match = sum(1 for t in task_results if t["keyword_match"])
    metrics.query_accuracy = metrics.tasks_keyword_match / metrics.tasks_total

    print(f"\n  {'ID':>3}  {'Cat':<12}  {'Results':>7}  {'Match':>5}  Question")
    print(f"  {'-'*3}  {'-'*12}  {'-'*7}  {'-'*5}  {'-'*40}")
    for t in task_results:
        match_sym = "✓" if t["keyword_match"] else "✗"
        rows = t.get("rows_returned", 0)
        print(f"  {t['id']:>3}  {t['category']:<12}  {rows:>7}  {match_sym:>5}  {t['question'][:55]}")

    print(f"\n  Tasks with any results:   {metrics.tasks_with_results}/{metrics.tasks_total}")
    print(f"  Tasks keyword-matched:    {metrics.tasks_keyword_match}/{metrics.tasks_total}")
    print(f"  Query accuracy:           {metrics.query_accuracy:.1%}  (project target: >75%)")

    # --- Riskine alignment (optional) ---
    riskine_detail: dict = {}
    if run_riskine:
        print("\n[5/5] Riskine ontology alignment...")
        import riskine_loader
        import riskine_eval
        schemas = riskine_loader.fetch_and_cache()
        riskine_classes = riskine_loader.extract_riskine_classes(schemas)
        llm = ChatOllama(model=model, base_url=config.OLLAMA_BASE_URL, temperature=0)
        riskine_result = riskine_eval.measure_riskine_alignment(
            graph, llm, riskine_classes, suffix=suffix
        )
        metrics.riskine_precision = riskine_result["precision"]
        metrics.riskine_recall    = riskine_result["recall"]
        metrics.riskine_f1        = riskine_result["f1"]
        metrics.riskine_available = True
        riskine_detail = riskine_result
        print(f"  Riskine P={metrics.riskine_precision:.3f}  "
              f"R={metrics.riskine_recall:.3f}  F1={metrics.riskine_f1:.3f}")

    # --- Load ontology induction detail from pipeline run summary (if available) ---
    ontology_detail: dict = {}
    run_summary_path = os.path.join(
        config.RESULTS_DIR, f"baseline_run_summary_{suffix}.json"
    )
    if os.path.exists(run_summary_path):
        try:
            with open(run_summary_path, encoding="utf-8") as _f:
                _summary = json.load(_f)
            ontology_detail = _summary.get("ontology_induction", {})
            if ontology_detail:
                print(f"\n  [loaded ontology_induction from run summary: "
                      f"{ontology_detail.get('labels_mapped', 0)}/{ontology_detail.get('labels_seen', 0)} labels mapped]")
        except Exception:
            pass

    # --- Save results ---
    report = {
        "mode": suffix,
        "baseline_metrics": asdict(metrics),
        "duplication_detail": dup,
        "type_consistency_detail": tc,
        "task_results": task_results,
    }
    if ontology_detail:
        report["ontology_induction_detail"] = ontology_detail
    if riskine_detail:
        report["riskine_detail"] = riskine_detail
    out_path = os.path.join(config.RESULTS_DIR, f"baseline_eval_results_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("BASELINE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Nodes:              {metrics.total_nodes}")
    print(f"  Relationships:      {metrics.total_relationships}")
    print(f"  Unique labels:      {metrics.unique_labels}")
    print(f"  Duplication rate:   {metrics.duplication_rate:.1%}")
    print(f"  Type inconsistency: {metrics.type_inconsistency_rate:.1%}")
    print(f"  Query accuracy:     {metrics.query_accuracy:.1%}")
    print(f"\n  Full results → {out_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Baseline KG evaluation harness")
    parser.add_argument("--suffix", default="original",
                        help="Label for this run: 'original', 'zone1', 'zone1_qwen', ...")
    parser.add_argument("--riskine", action="store_true",
                        help="Run Riskine ontology alignment step [5/5] (slow, needs Ollama)")
    parser.add_argument("--model", default=config.OLLAMA_MODEL,
                        help=f"Ollama model for Riskine LLM judge (default: {config.OLLAMA_MODEL})")
    args = parser.parse_args()
    run_evaluation(suffix=args.suffix, run_riskine=args.riskine, model=args.model)
