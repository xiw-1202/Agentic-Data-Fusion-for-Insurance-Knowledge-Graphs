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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        "question": "What is the maximum building coverage amount under NFIP?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'building' AND (toLower(n.id) CONTAINS 'coverage' OR toLower(n.id) CONTAINS 'limit') RETURN n.id, labels(n) LIMIT 10",
        "keywords": ["500,000", "500000", "limit"],
    },
    {
        "id": 3,
        "category": "coverage",
        "question": "What is the maximum contents coverage amount?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'content' AND (toLower(n.id) CONTAINS 'coverage' OR toLower(n.id) CONTAINS 'limit') RETURN n.id LIMIT 10",
        "keywords": ["500,000", "contents", "limit"],
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
        "question": "What is the waiting period before a new NFIP policy takes effect?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'wait' OR toLower(n.id) CONTAINS 'effective' RETURN n.id LIMIT 10",
        "keywords": ["30", "days", "waiting period"],
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
        "cypher": "MATCH (n) WHERE n.id = 'Flood' OR n.id = 'flood' OR toLower(n.id) CONTAINS 'flood definition' RETURN n.id, labels(n) LIMIT 10",
        "keywords": ["overflow", "surface water", "inundation"],
    },
    {
        "id": 12,
        "category": "definitions",
        "question": "What is the definition of a 'building' under NFIP?",
        "cypher": "MATCH (n) WHERE n.id = 'Building' OR n.id = 'building' RETURN n.id, labels(n) LIMIT 10",
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
        "cypher": "MATCH (n)-[r]->(m) WHERE toLower(n.id) CONTAINS 'content' AND toLower(type(r)) CONTAINS 'cover' RETURN n.id, type(r), m.id LIMIT 20",
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
        "question": "Can a policyholder appeal a claim decision?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'appeal' OR toLower(n.id) CONTAINS 'dispute' RETURN n.id LIMIT 10",
        "keywords": ["appeal", "dispute", "arbitration"],
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

def run_evaluation(suffix: str = "original"):
    """
    Run evaluation against the current Neo4j graph.
    suffix: 'original' | 'zone1'  — determines the output filename.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    label = "Zone1 Chunks (Ablation)" if suffix == "zone1" else "Original 512-Token Chunks"
    print("=" * 60)
    print(f"CS584 Capstone — Baseline Evaluation [{label}]")
    print("=" * 60)

    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )

    metrics = BaselineMetrics()

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

    # --- Save results ---
    report = {
        "mode": suffix,
        "baseline_metrics": asdict(metrics),
        "duplication_detail": dup,
        "type_consistency_detail": tc,
        "task_results": task_results,
    }
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="original",
                        help="Label for this run: 'original' or 'zone1'")
    args = parser.parse_args()
    run_evaluation(suffix=args.suffix)
