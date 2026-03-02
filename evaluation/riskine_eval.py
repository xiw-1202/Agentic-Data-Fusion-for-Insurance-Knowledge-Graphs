"""
evaluation/riskine_eval.py
==========================
Riskine Insurance Ontology Alignment Evaluation.

Computes Precision/Recall/F1 between:
  - Induced labels in the Neo4j graph (from LLMGraphTransformer)
  - Ground-truth Riskine ontology classes (10 flood-relevant schemas)

Algorithm:
  1. Query live Neo4j for all labels (CALL db.labels())
  2. Embed induced labels + Riskine class names with all-MiniLM-L6-v2
  3. Cosine similarity filter (>= CANDIDATE_THRESHOLD) for candidate pairs
  4. LLM judge for each candidate pair (MATCH / PARTIAL / NO_MATCH)
  5. Score: MATCH=1.0, PARTIAL=0.5, NO_MATCH=0.0
  6. Precision = sum(scores) / len(induced_labels)
  7. Recall = riskine_classes_covered / 10
  8. F1 = 2*P*R / (P+R) if P+R > 0 else 0

Saves: data/results/riskine_eval_{suffix}.json
CLI:   python3 evaluation/riskine_eval.py --suffix zone1
"""

import json
import os
import sys
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Path setup — add project root and evaluation dir
# ---------------------------------------------------------------------------

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_EVAL_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _EVAL_DIR)

from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

import config
import riskine_loader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANDIDATE_THRESHOLD = 0.60   # cosine similarity to pass candidate pair to LLM judge
PARTIAL_WEIGHT = 0.5         # PARTIAL match contributes 0.5 to precision score

# Labels to skip — LangChain infrastructure labels
EXCLUDED_LABELS = {"__Entity__", "Document"}


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_model_cache: SentenceTransformer | None = None


def embed_labels(labels: list[str]) -> np.ndarray:
    """
    Embed a list of label strings with all-MiniLM-L6-v2.
    Returns L2-normalized embeddings of shape (len(labels), 384).
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache.encode(labels, convert_to_numpy=True, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _llm_judge(llm: ChatOllama, induced: str, riskine_class: str, properties: list[str]) -> str:
    """
    Ask the LLM whether induced and riskine_class refer to the same conceptual category.
    Returns: "MATCH" | "PARTIAL" | "NO_MATCH"
    """
    props_str = ", ".join(properties[:8])  # show up to 8 properties to avoid token bloat
    prompt = (
        "Insurance ontology alignment task.\n"
        f"Induced class:  {induced}\n"
        f"Riskine class:  {riskine_class} (properties: {props_str})\n"
        "Do these refer to the same conceptual category?\n"
        "Answer with exactly one of: MATCH / PARTIAL / NO_MATCH"
    )
    try:
        response = llm.invoke(prompt)
        text = response.content.strip().upper()
        # Parse carefully — "NO_MATCH" contains "MATCH", check it first
        if "NO_MATCH" in text or ("NO" in text.split() and "MATCH" in text.split()):
            return "NO_MATCH"
        elif "MATCH" in text:
            return "MATCH"
        elif "PARTIAL" in text:
            return "PARTIAL"
        else:
            return "NO_MATCH"
    except Exception as e:
        print(f"    [riskine] LLM judge error: {e}")
        return "NO_MATCH"


# ---------------------------------------------------------------------------
# Main alignment measurement
# ---------------------------------------------------------------------------

def measure_riskine_alignment(
    graph: Neo4jGraph,
    llm: ChatOllama,
    riskine_classes: list[dict],
    suffix: str = "zone1",
) -> dict:
    """
    Compute Riskine alignment P/R/F1 against the live Neo4j graph.

    Returns a dict with:
        precision, recall, f1,
        induced_label_count, riskine_class_count, riskine_covered_count,
        alignment_table, unmatched_induced, unmatched_riskine

    Also saves: data/results/riskine_eval_{suffix}.json
    """
    print("  Querying Neo4j for induced labels...")
    rows = graph.query("CALL db.labels() YIELD label RETURN label")
    induced_labels = [
        r["label"] for r in rows
        if r["label"] not in EXCLUDED_LABELS
    ]

    if not induced_labels:
        print("  [riskine] WARNING: No induced labels found in graph.")
        return {
            "suffix": suffix,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "induced_label_count": 0,
            "riskine_class_count": len(riskine_classes),
            "riskine_covered_count": 0,
            "alignment_table": [],
            "unmatched_induced": [],
            "unmatched_riskine": [c["name"] for c in riskine_classes],
        }

    riskine_names = [c["name"] for c in riskine_classes]
    riskine_by_name = {c["name"]: c for c in riskine_classes}

    print(f"  Induced labels: {len(induced_labels)}, Riskine classes: {len(riskine_names)}")
    print("  Embedding labels with all-MiniLM-L6-v2...")

    induced_embs = embed_labels(induced_labels)   # (N_induced, 384)
    riskine_embs = embed_labels(riskine_names)    # (N_riskine, 384)

    # Cosine similarity matrix — embeddings are L2-normalized, so dot product = cosine sim
    sim_matrix = np.dot(induced_embs, riskine_embs.T)   # (N_induced, N_riskine)

    print(f"  Running LLM judge for candidate pairs (threshold ≥ {CANDIDATE_THRESHOLD})...")

    alignment_table: list[dict] = []
    riskine_covered: set[str] = set()
    scores: list[float] = []

    for i, induced in enumerate(induced_labels):
        candidate_indices = np.where(sim_matrix[i] >= CANDIDATE_THRESHOLD)[0]

        if len(candidate_indices) == 0:
            scores.append(0.0)
            alignment_table.append({
                "induced": induced,
                "riskine": None,
                "cosine": None,
                "verdict": "NO_CANDIDATE",
                "score": 0.0,
            })
            continue

        best_score = 0.0
        best_entry: dict = {
            "induced": induced,
            "riskine": None,
            "cosine": None,
            "verdict": "NO_MATCH",
            "score": 0.0,
        }

        for j in candidate_indices:
            riskine_name = riskine_names[j]
            cosine_val = float(sim_matrix[i][j])
            props = riskine_by_name[riskine_name]["properties"]

            print(
                f"    [{induced[:35]}] ←→ [{riskine_name}]  "
                f"sim={cosine_val:.3f}  ... ",
                end="", flush=True,
            )
            verdict = _llm_judge(llm, induced, riskine_name, props)
            print(verdict)

            score = (
                1.0 if verdict == "MATCH"
                else (PARTIAL_WEIGHT if verdict == "PARTIAL" else 0.0)
            )
            if score > best_score:
                best_score = score
                best_entry = {
                    "induced": induced,
                    "riskine": riskine_name,
                    "cosine": round(cosine_val, 4),
                    "verdict": verdict,
                    "score": score,
                }
                if verdict in ("MATCH", "PARTIAL"):
                    riskine_covered.add(riskine_name)

        scores.append(best_score)
        alignment_table.append(best_entry)

    precision = sum(scores) / len(induced_labels) if induced_labels else 0.0
    recall = len(riskine_covered) / len(riskine_names) if riskine_names else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    unmatched_induced = [e["induced"] for e in alignment_table if e["score"] == 0.0]
    unmatched_riskine = [n for n in riskine_names if n not in riskine_covered]

    result = {
        "suffix": suffix,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "induced_label_count": len(induced_labels),
        "riskine_class_count": len(riskine_names),
        "riskine_covered_count": len(riskine_covered),
        "alignment_table": alignment_table,
        "unmatched_induced": unmatched_induced,
        "unmatched_riskine": unmatched_riskine,
    }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"riskine_eval_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✓ Riskine alignment saved → {out_path}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Riskine ontology alignment evaluator")
    parser.add_argument(
        "--suffix", default="zone1",
        help="Which graph suffix to evaluate: 'zone1', 'original', 'zone1_qwen', ..."
    )
    parser.add_argument(
        "--model", default=config.OLLAMA_MODEL,
        help=f"Ollama model for LLM judge (default: {config.OLLAMA_MODEL})"
    )
    args = parser.parse_args()

    print(f"Riskine Alignment Evaluation — suffix={args.suffix}, model={args.model}")
    print("=" * 60)

    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )
    llm = ChatOllama(model=args.model, base_url=config.OLLAMA_BASE_URL, temperature=0)

    schemas = riskine_loader.fetch_and_cache()
    riskine_classes = riskine_loader.extract_riskine_classes(schemas)

    result = measure_riskine_alignment(graph, llm, riskine_classes, suffix=args.suffix)

    print(f"\n{'=' * 60}")
    print(f"RISKINE ALIGNMENT SUMMARY  [{args.suffix}]")
    print(f"{'=' * 60}")
    print(f"  Induced labels:    {result['induced_label_count']}")
    print(f"  Riskine classes:   {result['riskine_class_count']}")
    print(f"  Riskine covered:   {result['riskine_covered_count']}")
    print(f"  Precision:         {result['precision']:.3f}")
    print(f"  Recall:            {result['recall']:.3f}")
    print(f"  F1:                {result['f1']:.3f}")
