"""evaluation/riskine_eval.py
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

from __future__ import annotations

import json
import os
import sys
import argparse
import re
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


def _humanize_label(label: str) -> str:
    """Convert PascalCase/CamelCase label to space-separated words for embedding.

    Examples:
        BuildingCoverage     → Building Coverage
        FloodEventCoverage   → Flood Event Coverage
        NFIPSuspension       → NFIP Suspension
        Class90days          → Class 90days
    """
    # Handle sequences like ACRONYMWord → ACRONYM Word
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', label)
    # Handle camelCase boundary: lowercase→uppercase
    s = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', s)
    return s

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
import ontology_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANDIDATE_THRESHOLD = 0.60   # cosine similarity to pass candidate pair to LLM judge
PARTIAL_WEIGHT = 0.5         # PARTIAL match contributes 0.5 to precision score

# Labels to skip — LangChain / Zone 3 infrastructure labels, not ontology concepts
EXCLUDED_LABELS = {"__Entity__", "Document", "OntologyClass", "Entity"}


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_model_cache: Optional[SentenceTransformer] = None


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
        # Parse order matters: NO_MATCH first (contains "MATCH"), then PARTIAL
        # (a response like "PARTIAL MATCH" must → PARTIAL, not MATCH)
        if "NO_MATCH" in text:
            return "NO_MATCH"
        elif "PARTIAL" in text:
            return "PARTIAL"
        elif "MATCH" in text:
            return "MATCH"
        else:
            return "NO_MATCH"
    except Exception as e:
        print(f"    [riskine] LLM judge error: {e}")
        return "NO_MATCH"


# ---------------------------------------------------------------------------
# Main alignment measurement
# ---------------------------------------------------------------------------

# Prefixes for structured record entities (opaque IDs, not semantic names).
# These are filtered from member lists for embedding-based evaluation because
# "POL-3d2371dd" has no semantic content — it would add noise to class centroids.
_RECORD_PREFIXES = ("POL-", "CLM-", "REC-", "PER-", "PROP-")


def _get_label_members(
    graph: Neo4jGraph, label: str, *, exclude_records: bool = True,
) -> list[str]:
    """Return entity IDs that carry a given Neo4j label.

    Args:
        graph: Neo4j connection.
        label: Neo4j node label (PascalCase class name).
        exclude_records: If True, filter out structured record IDs (POL-xxx, etc.)
            that have no semantic content for embedding-based evaluation.

    Returns:
        Up to 30 semantic entity names for the class.
    """
    safe = re.sub(r"[^A-Za-z0-9_]", "", label)
    if not safe:
        return []
    try:
        rows = graph.query(
            f"MATCH (n:{safe}) WHERE n.id IS NOT NULL RETURN n.id AS id ORDER BY n.id ASC LIMIT 100"
        )
        members = [r["id"] for r in rows if r.get("id")]
        if exclude_records:
            members = [m for m in members if not m.startswith(_RECORD_PREFIXES)]
        return members[:30]
    except Exception:
        return []


def measure_entity_assignment(
    graph: Neo4jGraph,
    riskine_classes: list[dict],
    induced_labels: list[str],
) -> dict:
    """
    Entity Assignment F1 — evaluates whether entities are placed in the correct
    ontological class by comparing member centroids to Riskine class descriptions.

    Unlike BERTScore/Graph F1 (which compare class *names* and *structure*),
    this metric evaluates the actual *entity-to-class assignments* — the primary
    output of ontology induction.

    Method:
        For each induced class, embed its member entity IDs and average them
        into a cluster centroid. Compare centroids to Riskine class descriptions
        (class name + property names) via cosine similarity.

    Returns:
        entity_assignment_precision, entity_assignment_recall, entity_assignment_f1,
        entity_assignment_riskine_covered, entity_assignment_table
    """
    MEMBER_CANDIDATE_THRESHOLD = 0.42   # centroid cosines are denser — lower threshold
    MEMBER_MATCH_THRESHOLD     = 0.58   # above this → MATCH (score 1.0), else PARTIAL (0.5)

    print("\n  [entity-assignment] Building member-centroid embeddings (v2: enriched + dual)...")
    centroids: list[np.ndarray] = []
    descriptions: list[str] = []

    for label in induced_labels:
        members = _get_label_members(graph, label)
        human_label = _humanize_label(label)
        if members:
            # Enriched representation: "ClassName: member1, member2, ..."
            # Anchors embedding to class semantics while incorporating member evidence.
            # Raw member-centroid averaging loses the class context and produces
            # embeddings in a different semantic space from Riskine's property-based
            # descriptions (see F-19: EA eval sensitivity).
            enriched_text = f"{human_label}: {', '.join(members[:15])}"
            centroid = embed_labels([enriched_text])[0]           # (384,)
        else:
            centroid = embed_labels([human_label])[0]             # fallback: class name
        centroids.append(centroid)
        descriptions.append(
            f"{label}: [{', '.join(members[:5])}{'...' if len(members) > 5 else ''}]"
            if members else label
        )

    induced_matrix = np.vstack(centroids)                         # (N, 384)

    # Riskine dual representation: embed BOTH property-based and name-based
    # descriptions, then take the element-wise max similarity.
    # Rationale (see F-19): property descriptions ("Organization: business-name,
    # founding-date, company-registry-number") live in a different semantic space
    # from member entity names ("FEMA, Insurer"). Some classes match better on
    # name similarity (Organization→Organization), others on property similarity
    # (Coverage→is-included, sum-insured). Taking the max is fair to both.
    riskine_names = [c["name"] for c in riskine_classes]
    riskine_prop_texts = [
        f"{c['name']}: {', '.join(c['properties'][:8])}"
        for c in riskine_classes
    ]
    riskine_name_texts = [_humanize_label(c["name"]) for c in riskine_classes]

    riskine_prop_embs = embed_labels(riskine_prop_texts)          # (K, 384)
    riskine_name_embs = embed_labels(riskine_name_texts)          # (K, 384)

    sim_props = np.dot(induced_matrix, riskine_prop_embs.T)       # (N, K)
    sim_names = np.dot(induced_matrix, riskine_name_embs.T)       # (N, K)
    sim_matrix = np.maximum(sim_props, sim_names)                 # element-wise max

    scores: list[float] = []
    riskine_covered: set[str] = set()
    member_table: list[dict] = []

    for i, induced in enumerate(induced_labels):
        best_j    = int(np.argmax(sim_matrix[i]))
        best_sim  = float(sim_matrix[i][best_j])
        rname     = riskine_names[best_j]

        if best_sim >= MEMBER_MATCH_THRESHOLD:
            score = 1.0
            riskine_covered.add(rname)
        elif best_sim >= MEMBER_CANDIDATE_THRESHOLD:
            score = 0.5
            riskine_covered.add(rname)
        else:
            score   = 0.0
            rname   = None       # type: ignore[assignment]
            best_sim = float(sim_matrix[i][best_j])

        scores.append(score)
        member_table.append({
            "induced":        induced,
            "members_sample": descriptions[i],
            "riskine":        rname,
            "cosine":         round(best_sim, 4),
            "score":          score,
        })

    precision = sum(scores) / len(induced_labels)    if induced_labels  else 0.0
    recall    = len(riskine_covered) / len(riskine_names) if riskine_names else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- Present-class metrics ---
    # Determine which Riskine classes have ANY matching entities in the graph.
    # Compute a separate F1 only against those "evidenced" classes.
    # This gives an honest metric when the source data doesn't cover all classes.
    evidenced_riskine = set()
    for entry in member_table:
        if entry["riskine"] is not None and entry["score"] > 0:
            evidenced_riskine.add(entry["riskine"])
    # Also check: does ANY induced class centroid exceed candidate threshold
    # against a Riskine class? If so, that class is "present in data"
    for j, rname in enumerate(riskine_names):
        col_sims = sim_matrix[:, j]
        if col_sims.max() >= MEMBER_CANDIDATE_THRESHOLD:
            evidenced_riskine.add(rname)

    n_evidenced = len(evidenced_riskine)
    recall_present = len(riskine_covered & evidenced_riskine) / n_evidenced if n_evidenced > 0 else 0.0
    f1_present = (
        2 * precision * recall_present / (precision + recall_present)
        if (precision + recall_present) > 0 else 0.0
    )

    print(f"  [entity-assignment] P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  (full {len(riskine_names)}-class)")
    print(f"  [entity-assignment] P={precision:.3f}  R={recall_present:.3f}  F1={f1_present:.3f}  (present {n_evidenced}-class)")
    print(f"  [entity-assignment] Riskine classes covered: {sorted(riskine_covered)}")
    print(f"  [entity-assignment] Evidenced classes: {sorted(evidenced_riskine)}")

    return {
        "entity_assignment_precision":       round(precision, 4),
        "entity_assignment_recall":          round(recall,    4),
        "entity_assignment_f1":              round(f1,        4),
        "entity_assignment_riskine_covered": sorted(riskine_covered),
        "entity_assignment_table":           member_table,
        # Present-class metrics (only classes evidenced in source data)
        "entity_assignment_evidenced_classes": sorted(evidenced_riskine),
        "entity_assignment_evidenced_count":  n_evidenced,
        "entity_assignment_recall_present":   round(recall_present, 4),
        "entity_assignment_f1_present":       round(f1_present, 4),
    }


def measure_riskine_alignment(
    graph: Neo4jGraph,
    llm: ChatOllama,
    riskine_classes: list[dict],
    suffix: str = "zone1",
    use_all_classes: bool = False,
    results_dir: str | None = None,
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
    print("  Embedding labels with all-MiniLM-L6-v2 (PascalCase → human-readable)...")

    # Humanize + add domain context for embedding.
    # Bare words like "Policy" vs "Product" have low cosine (0.37) but with
    # domain context similarity jumps significantly — essential for cross-name
    # matching in ontology alignment.
    # Domain is read from zone3 summary (LLM-detected, not hardcoded).
    # Search both results_dir and its parent (zone3 summary may be one level up).
    rdir = results_dir or config.RESULTS_DIR
    detected_domain = ""
    search_dirs = [rdir, os.path.dirname(rdir.rstrip("/"))]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for fn in os.listdir(search_dir):
            if fn.startswith("zone3_svloi_summary") and fn.endswith(".json"):
                try:
                    with open(os.path.join(search_dir, fn)) as fh:
                        z3 = json.load(fh)
                    detected_domain = z3.get("domain", "")
                except Exception:
                    pass
                if detected_domain:
                    break
        if detected_domain:
            break
    domain_ctx = f"({detected_domain} ontology class)" if detected_domain else "(ontology class)"
    print(f"  Embedding context: {domain_ctx}")
    induced_for_embed = [f"{_humanize_label(l)} {domain_ctx}" for l in induced_labels]
    riskine_for_embed = [f"{_humanize_label(n)} {domain_ctx}" for n in riskine_names]

    induced_embs = embed_labels(induced_for_embed)   # (N_induced, 384)
    riskine_embs = embed_labels(riskine_for_embed)   # (N_riskine, 384)

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

    # Entity Assignment F1 — evaluates entity-to-class placement quality.
    # Embeds cluster members (not class names) to remove naming-convention bias (F-07).
    try:
        ea_results = measure_entity_assignment(graph, riskine_classes, induced_labels)
        result.update(ea_results)
    except Exception as exc:
        print(f"  [entity-assignment] WARNING: entity assignment eval failed: {exc}")

    # Standard ontology metrics (OLLM NeurIPS'24, AutoSchemaKG'25)
    # Fuzzy F1, Continuous F1, Graph F1, BERTScore, Taxonomy Edge F1, Wu-Palmer, AUC
    try:
        schemas = riskine_loader.fetch_and_cache(use_all=use_all_classes)
        std_metrics = ontology_metrics.evaluate_ontology(
            graph, schemas, riskine_classes,
            alignment_table=alignment_table,
        )
        result["standard_metrics"] = std_metrics
    except Exception as exc:
        print(f"  [ontology-metrics] WARNING: standard metrics failed: {exc}")

    rdir = results_dir or config.RESULTS_DIR
    os.makedirs(rdir, exist_ok=True)
    out_path = os.path.join(rdir, f"riskine_eval_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✓ Riskine alignment saved → {out_path}")
    if "entity_assignment_f1" in result:
        ep = result["entity_assignment_precision"]
        er = result["entity_assignment_recall"]
        ef = result["entity_assignment_f1"]
        print(f"  Entity Assignment P={ep:.3f}  R={er:.3f}  F1={ef:.3f}")
    print(f"  (Legacy name F1={f1:.3f} — deprecated, see F-07)")

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
    parser.add_argument(
        "--all-classes", action="store_true",
        help="Use ALL 26 Riskine classes (not just 10 flood-relevant) for evaluation"
    )
    args = parser.parse_args()

    use_all = getattr(args, 'all_classes', False)
    n_classes = "26 (full)" if use_all else "10 (flood)"
    print(f"Riskine Alignment Evaluation — suffix={args.suffix}, model={args.model}, classes={n_classes}")
    print("=" * 60)

    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )
    llm = ChatOllama(model=args.model, base_url=config.OLLAMA_BASE_URL, temperature=0)

    schemas = riskine_loader.fetch_and_cache(use_all=use_all)
    riskine_classes = riskine_loader.extract_riskine_classes(schemas)

    result = measure_riskine_alignment(
        graph, llm, riskine_classes, suffix=args.suffix, use_all_classes=use_all,
    )

    print(f"\n{'=' * 60}")
    print(f"RISKINE ALIGNMENT SUMMARY  [{args.suffix}]")
    print(f"{'=' * 60}")
    print(f"  Induced labels:    {result['induced_label_count']}")
    print(f"  Riskine classes:   {result['riskine_class_count']}")
    print(f"  Riskine covered:   {result['riskine_covered_count']}")
    if "entity_assignment_f1" in result:
        print(f"  Entity Assignment (full {result['riskine_class_count']}-class Riskine):")
        print(f"    Precision:       {result['entity_assignment_precision']:.3f}")
        print(f"    Recall:          {result['entity_assignment_recall']:.3f}")
        print(f"    F1:              {result['entity_assignment_f1']:.3f}")
        print(f"    Riskine covered: {result['entity_assignment_riskine_covered']}")
        if "entity_assignment_f1_present" in result:
            n_ev = result.get('entity_assignment_evidenced_count', '?')
            print(f"  Entity Assignment (present {n_ev}-class — classes evidenced in data):")
            print(f"    Recall:          {result['entity_assignment_recall_present']:.3f}")
            print(f"    F1:              {result['entity_assignment_f1_present']:.3f}")
            print(f"    Evidenced:       {result['entity_assignment_evidenced_classes']}")
    if "standard_metrics" in result:
        sm = result["standard_metrics"]
        print(f"  Standard ontology metrics (OLLM NeurIPS'24 / AutoSchemaKG'25):")
        print(f"    BERTScore F1:    {sm.get('bertscore_f1', 0):.3f}")
        print(f"    Graph F1:        {sm.get('graph_f1', 0):.3f}")
        print(f"    Continuous F1:   {sm.get('continuous_f1', 0):.3f}")
        print(f"    Fuzzy F1:        {sm.get('fuzzy_f1', 0):.3f}")
        print(f"    Wu-Palmer:       {sm.get('avg_wu_palmer', 0):.3f}")
        # AUC metrics (independent ground truth)
        if "auc_roc" in sm:
            print(f"  AUC-ROC class alignment (independent ground truth):")
            print(f"    AUC-ROC:         {sm.get('auc_roc', 0):.3f}")
            print(f"    MRR:             {sm.get('mrr', 0):.3f}")
            print(f"    Recall@1:        {sm.get('recall_at_1', 0):.3f}")
            print(f"    Recall@3:        {sm.get('recall_at_3', 0):.3f}")
            print(f"    GT matches:      {sm.get('gt_matches', 0)} (threshold={sm.get('match_threshold', '?')})")
            print(f"    Classes matched: {sm.get('auc_classes_matched', 0)}/{sm.get('auc_classes_total', 0)}")
        # Per-class confusion (top matches)
        if "confusion_matrix" in sm:
            print(f"  Per-class confusion (reference → best induced match):")
            for entry in sm["confusion_matrix"]:
                ref = entry["reference"]
                best = entry.get("best_match", "—")
                sim_val = entry.get("best_similarity", 0)
                marker = "✓" if sim_val >= 0.5 else "✗"
                print(f"    {marker} {ref:<20} → {best:<20} (sim={sim_val:.3f})")
        # Intrinsic quality metrics (no reference needed)
        if "intrinsic" in sm:
            iq = sm["intrinsic"]
            print(f"  Intrinsic Ontology Quality (reference-free):")
            print(f"    Class coverage:  {iq.get('class_coverage', 0):.1%} "
                  f"({iq.get('entities_labeled', 0)}/{iq.get('entities_total', 0)} entities)")
            print(f"    Class coherence: {iq.get('class_coherence', 0):.3f}")
            print(f"    Class balance:   {iq.get('class_balance_entropy', 0):.3f}")
            print(f"    Hierarchy:       {iq.get('n_hierarchy_edges', 0)} edges, "
                  f"depth {iq.get('hierarchy_max_depth', 0)}")
            print(f"    Classes:         {iq.get('n_classes', 0)}, "
                  f"avg size {iq.get('avg_class_size', 0):.1f}")
    # Legacy name-based F1 still computed for backward compat but not highlighted
    print(f"  (Legacy name-based F1: {result['f1']:.3f} — deprecated, see F-07)")
