"""Compare Evaluation Results — Ablation Study
============================================
Prints a side-by-side comparison of:
  - Baseline (original 512-token chunks) + LLMGraphTransformer
  - Baseline (Zone 1 section-aware chunks) + LLMGraphTransformer  ← ablation

This isolates the contribution of Zone 1 chunking alone.
Later: add the full novel pipeline results for the final Table 1.

Usage:
  python3 evaluation/compare_results.py
  python3 evaluation/compare_results.py --add zone2   # after Zone 2 is built
"""

from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Allow imports from project root (config.py lives there)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

RESULTS_DIR = config.RESULTS_DIR

RUNS = {
    "original":   ("Baseline (512-token)",         "baseline_eval_results_original.json"),
    "zone1":      ("Baseline + Zone1 (Llama)",     "baseline_eval_results_zone1.json"),
    "zone1_qwen": ("Baseline + Zone1 (Qwen2.5)",  "baseline_eval_results_zone1_qwen.json"),
    "novel":      ("Novel Pipeline (full)",        "novel_eval_results.json"),   # future
}


def load_result(filename: str) -> dict | None:
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(value, is_pct: bool = False, is_good_low: bool = False) -> str:
    """Format a metric value with directional arrow vs baseline."""
    if value is None:
        return "N/A"
    if is_pct:
        return f"{value:.1%}"
    return str(value)


def compare():
    print("=" * 72)
    print("CS584 Capstone — Evaluation Comparison")
    print("=" * 72)

    results = {}
    for key, (label, fname) in RUNS.items():
        r = load_result(fname)
        if r:
            results[key] = (label, r)

    if not results:
        print("No result files found. Run the pipeline first.")
        return

    available = list(results.keys())
    print(f"Loaded runs: {[results[k][0] for k in available]}\n")

    # --- Header ---
    col_w = 26
    header = f"{'Metric':<28}" + "".join(f"{results[k][0]:>{col_w}}" for k in available)
    print(header)
    print("-" * len(header))

    def row(name, getter, is_pct=False):
        vals = []
        base_val = None
        for i, k in enumerate(available):
            m = results[k][1].get("baseline_metrics", {})
            v = getter(m)
            if i == 0:
                base_val = v
            if v is None:
                vals.append("N/A")
            elif is_pct:
                cell = f"{v:.1%}"
                if i > 0 and base_val is not None:
                    delta = v - base_val
                    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                    cell += f" ({arrow}{abs(delta):.1%})"
                vals.append(cell)
            else:
                cell = str(v)
                if i > 0 and base_val is not None and isinstance(v, (int, float)):
                    delta = v - base_val
                    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                    cell += f" ({arrow}{abs(int(delta))})"
                vals.append(cell)
        print(f"{name:<28}" + "".join(f"{v:>{col_w}}" for v in vals))

    row("Nodes",              lambda m: m.get("total_nodes"))
    row("Relationships",      lambda m: m.get("total_relationships"))
    row("Unique labels",      lambda m: m.get("unique_labels"))
    row("Duplication rate",   lambda m: m.get("duplication_rate"),   is_pct=True)
    row("Type inconsistency", lambda m: m.get("type_inconsistency_rate"), is_pct=True)
    row("Query accuracy",     lambda m: m.get("query_accuracy"),     is_pct=True)
    row("Riskine Precision",  lambda m: m.get("riskine_precision") if m.get("riskine_available") else None, is_pct=True)
    row("Riskine Recall",     lambda m: m.get("riskine_recall")    if m.get("riskine_available") else None, is_pct=True)
    row("Riskine F1",         lambda m: m.get("riskine_f1")        if m.get("riskine_available") else None, is_pct=True)

    print()

    # --- Per-task breakdown ---
    print("=" * 72)
    print("Per-Task Query Accuracy Breakdown")
    print("=" * 72)
    task_col = f"{'ID':<4}{'Category':<14}{'Question':<40}"
    for k in available:
        task_col += f"  {results[k][0][:12]:>12}"
    print(task_col)
    print("-" * len(task_col))

    # Align tasks by id across runs
    all_tasks = {t["id"]: t for t in results[available[0]][1].get("task_results", [])}
    for tid in sorted(all_tasks.keys()):
        t0 = all_tasks[tid]
        line = f"{tid:<4}{t0['category']:<14}{t0['question'][:38]:<40}"
        for k in available:
            tasks = {t["id"]: t for t in results[k][1].get("task_results", [])}
            t = tasks.get(tid, {})
            sym = "✓" if t.get("keyword_match") else "✗"
            rows = t.get("rows_returned", 0)
            line += f"  {sym} ({rows:>3} rows)"
        print(line)

    # --- Summary row ---
    print("-" * len(task_col))
    summary_line = f"{'TOTAL':<4}{'':<14}{'Keyword-matched / 20':<40}"
    for k in available:
        tasks = results[k][1].get("task_results", [])
        matched = sum(1 for t in tasks if t.get("keyword_match"))
        pct = matched / len(tasks) if tasks else 0
        summary_line += f"  {matched}/20 ({pct:.0%})      "
    print(summary_line)

    # --- Key insight ---
    if "original" in results and "zone1" in results:
        orig_acc = results["original"][1]["baseline_metrics"]["query_accuracy"]
        z1_acc = results["zone1"][1]["baseline_metrics"]["query_accuracy"]
        delta = z1_acc - orig_acc
        print(f"\n{'=' * 72}")
        print("KEY FINDING: Contribution of Zone 1 chunking alone")
        print(f"{'=' * 72}")
        print(f"  Query accuracy:  {orig_acc:.1%} → {z1_acc:.1%}  (Δ = {delta:+.1%})")

        orig_labels = results["original"][1]["baseline_metrics"]["unique_labels"]
        z1_labels   = results["zone1"][1]["baseline_metrics"]["unique_labels"]
        print(f"  Unique labels:   {orig_labels} → {z1_labels}  "
              f"({'fewer' if z1_labels < orig_labels else 'more'} label proliferation)")

        orig_dup = results["original"][1]["baseline_metrics"]["duplication_rate"]
        z1_dup   = results["zone1"][1]["baseline_metrics"]["duplication_rate"]
        print(f"  Duplication:     {orig_dup:.1%} → {z1_dup:.1%}")

        if delta > 0:
            print(f"\n  ✓ Zone 1 chunking improves query accuracy by {delta:+.1%}")
            print(f"    Section-aware splits give the LLM more coherent context per chunk.")
        elif delta == 0:
            print(f"\n  → Same accuracy — extraction quality is the bottleneck, not chunking.")
            print(f"    Zone 2 Open IE with quality filters is needed for further gains.")
        else:
            print(f"\n  ✗ Zone 1 chunking hurt accuracy by {delta:+.1%}")
            print(f"    Larger sections may overwhelm the LLMGraphTransformer — check chunk sizes.")


def compare_zone3():
    """Compare Zone 3 ontology induction methods — comprehensive table
    including AUC, F1, BERTScore, and ablation results."""
    print("=" * 80)
    print("CS584 Capstone — Zone 3 Ontology Induction Comparison")
    print("=" * 80)

    # Discover all riskine_eval_zone3*.json files
    result_dir = Path(RESULTS_DIR)
    z3_files = sorted(result_dir.glob("riskine_eval_zone3*.json"))
    if not z3_files:
        print("No Zone 3 evaluation results found. Run eval.py --riskine first.")
        return

    results: dict[str, dict] = {}
    for f in z3_files:
        with open(f) as fh:
            data = json.load(fh)
        suffix = data.get("suffix", f.stem.replace("riskine_eval_", ""))
        results[suffix] = data

    if not results:
        print("No results loaded.")
        return

    suffixes = list(results.keys())
    col_w = 18

    # Header
    header = f"{'Metric':<30}" + "".join(f"{s:>{col_w}}" for s in suffixes)
    print(header)
    print("-" * len(header))

    def _get(data: dict, *keys, default=0.0):
        """Safely navigate nested keys."""
        for k in keys:
            if isinstance(data, dict):
                data = data.get(k, default)
            else:
                return default
        return data

    def row(name: str, getter, fmt_str: str = "{:.3f}"):
        vals = []
        for s in suffixes:
            v = getter(results[s])
            if v is None:
                vals.append("N/A")
            else:
                vals.append(fmt_str.format(v))
        print(f"{name:<30}" + "".join(f"{v:>{col_w}}" for v in vals))

    # --- Name-based metrics ---
    print("\n  Name-based alignment (legacy):")
    row("  Name Precision", lambda d: d.get("precision"))
    row("  Name Recall", lambda d: d.get("recall"))
    row("  Name F1", lambda d: d.get("f1"))

    # --- Entity Assignment ---
    print("\n  Entity Assignment:")
    row("  EA Precision", lambda d: d.get("entity_assignment_precision"))
    row("  EA Recall", lambda d: d.get("entity_assignment_recall"))
    row("  EA F1 (full)", lambda d: d.get("entity_assignment_f1"))
    row("  EA F1 (present)", lambda d: d.get("entity_assignment_f1_present"))
    row("  Evidenced classes", lambda d: d.get("entity_assignment_evidenced_count"), fmt_str="{:.0f}")

    # --- Standard metrics ---
    print("\n  Standard ontology metrics:")
    row("  BERTScore F1", lambda d: _get(d, "standard_metrics", "bertscore_f1"))
    row("  Graph F1", lambda d: _get(d, "standard_metrics", "graph_f1"))
    row("  Continuous F1", lambda d: _get(d, "standard_metrics", "continuous_f1"))
    row("  Fuzzy F1", lambda d: _get(d, "standard_metrics", "fuzzy_f1"))
    row("  Wu-Palmer", lambda d: _get(d, "standard_metrics", "avg_wu_palmer"))

    # --- AUC metrics ---
    print("\n  AUC-ROC (threshold-independent):")
    row("  AUC macro", lambda d: _get(d, "standard_metrics", "auc_macro"))
    row("  AUC weighted", lambda d: _get(d, "standard_metrics", "auc_weighted"))
    row("  mAP", lambda d: _get(d, "standard_metrics", "map_score"))
    row("  AUC classes eval'd", lambda d: _get(d, "standard_metrics", "auc_classes_evaluated"), fmt_str="{:.0f}")

    # --- Summary ---
    print("\n  Summary:")
    row("  Induced labels", lambda d: d.get("induced_label_count"), fmt_str="{:.0f}")
    row("  Riskine classes", lambda d: d.get("riskine_class_count"), fmt_str="{:.0f}")
    row("  Riskine covered", lambda d: d.get("riskine_covered_count"), fmt_str="{:.0f}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone3", action="store_true",
                        help="Compare Zone 3 ontology induction results")
    args = parser.parse_args()

    if args.zone3:
        compare_zone3()
    else:
        compare()
