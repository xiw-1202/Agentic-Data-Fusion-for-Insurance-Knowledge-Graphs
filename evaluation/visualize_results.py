"""
evaluation/visualize_results.py
================================
Clean evaluation report — metrics tables + actual QA returns.

Sections:
  1. Metrics Summary      — comparison table across models
  2. Query Accuracy       — 10 representative tasks (5 pass + 5 fail): question, expected, actual return
  3. Riskine Alignment    — induced labels vs 10 Riskine classes, missed classes
  4. Label Explosion      — type-proliferation examples from the graph

Usage:
  python3 evaluation/visualize_results.py [--suffix zone1] [--html]
"""

import html as _html
import json
import os
import sys
import argparse

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_EVAL_DIR)
_BASELINE = os.path.join(_ROOT, "baseline")
if _ROOT     not in sys.path: sys.path.insert(0, _ROOT)
if _EVAL_DIR not in sys.path: sys.path.insert(0, _EVAL_DIR)
if _BASELINE not in sys.path: sys.path.insert(0, _BASELINE)

import config
from eval import EVAL_TASKS  # expected keywords + cypher for each task


# ---------------------------------------------------------------------------
# Tiny HTML helpers — no CDN, no JS
# ---------------------------------------------------------------------------

def _e(s):
    return _html.escape(str(s))

def _cell(s, raw=False):
    return f"<td>{s if raw else _e(s)}</td>"

def _hcell(s):
    return f"<th>{_e(s)}</th>"

def _row(*cells, cls=""):
    inner = "".join(cells)
    return f'<tr class="{cls}">{inner}</tr>' if cls else f"<tr>{inner}</tr>"

def _table(headers, rows, raw_cols=None, row_classes=None):
    """
    raw_cols:    set of column indices where content is already HTML (skip escaping).
    row_classes: optional list of CSS class strings, one per row.
    """
    raw_cols = raw_cols or set()
    head = "<tr>" + "".join(_hcell(h) for h in headers) + "</tr>"
    body = ""
    for i, row in enumerate(rows):
        cls   = row_classes[i] if row_classes and i < len(row_classes) else ""
        cells = "".join(_cell(v, raw=(j in raw_cols)) for j, v in enumerate(row))
        body += f'<tr class="{cls}">{cells}</tr>' if cls else f"<tr>{cells}</tr>"
    return f'<table><thead>{head}</thead><tbody>{body}</tbody></table>'

def _section(title, subtitle, body):
    sub = f'<div class="subtitle">{_e(subtitle)}</div>' if subtitle else ""
    return f'<div class="sec"><h2>{_e(title)}</h2>{sub}{body}</div>'

def _note(text):
    return f'<p class="note">{_e(text)}</p>'


# ---------------------------------------------------------------------------
# HTML page template — self-contained, no CDN
# ---------------------------------------------------------------------------

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 14px;
  background: #f5f6f8;
  color: #222;
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 20px;
}
h1  { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
h2  { font-size: 15px; font-weight: 700; margin-bottom: 6px; color: #1a1a2e; }
.subtitle { font-size: 12px; color: #666; margin-bottom: 10px; }
.meta { font-size: 12px; color: #888; margin-bottom: 20px; }

/* Sections */
.sec {
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,.08);
  padding: 18px 20px;
  margin-bottom: 16px;
}

/* Tables */
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
  margin-top: 6px;
}
th, td {
  border: 1px solid #e2e4e8;
  padding: 5px 9px;
  text-align: left;
  vertical-align: top;
}
th { background: #f0f2f5; font-weight: 600; }
tr:nth-child(even) { background: #fafbfc; }
tr.pass td:first-child { border-left: 3px solid #27ae60; }
tr.fail td:first-child { border-left: 3px solid #e74c3c; }
tr.pass { background: #f6fff8; }
tr.fail { background: #fff6f6; }

/* Inline badges */
.v-match   { color: #1a7a4a; font-weight: 600; }
.v-partial { color: #9a6800; font-weight: 600; }
.v-none    { color: #999; }
.pass-mark { color: #27ae60; font-weight: 700; }
.fail-mark { color: #e74c3c; font-weight: 700; }

/* Actual-returns cell */
.ret-list { list-style: none; padding: 0; margin: 0; }
.ret-list li {
  background: #f4f5f7;
  border-radius: 3px;
  padding: 2px 6px;
  margin-bottom: 2px;
  font-size: 12px;
  word-break: break-word;
}
.ret-none { color: #aaa; font-style: italic; font-size: 12px; }

/* Keywords */
.kw-list { font-size: 12px; color: #555; }
.kw { display: inline-block; background: #eef2ff; border-radius: 3px;
      padding: 1px 5px; margin: 1px 2px; }

/* Note */
.note { font-size: 12px; color: #888; margin-top: 8px; }

/* Missed chips */
.missed-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.missed-chip {
  background: #fde8e8;
  border: 1px solid #f5c6cb;
  border-radius: 4px;
  padding: 3px 10px;
  font-size: 12px;
  font-weight: 600;
  color: #c0392b;
}
"""

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>KG Evaluation — {suffix}</title>
<style>
{css}
</style>
</head>
<body>
<h1>KG Evaluation Report</h1>
<p class="meta">Suffix: <strong>{suffix}</strong> &nbsp;·&nbsp; Baseline evaluation · {suffix}</p>
{sections}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load(filename):
    path = os.path.join(config.RESULTS_DIR, filename)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None

def _m(data):
    return (data or {}).get("baseline_metrics", {})

def _fmt(v, pct=False):
    if v is None:
        return "N/A"
    return f"{v:.1%}" if pct else str(v)


# ---------------------------------------------------------------------------
# Section 1 — Metrics Summary
# ---------------------------------------------------------------------------

def section1_metrics(suffix, html_parts):
    orig  = _load("baseline_eval_results_original.json")
    llama = _load(f"baseline_eval_results_{suffix}.json")
    qwen  = _load(f"baseline_eval_results_{suffix}_qwen.json")

    om, lm, qm = _m(orig), _m(llama), _m(qwen)

    orig_label  = "Baseline (512-tok)"
    llama_label = lm.get("model", "llama3.1:8b") if lm else "llama3.1:8b"
    qwen_label  = qm.get("model", "qwen2.5:7b")  if qm else "qwen2.5:7b"

    rows = [
        ("Nodes",               "total_nodes",             False),
        ("Relationships",       "total_relationships",     False),
        ("Unique labels",       "unique_labels",           False),
        ("Duplicate nodes",     "duplicate_node_count",    False),
        ("Duplication rate",    "duplication_rate",        True),
        ("Type inconsistency",  "type_inconsistency_rate", True),
        ("Query accuracy",      "query_accuracy",          True),
        ("Riskine Precision",   "riskine_precision",       True),
        ("Riskine Recall",      "riskine_recall",          True),
        ("Riskine F1",          "riskine_f1",              True),
    ]

    table_rows = []
    for name, key, pct in rows:
        table_rows.append([name,
                           _fmt(om.get(key), pct),
                           _fmt(lm.get(key) if lm else None, pct),
                           _fmt(qm.get(key) if qm else None, pct)])

    body = _table([" ", orig_label, llama_label, qwen_label], table_rows)
    html_parts.append(_section("1. Metrics Summary", "All three model runs side-by-side", body))


# ---------------------------------------------------------------------------
# Section 2 — Query Accuracy (20 tasks, actual returns shown)
# ---------------------------------------------------------------------------

def _format_sample(sample):
    """Turn a list of Neo4j row dicts into a compact HTML snippet."""
    if not sample:
        return '<span class="ret-none">no rows returned</span>'

    items = []
    for row in sample[:3]:   # show up to 3 rows
        # Extract the most meaningful value from each row
        # Prefer n.id, then m.id, then any string value
        val = (row.get("n.id") or row.get("m.id")
               or next((v for v in row.values() if isinstance(v, str)), None)
               or str(row))
        # Truncate long raw-text nodes
        if isinstance(val, str) and len(val) > 90:
            val = val[:87] + "…"
        items.append(f'<li>{_e(val)}</li>')

    extra = len(sample) - 3
    if extra > 0:
        items.append(f'<li class="ret-none">… {extra} more row(s)</li>')

    return '<ul class="ret-list">' + "".join(items) + "</ul>"


def _format_keywords(kws):
    chips = "".join(f'<span class="kw">{_e(k)}</span>' for k in kws)
    return f'<span class="kw-list">{chips}</span>'


def section2_qa(suffix, html_parts):
    data = _load(f"baseline_eval_results_{suffix}.json")
    if not data:
        html_parts.append(_section("2. Query Accuracy",
                                   f"baseline_eval_results_{suffix}.json not found",
                                   _note("Run: python3 baseline/eval.py --suffix " + suffix)))
        return

    task_results = {t["id"]: t for t in data.get("task_results", [])}
    model  = _m(data).get("model", "unknown")
    acc    = _m(data).get("query_accuracy", 0)
    passed = sum(1 for t in task_results.values() if t.get("keyword_match"))

    task_map = {t["id"]: t for t in EVAL_TASKS}

    # Derive from actual results so headers always match row colours for any suffix
    passed_ids = sorted(tid for tid, t in task_results.items() if     t.get("keyword_match"))[:5]
    failed_ids = sorted(tid for tid, t in task_results.items() if not t.get("keyword_match"))[:5]

    def build_rows(ids):
        rows, classes = [], []
        for tid in ids:
            task = task_map.get(tid)   # safe: skip if ID not in EVAL_TASKS
            if task is None:
                continue
            res  = task_results.get(tid, {})
            ok   = res.get("keyword_match", False)
            mark = '<span class="pass-mark">✓</span>' if ok else '<span class="fail-mark">✗</span>'
            rows.append([
                str(tid),
                task["category"],
                task["question"],
                _format_keywords(task["keywords"]),
                _format_sample(res.get("sample", [])),
                mark,
            ])
            classes.append("pass" if ok else "fail")
        return rows, classes

    pass_rows, pass_cls = build_rows(passed_ids)
    fail_rows, fail_cls = build_rows(failed_ids)

    headers = ["#", "Cat", "Question", "Expected", "Graph returned", "✓/✗"]
    body = (
        '<h3 style="margin:10px 0 4px;color:#27ae60">Passing examples</h3>'
        + _table(headers, pass_rows, raw_cols={3, 4, 5}, row_classes=pass_cls)
        + '<h3 style="margin:16px 0 4px;color:#e74c3c">Failing examples</h3>'
        + _table(headers, fail_rows, raw_cols={3, 4, 5}, row_classes=fail_cls)
    )
    subtitle = (f"Model: {model}  ·  {passed}/20 tasks passed  ·  "
                f"Query accuracy: {acc:.0%}  ·  showing 5 pass + 5 fail")
    html_parts.append(_section("2. Query Accuracy — Representative Examples", subtitle, body))


# ---------------------------------------------------------------------------
# Section 3 — Riskine Alignment
# ---------------------------------------------------------------------------

def section3_riskine(suffix, html_parts):
    data = _load(f"riskine_eval_{suffix}.json")
    if not data:
        html_parts.append(_section("3. Riskine Ontology Alignment",
                                   "riskine_eval file not found",
                                   _note(f"Run: python3 baseline/eval.py --suffix {suffix} --riskine")))
        return

    try:
        p       = data.get("precision", 0.0)
        r       = data.get("recall",    0.0)
        f1      = data.get("f1",        0.0)
        covered = data.get("riskine_covered_count", "?")
        induced = data.get("induced_label_count",   "?")
        table   = data.get("alignment_table", [])
        missed  = data.get("unmatched_riskine", [])

        # Sort: MATCH → PARTIAL → NO_CANDIDATE
        order = {"MATCH": 0, "PARTIAL": 1, "NO_CANDIDATE": 2, "NO_MATCH": 3}
        table = sorted(table, key=lambda e: order.get(e["verdict"], 9))

        headers = ["Induced label (LLM)", "Riskine class", "Verdict", "Cosine"]
        rows = []
        for e in table:
            v = e["verdict"]
            cls_map = {"MATCH": "v-match", "PARTIAL": "v-partial"}
            v_html  = f'<span class="{cls_map.get(v, "v-none")}">{_e(v)}</span>'
            cos     = f'{e["cosine"]:.3f}' if e.get("cosine") is not None else "—"
            rows.append([e["induced"], e["riskine"] or "—", v_html, cos])

        body = _table(headers, rows, raw_cols={2})

        # Missed chips
        chips = "".join(f'<div class="missed-chip">{_e(c)}</div>' for c in missed)
        missed_html = (
            f'<div style="margin-top:14px"><strong>Riskine classes never extracted '
            f'({len(missed)}/10):</strong>'
            f'<div class="missed-row">{chips}</div></div>'
            if missed else ""
        )

        subtitle = (f"P={p:.2f}  ·  R={r:.2f}  ·  F1={f1:.2f}  ·  "
                    f"{covered}/10 Riskine classes covered  ·  "
                    f"{induced} induced labels evaluated")
        html_parts.append(_section("3. Riskine Ontology Alignment",
                                   subtitle,
                                   body + missed_html))
    except Exception as exc:
        html_parts.append(_section("3. Riskine Ontology Alignment",
                                   "Error rendering section",
                                   _note(f"Details: {exc}")))


# ---------------------------------------------------------------------------
# Section 4 — Label Explosion (type proliferation)
# ---------------------------------------------------------------------------

def section4_labels(suffix, html_parts):
    data = _load(f"baseline_eval_results_{suffix}.json")
    if not data:
        return

    tc  = data.get("type_consistency_detail", {})
    ex  = tc.get("examples", {})
    if not ex:
        return

    headers = ["Root concept", "Labels the LLM created", "Count"]
    rows = []
    for root, variants in ex.items():
        rows.append([root, "  ·  ".join(variants), str(len(variants))])

    rate = _m(data).get("type_inconsistency_rate", 0)
    subtitle = (f"{tc.get('concepts_with_multiple_labels', 0)} concepts exploded into multiple labels  ·  "
                f"Type inconsistency rate: {rate:.1%}  ·  "
                f"Total unique labels: {tc.get('total_labels', '?')}")

    body = _table(headers, rows)
    html_parts.append(_section("4. Label Explosion (Type Inconsistency)", subtitle, body))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="zone1")
    parser.add_argument("--html",   action="store_true")
    args = parser.parse_args()

    html_parts = []

    section1_metrics(args.suffix, html_parts)
    section2_qa(args.suffix, html_parts)
    section3_riskine(args.suffix, html_parts)
    section4_labels(args.suffix, html_parts)

    # Plain-text summary
    data = _load(f"baseline_eval_results_{args.suffix}.json")
    if data:
        m = _m(data)
        nodes  = m.get("total_nodes", "?")
        rels   = m.get("total_relationships", "?")
        labels = m.get("unique_labels", "?")
        print(f"\nKG Evaluation — {args.suffix}")
        print(f"  Nodes: {nodes}  Rels: {rels}  Labels: {labels}")
        acc = m.get("query_accuracy")
        kw  = m.get("tasks_keyword_match", "?")
        if acc is not None:
            print(f"  Query accuracy: {acc:.0%}  ({kw}/20 tasks)")
        ti = m.get("type_inconsistency_rate")
        if ti is not None:
            print(f"  Type inconsistency: {ti:.1%}")
        if m.get("riskine_available"):
            f1 = m.get("riskine_f1", 0)
            rp = m.get("riskine_precision", 0)
            rr = m.get("riskine_recall", 0)
            print(f"  Riskine F1: {f1:.3f}  (P={rp:.3f}, R={rr:.3f})")

    if args.html:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        out = os.path.join(config.RESULTS_DIR, f"visualization_report_{args.suffix}.html")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(_HTML_TEMPLATE.format(
                suffix=_e(args.suffix),
                css=_CSS,
                sections="\n".join(html_parts),
            ))
        print(f"  HTML → {out}")


if __name__ == "__main__":
    main()
