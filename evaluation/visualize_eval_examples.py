#!/usr/bin/env python3
"""
Presentation visualization script for CS584 AI Capstone.

Generates 3 PNG figures to data/results/:
  1. presentation_eval_qa.png     — 4-panel eval Q&A (2 correct, 2 wrong)
  2. presentation_prompts.png     — 3 prompt examples (bootstrap, extraction, cluster naming)
  3. presentation_kg_subgraph.png — KG subgraph with ontology class coloring

Run from project root:
  /opt/anaconda3/bin/python3 evaluation/visualize_eval_examples.py
"""

import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "data", "results")
ZONE2_SUMMARY = os.path.join(RESULTS_DIR, "zone2_run_summary.json")
ZONE3_SUMMARY = os.path.join(RESULTS_DIR, "zone3_run_summary.json")

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BG_DARK     = "#1E2130"
BG_PANEL    = "#252A3E"
BG_CARD     = "#2D3250"
GREEN       = "#27AE60"
GREEN_GLOW  = "#2ECC71"
RED         = "#E74C3C"
ORANGE      = "#E67E22"
BLUE        = "#4A90D9"
PURPLE      = "#9B59B6"
GOLD        = "#F1C40F"
GRAY        = "#95A5A6"
WHITE       = "#ECEFF4"
LIGHT_GRAY  = "#BDC3C7"
CYAN        = "#1ABC9C"

# Class → color mapping for KG visualization
CLASS_COLORS = {
    "InsuredAsset":              "#4A90D9",   # blue
    "InsuranceTerm":             "#27AE60",   # green
    "Timeframe":                 "#E67E22",   # orange
    "PolicyParticipantOrEvent":  "#9B59B6",   # purple
    "InsuredProperty":           "#E74C3C",   # red
    "InsurancePolicyProvisions": "#1ABC9C",   # cyan
    "CoverageExpansionProvision":"#F39C12",   # amber
    "ReplacementValue":          "#8E44AD",   # deep purple
    # coarse/fine level classes
    "InsuranceConcept":          "#2980B9",
    "InsuranceCoverageTerm":     "#16A085",
    "Event":                     "#D35400",
    "InsurableProperty":         "#C0392B",
    # singletons / unknown
    "_singleton":                "#95A5A6",
}


# ---------------------------------------------------------------------------
# Helper: rounded box with text
# ---------------------------------------------------------------------------
def _draw_box(ax, x, y, width, height, text, facecolor, textcolor=WHITE,
              fontsize=8, wrap_width=45, linecolor=None, linestyle="-", lw=1.5,
              bold=False, family="monospace"):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02",
                         facecolor=facecolor,
                         edgecolor=linecolor or facecolor,
                         linewidth=lw,
                         linestyle=linestyle,
                         transform=ax.transData, clip_on=False)
    ax.add_patch(box)
    wrapped = "\n".join(textwrap.wrap(text, wrap_width))
    ax.text(x + width / 2, y + height / 2, wrapped,
            ha="center", va="center",
            fontsize=fontsize, color=textcolor,
            fontfamily=family,
            fontweight="bold" if bold else "normal",
            transform=ax.transData,
            wrap=False)


def _arrow(ax, x0, y0, x1, y1, color=LIGHT_GRAY, lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


# ===========================================================================
# FIGURE 1: Eval Q&A Examples (4-panel 2×2)
# ===========================================================================

def make_eval_qa_figure(outpath: str):
    fig = plt.figure(figsize=(18, 12), facecolor=BG_DARK)
    fig.suptitle("Evaluation Q&A Examples — Zone 3 (8b+EDC+MultiLeiden, 75% Accuracy)",
                 color=WHITE, fontsize=15, fontweight="bold", y=0.97)

    panels = [
        # (row, col, correct, task_id, title, question, cypher, result, why)
        (0, 0, True,  "Q2",
         "Q2 — ICC Coverage Limit",
         "What is the maximum coverage limit\nfor ICC (Coverage D)?",
         "MATCH (n)-[r:HAS_COVERAGE_LIMIT]->(m)\nWHERE n.id CONTAINS 'Coverage D'\nRETURN n, r, m",
         "Coverage D ──[HAS_COVERAGE_LIMIT]──► $30,000  ✓\nCoverage D ──[PROVIDES_PERSONAL_PROPERTY_COVERAGE]──► Increased Cost of Compliance",
         "Zone 2 extracted the exact numeric triple in\npass 2 (numeric focus pass).",
        ),
        (0, 1, True,  "Q6",
         "Q6 — Lake Flood Duration",
         "Under NFIP, how many continuous days must a\nlake flood an insured building before a\npre-loss claim may be filed?",
         "MATCH (n)-[r:HAS_COVERAGE_LIMIT]->(m)\nWHERE n.id CONTAINS 'Continuous Lake'\nRETURN n, r, m",
         "Continuous Lake Flood ──[HAS_COVERAGE_LIMIT]──► 90 days  ✓",
         "Section VII.Q chunk extracted the temporal\ncondition correctly. '90 days' captured\nin pass 2 (numeric focus).",
        ),
        (1, 0, False, "Q12",
         "Q12 — Building Definition",
         "What is the definition of a 'building'\nunder NFIP?",
         "MATCH (n)-[r:DEFINED_AS]->(m)\nWHERE toLower(n.id) CONTAINS 'building'\nRETURN n, r, m",
         "Result: 0 rows  ✗",
         "Entity 'Building' not extracted as standalone\nnode. Zone 2 extracted 'Building Property',\n'Building Coverage', 'Building and Personal\nProperty' — but not the definition-bearing\nentity from Section II.",
        ),
        (1, 1, False, "Q17",
         "Q17 — Improvements & Betterments",
         "What improvements and betterments are\ncovered under NFIP?",
         "MATCH (n)-[r]->(m)\nWHERE toLower(n.id) CONTAINS 'improvement'\nRETURN n, r, m",
         "Result: 0 rows  ✗",
         "Zone 1 chunked improvements/betterments\ncontent into a larger section. Zone 2\nextracted coverage triples for that chunk\nbut 'improvements and betterments' was not\ncaptured as a standalone entity — absorbed\ninto generic 'Personal Property'.",
        ),
    ]

    for (row, col, correct, tag, title, question, cypher, result, why) in panels:
        # Create subplot axes
        ax = fig.add_axes([0.03 + col * 0.495,
                           0.06 + (1 - row) * 0.455,
                           0.46, 0.42])
        ax.set_facecolor(BG_PANEL)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        accent = GREEN if correct else RED
        badge  = "[CORRECT]" if correct else "[WRONG]"

        # Title bar
        title_box = FancyBboxPatch((0, 8.8), 10, 1.0,
                                   boxstyle="round,pad=0.05",
                                   facecolor=accent, edgecolor="none",
                                   transform=ax.transData)
        ax.add_patch(title_box)
        ax.text(0.25, 9.3, badge, ha="left", va="center",
                fontsize=11, fontweight="bold", color=WHITE, transform=ax.transData)
        ax.text(9.75, 9.3, title, ha="right", va="center",
                fontsize=9, color=WHITE, transform=ax.transData)

        # Question section
        ax.text(0.2, 8.5, "Question", ha="left", va="bottom",
                fontsize=8, color=LIGHT_GRAY, fontstyle="italic", transform=ax.transData)
        q_box = FancyBboxPatch((0.1, 7.2), 9.8, 1.2,
                               boxstyle="round,pad=0.05",
                               facecolor=BG_CARD, edgecolor=BLUE, linewidth=1,
                               transform=ax.transData)
        ax.add_patch(q_box)
        ax.text(5.0, 7.8, question, ha="center", va="center",
                fontsize=8.5, color=WHITE, transform=ax.transData)

        # Arrow
        ax.annotate("", xy=(5, 6.9), xytext=(5, 7.2),
                    arrowprops=dict(arrowstyle="-|>", color=LIGHT_GRAY, lw=1.2))

        # Cypher section
        ax.text(0.2, 6.8, "Cypher Query", ha="left", va="bottom",
                fontsize=8, color=LIGHT_GRAY, fontstyle="italic", transform=ax.transData)
        c_box = FancyBboxPatch((0.1, 5.4), 9.8, 1.3,
                               boxstyle="round,pad=0.05",
                               facecolor="#1A1D2E", edgecolor=CYAN, linewidth=1,
                               transform=ax.transData)
        ax.add_patch(c_box)
        ax.text(0.4, 6.0, cypher, ha="left", va="center",
                fontsize=7, color=CYAN, fontfamily="monospace",
                transform=ax.transData)

        # Arrow
        ax.annotate("", xy=(5, 5.1), xytext=(5, 5.4),
                    arrowprops=dict(arrowstyle="-|>", color=LIGHT_GRAY, lw=1.2))

        # Result section
        ax.text(0.2, 5.0, "Result", ha="left", va="bottom",
                fontsize=8, color=LIGHT_GRAY, fontstyle="italic", transform=ax.transData)
        r_edge = GREEN_GLOW if correct else RED
        r_box = FancyBboxPatch((0.1, 3.7), 9.8, 1.2,
                               boxstyle="round,pad=0.05",
                               facecolor="#1A1D2E", edgecolor=r_edge, linewidth=2,
                               transform=ax.transData)
        ax.add_patch(r_box)
        ax.text(0.4, 4.3, result, ha="left", va="center",
                fontsize=7.5, color=GREEN_GLOW if correct else RED,
                fontfamily="monospace", transform=ax.transData)

        # Why section
        why_bg = "#1A2E1A" if correct else "#2E1A1A"
        w_box = FancyBboxPatch((0.1, 0.3), 9.8, 3.1,
                               boxstyle="round,pad=0.05",
                               facecolor=why_bg, edgecolor=accent, linewidth=1,
                               linestyle="--", transform=ax.transData)
        ax.add_patch(w_box)
        ax.text(0.3, 3.1, "Why " + ("[CORRECT]" if correct else "[WRONG]"), ha="left", va="bottom",
                fontsize=8, color=accent, fontweight="bold", transform=ax.transData)
        ax.text(0.3, 1.85, why, ha="left", va="center",
                fontsize=7.5, color=LIGHT_GRAY, transform=ax.transData)

    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print(f"  ✓ Saved: {outpath}")


# ===========================================================================
# FIGURE 2: Prompt Examples (3 panels, vertical stack)
# ===========================================================================

BOOTSTRAP_PROMPT = """\
RELATION_BOOTSTRAP_PROMPT (zone2/prompts.py)

You are a knowledge graph expert.
Read these sample passages from an insurance domain document.
Propose 12-18 SNAKE_CASE relation type names that capture the
relationships described in these passages.

You MUST include at least 2 relation types from EACH of these 5 categories:
  1. Coverage / exclusion  (e.g. COVERS, EXCLUDED_FROM, DOES_NOT_COVER)
  2. Amounts / limits      (e.g. HAS_COVERAGE_LIMIT, HAS_DEDUCTIBLE)
  3. Time periods          (e.g. HAS_WAITING_PERIOD, HAS_DEADLINE)
  4. Definitions           (e.g. DEFINED_AS, IS_CLASSIFIED_AS)
  5. Obligations / steps   (e.g. MUST_NOTIFY, MUST_FILE, PRECEDES)

Do NOT include generic types: HAS, IS, CONTAINS, INCLUDES, RELATES_TO.
Respond with ONLY a JSON array of strings.
"""

EXTRACTION_PROMPT = """\
SYSTEM_PROMPT_TEMPLATE + PASS_FOCUS_INSTRUCTIONS (zone2/prompts.py)

You are a knowledge graph extractor for insurance policy documents.
Extract (subject, relation, object) triples from the passage provided.

Rules:
  - Extract ALL important facts, not just the most prominent one
  - Subject of a definition triple MUST be the exact term being defined
  - Dollar amounts and time periods are valid objects: "$30,000", "90 days"
  - Negation ("not covered", "excluded") → use an exclusion relation
  - MANDATORY LIST EXTRACTION: When a passage lists items
    "A, B, C are excluded", extract EACH item as a separate triple
  - Extract 3-5 triples per passage

Pass 2 focus (numeric facts only):
  Extract ONLY triples where the object is a specific numeric value:
  a dollar amount, time period, percentage, or count/threshold.
  Examples: HAS_COVERAGE_LIMIT, HAS_DEDUCTIBLE, HAS_WAITING_PERIOD
"""

CLUSTER_NAMING_PROMPT = """\
CLUSTER NAMING PROMPT (zone3/pipeline.py — _name_cluster_list)

You are an ontology engineer building an insurance domain ontology.
These entity names all belong to the same semantic cluster:

  ["Coverage D", "Building Coverage", "$30,000",
   "Deductible", "Contents Coverage", "Building and Personal Property"]

Propose ONE canonical PascalCase ontology class name that best describes
the abstract concept shared by all these entities.
The name should be general enough to cover all members but specific
enough to be meaningful in an insurance ontology.

Respond with ONLY the class name (one word or compound word, PascalCase):

→ Model output: InsuranceTerm
"""


def make_prompts_figure(outpath: str):
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), facecolor=BG_DARK)
    fig.suptitle("Pipeline Prompt Examples — Domain-Agnostic Design",
                 color=WHITE, fontsize=14, fontweight="bold", y=0.99)
    plt.subplots_adjust(hspace=0.15, top=0.96, bottom=0.02, left=0.02, right=0.98)

    prompts = [
        ("Zone 2 — Step 1: Relation Bootstrapping",
         "LLM reads document samples and proposes relation types — no hardcoding",
         BOOTSTRAP_PROMPT, BLUE),
        ("Zone 2 — Step 2: 3-Pass Extraction",
         "3 extraction passes with different focus; dedup by (subject, relation, object)",
         EXTRACTION_PROMPT, ORANGE),
        ("Zone 3 — Cluster Naming (One-Shot)",
         "LLM receives entity cluster members and proposes a PascalCase ontology class",
         CLUSTER_NAMING_PROMPT, PURPLE),
    ]

    for ax, (title, subtitle, prompt_text, color) in zip(axes, prompts):
        ax.set_facecolor(BG_PANEL)
        ax.axis("off")

        # Header band
        ax.add_patch(FancyBboxPatch((0, 0.82), 1, 0.17,
                                    boxstyle="square,pad=0",
                                    facecolor=color, edgecolor="none",
                                    transform=ax.transAxes))
        ax.text(0.012, 0.92, title, ha="left", va="center",
                fontsize=11, fontweight="bold", color=WHITE,
                transform=ax.transAxes)
        ax.text(0.012, 0.85, subtitle, ha="left", va="center",
                fontsize=8.5, color=WHITE, alpha=0.85,
                transform=ax.transAxes)

        # Code block background
        ax.add_patch(FancyBboxPatch((0.005, 0.01), 0.99, 0.80,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#141622", edgecolor=color,
                                    linewidth=1.5,
                                    transform=ax.transAxes))

        ax.text(0.015, 0.795, prompt_text,
                ha="left", va="top",
                fontsize=8, color="#E8F4FD",
                fontfamily="monospace",
                transform=ax.transAxes,
                linespacing=1.4)

    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print(f"  ✓ Saved: {outpath}")


# ===========================================================================
# FIGURE 3: KG Subgraph Visualization
# ===========================================================================

def _load_graph_data():
    """Load triples and named_clusters from run summaries."""
    # Triples
    triples = []
    if os.path.exists(ZONE2_SUMMARY):
        with open(ZONE2_SUMMARY) as f:
            d = json.load(f)
            triples = d.get("triples", [])
    else:
        print(f"  ⚠  {ZONE2_SUMMARY} not found — using empty triples")

    # Named clusters + hierarchy
    named_clusters = []
    hierarchy = []
    if os.path.exists(ZONE3_SUMMARY):
        with open(ZONE3_SUMMARY) as f:
            d = json.load(f)
            named_clusters = d.get("named_clusters", [])
            hierarchy = d.get("hierarchy", [])
    else:
        print(f"  ⚠  {ZONE3_SUMMARY} not found — using empty clusters")

    return triples, named_clusters, hierarchy


def _build_entity_class_map(named_clusters):
    """Map entity → class name from named clusters."""
    entity_to_class = {}
    for cluster in named_clusters:
        class_name = cluster.get("class_name", "_singleton")
        for member in cluster.get("members", []):
            entity_to_class[member] = class_name
    return entity_to_class


def _node_color(node, entity_to_class, class_nodes):
    """Return color for a node."""
    if node in class_nodes:
        return GOLD
    cls = entity_to_class.get(node, None)
    if cls is None:
        return GRAY
    return CLASS_COLORS.get(cls, GRAY)


# Nodes that answered evaluation questions correctly
CORRECT_NODES = {"Coverage D", "$30,000", "Continuous Lake Flood", "90 days"}
# Gap node to annotate
GAP_NODE = "Building"

# Key triples to highlight (subject, object)
KEY_EDGES_CORRECT = {
    ("Coverage D", "$30,000"),
    ("Continuous Lake Flood", "90 days"),
}


def make_kg_subgraph_figure(outpath: str):
    triples, named_clusters, hierarchy = _load_graph_data()

    if not triples:
        print("  ⚠  No triples data — skipping KG figure")
        return

    entity_to_class = _build_entity_class_map(named_clusters)

    # Build networkx graph from triples
    G = nx.DiGraph()
    edge_labels = {}
    for t in triples:
        s = t.get("subject", "")
        r = t.get("relation", "")
        o = t.get("object", "")
        if s and o and len(s) < 50 and len(o) < 50:
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, relation=r)
            edge_labels[(s, o)] = r.replace("_", " ").lower()

    # Add OntologyClass nodes for primary classes
    class_nodes = set()
    for cluster in named_clusters:
        cls = cluster.get("class_name", "")
        if cls and len(cluster.get("members", [])) > 1:
            class_nodes.add(f"[{cls}]")
            G.add_node(f"[{cls}]")

    # Add SUBCLASS_OF edges between class nodes
    hier_edges = []
    for h in hierarchy:
        child_node  = f"[{h.get('child', '')}]"
        parent_node = f"[{h.get('parent', '')}]"
        if child_node in class_nodes and parent_node in G.nodes:
            G.add_node(parent_node)
            G.add_edge(child_node, parent_node, relation="SUBCLASS_OF")
            hier_edges.append((child_node, parent_node))

    # Limit graph to a readable subgraph centered on key entities
    # Focus: entities in the primary clusters that were evaluated
    focus_entities = set()
    FOCUS_CLASSES = {
        "InsuredProperty", "InsuranceTerm", "PolicyParticipantOrEvent",
        "Timeframe", "InsuredAsset",
    }
    for cluster in named_clusters:
        if cluster.get("class_name", "") in FOCUS_CLASSES:
            for m in cluster.get("members", []):
                focus_entities.add(m)

    # Keep nodes reachable from focus entities (1-hop)
    subgraph_nodes = set(focus_entities)
    for s, o in list(G.edges()):
        if s in focus_entities or o in focus_entities:
            subgraph_nodes.add(s)
            subgraph_nodes.add(o)
    subgraph_nodes |= class_nodes  # Always include class nodes
    # Limit to avoid overcrowding
    subgraph_nodes = {n for n in subgraph_nodes if len(n) < 45}

    Gsub = G.subgraph(subgraph_nodes).copy()
    # Also add hierarchy edges even if nodes were trimmed
    for c, p in hier_edges:
        Gsub.add_edge(c, p, relation="SUBCLASS_OF")

    # Layout
    np.random.seed(42)
    pos = nx.spring_layout(Gsub, seed=42, k=2.5, iterations=80)

    # Separate class nodes and entity nodes
    entity_nodes = [n for n in Gsub.nodes() if n not in class_nodes]
    ont_nodes    = [n for n in Gsub.nodes() if n in class_nodes]

    # Compute degree-based sizes
    degrees = dict(Gsub.degree())
    base_entity_size = 300
    base_class_size  = 1200

    entity_sizes = [max(base_entity_size, base_entity_size + degrees.get(n, 0) * 80)
                    for n in entity_nodes]
    class_sizes  = [base_class_size for _ in ont_nodes]

    # Colors
    entity_colors = [_node_color(n, entity_to_class, class_nodes) for n in entity_nodes]

    # Separate regular edges from hierarchy edges
    regular_edges = [(s, o) for s, o in Gsub.edges()
                     if (s, o) not in hier_edges and
                        Gsub.edges[s, o].get("relation") != "SUBCLASS_OF"]
    subclass_edges = [(s, o) for s, o in Gsub.edges()
                      if Gsub.edges[s, o].get("relation") == "SUBCLASS_OF"]

    correct_edges = [(s, o) for s, o in regular_edges if (s, o) in KEY_EDGES_CORRECT]
    normal_edges  = [(s, o) for s, o in regular_edges if (s, o) not in KEY_EDGES_CORRECT]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(20, 14), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")
    ax.set_title("Knowledge Graph Subgraph — Zone 3 (8b+EDC+MultiLeiden)\n"
                 "Node color = ontology class  |  Gold border = OntologyClass node  |  "
                 "Green glow = correct answer node  |  [WRONG] = extraction gap",
                 color=WHITE, fontsize=11, pad=12)

    # Draw normal edges
    nx.draw_networkx_edges(Gsub, pos, edgelist=normal_edges, ax=ax,
                           edge_color=LIGHT_GRAY, alpha=0.35, width=0.8,
                           arrows=True, arrowsize=10,
                           connectionstyle="arc3,rad=0.05")

    # Draw correct answer edges in green
    nx.draw_networkx_edges(Gsub, pos, edgelist=correct_edges, ax=ax,
                           edge_color=GREEN_GLOW, alpha=0.85, width=2.5,
                           arrows=True, arrowsize=14,
                           connectionstyle="arc3,rad=0.05")

    # Draw SUBCLASS_OF edges as thick dashed arrows
    nx.draw_networkx_edges(Gsub, pos, edgelist=subclass_edges, ax=ax,
                           edge_color=GOLD, alpha=0.7, width=2.0,
                           style="dashed", arrows=True, arrowsize=16,
                           connectionstyle="arc3,rad=0.15")

    # Draw entity nodes
    nx.draw_networkx_nodes(Gsub, pos, nodelist=entity_nodes, ax=ax,
                           node_color=entity_colors,
                           node_size=entity_sizes, alpha=0.9)

    # Draw OntologyClass nodes with gold border
    nx.draw_networkx_nodes(Gsub, pos, nodelist=ont_nodes, ax=ax,
                           node_color="#2A2A10",
                           node_size=class_sizes,
                           edgecolors=GOLD, linewidths=3, alpha=0.95)

    # Green glow around correct answer nodes
    correct_in_graph = [n for n in CORRECT_NODES if n in Gsub.nodes()]
    if correct_in_graph:
        nx.draw_networkx_nodes(Gsub, pos, nodelist=correct_in_graph, ax=ax,
                               node_color=GREEN_GLOW,
                               node_size=[degrees.get(n, 1) * 80 + 600
                                          for n in correct_in_graph],
                               alpha=0.3)
        nx.draw_networkx_nodes(Gsub, pos, nodelist=correct_in_graph, ax=ax,
                               node_color=GREEN_GLOW,
                               node_size=[degrees.get(n, 1) * 80 + 350
                                          for n in correct_in_graph],
                               alpha=0.85)

    # Node labels — shorten long names
    def _shorten(n):
        if n.startswith("[") and n.endswith("]"):
            return n[1:-1]  # remove brackets for display
        if len(n) > 22:
            return n[:20] + "…"
        return n

    labels = {n: _shorten(n) for n in Gsub.nodes()}
    nx.draw_networkx_labels(Gsub, pos, labels=labels, ax=ax,
                            font_size=6.5, font_color=WHITE,
                            font_weight="bold")

    # ❌ Gap annotation for missing "Building" entity
    # Place it near the "Building Property" node if present
    gap_anchor = None
    for candidate in ["Building Property", "Building Coverage",
                       "Building and Personal Property", "Coverage A"]:
        if candidate in Gsub.nodes():
            gap_anchor = pos[candidate]
            break
    if gap_anchor is None:
        gap_anchor = (0.0, 0.5)
        # Convert from data coords to axes coords manually if needed

    gx, gy = gap_anchor
    ax.annotate(
        "[WRONG] 'Building'\n(definition entity\nnot extracted)",
        xy=(gx, gy),
        xytext=(gx - 0.6, gy + 0.55),
        fontsize=8, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5,
                        connectionstyle="arc3,rad=-0.3"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2E1A1A",
                  edgecolor=RED, linewidth=1.5),
    )

    # Legend
    legend_items = []
    for cls, col in [
        ("InsuredAsset",              "#4A90D9"),
        ("InsuranceTerm",             "#27AE60"),
        ("Timeframe",                 "#E67E22"),
        ("PolicyParticipantOrEvent",  "#9B59B6"),
        ("InsuredProperty",           "#E74C3C"),
        ("InsurancePolicyProvisions", "#1ABC9C"),
        ("OntologyClass node",        GOLD),
        ("Correct answer node",       GREEN_GLOW),
    ]:
        legend_items.append(mpatches.Patch(color=col, label=cls))

    ax.legend(handles=legend_items, loc="lower left",
              facecolor=BG_PANEL, edgecolor=LIGHT_GRAY,
              labelcolor=WHITE, fontsize=7.5,
              framealpha=0.9, ncol=2)

    # SUBCLASS_OF edge legend note
    ax.text(0.01, 0.02,
            "─ ─ ► Gold dashed = SUBCLASS_OF (algorithmic, 5 edges)",
            color=GOLD, fontsize=7.5, transform=ax.transAxes)
    ax.text(0.01, 0.055,
            "──── ► Green = correct answer edge",
            color=GREEN_GLOW, fontsize=7.5, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print(f"  ✓ Saved: {outpath}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n── Generating presentation figures ──────────────────────────────")

    print("\n[1/3] Eval Q&A figure (4-panel)...")
    make_eval_qa_figure(os.path.join(RESULTS_DIR, "presentation_eval_qa.png"))

    print("\n[2/3] Prompt examples figure (3-panel)...")
    make_prompts_figure(os.path.join(RESULTS_DIR, "presentation_prompts.png"))

    print("\n[3/3] KG subgraph figure...")
    make_kg_subgraph_figure(os.path.join(RESULTS_DIR, "presentation_kg_subgraph.png"))

    print("\n── Done ─────────────────────────────────────────────────────────")
    print(f"Output directory: {RESULTS_DIR}")
    print("  presentation_eval_qa.png")
    print("  presentation_prompts.png")
    print("  presentation_kg_subgraph.png")
    print("\nOpen with:  open data/results/presentation_*.png")


if __name__ == "__main__":
    main()
