"""Streamlit renderers for each viz type Claude can pick."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from chatbot.qa_chain import QAResult


def _df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _coerce_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col and col in df.columns:
        df = df.copy()
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def render(result: QAResult) -> None:
    if not result.guardrail_ok:
        st.error(result.summary)
        return

    if result.intent:
        with st.expander("Claude's reading of the question", expanded=False):
            st.markdown(f"**Intent:** {result.intent}")

    st.markdown("### Answer")
    st.write(result.summary)

    if result.key_insight:
        st.info(f"**Key insight:** {result.key_insight}")

    df = _df(result.rows)
    viz = result.viz

    st.markdown(f"### {viz.title or 'Result'}")

    if viz.type == "text" or df.empty:
        if df.empty:
            st.caption("Query returned no rows.")
        return

    if viz.type == "scalar" and viz.value in df.columns:
        val = df[viz.value].iloc[0]
        st.metric(label=viz.title or viz.value, value=val)
        return

    if viz.type == "bar" and viz.x in df.columns and viz.y in df.columns:
        df = _coerce_numeric(df, viz.y)
        fig = px.bar(df, x=viz.x, y=viz.y, title=viz.title)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Data"):
            st.dataframe(df, use_container_width=True)
        return

    if viz.type == "line" and viz.x in df.columns and viz.y in df.columns:
        df = _coerce_numeric(df, viz.y)
        fig = px.line(df, x=viz.x, y=viz.y, title=viz.title, markers=True)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Data"):
            st.dataframe(df, use_container_width=True)
        return

    if viz.type == "pie" and viz.label in df.columns and viz.value in df.columns:
        df = _coerce_numeric(df, viz.value)
        fig = px.pie(df, names=viz.label, values=viz.value, title=viz.title, hole=0.35)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Data"):
            st.dataframe(df, use_container_width=True)
        return

    if viz.type == "graph" and viz.source in df.columns and viz.target in df.columns:
        _render_graph(df, viz.source, viz.target, title=viz.title)
        with st.expander("Edges"):
            st.dataframe(df, use_container_width=True)
        return

    # default: table
    st.dataframe(df, use_container_width=True, height=min(600, 40 + 35 * len(df)))


def render_sources(sources: list[dict[str, str]], graph) -> None:
    """Render each source chunk as a clickable expander showing the original text.

    Looks up :Chunk by composite (id, source) so PDF chunk #16 and CSV row #16
    don't collide.
    """
    if not sources:
        return
    st.markdown(f"### Sources ({len(sources)})")
    pairs = [
        {"id": str(s["chunk_id"]), "source": s.get("source") or ""}
        for s in sources
        if s.get("chunk_id") is not None
    ]
    if not pairs:
        return
    rows = graph.query(
        """
        UNWIND $pairs AS p
        OPTIONAL MATCH (c:Chunk {id: p.id, source: p.source})
        RETURN p.id AS id, p.source AS source, c.text AS text
        """,
        params={"pairs": pairs},
    )
    by_key = {(r["id"], r["source"]): r for r in rows}
    for s in sources:
        cid = str(s.get("chunk_id", ""))
        src = s.get("source") or ""
        rec = by_key.get((cid, src))
        label = f"📄 {cid}" + (f" — {src}" if src else "")
        with st.expander(label, expanded=False):
            if rec and rec.get("text"):
                st.caption(f"Source file: `{src or '(unknown)'}`")
                st.text(rec["text"])
            else:
                st.caption(f"Source file: `{src or '(unknown)'}`")
                st.info(
                    f"No stored text for chunk `{cid}` from `{src}`. "
                    "Open the source file directly to see the original record."
                )


def _render_graph(df: pd.DataFrame, src_col: str, tgt_col: str, title: str = "") -> None:
    """Render a node-edge graph using streamlit-agraph.

    Falls back to a graphviz DOT rendering if streamlit-agraph isn't installed.
    """
    edges = df[[src_col, tgt_col]].dropna().astype(str).values.tolist()
    nodes = sorted({n for pair in edges for n in pair})

    try:
        from streamlit_agraph import Config, Edge, Node, agraph

        ag_nodes = [Node(id=n, label=n, size=15) for n in nodes]
        ag_edges = [Edge(source=s, target=t) for s, t in edges]
        cfg = Config(
            width="100%",
            height=500,
            directed=True,
            physics=True,
            hierarchical=False,
        )
        agraph(nodes=ag_nodes, edges=ag_edges, config=cfg)
        if title:
            st.caption(title)
    except ImportError:
        dot = ["digraph G {", '  rankdir=LR;']
        for n in nodes:
            dot.append(f'  "{n}";')
        for s, t in edges:
            dot.append(f'  "{s}" -> "{t}";')
        dot.append("}")
        st.graphviz_chart("\n".join(dot))
