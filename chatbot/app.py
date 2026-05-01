"""Streamlit UI for the Emory KG chatbot.

Run:
    pip install -r chatbot/requirements.txt
    streamlit run chatbot/app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st
from langchain_neo4j import Neo4jGraph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from chatbot.examples import EXAMPLES
from chatbot.qa_chain import QAResult, _coerce_viz, ask, ask_stream, build_schema_prefix
from chatbot.render import render, render_sources


st.set_page_config(page_title="SEAF-KG Chatbot", layout="wide")

st.title("SEAF-KG — Emory Insurance KG Chatbot")
st.caption(
    "Ask questions in English. Claude reasons about your intent, writes Cypher "
    "against the induced ontology, runs it on Neo4j, and picks the best way to "
    "present the answer (table, bar, pie, graph, …)."
)


@st.cache_resource(show_spinner="Connecting to Neo4j...")
def get_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )


@st.cache_resource(show_spinner="Building schema prefix...")
def get_schema_prefix(_graph: Neo4jGraph) -> str:
    return build_schema_prefix(_graph)


graph = get_graph()
schema_prefix = get_schema_prefix(graph)

with st.sidebar:
    st.header("Target KG")
    st.code(f"{config.NEO4J_URI}\ndb: {config.NEO4J_DATABASE}", language="text")

    counts = graph.query(
        """
        MATCH (e:Entity) WITH count(e) AS entities
        MATCH (c:OntologyClass) WITH entities, count(c) AS classes
        MATCH ()-[r]->() RETURN entities, classes, count(r) AS rels
        """
    )
    if counts:
        row = counts[0]
        st.metric("Entities", row["entities"])
        st.metric("Classes", row["classes"])
        st.metric("Relationships", row["rels"])

    st.divider()
    st.header("Try one of these")
    for ex in EXAMPLES:
        if st.button(ex["question"], key=ex["question"], use_container_width=True):
            st.session_state["q"] = ex["question"]

    with st.expander("Schema prefix sent to Claude"):
        st.text(schema_prefix)


question = st.text_input(
    "Ask a question",
    value=st.session_state.get("q", ""),
    placeholder="e.g. Which device types have the most claims?",
)

# Run only triggers a new stream. Rendering reads from session_state so
# subsequent reruns (e.g. from feedback buttons) don't lose the answer.
if st.button("Run", type="primary") and question:
    steps: list = []
    final_payload: dict = {}
    classify_payload: dict = {}

    for step in ask_stream(question, graph=graph, schema_prefix=schema_prefix):
        steps.append(step)
        with st.status(step.title, expanded=(step.name in {"classify", "plan", "cite"})) as status:
            if step.name == "classify":
                classify_payload = step.payload
                st.write(f"**Kind:** `{step.payload['kind']}`")
                st.write(f"**Why:** {step.payload['reason']}")
                st.progress(step.payload["confidence"], text=f"confidence {step.payload['confidence']:.0%}")
            elif step.name == "plan":
                st.write(f"**Intent:** {step.payload.get('intent','')}")
                st.write(f"**Approach:** {step.payload.get('approach','')}")
                if step.payload.get("ontology_classes_used"):
                    st.write("**Classes used:** " + ", ".join(step.payload["ontology_classes_used"]))
            elif step.name == "cypher":
                st.code(step.payload["cypher"], language="cypher")
            elif step.name == "execute":
                st.write(f"Returned {len(step.payload['rows'])} rows.")
            elif step.name == "interpret":
                st.json(step.payload)
            elif step.name == "cite":
                final_payload = step.payload
            status.update(
                label=("❌ " + step.title) if not step.ok else ("✅ " + step.title),
                state=("error" if not step.ok else "complete"),
            )

    # Persist for re-render after rerun (feedback button etc.)
    st.session_state["last_question"] = question
    st.session_state["last_steps"] = [
        {"name": s.name, "title": s.title, "payload": s.payload, "ok": s.ok} for s in steps
    ]
    st.session_state["last_classify"] = classify_payload
    st.session_state["last_payload"] = final_payload

# --- Render answer block from session_state (survives reruns) ---
final_payload = st.session_state.get("last_payload", {})
classify_payload = st.session_state.get("last_classify", {})
question_for_render = st.session_state.get("last_question", "")

if final_payload:
    if classify_payload.get("kind") == "open_interpretive":
        st.warning("⚠️ Interpretive answer — the KG doesn't directly answer this. Verify against the sources below.")
    elif classify_payload.get("kind") == "out_of_scope":
        st.error("This question is out of scope for the current KG.")

    if final_payload.get("summary"):
        if final_payload.get("rows"):
            # render() owns Answer + Key insight + viz when there are rows
            result = QAResult(
                question=question_for_render,
                cypher="",
                rows=final_payload["rows"],
                summary=final_payload["summary"],
                key_insight=final_payload.get("key_insight", ""),
                viz=_coerce_viz({"viz": final_payload.get("viz", {})}, final_payload["rows"]),
            )
            render(result)
        else:
            # No rows (text/open answers) — render Answer ourselves
            st.markdown("### Answer")
            st.write(final_payload["summary"])
            if final_payload.get("key_insight"):
                st.info(f"**Key insight:** {final_payload['key_insight']}")

        render_sources(final_payload.get("sources", []), graph)

        # --- CSV ground-truth check ---
        # For each relation in the generated Cypher, compare KG edge count
        # to the source CSV's non-null count for the same column. Surfaces
        # extraction gaps so the user can judge how trustworthy the answer is.
        steps_persisted = st.session_state.get("last_steps", [])
        cypher_step = next(
            (s for s in steps_persisted if s["name"] == "cypher"), None
        )
        if cypher_step:
            from chatbot.eval.verify import verify_relations
            checks = verify_relations(cypher_step["payload"].get("cypher", ""))
            if checks:
                st.markdown("### CSV ground-truth check")
                check_rows = []
                for c in checks:
                    # Find the corresponding kg_edges count for this rel
                    kg_count = graph.query(
                        "MATCH ()-[r]->() WHERE type(r) = $rel RETURN count(r) AS n",
                        params={"rel": c["relation"]},
                    )
                    n_kg = kg_count[0]["n"] if kg_count else 0
                    cov = (n_kg / c["csv_non_null"] * 100) if c["csv_non_null"] else 0
                    flag = "🟢" if cov >= 70 else ("🟡" if cov >= 30 else "🔴")
                    check_rows.append(
                        {
                            "": flag,
                            "Relation": c["relation"],
                            "CSV column": f"{c['column']} ({c['csv_file']})",
                            "CSV non-null": c["csv_non_null"],
                            "KG edges": n_kg,
                            "Coverage": f"{cov:.0f}%",
                        }
                    )
                st.dataframe(check_rows, use_container_width=True, hide_index=True)
                low_cov = [c for c in check_rows if c[""] == "🔴"]
                if low_cov:
                    rels = ", ".join(c["Relation"] for c in low_cov)
                    st.warning(
                        f"⚠️ Low extraction coverage on {rels} — answer may "
                        "miss rows that exist in the source CSV."
                    )

# --- Feedback (only show after an answer exists) ---
if final_payload:
    st.divider()
    st.markdown("**Was this answer helpful?**")
    col_up, col_down, col_comment = st.columns([1, 1, 6])
    with col_up:
        up_clicked = st.button("👍", key="fb_up")
    with col_down:
        down_clicked = st.button("👎", key="fb_down")
    comment = col_comment.text_input(
        "Comment (optional)", key="fb_comment", label_visibility="collapsed"
    )

    verdict = "up" if up_clicked else ("down" if down_clicked else None)
    if verdict:
        from chatbot.feedback import log_feedback
        from pathlib import Path
        log_feedback(
            Path("chatbot/feedback.jsonl"),
            question=st.session_state.get("last_question", ""),
            verdict=verdict,
            comment=comment,
            trace=st.session_state.get("last_steps", []),
        )
        st.toast(f"Thanks — logged {verdict}.")
