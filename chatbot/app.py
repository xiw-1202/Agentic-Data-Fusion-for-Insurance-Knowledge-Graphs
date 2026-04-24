"""Streamlit UI for the Emory KG chatbot.

Run:
    pip install -r chatbot/requirements.txt
    streamlit run chatbot/app.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st
from langchain_neo4j import Neo4jGraph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from chatbot.examples import EXAMPLES
from chatbot.qa_chain import ask, build_schema_prefix


st.set_page_config(page_title="SEAF-KG Chatbot", layout="wide")

st.title("SEAF-KG — Emory Insurance KG Chatbot")
st.caption(
    "Ask questions in English. Claude generates Cypher against the induced "
    "ontology, runs it on Neo4j, and explains the result."
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
        MATCH (c:Class) WITH entities, count(c) AS classes
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

if st.button("Run", type="primary") and question:
    with st.spinner("Claude is generating Cypher..."):
        result = ask(question, graph=graph, schema_prefix=schema_prefix)

    st.subheader("Answer")
    if not result.guardrail_ok:
        st.error(result.answer)
    else:
        st.write(result.answer)

    with st.expander("Generated Cypher", expanded=True):
        st.code(result.cypher, language="cypher")

    if result.rows:
        st.subheader(f"Rows ({len(result.rows)})")
        st.dataframe(pd.DataFrame(result.rows), use_container_width=True)
    else:
        st.info("Query returned no rows.")
