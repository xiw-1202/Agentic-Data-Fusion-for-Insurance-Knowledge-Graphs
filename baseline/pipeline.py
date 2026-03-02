"""
Baseline Pipeline — LangChain LLMGraphTransformer → Neo4j + LLM Ontology Induction

This is the improved comparison baseline described in the project plan (Appendix D).
No entity resolution, no cross-domain transfer.
Includes LLM-based ontology induction (maps extracted labels → Riskine classes)
as a simpler alternative to Zone 3's Leiden community detection approach.
Used to measure: entity duplication, type inconsistency, query accuracy, Riskine alignment.

Usage:
  python3 baseline/pipeline.py              # original 512-token chunks
  python3 baseline/pipeline.py --zone1      # Zone 1 section-aware chunks (ablation)
"""

import json
import time
import os
import sys
import argparse
from typing import TypedDict, Annotated
import operator

# Allow imports from project root (config.py lives there)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

import config
from ontology_induction import run_ontology_induction


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    chunks: list[dict]
    graph_documents: list          # GraphDocument objects from LLMGraphTransformer
    neo4j_stats: dict              # nodes/rels inserted, errors
    ontology_induction: dict       # label → Riskine class mapping + metrics
    errors: Annotated[list, operator.add]
    model: str                     # Ollama model name


# ---------------------------------------------------------------------------
# LLM + Graph
# ---------------------------------------------------------------------------

def get_llm(model: str = config.OLLAMA_MODEL):
    return ChatOllama(
        model=model,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
    )


def get_neo4j_graph():
    return Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

ZONE1_CHUNKS_FILE = config.ZONE1_CHUNKS_FILE
PDF_SOURCE_KEY = "fema_F-123-general-property-SFIP_2021.pdf"  # zone1 source name for PDF


def load_chunks(state: PipelineState) -> PipelineState:
    """Load original 512-token chunks."""
    print("\n[1/4] Loading chunks (original 512-token)...")
    with open(config.CHUNKS_FILE) as f:
        chunks = json.load(f)
    print(f"  ✓ Loaded {len(chunks)} chunks from {config.CHUNKS_FILE}")
    return {"chunks": chunks}


def load_chunks_zone1(state: PipelineState) -> PipelineState:
    """Load Zone 1 section-aware chunks (PDF only — CSV chunks too large for LLM)."""
    print("\n[1/4] Loading Zone 1 section-aware PDF chunks...")
    with open(ZONE1_CHUNKS_FILE) as f:
        all_chunks = json.load(f)
    # Keep only PDF chunks (CSV chunks are 10K+ tokens — not suitable for LLMGraphTransformer)
    pdf_chunks = [c for c in all_chunks if c["source"] == PDF_SOURCE_KEY]
    print(f"  ✓ Loaded {len(pdf_chunks)} PDF chunks from {ZONE1_CHUNKS_FILE}")
    print(f"    (filtered from {len(all_chunks)} total; excluded CSV chunks)")
    return {"chunks": pdf_chunks}


def extract_triples(state: PipelineState) -> PipelineState:
    """
    Baseline extraction: LLMGraphTransformer with no filtering.
    Direct extraction — no entity resolution, no type clustering.
    Expected issues: 20-30% entity duplication, type inconsistency.
    """
    model = state.get("model", config.OLLAMA_MODEL)
    print(f"\n[2/4] Extracting graph triples with LLMGraphTransformer ({model})...")
    llm = get_llm(model)

    transformer = LLMGraphTransformer(
        llm=llm,
        # No allowed_nodes or allowed_relationships — let LLM decide freely
        # This is the baseline: unconstrained extraction
    )

    chunks = state["chunks"]
    docs = [
        Document(
            page_content=c["content"],
            metadata={
                "chunk_id": c["chunk_id"],
                "page": c.get("page", c.get("pages", [-1])[0] if c.get("pages") else -1),
                "source": c["source"],
                "section_hierarchy": c.get("section_hierarchy", []),
            },
        )
        for c in chunks
    ]

    graph_documents = []
    errors = []

    for i, doc in enumerate(docs):
        hierarchy = doc.metadata.get("section_hierarchy", [])
        label = " > ".join(hierarchy) if hierarchy else f"page {doc.metadata['page']}"
        print(f"  Chunk {i+1}/{len(docs)} [{label[:55]}]...", end=" ", flush=True)
        try:
            result = transformer.convert_to_graph_documents([doc])
            n_nodes = sum(len(gd.nodes) for gd in result)
            n_rels = sum(len(gd.relationships) for gd in result)
            print(f"→ {n_nodes} nodes, {n_rels} rels")
            graph_documents.extend(result)
        except Exception as e:
            print(f"✗ ERROR: {e}")
            errors.append({"chunk_id": i, "error": str(e)})
        time.sleep(0.1)  # small pause to avoid hammering Ollama

    total_nodes = sum(len(gd.nodes) for gd in graph_documents)
    total_rels = sum(len(gd.relationships) for gd in graph_documents)
    print(f"\n  ✓ Total extracted: {total_nodes} nodes, {total_rels} relationships")
    print(f"  ✗ Errors: {len(errors)} chunks failed")

    return {"graph_documents": graph_documents, "errors": errors}


def insert_to_neo4j(state: PipelineState) -> PipelineState:
    """
    Baseline Neo4j insertion: direct, no deduplication.
    This will create duplicate nodes — intentional for baseline measurement.
    """
    print("\n[3/4] Inserting into Neo4j AuraDB (direct, no entity resolution)...")

    if not state["graph_documents"]:
        print("  ✗ No graph documents to insert.")
        return {"neo4j_stats": {"nodes_inserted": 0, "rels_inserted": 0, "error": "no documents"}}

    try:
        graph = get_neo4j_graph()
        graph.add_graph_documents(
            state["graph_documents"],
            baseEntityLabel=True,   # adds :__Entity__ label to all nodes
            include_source=True,    # links nodes back to source Document nodes
        )

        # Count what's in the graph
        node_count = graph.query("MATCH (n) RETURN count(n) AS count")[0]["count"]
        rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
        label_count = graph.query(
            "CALL db.labels() YIELD label RETURN count(label) AS count"
        )[0]["count"]

        stats = {
            "nodes_inserted": node_count,
            "rels_inserted": rel_count,
            "unique_labels": label_count,
        }
        print(f"  ✓ Graph: {node_count} nodes, {rel_count} rels, {label_count} unique labels")
        return {"neo4j_stats": stats}

    except Exception as e:
        print(f"  ✗ Neo4j error: {e}")
        return {"neo4j_stats": {"error": str(e)}}


def induce_ontology(state: PipelineState) -> PipelineState:
    """
    LLM-based ontology induction step.
    Runs after Neo4j insertion; maps free-form extracted labels → Riskine
    ontology classes and adds those classes as additional node labels.
    """
    model = state.get("model", config.OLLAMA_MODEL)
    print(f"\n[4/4] Running LLM ontology induction ({model})...")
    llm   = get_llm(model)
    graph = get_neo4j_graph()
    result = run_ontology_induction(graph, llm)
    print(f"  ✓ Labels seen:      {result['labels_seen']}")
    print(f"  ✓ Labels mapped:    {result['labels_mapped']}  (→ Riskine class)")
    print(f"  - Labels unmapped:  {result['labels_unmapped']}  (→ Other)")
    print(f"  ✓ Nodes relabelled: {result['nodes_relabelled']}")
    for orig, cls in result.get("mapping", {}).items():
        if cls != "Other":
            print(f"      {orig!r:30s} → {cls}")
    return {"ontology_induction": result}


# ---------------------------------------------------------------------------
# Graph / Compile
# ---------------------------------------------------------------------------

def clear_neo4j():
    """Wipe all nodes and relationships from the AuraDB database."""
    print("  Clearing Neo4j graph...")
    graph = get_neo4j_graph()
    graph.query("MATCH (n) DETACH DELETE n")
    count = graph.query("MATCH (n) RETURN count(n) AS cnt")[0]["cnt"]
    print(f"  ✓ Graph cleared (nodes remaining: {count})")


def build_pipeline(zone1: bool = False):
    builder = StateGraph(PipelineState)

    loader_node = load_chunks_zone1 if zone1 else load_chunks
    builder.add_node("load_chunks",      loader_node)
    builder.add_node("extract_triples",  extract_triples)
    builder.add_node("insert_to_neo4j",  insert_to_neo4j)
    builder.add_node("induce_ontology",  induce_ontology)

    builder.set_entry_point("load_chunks")
    builder.add_edge("load_chunks",     "extract_triples")
    builder.add_edge("extract_triples", "insert_to_neo4j")
    builder.add_edge("insert_to_neo4j", "induce_ontology")
    builder.add_edge("induce_ontology", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline(zone1: bool = False, model: str = config.OLLAMA_MODEL):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    mode_label = "Zone1 Chunks (Ablation)" if zone1 else "Original 512-Token Chunks"
    print("=" * 60)
    print(f"CS584 Capstone — Baseline Pipeline [{mode_label}]")
    print(f"LangChain LLMGraphTransformer → Neo4j AuraDB  (model: {model})")
    print("=" * 60)

    print("\n[0/4] Clearing existing Neo4j graph for clean run...")
    clear_neo4j()

    pipeline = build_pipeline(zone1=zone1)
    start = time.time()
    result = pipeline.invoke({
        "chunks": [], "graph_documents": [], "neo4j_stats": {},
        "ontology_induction": {}, "errors": [],
        "model": model,
    })
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Errors: {len(result.get('errors', []))}")
    print(f"Neo4j stats: {result.get('neo4j_stats', {})}")

    oi = result.get("ontology_induction", {})
    if oi:
        print(f"Ontology induction: {oi.get('labels_mapped', 0)}/{oi.get('labels_seen', 0)} labels mapped, "
              f"{oi.get('nodes_relabelled', 0)} nodes relabelled")

    # Save run summary
    suffix = "_zone1" if zone1 else "_original"
    summary = {
        "mode": "zone1" if zone1 else "original",
        "model": model,
        "elapsed_seconds": round(elapsed, 2),
        "chunks_processed": len(result.get("chunks", [])),
        "graph_documents": len(result.get("graph_documents", [])),
        "errors": result.get("errors", []),
        "neo4j_stats": result.get("neo4j_stats", {}),
        "ontology_induction": result.get("ontology_induction", {}),
    }
    out_path = os.path.join(config.RESULTS_DIR, f"baseline_run_summary{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline pipeline runner")
    parser.add_argument("--zone1", action="store_true",
                        help="Use Zone 1 section-aware chunks instead of original 512-token chunks")
    parser.add_argument("--model", default=config.OLLAMA_MODEL,
                        help=f"Ollama model name (default: {config.OLLAMA_MODEL})")
    args = parser.parse_args()
    run_baseline(zone1=args.zone1, model=args.model)
