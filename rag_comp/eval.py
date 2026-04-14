import os
import sys
import json
import time
import argparse
from typing import List, Dict

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config
from rag_comp.document_rag import DocumentRAG
from rag_comp.graph_rag import GraphRAG
from rag_comp.hybrid_rag import HybridRAG
from rag_comp.judge import RAGJudge

def load_tasks():
    path = os.path.join(_ROOT, config.RESULTS_DIR, 'evaluation_datasets', 'auto_gold_standard.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"Warning: {path} not found. Ensure you run rag_comp/generate_eval.py first.")
    return []

def run_benchmark(num_tasks: int = None, mode: str = "compare", model_name: str = None):
    print("="*60)
    print(f"RAG Comparison Benchmark ({mode.upper()})")
    print(f"Model: {model_name or config.OLLAMA_MODEL}")
    print("="*60)
    
    # Initialize RAG engines
    doc_rag = None
    graph_rag = None
    hybrid_rag = None
    
    if mode in ["doc", "compare"]:
        doc_rag = DocumentRAG(model_name=model_name)
    if mode in ["graph", "compare"]:
        graph_rag = GraphRAG(model_name=model_name)
    if mode in ["hybrid", "compare"]:
        hybrid_rag = HybridRAG(model_name=model_name)
        
    judge = RAGJudge(model_name=model_name)
    tasks = load_tasks()
    if num_tasks:
        tasks = tasks[:num_tasks]
        
    if not tasks:
        print("Error: No evaluation tasks found.")
        return
        
    results = []
    
    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] Task: {task['question'][:50]}...")
        question = task['question']
        keywords = task['keywords']
        
        answers = {}
        
        if doc_rag:
            res = doc_rag.generate_answer(question)
            answers["doc"] = res.get("answer", "ERROR")
            
        if graph_rag:
            res = graph_rag.generate_answer(question)
            answers["graph"] = res.get("answer", "ERROR")
            
        if hybrid_rag:
            res = hybrid_rag.generate_answer(question)
            answers["hybrid"] = res.get("answer", "ERROR")
            
        # Judge the results
        judgment = judge.judge_answers(question, keywords, answers)
        
        entry = {
            "task_id": task.get("id", i),
            "question": question,
            "keywords": keywords,
            "mode_results": answers,
            "judgment": judgment
        }
        results.append(entry)
        
    # Summarize and Save
    out_path = os.path.join(_ROOT, config.RESULTS_DIR, f"benchmark_{mode}_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✓ Benchmark complete. Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified RAG Benchmark")
    parser.add_argument("-n", "--num", type=int, help="Number of tasks to run.")
    parser.add_argument("--mode", choices=["doc", "graph", "hybrid", "compare"], default="compare")
    parser.add_argument("--model", help="Ollama model name (overrides config)")
    
    args = parser.parse_args()
    
    # Manual fix for a common user request: handle qwen vs llama
    selected_model = args.model
    if selected_model == "alt":
        selected_model = config.OLLAMA_MODEL_ALT
        
    run_benchmark(num_tasks=args.num, mode=args.mode, model_name=selected_model)
