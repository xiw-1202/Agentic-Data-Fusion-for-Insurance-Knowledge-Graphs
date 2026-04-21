import os
import sys
import json
import time
import argparse
import signal
from contextlib import contextmanager
from typing import List, Dict

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config
from rag_comp.document_rag import DocumentRAG
from rag_comp.graph_rag import GraphRAG
from rag_comp.hybrid_rag import HybridRAG
from rag_comp.judge import RAGJudge

DATASET_ALIASES = {
    "doc_primary": config.DOC_PRIMARY_EVAL_FILE,
    "graph_primary": config.GRAPH_PRIMARY_EVAL_FILE,
    "combined": None,
}


@contextmanager
def _time_limit(seconds: int | None):
    if not seconds or seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _normalize_task(task: Dict, dataset_name: str) -> Dict:
    normalized = dict(task)
    normalized.setdefault("dataset", dataset_name)
    return normalized


def _load_dataset_file(relative_or_absolute_path: str, dataset_name: str) -> List[Dict]:
    path = (
        relative_or_absolute_path
        if os.path.isabs(relative_or_absolute_path)
        else os.path.join(_ROOT, relative_or_absolute_path)
    )

    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return []

    with open(path, "r", encoding="utf-8") as f:
        return [_normalize_task(task, dataset_name) for task in json.load(f)]


def load_tasks(input_file: str = None, mode: str = "compare"):
    if input_file:
        if input_file == "combined":
            doc_tasks = _load_dataset_file(config.DOC_PRIMARY_EVAL_FILE, "doc_primary")
            graph_tasks = _load_dataset_file(config.GRAPH_PRIMARY_EVAL_FILE, "graph_primary")
            return doc_tasks + graph_tasks

        resolved_input = DATASET_ALIASES.get(input_file, input_file)
        dataset_name = input_file if input_file in DATASET_ALIASES else "custom"
        return _load_dataset_file(resolved_input, dataset_name)

    if mode == "doc":
        return _load_dataset_file(config.DOC_PRIMARY_EVAL_FILE, "doc_primary")

    if mode in {"graph", "hybrid", "compare"}:
        doc_tasks = _load_dataset_file(config.DOC_PRIMARY_EVAL_FILE, "doc_primary")
        graph_tasks = _load_dataset_file(config.GRAPH_PRIMARY_EVAL_FILE, "graph_primary")
        return doc_tasks + graph_tasks

    return []


def _run_stage(label: str, fn, timeout_seconds: int | None):
    start = time.time()
    try:
        with _time_limit(timeout_seconds):
            result = fn()
        return {
            "ok": True,
            "result": result,
            "latency_seconds": time.time() - start,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "result": None,
            "latency_seconds": time.time() - start,
            "error": str(exc),
        }


def _build_failed_mode_result(mode_name: str, error: str) -> Dict:
    return {
        "answer": f"ERROR: {mode_name} stage failed: {error}",
        "error": error,
    }


def _build_failed_judgment(error: str, answers: Dict[str, str], latencies: Dict[str, float]) -> Dict:
    return {
        "error": error,
        "winner": "error",
        "ranking": list(answers.keys()),
        "rationale": "Evaluation failed due to timeout or runtime error.",
        "modes": {
            mode_name: {
                "keyword_coverage": 0.0,
                "faithfulness": 0.0,
                "relevancy": 0.0,
                "completeness": 0.0,
                "insurance_factor": 0.0,
                "overall_score": 0,
                "hallucination_detected": False,
                "latency_seconds": latency,
                "error": answers.get(mode_name, ""),
            }
            for mode_name, latency in latencies.items()
        },
    }


def run_benchmark(
    num_tasks: int = None,
    mode: str = "compare",
    model_name: str = None,
    output_file: str = None,
    resume: bool = False,
    input_file: str = None,
    stage_timeout: int | None = 90,
    judge_timeout: int | None = 90,
):
    print("="*60)
    print(f"RAG Comparison Benchmark ({mode.upper()})")
    print(f"Input: {input_file or 'Default'}")
    print(f"Model: {model_name or config.OLLAMA_MODEL}")
    print(f"Stage timeout: {stage_timeout or 'disabled'}s | Judge timeout: {judge_timeout or 'disabled'}s")
    print("="*60)
    
    # Setup output path
    if output_file:
        out_path = os.path.join(_ROOT, output_file) if not os.path.isabs(output_file) else output_file
    else:
        out_path = os.path.join(_ROOT, config.RESULTS_DIR, f"benchmark_{mode}_{int(time.time())}.json")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Initialize results and check for resume
    results = []
    completed_task_ids = set()
    
    if resume and os.path.exists(out_path):
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                completed_task_ids = {r.get("task_id") for r in results if "task_id" in r}
                print(f"✓ Resuming from {out_path}. Found {len(results)} existing results.")
        except Exception as e:
            print(f"Warning: Could not load existing results for resume: {e}")
            results = []

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
    # Load tasks
    tasks = load_tasks(input_file=input_file, mode=mode)
    
    # Filter tasks if resuming
    if completed_task_ids:
        tasks = [t for t in tasks if str(t.get("id")) not in (str(tid) for tid in completed_task_ids)]
        print(f"✓ Skipped {len(completed_task_ids)} tasks. Remaining: {len(tasks)}")

    if num_tasks:
        tasks = tasks[:num_tasks]
        
    if not tasks:
        if completed_task_ids:
            print("All tasks are already completed.")
        else:
            print("Error: No evaluation tasks found.")
        return
    
    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] Task: {task['question'][:50]}...")
        question = task['question']
        keywords = task['keywords']
        expected_answer = task.get("expected_answer", "")
        
        answers = {}
        latencies = {}
        evidence_by_mode = {}
        mode_results = {}
        
        if doc_rag:
            print("  -> Running doc stage...")
            stage = _run_stage("doc", lambda: doc_rag.generate_answer(question), stage_timeout)
            latencies["doc"] = stage["latency_seconds"]
            if stage["ok"]:
                res = stage["result"]
                answers["doc"] = res.get("answer", "ERROR")
                evidence_by_mode["doc"] = res.get("context_text", "No document context.")
                mode_results["doc"] = res
                print(f"     doc done in {latencies['doc']:.2f}s")
            else:
                error = stage["error"] or "Unknown error."
                answers["doc"] = f"ERROR: {error}"
                evidence_by_mode["doc"] = f"doc stage failed: {error}"
                mode_results["doc"] = _build_failed_mode_result("doc", error)
                print(f"     doc failed in {latencies['doc']:.2f}s: {error}")
            
        if graph_rag:
            print("  -> Running graph stage...")
            stage = _run_stage("graph", lambda: graph_rag.generate_answer(question), stage_timeout)
            latencies["graph"] = stage["latency_seconds"]
            if stage["ok"]:
                res = stage["result"]
                answers["graph"] = res.get("answer", "ERROR")
                evidence_by_mode["graph"] = res.get("context_used", "No graph context.")
                mode_results["graph"] = res
                print(f"     graph done in {latencies['graph']:.2f}s")
            else:
                error = stage["error"] or "Unknown error."
                answers["graph"] = f"ERROR: {error}"
                evidence_by_mode["graph"] = f"graph stage failed: {error}"
                mode_results["graph"] = _build_failed_mode_result("graph", error)
                print(f"     graph failed in {latencies['graph']:.2f}s: {error}")
            
        if hybrid_rag:
            print("  -> Running hybrid stage...")
            stage = _run_stage("hybrid", lambda: hybrid_rag.generate_answer(question), stage_timeout)
            latencies["hybrid"] = stage["latency_seconds"]
            if stage["ok"]:
                res = stage["result"]
                answers["hybrid"] = res.get("answer", "ERROR")
                evidence_by_mode["hybrid"] = (
                    f"Knowledge Graph Context:\n{res.get('graph_context', 'No graph context.')}\n\n"
                    f"Document Context:\n{res.get('doc_context', 'No document context.')}"
                )
                mode_results["hybrid"] = res
                print(f"     hybrid done in {latencies['hybrid']:.2f}s")
            else:
                error = stage["error"] or "Unknown error."
                answers["hybrid"] = f"ERROR: {error}"
                evidence_by_mode["hybrid"] = f"hybrid stage failed: {error}"
                mode_results["hybrid"] = _build_failed_mode_result("hybrid", error)
                print(f"     hybrid failed in {latencies['hybrid']:.2f}s: {error}")
            
        # Judge the results
        print("  -> Running judge stage...")
        judgment_stage = _run_stage(
            "judge",
            lambda: judge.judge_answers(
                question,
                keywords,
                answers,
                expected_answer=expected_answer,
                evidence_by_mode=evidence_by_mode,
            ),
            judge_timeout,
        )
        if judgment_stage["ok"]:
            judgment = judgment_stage["result"]
            print(f"     judge done in {judgment_stage['latency_seconds']:.2f}s")
        else:
            error = judgment_stage["error"] or "Unknown error."
            judgment = _build_failed_judgment(error, answers, latencies)
            print(f"     judge failed in {judgment_stage['latency_seconds']:.2f}s: {error}")
        
        # Merge automated metrics into judgment for recording
        if "modes" in judgment:
            for mode_name, latency in latencies.items():
                if mode_name in judgment["modes"]:
                    judgment["modes"][mode_name]["latency_seconds"] = latency

        entry = {
            "task_id": task.get("id", i),
            "task": task,
            "mode_results": mode_results,
            "judgment": judgment
        }
        results.append(entry)
        
        # Iterative Save
        with open(out_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
    print(f"\n✓ Benchmark complete. Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified RAG Benchmark")
    parser.add_argument("-n", "--num", type=int, help="Number of tasks to run.")
    parser.add_argument("--mode", choices=["doc", "graph", "hybrid", "compare"], default="compare")
    parser.add_argument("--model", help="Ollama model name (overrides config)")
    parser.add_argument("-o", "--output", help="Output JSON filename (relative to root or absolute)")
    parser.add_argument("-i", "--input", help="Input dataset JSON filename")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--stage-timeout", type=int, default=90, help="Per-mode timeout in seconds; 0 disables.")
    parser.add_argument("--judge-timeout", type=int, default=90, help="Judge timeout in seconds; 0 disables.")
    
    args = parser.parse_args()
    
    # Manual fix for a common user request: handle qwen vs llama
    selected_model = args.model
    if selected_model == "alt":
        selected_model = config.OLLAMA_MODEL_ALT
        
    run_benchmark(
        num_tasks=args.num, 
        mode=args.mode, 
        model_name=selected_model,
        output_file=args.output,
        resume=args.resume,
        input_file=args.input,
        stage_timeout=args.stage_timeout or None,
        judge_timeout=args.judge_timeout or None,
    )
