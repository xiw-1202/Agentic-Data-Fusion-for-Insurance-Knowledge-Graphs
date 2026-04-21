import argparse
import os
import subprocess
import sys
from typing import List


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVAL_SCRIPT = os.path.join(_ROOT, "rag_comp", "eval.py")
_AGGREGATE_SCRIPT = os.path.join(_ROOT, "rag_comp", "aggregate_results.py")


def _run_command(command: List[str]) -> None:
    print("=" * 80)
    print("Running:")
    print(" ".join(command))
    print("=" * 80)
    subprocess.run(command, check=True, cwd=_ROOT)


def _build_eval_command(
    dataset_name: str,
    output_path: str,
    model_name: str | None,
    num_tasks: int | None,
    stage_timeout: int,
    judge_timeout: int,
    resume: bool,
) -> List[str]:
    command = [
        sys.executable,
        _EVAL_SCRIPT,
        "--mode",
        "compare",
        "-i",
        dataset_name,
        "-o",
        output_path,
        "--stage-timeout",
        str(stage_timeout),
        "--judge-timeout",
        str(judge_timeout),
    ]

    if model_name:
        command.extend(["--model", model_name])
    if num_tasks is not None:
        command.extend(["-n", str(num_tasks)])
    if resume:
        command.append("--resume")

    return command


def _build_aggregate_command(output_path: str) -> List[str]:
    return [
        sys.executable,
        _AGGREGATE_SCRIPT,
        output_path,
    ]


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Run doc/graph/combined benchmarks and aggregate their results."
    )
    parser.add_argument(
        "--model",
        help="Optional Ollama model name to pass through to eval.py.",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        help="Optional limit on the number of tasks for each benchmark run.",
    )
    parser.add_argument(
        "--stage-timeout",
        type=int,
        default=60,
        help="Per-mode timeout in seconds passed through to eval.py.",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=45,
        help="Judge timeout in seconds passed through to eval.py.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume any benchmark output files that already exist.",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Run only doc_primary and graph_primary, without the combined benchmark.",
    )
    parser.add_argument(
        "--doc-output",
        default="data/results/doc_primary_benchmark.json",
        help="Output path for the doc_primary benchmark JSON.",
    )
    parser.add_argument(
        "--graph-output",
        default="data/results/graph_primary_benchmark.json",
        help="Output path for the graph_primary benchmark JSON.",
    )
    parser.add_argument(
        "--combined-output",
        default="data/results/final_benchmark.json",
        help="Output path for the combined benchmark JSON.",
    )
    args = parser.parse_args()

    benchmark_runs = [
        ("doc_primary", args.doc_output),
        ("graph_primary", args.graph_output),
    ]
    if not args.skip_combined:
        benchmark_runs.append(("combined", args.combined_output))

    print("Benchmark plan:")
    for dataset_name, output_path in benchmark_runs:
        print(f"  - {dataset_name} -> {output_path}")

    for dataset_name, output_path in benchmark_runs:
        eval_command = _build_eval_command(
            dataset_name=dataset_name,
            output_path=output_path,
            model_name=args.model,
            num_tasks=args.num,
            stage_timeout=args.stage_timeout,
            judge_timeout=args.judge_timeout,
            resume=args.resume,
        )
        _run_command(eval_command)

        aggregate_command = _build_aggregate_command(output_path)
        _run_command(aggregate_command)

    print("=" * 80)
    print("Finished running benchmark and aggregation pipeline.")
    print("=" * 80)


if __name__ == "__main__":
    run()
