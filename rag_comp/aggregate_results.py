import json
import os
import sys
from collections import defaultdict

def aggregate_benchmark(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("Error: JSON file is empty.")
        return

    # Metrics to track
    metrics = [
        "keyword_coverage", 
        "faithfulness", 
        "relevancy", 
        "completeness", 
        "insurance_factor", 
        "overall_score", 
        "latency_seconds"
    ]
    
    # Aggregators
    overall_sums = defaultdict(lambda: defaultdict(float))
    overall_counts = defaultdict(lambda: defaultdict(int))
    category_data = defaultdict(lambda: {
        "sums": defaultdict(lambda: defaultdict(float)),
        "counts": defaultdict(lambda: defaultdict(int)),
        "wins": defaultdict(int),
        "total": 0
    })
    
    overall_wins = defaultdict(int)
    total_tasks = len(data)

    for entry in data:
        category = entry.get("task", {}).get("category", "unknown")
        judgment = entry.get("judgment", {})
        modes = judgment.get("modes", {})
        winner = judgment.get("winner")
        
        category_data[category]["total"] += 1
        if winner:
            overall_wins[winner] += 1
            category_data[category]["wins"][winner] += 1
            
        for mode_name, mode_metrics in modes.items():
            for m in metrics:
                if m in mode_metrics and mode_metrics[m] is not None:
                    val = mode_metrics[m]
                    # Overall
                    overall_sums[mode_name][m] += val
                    overall_counts[mode_name][m] += 1
                    # Category-wise
                    category_data[category]["sums"][mode_name][m] += val
                    category_data[category]["counts"][mode_name][m] += 1

    # Print Overall Results
    print("=" * 80)
    print(f"REPORT: {os.path.basename(file_path)}")
    print(f"Total Tasks: {total_tasks}")
    print("=" * 80)

    def print_table(header_text, sums, counts, wins, total):
        print(f"\n>>> {header_text} ({total} tasks)")
        header = f"{'Metric':<20} | {'Document RAG':<12} | {'Graph RAG':<12} | {'Hybrid RAG':<12}"
        print(header)
        print("-" * len(header))
        for m in metrics:
            row = f"{m:<20} | "
            for mode in ["doc", "graph", "hybrid"]:
                if counts[mode][m] > 0:
                    avg = sums[mode][m] / counts[mode][m]
                    row += f"{avg:>12.4f} | "
                else:
                    row += f"{'N/A':>12} | "
            print(row.rstrip(" | "))
        
        win_row = f"{'Win Rate (%)':<20} | "
        for mode in ["doc", "graph", "hybrid"]:
            rate = (wins[mode] / total) * 100 if total > 0 else 0
            win_row += f"{rate:>11.1f}% | "
        print(win_row.rstrip(" | "))
        print("-" * len(header))

    # 1. Overall Table
    print_table("OVERALL PERFORMANCE", overall_sums, overall_counts, overall_wins, total_tasks)

    # 2. Category-wise Tables
    print("\n" + "="*80)
    print("CATEGORY-WISE BREAKDOWN")
    print("="*80)
    for category in sorted(category_data.keys()):
        c_info = category_data[category]
        print_table(f"CATEGORY: {category.upper()}", c_info["sums"], c_info["counts"], c_info["wins"], c_info["total"])

    print(f"\nActual Total Wins: {dict(overall_wins)}")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Default to the one the user specified
        path = "data/results/final_benchmark.json"
        
    # Handle absolute paths or relative to execution
    if not os.path.exists(path) and not path.startswith("/"):
        # Try relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(script_dir)
        new_path = os.path.join(root, path)
        if os.path.exists(new_path):
            path = new_path
        
    aggregate_benchmark(path)
