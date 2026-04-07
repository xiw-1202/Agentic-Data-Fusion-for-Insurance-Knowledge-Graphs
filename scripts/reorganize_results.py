#!/usr/bin/env python3
"""Reorganize data/results/ from flat directory to structured layout.

Dry-run by default. Pass --execute to actually move files.

Layout:
  data/results/
  ├── flood/           (flood insurance results)
  │   ├── eval/        (all eval results)
  │   ├── svloi/       (SV-LOI method versions)
  │   └── rsi/         (RSI-LCR method results)
  ├── emory/           (Emory Spring 2026 results)
  │   └── eval/
  └── visualizations/  (charts/reports)
"""

import os
import shutil
import sys

RESULTS_DIR = "data/results"


def classify_file(filename: str) -> str:
    """Return destination subdirectory for a result file."""
    f = filename.lower()

    # Visualizations
    if f.endswith((".png", ".html")):
        return "visualizations"

    # Emory/new data results
    if "emory" in f:
        if any(k in f for k in ["eval", "riskine", "extraction_quality", "baseline_eval"]):
            return "emory/eval"
        return "emory"

    # SV-LOI results (flood)
    if "svloi" in f or "sv_loi" in f:
        if "provenance" in f or "summary" in f:
            return "flood/svloi"
        if any(k in f for k in ["riskine", "baseline_eval", "extraction"]):
            return "flood/eval"
        return "flood/svloi"

    # RSI-LCR results (flood)
    if "rsi" in f:
        if any(k in f for k in ["riskine", "baseline_eval"]):
            return "flood/eval"
        return "flood/rsi"

    # Eval results (flood)
    if any(k in f for k in ["riskine_eval", "baseline_eval", "extraction_quality"]):
        return "flood/eval"

    # Zone 2/3 core outputs (flood)
    if any(k in f for k in ["zone2_", "zone3_"]):
        return "flood"

    # Catch-all
    return "flood"


def main():
    execute = "--execute" in sys.argv

    if not os.path.exists(RESULTS_DIR):
        print(f"ERROR: {RESULTS_DIR} not found")
        sys.exit(1)

    # Collect files (skip subdirectories that already exist)
    files = [
        f for f in os.listdir(RESULTS_DIR)
        if os.path.isfile(os.path.join(RESULTS_DIR, f))
    ]

    if not files:
        print("No files to reorganize.")
        return

    # Plan moves
    moves: list[tuple[str, str]] = []
    for f in sorted(files):
        dest_subdir = classify_file(f)
        src = os.path.join(RESULTS_DIR, f)
        dst = os.path.join(RESULTS_DIR, dest_subdir, f)
        moves.append((src, dst))

    # Print plan
    print(f"{'DRY RUN' if not execute else 'EXECUTING'}: "
          f"{len(moves)} files to reorganize\n")

    subdirs = set()
    for src, dst in moves:
        rel_src = os.path.relpath(src, RESULTS_DIR)
        rel_dst = os.path.relpath(dst, RESULTS_DIR)
        print(f"  {rel_src:55s} → {rel_dst}")
        subdirs.add(os.path.dirname(dst))

    if not execute:
        print(f"\nDry run complete. Pass --execute to move files.")
        print(f"Directories to create: {sorted(os.path.relpath(d, RESULTS_DIR) for d in subdirs)}")
        return

    # Execute moves
    for d in subdirs:
        os.makedirs(d, exist_ok=True)

    moved = 0
    for src, dst in moves:
        if os.path.exists(dst):
            print(f"  SKIP (exists): {dst}")
            continue
        shutil.move(src, dst)
        moved += 1

    print(f"\nMoved {moved}/{len(moves)} files.")


if __name__ == "__main__":
    main()
