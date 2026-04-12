"""Feedback loop: extract corrections from eval mismatches for few-shot learning.

After running riskine_eval.py, this script compares the entity_assignment_table
(which shows how induced classes map to Riskine classes) with the SV-LOI provenance
log to identify entity misclassifications. These corrections are formatted as
few-shot examples that can be prepended to the typing prompt in future runs.

Usage:
    python3 evaluation/feedback_loop.py \
        --eval-results data/results/emory/eval/riskine_eval_Emory_Spring2026_qwen2_5_72b.json \
        --provenance data/results/emory/svloi_provenance_Emory_Spring2026_qwen2_5_72b.json \
        --output data/results/emory/few_shot_corrections.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_corrections(
    eval_results_path: str,
    provenance_path: str,
    max_corrections: int = 20,
) -> list[dict]:
    """Compare eval mismatches with provenance to generate few-shot corrections.

    Looks at the entity_assignment_table from riskine_eval to find induced classes
    that mapped to a Riskine class with low score, then checks provenance for
    entities that were flagged/arbitrated in those classes.

    Returns list of correction dicts:
        [{entity, wrong_class, correct_class, reason, confidence}, ...]
    """
    with open(eval_results_path) as f:
        eval_data = json.load(f)

    with open(provenance_path) as f:
        provenance = json.load(f)

    corrections: list[dict] = []

    # Get the assignment table: [{induced, riskine, score, members}, ...]
    assignment_table = eval_data.get("entity_assignment_table", [])

    # Build map: induced_class -> (riskine_class, score)
    class_mapping: dict[str, tuple[str | None, float]] = {}
    for entry in assignment_table:
        induced = entry.get("induced", "")
        riskine = entry.get("riskine")
        score = entry.get("score", 0.0)
        if induced:
            class_mapping[induced] = (riskine, score)

    # Find low-confidence entities that were arbitrated (class changed)
    # These are the most informative corrections.
    for eid, prov in provenance.items():
        if not prov.get("arbitrated"):
            continue

        final_class = prov.get("final_type", "Other")
        pre_arb_class = prov.get("pre_arb_class", "")

        if final_class == "Other" or not pre_arb_class:
            continue

        # Check if the final class has a good Riskine match
        final_mapping = class_mapping.get(final_class, (None, 0.0))
        pre_mapping = class_mapping.get(pre_arb_class, (None, 0.0))

        # If the final class maps better to Riskine than the pre-arb class,
        # this is a useful correction example
        if final_mapping[1] > pre_mapping[1]:
            corrections.append({
                "entity": eid,
                "wrong_class": pre_arb_class,
                "correct_class": final_class,
                "reason": (
                    f"structural verification showed it belongs with "
                    f"{final_class} entities (maps to Riskine: {final_mapping[0]})"
                ),
                "confidence": prov.get("confidence", 0.5),
            })

    # Also find entities in classes with NO Riskine match (score=0)
    # that have low confidence — these might be systematically misclassified
    for eid, prov in provenance.items():
        if prov.get("confidence", 1.0) <= 0.3:
            final_class = prov.get("final_type", "Other")
            if final_class == "Other":
                continue
            mapping = class_mapping.get(final_class, (None, 0.0))
            if mapping[1] == 0.0:
                # This entity is in a class with no Riskine match
                # and has very low confidence — likely a misclassification
                llm_type = prov.get("llm_type", "Unknown")
                if llm_type != final_class and llm_type != "Other":
                    corrections.append({
                        "entity": eid,
                        "wrong_class": final_class,
                        "correct_class": llm_type,
                        "reason": "class has no domain match; original LLM typing may be better",
                        "confidence": 0.3,
                    })

    # Deduplicate by entity and sort by confidence (most confident first)
    seen = set()
    unique_corrections = []
    for c in corrections:
        if c["entity"] not in seen:
            seen.add(c["entity"])
            unique_corrections.append(c)

    unique_corrections.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return unique_corrections[:max_corrections]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract corrections from eval for few-shot learning")
    parser.add_argument("--eval-results", required=True, help="Path to riskine_eval output JSON")
    parser.add_argument("--provenance", required=True, help="Path to svloi_provenance JSON")
    parser.add_argument("--output", required=True, help="Output path for corrections JSON")
    parser.add_argument("--max-corrections", type=int, default=20, help="Max corrections to extract")
    args = parser.parse_args()

    corrections = extract_corrections(
        args.eval_results, args.provenance, args.max_corrections,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(corrections, f, indent=2)

    print(f"Extracted {len(corrections)} corrections → {args.output}")
    for c in corrections[:5]:
        print(f"  {c['entity']}: {c['wrong_class']} -> {c['correct_class']} ({c['reason'][:60]})")


if __name__ == "__main__":
    main()
