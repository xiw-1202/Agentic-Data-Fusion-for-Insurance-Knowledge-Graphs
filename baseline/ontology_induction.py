"""
baseline/ontology_induction.py
================================
LLM-based ontology induction for the improved baseline.

After unconstrained LLMGraphTransformer extraction and Neo4j insertion:
  1. Collect unique node labels from Neo4j (excluding internal labels)
  2. Batch-prompt LLM to map each label → Riskine class (or "Other")
  3. Write Riskine class as additional Neo4j label on matching nodes
  4. Return metrics dict for storage in pipeline results

This mirrors what Zone 3 does (Leiden community detection + ontology mapping)
but uses a simpler LLM-only approach — a fair comparison baseline.
"""

import re
from langchain_core.messages import HumanMessage

RISKINE_CLASSES = [
    "Coverage", "Product", "Damage", "Risk", "Structure",
    "Property", "Person", "Object", "Organization", "Address"
]

_INDUCTION_PROMPT = """\
You are an insurance domain expert. Map each extracted entity type to the \
closest Riskine flood-insurance ontology class.

Riskine classes (choose ONLY from these):
{classes}

Extracted entity types:
{labels}

Rules:
- Output exactly one line per entity type in the format: <EntityType> -> <RiskineClass>
- Use "Other" if no Riskine class fits well
- Be conservative: prefer "Other" over a weak or speculative match
- Use the exact Riskine class names listed above (case-sensitive)

Example output:
Policy -> Product
Flood zone -> Risk
Claimant -> Person
Event section -> Other
"""

_LINE_RE = re.compile(r'^(.+?)\s*->\s*(\w+)\s*$')


def get_unique_labels(graph) -> list[str]:
    """Query Neo4j for all unique node labels (excluding internal/meta labels)."""
    skip = {"__Entity__", "Document"}
    try:
        result = graph.query(
            "CALL db.labels() YIELD label RETURN collect(label) AS labels"
        )
        all_labels = result[0]["labels"] if result else []
        return [lbl for lbl in all_labels if lbl not in skip]
    except Exception as exc:
        print(f"  [ontology] Warning: could not retrieve labels — {exc}")
        return []


def induce_ontology_labels(labels: list[str], llm) -> dict[str, str]:
    """
    Single LLM call: map every extracted label → Riskine class (or 'Other').
    Returns dict of {extracted_label: riskine_class}.
    """
    if not labels:
        return {}

    prompt = _INDUCTION_PROMPT.format(
        classes=", ".join(RISKINE_CLASSES),
        labels="\n".join(f"- {lbl}" for lbl in labels),
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        print(f"  [ontology] LLM call failed — {exc}")
        return {lbl: "Other" for lbl in labels}

    mapping: dict[str, str] = {}
    for line in raw.splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        entity_type   = m.group(1).strip()
        riskine_class = m.group(2).strip()
        # Only accept valid Riskine class names or "Other"
        if riskine_class in RISKINE_CLASSES or riskine_class == "Other":
            mapping[entity_type] = riskine_class

    # Mark anything the LLM omitted as "Other"
    for lbl in labels:
        if lbl not in mapping:
            mapping[lbl] = "Other"

    return mapping


def apply_ontology_labels(graph, mapping: dict[str, str]) -> int:
    """
    For each extracted label mapped to a Riskine class, SET that class
    as an additional Neo4j label on every matching node.
    Original labels are preserved; the Riskine label is additive.
    Returns total count of nodes that received a new label.
    """
    total_relabelled = 0
    for original_label, riskine_class in mapping.items():
        if riskine_class == "Other":
            continue
        # Strip backticks defensively before interpolating into Cypher
        safe_original = original_label.replace("`", "")
        safe_class    = riskine_class.replace("`", "")
        try:
            result = graph.query(
                f"MATCH (n:`{safe_original}`) "
                f"WHERE NOT n:`{safe_class}` "
                f"SET n:`{safe_class}` "
                f"RETURN count(n) AS updated"
            )
            updated = result[0]["updated"] if result else 0
            total_relabelled += updated
        except Exception as exc:
            print(f"  [ontology] Could not relabel '{original_label}' → '{riskine_class}': {exc}")

    return total_relabelled


def run_ontology_induction(graph, llm) -> dict:
    """
    Full orchestration:
      1. Collect unique labels from Neo4j
      2. LLM mapping (single batch call)
      3. Apply Riskine class labels to Neo4j nodes
      4. Return metrics dict

    Return shape:
    {
        "labels_seen": int,
        "labels_mapped": int,       # mapped to a real Riskine class
        "labels_unmapped": int,     # mapped to "Other"
        "nodes_relabelled": int,    # total nodes that got a new label
        "mapping": {label: class}
    }
    """
    labels = get_unique_labels(graph)
    if not labels:
        return {
            "labels_seen": 0,
            "labels_mapped": 0,
            "labels_unmapped": 0,
            "nodes_relabelled": 0,
            "mapping": {},
        }

    mapping = induce_ontology_labels(labels, llm)

    mapped   = [lbl for lbl, cls in mapping.items() if cls != "Other"]
    unmapped = [lbl for lbl, cls in mapping.items() if cls == "Other"]

    nodes_relabelled = apply_ontology_labels(graph, mapping)

    return {
        "labels_seen":      len(labels),
        "labels_mapped":    len(mapped),
        "labels_unmapped":  len(unmapped),
        "nodes_relabelled": nodes_relabelled,
        "mapping":          mapping,
    }
