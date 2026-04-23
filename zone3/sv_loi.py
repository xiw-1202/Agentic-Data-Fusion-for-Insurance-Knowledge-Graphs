"""SV-LOI facade — preserves the public API. Implementation lives in zone3/_svloi/."""
from zone3._svloi.utils import (
    get_llm,
    get_neo4j_graph,
    _invoke_llm,
    _sanitize_label,
    _parse_json_safely,
)
from zone3._svloi.typing import type_value_entities, propagate_to_records
from zone3._svloi.hierarchy import derive_interclass_edges
from zone3._svloi.writer import write_ontology
from zone3._svloi.pipeline import run_sv_loi, main

__all__ = [
    "get_llm",
    "get_neo4j_graph",
    "_invoke_llm",
    "_sanitize_label",
    "_parse_json_safely",
    "type_value_entities",
    "propagate_to_records",
    "derive_interclass_edges",
    "write_ontology",
    "run_sv_loi",
]


if __name__ == "__main__":
    main()
