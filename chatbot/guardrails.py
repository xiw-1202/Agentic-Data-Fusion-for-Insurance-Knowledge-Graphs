"""Safety guardrails for LLM-generated Cypher."""
from __future__ import annotations

import re

FORBIDDEN_KEYWORDS = {
    "CREATE",
    "MERGE",
    "DELETE",
    "REMOVE",
    "SET",
    "DROP",
    "DETACH",
    "LOAD",
    "CALL",
    "FOREACH",
}

MAX_RESULT_ROWS = 200


def is_read_only(cypher: str) -> tuple[bool, str]:
    """Return (ok, reason). Rejects any write keyword outside string literals."""
    stripped = re.sub(r"'[^']*'|\"[^\"]*\"", "", cypher)
    upper = stripped.upper()
    tokens = re.findall(r"\b[A-Z]+\b", upper)
    for t in tokens:
        if t in FORBIDDEN_KEYWORDS:
            return False, f"forbidden keyword: {t}"
    return True, "ok"


def clamp_limit(cypher: str, cap: int = MAX_RESULT_ROWS) -> str:
    """Append LIMIT if the query doesn't already have one."""
    if re.search(r"\bLIMIT\b", cypher, flags=re.I):
        return cypher
    return cypher.rstrip().rstrip(";") + f"\nLIMIT {cap}"
