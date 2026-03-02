"""
evaluation/riskine_loader.py
============================
Fetch and cache Riskine insurance ontology schemas from GitHub.
Used by riskine_eval.py to build ground-truth class list for P/R/F1 scoring.

Riskine ontology: https://github.com/riskine/ontology/tree/master/schemas/core
10 flood-relevant schemas are fetched; all 27 are cached for completeness.
"""

import json
import os
import urllib.request
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RISKINE_BASE = (
    "https://raw.githubusercontent.com/riskine/ontology/master/schemas/core/"
)

# 10 flood-relevant schemas used in P/R/F1 scoring
FLOOD_SCHEMAS = [
    "product",
    "coverage",
    "damage",
    "risk",
    "structure",
    "property",
    "person",
    "object",
    "organization",
    "address",
]

# Default local cache directory
CACHE_DIR = "data/riskine/schemas"


# ---------------------------------------------------------------------------
# Schema fetching and caching
# ---------------------------------------------------------------------------

def fetch_and_cache(
    cache_dir: str = CACHE_DIR,
    schemas: Optional[list[str]] = None,
    force_refresh: bool = False,
) -> dict[str, dict]:
    """
    Fetch Riskine schemas from GitHub; write each to cache_dir as <name>.json.
    On subsequent calls, reads from disk unless force_refresh=True.

    Returns:
        {schema_name: schema_dict}  — e.g. {"coverage": {...}, "product": {...}}
    """
    if schemas is None:
        schemas = FLOOD_SCHEMAS

    os.makedirs(cache_dir, exist_ok=True)
    result: dict[str, dict] = {}

    for name in schemas:
        cache_path = os.path.join(cache_dir, f"{name}.json")

        # Use cache if available and not stale
        if os.path.exists(cache_path) and not force_refresh:
            with open(cache_path) as f:
                result[name] = json.load(f)
            continue

        # Fetch from GitHub
        url = RISKINE_BASE + f"{name}.json"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
            result[name] = data
            print(f"  [riskine] Fetched {name}.json → {cache_path}")
        except Exception as e:
            print(f"  [riskine] WARNING: could not fetch {name}.json: {e}")
            # If cache exists from a previous run, fall back to it
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    result[name] = json.load(f)
                print(f"  [riskine] Using cached {name}.json (stale)")

    return result


# ---------------------------------------------------------------------------
# Schema → class list
# ---------------------------------------------------------------------------

def extract_riskine_classes(schemas: dict[str, dict]) -> list[dict]:
    """
    Convert loaded schemas into a flat list of class descriptors.

    Each descriptor:
        {
          "name":       str,         # PascalCase, e.g. "Coverage"
          "schema_id":  str,         # $id from JSON schema, e.g. ".../coverage.json"
          "properties": list[str],   # top-level property names
        }

    Class name is derived by PascalCase-ing the filename stem:
        "coverage" → "Coverage"
        "address"  → "Address"
        "product"  → "Product"
    """
    classes: list[dict] = []
    for name, schema in schemas.items():
        class_name = _to_pascal_case(name)
        schema_id  = schema.get("$id", name)

        # Collect top-level property names (skip $defs, allOf meta-keys)
        raw_props: dict = schema.get("properties", {})
        if not raw_props:
            # Some schemas nest properties inside allOf or definitions
            for entry in schema.get("allOf", []):
                if "properties" in entry:
                    raw_props = entry["properties"]
                    break

        prop_names = list(raw_props.keys())

        classes.append({
            "name":       class_name,
            "schema_id":  schema_id,
            "properties": prop_names,
        })

    return classes


def _to_pascal_case(s: str) -> str:
    """'some-hyphen-name' or 'snake_case' or 'lowercase' → PascalCase."""
    # Split on hyphens, underscores, or spaces
    import re
    parts = re.split(r'[-_\s]+', s)
    return "".join(p.capitalize() for p in parts if p)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Fetching Riskine flood schemas...")
    schemas = fetch_and_cache()
    print(f"\nLoaded {len(schemas)} schemas: {list(schemas.keys())}")

    classes = extract_riskine_classes(schemas)
    print(f"\nExtracted {len(classes)} Riskine classes:")
    for cls in classes:
        props_preview = cls["properties"][:4]
        print(f"  {cls['name']:<16}  props: {props_preview}")
