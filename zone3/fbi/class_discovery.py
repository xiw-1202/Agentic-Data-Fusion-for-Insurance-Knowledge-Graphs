"""Phase 2: Multi-iteration class discovery from file fingerprints.

Iteration 1 — Prefix grouping (algorithmic): find_prefix_groups, detect_sibling_patterns
Iteration 2 — Semantic grouping (LLM): semantic_group_headers
Iteration 3 — Cross-file merging (algorithmic): merge_cross_file_classes
Iteration 4 — Naming (LLM): name_classes
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from zone3.fbi.llm_utils import llm_call_json


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CandidateClass:
    """A candidate ontology class discovered from file headers."""

    prefix: str
    headers: list[str] = field(default_factory=list)
    suffixes: list[str] = field(default_factory=list)
    source_file: str = ""
    name: str = ""
    children: list[CandidateClass] = field(default_factory=list)
    parent: CandidateClass | None = field(default=None, repr=False)
    level: int = 0  # 0=root, 1=top, 2=sub, 3=leaf
    source_files: list[str] = field(default_factory=list)
    unique_headers: list[str] = field(default_factory=list)
    shared_headers: list[str] = field(default_factory=list)


@dataclass
class SiblingGroup:
    """A group of CandidateClasses that share the same suffix pattern."""

    common_prefix: str
    suffix_pattern: list[str] = field(default_factory=list)
    children: list[CandidateClass] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _longest_common_prefix(strings: list[str]) -> str:
    """Find the longest common prefix among a list of strings."""
    if not strings:
        return ""
    shortest = min(strings, key=len)
    for i, char in enumerate(shortest):
        if any(s[i] != char for s in strings):
            return shortest[:i]
    return shortest


# ---------------------------------------------------------------------------
# Iteration 1 — Prefix grouping (algorithmic)
# ---------------------------------------------------------------------------


def find_prefix_groups(
    headers: list[str],
    min_group_size: int = 3,
    separator: str = "_",
) -> list[CandidateClass]:
    """Build prefix trie and find groups where >= min_group_size headers share a prefix.

    Longer (more specific) prefixes take priority. Headers claimed by a
    specific prefix are not available for broader prefixes.
    """
    if not headers:
        return []

    # Build mapping: prefix -> list of headers that have that prefix
    prefix_to_headers: dict[str, list[str]] = defaultdict(list)

    for header in headers:
        parts = header.split(separator)
        # Generate all possible prefixes (from longest to shortest, at least 1 part)
        for length in range(1, len(parts)):
            prefix = separator.join(parts[:length])
            prefix_to_headers[prefix].append(header)

    # Filter to prefixes with enough members
    valid_prefixes = {
        prefix: hdrs
        for prefix, hdrs in prefix_to_headers.items()
        if len(hdrs) >= min_group_size
    }

    if not valid_prefixes:
        return []

    # Sort by prefix length descending — longer (more specific) prefixes first
    sorted_prefixes = sorted(valid_prefixes.keys(), key=len, reverse=True)

    claimed: set[str] = set()
    groups: list[CandidateClass] = []

    for prefix in sorted_prefixes:
        # Only consider unclaimed headers
        available = [h for h in valid_prefixes[prefix] if h not in claimed]
        if len(available) < min_group_size:
            continue

        # Compute suffixes: the part after the prefix + separator
        prefix_with_sep = prefix + separator
        suffixes = []
        for h in available:
            if h.startswith(prefix_with_sep):
                suffixes.append(h[len(prefix_with_sep):])
            elif h == prefix:
                suffixes.append("")

        claimed.update(available)

        groups.append(
            CandidateClass(
                prefix=prefix,
                headers=list(available),
                suffixes=suffixes,
                level=1,
            )
        )

    return groups


def detect_sibling_patterns(
    groups: list[CandidateClass],
    min_suffix_overlap: float = 0.8,
) -> list[SiblingGroup]:
    """Find groups sharing the same suffix pattern (Jaccard >= threshold).

    Groups with identical or near-identical suffix sets are siblings
    under a common parent.
    """
    if len(groups) < 2:
        return []

    siblings: list[SiblingGroup] = []
    used: set[int] = set()

    for i, group_a in enumerate(groups):
        if i in used:
            continue
        suffix_set_a = set(group_a.suffixes)
        if not suffix_set_a:
            continue

        cluster = [group_a]
        cluster_indices = {i}

        for j, group_b in enumerate(groups):
            if j <= i or j in used:
                continue
            suffix_set_b = set(group_b.suffixes)
            if not suffix_set_b:
                continue

            # Jaccard similarity
            intersection = suffix_set_a & suffix_set_b
            union = suffix_set_a | suffix_set_b
            jaccard = len(intersection) / len(union) if union else 0.0

            if jaccard >= min_suffix_overlap:
                cluster.append(group_b)
                cluster_indices.add(j)

        if len(cluster) >= 2:
            used.update(cluster_indices)

            # Find common prefix among the group prefixes
            prefixes = [c.prefix for c in cluster]
            common = _longest_common_prefix(prefixes)
            # Trim trailing separator
            if common.endswith("_"):
                common = common[:-1]

            # Shared suffix pattern is the intersection of all suffix sets
            shared_suffixes = set(cluster[0].suffixes)
            for c in cluster[1:]:
                shared_suffixes &= set(c.suffixes)

            siblings.append(
                SiblingGroup(
                    common_prefix=common,
                    suffix_pattern=sorted(shared_suffixes),
                    children=list(cluster),
                )
            )

    return siblings


def get_ungrouped_headers(
    all_headers: list[str],
    groups: list[CandidateClass],
) -> list[str]:
    """Return headers not claimed by any prefix group."""
    claimed: set[str] = set()
    for group in groups:
        claimed.update(group.headers)
    return [h for h in all_headers if h not in claimed]


# ---------------------------------------------------------------------------
# Iteration 2 — Semantic grouping (LLM)
# ---------------------------------------------------------------------------


def build_semantic_grouping_prompt(headers: list[str]) -> str:
    """Build prompt asking LLM to group headers by concept.

    Request JSON output: {"groups": [{"name": "...", "headers": [...]}]}
    """
    header_list = "\n".join(f"- {h}" for h in headers)

    return (
        "You are an insurance data-model expert.\n"
        "Group the following column headers by semantic concept.\n"
        "Each group should represent a single ontological class or entity type.\n\n"
        "Return ONLY a JSON object in this exact format:\n"
        '{"groups": [{"name": "ConceptName", "headers": ["HEADER1", "HEADER2"]}]}\n\n'
        "Rules:\n"
        "- Every header must appear in exactly one group\n"
        "- Use concise PascalCase names for groups\n"
        "- Aim for 3-10 groups\n"
        "- Group by domain concept, not by data type\n\n"
        f"Headers:\n{header_list}"
    )


def semantic_group_headers(
    headers: list[str],
    model: str | None = None,
) -> list[CandidateClass]:
    """Call LLM to semantically group headers into candidate classes."""
    if not headers:
        return []

    prompt = build_semantic_grouping_prompt(headers)
    result = llm_call_json(prompt, model=model)

    groups: list[CandidateClass] = []

    raw_groups = result.get("groups", []) if isinstance(result, dict) else []
    for g in raw_groups:
        if not isinstance(g, dict):
            continue
        group_name = g.get("name", "")
        group_headers = g.get("headers", [])
        if not isinstance(group_headers, list):
            continue

        groups.append(
            CandidateClass(
                prefix="",
                headers=[str(h) for h in group_headers],
                name=group_name,
                level=1,
            )
        )

    return groups


# ---------------------------------------------------------------------------
# Iteration 3 — Cross-file merging (algorithmic)
# ---------------------------------------------------------------------------


def merge_cross_file_classes(
    classes: list[CandidateClass],
    overlap_threshold: float = 0.3,
) -> list[CandidateClass]:
    """Merge classes from different files that represent the same concept.

    Two classes merge if:
    - They share the same prefix (non-empty), OR
    - Their header sets have Jaccard > threshold

    Shared headers become the parent; unique headers define each child.
    """
    if len(classes) < 2:
        return list(classes)

    used: set[int] = set()
    merged: list[CandidateClass] = []

    for i, cls_a in enumerate(classes):
        if i in used:
            continue

        merge_group = [cls_a]
        merge_indices = {i}

        set_a = set(cls_a.headers)

        for j, cls_b in enumerate(classes):
            if j <= i or j in used:
                continue

            set_b = set(cls_b.headers)

            # Check prefix match (both non-empty)
            prefix_match = (
                cls_a.prefix
                and cls_b.prefix
                and cls_a.prefix == cls_b.prefix
            )

            # Check Jaccard overlap
            intersection = set_a & set_b
            union = set_a | set_b
            jaccard = len(intersection) / len(union) if union else 0.0

            if prefix_match or jaccard > overlap_threshold:
                merge_group.append(cls_b)
                merge_indices.add(j)

        if len(merge_group) == 1:
            # No merge needed — keep as-is
            merged.append(cls_a)
        else:
            used.update(merge_indices)

            # Compute shared and unique headers
            header_sets = [set(c.headers) for c in merge_group]
            shared = header_sets[0]
            for hs in header_sets[1:]:
                shared = shared & hs
            shared_list = sorted(shared)

            # Collect all source files
            all_sources: list[str] = []
            for c in merge_group:
                if c.source_file and c.source_file not in all_sources:
                    all_sources.append(c.source_file)
                for sf in c.source_files:
                    if sf not in all_sources:
                        all_sources.append(sf)

            # Build parent
            parent = CandidateClass(
                prefix=merge_group[0].prefix,
                headers=sorted(set().union(*header_sets)),
                shared_headers=shared_list,
                source_files=all_sources,
                level=1,
            )

            # Build children with unique headers
            children: list[CandidateClass] = []
            for c in merge_group:
                unique = sorted(set(c.headers) - shared)
                if unique or c.source_file:
                    child = CandidateClass(
                        prefix=c.prefix,
                        headers=list(c.headers),
                        unique_headers=unique,
                        shared_headers=shared_list,
                        source_file=c.source_file,
                        source_files=[c.source_file] if c.source_file else list(c.source_files),
                        parent=parent,
                        level=2,
                    )
                    children.append(child)

            parent.children = children
            merged.append(parent)

    return merged


# ---------------------------------------------------------------------------
# Iteration 4 — Naming (LLM)
# ---------------------------------------------------------------------------


def build_naming_prompt(parent_class: CandidateClass) -> str:
    """Build prompt showing shared + unique headers for LLM naming.

    Returns a prompt that asks the LLM to name a parent class and its children.
    """
    lines = [
        "You are an insurance ontology expert.",
        "Name the following class and its subclasses based on the evidence below.",
        "",
        "Shared headers (define the parent concept):",
    ]
    for h in parent_class.shared_headers:
        lines.append(f"  - {h}")

    lines.append("")
    lines.append("Subclasses (each has unique headers):")

    for i, child in enumerate(parent_class.children):
        source_info = ""
        if child.source_file:
            source_info = f" (from: {child.source_file})"
        elif child.source_files:
            source_info = f" (from: {', '.join(child.source_files)})"

        lines.append(f"  Child {i + 1}{source_info}:")
        for h in child.unique_headers:
            lines.append(f"    - {h}")

    lines.extend([
        "",
        "Return ONLY a JSON object in this exact format:",
        '{"parent_name": "ParentClassName",',
        ' "children": [{"name": "ChildName", "is_subclass": true}]}',
        "",
        "Rules:",
        "- Use concise PascalCase names from insurance domain vocabulary",
        "- Parent name should capture the shared concept",
        "- Child names should capture what makes each variant unique",
        "- is_subclass should be true for all children",
    ])

    return "\n".join(lines)


def _build_standalone_naming_prompt(cls: CandidateClass) -> str:
    """Build a simpler naming prompt for classes without children."""
    header_list = "\n".join(f"  - {h}" for h in cls.headers)
    source_info = ""
    if cls.source_file:
        source_info = f"\nSource file: {cls.source_file}"
    elif cls.source_files:
        source_info = f"\nSource files: {', '.join(cls.source_files)}"

    return (
        "You are an insurance ontology expert.\n"
        "Name the following class based on its headers.\n\n"
        f"Headers:\n{header_list}\n"
        f"{source_info}\n\n"
        "Return ONLY a JSON object: {\"name\": \"ClassName\"}\n"
        "Use concise PascalCase from insurance domain vocabulary."
    )


def name_classes(
    classes: list[CandidateClass],
    model: str | None = None,
) -> None:
    """Name all classes in-place via LLM.

    For classes with children, uses build_naming_prompt.
    For standalone classes, uses a simpler prompt.
    """
    for cls in classes:
        if cls.children:
            prompt = build_naming_prompt(cls)
            result = llm_call_json(prompt, model=model)

            if isinstance(result, dict):
                cls.name = result.get("parent_name", cls.prefix)

                child_names = result.get("children", [])
                for i, child in enumerate(cls.children):
                    if i < len(child_names) and isinstance(child_names[i], dict):
                        child.name = child_names[i].get("name", child.prefix)
                    else:
                        child.name = child.prefix
        else:
            prompt = _build_standalone_naming_prompt(cls)
            result = llm_call_json(prompt, model=model)

            if isinstance(result, dict):
                cls.name = result.get("name", cls.prefix)
