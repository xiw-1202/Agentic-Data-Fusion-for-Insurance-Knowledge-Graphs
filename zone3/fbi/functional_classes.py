"""Phase 2 — Tasks 3 & 4: Function-first class hierarchy with sibling sub-classes.

Convert ``FunctionGroup`` objects into ``CandidateClass`` hierarchies:

- Multi-file groups produce a parent class (with shared headers) and one child
  per source file (showing LOB-specific unique columns).
- Singleton groups produce a single class using that file's headers (or
  sections + defined terms for PDF/TXT files).

Within each resulting class, sibling-pattern sub-classes are detected and
attached. Non-sibling prefix groups are NOT promoted — they remain implicit
internal structure captured by the class's ``headers`` field.
"""

from __future__ import annotations

from zone3.fbi.class_discovery import (
    CandidateClass,
    SiblingGroup,
    detect_sibling_patterns,
    find_prefix_groups,
)
from zone3.fbi.function_grouping import FunctionGroup


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_multi_file_class(group: FunctionGroup) -> CandidateClass:
    """Build parent class with one child per source file.

    Parent ``headers`` = sorted shared headers across all files.
    Each child ``headers`` = that file's raw headers; ``unique_headers`` =
    headers minus the shared set.
    """
    shared_set = set(group.shared_headers)
    shared_sorted = sorted(shared_set)

    source_files = [fp.file_path for fp in group.files]

    parent = CandidateClass(
        prefix="",
        headers=list(shared_sorted),
        shared_headers=list(shared_sorted),
        source_files=list(source_files),
        level=1,
    )

    children: list[CandidateClass] = []
    for fp in group.files:
        file_headers = list(fp.headers_raw)
        unique = sorted(set(file_headers) - shared_set)
        child = CandidateClass(
            prefix="",
            headers=file_headers,
            unique_headers=unique,
            shared_headers=list(shared_sorted),
            source_file=fp.file_path,
            source_files=[fp.file_path],
            parent=parent,
            level=2,
        )
        children.append(child)

    parent.children = children
    return parent


def _build_singleton_class(group: FunctionGroup) -> CandidateClass:
    """Build a single class for a singleton ``FunctionGroup``.

    CSV: ``headers`` = file's ``headers_raw``.
    PDF/TXT: ``headers`` = ``sections + defined_terms``.
    """
    fp = group.files[0]

    if fp.file_type == "csv":
        headers = list(fp.headers_raw)
    else:
        headers = list(fp.sections) + list(fp.defined_terms)

    return CandidateClass(
        prefix="",
        headers=headers,
        source_file=fp.file_path,
        source_files=[fp.file_path],
        level=1,
    )


# ---------------------------------------------------------------------------
# Sibling sub-class attachment
# ---------------------------------------------------------------------------


def _attach_sibling_subclasses(cls: CandidateClass) -> None:
    """Detect sibling patterns within ``cls.headers`` and attach as sub-classes.

    For each detected ``SiblingGroup`` a parent sub-class is created at
    ``cls.level + 1`` and the matching sibling children are re-parented at
    ``cls.level + 2``. Non-sibling prefix groups are ignored — their headers
    are already reflected in ``cls.headers``.
    """
    if not cls.headers:
        return

    prefix_groups = find_prefix_groups(cls.headers)
    if len(prefix_groups) < 2:
        return

    sibling_groups: list[SiblingGroup] = detect_sibling_patterns(prefix_groups)
    if not sibling_groups:
        return

    for sg in sibling_groups:
        sibling_parent = CandidateClass(
            prefix=sg.common_prefix,
            headers=[],
            suffixes=list(sg.suffix_pattern),
            parent=cls,
            level=cls.level + 1,
        )

        sibling_children: list[CandidateClass] = []
        parent_headers: list[str] = []
        for child_group in sg.children:
            sibling_child = CandidateClass(
                prefix=child_group.prefix,
                headers=list(child_group.headers),
                suffixes=list(child_group.suffixes),
                parent=sibling_parent,
                level=cls.level + 2,
            )
            sibling_children.append(sibling_child)
            parent_headers.extend(child_group.headers)

        sibling_parent.children = sibling_children
        sibling_parent.headers = parent_headers

        cls.children.append(sibling_parent)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_functional_classes(
    groups: list[FunctionGroup],
) -> list[CandidateClass]:
    """Convert ``FunctionGroup`` objects into a function-first class hierarchy.

    For each group:
    - Multi-file → parent + children (one per file).
    - Singleton → single class.

    Then for every resulting class (parent, child, or singleton) run
    sibling-pattern detection on its ``headers`` and promote only sibling
    groups to sub-classes.
    """
    if not groups:
        return []

    top_classes: list[CandidateClass] = []

    for group in groups:
        if group.is_singleton:
            cls = _build_singleton_class(group)
            _attach_sibling_subclasses(cls)
            top_classes.append(cls)
        else:
            parent = _build_multi_file_class(group)
            _attach_sibling_subclasses(parent)
            for child in list(parent.children):
                # Only attach sibling sub-classes to file-variant children,
                # not to sibling sub-classes we just added.
                if child.level == 2 and child.source_file:
                    _attach_sibling_subclasses(child)
            top_classes.append(parent)

    return top_classes
