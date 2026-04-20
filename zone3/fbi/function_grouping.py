"""Function-based file grouping across LOBs.

Groups data files by their **functional role** (claim, policy, survey, ...)
rather than by their line-of-business (GEICO, TMobile, ...).

Input: ``FileFingerprint`` list + ``TokenClassification`` (from Phase 1/2).

Algorithm (per ``group_files_by_function``):

1. Only consider pairs of files from **different** LOB groups — files from the
   same LOB serve different functions and must not be merged.
2. Score each cross-LOB pair with three weighted signals:
   - +0.5 for any shared (normalized) function token
   - +0.3 for any shared dominant header prefix
   - +0.2 for header Jaccard > 0.1
3. Merge via union-find when score > threshold (default 0.5). Transitive
   closure: if A~B and B~C, then A/B/C share one group.
4. Files that do not merge remain singletons (still valid groups).
5. Each resulting group exposes its intersected function tokens, intersected
   dominant prefixes, and intersected headers for downstream class discovery.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field

from zone3.fbi.fingerprint import FileFingerprint
from zone3.fbi.token_classifier import TokenClassification
from zone3.fbi.token_utils import normalize_token as _normalize_token


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------


@dataclass
class FunctionGroup:
    """A group of files serving the same functional role across LOBs."""

    function_tokens: set[str] = field(default_factory=set)
    files: list[FileFingerprint] = field(default_factory=list)
    dominant_prefixes: set[str] = field(default_factory=set)
    shared_headers: set[str] = field(default_factory=set)

    @property
    def is_singleton(self) -> bool:
        return len(self.files) == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_dominant_prefixes(headers: list[str], min_size: int = 3) -> set[str]:
    """Return first-level prefixes that cover at least ``min_size`` headers.

    Only the portion before the first ``_`` is considered. Headers without
    an underscore contribute nothing.
    """
    counts: dict[str, int] = defaultdict(int)
    for h in headers:
        if "_" not in h:
            continue
        prefix = h.split("_", 1)[0]
        if not prefix:
            continue
        counts[prefix] += 1
    return {p for p, c in counts.items() if c >= min_size}


def _file_lob_group_id(fp: FileFingerprint, lob_groups: list[set[str]]) -> int:
    """Return the index of the LOB group containing ``fp``'s basename.

    Returns ``-1`` when the file is not present in any LOB group.
    """
    name = os.path.basename(fp.file_path)
    for idx, group in enumerate(lob_groups):
        if name in group:
            return idx
    return -1


def _header_jaccard(h1: list[str], h2: list[str]) -> float:
    s1, s2 = set(h1), set(h2)
    if not s1 and not s2:
        return 0.0
    union = s1 | s2
    if not union:
        return 0.0
    return len(s1 & s2) / len(union)


def _file_function_signature(
    fp: FileFingerprint, function_tokens_all: set[str]
) -> set[str]:
    """Return the (normalized) function tokens present in ``fp``'s filename tokens.

    When ``function_tokens_all`` is empty (i.e. no LOB structure was detected
    during token classification), fall back to treating every filename token
    as a candidate function signature — since there is no LOB/function split
    to enforce, matching on raw normalized tokens is the best signal we have.
    """
    if not function_tokens_all:
        return {_normalize_token(t) for t in fp.filename_tokens}

    norm_functions = {_normalize_token(t) for t in function_tokens_all}
    return {
        _normalize_token(t)
        for t in fp.filename_tokens
        if _normalize_token(t) in norm_functions
    }


def _compute_merge_score(
    fp1: FileFingerprint,
    fp2: FileFingerprint,
    function_tokens_all: set[str],
) -> float:
    """Weighted merge score for two files.

    Breakdown:
    - +0.5 shared function token (after normalization)
    - +0.3 shared dominant header prefix
    - +0.2 header Jaccard > 0.1
    """
    score = 0.0

    sig1 = _file_function_signature(fp1, function_tokens_all)
    sig2 = _file_function_signature(fp2, function_tokens_all)
    if sig1 & sig2:
        score += 0.5

    # Use a slightly looser threshold (min_size=2) inside the merge-score
    # check so small header lists (e.g. 2 CLAIM_* columns) still contribute.
    prefixes1 = _compute_dominant_prefixes(fp1.headers_raw, min_size=2)
    prefixes2 = _compute_dominant_prefixes(fp2.headers_raw, min_size=2)
    if prefixes1 & prefixes2:
        score += 0.3

    if _header_jaccard(fp1.headers_raw, fp2.headers_raw) > 0.1:
        score += 0.2

    return score


class _UnionFind:
    """Minimal union-find over integer indices."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return dict(out)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def group_files_by_function(
    fingerprints: list[FileFingerprint],
    token_classification: TokenClassification,
    score_threshold: float = 0.5,
) -> list[FunctionGroup]:
    """Group files by their functional role across LOB groups.

    Parameters
    ----------
    fingerprints:
        All fingerprints in the corpus.
    token_classification:
        Result of ``classify_tokens`` — used to restrict signature matching
        to function tokens only and to gate same-LOB merges.
    score_threshold:
        Strict threshold — pairs merge when score **> threshold** (default 0.5).
    """
    if not fingerprints:
        return []

    function_tokens_all = set(token_classification.function_tokens)
    lob_groups = token_classification.lob_groups

    n = len(fingerprints)
    lob_ids = [_file_lob_group_id(fp, lob_groups) for fp in fingerprints]
    uf = _UnionFind(n)

    # Score only cross-LOB pairs; files with lob_id == -1 (not in any LOB group)
    # can merge with anyone — they are effectively "standalone" LOBs.
    for i in range(n):
        for j in range(i + 1, n):
            lob_i, lob_j = lob_ids[i], lob_ids[j]
            # Same, non-negative LOB group → don't merge (different functions
            # within the same LOB represent distinct classes).
            if lob_i >= 0 and lob_j >= 0 and lob_i == lob_j:
                continue
            score = _compute_merge_score(
                fingerprints[i], fingerprints[j], function_tokens_all
            )
            # Use >= so that a clear function-token match (worth 0.5 alone)
            # is sufficient at the default 0.5 threshold.
            if score >= score_threshold:
                uf.union(i, j)

    # Build groups from union-find output
    groups: list[FunctionGroup] = []
    for indices in uf.groups().values():
        files = [fingerprints[i] for i in indices]

        # Intersected function signatures across all files in the group.
        # For singletons we just use the file's own signature.
        signatures = [_file_function_signature(f, function_tokens_all) for f in files]
        if signatures:
            func_intersection = set.intersection(*signatures)
        else:
            func_intersection = set()

        prefix_sets = [
            _compute_dominant_prefixes(f.headers_raw, min_size=2) for f in files
        ]
        if prefix_sets:
            prefix_intersection = set.intersection(*prefix_sets)
        else:
            prefix_intersection = set()

        header_sets = [set(f.headers_raw) for f in files]
        if header_sets:
            header_intersection = set.intersection(*header_sets)
        else:
            header_intersection = set()

        groups.append(
            FunctionGroup(
                function_tokens=func_intersection,
                files=files,
                dominant_prefixes=prefix_intersection,
                shared_headers=header_intersection,
            )
        )

    return groups
