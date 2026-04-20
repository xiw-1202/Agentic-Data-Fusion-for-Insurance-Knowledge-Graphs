"""Token classifier — separate LOB tokens from function tokens via co-occurrence.

Filename tokens in raw form mix two concerns:

1. **LOB tokens** — identifiers of a line-of-business/client, e.g. ``geico``,
   ``tmobile``, ``renters``. These co-occur consistently within the same
   corpus slice.
2. **Function tokens** — describe the file's purpose (claim, survey, policy)
   and appear across multiple LOBs.
3. **Modifier tokens** — qualify a function within a single file
   (e.g. ``cancel`` in ``geicorenterscancelsurvey``).

This module classifies tokens algorithmically (no LLM) using two passes of
union-find clustering over files by shared-token counts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from zone3.fbi.fingerprint import FileFingerprint


@dataclass
class TokenClassification:
    """Classification of filename tokens into LOB / function / modifier."""

    lob_tokens: set[str] = field(default_factory=set)
    function_tokens: set[str] = field(default_factory=set)
    modifier_tokens: set[str] = field(default_factory=set)
    lob_groups: list[set[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_token_index(
    fingerprints: list[FileFingerprint],
) -> dict[str, set[str]]:
    """Return a mapping ``token -> set of filenames`` containing it."""
    index: dict[str, set[str]] = defaultdict(set)
    for fp in fingerprints:
        for tok in fp.filename_tokens:
            index[tok].add(fp.basename)
    return dict(index)


class _UnionFind:
    """Minimal union-find over a fixed set of string keys."""

    def __init__(self, keys: list[str]) -> None:
        self.parent: dict[str, str] = {k: k for k in keys}

    def find(self, k: str) -> str:
        root = k
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while self.parent[k] != root:
            self.parent[k], k = root, self.parent[k]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> list[set[str]]:
        clusters: dict[str, set[str]] = defaultdict(set)
        for k in self.parent:
            clusters[self.find(k)].add(k)
        return list(clusters.values())


def _cluster_files_by_shared_tokens(
    fingerprints: list[FileFingerprint],
    min_shared: int = 2,
) -> list[set[str]]:
    """Cluster filenames by shared token count (strict, single pass).

    Two files belong to the same cluster if they share at least
    ``min_shared`` filename tokens. Returns a list of filename sets
    (includes singletons for files not sharing enough tokens with anyone).
    """
    names = [fp.basename for fp in fingerprints]
    tokens_by_name = {fp.basename: set(fp.filename_tokens) for fp in fingerprints}

    uf = _UnionFind(names)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if len(tokens_by_name[a] & tokens_by_name[b]) >= min_shared:
                uf.union(a, b)

    return uf.groups()


def _lob_defining_tokens(
    group: set[str],
    tokens_by_name: dict[str, set[str]],
    min_files: int = 2,
) -> set[str]:
    """Tokens that appear in >= ``min_files`` files within *group*.

    These are the tokens that define the LOB identity of the group.
    Returns an empty set for singleton groups.
    """
    if len(group) < min_files:
        return set()

    counts: dict[str, int] = defaultdict(int)
    for name in group:
        for tok in tokens_by_name.get(name, ()):
            counts[tok] += 1

    return {tok for tok, c in counts.items() if c >= min_files}


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


def classify_tokens(
    fingerprints: list[FileFingerprint],
) -> TokenClassification:
    """Classify filename tokens into LOB / function / modifier categories.

    Algorithm:

    1. Build a ``token -> files`` index.
    2. **Pass 1 clustering**: union-find files sharing >= 2 tokens.
    3. **Pass 2 rescue** (only if Pass 1 produced >= 1 multi-file cluster):
       a. *Singleton into existing cluster*: merge singleton ``f`` into cluster
          ``C`` if they share a token ``t`` that already appears in >= 2 files
          of ``C`` (i.e. ``t`` is LOB-defining for ``C``).
       b. *Two singletons into a new cluster*: merge two singletons sharing
          a token ``t`` where every file containing ``t`` is a singleton.
    4. For each cluster, compute LOB-defining tokens (appearing in >= 2 files).
    5. Classify each token by (#distinct clusters it appears in, #files):
       - in >= 2 clusters          -> function
       - in 1 cluster, >= 2 files  -> LOB
       - in 1 cluster, 1 file      -> modifier OR function (see below)
    6. *Function signature rule*: a single-file token is promoted to
       ``function`` if it is the ONLY non-LOB-defining token in its file
       (it is the file's functional signature).
    7. *Fallback*: if Pass 1 produced no multi-file clusters, tokens appearing
       in >= 2 files are function tokens.
    """
    if not fingerprints:
        return TokenClassification()

    tokens_by_name: dict[str, set[str]] = {
        fp.basename: set(fp.filename_tokens) for fp in fingerprints
    }
    token_index = _build_token_index(fingerprints)

    # ------------------------------------------------------------------
    # Pass 1: strict clustering (>= 2 shared tokens)
    # ------------------------------------------------------------------
    primary_groups = _cluster_files_by_shared_tokens(fingerprints, min_shared=2)

    # Did Pass 1 actually produce any multi-file cluster?
    has_multi_cluster = any(len(g) >= 2 for g in primary_groups)

    # Work on a mutable name -> group-id map
    name_to_gid: dict[str, int] = {}
    groups: list[set[str]] = []
    for gid, g in enumerate(primary_groups):
        groups.append(set(g))
        for name in g:
            name_to_gid[name] = gid

    # ------------------------------------------------------------------
    # Pass 2 (rescue) — only if Pass 1 produced a real cluster
    # ------------------------------------------------------------------
    if has_multi_cluster:
        # 2a. Singleton merges into existing multi-file cluster via
        #     a token that's LOB-defining for that cluster.
        changed = True
        while changed:
            changed = False
            # Recompute LOB-defining per cluster (may grow as we merge).
            lob_def_per_gid: dict[int, set[str]] = {
                gid: _lob_defining_tokens(groups[gid], tokens_by_name)
                for gid in range(len(groups))
            }

            for fp in fingerprints:
                name = fp.basename
                gid = name_to_gid[name]
                if len(groups[gid]) != 1:
                    continue  # only rescue singletons

                # Find a target cluster sharing an LOB-defining token with fp.
                fp_tokens = tokens_by_name[name]
                for target_gid, lob_def in lob_def_per_gid.items():
                    if target_gid == gid:
                        continue
                    if len(groups[target_gid]) < 2:
                        continue
                    if fp_tokens & lob_def:
                        # Merge singleton into target
                        groups[target_gid].add(name)
                        groups[gid].clear()
                        name_to_gid[name] = target_gid
                        changed = True
                        break

        # 2b. Two (or more) singletons merging into a new cluster via a
        #     token that appears ONLY in singleton files.
        # Recompute which files are currently singletons.
        singleton_names = [
            fp.basename for fp in fingerprints if len(groups[name_to_gid[fp.basename]]) == 1
        ]
        singleton_set = set(singleton_names)

        # Union-find over singletons using tokens that live only in singletons.
        if singleton_names:
            uf_s = _UnionFind(singleton_names)
            for tok, files in token_index.items():
                if len(files) < 2:
                    continue
                if not files.issubset(singleton_set):
                    continue
                files_list = sorted(files)
                for other in files_list[1:]:
                    uf_s.union(files_list[0], other)

            # Apply merges: for each non-trivial group, merge singletons
            # into a fresh cluster.
            for sub in uf_s.groups():
                if len(sub) < 2:
                    continue
                new_gid = len(groups)
                groups.append(set())
                for name in sub:
                    old_gid = name_to_gid[name]
                    groups[old_gid].discard(name)
                    groups[new_gid].add(name)
                    name_to_gid[name] = new_gid

    # Drop empty groups and build final list
    lob_groups = [g for g in groups if g]

    # ------------------------------------------------------------------
    # Step 4 — recompute LOB-defining tokens per final cluster
    # ------------------------------------------------------------------
    lob_def_per_group: list[set[str]] = [
        _lob_defining_tokens(g, tokens_by_name) for g in lob_groups
    ]

    # Map name -> index into lob_groups
    name_to_group_idx: dict[str, int] = {}
    for idx, g in enumerate(lob_groups):
        for name in g:
            name_to_group_idx[name] = idx

    # Count groups per token
    groups_per_token: dict[str, set[int]] = defaultdict(set)
    for tok, files in token_index.items():
        for name in files:
            if name in name_to_group_idx:
                groups_per_token[tok].add(name_to_group_idx[name])

    # ------------------------------------------------------------------
    # Step 5-7 — classify
    # ------------------------------------------------------------------
    lob_tokens: set[str] = set()
    function_tokens: set[str] = set()
    modifier_tokens: set[str] = set()

    all_tokens = set(token_index.keys())

    if not has_multi_cluster:
        # Fallback mode: no real LOB structure exists.
        for tok in all_tokens:
            if len(token_index[tok]) >= 2:
                function_tokens.add(tok)
            else:
                modifier_tokens.add(tok)
        return TokenClassification(
            lob_tokens=lob_tokens,
            function_tokens=function_tokens,
            modifier_tokens=modifier_tokens,
            lob_groups=lob_groups,
        )

    for tok in all_tokens:
        files = token_index[tok]
        group_ids = groups_per_token.get(tok, set())

        if len(group_ids) >= 2:
            function_tokens.add(tok)
        elif len(group_ids) == 1 and len(files) >= 2:
            lob_tokens.add(tok)
        else:
            # 1 group, 1 file — check "only non-LOB-defining token" rule.
            name = next(iter(files))
            g_idx = name_to_group_idx.get(name)
            if g_idx is None:
                modifier_tokens.add(tok)
                continue
            lob_def = lob_def_per_group[g_idx]
            non_lob_def_tokens = tokens_by_name[name] - lob_def
            if len(non_lob_def_tokens) == 1 and tok in non_lob_def_tokens:
                # This is the file's functional signature.
                function_tokens.add(tok)
            else:
                modifier_tokens.add(tok)

    return TokenClassification(
        lob_tokens=lob_tokens,
        function_tokens=function_tokens,
        modifier_tokens=modifier_tokens,
        lob_groups=lob_groups,
    )
