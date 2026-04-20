"""Shared token normalization utilities."""

from __future__ import annotations


def normalize_token(token: str) -> str:
    """Normalize a filename token for matching.

    - Lowercase
    - Strip trailing 's' for plural/singular matching (if length > 3)
    """
    t = token.lower()
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t
