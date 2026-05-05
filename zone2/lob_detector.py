"""LOB (Line of Business) detection for Zone 2 chunks.

Tags every chunk with one of Assurant's product LOBs so downstream
extraction can:
  • emit a class hierarchy ``<Lob>PolicyRecord IS_A PolicyRecord``
  • inject domain-specific context into LLM prompts
  • carry the LOB onto every triple/edge for Zone 3 induction

Detection signals (priority order):
  1. Filename keywords          — fastest, ~95% accurate for CSV
  2. Structured field signature — parse first ``RECORD <idx>:`` block
  3. Section hierarchy + first 500 chars of content (PDF/TXT fallback)
  4. Default: ``"generic"``
"""

from __future__ import annotations

import re

# Assurant LOB enum.  Order is documentation-only; lookups use the set.
LOBS: tuple[str, ...] = (
    # Global Lifestyle
    "device",          # mobile/electronics protection
    "appliance",       # consumer appliance extended service contracts
    "auto",            # vehicle protection / extended warranty
    "credit",          # credit & financial services insurance
    # Global Housing
    "flood",           # NFIP + voluntary flood
    "home",            # voluntary homeowners
    "renters",         # renters
    "condo",           # condominium
    "manufactured",    # manufactured housing
    "lender_placed",   # force-placed homeowners
    # Fallback
    "generic",
)


# ---------------------------------------------------------------------------
# Filename-driven rules
# ---------------------------------------------------------------------------

# (lob, ordered keyword tuple).  Order matters — more specific compounds first
# so that "manufactured_housing" wins over the substring "mobile" (device),
# and "lender_placed" wins over "home"/"renters" tokens that may appear nearby.
_FILENAME_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("manufactured",  ("manufactured_housing", "manufactured", "mobile_home")),
    ("lender_placed", ("lender_placed", "force_placed", "lpi", "forced_placed")),
    ("flood",         ("flood", "nfip", "fema", "sfip")),
    ("renters",       ("renter",)),
    ("condo",         ("condominium", "condo")),
    ("home",          ("homeowner", "dwelling", "home_insurance", "home_policy")),
    ("auto",          ("vehicle", "auto_insurance", "auto_polic", "auto_claim",
                       "automotive", "motorvehicle", "_mvr_", "auto_warranty")),
    ("device",        ("tmobile", "t_mobile", "mobile_device", "phone_insurance",
                       "imei", "att_phone", "verizon_phone", "device_protection")),
    ("appliance",     ("appliance", "electronics_warranty", "warranty")),
    ("credit",        ("credit_insurance", "credit_card_protection", "credit_protect",
                       "credit_card")),
]


def _filename_lob(source: str) -> str | None:
    """Match the *basename* of ``source`` against priority-ordered keywords.

    Path components are intentionally ignored — directory names like
    ``Emory_Spring2026`` would otherwise produce false positives.
    """
    if not source:
        return None
    base = source.lower().replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = re.sub(r"\.(csv|json|pdf|txt)$", "", base)
    cleaned = cleaned.replace("synthetic_data_sample_", "")
    # Pad with underscores so word-boundary patterns like "_mvr_" can match.
    padded = f"_{cleaned}_"

    for lob, kws in _FILENAME_RULES:
        for kw in kws:
            if kw in padded or kw in cleaned:
                return lob
    return None


# ---------------------------------------------------------------------------
# Field signature for structured chunks
# ---------------------------------------------------------------------------

# Distinctive field tokens (lowercased, non-alphanum stripped) per LOB.
# Renters and home are intentionally omitted — they share too many fields
# with each other and with auto for unambiguous structural detection;
# filename is the better signal there.
_FIELD_SIGNATURES: dict[str, frozenset[str]] = {
    "flood": frozenset({
        "ratedfloodzone", "floodzonecurrent", "iccpremium", "iccpolicy",
        "floodevent", "waterdepth", "nfipcommunityname", "nfipratedfloodzone",
    }),
    "auto": frozenset({
        "vin", "mvr", "vehicleyear", "vehiclemodel", "vehiclemake",
        "mileage", "collisionded", "comprehensiveded",
    }),
    "device": frozenset({
        "imei", "devicemodel", "deviceserial", "deviceiccid", "tac",
        "phonenumber", "msisdn", "devicemake",
    }),
    "manufactured": frozenset({
        "hudcode", "vinmh", "manufacturedhomemodel",
    }),
}


_RECORD_BLOCK = re.compile(r"RECORD(?:\s+\d+)?:\s*\n([\s\S]+?)(?=\n\nRECORD|\n\n[A-Z]|$)")
_GROUP_LINE = re.compile(r"^\s*\[[^\]]+\]\s*(.+)$", re.MULTILINE)


def _extract_field_tokens(content: str) -> set[str]:
    """Collect normalized field-name tokens from the first RECORD block."""
    m = _RECORD_BLOCK.search(content)
    if not m:
        return set()

    block = m.group(1)
    tokens: set[str] = set()
    for line_match in _GROUP_LINE.finditer(block):
        pairs_text = line_match.group(1)
        for pair in pairs_text.split("|"):
            if ":" not in pair:
                continue
            field, _, _ = pair.partition(":")
            normalized = re.sub(r"[^a-z0-9]", "", field.strip().lower())
            if normalized:
                tokens.add(normalized)
    return tokens


def _structured_field_lob(content: str) -> str | None:
    """Score each LOB by its field-signature overlap with the chunk."""
    tokens = _extract_field_tokens(content)
    if not tokens:
        return None

    best_lob: str | None = None
    best_score = 0
    for lob, sigs in _FIELD_SIGNATURES.items():
        score = sum(1 for s in sigs if s in tokens)
        if score > best_score:
            best_lob = lob
            best_score = score
    return best_lob


# ---------------------------------------------------------------------------
# Content fallback for prose / PDF
# ---------------------------------------------------------------------------

# Multi-keyword vote per LOB — at least 2 hits required so a single passing
# mention doesn't reroute the chunk.
_CONTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "flood":        ("flood", "nfip", "sfip", "flood zone", "floodplain"),
    "auto":         ("vehicle", "automobile", "collision coverage",
                     "comprehensive coverage", "motor vehicle"),
    "device":       ("mobile device", "smartphone", "imei", "phone insurance",
                     "device protection"),
    "renters":      ("renters insurance", "tenant", "personal property of the renter"),
    "home":         ("homeowners insurance", "dwelling coverage", "owner-occupied"),
    "manufactured": ("manufactured home", "mobile home park", "hud code"),
    "lender_placed":("lender-placed", "force-placed", "force placed"),
    "credit":       ("credit insurance", "credit life", "credit disability"),
    "appliance":    ("extended warranty", "service contract", "appliance protection"),
    "condo":        ("condominium", "condo association", "hoa master policy"),
}

_CONTENT_VOTE_THRESHOLD = 2


def _content_lob(content: str, hierarchy: list[str]) -> str | None:
    """Vote across content keywords; require ≥ threshold hits to commit."""
    text = " ".join(hierarchy).lower() + " " + content[:500].lower()
    if not text.strip():
        return None

    votes: dict[str, int] = {}
    for lob, kws in _CONTENT_KEYWORDS.items():
        n = sum(1 for kw in kws if kw in text)
        if n:
            votes[lob] = n

    if not votes:
        return None
    top = max(votes, key=votes.get)
    if votes[top] >= _CONTENT_VOTE_THRESHOLD:
        return top
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_lob(chunk: dict) -> str:
    """Identify the Line of Business for a single Zone 1 chunk.

    Returns one of :data:`LOBS`.  Defaults to ``"generic"`` when no
    signal is found.
    """
    source = chunk.get("source", "") or ""
    content = chunk.get("content", "") or ""
    hierarchy = chunk.get("section_hierarchy", []) or []

    lob = _filename_lob(source)
    if lob:
        return lob

    lob = _structured_field_lob(content)
    if lob:
        return lob

    lob = _content_lob(content, hierarchy)
    if lob:
        return lob

    return "generic"


def detect_lobs(state: dict) -> dict:
    """LangGraph node: stamp ``chunk["lob"]`` on every chunk in state.

    Mutates each chunk dict in place and returns the chunks list under
    the same key so LangGraph merges it back into state.
    """
    chunks = state.get("chunks", [])
    counts: dict[str, int] = {}
    for c in chunks:
        lob = detect_lob(c)
        c["lob"] = lob
        counts[lob] = counts.get(lob, 0) + 1

    if chunks:
        print(f"\n[1.25/4] LOB detection — {len(chunks)} chunks tagged")
        for lob in sorted(counts):
            print(f"  {lob:15s} {counts[lob]:>4} chunks")

    return {"chunks": chunks}
