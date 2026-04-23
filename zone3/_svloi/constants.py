"""SV-LOI constants: class vocabulary, prefixes, normalization maps."""
from __future__ import annotations

from zone3.graph_cache import STRUCTURED_PREFIXES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 15          # entities per LLM typing prompt (smaller = more context per entity)
MAX_MEMBERS_IN_PROMPT = 15
MIN_CLASS_SIZE = 10      # fallback floor — actual threshold is max(this, 1% of entities)
DEVIATION_THRESHOLD = 2.0  # σ threshold for structural flagging
MAX_ARBITRATION_BATCH = 10  # entities per arbitration prompt
MAX_CLASS_FRACTION = 0.25  # no single class should exceed 25% of entities

# Structured entity prefixes — these already have entity_type from Stage 1.
# SV-LOI should use their existing types directly, not re-classify via LLM.
_STRUCTURED_PREFIXES = STRUCTURED_PREFIXES

# Target class count range (guide for class discovery)
# Higher minimum to counteract consolidation over-merging
TARGET_CLASSES_MIN = 8
TARGET_CLASSES_MAX = 15

# REMOVED: PROTECTED_CLASS_NAMES was a hardcoded list that overlapped with
# Riskine reference classes (domain leakage). Protection is now data-driven:
# merge_leaf_classes() uses relational diversity (distinct_rels < 4) to
# distinguish property-value classes from real ontology classes.
# Classes with rich relational structure survive merging naturally.
PROTECTED_CLASS_NAMES: set[str] = set()  # empty — fully data-driven

# Forbidden class names — data types masquerading as ontology classes
FORBIDDEN_CLASS_NAMES = {
    "financialamount", "timperiod", "timeperiod", "measurement", "amount",
    "number", "date", "text", "quantity", "value", "metric", "event",
    "condition", "location", "data", "record", "entry", "item", "type",
    "category", "group", "class", "entity", "thing", "other",
}

# Zone 2 entity_type → ontology class name normalization.
# Maps extraction-level labels to domain-role class names.
# None = drop (data type / attribute, not an ontology class).
# Types not in this map pass through as-is.
ZONE2_TYPE_NORMALIZATION: dict[str, str | None] = {
    "InsurancePolicy": "Policy",
    "CoverageType": "Coverage",
    "ExcludedPeril": "Exclusion",
    "InsuredProperty": "Property",
    "InsuredItem": "Property",
    "Claimant": "Person",
    "ServiceProvider": "Organization",
    "WarrantyProvider": "Organization",
    "RepairFacility": "Organization",
    "DeductibleAmount": None,
    "FinancialTransaction": None,
    "ClaimStatus": None,
    "PolicyCoverageLimit": None,
    "StateRegulation": None,
    "ServiceContractTerm": None,
}
