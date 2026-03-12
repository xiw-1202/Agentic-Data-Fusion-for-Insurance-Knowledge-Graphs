"""
Zone 2 prompt constants — DOMAIN-AGNOSTIC version.

All prompts are generic to any insurance Line of Business (flood, auto, health,
liability). No SFIP text, no NFIP references, no Riskine class names.

The pipeline bootstraps domain-specific vocabulary from the documents themselves;
these prompts teach extraction PATTERNS, not domain FACTS.

Import via: from zone2.prompts import RELATION_BOOTSTRAP_PROMPT, ENTITY_BOOTSTRAP_PROMPT, SYSTEM_PROMPT_TEMPLATE, FEW_SHOT_PAIRS
"""

# ---------------------------------------------------------------------------
# Bootstrap prompts — LLM reads sample chunks and proposes schema
# ---------------------------------------------------------------------------

RELATION_BOOTSTRAP_PROMPT = """You are a knowledge graph expert.
Read these sample passages from an insurance domain document.
Propose 12-18 SNAKE_CASE relation type names that capture the relationships
described in these passages.

You MUST include at least 2 relation types from EACH of these 5 categories:
1. Coverage / exclusion  (e.g. COVERS, EXCLUDED_FROM, DOES_NOT_COVER)
2. Amounts / limits      (e.g. HAS_COVERAGE_LIMIT, HAS_DEDUCTIBLE, HAS_MAXIMUM)
3. Time periods          (e.g. HAS_WAITING_PERIOD, HAS_DEADLINE, EFFECTIVE_AFTER)
4. Definitions           (e.g. DEFINED_AS, IS_CLASSIFIED_AS)
5. Obligations / steps   (e.g. MUST_NOTIFY, MUST_FILE, PRECEDES, REQUIRES)

Also add any additional types useful for the sample passages below.
Do NOT include generic types: HAS, IS, CONTAINS, INCLUDES, RELATES_TO.

Sample passages:
{samples}

Respond with ONLY a JSON array of strings — no explanation:
["COVERS", "EXCLUDED_FROM", "HAS_COVERAGE_LIMIT", "HAS_WAITING_PERIOD", ...]"""

ENTITY_BOOTSTRAP_PROMPT = """You are a knowledge graph expert.
Read these sample passages from an insurance domain document.
Propose 10-20 PascalCase entity TYPE names that would be useful for organizing
the concepts in these passages into a knowledge graph ontology.

Think about: What kinds of things are mentioned? What categories do they fall
into? What would an ontology engineer name them?

Categories to consider:
- Types of coverage or insurance products
- Types of property, assets, or insured items
- Types of perils, risks, or causes of loss
- Types of people or organizations involved
- Types of rules, conditions, or procedures
- Types of financial amounts or limits

Sample passages:
{samples}

Respond with ONLY a JSON array of strings — no explanation:
["InsurancePolicy", "CoverageType", "ExcludedPeril", "InsuredProperty", ...]"""


# ---------------------------------------------------------------------------
# Extraction system prompt — domain-agnostic
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are a knowledge graph extractor for insurance policy documents.

Extract (subject, relation, object) triples from the passage provided.

Suggested relation types derived from this document (prefer these, but you MAY
create other SNAKE_CASE types if they capture something not covered):
{vocab_lines}

Do NOT use generic relations: HAS, IS, CONTAINS, INCLUDES, RELATES_TO.

Rules:
- Extract ALL important facts from the passage, not just the most prominent one
- Subject of a definition triple MUST be the exact term being defined
- Dollar amounts and time periods are valid objects: "$250,000", "30 days", "60 days"
- Percentages are valid objects: "10 percent", "80%"
- Negation ("not covered", "does not insure", "excluded") → use an exclusion relation
- Sequential steps → use PRECEDES (step A PRECEDES step B)
- MANDATORY LIST EXTRACTION: When a passage lists items "A, B, C are excluded/defined/required", extract EACH item as a separate triple — never collapse a list into one triple
- Extract 3-5 triples per passage. Only return 1-2 if the passage is very short.
- Include span: verbatim quote <=120 chars supporting the triple
- Include confidence: 0.0–1.0
- Return [] only if the passage contains no factual insurance content

Output format — JSON array only:
[{{"subject": "...", "relation": "...", "object": "...", "span": "...", "confidence": 0.9}}]"""


# ---------------------------------------------------------------------------
# Few-shot pairs — SYNTHETIC, domain-agnostic insurance patterns
# No real document text. These teach extraction PATTERNS that apply to any LOB.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-pass extraction focus instructions
#
# Running the same prompt 3× at temperature=0 gives identical output — useless.
# Each pass appends a DIFFERENT focus suffix to the system message so the model
# attends to different fact types in the same chunk.
#
# Pass 1: general extraction   (existing behavior, no suffix)
# Pass 2: numeric / limit facts (dollar amounts, time periods, percentages)
# Pass 3: obligation / procedure facts (must-do, sequential steps, conditions)
#
# After all 3 passes the caller deduplicates by (subject, relation, object).
# Expected uplift: ~1 triple/chunk → ~3-4 triples/chunk (after dedup).
# ---------------------------------------------------------------------------

PASS_FOCUS_INSTRUCTIONS: list[str] = [
    # Pass 1 — general (unmodified system prompt)
    "",

    # Pass 2 — numeric facts only
    (
        "\n\n## Extraction Focus (this pass only)\n"
        "Extract ONLY triples where the object is a specific numeric value: "
        "a dollar amount (e.g. \"$500,000\"), a time period (e.g. \"30 days\", \"60 days\"), "
        "a percentage (e.g. \"80%\"), or a count/threshold. "
        "Examples: HAS_COVERAGE_LIMIT, HAS_DEDUCTIBLE, HAS_DEADLINE, HAS_WAITING_PERIOD. "
        "Skip coverage/exclusion/definition triples — they were captured in the previous pass."
    ),

    # Pass 3 — obligation / procedure / condition facts only
    (
        "\n\n## Extraction Focus (this pass only)\n"
        "Extract ONLY triples about: "
        "(a) obligations and requirements — what parties MUST do (MUST_NOTIFY, MUST_FILE, MUST_PAY, REQUIRED_TO_*); "
        "(b) sequential procedures — step A happens before step B (PRECEDES, FOLLOWED_BY); "
        "(c) conditional coverage — when coverage applies or is suspended (HAS_CONDITION, APPLIES_WHEN, SUSPENDED_IF). "
        "Skip numeric and coverage/exclusion triples — they were captured in earlier passes."
    ),
]


FEW_SHOT_PAIRS: list[tuple[str, str]] = [
    # Pattern 1: Exclusion list — teaches list extraction + negation relation
    (
        "Text: This policy does not cover losses caused by: 1. War or military action; "
        "2. Nuclear hazard or radioactive contamination; 3. Intentional acts by the insured; "
        "4. Wear and tear or gradual deterioration.",
        '[{"subject": "Policy", "relation": "EXCLUDED_FROM", "object": "War or Military Action", '
        '"span": "This policy does not cover losses caused by: 1. War or military action", '
        '"confidence": 0.97}, '
        '{"subject": "Policy", "relation": "EXCLUDED_FROM", "object": "Nuclear Hazard", '
        '"span": "2. Nuclear hazard or radioactive contamination", "confidence": 0.96}, '
        '{"subject": "Policy", "relation": "EXCLUDED_FROM", "object": "Intentional Acts", '
        '"span": "3. Intentional acts by the insured", "confidence": 0.96}, '
        '{"subject": "Policy", "relation": "EXCLUDED_FROM", "object": "Wear and Tear", '
        '"span": "4. Wear and tear or gradual deterioration", "confidence": 0.95}]'
    ),
    # Pattern 2: Coverage with dollar limit — teaches numeric extraction
    (
        "Text: Coverage A provides protection for the insured dwelling up to a maximum "
        "of $350,000. Coverage B provides protection for personal property up to $100,000. "
        "A deductible of $1,000 applies to each claim.",
        '[{"subject": "Coverage A", "relation": "COVERS", "object": "Insured Dwelling", '
        '"span": "Coverage A provides protection for the insured dwelling", "confidence": 0.98}, '
        '{"subject": "Coverage A", "relation": "HAS_COVERAGE_LIMIT", "object": "$350,000", '
        '"span": "insured dwelling up to a maximum of $350,000", "confidence": 0.99}, '
        '{"subject": "Coverage B", "relation": "COVERS", "object": "Personal Property", '
        '"span": "Coverage B provides protection for personal property", "confidence": 0.98}, '
        '{"subject": "Coverage B", "relation": "HAS_COVERAGE_LIMIT", "object": "$100,000", '
        '"span": "personal property up to $100,000", "confidence": 0.99}, '
        '{"subject": "Policy", "relation": "HAS_DEDUCTIBLE", "object": "$1,000", '
        '"span": "A deductible of $1,000 applies to each claim", "confidence": 0.98}]'
    ),
    # Pattern 3: Definition block — teaches multi-definition extraction
    (
        "Text: 1. Peril means any event that may cause a loss. "
        "2. Dwelling means a residential structure used as a primary residence. "
        "3. Personal property means movable items owned by the insured.",
        '[{"subject": "Peril", "relation": "DEFINED_AS", '
        '"object": "Event that may cause a loss", '
        '"span": "Peril means any event that may cause a loss", "confidence": 0.98}, '
        '{"subject": "Dwelling", "relation": "DEFINED_AS", '
        '"object": "Residential structure used as primary residence", '
        '"span": "Dwelling means a residential structure used as a primary residence", '
        '"confidence": 0.98}, '
        '{"subject": "Personal Property", "relation": "DEFINED_AS", '
        '"object": "Movable items owned by the insured", '
        '"span": "Personal property means movable items owned by the insured", '
        '"confidence": 0.98}]'
    ),
    # Pattern 4: Obligation with deadline + sequential procedure
    (
        "Text: In the event of a loss, the insured must: (1) Notify the insurer within "
        "24 hours. (2) File a written proof of loss within 60 days. (3) Cooperate with "
        "the claims investigation and provide all requested documentation.",
        '[{"subject": "Insured", "relation": "MUST_NOTIFY", "object": "Insurer", '
        '"span": "the insured must: (1) Notify the insurer within 24 hours", "confidence": 0.98}, '
        '{"subject": "Loss Notification", "relation": "HAS_DEADLINE", "object": "24 hours", '
        '"span": "Notify the insurer within 24 hours", "confidence": 0.97}, '
        '{"subject": "Insured", "relation": "MUST_FILE", "object": "Proof of Loss", '
        '"span": "File a written proof of loss within 60 days", "confidence": 0.98}, '
        '{"subject": "Proof of Loss", "relation": "HAS_DEADLINE", "object": "60 days", '
        '"span": "proof of loss within 60 days", "confidence": 0.98}, '
        '{"subject": "Notify Insurer", "relation": "PRECEDES", "object": "File Proof of Loss", '
        '"span": "Notify the insurer within 24 hours. (2) File a written proof of loss", '
        '"confidence": 0.95}]'
    ),
    # Pattern 5: Waiting period + exception — teaches conditional extraction
    (
        "Text: There is a 30-day waiting period before coverage takes effect, unless "
        "the policy is purchased in connection with a new loan or property purchase.",
        '[{"subject": "Policy", "relation": "HAS_WAITING_PERIOD", "object": "30 days", '
        '"span": "30-day waiting period before coverage takes effect", "confidence": 0.98}, '
        '{"subject": "Waiting Period", "relation": "HAS_EXCEPTION", "object": "New Loan Purchase", '
        '"span": "unless the policy is purchased in connection with a new loan", '
        '"confidence": 0.95}]'
    ),
]
