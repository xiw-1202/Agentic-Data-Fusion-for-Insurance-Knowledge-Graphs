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

Entity types identified in this document (assign one to each subject and object):
{entity_type_lines}

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
- STRUCTURED DATA: When the passage contains records, tables, or key-value fields, extract EVERY field as a separate triple — dates, amounts, locations, codes, classifications. Each record should yield 5-10 triples.
- SHORT CLAUSES: Even brief passages (1-2 sentences) usually contain 2-3 extractable facts — look for implicit relations (options, conditions, parties involved).
- Include span: verbatim quote <=120 chars supporting the triple
- Include confidence: 0.0–1.0
- Include subject_type and object_type: assign one entity type from the list above to each (use "Unknown" if none fit)
- Return [] only if the passage contains no factual insurance content
- SEMANTIC ACCURACY (high priority — omit uncertain triples rather than guess wrong):
  - An organization that ADMINISTERS a program vs one that PROVIDES coverage
  - An exclusion from coverage vs a cause of damage (direction matters!)
  - A deductible amount (numeric) vs the property the deductible applies to (structure)
  - A peril being excluded FROM coverage vs a peril CAUSING a loss
  - WRONG: "Agency COVERS Program" — agencies administer, policies cover
  - WRONG: "Building HAS_DEDUCTIBLE Policy" — deductibles are dollar amounts, not structures
  - WRONG: "Flood HAS_CAUSE_OF_DAMAGE Earth Movement" — reversed direction

Output format — JSON array only:
[{{"subject": "...", "subject_type": "...", "relation": "...", "object": "...", "object_type": "...", "span": "...", "confidence": 0.9}}]"""


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


# ---------------------------------------------------------------------------
# Recall-oriented Pass 2 prompt
# ---------------------------------------------------------------------------
# Shows Pass 1 results and asks the LLM to find what it missed.
# The key insight: showing already-extracted triples shifts attention to
# what's NOT yet captured — list tails, conditions, cross-sentence facts.

RECALL_PASS_PROMPT = """You are a knowledge graph completeness checker.

The following triples were already extracted from this passage:
{existing_triples}

Re-read the passage below and extract ADDITIONAL facts that were NOT captured above.

Focus on:
- Items in lists that were only partially extracted (list tails)
- Conditional or exception clauses ("if", "unless", "except", "provided that")
- Numeric values (dollar amounts, dates, time periods) tied to specific entities
- Cross-sentence relations where the subject is in one sentence and the predicate/object in another
- Low-salience but factual statements (minor definitions, procedural details, party roles)
- Negation facts ("does not cover", "is void if", "excluded from")

Rules:
- Do NOT repeat any triple already listed above
- Each new triple must be supported by specific text in the passage
- Include span: verbatim quote <=120 chars from the passage supporting the triple
- Include confidence: 0.0–1.0
- Return [] if no additional facts exist

Output format — JSON array only:
[{{"subject": "...", "subject_type": "...", "relation": "...", "object": "...", "object_type": "...", "span": "...", "confidence": 0.85}}]

Text: {chunk_text}"""


SINGLE_PASS_FOCUS = (
    "\n\n## Extraction Focus\n"
    "Extract ALL of the following fact types from this passage:\n"
    "1. **Coverage & definitions**: what is covered, excluded, or defined.\n"
    "2. **Numeric facts**: dollar amounts (\"$500,000\"), time periods (\"30 days\"), "
    "percentages (\"80%\"), counts, deductibles, limits.\n"
    "3. **Obligations & procedures**: what parties MUST do (MUST_NOTIFY, MUST_FILE), "
    "sequential steps (PRECEDES, FOLLOWED_BY), and conditional coverage "
    "(HAS_CONDITION, APPLIES_WHEN, SUSPENDED_IF).\n"
    "Do not omit any category — cover all three in this single pass."
)


FEW_SHOT_PAIRS: list[tuple[str, str]] = [
    # Pattern 1: Exclusion list — teaches list extraction + negation relation
    (
        "Text: This policy does not cover losses caused by: 1. War or military action; "
        "2. Nuclear hazard or radioactive contamination; 3. Intentional acts by the insured; "
        "4. Wear and tear or gradual deterioration.",
        '[{"subject": "Policy", "subject_type": "InsurancePolicy", "relation": "EXCLUDED_FROM", "object": "War or Military Action", "object_type": "ExcludedPeril", '
        '"span": "This policy does not cover losses caused by: 1. War or military action", '
        '"confidence": 0.97}, '
        '{"subject": "Policy", "subject_type": "InsurancePolicy", "relation": "EXCLUDED_FROM", "object": "Nuclear Hazard", "object_type": "ExcludedPeril", '
        '"span": "2. Nuclear hazard or radioactive contamination", "confidence": 0.96}, '
        '{"subject": "Policy", "subject_type": "InsurancePolicy", "relation": "EXCLUDED_FROM", "object": "Intentional Acts", "object_type": "ExcludedPeril", '
        '"span": "3. Intentional acts by the insured", "confidence": 0.96}, '
        '{"subject": "Policy", "subject_type": "InsurancePolicy", "relation": "EXCLUDED_FROM", "object": "Wear and Tear", "object_type": "ExcludedPeril", '
        '"span": "4. Wear and tear or gradual deterioration", "confidence": 0.95}]'
    ),
    # Pattern 2: Coverage with dollar limit — teaches numeric extraction
    (
        "Text: Coverage A provides protection for the insured dwelling up to a maximum "
        "of $350,000. Coverage B provides protection for personal property up to $100,000. "
        "A deductible of $1,000 applies to each claim.",
        '[{"subject": "Coverage A", "subject_type": "CoverageType", "relation": "COVERS", "object": "Insured Dwelling", "object_type": "InsuredProperty", '
        '"span": "Coverage A provides protection for the insured dwelling", "confidence": 0.98}, '
        '{"subject": "Coverage A", "subject_type": "CoverageType", "relation": "HAS_COVERAGE_LIMIT", "object": "$350,000", "object_type": "FinancialAmount", '
        '"span": "insured dwelling up to a maximum of $350,000", "confidence": 0.99}, '
        '{"subject": "Coverage B", "subject_type": "CoverageType", "relation": "COVERS", "object": "Personal Property", "object_type": "InsuredProperty", '
        '"span": "Coverage B provides protection for personal property", "confidence": 0.98}, '
        '{"subject": "Coverage B", "subject_type": "CoverageType", "relation": "HAS_COVERAGE_LIMIT", "object": "$100,000", "object_type": "FinancialAmount", '
        '"span": "personal property up to $100,000", "confidence": 0.99}, '
        '{"subject": "Policy", "subject_type": "InsurancePolicy", "relation": "HAS_DEDUCTIBLE", "object": "$1,000", "object_type": "FinancialAmount", '
        '"span": "A deductible of $1,000 applies to each claim", "confidence": 0.98}]'
    ),
    # Pattern 3: Definition block — teaches multi-definition extraction
    (
        "Text: 1. Peril means any event that may cause a loss. "
        "2. Dwelling means a residential structure used as a primary residence. "
        "3. Personal property means movable items owned by the insured.",
        '[{"subject": "Peril", "subject_type": "RiskType", "relation": "DEFINED_AS", '
        '"object": "Event that may cause a loss", "object_type": "Definition", '
        '"span": "Peril means any event that may cause a loss", "confidence": 0.98}, '
        '{"subject": "Dwelling", "subject_type": "InsuredProperty", "relation": "DEFINED_AS", '
        '"object": "Residential structure used as primary residence", "object_type": "Definition", '
        '"span": "Dwelling means a residential structure used as a primary residence", '
        '"confidence": 0.98}, '
        '{"subject": "Personal Property", "subject_type": "InsuredProperty", "relation": "DEFINED_AS", '
        '"object": "Movable items owned by the insured", "object_type": "Definition", '
        '"span": "Personal property means movable items owned by the insured", '
        '"confidence": 0.98}]'
    ),
    # Pattern 4: Obligation with deadline + sequential procedure
    (
        "Text: In the event of a loss, the insured must: (1) Notify the insurer within "
        "24 hours. (2) File a written proof of loss within 60 days. (3) Cooperate with "
        "the claims investigation and provide all requested documentation.",
        '[{"subject": "Insured", "subject_type": "Party", "relation": "MUST_NOTIFY", "object": "Insurer", "object_type": "Party", '
        '"span": "the insured must: (1) Notify the insurer within 24 hours", "confidence": 0.98}, '
        '{"subject": "Loss Notification", "subject_type": "Procedure", "relation": "HAS_DEADLINE", "object": "24 hours", "object_type": "TimePeriod", '
        '"span": "Notify the insurer within 24 hours", "confidence": 0.97}, '
        '{"subject": "Insured", "subject_type": "Party", "relation": "MUST_FILE", "object": "Proof of Loss", "object_type": "Document", '
        '"span": "File a written proof of loss within 60 days", "confidence": 0.98}, '
        '{"subject": "Proof of Loss", "subject_type": "Document", "relation": "HAS_DEADLINE", "object": "60 days", "object_type": "TimePeriod", '
        '"span": "proof of loss within 60 days", "confidence": 0.98}, '
        '{"subject": "Notify Insurer", "subject_type": "Procedure", "relation": "PRECEDES", "object": "File Proof of Loss", "object_type": "Procedure", '
        '"span": "Notify the insurer within 24 hours. (2) File a written proof of loss", '
        '"confidence": 0.95}]'
    ),
    # Pattern 5: Waiting period + exception — teaches conditional extraction
    (
        "Text: There is a 30-day waiting period before coverage takes effect, unless "
        "the policy is purchased in connection with a new loan or property purchase.",
        '[{"subject": "Policy", "subject_type": "InsurancePolicy", "relation": "HAS_WAITING_PERIOD", "object": "30 days", "object_type": "TimePeriod", '
        '"span": "30-day waiting period before coverage takes effect", "confidence": 0.98}, '
        '{"subject": "Waiting Period", "subject_type": "Condition", "relation": "HAS_EXCEPTION", "object": "New Loan Purchase", "object_type": "Condition", '
        '"span": "unless the policy is purchased in connection with a new loan", '
        '"confidence": 0.95}]'
    ),
    # Pattern 6: Structured / tabular record — teaches extraction from key-value data
    (
        "Text: RECORD:\n"
        "  [Policy] policy effective date: 2024-03-15 | policy termination date: 2025-03-15 | policy cost: 850\n"
        "  [Coverage] total building insurance coverage: 250000 | building deductible code: A\n"
        "  [Property] occupancy type: 11 | number of floors: 2 | original construction date: 1995-06-01\n"
        "  [Location] rated flood zone: AE | property state: FL | reported zip code: 33019",
        '[{"subject": "Policy-2024-03-15", "subject_type": "InsurancePolicy", "relation": "HAS_EFFECTIVE_DATE", "object": "2024-03-15", "object_type": "TimePeriod", '
        '"span": "policy effective date: 2024-03-15", "confidence": 0.98}, '
        '{"subject": "Policy-2024-03-15", "subject_type": "InsurancePolicy", "relation": "HAS_DEADLINE", "object": "2025-03-15", "object_type": "TimePeriod", '
        '"span": "policy termination date: 2025-03-15", "confidence": 0.98}, '
        '{"subject": "Policy-2024-03-15", "subject_type": "InsurancePolicy", "relation": "HAS_COVERAGE_LIMIT", "object": "$250,000", "object_type": "FinancialAmount", '
        '"span": "total building insurance coverage: 250000", "confidence": 0.99}, '
        '{"subject": "Policy-2024-03-15", "subject_type": "InsurancePolicy", "relation": "HAS_DEDUCTIBLE", "object": "Code A", "object_type": "FinancialAmount", '
        '"span": "building deductible code: A", "confidence": 0.95}, '
        '{"subject": "Policy-2024-03-15", "subject_type": "InsurancePolicy", "relation": "COVERS", "object": "2-Story Residential Building", "object_type": "InsuredProperty", '
        '"span": "occupancy type: 11 | number of floors: 2", "confidence": 0.93}, '
        '{"subject": "Policy-2024-03-15", "subject_type": "InsurancePolicy", "relation": "IS_CLASSIFIED_AS", "object": "Flood Zone AE", "object_type": "RiskType", '
        '"span": "rated flood zone: AE", "confidence": 0.97}, '
        '{"subject": "Insured Property", "subject_type": "InsuredProperty", "relation": "IS_CLASSIFIED_AS", "object": "FL-33019", "object_type": "Location", '
        '"span": "property state: FL | reported zip code: 33019", "confidence": 0.95}]'
    ),
    # Pattern 7: Short clause — teaches extracting 2-3 triples even from brief text
    (
        "Text: The insurer may repair, rebuild, or replace the damaged property with "
        "material of like kind and quality within a reasonable time.",
        '[{"subject": "Insurer", "subject_type": "Party", "relation": "HAS_OPTION", "object": "Repair Property", "object_type": "Procedure", '
        '"span": "The insurer may repair, rebuild, or replace the damaged property", "confidence": 0.96}, '
        '{"subject": "Insurer", "subject_type": "Party", "relation": "HAS_OPTION", "object": "Rebuild Property", "object_type": "Procedure", '
        '"span": "The insurer may repair, rebuild, or replace the damaged property", "confidence": 0.96}, '
        '{"subject": "Insurer", "subject_type": "Party", "relation": "HAS_OPTION", "object": "Replace Property", "object_type": "Procedure", '
        '"span": "The insurer may repair, rebuild, or replace the damaged property", "confidence": 0.96}]'
    ),
    # Pattern 8: Semantic accuracy — teaches correct role assignment + exclusion direction
    (
        "Text: The National Insurance Program, administered by the Federal Agency, "
        "provides coverage for residential buildings. Earth movement, including "
        "landslide and mudflow, is excluded from coverage. The deductible for building "
        "coverage is $2,000.",
        '[{"subject": "National Insurance Program", "subject_type": "InsurancePolicy", "relation": "COVERS", "object": "Residential Buildings", "object_type": "InsuredProperty", '
        '"span": "provides coverage for residential buildings", "confidence": 0.97}, '
        '{"subject": "Federal Agency", "subject_type": "Organization", "relation": "ADMINISTERS", "object": "National Insurance Program", "object_type": "InsurancePolicy", '
        '"span": "administered by the Federal Agency", "confidence": 0.96}, '
        '{"subject": "National Insurance Program", "subject_type": "InsurancePolicy", "relation": "EXCLUDED_FROM", "object": "Earth Movement", "object_type": "ExcludedPeril", '
        '"span": "Earth movement, including landslide and mudflow, is excluded from coverage", "confidence": 0.97}, '
        '{"subject": "National Insurance Program", "subject_type": "InsurancePolicy", "relation": "EXCLUDED_FROM", "object": "Landslide", "object_type": "ExcludedPeril", '
        '"span": "Earth movement, including landslide and mudflow, is excluded", "confidence": 0.95}, '
        '{"subject": "Building Coverage", "subject_type": "CoverageType", "relation": "HAS_DEDUCTIBLE", "object": "$2,000", "object_type": "FinancialAmount", '
        '"span": "The deductible for building coverage is $2,000", "confidence": 0.99}]'
    ),
]
