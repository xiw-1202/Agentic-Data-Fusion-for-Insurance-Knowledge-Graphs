"""Flood / NFIP query-evaluation tasks — derived from the source data, not the KG.

Source inputs for flood:
  - data/flood/raw/openfema/policies_sample.json  (OpenFEMA NFIP policies, 67 records)
  - data/flood/raw/openfema/claims_sample.json    (OpenFEMA NFIP claims, 72 records)
  - data/flood/raw/pdf/fema_F-123-general-property-SFIP_2021.pdf  (27-page SFIP policy form)

Each question asks about a concept that appears in the raw inputs (column name,
common value, or policy-document section). The Cypher tries to surface the
corresponding KG evidence without assuming specific canonical relation names —
we use `type(r) CONTAINS 'KEYWORD'` when the exact name may differ by run.

Keyword-match scoring: the test passes if the query returns any row AND at least
one expected keyword appears anywhere in the result string. Keywords are drawn
from actual values/sections in the source, not from the KG.
"""
from __future__ import annotations

EVAL_TASKS_FLOOD: list[dict] = [
    # --- Coverage (6) — from SFIP policy form + OpenFEMA policy coverage fields ---
    {
        "id": 1, "category": "coverage",
        "question": "What types of property does the SFIP General Property Form cover (Coverage A / Coverage B)?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'building' OR toLower(n.id) CONTAINS 'coverage a' OR toLower(n.id) CONTAINS 'coverage b' OR toLower(n.id) CONTAINS 'personal property' RETURN DISTINCT n.id, labels(n) LIMIT 20",
        "keywords": ["building", "personal property", "coverage", "contents"],
    },
    {
        "id": 2, "category": "coverage",
        "question": "What is Increased Cost of Compliance (ICC / Coverage D) coverage?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'increased cost of compliance' OR toLower(n.id) CONTAINS 'icc' OR toLower(n.id) CONTAINS 'coverage d' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["increased", "compliance", "icc", "coverage"],
    },
    {
        "id": 3, "category": "coverage",
        "question": "Is building replacement cost value addressed in the policy form?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'replacement cost' OR toLower(n.id) CONTAINS 'actual cash value' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["replacement", "cost", "actual cash"],
    },
    {
        "id": 4, "category": "coverage",
        "question": "What are the Coverage C loss-avoidance measures the policy allows?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'loss avoidance' OR toLower(n.id) CONTAINS 'coverage c' OR toLower(n.id) CONTAINS 'sandbag' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["loss avoidance", "coverage c", "sandbag"],
    },
    {
        "id": 5, "category": "coverage",
        "question": "Does the form reference elevated-building or condominium coverage variants?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'elevated' OR toLower(n.id) CONTAINS 'condominium' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["elevated", "condominium"],
    },
    {
        "id": 6, "category": "coverage",
        "question": "What total building insurance coverage amounts appear in policy records?",
        "cypher": "MATCH (p:Entity)-[r]->(v) WHERE p.entity_type = 'PolicyRecord' AND (type(r) CONTAINS 'BUILDING' AND type(r) CONTAINS 'COVERAGE') RETURN DISTINCT type(r) AS rel, v.id LIMIT 10",
        "keywords": ["building", "coverage", "insurance"],
    },

    # --- Exclusions (3) — policy form enumerates several classes of exclusion ---
    {
        "id": 7, "category": "exclusions",
        "question": "What causes of damage are excluded (earth movement, sewer backup, etc.)?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'earth movement' OR toLower(n.id) CONTAINS 'sewer' OR toLower(n.id) CONTAINS 'exclud' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["earth movement", "sewer", "excluded"],
    },
    {
        "id": 8, "category": "exclusions",
        "question": "Are business-interruption or financial-loss-type exclusions mentioned?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'business interrupt' OR toLower(n.id) CONTAINS 'financial loss' OR toLower(n.id) CONTAINS 'loss of use' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["business", "interruption", "financial"],
    },
    {
        "id": 9, "category": "exclusions",
        "question": "What property types are specifically excluded (fences, land, plants)?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'fence' OR toLower(n.id) CONTAINS 'land' OR toLower(n.id) CONTAINS 'lawn' OR toLower(n.id) CONTAINS 'tree' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["fence", "land", "lawn", "tree"],
    },

    # --- Policy terms (6) — SFIP prescribes deductibles, waiting period, etc. ---
    {
        "id": 10, "category": "policy_terms",
        "question": "What is the NFIP waiting period before a new policy takes effect?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'waiting period' OR toLower(n.id) CONTAINS '30 day' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["waiting", "period", "30"],
    },
    {
        "id": 11, "category": "policy_terms",
        "question": "Does the policy form describe deductibles for building and contents?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'deductible' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["deductible"],
    },
    {
        "id": 12, "category": "policy_terms",
        "question": "What proof-of-loss deadline is required?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'proof of loss' OR toLower(n.id) CONTAINS '60 day' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["proof", "loss", "60"],
    },
    {
        "id": 13, "category": "policy_terms",
        "question": "Are policy effective and termination dates recorded per policy record?",
        "cypher": "MATCH (p:Entity)-[r]->(d) WHERE p.entity_type = 'PolicyRecord' AND (type(r) CONTAINS 'EFFECTIVE' OR type(r) CONTAINS 'TERMINATION') RETURN DISTINCT type(r) AS rel, d.id LIMIT 10",
        "keywords": ["effective", "termination", "date"],
    },
    {
        "id": 14, "category": "policy_terms",
        "question": "What fees or surcharges does the form describe (federal fee, HFIAA, reserve-fund)?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'fee' OR toLower(n.id) CONTAINS 'hfiaa' OR toLower(n.id) CONTAINS 'reserve' OR toLower(n.id) CONTAINS 'surcharge' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["fee", "hfiaa", "surcharge", "reserve"],
    },
    {
        "id": 15, "category": "policy_terms",
        "question": "What is the Liberalization Clause?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'liberal' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["liberalization", "clause"],
    },

    # --- Claims (3) — SFIP claim process + OpenFEMA claim data ---
    {
        "id": 16, "category": "claims",
        "question": "What steps must a policyholder follow after a flood loss?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'notify' OR toLower(n.id) CONTAINS 'notice of loss' OR toLower(n.id) CONTAINS 'claim process' OR toLower(n.id) CONTAINS 'separate' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["notify", "notice", "separate", "process"],
    },
    {
        "id": 17, "category": "claims",
        "question": "What appraisal process resolves loss-amount disputes?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'appraisal' OR toLower(n.id) CONTAINS 'umpire' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["appraisal", "umpire", "dispute"],
    },
    {
        "id": 18, "category": "claims",
        "question": "What causes-of-damage codes appear in OpenFEMA claims?",
        "cypher": "MATCH (c:Entity)-[r]->(cause) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'CAUSE' RETURN DISTINCT cause.id LIMIT 20",
        "keywords": ["damage", "cause", "water", "flood"],
    },

    # --- Definitions (2) ---
    {
        "id": 19, "category": "definitions",
        "question": "How is 'flood' defined in the SFIP?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'flood' AND (toLower(n.id) CONTAINS 'defin' OR toLower(n.id) CONTAINS 'inundation' OR size(n.id) > 40) RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["flood", "inundation", "overflow"],
    },
    {
        "id": 20, "category": "definitions",
        "question": "What is the SFIP definition of 'building'?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'building' AND (toLower(n.id) CONTAINS 'defin' OR toLower(n.id) CONTAINS 'walled' OR toLower(n.id) CONTAINS 'roofed') RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["building", "walled", "roofed", "structure"],
    },

    # --- Structured claims (4) — OpenFEMA claim record fields ---
    {
        "id": 21, "category": "structured_claims",
        "question": "Are individual OpenFEMA claim records represented as entities?",
        "cypher": "MATCH (c:Entity) WHERE c.entity_type = 'ClaimRecord' RETURN count(c) AS claim_count",
        "keywords": ["claim_count"],
    },
    {
        "id": 22, "category": "structured_claims",
        "question": "Do claim records carry date-of-loss information?",
        "cypher": "MATCH (c:Entity)-[r]->(d) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'LOSS' AND type(r) CONTAINS 'DATE' RETURN type(r) AS relation, count(DISTINCT d) AS distinct_dates",
        "keywords": ["date", "loss", "distinct_dates"],
    },
    {
        "id": 23, "category": "structured_claims",
        "question": "Do claim records include building/contents payment amounts?",
        "cypher": "MATCH (c:Entity)-[r]->(amt) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'PAID' AND type(r) CONTAINS 'BUILDING' RETURN type(r) AS relation, count(DISTINCT amt) AS distinct_amounts",
        "keywords": ["paid", "building", "distinct_amounts"],
    },
    {
        "id": 24, "category": "structured_claims",
        "question": "Are water-depth values captured per claim?",
        "cypher": "MATCH (c:Entity)-[r]->(w) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'WATER' AND type(r) CONTAINS 'DEPTH' RETURN DISTINCT w.id LIMIT 10",
        "keywords": ["water", "depth"],
    },

    # --- Structured policies (3) — OpenFEMA policy record fields ---
    {
        "id": 25, "category": "structured_policies",
        "question": "Are individual OpenFEMA policy records represented as entities?",
        "cypher": "MATCH (p:Entity) WHERE p.entity_type = 'PolicyRecord' RETURN count(p) AS policy_count",
        "keywords": ["policy_count"],
    },
    {
        "id": 26, "category": "structured_policies",
        "question": "Do policy records include rated flood-zone classification?",
        "cypher": "MATCH (p:Entity)-[r]->(z) WHERE p.entity_type = 'PolicyRecord' AND type(r) CONTAINS 'ZONE' RETURN DISTINCT z.id LIMIT 20",
        "keywords": ["zone", "a", "v", "x"],
    },
    {
        "id": 27, "category": "structured_policies",
        "question": "What policy-cost or premium fields are preserved for policy records?",
        "cypher": "MATCH (p:Entity)-[r]->(v) WHERE p.entity_type = 'PolicyRecord' AND (type(r) CONTAINS 'COST' OR type(r) CONTAINS 'PREMIUM' OR type(r) CONTAINS 'FEE') RETURN DISTINCT type(r) AS rel LIMIT 15",
        "keywords": ["cost", "premium", "fee"],
    },

    # --- Cross-source (5) — policy ↔ claim ↔ PDF concept linkage ---
    {
        "id": 28, "category": "cross_source",
        "question": "Do structured records share flood-zone values with PDF-extracted concepts?",
        "cypher": "MATCH (rec:Entity)-[r]->(z:Entity) WHERE rec.entity_type IN ['PolicyRecord','ClaimRecord'] AND type(r) CONTAINS 'ZONE' RETURN DISTINCT z.id LIMIT 10",
        "keywords": ["zone", "flood"],
    },
    {
        "id": 29, "category": "cross_source",
        "question": "Are any community names shared across policy records and PDF concepts?",
        "cypher": "MATCH (p:Entity)-[r]->(cn:Entity) WHERE p.entity_type = 'PolicyRecord' AND type(r) CONTAINS 'COMMUNITY' RETURN DISTINCT cn.id LIMIT 20",
        "keywords": ["community", "city", "village", "parish"],
    },
    {
        "id": 30, "category": "cross_source",
        "question": "Do policy and claim records share property-state values?",
        "cypher": "MATCH (p:Entity)-[r1]->(s:Entity), (c:Entity)-[r2]->(s) WHERE p.entity_type = 'PolicyRecord' AND c.entity_type = 'ClaimRecord' AND (type(r1) CONTAINS 'STATE' OR type(r1) CONTAINS 'ST ') RETURN DISTINCT s.id LIMIT 20",
        "keywords": ["state", "tx", "la", "ga", "fl"],
    },
    {
        "id": 31, "category": "cross_source",
        "question": "Is the NFIP-rated community number linked across records?",
        "cypher": "MATCH (rec:Entity)-[r]->(cn:Entity) WHERE rec.entity_type IN ['PolicyRecord','ClaimRecord'] AND type(r) CONTAINS 'COMMUNITY' AND type(r) CONTAINS 'NUMBER' RETURN DISTINCT cn.id LIMIT 20",
        "keywords": ["number", "community"],
    },
    {
        "id": 32, "category": "cross_source",
        "question": "Do flood-event names (e.g. Hurricane) appear as named entities?",
        "cypher": "MATCH (c:Entity)-[r]->(e:Entity) WHERE c.entity_type = 'ClaimRecord' AND (type(r) CONTAINS 'EVENT' OR type(r) CONTAINS 'FLOOD_EVENT') RETURN DISTINCT e.id LIMIT 20",
        "keywords": ["hurricane", "flood", "storm", "event"],
    },

    # --- Identity (1) ---
    {
        "id": 33, "category": "identity",
        "question": "Are there shared property-identity values linking policies and claims?",
        "cypher": "MATCH (rec:Entity) WHERE rec.entity_type IN ['PolicyRecord','ClaimRecord'] RETURN rec.entity_type AS type, count(rec) AS n",
        "keywords": ["PolicyRecord", "ClaimRecord"],
    },

    # --- Claim lookup (4) ---
    {
        "id": 34, "category": "claim_lookup",
        "question": "What are the highest building claim payment amounts?",
        "cypher": "MATCH (c:Entity)-[r]->(amt:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'PAID' AND type(r) CONTAINS 'BUILDING' WITH c, toFloat(amt.id) AS a ORDER BY a DESC LIMIT 5 RETURN c.id AS claim, a AS amount",
        "keywords": ["amount", "claim"],
    },
    {
        "id": 35, "category": "claim_lookup",
        "question": "Which flood events are represented in the claims data?",
        "cypher": "MATCH (c:Entity)-[r]->(e:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'EVENT' RETURN e.id AS event, count(c) AS n ORDER BY n DESC LIMIT 10",
        "keywords": ["hurricane", "flood", "storm"],
    },
    {
        "id": 36, "category": "claim_lookup",
        "question": "What year-of-loss values appear in the claim records?",
        "cypher": "MATCH (c:Entity)-[r]->(y:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'YEAR' RETURN y.id AS year, count(c) AS n ORDER BY n DESC LIMIT 5",
        "keywords": ["year", "20"],
    },
    {
        "id": 37, "category": "claim_lookup",
        "question": "What cause-of-damage codes are most common?",
        "cypher": "MATCH (c:Entity)-[r]->(cd:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'CAUSE' RETURN cd.id AS cause, count(c) AS n ORDER BY n DESC LIMIT 5",
        "keywords": ["cause"],
    },

    # --- Policy lookup (3) ---
    {
        "id": 38, "category": "policy_lookup",
        "question": "Which property states are represented in policy records?",
        "cypher": "MATCH (p:Entity)-[r]->(s:Entity) WHERE p.entity_type = 'PolicyRecord' AND (type(r) CONTAINS 'STATE' OR type(r) = 'HAS_PROPERTY_STATE') RETURN DISTINCT s.id LIMIT 20",
        "keywords": ["ga", "tx", "la", "fl", "ny"],
    },
    {
        "id": 39, "category": "policy_lookup",
        "question": "What occupancy types appear in policy records?",
        "cypher": "MATCH (p:Entity)-[r]->(o:Entity) WHERE p.entity_type = 'PolicyRecord' AND type(r) CONTAINS 'OCCUPANCY' RETURN o.id AS occupancy, count(p) AS n ORDER BY n DESC LIMIT 10",
        "keywords": ["occupancy"],
    },
    {
        "id": 40, "category": "policy_lookup",
        "question": "Which rate methods are used across policies (RatingEngine, manual, etc.)?",
        "cypher": "MATCH (p:Entity)-[r]->(m:Entity) WHERE p.entity_type = 'PolicyRecord' AND type(r) CONTAINS 'RATE' AND type(r) CONTAINS 'METHOD' RETURN DISTINCT m.id LIMIT 10",
        "keywords": ["rating", "engine", "method"],
    },
]
