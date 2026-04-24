"""Emory_Spring2026 query-evaluation tasks.

Mirrors the 40-task structure of baseline/eval.py::EVAL_TASKS (flood/NFIP)
but uses Emory's auto-service-contract + T-Mobile claims + GEICO renters
domain language. Each task:
  - question: natural-language analytical question
  - cypher  : read-only query against the loaded Neo4j KG
  - keywords: expected tokens in the result set for keyword-match scoring
"""
from __future__ import annotations

EVAL_TASKS_EMORY: list[dict] = [
    # --- Coverage (6) ---
    {
        "id": 1,
        "category": "coverage",
        "question": "What types of covered parts are listed in the Auto Service Contract?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'covered part' OR toLower(n.id) CONTAINS 'cylinder' OR toLower(n.id) CONTAINS 'engine' OR toLower(n.id) CONTAINS 'transmission' RETURN DISTINCT n.id, labels(n) LIMIT 20",
        "keywords": ["engine", "transmission", "cylinder", "part", "covered"],
    },
    {
        "id": 2,
        "category": "coverage",
        "question": "What emergency roadside services does the Auto Service Contract cover?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'roadside' OR toLower(n.id) CONTAINS 'towing' OR toLower(n.id) CONTAINS 'emergency' RETURN DISTINCT n.id, labels(n) LIMIT 20",
        "keywords": ["roadside", "towing", "emergency"],
    },
    {
        "id": 3,
        "category": "coverage",
        "question": "What device types are covered under T-Mobile insurance claims?",
        "cypher": "MATCH (c:Entity)-[:HAS_DEVICE_TYPE]->(d:Entity) WHERE c.entity_type = 'ClaimRecord' RETURN DISTINCT d.id LIMIT 20",
        "keywords": ["cellular", "phone", "device", "smart"],
    },
    {
        "id": 4,
        "category": "coverage",
        "question": "Does the Auto Service Contract cover manufacturer deductible reimbursement?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'deductible' OR toLower(n.id) CONTAINS 'reimburs' OR toLower(n.id) CONTAINS \"manufacturer\" RETURN DISTINCT n.id, labels(n) LIMIT 20",
        "keywords": ["deductible", "manufacturer", "reimbursement"],
    },
    {
        "id": 5,
        "category": "coverage",
        "question": "What loss types are recorded in the T-Mobile device claims?",
        "cypher": "MATCH (c:Entity)-[:HAS_LOSS_TYPE]->(l:Entity) WHERE c.entity_type = 'ClaimRecord' RETURN DISTINCT l.id LIMIT 20",
        "keywords": ["physical damage", "loss", "damage"],
    },
    {
        "id": 6,
        "category": "coverage",
        "question": "What kinds of coverage limitations apply to the service contract?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'limitation' OR toLower(n.id) CONTAINS 'service contract limit' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["limit", "contract", "limitation"],
    },

    # --- Exclusions (3) ---
    {
        "id": 7,
        "category": "exclusions",
        "question": "What parts are excluded from the Auto Service Contract?",
        "cypher": "MATCH (n)-[r:EXCLUDED_FROM]->(c) RETURN n.id, c.id LIMIT 20",
        "keywords": ["brake", "spark plug", "wheel", "tire", "excluded"],
    },
    {
        "id": 8,
        "category": "exclusions",
        "question": "What conditions void the service contract coverage?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'void' OR toLower(n.id) CONTAINS 'misrepresent' OR toLower(n.id) CONTAINS 'odometer' OR toLower(n.id) CONTAINS 'unsafe' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["misrepresentation", "odometer", "unsafe", "void"],
    },
    {
        "id": 9,
        "category": "exclusions",
        "question": "What business or commercial uses void vehicle coverage?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'business' OR toLower(n.id) CONTAINS 'commercial' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["business", "commercial"],
    },

    # --- Policy terms (6) ---
    {
        "id": 10,
        "category": "policy_terms",
        "question": "What is the Service Contract Administrative Fee?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'administrative fee' OR toLower(n.id) CONTAINS 'service contract fee' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["administrative", "fee", "service contract"],
    },
    {
        "id": 11,
        "category": "policy_terms",
        "question": "How does a pro-rata refund of the service contract work?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'refund' OR toLower(n.id) CONTAINS 'pro-rata' OR toLower(n.id) CONTAINS 'unused term' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["refund", "pro-rata", "unused"],
    },
    {
        "id": 12,
        "category": "policy_terms",
        "question": "What arbitration or umpire process is defined for disputes?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'arbitrat' OR toLower(n.id) CONTAINS 'umpire' OR toLower(n.id) CONTAINS 'dispute' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["arbitration", "umpire", "dispute"],
    },
    {
        "id": 13,
        "category": "policy_terms",
        "question": "What is the lienholder priority for refunds?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'lienholder' OR toLower(n.id) CONTAINS 'priority' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["lienholder", "priority", "refund"],
    },
    {
        "id": 14,
        "category": "policy_terms",
        "question": "Is there a toll-free call process for service claims?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'toll-free' OR toLower(n.id) CONTAINS 'toll free' OR toLower(n.id) CONTAINS 'customer service' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["toll-free", "call", "customer"],
    },
    {
        "id": 15,
        "category": "policy_terms",
        "question": "What state amendments apply to the service contract?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'state' AND (toLower(n.id) CONTAINS 'amendment' OR toLower(n.id) CONTAINS 'law') RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["state", "amendment", "law"],
    },

    # --- Claims (3) ---
    {
        "id": 16,
        "category": "claims",
        "question": "What steps are in the claim review / repair-authorization process?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'claim review' OR toLower(n.id) CONTAINS 'authoriz' OR toLower(n.id) CONTAINS 'repair' OR toLower(n.id) CONTAINS 'claim process' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["claim", "review", "authorization", "repair"],
    },
    {
        "id": 17,
        "category": "claims",
        "question": "What agent or device CSAT categories appear in the claims data?",
        "cypher": "MATCH (c:Entity)-[r]->(v:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) IN ['HAS_AGENT_CSAT','HAS_DEVICE_CSAT','HAS_Q_NPS','HAS_NPS_CATEGORY'] RETURN DISTINCT type(r) AS rel, v.id LIMIT 20",
        "keywords": ["csat", "nps", "category"],
    },
    {
        "id": 18,
        "category": "claims",
        "question": "What issue-resolution methods appear in T-Mobile claims?",
        "cypher": "MATCH (c:Entity)-[r]->(m:Entity) WHERE c.entity_type = 'ClaimRecord' AND (type(r) CONTAINS 'RESOLV' OR type(r) CONTAINS 'RESOLUTION') RETURN DISTINCT type(r) AS relation, m.id LIMIT 20",
        "keywords": ["resolve", "resolution", "yes", "no"],
    },

    # --- Definitions (2) ---
    {
        "id": 19,
        "category": "definitions",
        "question": "What is a 'breakdown' under the Auto Service Contract?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'breakdown' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["breakdown", "mechanical", "part"],
    },
    {
        "id": 20,
        "category": "definitions",
        "question": "What parts, procedures, or service terms are defined in the contract?",
        "cypher": "MATCH (n)-[:INSTANCE_OF]->(c:Class) WHERE c.name IN ['ServiceProcedure','WarrantyServiceProcedure','Coverage'] AND size(n.id) < 60 RETURN DISTINCT n.id, c.name LIMIT 20",
        "keywords": ["service", "procedure", "coverage", "warranty"],
    },

    # --- Structured claims (4) ---
    {
        "id": 21,
        "category": "structured_claims",
        "question": "Are individual T-Mobile / GEICO claim records represented as entities in the KG?",
        "cypher": "MATCH (c:Entity) WHERE c.entity_type = 'ClaimRecord' RETURN count(c) AS claim_count",
        "keywords": ["claim_count"],
    },
    {
        "id": 22,
        "category": "structured_claims",
        "question": "Do claim records have time-to-resolve (hours) information?",
        "cypher": "MATCH (c:Entity)-[r:HAS_TIME_TO_RESOLVE_HR]->(h:Entity) WHERE c.entity_type = 'ClaimRecord' RETURN type(r) AS relation, count(DISTINCT h) AS distinct_values LIMIT 10",
        "keywords": ["time_to_resolve", "resolve", "distinct_values"],
    },
    {
        "id": 23,
        "category": "structured_claims",
        "question": "Do claim records carry device damage type information?",
        "cypher": "MATCH (c:Entity)-[:HAS_DEVICE_DAMAGE_TYPE]->(d:Entity) WHERE c.entity_type = 'ClaimRecord' RETURN DISTINCT d.id LIMIT 10",
        "keywords": ["damage", "device"],
    },
    {
        "id": 24,
        "category": "structured_claims",
        "question": "Do claim records include NPS scores from customers?",
        "cypher": "MATCH (c:Entity)-[r]->(n:Entity) WHERE c.entity_type = 'ClaimRecord' AND (type(r) CONTAINS 'NPS') RETURN type(r) AS relation, count(DISTINCT n) AS distinct_values LIMIT 10",
        "keywords": ["nps", "has_nps", "distinct_values"],
    },

    # --- Structured policies (3) ---
    {
        "id": 25,
        "category": "structured_policies",
        "question": "Are individual GEICO renters policies represented as entities?",
        "cypher": "MATCH (p:Entity) WHERE p.entity_type = 'PolicyRecord' RETURN count(p) AS policy_count",
        "keywords": ["policy_count"],
    },
    {
        "id": 26,
        "category": "structured_policies",
        "question": "Do policy records carry premium / cost / fee information?",
        "cypher": "MATCH (p:Entity)-[r]->(v:Entity) WHERE p.entity_type = 'PolicyRecord' AND (type(r) CONTAINS 'COST' OR type(r) CONTAINS 'PREMIUM' OR type(r) CONTAINS 'FEE') RETURN DISTINCT type(r) AS rel, v.id LIMIT 10",
        "keywords": ["cost", "premium", "fee", "amount"],
    },
    {
        "id": 27,
        "category": "structured_policies",
        "question": "Do policy records have effective or expiration dates?",
        "cypher": "MATCH (p:Entity)-[r]->(d:Entity) WHERE p.entity_type = 'PolicyRecord' AND (type(r) CONTAINS 'EFF' OR type(r) CONTAINS 'EXP' OR type(r) CONTAINS 'EFFECTIVE' OR type(r) CONTAINS 'EXPIR') RETURN DISTINCT type(r) AS rel, d.id LIMIT 10",
        "keywords": ["effective", "expiration", "date"],
    },

    # --- Surveys (NEW category, 3) ---
    {
        "id": 28,
        "category": "surveys",
        "question": "What channels were used for claim / survey responses?",
        "cypher": "MATCH (s:Entity)-[r]->(ch:Entity) WHERE type(r) IN ['HAS_CLAIM_CHANNEL','HAS_SURVEY_CHANNEL'] RETURN DISTINCT ch.id LIMIT 20",
        "keywords": ["web", "phone", "email", "channel"],
    },
    {
        "id": 29,
        "category": "surveys",
        "question": "What survey gateways / channels appear in the renters survey data?",
        "cypher": "MATCH (s:Entity)-[r]->(g:Entity) WHERE (type(r) CONTAINS 'GATEWAY' OR type(r) CONTAINS 'CHANNEL') AND s.entity_type IN ['SurveyRecord','ClaimRecord'] RETURN DISTINCT g.id LIMIT 20",
        "keywords": ["cwa", "email", "gateway", "web", "phone", "chat"],
    },
    {
        "id": 30,
        "category": "surveys",
        "question": "What distinct NPS scores were reported across surveys?",
        "cypher": "MATCH (s:Entity)-[r]->(n:Entity) WHERE type(r) IN ['HAS_NPS_SCORE','HAS_NPS'] RETURN DISTINCT n.id ORDER BY n.id LIMIT 20",
        "keywords": ["nps", "score"],
    },

    # --- Cross-source (5) ---
    {
        "id": 31,
        "category": "cross_source",
        "question": "Are claim records linked to policy numbers in the KG?",
        "cypher": "MATCH (c:Entity)-[:HAS_POLICY_NUMBER]->(pn:Entity) WHERE c.entity_type = 'ClaimRecord' RETURN count(DISTINCT pn) AS linked_policies",
        "keywords": ["policy", "linked_policies"],
    },
    {
        "id": 32,
        "category": "cross_source",
        "question": "Do different record types (survey, claim, policy) share the same organization name?",
        "cypher": "MATCH (a:Entity)-[:HAS_ORGANIZATION_NAME]->(o:Entity), (b:Entity)-[:HAS_ORGANIZATION_NAME]->(o) WHERE a.entity_type <> b.entity_type AND a <> b RETURN DISTINCT o.id LIMIT 10",
        "keywords": ["organization", "assurant", "geico"],
    },
    {
        "id": 33,
        "category": "cross_source",
        "question": "How many claims include both a device type and an NPS score?",
        "cypher": "MATCH (c:Entity)-[:HAS_DEVICE_TYPE]->(d:Entity), (c)-[r]->(n:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'NPS' RETURN count(DISTINCT c) AS claims_with_device_and_nps, count(DISTINCT d) AS distinct_devices LIMIT 10",
        "keywords": ["claims_with_device_and_nps", "distinct_devices"],
    },
    {
        "id": 34,
        "category": "cross_source",
        "question": "Are any Auto Service Contract procedures referenced by claim records?",
        "cypher": "MATCH (c:Entity)-[r]->(proc:Entity)-[:INSTANCE_OF]->(:Class) WHERE c.entity_type = 'ClaimRecord' AND (toLower(proc.id) CONTAINS 'roadside' OR toLower(proc.id) CONTAINS 'towing' OR toLower(proc.id) CONTAINS 'repair') RETURN DISTINCT proc.id LIMIT 10",
        "keywords": ["roadside", "towing", "repair", "procedure"],
    },
    {
        "id": 35,
        "category": "cross_source",
        "question": "Do survey records and claim records share organization-name values?",
        "cypher": "MATCH (s:Entity)-[:HAS_ORGANIZATION_NAME]->(o:Entity), (c:Entity)-[:HAS_ORGANIZATION_NAME]->(o) WHERE s.entity_type = 'SurveyRecord' AND c.entity_type = 'ClaimRecord' RETURN DISTINCT o.id LIMIT 10",
        "keywords": ["assurant", "geico", "organization"],
    },

    # --- Identity (1) ---
    {
        "id": 36,
        "category": "identity",
        "question": "Are there shared policy-number values linking survey and claim records?",
        "cypher": "MATCH (s:Entity)-[:HAS_POLICY_NUMBER]->(pn:Entity), (c:Entity)-[:HAS_POLICY_NUMBER]->(pn) WHERE s.entity_type <> c.entity_type RETURN count(DISTINCT pn) AS shared_policies",
        "keywords": ["shared_policies", "policy"],
    },

    # --- Claim lookup (3) ---
    {
        "id": 37,
        "category": "claim_lookup",
        "question": "What are the longest claim resolution times in the T-Mobile data?",
        "cypher": "MATCH (c:Entity)-[:HAS_TIME_TO_RESOLVE_HR]->(h:Entity) WHERE c.entity_type = 'ClaimRecord' WITH c, toFloat(h.id) AS hrs ORDER BY hrs DESC LIMIT 5 RETURN c.id AS claim, hrs",
        "keywords": ["hrs", "claim"],
    },
    {
        "id": 38,
        "category": "claim_lookup",
        "question": "What loss type accounts for the most claims?",
        "cypher": "MATCH (c:Entity)-[:HAS_LOSS_TYPE]->(l:Entity) WHERE c.entity_type = 'ClaimRecord' RETURN l.id AS loss_type, count(c) AS n ORDER BY n DESC LIMIT 5",
        "keywords": ["physical damage", "loss", "type"],
    },
    {
        "id": 39,
        "category": "claim_lookup",
        "question": "Which device types received the most claims?",
        "cypher": "MATCH (c:Entity)-[:HAS_DEVICE_TYPE]->(d:Entity) WHERE c.entity_type = 'ClaimRecord' RETURN d.id AS device, count(c) AS n ORDER BY n DESC LIMIT 5",
        "keywords": ["cellular", "phone", "device"],
    },

    # --- Policy lookup (1) ---
    {
        "id": 40,
        "category": "policy_lookup",
        "question": "Which organizations underwrite or administer the renters policies?",
        "cypher": "MATCH (p:Entity)-[:HAS_ORGANIZATION_NAME]->(o:Entity) WHERE p.entity_type = 'PolicyRecord' RETURN DISTINCT o.id LIMIT 10",
        "keywords": ["assurant", "geico", "organization"],
    },
]
