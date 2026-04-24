"""Emory Spring 2026 query-evaluation tasks — derived from source CSVs + PDF,
not from inspecting the KG.

Source inputs for Emory:
  - synthetic_data_sample__geicorenterssurveysample.csv   (100 rows, 40 cols)
  - synthetic_data_sample_geicorenterscancelsurvey.csv    (100 rows, 45 cols)
  - synthetic_data_sample_geicorentersclaims.csv          (100 rows, 23 cols)
  - synthetic_data_sample_geicorenterspoliciesdetails.csv (100 rows, 187 cols)
  - synthetic_data_sample_geicorenterssurvey.csv          (100 rows, 40 cols)
  - synthetic_data_sample_tmobilechatsurveysample.csv     (100 rows, 81 cols)
  - synthetic_data_sample_tmobileclaimsample.csv          (100 rows, 51 cols)
  - synthetic_data_sample_tmobilesurveysample.csv         (100 rows, 72 cols)
  - Auto_Service_form_masked.pdf                          (24-page Assurant form)

Each question asks about a column, value, or section that exists in the raw
sources. Cypher uses `type(r) CONTAINS 'TOKEN'` or `toLower(n.id) CONTAINS …`
so queries stay robust to minor relation-name changes across runs.

Keyword-match scoring: passes if the Cypher returns any row AND any keyword
appears in the stringified result. Keywords are drawn from the actual CSV
values/column names or PDF section titles — not from KG inspection.
"""
from __future__ import annotations

EVAL_TASKS_EMORY: list[dict] = [
    # --- Coverage (6) — Auto Service Contract PDF + T-Mobile claim coverage + GEICO renters coverage ---
    {
        "id": 1, "category": "coverage",
        "question": "What parts does the Auto Service Contract cover under Section II?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'covered part' OR toLower(n.id) CONTAINS 'engine' OR toLower(n.id) CONTAINS 'transmission' OR toLower(n.id) CONTAINS 'cylinder' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["engine", "transmission", "cylinder", "part"],
    },
    {
        "id": 2, "category": "coverage",
        "question": "What emergency roadside / towing services are included in the Additional Benefits section?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'roadside' OR toLower(n.id) CONTAINS 'towing' OR toLower(n.id) CONTAINS 'emergency' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["roadside", "towing", "emergency"],
    },
    {
        "id": 3, "category": "coverage",
        "question": "What device types are covered under T-Mobile device claims (DEVICE_TYPE column)?",
        "cypher": "MATCH (c:Entity)-[r]->(d:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'DEVICE' AND type(r) CONTAINS 'TYPE' RETURN DISTINCT d.id LIMIT 20",
        "keywords": ["cellular", "phone", "smart"],
    },
    {
        "id": 4, "category": "coverage",
        "question": "What coverage-type values appear in T-Mobile claims (COVERAGE_TYPE column: MISSING OR LOST, etc.)?",
        "cypher": "MATCH (c:Entity)-[r]->(v:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'COVERAGE' AND type(r) CONTAINS 'TYPE' RETURN DISTINCT v.id LIMIT 20",
        "keywords": ["missing", "lost", "damage", "coverage"],
    },
    {
        "id": 5, "category": "coverage",
        "question": "What personal-property and liability coverage amounts (COVAMT_PERS / COVAMT_LIAB) are recorded on GEICO renters policies?",
        "cypher": "MATCH (p:Entity)-[r]->(v:Entity) WHERE p.entity_type = 'PolicyRecord' AND (type(r) CONTAINS 'COVAMT' OR type(r) CONTAINS 'PERS' OR type(r) CONTAINS 'LIAB') RETURN DISTINCT type(r) AS rel, v.id LIMIT 10",
        "keywords": ["covamt", "pers", "liab", "10000", "100000"],
    },
    {
        "id": 6, "category": "coverage",
        "question": "What endorsements does the GEICO renters policy schema list (earthquake, jewelry, pet damage, water damage, etc.)?",
        "cypher": "MATCH (p:Entity)-[r]->(v:Entity) WHERE p.entity_type = 'PolicyRecord' AND type(r) CONTAINS 'ENDORSEMENT' RETURN DISTINCT type(r) AS endorsement LIMIT 20",
        "keywords": ["earthquake", "jewelry", "pet", "water", "endorsement"],
    },

    # --- Exclusions (3) — Auto service contract limitations + void conditions ---
    {
        "id": 7, "category": "exclusions",
        "question": "What parts are excluded from the Auto Service Contract (brake pads, spark plugs, wheel covers)?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'brake pad' OR toLower(n.id) CONTAINS 'spark plug' OR toLower(n.id) CONTAINS 'wheel cover' OR toLower(n.id) CONTAINS 'exclud' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["brake", "spark", "wheel", "excluded"],
    },
    {
        "id": 8, "category": "exclusions",
        "question": "What conditions void the service contract (odometer tamper, misrepresentation, unsafe condition)?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'odometer' OR toLower(n.id) CONTAINS 'misrepresent' OR toLower(n.id) CONTAINS 'unsafe' OR toLower(n.id) CONTAINS 'void' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["odometer", "misrepresentation", "unsafe"],
    },
    {
        "id": 9, "category": "exclusions",
        "question": "Does the service contract exclude commercial or business use?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'commercial' OR toLower(n.id) CONTAINS 'business' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["commercial", "business"],
    },

    # --- Policy terms (6) ---
    {
        "id": 10, "category": "policy_terms",
        "question": "What is the Service Contract Administrative Fee?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'administrative fee' OR toLower(n.id) CONTAINS 'service contract fee' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["administrative", "fee", "service"],
    },
    {
        "id": 11, "category": "policy_terms",
        "question": "How does pro-rata refund work (unused months/miles, whichever is less)?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'pro-rata' OR toLower(n.id) CONTAINS 'unused' OR toLower(n.id) CONTAINS 'refund' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["pro-rata", "unused", "refund", "month", "mile"],
    },
    {
        "id": 12, "category": "policy_terms",
        "question": "What arbitration / umpire process resolves disputes under the service contract?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'arbitrat' OR toLower(n.id) CONTAINS 'umpire' OR toLower(n.id) CONTAINS 'dispute' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["arbitration", "umpire", "dispute"],
    },
    {
        "id": 13, "category": "policy_terms",
        "question": "What is the lienholder-priority rule for refunds?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'lienholder' OR toLower(n.id) CONTAINS 'priority' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["lienholder", "priority"],
    },
    {
        "id": 14, "category": "policy_terms",
        "question": "What payment plans / term lengths does the GEICO renters policy schema record (PAYPLAN, TERM columns)?",
        "cypher": "MATCH (p:Entity)-[r]->(v:Entity) WHERE p.entity_type = 'PolicyRecord' AND (type(r) CONTAINS 'PAYPLAN' OR type(r) CONTAINS 'PAYCODE' OR type(r) = 'HAS_TERM' OR type(r) CONTAINS 'NUMPMTS') RETURN DISTINCT type(r) AS rel, v.id LIMIT 10",
        "keywords": ["payplan", "term", "pay"],
    },
    {
        "id": 15, "category": "policy_terms",
        "question": "Does Section IX describe state-specific amendments to the contract?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'state amendment' OR (toLower(n.id) CONTAINS 'state' AND toLower(n.id) CONTAINS 'law') RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["state", "amendment", "law"],
    },

    # --- Claims (3) ---
    {
        "id": 16, "category": "claims",
        "question": "What claim statuses appear in the T-Mobile claim data (CLAIM_STATUS column)?",
        "cypher": "MATCH (c:Entity)-[r]->(s:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'STATUS' RETURN DISTINCT s.id LIMIT 20",
        "keywords": ["authorized", "closed", "pending", "denied", "status"],
    },
    {
        "id": 17, "category": "claims",
        "question": "What claim channels are recorded (CLAIM_CHANNEL: Web, DISPATCH, EzPass, IVR)?",
        "cypher": "MATCH (c:Entity)-[r]->(ch:Entity) WHERE (c.entity_type = 'ClaimRecord' OR c.entity_type = 'SurveyRecord') AND type(r) CONTAINS 'CHANNEL' RETURN DISTINCT ch.id LIMIT 20",
        "keywords": ["web", "dispatch", "ezpass", "ivr"],
    },
    {
        "id": 18, "category": "claims",
        "question": "What cause-of-loss values appear in GEICO renters claims (CAUSE_OF_LOSS column)?",
        "cypher": "MATCH (c:Entity)-[r]->(cl:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'CAUSE' RETURN DISTINCT cl.id LIMIT 20",
        "keywords": ["other", "water", "fire", "theft", "loss", "cause"],
    },

    # --- Definitions (2) ---
    {
        "id": 19, "category": "definitions",
        "question": "What does Section I of the contract define as 'breakdown' or 'covered part'?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'breakdown' OR toLower(n.id) CONTAINS 'covered part' OR toLower(n.id) CONTAINS 'key term' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["breakdown", "part", "covered"],
    },
    {
        "id": 20, "category": "definitions",
        "question": "Does the contract define what constitutes a 'lienholder' or 'named insured'?",
        "cypher": "MATCH (n) WHERE toLower(n.id) CONTAINS 'lienholder' OR toLower(n.id) CONTAINS 'named insured' OR toLower(n.id) CONTAINS 'policyholder' RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["lienholder", "insured", "holder"],
    },

    # --- Structured claims (4) — T-Mobile + GEICO renters claim columns ---
    {
        "id": 21, "category": "structured_claims",
        "question": "Are individual T-Mobile / GEICO renters claim records represented as entities?",
        "cypher": "MATCH (c:Entity) WHERE c.entity_type = 'ClaimRecord' RETURN count(c) AS claim_count",
        "keywords": ["claim_count"],
    },
    {
        "id": 22, "category": "structured_claims",
        "question": "Do T-Mobile claim records include claim-loss-date information (CLAIM_LOSS_DATE column)?",
        "cypher": "MATCH (c:Entity)-[r]->(d:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'LOSS' AND type(r) CONTAINS 'DATE' RETURN type(r) AS rel, count(DISTINCT d) AS distinct_dates LIMIT 5",
        "keywords": ["loss", "date", "distinct_dates"],
    },
    {
        "id": 23, "category": "structured_claims",
        "question": "Do T-Mobile claims capture total claim time (TOTAL_CLAIM_TIME column)?",
        "cypher": "MATCH (c:Entity)-[r]->(t:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'TOTAL' AND type(r) CONTAINS 'TIME' RETURN type(r) AS rel, count(DISTINCT t) AS distinct_values LIMIT 5",
        "keywords": ["total", "time", "distinct_values"],
    },
    {
        "id": 24, "category": "structured_claims",
        "question": "Do GEICO renters claims carry TOTAL_LOSSES dollar amounts?",
        "cypher": "MATCH (c:Entity)-[r]->(v:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'TOTAL_LOSSES' RETURN count(DISTINCT v) AS distinct_amounts",
        "keywords": ["distinct_amounts"],
    },

    # --- Structured policies (3) ---
    {
        "id": 25, "category": "structured_policies",
        "question": "Are individual GEICO renters policies represented as entities?",
        "cypher": "MATCH (p:Entity) WHERE p.entity_type = 'PolicyRecord' RETURN count(p) AS policy_count",
        "keywords": ["policy_count"],
    },
    {
        "id": 26, "category": "structured_policies",
        "question": "Do renters policies preserve gross written premium (GWP) and fee data (FEES, TAXES)?",
        "cypher": "MATCH (p:Entity)-[r]->(v:Entity) WHERE p.entity_type = 'PolicyRecord' AND (type(r) CONTAINS 'GWP' OR type(r) CONTAINS 'NWP' OR type(r) CONTAINS 'FEE' OR type(r) CONTAINS 'TAX') RETURN DISTINCT type(r) AS rel LIMIT 15",
        "keywords": ["gwp", "nwp", "fee", "tax"],
    },
    {
        "id": 27, "category": "structured_policies",
        "question": "Do renters policies include effective / expiration / issue dates (EFF_DATE, EXP_DATE, POLICY_ISSUE_INVOICED)?",
        "cypher": "MATCH (p:Entity)-[r]->(d:Entity) WHERE p.entity_type = 'PolicyRecord' AND (type(r) CONTAINS 'EFF' OR type(r) CONTAINS 'EXP' OR type(r) CONTAINS 'ISSUE' OR type(r) CONTAINS 'DATE') RETURN DISTINCT type(r) AS rel LIMIT 15",
        "keywords": ["eff", "exp", "issue", "date"],
    },

    # --- Surveys (3) — GEICO renters survey + T-Mobile survey NPS/CSAT fields ---
    {
        "id": 28, "category": "surveys",
        "question": "What NPS scores and categories (NPS_CATEGORY: Promoter/Passive/Detractor) appear in the surveys?",
        "cypher": "MATCH (s:Entity)-[r]->(v:Entity) WHERE type(r) CONTAINS 'NPS' RETURN DISTINCT type(r) AS rel, v.id LIMIT 20",
        "keywords": ["promoter", "detractor", "passive", "nps"],
    },
    {
        "id": 29, "category": "surveys",
        "question": "Do surveys record CSAT values (DEVICE_CSAT, AGENT_CSAT, REPAIR_CSAT, etc.)?",
        "cypher": "MATCH (s:Entity)-[r]->(v:Entity) WHERE type(r) CONTAINS 'CSAT' RETURN DISTINCT type(r) AS rel LIMIT 20",
        "keywords": ["csat", "device", "agent", "repair"],
    },
    {
        "id": 30, "category": "surveys",
        "question": "What renters-cancellation reasons appear in the cancel survey (RENTERS_MAIN_CANCELLATION_REASON, e.g. 'I purchased a new home')?",
        "cypher": "MATCH (s:Entity)-[r]->(v:Entity) WHERE s.entity_type = 'SurveyRecord' AND (type(r) CONTAINS 'CANCEL' OR type(r) CONTAINS 'REASON') RETURN DISTINCT v.id LIMIT 20",
        "keywords": ["purchased", "home", "cancel", "reason", "website"],
    },

    # --- Cross-source (5) — organization-name / policy-number / device sharing ---
    {
        "id": 31, "category": "cross_source",
        "question": "Do claim records reference the same organization name (Assurant Global Home, GEICO) as survey records?",
        "cypher": "MATCH (a:Entity)-[:HAS_ORGANIZATION_NAME]->(o:Entity), (b:Entity)-[:HAS_ORGANIZATION_NAME]->(o) WHERE a.entity_type <> b.entity_type RETURN DISTINCT o.id LIMIT 10",
        "keywords": ["assurant", "geico"],
    },
    {
        "id": 32, "category": "cross_source",
        "question": "Are policy numbers preserved across claim, survey, and policy sources?",
        "cypher": "MATCH (rec:Entity)-[r]->(pn:Entity) WHERE type(r) CONTAINS 'POLICY' AND (type(r) CONTAINS 'NUMBER' OR type(r) CONTAINS 'NO') RETURN DISTINCT rec.entity_type AS src, count(DISTINCT pn) AS distinct_policy_numbers LIMIT 5",
        "keywords": ["distinct_policy_numbers"],
    },
    {
        "id": 33, "category": "cross_source",
        "question": "Are T-Mobile device types referenced from both claim records and survey records?",
        "cypher": "MATCH (a:Entity)-[r1]->(d:Entity), (b:Entity)-[r2]->(d) WHERE a.entity_type <> b.entity_type AND type(r1) CONTAINS 'DEVICE' AND type(r2) CONTAINS 'DEVICE' RETURN DISTINCT d.id LIMIT 10",
        "keywords": ["cellular", "phone", "device"],
    },
    {
        "id": 34, "category": "cross_source",
        "question": "Does the Auto Service Contract share any concept names with T-Mobile claim coverage types?",
        "cypher": "MATCH (n:Entity) WHERE (toLower(n.id) CONTAINS 'coverage' OR toLower(n.id) CONTAINS 'service') AND n.entity_type IS NULL RETURN DISTINCT n.id LIMIT 20",
        "keywords": ["coverage", "service"],
    },
    {
        "id": 35, "category": "cross_source",
        "question": "Are manufacturer names (Samsung, Apple) shared across T-Mobile claim and survey records?",
        "cypher": "MATCH (a:Entity)-[r1]->(m:Entity), (b:Entity)-[r2]->(m) WHERE a.entity_type <> b.entity_type AND type(r1) CONTAINS 'MANUFACTURER' AND type(r2) CONTAINS 'MANUFACTURER' RETURN DISTINCT m.id LIMIT 10",
        "keywords": ["samsung", "apple", "manufacturer"],
    },

    # --- Identity (1) ---
    {
        "id": 36, "category": "identity",
        "question": "Do policy records and claim records share any common policy-number identities?",
        "cypher": "MATCH (p:Entity)-[r1]->(pn:Entity), (c:Entity)-[r2]->(pn) WHERE p.entity_type = 'PolicyRecord' AND c.entity_type = 'ClaimRecord' AND type(r1) CONTAINS 'POLICY' AND type(r1) CONTAINS 'NUMBER' AND type(r2) CONTAINS 'POLICY' AND type(r2) CONTAINS 'NUMBER' RETURN count(DISTINCT pn) AS shared_policy_numbers",
        "keywords": ["shared_policy_numbers"],
    },

    # --- Claim lookup (3) ---
    {
        "id": 37, "category": "claim_lookup",
        "question": "Which device types receive the most T-Mobile claims (DEVICE_TYPE most-frequent)?",
        "cypher": "MATCH (c:Entity)-[r]->(d:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'DEVICE' AND type(r) CONTAINS 'TYPE' RETURN d.id AS device, count(c) AS n ORDER BY n DESC LIMIT 5",
        "keywords": ["cellular", "phone"],
    },
    {
        "id": 38, "category": "claim_lookup",
        "question": "Which loss types (LOSS_TYPE: Physical Damage, Missing or Lost, Malfunction) are most common?",
        "cypher": "MATCH (c:Entity)-[r]->(l:Entity) WHERE c.entity_type = 'ClaimRecord' AND type(r) CONTAINS 'LOSS' AND type(r) CONTAINS 'TYPE' RETURN l.id AS loss_type, count(c) AS n ORDER BY n DESC LIMIT 5",
        "keywords": ["physical damage", "missing", "lost"],
    },
    {
        "id": 39, "category": "claim_lookup",
        "question": "Which US states (CLAIMANT_STATE column) are represented in GEICO renters claims?",
        "cypher": "MATCH (c:Entity)-[r]->(s:Entity) WHERE c.entity_type = 'ClaimRecord' AND (type(r) CONTAINS 'STATE' OR type(r) CONTAINS 'CLAIMANT_STATE') RETURN DISTINCT s.id LIMIT 20",
        "keywords": ["tx", "ca", "fl", "ny", "pa", "state"],
    },

    # --- Policy lookup (1) ---
    {
        "id": 40, "category": "policy_lookup",
        "question": "Which counties (COUNTY column) and risk states (RISK_ST column) are represented in GEICO renters policies?",
        "cypher": "MATCH (p:Entity)-[r]->(v:Entity) WHERE p.entity_type = 'PolicyRecord' AND (type(r) CONTAINS 'COUNTY' OR type(r) CONTAINS 'RISK_ST' OR type(r) CONTAINS 'STATE') RETURN DISTINCT type(r) AS rel, v.id LIMIT 20",
        "keywords": ["county", "state", "hillsborough"],
    },
]
