# Fingerprint-Based Ontology Induction (FBI) — Design Spec

**Date**: 2026-04-16
**Author**: Sam (Xiaofei Wang)
**Project**: SEAF-KG Zone 3 Redesign
**Branch**: `claude/suspicious-sanderson`

---

## 1. Problem Statement

Zone 3 (ontology induction) currently produces flat, coarse classes (9-11 total) and fails to discover subclasses — even when the raw data clearly contains evidence for them (e.g., different policy types, endorsement subtypes, claim specializations).

**Root causes:**
1. **Information loss**: Zone 2 stores source filename, confidence, source_type, and section hierarchy in Neo4j, but Zone 3's `graph_cache.py` drops all of it. Zone 3 only sees entity names + 5 truncated relation summaries.
2. **Wrong signal source**: Current approaches (SV-LOI, Leiden) try to classify entities by asking the LLM "what type is this entity?" — but a local 72B model produces inconsistent answers. The real class signal is in file-level structure (filenames, column headers, section headings) which is never examined.
3. **No multi-level hierarchy discovery**: Current methods produce one flat level of classes. The data contains 3+ levels of hierarchy encoded in header prefix structure (e.g., `ENDORSEMENT` → `ENDORSEMENT_EARTHQUAKE` → `ENDORSEMENT_EARTHQUAKE_CODE`).

## 2. Design Constraints

1. **Domain-agnostic**: Zero hardcoded domain terms. Must work identically on flood insurance, auto insurance, pet insurance, or any other LOB with zero code changes.
2. **No fine-tuning**: Uses qwen2.5:72b as-is via Ollama. No training data, no parameter updates.
3. **No leakage**: Never reference Riskine, ACORD, or any evaluation ontology in prompts or logic.
4. **Extensible**: When new LOB files are added, the pipeline discovers new classes automatically. No schema changes needed.
5. **LLM-minimal**: The 72B local model is an assistant (naming, grouping ambiguous items), not the driver. All structural decisions are algorithmic.
6. **Self-contained prompts**: Every LLM prompt must contain all information the model needs to answer. No references to external standards or assumed domain knowledge.

## 3. Data Sources

Working directory: `/data/Emory_Spring2026/`

### CSV Files (8 files)
| File | LOB | Function | Columns |
|------|-----|----------|---------|
| `geicorenterspoliciesdetails.csv` | GEICO Renters | Policy | 187 |
| `geicorentersclaims.csv` | GEICO Renters | Claims | 23 |
| `geicorenterssurvey.csv` | GEICO Renters | Survey | 40 |
| `_geicorenterssurveysample.csv` | GEICO Renters | Survey | 40 |
| `geicorenterscancelsurvey.csv` | GEICO Renters | Cancel Survey | 45 |
| `tmobileclaimsample.csv` | T-Mobile Device | Claims | 51 |
| `tmobilesurveysample.csv` | T-Mobile Device | Survey | 72 |
| `tmobilechatsurveysample.csv` | T-Mobile Device | Chat Survey | 81 |

### PDF Files (1 file)
| File | LOB | Function |
|------|-----|----------|
| `Auto_Service_form_masked.pdf` | Auto Service | Vehicle Service Contract |

### TXT Files (5 files, in `web_policies/`)
| File | LOB | Function |
|------|-----|----------|
| `tmobile_p360_terms.txt` | T-Mobile | Protection 360 Terms |
| `tmobile_standard_device_protection_terms.txt` | T-Mobile | Standard Protection Terms |
| `tmobile_device_protection_claim.txt` | T-Mobile | Claim Process |
| `geico_renters_personal_liability.txt` | GEICO | Personal Liability Coverage |
| `fastclaim_metro_php_plan_coverage.txt` | Metro | PHP Plan Coverage |

## 4. Architecture Overview

```
Raw Files ──→ Phase 1 (Header Intelligence)
                  │
                  ├──→ Phase 2 (Multi-Iteration Class Discovery)
                  │         │
                  │         ├──→ Phase 3 (Relationship Discovery)
                  │         │         │
                  │         │         ▼
                  │    Zone 2 (Extract) ──→ Phase 4 (Entity Assignment)
                  │                              │
                  │                              ▼
                  │                         Neo4j Ontology
                  │
                  └── Phases 1-3 do NOT depend on Zone 2
```

**Key architectural decision**: Phases 1-3 work directly on raw file metadata. They do not need Zone 2's extracted triples. Only Phase 4 uses Zone 2 output. This means class discovery is independent of extraction quality.

## 5. Phase 1: Header Intelligence

**Purpose**: Make cryptic column headers and filenames readable for both the LLM and the clustering algorithm.

**Input**: Raw data files (CSV, PDF, TXT)
**Output**: Per-file metadata record with expanded headers and filename tokens

### Step 1a: Extract Raw Headers (Algorithmic)

For each file type:
- **CSV**: Read the first row (column headers)
- **PDF**: Extract section headings from document structure (e.g., "VEHICLE INFORMATION", "KEY TERMS", "WHAT THIS SERVICE CONTRACT COVERS")
- **TXT**: Extract section headings and defined terms (terms in bold or quotes, section dividers)

Strip universal audit columns that appear in every file and carry no domain meaning:
- Pattern: `BI_CREATED_DT`, `BI_CREATED_BY`, `BI_MODIFIED_DT`, `BI_MODIFIED_BY`
- Detection: any column appearing in ALL files with identical name → likely audit/metadata, flag for review

### Step 1b: LLM Expands Cryptic Headers (Batched, ~2-3 calls)

Many headers are abbreviated: `SB`, `MCO`, `PCO`, `UW_POS5`, `COVAMT_PERS`, `COVLIAB_DED2`, `CXL_YYMM`.

```
Prompt template:
"Below is a list of abbreviated column headers from a business database.
For each one, expand the abbreviation into its full descriptive name.
If the meaning is unclear, make your best guess based on the abbreviation pattern.

Format your response as:
ABBREVIATION → Full Name

Headers:
{batch of 40-60 headers}
"
```

Example output:
```
SB → Sub-Business
MCO → Master Company Organization
COVAMT_PERS → Coverage Amount Personal Property
COVLIAB_DED2 → Coverage Liability Deductible 2
ENDORSEMENT_SCHED_JEWELRY_NWP → Endorsement Scheduled Jewelry Net Written Premium
CXL_YYMM → Cancellation Year-Month
POLNO → Policy Number
```

Batch size: ~50 headers per call. Total calls: ceil(unique_headers / 50).

### Step 1c: LLM Parses Filenames (1 call)

Filenames can have arbitrary conventions. The LLM extracts semantic tokens.

```
Prompt template:
"Below are filenames from a data directory.
For each filename, extract the meaningful domain tokens.
Ignore technical prefixes like 'synthetic_data_sample_' or suffixes like '_masked'.
Ignore file extensions.

Format: filename → [token1, token2, ...]

Filenames:
{list of filenames}
"
```

Example output:
```
synthetic_data_sample_geicorentersclaims → [geico, renters, claims]
synthetic_data_sample_tmobileclaimsample → [tmobile, claim]
Auto_Service_form_masked → [auto, service]
tmobile_p360_terms → [tmobile, p360, terms]
geico_renters_personal_liability → [geico, renters, personal, liability]
```

### Step 1d: PDF/TXT Section Extraction (Algorithmic + Light LLM)

For PDF files: extract section headings from document structure.
For TXT files: extract section headings (lines in ALL CAPS, or bold markers) and defined terms (quoted terms with definitions).

Example from `Auto_Service_form_masked.pdf`:
```
Sections: [VEHICLE INFORMATION, MEMBER INFORMATION, PRODUCER INFORMATION,
           LIENHOLDER INFORMATION, VEHICLE SERVICE CONTRACT INFORMATION,
           SERVICE CONTRACT PRICE, KEY TERMS, WHAT THIS SERVICE CONTRACT COVERS,
           ADDITIONAL BENEFITS, MAINTENANCE AND PARTS NOT COVERED,
           SERVICE CONTRACT LIMITATIONS, WHAT TO DO IF REPAIRS ARE NEEDED,
           YOUR RESPONSIBILITIES, GENERAL PROVISIONS, STATE AMENDMENTS]

Defined Terms: [Breakdown, Commercial Purposes, Cost, Deductible, Producer,
                Purchase Date, Repair Facility, Service Contract Price,
                Term Miles, Term Months, Vehicle, Warranty, Worn]
```

### Phase 1 Output Format

Per-file metadata record:
```json
{
  "file": "synthetic_data_sample_geicorenterspoliciesdetails.csv",
  "file_type": "csv",
  "filename_tokens": ["geico", "renters", "policies", "details"],
  "headers_raw": ["MCO", "POLNO", "COVAMT_PERS", "..."],
  "headers_expanded": ["Master Company Organization", "Policy Number",
                        "Coverage Amount Personal Property", "..."],
  "record_count": 500
}
```

## 6. Phase 2: Multi-Iteration Class Discovery

**Purpose**: Discover ontology classes at multiple hierarchy levels from header structure.

**Input**: Per-file metadata from Phase 1
**Output**: Multi-level class hierarchy with class-to-file mappings

### Iteration 1: Prefix-Based Grouping (Within Each File — Algorithmic)

Build a prefix trie from all headers in each file. Identify prefix groups where ≥3 headers share a common prefix.

**Algorithm**:
1. For each file, collect all expanded headers
2. Build prefix trie
3. At each trie node, if the subtree has ≥3 leaf headers, mark it as a candidate class
4. Record the prefix (= class identity) and the suffixes (= class attributes)
5. Detect **sibling patterns**: if multiple prefix groups share the same set of suffixes, they are siblings under a common parent

**Example from `geicorenterspoliciesdetails.csv`** (187 columns):

```
ENDORSEMENT_* (23 subtypes × 4 attributes each = 92 headers)
  All subtypes share suffix pattern: {CODE, NWP, DED_CODE, EXPOSURE}
  
  Subtypes found:
    ENDORSEMENT_EARTHQUAKE_*
    ENDORSEMENT_SCHED_JEWELRY_*
    ENDORSEMENT_OTHER_PROPERTY_*
    ENDORSEMENT_UNSCHED_JEWELRY_*
    ENDORSEMENT_ADD_*
    ENDORSEMENT_WORKCOMP_RI_*
    ENDORSEMENT_SEWER_BACKUP_*
    ENDORSEMENT_DAYCARE_*
    ENDORSEMENT_IUI_SINGLE_*
    ENDORSEMENT_IUI_JOINT_*
    ENDORSEMENT_MOLD_PROP_RI_*
    ENDORSEMENT_MOLD_LIAB_RI_*
    ENDORSEMENT_ID_THEFT_*
    ENDORSEMENT_PET_DAMAGE_*
    ENDORSEMENT_ALT_LOC_EXP_*
    ENDORSEMENT_WATER_DAMAGE_*
    ENDORSEMENT_WATERGARD_*
    ENDORSEMENT_IUI_*
    ENDORSEMENT_WORKCOMP_ERL_*
    ENDORSEMENT_MOLD_INCR_ERL_*
    ... (23 total)

  → Parent class: "Endorsement" (from shared prefix)
  → 23 child classes (from prefix subtypes)
  → 4 shared attributes per child (from suffix pattern)

COV* (10 headers)
  COVPROP_* → {CODE1, PREM1, DED1, AINS1}
  COVLIAB_* → {CODE2, PREM2, DED2, AINS2}
  COVAMT_*  → {PERS, LIAB}
  
  → Parent class: "Coverage"
  → 3 child classes: PropertyCoverage, LiabilityCoverage, CoverageAmount

CLAIM_* (from tmobileclaimsample.csv, 14 headers)
  CLAIM_STATUS, CLAIM_LOSS_DATE, CLAIM_OPEN_PERIOD, CLAIM_CHANNEL,
  CLAIM_ISSUE, CLAIM_FULFILLMENT_TYPE, CLAIM_PENDING_STATUS_DATE,
  CLAIM_IN_REVIEW_STATUS_DATE, CLAIM_CLOSED_STATUS_DATE,
  CLAIM_APPROVED_DATE, CLAIM_AUTHORIZED_DATE, CLAIM_DENIED_DATE, ...
  
  → Class: "Claim" with 14 attributes

REPAIR_* (from tmobileclaimsample.csv)
  REPAIR_STARTED_DATE, REPAIR_COMPLETED_DATE, REPAIR_LOCATION
  → Class: "Repair"

SHIPPED_* (from tmobileclaimsample.csv)
  SHIPPED_SERIAL_NUMBER, SHIPPED_MANUFACTURER, SHIPPED_MODEL, SHIPPED_SERIAL
  → Class: "Shipment"
```

### Iteration 2: Semantic Grouping of Ungrouped Headers (LLM-Assisted)

Headers without clear shared prefixes need LLM grouping. ~1 call per file with ungrouped headers.

```
Prompt template:
"Below are column headers from a single data file that don't share
obvious naming prefixes with each other.

Group them by the concept they describe. Each group should represent
one coherent business concept.

Format:
ConceptName: [Header1, Header2, Header3]

Headers:
{list of ungrouped expanded headers}
"
```

Example input (from geicorenterspoliciesdetails.csv ungrouped):
```
[Insured Name, Location Address Line 1, Location Address Line 2,
 Location Address Line 3, Location Zip, Risk State, County, Territory,
 Policy Number, Previous Policy, Parent Policy, Policy Company Organization,
 Effective Date, Expiration Date, Cancellation Date, Original Year-Month,
 Gross Written Premium, Net Written Premium, Fees, Taxes, Credit,
 Payment Plan, Payment Code, Billing Date, Billing Indicator,
 Agency, Agency Name, Earned, Term, Number of Payments]
```

Example output:
```
Person: [Insured Name]
Address: [Location Address Line 1, Location Address Line 2,
          Location Address Line 3, Location Zip, Risk State, County, Territory]
PolicyIdentification: [Policy Number, Previous Policy, Parent Policy,
                        Policy Company Organization]
PolicyDates: [Effective Date, Expiration Date, Cancellation Date,
              Original Year-Month]
Financial: [Gross Written Premium, Net Written Premium, Fees, Taxes,
            Credit, Earned]
Billing: [Payment Plan, Payment Code, Billing Date, Billing Indicator,
          Term, Number of Payments]
Agency: [Agency, Agency Name]
```

### Iteration 3: Cross-File Class Merging

Compare prefix groups and semantic groups across ALL files. Groups with matching or overlapping headers across files → shared parent class. Unique headers within the merged group → evidence for subclasses.

**Algorithm**:
1. Collect all candidate classes from all files (from Iterations 1 and 2)
2. For each pair of candidate classes from different files:
   a. Compute header overlap (Jaccard on expanded header names, with fuzzy matching for synonyms)
   b. If overlap > 0.3 → candidates are related, merge into shared parent
3. Shared headers → define parent class attributes
4. Unique-to-file headers → define what makes each file's version a distinct subclass
5. If overlap < 0.1 → independent classes (no parent-child relationship)

**Example: "Claim" class merging**:
```
geicorentersclaims.csv:
  CLAIM_* = {CLAIM_STATUS, CLAIM_MONTH_ID, CLAIM_CYCLE_TIME_NUM, CLAIM_CYCLE_TIME_DEN}
  Plus: {CAUSE_OF_LOSS, IS_CAT, CAT_CLAIMS, TOTAL_LOSSES, AUTHORIZED_CLAIMS,
         CLAIMS_WITHOUT_PAYMENT, CLOSED_CLAIMS, REPORTED_CLAIMS, CWA_OR_CWOP}

tmobileclaimsample.csv:
  CLAIM_* = {CLAIM_STATUS, CLAIM_NUMBER, CLAIM_LOSS_DATE, CLAIM_OPEN_DATE,
             CLAIM_CHANNEL, CLAIM_ISSUE, CLAIM_FULFILLMENT_TYPE, ...}
  Plus: {DEVICE_DAMAGE_TYPE, DEDUCTIBLE_AMOUNT, COVERAGE_TYPE,
         CLAIMED_MANUFACTURER, REPAIR_*, SHIPPED_*, AUTH_TO_*, ...}

Shared (→ parent "Claim"): {CLAIM_STATUS, CLAIM_NUMBER-like ID}
Unique to GEICO: {CAUSE_OF_LOSS, IS_CAT, TOTAL_LOSSES, ...} → property-loss characteristics
Unique to T-Mobile: {DEVICE_DAMAGE_TYPE, FULFILLMENT_TYPE, REPAIR_*, ...} → device characteristics
→ Two subclasses under shared "Claim" parent
```

**Example: "Survey" class merging**:
```
GEICO surveys: {SURVEY_ID, SURVEY_NAME, Q_NPS, A_NPS, NPS_CATEGORY,
                A_AGENT_COMMUNICATION, A_AGENT_COURTEOUS, A_AGENT_OSAT, ...}
T-Mobile surveys: {GATEWAY_SURVEY_ID, NPS_SCORE, NPS_CATEGORY,
                   DEVICE_CSAT, REPAIR_CSAT, CLAIM_FILING_CSAT, *_CES, ...}
Cancel survey: {SURVEY_ID, RENTERS_MAIN_CANCELLATION_REASON,
               RENTERS_CANCEL_*, RENTER_*, BILLING_CANCELLATION_REASON, ...}

Shared: {survey ID, NPS score/category}
GEICO unique: {A_AGENT_* agent satisfaction scores}
T-Mobile unique: {*_CSAT, *_CES satisfaction/effort scores}
Cancel unique: {CANCELLATION_*, CANCEL_* reasons}
→ 3-4 subclasses under shared "Survey" parent
```

### Iteration 4: Hierarchy Assembly + LLM Naming

Assemble all iterations into a tree. LLM names each class given its evidence.

```
Prompt template:
"I have groups of column headers from business data files.
Each group represents a distinct concept in the data.

For each group, provide a short name (1-2 words) that describes
what concept these columns represent.

Group 1 (appears in multiple files, shared columns):
  {CLAIM_STATUS, CLAIM_NUMBER}
  
Group 1a (only in files about [geico, renters]):
  adds: {CAUSE_OF_LOSS, IS_CATASTROPHE, CATASTROPHE_CLAIMS, TOTAL_LOSSES,
         AUTHORIZED_CLAIMS, CLAIMS_WITHOUT_PAYMENT, CLOSED_CLAIMS}
  
Group 1b (only in files about [tmobile, device]):
  adds: {DEVICE_DAMAGE_TYPE, FULFILLMENT_TYPE, COVERAGE_TYPE,
         DEDUCTIBLE_AMOUNT, CLAIMED_MANUFACTURER, REPAIR_LOCATION}

For each group:
1. Name the group (1-2 words)
2. Is Group 1a a more specific type of Group 1? (yes/no)
3. Is Group 1b a more specific type of Group 1? (yes/no)
"
```

LLM calls: ~5-8 total (one per top-level class cluster with its children).

## 7. Phase 3: Relationship Discovery

**Purpose**: Discover inter-class relationships from bridge columns (headers appearing across multiple classes/files).

**Input**: Class hierarchy from Phase 2, per-file header assignments
**Output**: Named relationships between classes

### Step 3a: Identify Bridge Columns (Algorithmic)

A bridge column is any header that appears in files belonging to different classes.

```
POLICY_NUMBER appears in:
  - geicorenterspoliciesdetails.csv (Policy class)
  - geicorentersclaims.csv (Claim class)
  - geicorenterssurvey.csv (Survey class)
  - geicorenterscancelsurvey.csv (CancelSurvey class)

CLAIM_NUMBER appears in:
  - geicorentersclaims.csv (Claim class)
  - tmobileclaimsample.csv (Claim class)
  - geicorenterssurvey.csv (Survey class)
  - tmobilesurveysample.csv (Survey class)
  - tmobilechatsurveysample.csv (Survey class)

CLIENT_NAME appears in:
  - tmobileclaimsample.csv (Claim class)
  - tmobilesurveysample.csv (Survey class)
  - tmobilechatsurveysample.csv (Survey class)
```

### Step 3b: LLM Names Relationships (~3-5 calls)

```
Prompt template:
"Column '{column_name}' appears in data about both '{class_A}' and '{class_B}'.

What is the relationship between these two concepts?
Express as: ClassA --relationship_name--> ClassB

Example: if EMPLOYEE_ID appears in both 'Department' and 'Payroll',
the relationship might be: Department --employs--> Payroll
"
```

### Step 3c: Build Relationship Graph (Algorithmic)

Output relationship edges:
```
Claim --references--> Policy     (via POLICY_NUMBER)
Survey --about--> Claim          (via CLAIM_NUMBER)
Survey --for--> Policy           (via POLICY_NUMBER)
CancelSurvey --about--> Policy   (via POLICY_NUMBER)
```

## 8. Phase 4: Entity Assignment + Materialization

**Purpose**: Assign Zone 2's extracted entities to discovered classes and write the complete ontology to Neo4j.

**Input**: Zone 2 entities (with source_file preserved) + class hierarchy from Phase 2-3
**Output**: Complete ontology in Neo4j

### Step 4a: Fix the Information Pipeline

Modify `graph_cache.py` to extract from Neo4j edges:
- `source` (filename) — already stored, just not loaded
- `source_type` (llm/structured/regex) — already stored
- `confidence` (0.0-1.0) — already stored

Each entity gets: `source_files: set[str]` — all files it was extracted from.

### Step 4b: Entity-to-Class Assignment (Algorithmic)

```
For each entity:
  1. Look up its source_files
  2. Map source_file → class (from Phase 2 file-to-class mapping)
  3. If entity appears in 1 class → assign directly
  4. If entity appears in multiple classes → it's a bridge entity
     → assign to the class where it has the MOST relations
     → create relationship edges to other classes
```

### Step 4c: LLM Validation (Edge cases only, ~3-5 calls)

Only for entities that appear in 3+ classes or have ambiguous assignment.

```
Prompt template:
"Entity '{entity_name}' appears in data about multiple concepts:
  - In '{class_A}' context: connected to {relation_summary_A}
  - In '{class_B}' context: connected to {relation_summary_B}

Which concept does this entity primarily belong to?
Answer with just the concept name."
```

### Step 4d: Write to Neo4j

1. Create `OntologyClass` nodes:
   - Properties: `name`, `level` (0=root, 1=top, 2=sub, 3=leaf), `source_files[]`, `header_count`
2. Create `SUBCLASS_OF` edges between classes
3. Create `RELATES_TO` edges between classes (from Phase 3)
4. Update each `Entity` node with:
   - `ontology_class` label
   - `lob_tag` (from filename tokens — for per-client filtering)
   - `source_file` (preserved provenance)

## 9. Expected Output

### Class Hierarchy (from actual data analysis)

```
Root
├── Policy
│   ├── Coverage
│   │   ├── PropertyCoverage       (COVPROP_*)
│   │   └── LiabilityCoverage     (COVLIAB_*)
│   ├── Endorsement                (ENDORSEMENT_*, 23 subtypes)
│   │   ├── EarthquakeEndorsement
│   │   ├── JewelryEndorsement
│   │   ├── WaterDamageEndorsement
│   │   ├── PetDamageEndorsement
│   │   ├── IdentityTheftEndorsement
│   │   ├── SewerBackupEndorsement
│   │   └── ... (17 more)
│   ├── Financial                  (GWP, NWP, FEES, TAXES)
│   ├── Billing                    (PAYPLAN, PAYCODE, BILLING*)
│   └── PolicyDates                (EFF_DATE, EXP_DATE, CXL_*)
│
├── Claim
│   ├── [GEICO subclass]           (CAUSE_OF_LOSS, IS_CAT, TOTAL_LOSSES)
│   └── [T-Mobile subclass]        (DEVICE_DAMAGE_TYPE, FULFILLMENT_TYPE)
│       ├── Shipment               (SHIPPED_*)
│       ├── Repair                 (REPAIR_*)
│       └── ClaimTimeline          (AUTH_TO_*, REPORT_TO_*, SPEED_TO_*)
│
├── Survey
│   ├── [GEICO service survey]     (A_AGENT_*, satisfaction scores)
│   ├── [T-Mobile experience]      (*_CSAT, *_CES scores)
│   │   └── ChatSurvey            (CHAT_*, CONVERSATION_*, DISPOSITION_*)
│   └── CancellationSurvey        (RENTERS_CANCEL_*, CANCELLATION_*)
│
├── Person                         (INSURED, INSURED_NAME)
├── Address                        (LADD*, LAZIP, RISK_ST, COUNTY)
├── Organization                   (ORGANIZATION_*, MCO, ACCOUNT)
├── Device                         (DEVICE_TYPE, MANUFACTURER)
│
└── ServiceContract                (from Auto_Service PDF)
    ├── Vehicle                    (VEHICLE INFORMATION section)
    ├── Member                     (MEMBER INFORMATION section)
    ├── Producer                   (PRODUCER INFORMATION section)
    ├── ContractTerms              (KEY TERMS section)
    └── ContractCoverage           (WHAT THIS CONTRACT COVERS section)
```

Names in [brackets] will be determined by the LLM at runtime — they are NOT hardcoded.

### Estimated Metrics

| Metric | Old SV-LOI | Expected FBI |
|--------|:----------:|:------------:|
| Total classes | 9-11 | 30-50 |
| Hierarchy depth | 1 level | 3-4 levels |
| Endorsement subtypes | 0 | ~23 |
| Coverage subtypes | 0 | 2-3 |
| Claim subtypes | 0 | 2+ |
| Survey subtypes | 0 | 3-4 |
| LLM calls total | 100+ | ~20-30 |
| Domain-agnostic | Yes | Yes |
| New LOB extensible | Partial | Full |

## 10. LLM Budget Summary

| Phase | Calls | Purpose |
|-------|:-----:|---------|
| 1b: Expand headers | 2-3 | Batch-expand cryptic abbreviations |
| 1c: Parse filenames | 1 | Tokenize filenames |
| 2 iter 2: Semantic grouping | 2-3 | Group ungrouped headers |
| 2 iter 4: Hierarchy naming | 5-8 | Name classes at each level |
| 3b: Relationship naming | 3-5 | Name bridge relationships |
| 4c: Validation | 3-5 | Edge case entity assignment |
| **Total** | **~20-30** | **vs. 100+ in current SV-LOI** |

All prompts are self-contained. No references to external standards, domain ontologies, or assumed knowledge.

## 11. Integration with Zone 2

### What Changes in Zone 2
- **Nothing in extraction logic** — Zone 2 continues extracting as-is
- `graph_cache.py` modified to also load: `source` (filename), `source_type`, `confidence` from Neo4j edges

### What Phases 1-3 Need from Zone 2
- **Nothing** — Phases 1-3 work on raw file metadata only

### What Phase 4 Needs from Zone 2
- Entity nodes with their relation edges
- `source` property on edges (to map entities to source files → classes)

## 12. Extensibility Design

When a new LOB arrives (e.g., `synthetic_data_sample_petinsuranceclaims.csv`):

1. **Phase 1**: Extracts headers, expands abbreviations, parses filename → `[pet, insurance, claims]`
2. **Phase 2 Iter 1**: Finds prefix groups in new file's headers
3. **Phase 2 Iter 3**: Compares to existing class fingerprints
   - If new file's `CLAIM_*` headers overlap >0.3 with existing Claim class → maps to Claim, creates new subclass
   - If unique headers define a new concept → creates new top-level class
4. **Phase 4**: New entities assigned to discovered classes with `lob_tag=pet_insurance`

No code changes. No retraining. No manual schema updates.

## 13. Novelty Claim

**No existing work uses file-level structural metadata (filenames, column headers, section structure) as the primary signal for ontology class discovery.**

Prior approaches:
- AutoSchemaKG (2025): LLM conceptualizes entities from graph neighbors → requires strong LLM
- OLLM (NeurIPS 2024): Fine-tunes LLM on ontology subgraphs → requires training data
- RIGOR: Uses DDL schemas (CREATE TABLE) → requires formal database definitions
- Leiden/Louvain clustering: Uses entity embeddings → produces flat, impure clusters

Our approach:
- Works on raw file metadata without extraction
- Algorithmic core with LLM as naming assistant
- Discovers multi-level hierarchy from header prefix structure
- 10x fewer LLM calls than entity-level approaches
- Domain-agnostic by design

## 14. Open Questions

1. **Prefix trie threshold**: The ≥3 headers threshold for prefix grouping may need tuning. Too low → noise classes. Too high → misses small groups. Should be configurable.
2. **Cross-file fuzzy matching**: How to handle synonym headers across files (e.g., `SURVEY_ID` vs `GATEWAY_SURVEY_ID`, `POLICY_NUMBER` vs `POLNO` vs `POLICY_NO`)? The LLM expansion in Phase 1b helps, but may not catch all synonyms. May need an additional LLM call to confirm matches.
3. **PDF/TXT fingerprint quality**: Section headings from PDFs are less structured than CSV headers. The quality of class discovery for PDF-only LOBs may be lower than for CSV-heavy LOBs. Need to validate.
4. **Evaluation against Riskine**: Riskine has 26 classes designed for European P&C. Our data (US operational data) may discover classes that don't map to Riskine at all (e.g., Survey, SLA metrics). Need to decide how to evaluate classes with no Riskine equivalent.

## 15. Out of Scope

- Cross-domain transfer experiments (auto insurance → flood insurance)
- Evaluation metric design (will use existing Riskine alignment metrics)
- Zone 1 or Zone 2 redesign (only Zone 3)
- Chatbot/RAG integration (downstream consumer of the ontology)
- Performance benchmarking against ACORD/FIBO (these are not comparable — they are manually designed)
