# Auto Review Log — SV-LOI Ontology Induction

## Round 1 (2026-03-24)

### Assessment
- **Score: 3/10** — Not ready
- **Verdict**: Major prompt engineering failure. Class discovery produces data types, not domain concepts.

### Key Criticisms
1. **CRITICAL**: `discover_class_vocabulary()` showed only type statistics (e.g., "financial_entity: 577"), causing LLM to propose data types (FinancialAmount, TimePeriod, Measurement) instead of domain concepts (Coverage, Person, Risk)
2. **CRITICAL**: No entity NAME examples in discovery prompt — LLM can't infer domain concepts from abstract statistics
3. **HIGH**: No anti-data-type steering in any prompt
4. **HIGH**: FinancialAmount mega-class consumed 43% of all entities — no balance enforcement
5. **MEDIUM**: `batch_type_entities()` showed only 3 outgoing + 2 incoming relations — too sparse

### Actions Taken
1. **Rewrote `discover_class_vocabulary()`**: Now shows entity NAME samples per type (8 examples each), concrete triple examples with real entity names, and explicit steering: "propose WHAT entities ARE in the real world — NOT data types"
2. **Improved `batch_type_entities()` prompt**: Increased relation context (5 out + 3 in), added domain-concept steering ("Is it a person? An organization? A type of coverage?")
3. **Added `rebalance_mega_classes()`**: New Phase 1c — any class > 30% of entities gets split via LLM into 2-4 sub-classes with re-typing
4. **Tuned constants**: BATCH_SIZE 20→15, MIN_CLASS_SIZE 2→3, TARGET_CLASSES 8-20→10-25

### Results
- Pending cluster run. Code changes pushed to main.

### Status
- Continuing to Round 2 (after cluster results)
