# Midterm Presentation — Slide Content
## CS584 AI Capstone | Xiaofei Wang & Zechary Chou | March 2026

---

## Slide 1: Title (30 sec)

**Title:** Schema-Evolving Knowledge Graphs for Insurance
**Subtitle:** Automated Ontology Induction with Cross-Domain Transfer Learning
**Authors:** Xiaofei Wang, Zechary Chou
**Course:** CS 584 AI Capstone — Emory University, Spring 2026
**Advisor:** Dr. Yingying Kang (Assurant)

---

## Slide 2: The Problem (45 sec)

**Title:** The Ontology Engineering Bottleneck

**Content:**
- Insurance companies expanding into new lines of business (flood, auto, health) face a **3-6 month ontology engineering bottleneck** per domain
- Domain experts must manually design knowledge schemas from scratch
- Current tools (LLMGraphTransformer) extract knowledge graphs automatically, BUT:

| Problem | Impact |
|---------|--------|
| Entity duplication | "Policy 5123" vs "policy #5123" = 2 nodes |
| Type proliferation | Policy, Policy action, Policy type... |
| No ontology hierarchy | Flat labels, no SUBCLASS_OF |
| No cross-domain transfer | Every LOB starts from scratch |

**Key point:** Can we automate ontology induction?

---

## Slide 3: Our Solution — 4-Zone Pipeline (45 sec)

**Title:** 4-Zone Domain-Agnostic Pipeline

**Diagram:**
```
Zone 1          Zone 2           Zone 3              Zone 4
Ingestion  -->  Extraction  -->  Ontology       -->  Storage
                                 Induction
PDF + CSV       Bootstrap        SV-LOI (novel)      Neo4j
Section-aware   schema from      LLM typing +        3-layer
chunking        documents        Structural           graph
                No hardcoding    verification
```

**Key points:**
- Zero domain-specific code — same pipeline runs on flood AND auto insurance
- Zone 2 bootstraps vocabulary from YOUR documents (not hardcoded)
- Zone 3 is the research contribution: 3 methods compared

---

## Slide 4: Zone 2 — Extraction Results (30 sec)

**Title:** Extraction is Solved (Zone 2)

**Content:**
- Model: qwen2.5:72b on Emory Turing GPU cluster
- Input: 188 chunks (49 PDF + 67 policy CSV + 72 claims CSV)
- Output: **1,351 entities, 1,749 relationships**
- **95% query accuracy** on 20 evaluation questions
- Domain-agnostic: bootstrapped 20 entity types + 45 relation types from documents

**Takeaway:** Extraction quality is excellent. The bottleneck is ontology induction.

---

## Slide 5: Zone 3 — The Research Contribution (60 sec)

**Title:** Three Ontology Induction Methods Compared

**Method A: Leiden (baseline)**
- Community detection on entity similarity graph
- Result: clusters are semantically impure (mixes coverages, buildings, perils)

**Method B: RSI-LCR**
- Group entities by relation-type signatures
- Better structural separation, but still over-fragments

**Method C: SV-LOI (our novel contribution)**
1. Two-stage class discovery: detect domain, then propose classes
2. LLM assigns each entity to a class (batched, with relation context)
3. Structural verification flags inconsistencies
4. LLM-guided consolidation merges fine-grained classes

**Key insight:** Neither semantic (LLM) nor structural (graph) signal alone is sufficient. SV-LOI fuses both.

---

## Slide 6: Results — Main Comparison Table (45 sec)

**Title:** SV-LOI Outperforms on Every Metric

| Metric | Leiden | RSI-LCR | SV-LOI |
|--------|:------:|:-------:|:------:|
| Name F1 | 0.000 | 0.295 | **0.563** |
| BERTScore F1 | 0.451 | 0.577 | **0.732** |
| Graph F1 | 0.441 | 0.501 | **0.714** |
| Entity Assignment F1 | 0.226 | 0.302 | **0.326** |
| Wu-Palmer | 0.653 | 0.723 | **0.752** |
| Induced classes | 11 | 30 | **7** |

- All methods use same extraction (qwen2.5:72b, 1,351 entities)
- Metrics follow OLLM (NeurIPS'24) evaluation protocol
- SV-LOI discovers 7 clean classes vs Leiden's 11 impure clusters

---

## Slide 7: What SV-LOI Discovered (30 sec)

**Title:** Induced Ontology vs Riskine Reference

**Our induced classes → Riskine match:**
| Induced | Riskine | Match |
|---------|---------|:-----:|
| Coverage | Coverage | MATCH |
| Product | Product | MATCH |
| Property | Property | MATCH |
| Risk | Risk | PARTIAL |
| Address | Address | MATCH |
| Requirement | — | no match |
| Occupancy | — | no match |

**5 out of 10 Riskine classes matched** — without ever seeing Riskine during induction.
Person, Organization, Object are absent from flood data (expected).

---

## Slide 8: Key Findings (45 sec)

**Title:** What We Learned

1. **Extraction is solved, induction is the bottleneck**
   - 95% query accuracy with 72b model; 1 triple/chunk with 8b → 8-50/chunk with 72b

2. **LLMs default to data-type thinking**
   - First SV-LOI attempt produced "FinancialAmount", "TimePeriod" instead of "Coverage", "Risk"
   - Fix: two-stage discovery (detect domain first) + forbidden name filter

3. **Naming convention bias (F-07)**
   - Larger models produce compound names ("PropertyCoverageComponent") that score WORSE on F1
   - Solved by using BERTScore F1 instead of exact name matching

4. **Source data limits recall**
   - Flood insurance data has no Person/Organization/Object entities
   - Present-class F1 (evaluating only evidenced classes) gives fairer comparison

---

## Slide 9: Remaining Work (30 sec)

**Title:** Next Steps (Weeks 10-11)

- [ ] Reduce "Other" entities from 24% to <5% (rescue step implemented)
- [ ] Cross-domain transfer: run identical pipeline on auto insurance (zero code changes)
- [ ] CTR/PTR measurement: how many classes transfer between flood and auto?
- [ ] Final paper writing
- [ ] Streamlit demo for interactive ontology exploration

**Publication target:** Peer-reviewed venue (workshop or short paper)

---

## Slide 10: Thank You (15 sec)

**Title:** Questions?

**Key links:**
- GitHub: github.com/xiw-1202/Agentic-Data-Fusion-for-Insurance-Knowledge-Graphs
- Advisor: Dr. Yingying Kang (Assurant)
- GPU: Emory Turing cluster (2x Quadro RTX 8000, 48GB each)

**Summary:** We built a domain-agnostic pipeline that automatically induces ontology classes from insurance documents. SV-LOI achieves 0.732 BERTScore F1 and 0.714 Graph F1 against the Riskine reference ontology — without ever seeing it during induction.
