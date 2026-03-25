# CS584 Midterm Presentation — Speaker Script

**Total time: ~8 minutes**

---

## Slide 1: Title (30s)

Welcome everyone. Our project is **Schema-Evolving Knowledge Graphs for Insurance** — the key word is "automated." We want to eliminate the 3-6 month manual effort of building an ontology for each new insurance line of business.

I'm Xiaofei Wang, working with Zechary Chou, advised by Dr. Kang at Assurant.

---

## Slide 2: The Problem (45s)

Here's the problem. When an insurance company expands into a new line of business — say from flood to auto — they need a new ontology. That means hiring domain experts, spending 3-6 months, and the work doesn't transfer.

Existing tools like LangChain's LLMGraphTransformer can extract knowledge graphs, but they produce a **flat mess** — duplicated entities, inconsistent types, no hierarchy. You get a knowledge graph but **no ontology structure**.

Our research question: **can we automatically induce an ontology from an extracted knowledge graph?**

---

## Slide 3: Pipeline (30s)

Our pipeline has 4 zones. Zone 1 handles document ingestion. Zone 2 is extraction — bootstraps vocabularies from documents, extracts triples, resolves entities. Zone 3 is the **research contribution** — ontology induction. Zone 4 is structured storage.

The critical point: **the entire pipeline is domain-agnostic.** Same code runs on flood AND auto insurance with zero changes.

---

## Slide 4: Extraction Results (30s)

Zone 2 results: 1,351 entities, 1,749 relationships, **95% query accuracy.** The extraction is solved.

The bottleneck is ontology induction — that's where our research contribution lies.

---

## Slide 5: Three Methods Overview (30s)

For Zone 3, we compared three methods. Leiden is our baseline — standard graph clustering. RSI-LCR uses relation signatures. SV-LOI is our **novel contribution** — it fuses LLM typing with structural verification.

The key insight: **neither semantic nor structural signals alone are sufficient.**

---

## Slide 6: Leiden — Deep Dive (30s)

Leiden: the standard approach. Group similar entities together. It found 11 clusters — so bottom-up induction IS feasible.

But the clusters are **semantically impure.** "Hazard" mixed coverages, buildings, AND perils. Name F1 was literally **zero.**

**Lesson: entities that LOOK alike are not necessarily the same THING.**

---

## Slide 7: RSI-LCR — Deep Dive (30s)

RSI-LCR: fix Leiden by clustering on **relation patterns** instead of name embeddings. Coverage entities have COVERS and HAS_LIMIT; Person entities have HAS_NAME and RESIDES_AT.

Name F1 went from 0.000 to 0.295 — found Coverage, Risk, Property. But it **over-fragments** to 30 tiny clusters.

**Lesson: relation patterns tell you how entities BEHAVE, not what they ARE.**

---

## Slide 8: SV-LOI — Deep Dive (45s)

SV-LOI: our **novel contribution.** Semantic signal fails alone. Structural signal fails alone. So **fuse both.**

Six phases: detect domain, propose classes, assign entities, structural verification, arbitrate disagreements, consolidate.

Results: **Name F1 = 0.563. BERTScore F1 = 0.732. Graph F1 = 0.714.** Seven clean classes. Five out of ten reference classes matched — **without ever seeing the reference ontology.**

---

## Slide 9: Results Table (30s)

The full comparison. **Every single metric improves** from Leiden to RSI-LCR to SV-LOI. Name F1: zero to 0.295 to 0.563. BERTScore: 0.451 to 0.577 to 0.732. Graph F1: 0.441 to 0.501 to 0.714.

All three methods use the exact same extraction. The only difference is the induction algorithm.

---

## Slide 10: What It Discovered (30s)

Coverage matched Coverage. Product matched Product. Property matched Property. Risk matched Risk. Address matched Address.

The 5 unmatched classes — Person, Organization, Object — are **absent from flood insurance data.** This is a data limitation, not a method limitation. Our method correctly doesn't hallucinate classes that aren't there.

---

## Slide 11: Key Findings (30s)

Four findings: extraction scales with model size. LLMs default to data-type thinking — we fixed with two-stage discovery. Name-matching F1 is a poor metric — use BERTScore. Source data limits recall — report present-class F1 for fair comparison.

---

## Slide 12: Next Steps (20s)

Remaining weeks: reduce unclassified entities, cross-domain transfer on auto insurance, paper writing, and a Streamlit demo.

---

## Slide 13: Questions (15s)

Thank you. The key takeaway: SV-LOI achieves **0.732 BERTScore F1** without ever seeing the reference ontology. It's fully domain-agnostic and ready for cross-domain transfer.

Questions?
