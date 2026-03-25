# Speaker Script — Midterm Presentation (5-7 min)

---

## Slide 1: Title (30 sec)

"Hi everyone. I'm Xiaofei, and my partner is Zechary. Our capstone project is about automatically building knowledge graphs for insurance — specifically, can we get an AI system to discover the ontology structure of an insurance domain without any manual engineering?"

---

## Slide 2: The Problem (45 sec)

"So here's the problem. When an insurance company wants to expand into a new line of business — say from auto to flood insurance — they need a knowledge schema, an ontology, that describes the concepts in that domain: what's a coverage, what's a risk, what's a claim.

Today, this takes 3 to 6 months of manual work by domain experts. And it doesn't transfer — you have to start from scratch for each new domain.

Existing tools like LangChain's LLMGraphTransformer can extract knowledge graphs automatically, but they produce flat, messy graphs — duplicate entities, inconsistent labels, no hierarchy. We wanted to fix that."

---

## Slide 3: Our Solution (45 sec)

"Our solution is a 4-zone pipeline. Zone 1 chunks the documents with section awareness. Zone 2 extracts triples using an LLM — but crucially, it bootstraps its own vocabulary from the documents, no hardcoded schema. Zone 3 is where our research contribution lives — ontology induction. And Zone 4 stores everything in Neo4j.

The key design principle: zero domain-specific code. The exact same pipeline runs on flood insurance and auto insurance with no changes."

---

## Slide 4: Extraction Results (30 sec)

"Zone 2 works really well. Using qwen2.5 72-billion parameter model on our GPU cluster, we extracted 1,351 entities and over 1,700 relationships from 188 document chunks. Query accuracy is 95% — meaning the graph correctly answers 19 out of 20 evaluation questions. So extraction is solved. The hard problem is: can we organize these entities into meaningful ontology classes?"

---

## Slide 5: Three Methods (60 sec — this is the core)

"For Zone 3, we implemented and compared three ontology induction methods.

First, Leiden — standard community detection. It clusters entities by embedding similarity. The problem is the clusters are impure. A cluster called 'Hazard' ends up mixing coverages, buildings, and perils together.

Second, RSI-LCR — our relation-signature approach. It groups entities by what types of relations they participate in. Better structural separation, but it over-fragments into 30 tiny clusters.

Third — and this is our novel contribution — SV-LOI, Structurally-Verified LLM Ontology Induction. The key insight is: neither the LLM's semantic judgment nor the graph's structural signal is sufficient alone. The LLM knows that 'Coverage B' is a type of coverage — but it's inconsistent across batches. The graph structure knows which entities behave similarly — but can't name the classes.

SV-LOI fuses both: the LLM assigns classes, structural signatures verify the assignments, and disagreements are arbitrated with enriched context."

---

## Slide 6: Results Table (45 sec)

"Here are the results. All three methods use the exact same extraction — 1,351 entities from qwen2.5 72b. We evaluate using metrics from the OLLM paper at NeurIPS 2024.

SV-LOI wins on every metric. BERTScore F1 is 0.732 — meaning our induced class names semantically align with the Riskine reference ontology. Graph F1 is 0.714 — meaning the hierarchy structure matches. And it does this with just 7 clean classes, versus Leiden's 11 impure clusters or RSI-LCR's 30 fragments.

The important thing is: our pipeline never sees the Riskine ontology. It discovers these classes purely from the documents."

---

## Slide 7: What It Discovered (30 sec)

"Here's what SV-LOI actually found. It discovered Coverage, Product, Property, Risk, Address — all matching Riskine reference classes. The two it missed — Requirement and Occupancy — don't have direct Riskine equivalents.

And the three Riskine classes we didn't cover — Person, Organization, Object — those concepts simply don't appear in flood insurance documents. That's expected and honest."

---

## Slide 8: Key Findings (45 sec)

"A few interesting findings. First, extraction scales dramatically with model size — the 8-billion model extracts one triple per chunk, while the 72-billion model extracts 8 to 50. Extraction is solved.

Second, LLMs naturally think in data types, not domain concepts. Our first SV-LOI attempt produced classes like 'FinancialAmount' and 'TimePeriod' instead of 'Coverage' and 'Risk'. We fixed this with a two-stage approach: first ask what domain this is, then propose domain-appropriate classes.

Third, we discovered that standard name-matching F1 is a poor metric — it rewards naming convention similarity, not semantic correctness. That's why we adopted BERTScore and Graph F1 from recent NeurIPS papers."

---

## Slide 9: Next Steps (30 sec)

"For the remaining weeks: we're running the identical pipeline on auto insurance data to prove cross-domain transfer. We expect auto insurance to fill in the missing classes — Person, Organization, Vehicle. We're also writing the paper targeting a peer-reviewed venue. And we have a rescue step ready to reduce unclassified entities from 24% to under 5%."

---

## Slide 10: Questions (15 sec)

"That's our midterm update. The code is all on GitHub, and we're happy to take questions. Thank you."

---

## Anticipated Questions + Answers

**Q: How is this different from just prompting GPT to design an ontology?**
A: Pure LLM ontology generation (OLLM, NeurIPS 2024) produces ontologies that aren't grounded in the actual data. Our pipeline extracts entities first, THEN induces classes from them. The structural verification step catches LLM inconsistencies. We show this in the comparison — Leiden (pure algorithmic) and SV-LOI (LLM + structural) both outperform approaches that don't use graph structure.

**Q: Why not just use the Riskine ontology directly?**
A: That would defeat the purpose. Our thesis is that you DON'T need a pre-existing ontology — the pipeline discovers one automatically. Riskine is only used for evaluation, never in the pipeline. This is critical for generality — you can't assume a reference ontology exists for every new domain.

**Q: What's the practical application?**
A: An insurance company expanding into a new LOB (e.g., cyber insurance) could run our pipeline on their policy documents and get an initial ontology in hours instead of months. The ontology expert then refines it rather than building from scratch — augmenting, not replacing, human expertise.

**Q: Why are some Riskine classes missing?**
A: Person, Organization, and Object don't appear in flood insurance documents — the NFIP policy is about what's covered and how, not who's involved. When we run on auto insurance (which has driver/owner info), we expect those classes to appear. That's the cross-domain transfer experiment.

**Q: What model do you use?**
A: qwen2.5:72b running on Emory's Turing GPU cluster (2x Quadro RTX 8000, 48GB each). We also tested llama3.1:8b and 70b — the 72b model was dramatically better for extraction but all models work for the pipeline structure.
