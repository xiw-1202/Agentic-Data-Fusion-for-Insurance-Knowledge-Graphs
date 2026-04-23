# SEAF-KG Capstone Presentation — Speaker Notes

**Total time: ~12-15 minutes**

---

## Slide 1: Title (30 seconds)

"Hi everyone, I'm Xiaofei Wang. My capstone project is SEAF-KG — Structurally-Enhanced Agentic Fusion for Insurance Knowledge Graphs. I worked under Professor Agichtein with industry mentoring from Dr. Yingying Kang at Assurant, one of the largest insurance service companies in the US."

---

## Slide 2: Problem & Motivation (1.5 minutes)

"So what problem are we solving? Insurance companies like Assurant deal with extremely heterogeneous data. They have policy documents in PDF format — legal text describing what's covered and what's excluded. They have claims data in CSV files — structured records with claim numbers, dates, amounts. They have customer surveys with NPS scores and satisfaction ratings.

The problem is there's no unified knowledge representation across these sources. If a business analyst asks 'Which claims took the longest to resolve, and what was the customer satisfaction for those cases?' — they'd have to manually join data from three different systems.

Manual ontology creation — where domain experts define the categories and relationships — is expensive and doesn't transfer. An ontology built for flood insurance won't work for auto insurance.

Our goal: automatically build a domain ontology from raw insurance data using LLMs combined with graph structure analysis."

---

## Slide 3: Pipeline Overview (2 minutes)

"Here's the pipeline architecture. It has four zones that process data left to right.

Zone 1 handles ingestion — PDFs get chunked into manageable text blocks, CSVs get parsed into structured records with automatic schema detection and column name expansion.

Zone 2 is extraction. This is a dual-path approach — LLMs extract knowledge triples from PDF text, while a deterministic structured mapper converts CSV records into triples without needing an LLM. Entity resolution then merges duplicate entities using cosine similarity.

Zone 3 is where our novel contribution lives — SV-LOI, which I'll explain in detail next. This is the ontology induction step that discovers what classes exist and assigns every entity to a class.

Zone 4 stores everything in Neo4j, a graph database, where business analysts can query with Cypher.

The bottom bar shows the scale: 271 input chunks produced over 22,000 triples, 5,255 entities, organized into 9 ontology classes with 27,000 relationships."

---

## Slide 4: Zone 2 — Knowledge Extraction (1.5 minutes)

"Let me dive into Zone 2 briefly. The dual-path design means we don't force LLMs to handle structured data they're bad at, and we don't force templates onto unstructured text.

The interesting technical challenge was property collapse — deciding which extracted values should remain as entities in the graph versus becoming properties on existing entities.

We developed a data-driven approach with three rules, zero hardcoded names. If a relation's values are mostly numeric with high cardinality but not too high — like claim resolution times, NPS scores — it's a MEASURE and stays as a queryable entity. If values are reused across many records — like claim channels (Web, Phone, EzPass) — it's a DIMENSION and stays. Everything else — single-use dates, IDs, serial numbers — collapses to a node property.

The key insight: we detect IDs versus measures by checking digit count. A claim number with 9 digits is an ID. A resolution time with 4 digits is a measure. All data-driven, no domain-specific rules needed."

---

## Slide 5: Zone 3 — SV-LOI (2 minutes)

"Now to the core contribution — SV-LOI, Structurally-Verified LLM Ontology Induction.

The key insight is that neither LLMs nor graph algorithms alone produce good ontologies. LLMs are accurate — they correctly recognize that 'flood insurance' is a type of Coverage — but they're inconsistent across batches. The same entity might get classified as Coverage in one batch and Product in another.

Graph structural clustering is consistent — entities with similar relation patterns always cluster together — but it over-fragments, creating too many small classes that don't correspond to real concepts.

SV-LOI fuses both signals. The pipeline has 7 stages.

Stage 3 does LLM batch typing — 15 entities per prompt, with each entity's name and top relationships as context.

Stage 4 is the structural verification — we compute cosine similarity signatures based on each entity's relation patterns, build class centroids, and flag any entity that's more than 2 standard deviations from its class centroid. These flagged entities go back to the LLM with enriched context: 'You typed this as Coverage, but structurally it looks like Risk. Which is correct?'

Stage 5 does 5-way class relation inference — for each pair of classes, the LLM decides: are they equivalent, parent-child, overlapping, or distinct? Only equivalence triggers a merge.

This fusion approach eliminates both failure modes — LLM inconsistency and structural over-fragmentation."

---

## Slide 6: Zone 2 Evaluation (1.5 minutes)

"For extraction quality, we use three metrics.

Triple Precision measures whether extracted triples are factually correct. An LLM judge evaluates each triple. We got 91.1% — out of 100 sampled triples, 82 were correct, 8 incorrect, and 10 uncertain. For example, 'Service Contract covers Heated Back Glass' is correct — that's in the policy document. But 'Claim Management can be canceled at mytmoclaim.com' is incorrect — that's a URL, not a cancellation condition.

Fact Recall measures whether important facts from the source documents made it into the KG. We extract key facts from 30 source chunks and check if matching triples exist using BERTScore similarity. 73.2% — 175 out of 239 facts were captured.

Source Grounding verifies traceability — for each triple, can we find the source document that supports it? 80% of triples were fully supported. The 9 with no source found were typically inferences the LLM made that weren't explicitly stated in any single chunk."

---

## Slide 7: Zone 3 Evaluation Part 1 (1.5 minutes)

"For ontology quality, we evaluate against Riskine, an open-source insurance reference ontology with 26 classes.

BERTScore F1 measures how well our class names match Riskine's, using BERT embeddings for semantic similarity. We got 0.617 with precision of 0.758. This means most of our classes have meaningful matches — Organization matches Organization exactly, Policy matches Product with 0.90 cosine similarity, Coverage matches Coverage perfectly.

Graph F1 compares the structure of our ontology graph against Riskine's using Soft Graph Correspondence. Our precision is very high at 84.2% — the edges we created are correct. Recall is lower at 29.1% because Riskine has 46 edges and we only induced 26. This is expected — our data only covers a subset of the insurance domain."

---

## Slide 8: Zone 3 Evaluation Part 2 (1 minute)

"Wu-Palmer similarity measures taxonomy distance — how close our class hierarchy is to Riskine's. We got 0.621, meaning on average our classes are in roughly the right neighborhood of the reference hierarchy.

Entity Assignment F1 is the hardest metric. For each induced class, we embed all its member entities and find the nearest Riskine class by centroid distance. Against all 26 Riskine classes we get 0.206 — but many of those 26 classes simply don't exist in our data, like BankAccount or DrivingLicense. Against only the 7 classes with evidence in our data, we get 0.404 — Coverage matches Coverage, Risk matches Risk, Person partially matches Identification."

---

## Slide 9: Query Showcase (1.5 minutes)

"But metrics don't tell the full story. Can the KG actually answer real business questions?

We tested 5 analytical queries, running Cypher against our Neo4j graph and comparing results against the raw CSV files using pandas.

Query 1: Which claims had the longest resolution time? The KG returns 61,606 hours for a Physical Damage claim on a Cellular Phone. This matches the ground truth exactly — verified against the tmobilechatsurveysample CSV.

Query 2: What device types have the most claims? The KG correctly identifies Cellular Phone as dominant with 274 entries, capturing 96% of the ground truth count.

Query 5: What loss types are most common? Physical Damage leads with 144, and 3 out of 4 loss type counts match exactly.

This demonstrates that the KG is not just structurally valid — it's analytically useful. Business users can write Cypher queries and get answers they can trust."

---

## Slide 10: Results Summary (30 seconds)

"Here's the full scorecard. 91% extraction precision, 73% fact recall, 80% source grounding. 9 induced classes with 100% backbone coverage — every record entity is connected to a concept entity through the ontology layer. Zero duplication, zero type inconsistency."

---

## Slide 11: Technical Innovations (1 minute)

"Seven technical innovations came out of this project. The data-driven property collapse that detects measures versus IDs without any hardcoded names. The dual-path extraction combining LLMs and deterministic mappers. The SV-LOI fusion method itself. LLM pairwise taxonomy with DAG enforcement. Record decomposition into domain sub-nodes. Versioned graph cache with automatic staleness detection. And scale-adaptive thresholds that work whether you have 2,000 or 2 million entities."

---

## Slide 12: Future Work & Conclusion (1 minute)

"For future work — cross-domain transfer to test whether an ontology induced from auto insurance data works on flood insurance. Variant comparison against published baselines like AutoSchemaKG and OLLM. And reducing the 'Other' fraction — currently 56.5% of entities aren't classified, mostly numeric values.

The core conclusion: SV-LOI demonstrates that fusing LLM semantic typing with graph-structural verification produces higher-quality ontologies than either signal alone. The pipeline processes heterogeneous insurance data into a queryable knowledge graph with 91% precision and domain-aligned classes — all without manual ontology engineering.

Thank you. I'm happy to take questions."
