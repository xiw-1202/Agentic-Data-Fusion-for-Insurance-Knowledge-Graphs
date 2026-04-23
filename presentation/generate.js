const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Xiaofei Wang";
pres.title = "SEAF-KG: Insurance Knowledge Graphs";

// Color palette — Midnight Executive
const C = {
  navy: "1E2761",
  ice: "CADCFC",
  white: "FFFFFF",
  dark: "0F1B3D",
  accent: "4A90D9",
  accentLight: "6BABEB",
  gray: "8899AA",
  lightBg: "F4F6FA",
  green: "2ECC71",
  red: "E74C3C",
  orange: "F39C12",
  text: "2C3E50",
  muted: "7F8C9B",
};

// Helper: create factory functions to avoid object reuse corruption
const makeShadow = () => ({ type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.12 });

// ========================================================================
// SLIDE 1: Title
// ========================================================================
let s1 = pres.addSlide();
s1.background = { color: C.navy };

// Top accent bar
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

s1.addText("SEAF-KG", {
  x: 0.8, y: 1.0, w: 8.4, h: 1.0,
  fontSize: 48, fontFace: "Georgia", bold: true, color: C.white, margin: 0,
});
s1.addText("Structurally-Enhanced Agentic Fusion\nfor Insurance Knowledge Graphs", {
  x: 0.8, y: 1.9, w: 8.4, h: 0.9,
  fontSize: 20, fontFace: "Calibri", color: C.ice, margin: 0,
});
s1.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 3.0, w: 2.5, h: 0.04, fill: { color: C.accent } });
s1.addText("CS584 AI Capstone  |  Emory University  |  Spring 2026", {
  x: 0.8, y: 3.3, w: 8.4, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: C.gray, margin: 0,
});
s1.addText("Xiaofei Wang", {
  x: 0.8, y: 4.2, w: 8.4, h: 0.4,
  fontSize: 16, fontFace: "Calibri", bold: true, color: C.white, margin: 0,
});
s1.addText("Advisor: Prof. Eugene Agichtein  |  Industry Mentor: Dr. Yingying Kang (Assurant)", {
  x: 0.8, y: 4.6, w: 8.4, h: 0.4,
  fontSize: 12, fontFace: "Calibri", color: C.gray, margin: 0,
});

// ========================================================================
// SLIDE 2: Problem & Motivation
// ========================================================================
let s2 = pres.addSlide();
s2.background = { color: C.lightBg };

s2.addText("Problem & Motivation", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.6,
  fontSize: 32, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});

// Challenge cards
const challenges = [
  { title: "Heterogeneous Data", desc: "PDFs (policy docs), CSVs (claims, surveys, policies) — no unified format" },
  { title: "No Knowledge Representation", desc: "Siloed data prevents cross-source queries and analytics" },
  { title: "Manual Ontology Is Expensive", desc: "Domain experts spend months building ontologies that don't transfer" },
  { title: "Our Goal", desc: "Automatically induce a domain ontology from raw insurance data using LLMs + graph structure" },
];

challenges.forEach((c, i) => {
  const row = Math.floor(i / 2);
  const col = i % 2;
  const x = 0.6 + col * 4.5;
  const y = 1.3 + row * 1.9;

  s2.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w: 4.2, h: 1.6,
    fill: { color: C.white }, rectRadius: 0.1, shadow: makeShadow(),
  });
  s2.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.08, h: 1.6,
    fill: { color: i === 3 ? C.accent : C.navy },
  });
  s2.addText(c.title, {
    x: x + 0.25, y: y + 0.15, w: 3.7, h: 0.4,
    fontSize: 16, fontFace: "Calibri", bold: true, color: C.navy, margin: 0,
  });
  s2.addText(c.desc, {
    x: x + 0.25, y: y + 0.6, w: 3.7, h: 0.8,
    fontSize: 12, fontFace: "Calibri", color: C.text, margin: 0,
  });
});

// ========================================================================
// SLIDE 3: Pipeline Overview
// ========================================================================
let s3 = pres.addSlide();
s3.background = { color: C.white };

s3.addText("Pipeline Overview: 4-Zone Architecture", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.6,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});

const zones = [
  { label: "Zone 1", title: "Ingestion", desc: "PDFs \u2192 chunks\nCSVs \u2192 records\nSchema detection", color: "3498DB" },
  { label: "Zone 2", title: "Extraction", desc: "LLM triples (PDFs)\nStructured mapper\nEntity resolution", color: "2ECC71" },
  { label: "Zone 3", title: "SV-LOI", desc: "7-stage induction\nLLM + structural\nNovel method", color: "E67E22" },
  { label: "Zone 4", title: "Storage", desc: "Neo4j graph DB\nCypher queries\nOntology layer", color: "9B59B6" },
];

zones.forEach((z, i) => {
  const x = 0.4 + i * 2.4;
  const y = 1.2;

  // Zone box
  s3.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w: 2.1, h: 3.4,
    fill: { color: C.white }, rectRadius: 0.1, shadow: makeShadow(),
    line: { color: z.color, width: 1.5 },
  });

  // Zone label pill
  s3.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: x + 0.2, y: y + 0.15, w: 1.1, h: 0.35,
    fill: { color: z.color }, rectRadius: 0.15,
  });
  s3.addText(z.label, {
    x: x + 0.2, y: y + 0.15, w: 1.1, h: 0.35,
    fontSize: 11, fontFace: "Calibri", bold: true, color: C.white, align: "center", valign: "middle", margin: 0,
  });

  // Title
  s3.addText(z.title, {
    x: x + 0.15, y: y + 0.7, w: 1.8, h: 0.4,
    fontSize: 18, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
  });

  // Description
  s3.addText(z.desc, {
    x: x + 0.15, y: y + 1.2, w: 1.8, h: 2.0,
    fontSize: 11, fontFace: "Calibri", color: C.text, margin: 0,
  });

  // Arrow between zones
  if (i < 3) {
    s3.addText("\u2192", {
      x: x + 2.1, y: y + 1.2, w: 0.3, h: 0.5,
      fontSize: 24, color: C.gray, align: "center", valign: "middle", margin: 0,
    });
  }
});

// Bottom stats bar
s3.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.85, w: 10, h: 0.75, fill: { color: C.navy } });
const stats = [
  { val: "271", label: "Chunks" },
  { val: "22,126", label: "Triples" },
  { val: "5,255", label: "Entities" },
  { val: "9", label: "Classes" },
  { val: "27,380", label: "Relations" },
];
stats.forEach((st, i) => {
  const x = 0.5 + i * 1.9;
  s3.addText(st.val, {
    x, y: 4.85, w: 1.6, h: 0.4,
    fontSize: 20, fontFace: "Georgia", bold: true, color: C.white, align: "center", margin: 0,
  });
  s3.addText(st.label, {
    x, y: 5.2, w: 1.6, h: 0.3,
    fontSize: 10, fontFace: "Calibri", color: C.ice, align: "center", margin: 0,
  });
});

// ========================================================================
// SLIDE 4: Zone 2 — Knowledge Extraction
// ========================================================================
let s4 = pres.addSlide();
s4.background = { color: C.lightBg };

s4.addText("Zone 2: Knowledge Extraction", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.6,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});

// Dual-path section
s4.addText("Dual-Path Extraction", {
  x: 0.6, y: 1.0, w: 4, h: 0.35,
  fontSize: 16, fontFace: "Calibri", bold: true, color: C.accent, margin: 0,
});

// LLM path card
s4.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.6, y: 1.45, w: 4.2, h: 1.1,
  fill: { color: C.white }, rectRadius: 0.08, shadow: makeShadow(),
});
s4.addText("LLM Path (PDFs)", {
  x: 0.8, y: 1.5, w: 3.8, h: 0.3,
  fontSize: 13, fontFace: "Calibri", bold: true, color: C.navy, margin: 0,
});
s4.addText("Qwen 72B extracts (subject, relation, object) triples\nfrom policy documents, coverage terms, exclusions", {
  x: 0.8, y: 1.8, w: 3.8, h: 0.6,
  fontSize: 11, fontFace: "Calibri", color: C.text, margin: 0,
});

// Structured path card
s4.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 5.2, y: 1.45, w: 4.2, h: 1.1,
  fill: { color: C.white }, rectRadius: 0.08, shadow: makeShadow(),
});
s4.addText("Structured Mapper (CSVs)", {
  x: 5.4, y: 1.5, w: 3.8, h: 0.3,
  fontSize: 13, fontFace: "Calibri", bold: true, color: C.navy, margin: 0,
});
s4.addText("Deterministic record-to-triple conversion\nClaims, policies, surveys \u2192 typed records + identity linking", {
  x: 5.4, y: 1.8, w: 3.8, h: 0.6,
  fontSize: 11, fontFace: "Calibri", color: C.text, margin: 0,
});

// Data-driven collapse section
s4.addText("Data-Driven Property Collapse", {
  x: 0.6, y: 2.8, w: 8, h: 0.35,
  fontSize: 16, fontFace: "Calibri", bold: true, color: C.accent, margin: 0,
});

const collapseRules = [
  { type: "MEASURE", rule: ">80% numeric, 50\u201395% cardinality, \u22648 digits", ex: "TOTAL_CLAIM_TIME, NPS_SCORE", icon: "\u2705", color: "27AE60" },
  { type: "DIMENSION", rule: "\u22652x value reuse across subjects", ex: "CLAIM_CHANNEL, DCC_LOCATION", icon: "\u2705", color: "2980B9" },
  { type: "COLLAPSE", rule: "Everything else (IDs, single-use literals)", ex: "CLAIM_NUMBER, RESPONSE_ID", icon: "\u2B07\uFE0F", color: "95A5A6" },
];

collapseRules.forEach((cr, i) => {
  const y = 3.25 + i * 0.7;
  s4.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y, w: 8.8, h: 0.6,
    fill: { color: C.white }, rectRadius: 0.06, shadow: makeShadow(),
  });
  s4.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y, w: 0.06, h: 0.6,
    fill: { color: cr.color },
  });
  s4.addText(cr.type, {
    x: 0.85, y, w: 1.2, h: 0.6,
    fontSize: 13, fontFace: "Calibri", bold: true, color: cr.color, valign: "middle", margin: 0,
  });
  s4.addText(cr.rule, {
    x: 2.1, y, w: 4.0, h: 0.6,
    fontSize: 11, fontFace: "Calibri", color: C.text, valign: "middle", margin: 0,
  });
  s4.addText(cr.ex, {
    x: 6.2, y, w: 3.0, h: 0.6,
    fontSize: 10, fontFace: "Consolas", color: C.muted, valign: "middle", margin: 0,
  });
});

// ========================================================================
// SLIDE 5: Zone 3 — SV-LOI
// ========================================================================
let s5 = pres.addSlide();
s5.background = { color: C.navy };

s5.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

s5.addText("SV-LOI: Structurally-Verified LLM Ontology Induction", {
  x: 0.6, y: 0.25, w: 8.8, h: 0.55,
  fontSize: 24, fontFace: "Georgia", bold: true, color: C.white, margin: 0,
});

// Key insight box
s5.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.6, y: 0.95, w: 8.8, h: 0.7,
  fill: { color: "1A2555" }, rectRadius: 0.08,
  line: { color: C.accent, width: 1 },
});
s5.addText("Key Insight: LLM typing is accurate but inconsistent across batches. Structural clustering is consistent but over-fragments. Fusing both with disagreement arbitration eliminates each signal\u2019s failure mode.", {
  x: 0.9, y: 0.95, w: 8.2, h: 0.7,
  fontSize: 11, fontFace: "Calibri", italic: true, color: C.ice, valign: "middle", margin: 0,
});

// 7 Stages
const stages = [
  { num: "1", name: "Load", desc: "Graph cache" },
  { num: "2", name: "Discover", desc: "Class vocab" },
  { num: "3", name: "Classify", desc: "LLM batch" },
  { num: "4", name: "Verify", desc: "Structural" },
  { num: "5", name: "Consolidate", desc: "5-way merge" },
  { num: "6", name: "Structure", desc: "Taxonomy" },
  { num: "7", name: "Write", desc: "Neo4j" },
];

stages.forEach((st, i) => {
  const x = 0.35 + i * 1.35;
  const y = 1.85;

  // Circle number
  s5.addShape(pres.shapes.OVAL, {
    x: x + 0.3, y, w: 0.5, h: 0.5,
    fill: { color: C.accent },
  });
  s5.addText(st.num, {
    x: x + 0.3, y, w: 0.5, h: 0.5,
    fontSize: 16, fontFace: "Georgia", bold: true, color: C.white, align: "center", valign: "middle", margin: 0,
  });
  s5.addText(st.name, {
    x: x, y: y + 0.55, w: 1.1, h: 0.3,
    fontSize: 12, fontFace: "Calibri", bold: true, color: C.white, align: "center", margin: 0,
  });
  s5.addText(st.desc, {
    x: x, y: y + 0.8, w: 1.1, h: 0.3,
    fontSize: 9, fontFace: "Calibri", color: C.gray, align: "center", margin: 0,
  });

  if (i < 6) {
    s5.addText("\u2192", {
      x: x + 1.1, y, w: 0.25, h: 0.5,
      fontSize: 16, color: C.gray, align: "center", valign: "middle", margin: 0,
    });
  }
});

// Detail cards
const details = [
  { title: "Stage 3: LLM Batch Typing", desc: "15 entities/prompt with name + relations context. LLM assigns class from discovered vocabulary. ~70 LLM calls for 1,500+ entities." },
  { title: "Stage 4: Structural Verification", desc: "Cosine signature vectors per entity. Flag outliers >2\u03C3 from class centroid. Re-query LLM with enriched structural context." },
  { title: "Stage 5: 5-Way Class Relations", desc: "Pairwise: equivalent (merge), parent/child (SUBCLASS_OF), overlap, distinct. Protected class names never renamed." },
];

details.forEach((d, i) => {
  const x = 0.6 + i * 3.1;
  const y = 3.1;
  s5.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w: 2.8, h: 2.2,
    fill: { color: "1A2555" }, rectRadius: 0.08,
  });
  s5.addText(d.title, {
    x: x + 0.15, y: y + 0.1, w: 2.5, h: 0.4,
    fontSize: 12, fontFace: "Calibri", bold: true, color: C.accent, margin: 0,
  });
  s5.addText(d.desc, {
    x: x + 0.15, y: y + 0.55, w: 2.5, h: 1.5,
    fontSize: 10, fontFace: "Calibri", color: C.ice, margin: 0,
  });
});

// ========================================================================
// SLIDE 6: Zone 2 Eval
// ========================================================================
let s6 = pres.addSlide();
s6.background = { color: C.lightBg };

s6.addText("Zone 2 Evaluation: Extraction Quality", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.6,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});

const z2metrics = [
  {
    name: "Triple Precision",
    value: "91.1%",
    what: "LLM judge evaluates each extracted triple as Correct / Incorrect / Uncertain",
    detail: "82 correct, 8 incorrect, 10 uncertain (out of 100 sampled)",
    exGood: 'SERVICE CONTRACT \u2014COVERS\u2192 Heated Back Glass \u2192 CORRECT',
    exBad: 'Claim Management \u2014CAN_BE_CANCELED_AT\u2192 mytmoclaim.com \u2192 INCORRECT',
  },
  {
    name: "Fact Recall",
    value: "73.2%",
    what: "For 30 source chunks, extract key facts, check if KG contains matching triples",
    detail: "175 of 239 facts found (BERTScore cosine > 0.65)",
    exGood: "Coverage terms, exclusions, policy conditions captured",
    exBad: "Complex multi-clause legal language sometimes missed",
  },
  {
    name: "Source Grounding",
    value: "80.0%",
    what: "For 50 sampled triples, verify the source document supports the claim",
    detail: "39 supported, 2 partially, 9 no source found (of 50 checked)",
    exGood: 'Vehicle Service Contract \u2014IS_ADMINISTERED_BY\u2192 AWF Inc. \u2192 SUPPORTED',
    exBad: 'Device Protection Enrollment \u2014REQUIRES\u2192 VMI \u2192 NO SOURCE',
  },
];

z2metrics.forEach((m, i) => {
  const y = 1.1 + i * 1.45;

  s6.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y, w: 8.8, h: 1.3,
    fill: { color: C.white }, rectRadius: 0.08, shadow: makeShadow(),
  });

  // Big number
  s6.addText(m.value, {
    x: 0.8, y, w: 1.5, h: 1.3,
    fontSize: 32, fontFace: "Georgia", bold: true, color: C.accent, valign: "middle", align: "center", margin: 0,
  });

  // Name + description
  s6.addText(m.name, {
    x: 2.5, y: y + 0.08, w: 6.5, h: 0.3,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.navy, margin: 0,
  });
  s6.addText(m.what, {
    x: 2.5, y: y + 0.35, w: 6.5, h: 0.25,
    fontSize: 10, fontFace: "Calibri", color: C.text, margin: 0,
  });
  s6.addText(m.detail, {
    x: 2.5, y: y + 0.58, w: 6.5, h: 0.25,
    fontSize: 10, fontFace: "Calibri", italic: true, color: C.muted, margin: 0,
  });
  // Examples
  s6.addText("\u2705 " + m.exGood, {
    x: 2.5, y: y + 0.82, w: 6.5, h: 0.2,
    fontSize: 9, fontFace: "Consolas", color: "27AE60", margin: 0,
  });
  s6.addText("\u274C " + m.exBad, {
    x: 2.5, y: y + 1.02, w: 6.5, h: 0.2,
    fontSize: 9, fontFace: "Consolas", color: C.red, margin: 0,
  });
});

// ========================================================================
// SLIDE 7: Zone 3 Eval Part 1
// ========================================================================
let s7 = pres.addSlide();
s7.background = { color: C.lightBg };

s7.addText("Zone 3 Evaluation: Ontology Quality (1/2)", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.6,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});

// BERTScore card
s7.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.6, y: 1.1, w: 8.8, h: 2.0,
  fill: { color: C.white }, rectRadius: 0.08, shadow: makeShadow(),
});
s7.addText("BERTScore F1 = 0.617", {
  x: 0.8, y: 1.15, w: 3, h: 0.4,
  fontSize: 20, fontFace: "Georgia", bold: true, color: C.accent, margin: 0,
});
s7.addText("P = 0.758  |  R = 0.520", {
  x: 4.0, y: 1.15, w: 3, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: C.muted, margin: 0,
});
s7.addText("Semantic similarity between our induced class names and Riskine reference classes using BERT embeddings. Measures whether we discovered the right concepts, even with different names.", {
  x: 0.8, y: 1.6, w: 8.4, h: 0.4,
  fontSize: 11, fontFace: "Calibri", color: C.text, margin: 0,
});

// Alignment examples
const alignments = [
  { ours: "Organization", ref: "Organization", cos: "1.00", verdict: "MATCH", color: "27AE60" },
  { ours: "Policy", ref: "Product", cos: "0.90", verdict: "MATCH", color: "27AE60" },
  { ours: "Person", ref: "Person", cos: "1.00", verdict: "MATCH", color: "27AE60" },
  { ours: "Coverage", ref: "Coverage", cos: "1.00", verdict: "MATCH", color: "27AE60" },
  { ours: "Risk", ref: "Address", cos: "0.80", verdict: "PARTIAL", color: C.orange },
];

// Header
s7.addText("Induced", { x: 1.0, y: 2.1, w: 1.8, h: 0.3, fontSize: 10, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
s7.addText("Riskine Ref", { x: 2.8, y: 2.1, w: 1.8, h: 0.3, fontSize: 10, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
s7.addText("Cosine", { x: 4.6, y: 2.1, w: 1.0, h: 0.3, fontSize: 10, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
s7.addText("Verdict", { x: 5.6, y: 2.1, w: 1.5, h: 0.3, fontSize: 10, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });

alignments.forEach((a, i) => {
  const y = 2.4 + i * 0.22;
  s7.addText(a.ours, { x: 1.0, y, w: 1.8, h: 0.22, fontSize: 10, fontFace: "Consolas", color: C.text, margin: 0 });
  s7.addText(a.ref, { x: 2.8, y, w: 1.8, h: 0.22, fontSize: 10, fontFace: "Consolas", color: C.text, margin: 0 });
  s7.addText(a.cos, { x: 4.6, y, w: 1.0, h: 0.22, fontSize: 10, fontFace: "Consolas", color: C.text, margin: 0 });
  s7.addText(a.verdict, { x: 5.6, y, w: 1.5, h: 0.22, fontSize: 10, fontFace: "Calibri", bold: true, color: a.color, margin: 0 });
});

// Graph F1 card
s7.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.6, y: 3.4, w: 8.8, h: 1.8,
  fill: { color: C.white }, rectRadius: 0.08, shadow: makeShadow(),
});
s7.addText("Graph F1 = 0.433", {
  x: 0.8, y: 3.45, w: 3, h: 0.4,
  fontSize: 20, fontFace: "Georgia", bold: true, color: C.accent, margin: 0,
});
s7.addText("P = 0.842  |  R = 0.291", {
  x: 4.0, y: 3.45, w: 3, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: C.muted, margin: 0,
});
s7.addText("Structural comparison using Soft Graph Correspondence (2-hop neighborhood). Compares topology of induced vs reference ontology graphs.", {
  x: 0.8, y: 3.9, w: 8.4, h: 0.35,
  fontSize: 11, fontFace: "Calibri", color: C.text, margin: 0,
});
s7.addText([
  { text: "High Precision (84.2%)", options: { bold: true, color: "27AE60", breakLine: true } },
  { text: "  Our edges are correct \u2014 induced relationships match reference patterns", options: { color: C.text, breakLine: true } },
  { text: "Lower Recall (29.1%)", options: { bold: true, color: C.orange, breakLine: true } },
  { text: "  Missing many Riskine edges \u2014 reference has 46 edges vs our 26", options: { color: C.text } },
], {
  x: 0.8, y: 4.3, w: 8.4, h: 0.8,
  fontSize: 11, fontFace: "Calibri", margin: 0,
});

// ========================================================================
// SLIDE 8: Zone 3 Eval Part 2
// ========================================================================
let s8 = pres.addSlide();
s8.background = { color: C.lightBg };

s8.addText("Zone 3 Evaluation: Ontology Quality (2/2)", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.6,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});

// Wu-Palmer card
s8.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.6, y: 1.1, w: 4.2, h: 2.0,
  fill: { color: C.white }, rectRadius: 0.08, shadow: makeShadow(),
});
s8.addText("Wu-Palmer = 0.621", {
  x: 0.8, y: 1.15, w: 3.8, h: 0.4,
  fontSize: 20, fontFace: "Georgia", bold: true, color: C.accent, margin: 0,
});
s8.addText("(9 matched pairs)", {
  x: 0.8, y: 1.5, w: 3.8, h: 0.25,
  fontSize: 11, fontFace: "Calibri", color: C.muted, margin: 0,
});
s8.addText("Taxonomy distance in a shared hierarchy.\n1.0 = same node, 0.0 = completely unrelated.\n\nMeasures how close our class placement is to Riskine\u2019s hierarchy structure. Higher = better structural alignment.", {
  x: 0.8, y: 1.85, w: 3.8, h: 1.1,
  fontSize: 11, fontFace: "Calibri", color: C.text, margin: 0,
});

// Entity Assignment F1 card
s8.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 5.2, y: 1.1, w: 4.2, h: 2.0,
  fill: { color: C.white }, rectRadius: 0.08, shadow: makeShadow(),
});
s8.addText("Entity Assignment F1", {
  x: 5.4, y: 1.15, w: 3.8, h: 0.35,
  fontSize: 18, fontFace: "Georgia", bold: true, color: C.accent, margin: 0,
});
s8.addText([
  { text: "Full 26-class: ", options: { bold: true, color: C.navy } },
  { text: "0.206", options: { color: C.text, breakLine: true } },
  { text: "Present 7-class: ", options: { bold: true, color: C.navy } },
  { text: "0.404", options: { color: "27AE60" } },
], {
  x: 5.4, y: 1.55, w: 3.8, h: 0.5,
  fontSize: 13, fontFace: "Calibri", margin: 0,
});
s8.addText("Embed class members, find nearest Riskine class by centroid distance. Full = all 26 Riskine classes (many absent from data). Present = only classes with evidence.", {
  x: 5.4, y: 2.1, w: 3.8, h: 0.8,
  fontSize: 10, fontFace: "Calibri", color: C.text, margin: 0,
});

// Entity assignment examples
s8.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.6, y: 3.3, w: 8.8, h: 1.9,
  fill: { color: C.white }, rectRadius: 0.08, shadow: makeShadow(),
});
s8.addText("Entity Assignment Examples", {
  x: 0.8, y: 3.35, w: 8, h: 0.35,
  fontSize: 14, fontFace: "Calibri", bold: true, color: C.navy, margin: 0,
});

const assignments = [
  { ours: "Coverage", ref: "Coverage", cos: "0.67", score: "1.0", color: "27AE60" },
  { ours: "Risk", ref: "Risk", cos: "0.45", score: "0.5", color: "27AE60" },
  { ours: "Person", ref: "Identification", cos: "0.45", score: "0.5", color: C.orange },
  { ours: "Property", ref: "Property", cos: "0.45", score: "0.5", color: "27AE60" },
  { ours: "Organization", ref: "\u2014", cos: "0.31", score: "0.0", color: C.red },
];

s8.addText("Induced", { x: 1.0, y: 3.75, w: 1.8, h: 0.25, fontSize: 10, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
s8.addText("Best Riskine Match", { x: 3.0, y: 3.75, w: 2.0, h: 0.25, fontSize: 10, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
s8.addText("Cosine", { x: 5.2, y: 3.75, w: 1.0, h: 0.25, fontSize: 10, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });
s8.addText("Score", { x: 6.4, y: 3.75, w: 1.0, h: 0.25, fontSize: 10, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });

assignments.forEach((a, i) => {
  const y = 4.05 + i * 0.25;
  s8.addText(a.ours, { x: 1.0, y, w: 1.8, h: 0.25, fontSize: 10, fontFace: "Consolas", color: C.text, margin: 0 });
  s8.addText(a.ref, { x: 3.0, y, w: 2.0, h: 0.25, fontSize: 10, fontFace: "Consolas", color: C.text, margin: 0 });
  s8.addText(a.cos, { x: 5.2, y, w: 1.0, h: 0.25, fontSize: 10, fontFace: "Consolas", color: C.text, margin: 0 });
  s8.addText(a.score, { x: 6.4, y, w: 1.0, h: 0.25, fontSize: 10, fontFace: "Calibri", bold: true, color: a.color, margin: 0 });
});

// ========================================================================
// SLIDE 9: Query Showcase
// ========================================================================
let s9 = pres.addSlide();
s9.background = { color: C.white };

s9.addText("Query Showcase: KG vs Ground Truth", {
  x: 0.6, y: 0.2, w: 8.8, h: 0.5,
  fontSize: 26, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});
s9.addText("Can the Knowledge Graph answer real business questions?", {
  x: 0.6, y: 0.65, w: 8.8, h: 0.3,
  fontSize: 13, fontFace: "Calibri", italic: true, color: C.muted, margin: 0,
});

const queries = [
  {
    q: "Q1: Which claims had the longest resolution time?",
    kg: "61,606 hours (Physical Damage, Cellular Phone)",
    gt: "61,606 hours (tmobilechatsurveysample.csv)",
    match: "100%", color: "27AE60",
  },
  {
    q: "Q2: What device types have the most claims?",
    kg: "Cellular Phone: 274 | (Unknown): 3 | Smart Watch: 1",
    gt: "Cellular Phone: 287 | (Unknown): 3 | Smart Watch: 1",
    match: "96%", color: "27AE60",
  },
  {
    q: "Q5: What loss types are most common?",
    kg: "Physical Damage: 144 | Missing/Lost: 22 | Theft: 7 | Mech. Breakdown: 5",
    gt: "Physical Damage: 151 | Missing/Lost: 22 | Theft: 7 | Mech. Breakdown: 5",
    match: "96%", color: "27AE60",
  },
];

queries.forEach((qr, i) => {
  const y = 1.1 + i * 1.4;

  s9.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y, w: 8.8, h: 1.25,
    fill: { color: C.lightBg }, rectRadius: 0.08, shadow: makeShadow(),
  });

  // Match badge
  s9.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 8.3, y: y + 0.08, w: 0.9, h: 0.3,
    fill: { color: qr.color }, rectRadius: 0.12,
  });
  s9.addText(qr.match, {
    x: 8.3, y: y + 0.08, w: 0.9, h: 0.3,
    fontSize: 11, fontFace: "Calibri", bold: true, color: C.white, align: "center", valign: "middle", margin: 0,
  });

  s9.addText(qr.q, {
    x: 0.8, y: y + 0.05, w: 7.3, h: 0.3,
    fontSize: 12, fontFace: "Calibri", bold: true, color: C.navy, margin: 0,
  });
  s9.addText("KG Result:  " + qr.kg, {
    x: 0.8, y: y + 0.4, w: 8.2, h: 0.3,
    fontSize: 10, fontFace: "Consolas", color: C.accent, margin: 0,
  });
  s9.addText("Ground Truth:  " + qr.gt, {
    x: 0.8, y: y + 0.7, w: 8.2, h: 0.3,
    fontSize: 10, fontFace: "Consolas", color: C.muted, margin: 0,
  });
});

// Bottom note
s9.addText("Queries executed via Cypher against Neo4j. Ground truth from raw CSV files (pandas verification).", {
  x: 0.6, y: 5.1, w: 8.8, h: 0.3,
  fontSize: 10, fontFace: "Calibri", italic: true, color: C.muted, margin: 0,
});

// ========================================================================
// SLIDE 10: Results Summary
// ========================================================================
let s10 = pres.addSlide();
s10.background = { color: C.navy };

s10.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

s10.addText("Results Summary", {
  x: 0.6, y: 0.25, w: 8.8, h: 0.5,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.white, margin: 0,
});

// Left: metrics table
const metricsData = [
  ["Metric", "Value"],
  ["Entities", "5,255"],
  ["Relationships", "27,380"],
  ["Ontology Classes", "9"],
  ["Triple Precision", "91.1%"],
  ["Fact Recall", "73.2%"],
  ["Source Grounding", "80.0%"],
  ["BERTScore F1", "0.617"],
  ["Graph F1", "0.433"],
  ["Riskine Class F1", "0.485"],
  ["Entity Assign. F1 (present)", "0.404"],
  ["Backbone Coverage", "100%"],
  ["Duplication Rate", "0.0%"],
];

const tableRows = metricsData.map((row, i) => {
  if (i === 0) {
    return row.map(cell => ({
      text: cell,
      options: { bold: true, color: "FFFFFF", fill: { color: C.accent }, fontSize: 11, fontFace: "Calibri" }
    }));
  }
  return row.map((cell, j) => ({
    text: cell,
    options: {
      color: i % 2 === 0 ? "CADCFC" : "FFFFFF",
      fill: { color: i % 2 === 0 ? "1A2555" : "0F1B3D" },
      fontSize: 11, fontFace: j === 1 ? "Consolas" : "Calibri",
      bold: j === 1,
    }
  }));
});

s10.addTable(tableRows, {
  x: 0.6, y: 0.95, w: 4.5,
  colW: [3.0, 1.5],
  border: { pt: 0.5, color: "2A3B6E" },
});

// Right: induced classes
s10.addText("Induced Ontology Classes", {
  x: 5.5, y: 0.95, w: 4, h: 0.4,
  fontSize: 16, fontFace: "Calibri", bold: true, color: C.accent, margin: 0,
});

const classes = [
  { name: "Claim", count: 598, match: "Coverage" },
  { name: "Service", count: 351, match: "BusinessProcess" },
  { name: "Person", count: 238, match: "Person" },
  { name: "Document", count: 210, match: "\u2014" },
  { name: "Policy", count: 155, match: "Product" },
  { name: "Property", count: 102, match: "Property" },
  { name: "Organization", count: 55, match: "Organization" },
  { name: "Coverage", count: 46, match: "Coverage" },
  { name: "Risk", count: 33, match: "Address" },
];

classes.forEach((cl, i) => {
  const y = 1.45 + i * 0.38;
  s10.addText(cl.name, {
    x: 5.5, y, w: 1.8, h: 0.3,
    fontSize: 12, fontFace: "Calibri", bold: true, color: C.white, margin: 0,
  });
  s10.addText(`${cl.count}`, {
    x: 7.3, y, w: 0.6, h: 0.3,
    fontSize: 11, fontFace: "Consolas", color: C.ice, align: "right", margin: 0,
  });
  s10.addText(`\u2192 ${cl.match}`, {
    x: 8.1, y, w: 1.5, h: 0.3,
    fontSize: 10, fontFace: "Calibri", color: C.gray, margin: 0,
  });
});

// ========================================================================
// SLIDE 11: Technical Innovations
// ========================================================================
let s11 = pres.addSlide();
s11.background = { color: C.lightBg };

s11.addText("Technical Innovations", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.6,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.navy, margin: 0,
});

const innovations = [
  { num: "1", title: "Data-Driven Property Collapse", desc: "Measure/dimension/ID classification with zero hardcoded names. Scales to any domain." },
  { num: "2", title: "Dual-Path Extraction", desc: "LLM for unstructured PDFs + deterministic mapper for structured CSVs. Best of both worlds." },
  { num: "3", title: "SV-LOI Fusion", desc: "LLM semantic typing + graph-structural verification with disagreement arbitration." },
  { num: "4", title: "LLM Pairwise Taxonomy", desc: "N\u00D7(N-1) ordered pair evaluation with DAG enforcement and max depth 3." },
  { num: "5", title: "Record Decomposition", desc: "Record entities \u2192 domain-specific sub-nodes via relation-range mapping." },
  { num: "6", title: "Versioned Graph Cache", desc: "Staleness detection by mtime + normalization version. Auto-rebuilds on code changes." },
  { num: "7", title: "Scale-Adaptive Thresholds", desc: "Fixed floors for intrinsic properties, fraction-of-class for relative decisions." },
];

innovations.forEach((inn, i) => {
  const col = i < 4 ? 0 : 1;
  const row = i < 4 ? i : i - 4;
  const x = 0.6 + col * 4.5;
  const y = 1.1 + row * 1.05;

  s11.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w: 4.2, h: 0.9,
    fill: { color: C.white }, rectRadius: 0.06, shadow: makeShadow(),
  });
  s11.addShape(pres.shapes.OVAL, {
    x: x + 0.12, y: y + 0.2, w: 0.5, h: 0.5,
    fill: { color: C.accent },
  });
  s11.addText(inn.num, {
    x: x + 0.12, y: y + 0.2, w: 0.5, h: 0.5,
    fontSize: 16, fontFace: "Georgia", bold: true, color: C.white, align: "center", valign: "middle", margin: 0,
  });
  s11.addText(inn.title, {
    x: x + 0.75, y: y + 0.08, w: 3.2, h: 0.3,
    fontSize: 13, fontFace: "Calibri", bold: true, color: C.navy, margin: 0,
  });
  s11.addText(inn.desc, {
    x: x + 0.75, y: y + 0.4, w: 3.2, h: 0.45,
    fontSize: 10, fontFace: "Calibri", color: C.text, margin: 0,
  });
});

// ========================================================================
// SLIDE 12: Future Work & Conclusion
// ========================================================================
let s12 = pres.addSlide();
s12.background = { color: C.navy };

s12.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

s12.addText("Future Work & Conclusion", {
  x: 0.6, y: 0.3, w: 8.8, h: 0.5,
  fontSize: 28, fontFace: "Georgia", bold: true, color: C.white, margin: 0,
});

// Future work
s12.addText("Future Work", {
  x: 0.6, y: 1.0, w: 4, h: 0.35,
  fontSize: 18, fontFace: "Calibri", bold: true, color: C.accent, margin: 0,
});

const futures = [
  "Cross-domain transfer: auto insurance \u2192 flood insurance",
  "Variant comparison: SV-LOI vs AutoSchemaKG vs OLLM baselines",
  'Reduce "Other" fraction (56.5%) via improved value typing',
  "Publication target: peer-reviewed venue",
];
futures.forEach((f, i) => {
  s12.addText([
    { text: "\u25B8 ", options: { color: C.accent } },
    { text: f, options: { color: C.ice } },
  ], {
    x: 0.8, y: 1.45 + i * 0.4, w: 8, h: 0.35,
    fontSize: 13, fontFace: "Calibri", margin: 0,
  });
});

// Conclusion box
s12.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.6, y: 3.2, w: 8.8, h: 1.8,
  fill: { color: "1A2555" }, rectRadius: 0.1,
  line: { color: C.accent, width: 1.5 },
});
s12.addText("Conclusion", {
  x: 0.8, y: 3.3, w: 8, h: 0.35,
  fontSize: 18, fontFace: "Calibri", bold: true, color: C.accent, margin: 0,
});
s12.addText("SV-LOI demonstrates that fusing LLM semantic typing with graph-structural verification produces higher-quality ontologies than either signal alone. The pipeline processes heterogeneous insurance data (PDFs + CSVs) into a queryable knowledge graph with 91% extraction precision and domain-aligned ontology classes \u2014 all without manual ontology engineering.", {
  x: 0.8, y: 3.7, w: 8.4, h: 1.1,
  fontSize: 13, fontFace: "Calibri", color: C.ice, margin: 0,
});

// Footer
s12.addText("Thank You  |  Questions?", {
  x: 0, y: 5.1, w: 10, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: C.gray, align: "center", margin: 0,
});

// ========================================================================
// Save
// ========================================================================
pres.writeFile({ fileName: "/Users/sam/Documents/School/Emory/CS584_AI_Capstone/presentation/capstone_showcase.pptx" })
  .then(() => console.log("Created: presentation/capstone_showcase.pptx"))
  .catch(err => console.error("Error:", err));
