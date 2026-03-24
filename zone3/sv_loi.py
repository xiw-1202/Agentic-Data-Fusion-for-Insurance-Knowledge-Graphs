"""Zone 3 — SV-LOI: Structurally-Verified LLM Ontology Induction

Novel ontology induction method that fuses LLM semantic typing with
graph-structural verification. Neither signal alone is sufficient:
LLM typing is accurate but inconsistent across batches; structural
clustering is consistent but over-fragments.

Key insight: entities of the same ontological class share two independent
signals — (1) the LLM recognizes what they ARE (semantic), and
(2) they participate in the same types of relations (structural).
Fusing both with disagreement arbitration eliminates each signal's
failure mode.

Algorithm:
  Phase 1 — LLM Entity Typing:
    Batch entities (20/prompt) with name + type + top relations.
    LLM assigns ontology class from a discovered class vocabulary.
    ~70 LLM calls for 1,351 entities.

  Phase 2 — Structural Consensus Verification:
    Build relation-signature vectors per entity.
    Compute class centroid for each LLM-assigned class.
    Flag entities whose signature deviates >2σ from their class centroid.

  Phase 3 — Disagreement Arbitration:
    For flagged entities, re-query LLM with enriched structural context:
    "You typed this as X, but it structurally resembles Y. Which is correct?"

  Phase 4 — Hierarchy Derivation:
    LLM proposes SUBCLASS_OF from final class set.
    Write ontology layer to Neo4j.

Usage:
  python3 zone3/sv_loi.py
  python3 zone3/sv_loi.py --model qwen2.5:72b
  python3 zone3/sv_loi.py --model qwen2.5:72b --suffix zone3_svloi

Pre-requisite: Zone 2 must have run first (Neo4j populated with :Entity nodes).
"""

from __future__ import annotations

import json
import re
import time
import os
import sys
import argparse
from typing import Optional, Union
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 15          # entities per LLM typing prompt (smaller = more context per entity)
MAX_MEMBERS_IN_PROMPT = 15
MIN_CLASS_SIZE = 3       # classes with fewer members get merged into nearest
DEVIATION_THRESHOLD = 2.0  # σ threshold for structural flagging
MAX_ARBITRATION_BATCH = 10  # entities per arbitration prompt
MAX_CLASS_FRACTION = 0.30  # no single class should exceed 30% of entities
ENTITY_SAMPLES_PER_TYPE = 8  # entity name examples shown in discovery prompt

# Target class count range (guide for class discovery)
TARGET_CLASSES_MIN = 10
TARGET_CLASSES_MAX = 25


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_llm(model: str) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
    )


def get_neo4j_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )


def _sanitize_label(name: str) -> str:
    """Make a PascalCase name safe for Neo4j label."""
    cleaned = re.sub(r'[^A-Za-z0-9]', '', name.strip())
    if not cleaned:
        return "UnknownClass"
    if cleaned[0].isdigit():
        cleaned = "Class" + cleaned
    return cleaned


def _parse_json_safely(text: str) -> Union[dict, list]:
    """Try to parse JSON from LLM output with fallbacks."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL) or re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _invoke_llm(llm: ChatOllama, prompt: str) -> str:
    """Call LLM and return content string."""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        print(f"    [llm] Error: {e}")
        return ""


# ---------------------------------------------------------------------------
# Step 1: Load Entities from Neo4j (reused from RSI-LCR)
# ---------------------------------------------------------------------------

def load_entities(graph: Neo4jGraph) -> list[dict]:
    """Load all Entity nodes with structural context."""
    print("\n[Phase 0] Loading entities from Neo4j...", flush=True)

    rows = graph.query("""
        MATCH (n:Entity)
        OPTIONAL MATCH (n)-[r]->(m:Entity)
        OPTIONAL MATCH (p:Entity)-[s]->(n)
        RETURN n.id AS id,
               n.entity_type AS entity_type,
               collect(DISTINCT {rel: type(r), target: m.id, target_type: m.entity_type}) AS out_rels,
               collect(DISTINCT {rel: type(s), source: p.id, source_type: p.entity_type}) AS in_rels
    """)
    if not rows:
        print("  ✗ No Entity nodes found. Run zone2/pipeline.py first.")
        return []

    entities: list[dict] = []
    for row in rows:
        eid = row["id"]
        etype = row.get("entity_type") or "Unknown"

        out_rels = [r for r in row.get("out_rels", []) if r.get("rel")]
        in_rels = [r for r in row.get("in_rels", []) if r.get("rel")]

        # Relation summaries for LLM prompt
        out_summary = []
        for r in out_rels[:5]:
            out_summary.append(f"--{r['rel']}--> {r.get('target', '?')}")
        in_summary = []
        for r in in_rels[:5]:
            in_summary.append(f"<--{r['rel']}-- {r.get('source', '?')}")

        # Relation type counts for structural signature
        out_counts: dict[str, int] = defaultdict(int)
        in_counts: dict[str, int] = defaultdict(int)
        for r in out_rels:
            out_counts[r["rel"]] += 1
        for r in in_rels:
            in_counts[r["rel"]] += 1

        entities.append({
            "id": eid,
            "entity_type": etype,
            "out_rels": out_rels,
            "in_rels": in_rels,
            "out_summary": out_summary,
            "in_summary": in_summary,
            "out_rel_counts": dict(out_counts),
            "in_rel_counts": dict(in_counts),
            "degree": len(out_rels) + len(in_rels),
        })

    typed = sum(1 for e in entities if e["entity_type"] != "Unknown")
    print(f"  ✓ {len(entities)} entities loaded ({typed} typed, {len(entities) - typed} untyped)")
    return entities


# ---------------------------------------------------------------------------
# Phase 1: Class Discovery + Batched LLM Entity Typing
# ---------------------------------------------------------------------------

def discover_class_vocabulary(entities: list[dict], llm: ChatOllama) -> list[str]:
    """Discover ontology classes by showing LLM actual entity examples + relations.

    Key improvement over v1: shows entity NAME examples (not just type stats)
    and explicitly steers toward domain concepts over data types.
    """
    print("\n[Phase 1a] Discovering class vocabulary from entity examples...", flush=True)

    # Group entities by type and collect name samples
    type_groups: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        type_groups[e["entity_type"]].append(e["id"])

    # Build rich entity type descriptions with NAME examples
    type_descriptions = []
    for etype, members in sorted(type_groups.items(), key=lambda x: -len(x[1])):
        if etype in ("Unknown", "unknown"):
            continue
        # Sample diverse entity names (not just first N — sample from spread)
        sample_size = min(ENTITY_SAMPLES_PER_TYPE, len(members))
        indices = np.linspace(0, len(members) - 1, sample_size, dtype=int)
        samples = [members[i] for i in indices]
        type_descriptions.append(
            f"  {etype} ({len(members)} entities): {', '.join(samples)}"
        )

    # Collect relation types
    rel_counts: Counter = Counter()
    for e in entities:
        for r in e.get("out_rel_counts", {}):
            rel_counts[r] += 1
        for r in e.get("in_rel_counts", {}):
            rel_counts[r] += 1
    top_rels = rel_counts.most_common(25)

    # Sample concrete triples (with real entity names, not just types)
    import random
    random.seed(42)
    triple_examples = []
    sampled = random.sample(entities, min(100, len(entities)))
    for e in sampled:
        for r in e.get("out_rels", [])[:3]:
            target = r.get("target", "?")
            triple_examples.append(f"  {e['id']} --{r['rel']}--> {target}")
            if len(triple_examples) >= 30:
                break
        if len(triple_examples) >= 30:
            break

    prompt = f"""You are an ontology engineer designing a domain ontology for a knowledge graph.

ENTITY TYPES with example entity names:
{chr(10).join(type_descriptions[:20])}

RELATION TYPES (most common):
{chr(10).join(f'  {r} ({c}x)' for r, c in top_rels)}

EXAMPLE TRIPLES (entity --relation--> entity):
{chr(10).join(triple_examples)}

Based on these REAL entities and their relationships, propose {TARGET_CLASSES_MIN}-{TARGET_CLASSES_MAX} \
ontology classes that capture the DOMAIN CONCEPTS represented by these entities.

CRITICAL RULES:
- Propose classes for WHAT entities ARE in the real world (e.g., Person, Organization, \
Product, Coverage, Risk, Location, Document, Event) — NOT data types (e.g., Amount, \
Number, Date, Text, Measurement, String)
- Each class = a distinct real-world concept with a clear role in the domain
- Use single-word PascalCase names (e.g., Coverage, Person, Risk, Structure, Product)
- Avoid compound names like "InsuredProperty" or "FinancialAmount" — prefer the simpler \
word that captures the core concept (Property, Amount)
- Every entity in the graph should fit into exactly one class
- Aim for balanced classes — no class should contain more than 30% of all entities

Output as JSON array:
[{{"name": "ClassName", "definition": "one-line definition of what entities belong here"}}, ...]
"""
    raw = _invoke_llm(llm, prompt)
    parsed = _parse_json_safely(raw)

    if isinstance(parsed, list):
        classes = [item["name"] for item in parsed if isinstance(item, dict) and "name" in item]
    elif isinstance(parsed, dict):
        classes = list(parsed.keys())
    else:
        classes = []

    # Sanitize
    classes = [_sanitize_label(c) for c in classes if c]

    # Deduplicate (case-insensitive)
    seen_lower: set[str] = set()
    deduped = []
    for c in classes:
        if c.lower() not in seen_lower:
            deduped.append(c)
            seen_lower.add(c.lower())
    classes = deduped

    # Ensure minimum set — fallback to entity type names
    if len(classes) < TARGET_CLASSES_MIN:
        for t, _ in sorted(type_groups.items(), key=lambda x: -len(x[1])):
            if t not in ("Unknown", "unknown") and _sanitize_label(t) not in classes:
                classes.append(_sanitize_label(t))
            if len(classes) >= TARGET_CLASSES_MIN:
                break

    # Always include "Other" for unclassifiable entities
    if "Other" not in classes:
        classes.append("Other")

    print(f"  ✓ Discovered {len(classes) - 1} classes + Other:", flush=True)
    for c in classes:
        print(f"    - {c}", flush=True)
    return classes


def batch_type_entities(
    entities: list[dict],
    class_vocab: list[str],
    llm: ChatOllama,
) -> dict[str, str]:
    """Assign ontology class to each entity via batched LLM prompts.

    Returns:
        {entity_id: class_name}
    """
    print(f"\n[Phase 1b] Batched LLM entity typing ({len(entities)} entities, batch={BATCH_SIZE})...")

    class_list = ", ".join(c for c in class_vocab if c != "Other")
    assignments: dict[str, str] = {}
    n_batches = (len(entities) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        batch = entities[start:start + BATCH_SIZE]

        entity_descriptions = []
        for e in batch:
            desc = f"- {e['id']} (extracted type: {e['entity_type']})"
            if e["out_summary"]:
                desc += f"\n    outgoing: {'; '.join(e['out_summary'][:5])}"
            if e["in_summary"]:
                desc += f"\n    incoming: {'; '.join(e['in_summary'][:3])}"
            entity_descriptions.append(desc)

        prompt = f"""Classify each entity into the ontology class that best describes \
WHAT IT IS in the real world.

AVAILABLE CLASSES: {class_list}, Other

Think about each entity: Is it a person? An organization? A type of coverage or product? \
A physical structure or location? A risk or peril? Classify by the entity's ROLE in the \
domain, not by its data format.

ENTITIES:
{chr(10).join(entity_descriptions)}

For each entity, output exactly one line: entity_name -> ClassName
Use "Other" only if the entity truly does not fit any class.
"""
        raw = _invoke_llm(llm, prompt)

        # Parse "entity_id -> ClassName" lines
        line_re = re.compile(r'^[- ]*(.+?)\s*->\s*(\w+)\s*$', re.MULTILINE)
        for m in line_re.finditer(raw):
            eid = m.group(1).strip().strip('"').strip("'")
            cls = m.group(2).strip()
            # Validate class name
            sanitized = _sanitize_label(cls)
            if sanitized in class_vocab or sanitized == "Other":
                assignments[eid] = sanitized
            else:
                # Find closest class by prefix match
                for cv in class_vocab:
                    if cv.lower() == sanitized.lower():
                        assignments[eid] = cv
                        break
                else:
                    assignments[eid] = "Other"

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"    Batch {batch_idx + 1}/{n_batches}: {len(assignments)} assigned so far")

    # Assign "Other" to any untyped entities
    for e in entities:
        if e["id"] not in assignments:
            assignments[e["id"]] = "Other"

    # Distribution
    dist = Counter(assignments.values())
    print(f"  ✓ Typed {len(assignments)} entities across {len(dist)} classes", flush=True)
    for cls, cnt in dist.most_common():
        print(f"    {cls}: {cnt}", flush=True)

    return assignments


def rebalance_mega_classes(
    assignments: dict[str, str],
    entities: list[dict],
    class_vocab: list[str],
    llm: ChatOllama,
    max_fraction: float = MAX_CLASS_FRACTION,
) -> tuple[dict[str, str], list[str]]:
    """Split any class that exceeds max_fraction of total entities.

    Uses LLM to propose sub-classes for the mega-class members.
    Returns updated assignments and updated class vocabulary.
    """
    total = len(assignments)
    threshold = int(total * max_fraction)
    dist = Counter(assignments.values())
    updated_vocab = list(class_vocab)

    mega_classes = [(cls, cnt) for cls, cnt in dist.items()
                    if cnt > threshold and cls != "Other"]

    if not mega_classes:
        print("\n[Phase 1c] No mega-classes detected — skipping rebalance.", flush=True)
        return assignments, updated_vocab

    print(f"\n[Phase 1c] Rebalancing {len(mega_classes)} mega-classes (threshold={threshold})...", flush=True)
    entity_map = {e["id"]: e for e in entities}
    updated = dict(assignments)

    for mega_cls, mega_cnt in mega_classes:
        print(f"  Splitting {mega_cls} ({mega_cnt} members)...", flush=True)

        # Get members with context
        members = [eid for eid, c in updated.items() if c == mega_cls]
        # Sample for the LLM
        import random
        random.seed(42)
        sample = random.sample(members, min(40, len(members)))
        sample_descs = []
        for eid in sample:
            e = entity_map.get(eid, {})
            desc = f"  {eid}"
            if e.get("out_summary"):
                desc += f" (rels: {'; '.join(e['out_summary'][:2])})"
            sample_descs.append(desc)

        other_classes = [c for c in updated_vocab if c != mega_cls and c != "Other"]

        prompt = f"""The ontology class "{mega_cls}" contains {mega_cnt} entities — too many for a single class.

Here are example members:
{chr(10).join(sample_descs)}

Other existing classes: {', '.join(other_classes)}

These entities probably represent DIFFERENT real-world concepts that were lumped together. \
Propose 2-4 sub-classes to replace "{mega_cls}". Each sub-class should capture a distinct \
domain concept.

RULES:
- Use PascalCase single-word names
- Propose concepts for WHAT these entities ARE (e.g., Person, Product, Coverage, Risk, \
Structure, Document, Event) — not data types
- Some members might actually belong to existing classes listed above — that's fine
- Output JSON: [{{"name": "SubClass1", "definition": "...", "example_members": ["entity1", "entity2"]}}]
"""
        raw = _invoke_llm(llm, prompt)
        parsed = _parse_json_safely(raw)

        if not isinstance(parsed, list) or len(parsed) < 2:
            print(f"    ✗ Could not split {mega_cls} — LLM did not propose sub-classes", flush=True)
            continue

        new_classes = []
        for item in parsed:
            if isinstance(item, dict) and "name" in item:
                name = _sanitize_label(item["name"])
                if name and name != mega_cls and name not in updated_vocab:
                    new_classes.append(name)
                    updated_vocab.append(name)

        if not new_classes:
            print(f"    ✗ No valid new classes proposed for {mega_cls}", flush=True)
            continue

        # Remove mega_cls from vocab, replace with new classes
        if mega_cls in updated_vocab:
            updated_vocab.remove(mega_cls)

        # Re-type the mega-class members using new sub-classes + existing classes
        all_target_classes = new_classes + other_classes
        retype_vocab = all_target_classes + ["Other"]

        # Re-type in batches
        n_batches = (len(members) + BATCH_SIZE - 1) // BATCH_SIZE
        retype_class_list = ", ".join(all_target_classes)

        for bi in range(n_batches):
            batch = members[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            batch_descs = []
            for eid in batch:
                e = entity_map.get(eid, {})
                desc = f"- {eid} (type: {e.get('entity_type', '?')})"
                if e.get("out_summary"):
                    desc += f"\n    rels: {'; '.join(e['out_summary'][:4])}"
                batch_descs.append(desc)

            rprompt = f"""Classify each entity by WHAT IT IS in the real world.

CLASSES: {retype_class_list}, Other

ENTITIES:
{chr(10).join(batch_descs)}

Output: entity_name -> ClassName
"""
            raw2 = _invoke_llm(llm, rprompt)
            line_re = re.compile(r'^[- ]*(.+?)\s*->\s*(\w+)\s*$', re.MULTILINE)
            for m in line_re.finditer(raw2):
                eid = m.group(1).strip().strip('"').strip("'")
                cls = _sanitize_label(m.group(2).strip())
                if eid in updated and (cls in updated_vocab or cls == "Other"):
                    updated[eid] = cls

        new_dist = Counter(updated[m] for m in members)
        print(f"    ✓ Split into: {dict(new_dist.most_common())}", flush=True)

    return updated, updated_vocab


# ---------------------------------------------------------------------------
# Phase 2: Structural Consensus Verification
# ---------------------------------------------------------------------------

def build_structural_signatures(entities: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """Build relation signature feature vectors.

    Returns:
        features: (n_entities, n_features) L2-normalized matrix
        entity_ids: ordered list
        feature_names: column names
    """
    all_rel_types: set[str] = set()
    all_entity_types: set[str] = set()
    for e in entities:
        all_rel_types.update(e.get("out_rel_counts", {}).keys())
        all_rel_types.update(e.get("in_rel_counts", {}).keys())
        all_entity_types.add(e["entity_type"])

    rel_types = sorted(all_rel_types)
    ent_types = sorted(all_entity_types)

    # Feature columns: [rel_OUT, rel_IN for each rel_type] + [entity_type one-hot]
    feature_names = []
    for rt in rel_types:
        feature_names.append(f"{rt}_OUT")
        feature_names.append(f"{rt}_IN")
    for et in ent_types:
        feature_names.append(f"type_{et}")

    n = len(entities)
    d = len(feature_names)
    features = np.zeros((n, d))

    entity_ids = []
    for i, e in enumerate(entities):
        entity_ids.append(e["id"])
        for rt in rel_types:
            j_out = feature_names.index(f"{rt}_OUT")
            j_in = feature_names.index(f"{rt}_IN")
            features[i, j_out] = e.get("out_rel_counts", {}).get(rt, 0)
            features[i, j_in] = e.get("in_rel_counts", {}).get(rt, 0)
        et = e["entity_type"]
        if f"type_{et}" in feature_names:
            features[i, feature_names.index(f"type_{et}")] = 1.0

    # L2 normalize
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    features = features / norms

    return features, entity_ids, feature_names


def structural_consensus_check(
    assignments: dict[str, str],
    features: np.ndarray,
    entity_ids: list[str],
) -> tuple[dict[str, dict], list[dict]]:
    """Verify LLM assignments against structural signatures.

    For each class, compute the centroid of member signatures.
    Flag entities whose cosine distance from centroid > DEVIATION_THRESHOLD * σ.

    Returns:
        class_stats: {class_name: {centroid, mean_sim, std_sim, member_count}}
        flagged: [{entity_id, assigned_class, similarity, nearest_class, nearest_sim}]
    """
    print(f"\n[Phase 2] Structural consensus verification (σ threshold={DEVIATION_THRESHOLD})...")

    eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

    # Group entities by assigned class
    class_members: dict[str, list[str]] = defaultdict(list)
    for eid, cls in assignments.items():
        class_members[cls].append(eid)

    # Compute class centroids
    class_stats: dict[str, dict] = {}
    for cls, members in class_members.items():
        if cls == "Other":
            continue
        indices = [eid_to_idx[m] for m in members if m in eid_to_idx]
        if not indices:
            continue

        member_features = features[indices]
        centroid = member_features.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # Cosine similarities to centroid
        sims = member_features @ centroid
        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims)) if len(sims) > 1 else 0.0

        class_stats[cls] = {
            "centroid": centroid,
            "mean_sim": mean_sim,
            "std_sim": std_sim,
            "member_count": len(indices),
        }

    # Flag outliers
    flagged: list[dict] = []
    all_centroids = {cls: stats["centroid"] for cls, stats in class_stats.items()}

    for eid, cls in assignments.items():
        if cls == "Other" or cls not in class_stats:
            continue
        idx = eid_to_idx.get(eid)
        if idx is None:
            continue

        stats = class_stats[cls]
        sim = float(features[idx] @ stats["centroid"])
        threshold = stats["mean_sim"] - DEVIATION_THRESHOLD * max(stats["std_sim"], 0.05)

        if sim < threshold:
            # Find nearest alternative class
            best_alt_cls = None
            best_alt_sim = -1.0
            for alt_cls, alt_centroid in all_centroids.items():
                if alt_cls == cls:
                    continue
                alt_sim = float(features[idx] @ alt_centroid)
                if alt_sim > best_alt_sim:
                    best_alt_sim = alt_sim
                    best_alt_cls = alt_cls

            flagged.append({
                "entity_id": eid,
                "assigned_class": cls,
                "similarity": round(sim, 4),
                "class_mean_sim": round(stats["mean_sim"], 4),
                "class_std_sim": round(stats["std_sim"], 4),
                "nearest_class": best_alt_cls,
                "nearest_sim": round(best_alt_sim, 4),
            })

    print(f"  ✓ {len(flagged)} entities flagged for arbitration ({len(flagged)}/{len(assignments)} = {100*len(flagged)/max(len(assignments),1):.1f}%)")
    for cls, stats in sorted(class_stats.items(), key=lambda x: x[1]["member_count"], reverse=True):
        n_flagged = sum(1 for f in flagged if f["assigned_class"] == cls)
        print(f"    {cls}: {stats['member_count']} members, mean_sim={stats['mean_sim']:.3f}, flagged={n_flagged}")

    return class_stats, flagged


# ---------------------------------------------------------------------------
# Phase 3: Disagreement Arbitration
# ---------------------------------------------------------------------------

def arbitrate_disagreements(
    flagged: list[dict],
    entities: list[dict],
    assignments: dict[str, str],
    class_vocab: list[str],
    llm: ChatOllama,
) -> dict[str, str]:
    """Re-query LLM for flagged entities with enriched structural context.

    Returns updated assignments dict.
    """
    if not flagged:
        print("\n[Phase 3] No disagreements to arbitrate.")
        return assignments

    print(f"\n[Phase 3] Arbitrating {len(flagged)} disagreements...")

    entity_map = {e["id"]: e for e in entities}
    updated = dict(assignments)
    changes = 0

    # Batch flagged entities
    n_batches = (len(flagged) + MAX_ARBITRATION_BATCH - 1) // MAX_ARBITRATION_BATCH

    for batch_idx in range(n_batches):
        start = batch_idx * MAX_ARBITRATION_BATCH
        batch = flagged[start:start + MAX_ARBITRATION_BATCH]

        entity_sections = []
        for f in batch:
            eid = f["entity_id"]
            e = entity_map.get(eid, {})

            section = (
                f"Entity: {eid}\n"
                f"  Current type: {e.get('entity_type', '?')}\n"
                f"  Assigned class: {f['assigned_class']} (structural sim: {f['similarity']:.3f})\n"
                f"  Nearest alternative: {f['nearest_class']} (structural sim: {f['nearest_sim']:.3f})\n"
            )
            if e.get("out_summary"):
                section += f"  Outgoing: {'; '.join(e['out_summary'][:3])}\n"
            if e.get("in_summary"):
                section += f"  Incoming: {'; '.join(e['in_summary'][:2])}\n"
            entity_sections.append(section)

        class_list = ", ".join(c for c in class_vocab if c != "Other")
        prompt = f"""You are verifying ontology class assignments. Some entities were flagged because
their structural relationships don't match their assigned class.

AVAILABLE CLASSES: {class_list}, Other

FLAGGED ENTITIES:
{chr(10).join(entity_sections)}

For each entity, decide the CORRECT class based on both its name/type AND its relationships.
Output one line per entity: entity_id -> CorrectClass
"""
        raw = _invoke_llm(llm, prompt)

        line_re = re.compile(r'^[- ]*(.+?)\s*->\s*(\w+)\s*$', re.MULTILINE)
        for m in line_re.finditer(raw):
            eid = m.group(1).strip().strip('"').strip("'")
            cls = _sanitize_label(m.group(2).strip())
            if eid in updated:
                old = updated[eid]
                if cls in class_vocab or cls == "Other":
                    if cls != old:
                        updated[eid] = cls
                        changes += 1

    print(f"  ✓ Arbitration complete: {changes} entities reassigned")
    return updated


# ---------------------------------------------------------------------------
# Phase 4: Post-processing — merge small classes, derive hierarchy
# ---------------------------------------------------------------------------

def merge_small_classes(
    assignments: dict[str, str],
    features: np.ndarray,
    entity_ids: list[str],
    min_size: int = MIN_CLASS_SIZE,
) -> dict[str, str]:
    """Merge classes with fewer than min_size members into the nearest class."""
    print(f"\n[Phase 4a] Merging small classes (min_size={min_size})...")

    eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

    # Count class sizes
    class_counts = Counter(assignments.values())
    small_classes = [c for c, n in class_counts.items() if n < min_size and c != "Other"]

    if not small_classes:
        print("  ✓ No small classes to merge")
        return assignments

    # Compute centroids for non-small classes
    centroids: dict[str, np.ndarray] = {}
    for cls, cnt in class_counts.items():
        if cls in small_classes or cls == "Other":
            continue
        indices = [eid_to_idx[eid] for eid, c in assignments.items() if c == cls and eid in eid_to_idx]
        if not indices:
            continue
        centroid = features[indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[cls] = centroid

    updated = dict(assignments)
    merge_map: dict[str, str] = {}

    for small_cls in small_classes:
        # Find nearest large class by centroid similarity
        indices = [eid_to_idx[eid] for eid, c in assignments.items() if c == small_cls and eid in eid_to_idx]
        if not indices:
            continue
        small_centroid = features[indices].mean(axis=0)
        norm = np.linalg.norm(small_centroid)
        if norm > 0:
            small_centroid = small_centroid / norm

        best_cls = None
        best_sim = -1.0
        for cls, centroid in centroids.items():
            sim = float(small_centroid @ centroid)
            if sim > best_sim:
                best_sim = sim
                best_cls = cls

        if best_cls:
            merge_map[small_cls] = best_cls
            for eid in list(updated.keys()):
                if updated[eid] == small_cls:
                    updated[eid] = best_cls

    if merge_map:
        print(f"  ✓ Merged {len(merge_map)} small classes:")
        for s, t in merge_map.items():
            print(f"    {s} ({class_counts[s]} members) → {t}")
    else:
        print("  ✓ No merges needed")

    return updated


def derive_hierarchy(
    assignments: dict[str, str],
    llm: ChatOllama,
) -> list[tuple[str, str]]:
    """LLM proposes SUBCLASS_OF relationships between classes."""
    print("\n[Phase 4b] Deriving class hierarchy...")

    class_counts = Counter(v for v in assignments.values() if v != "Other")
    class_names = sorted(class_counts.keys())

    if len(class_names) < 2:
        print("  ✓ Too few classes for hierarchy")
        return []

    classes_desc = "\n".join(f"  {c} ({class_counts[c]} members)" for c in class_names)

    prompt = f"""Given these ontology classes, propose SUBCLASS_OF relationships to form a hierarchy.

CLASSES:
{classes_desc}

Rules:
- Only propose relationships where one class is truly a subtype of another
- Use format: ChildClass -> ParentClass (meaning Child SUBCLASS_OF Parent)
- Prefer shallow hierarchies (max depth 3)
- Not every class needs a parent — top-level classes are fine
- Only output valid relationships, one per line
"""
    raw = _invoke_llm(llm, prompt)

    hierarchy: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    line_re = re.compile(r'^[- ]*(\w+)\s*->\s*(\w+)\s*$', re.MULTILINE)
    for m in line_re.finditer(raw):
        child = _sanitize_label(m.group(1))
        parent = _sanitize_label(m.group(2))
        if child != parent and child in class_names and parent in class_names:
            edge = (child, parent)
            if edge not in seen:
                hierarchy.append(edge)
                seen.add(edge)

    print(f"  ✓ {len(hierarchy)} SUBCLASS_OF edges")
    for child, parent in hierarchy:
        print(f"    {child} → {parent}")

    return hierarchy


# ---------------------------------------------------------------------------
# Phase 5: Write to Neo4j
# ---------------------------------------------------------------------------

def write_ontology(
    assignments: dict[str, str],
    hierarchy: list[tuple[str, str]],
) -> dict:
    """Write ontology layer to Neo4j."""
    print("\n[Phase 5] Writing ontology to Neo4j...")
    graph = get_neo4j_graph()

    # Clean previous ontology
    try:
        graph.query("MATCH (c:OntologyClass) DETACH DELETE c")
        # Remove old ontology labels from entities
        rows = graph.query("CALL db.labels() YIELD label RETURN label")
        existing = {r["label"] for r in rows}
        skip = {"__Entity__", "Document", "Entity", "OntologyClass"}
        for lbl in existing:
            if lbl not in skip and lbl != "Other":
                safe = re.sub(r'[^A-Za-z0-9_]', '', lbl)
                if safe and safe == lbl:
                    try:
                        graph.query(f"MATCH (n:`{safe}`) REMOVE n:`{safe}`")
                    except Exception:
                        pass
    except Exception as e:
        print(f"  Warning: cleanup error: {e}")

    # Get unique classes (excluding Other)
    class_counts = Counter(v for v in assignments.values() if v != "Other")
    class_names = sorted(class_counts.keys())

    # Create OntologyClass nodes
    for cls in class_names:
        safe = _sanitize_label(cls)
        try:
            graph.query(
                "MERGE (c:OntologyClass {name: $name})",
                params={"name": safe},
            )
        except Exception as e:
            print(f"  Warning: could not create class {safe}: {e}")

    # Label entities with their assigned class
    entities_labeled = 0
    for cls in class_names:
        safe = _sanitize_label(cls)
        members = [eid for eid, c in assignments.items() if c == cls]
        if not members:
            continue
        for batch_start in range(0, len(members), 100):
            batch = members[batch_start:batch_start + 100]
            try:
                graph.query(
                    f"MATCH (n:Entity) WHERE n.id IN $ids SET n:`{safe}`",
                    params={"ids": batch},
                )
                entities_labeled += len(batch)
            except Exception as e:
                print(f"  Warning: labeling batch for {safe}: {e}")

    # Create SUBCLASS_OF edges
    subclass_created = 0
    for child, parent in hierarchy:
        child_safe = _sanitize_label(child)
        parent_safe = _sanitize_label(parent)
        try:
            graph.query(
                "MATCH (c:OntologyClass {name: $child}), (p:OntologyClass {name: $parent}) "
                "MERGE (c)-[:SUBCLASS_OF]->(p)",
                params={"child": child_safe, "parent": parent_safe},
            )
            subclass_created += 1
        except Exception as e:
            print(f"  Warning: SUBCLASS_OF {child} → {parent}: {e}")

    stats = {
        "entities_labeled": entities_labeled,
        "ontology_classes": len(class_names),
        "subclass_of_edges": subclass_created,
        "class_names": class_names,
        "class_distribution": dict(class_counts),
        "method": "SV-LOI",
    }

    print(f"  ✓ Entities labeled:     {entities_labeled}")
    print(f"  ✓ OntologyClass nodes:  {len(class_names)}")
    print(f"  ✓ SUBCLASS_OF edges:    {subclass_created}")
    return stats


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def _flush_print(msg: str) -> None:
    """Print with immediate flush for Slurm log visibility."""
    print(msg, flush=True)


def run_sv_loi(
    model: str = config.OLLAMA_MODEL,
    suffix: str = "zone3_svloi",
) -> dict:
    """Run the full SV-LOI pipeline."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    _flush_print("=" * 70)
    _flush_print("CS584 Capstone — Zone 3: SV-LOI")
    _flush_print("Structurally-Verified LLM Ontology Induction")
    _flush_print(f"Model: {model} | Suffix: {suffix}")
    _flush_print("=" * 70)

    start = time.time()

    # Test Neo4j connection first
    _flush_print("\n[Pre-check] Testing Neo4j connection...")
    try:
        graph = get_neo4j_graph()
        test = graph.query("RETURN 1 AS ok")
        _flush_print(f"  ✓ Neo4j connected ({config.NEO4J_URI})")
    except Exception as e:
        _flush_print(f"  ✗ Neo4j connection FAILED: {e}")
        return {"error": f"Neo4j connection failed: {e}"}

    # Phase 0: Load entities
    entities = load_entities(graph)
    if not entities:
        return {"error": "no entities"}

    llm = get_llm(model)

    # Phase 1a: Discover class vocabulary
    class_vocab = discover_class_vocabulary(entities, llm)

    # Phase 1b: Batch LLM entity typing
    assignments = batch_type_entities(entities, class_vocab, llm)

    # Phase 1c: Rebalance mega-classes (split any class > 30% of entities)
    assignments, class_vocab = rebalance_mega_classes(
        assignments, entities, class_vocab, llm,
    )

    # Phase 2: Build structural signatures + consensus check
    features, entity_ids, feature_names = build_structural_signatures(entities)
    class_stats, flagged = structural_consensus_check(assignments, features, entity_ids)

    # Phase 3: Arbitrate disagreements
    assignments = arbitrate_disagreements(flagged, entities, assignments, class_vocab, llm)

    # Phase 4a: Merge small classes
    assignments = merge_small_classes(assignments, features, entity_ids)

    # Phase 4b: Derive hierarchy
    hierarchy = derive_hierarchy(assignments, llm)

    # Phase 5: Write to Neo4j
    neo4j_stats = write_ontology(assignments, hierarchy)

    elapsed = time.time() - start

    # Final distribution
    final_dist = Counter(v for v in assignments.values() if v != "Other")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SV-LOI pipeline complete in {elapsed:.1f}s")
    print(f"  Method:            SV-LOI (Structurally-Verified LLM Ontology Induction)")
    print(f"  Entities:          {len(entities)}")
    print(f"  Classes:           {len(final_dist)}")
    print(f"  SUBCLASS_OF:       {len(hierarchy)}")
    print(f"  Flagged/Arbitrated:{len(flagged)}")
    print(f"  Distribution:")
    for cls, cnt in final_dist.most_common():
        print(f"    {cls}: {cnt}")

    # Save summary
    summary = {
        "mode": "zone3_sv_loi",
        "model": model,
        "suffix": suffix,
        "elapsed_seconds": round(elapsed, 2),
        "entity_count": len(entities),
        "class_vocab_discovered": class_vocab,
        "classes_final": sorted(final_dist.keys()),
        "class_distribution": dict(final_dist),
        "flagged_count": len(flagged),
        "flagged_entities": flagged[:50],
        "hierarchy": [{"child": c, "parent": p} for c, p in hierarchy],
        "neo4j_stats": neo4j_stats,
    }
    out_path = os.path.join(config.RESULTS_DIR, f"zone3_svloi_summary_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✓ Summary saved to {out_path}")
    print(f"\nNext steps:")
    print(f"  python3 baseline/eval.py --suffix {suffix} --riskine --model {model}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zone 3: SV-LOI — Structurally-Verified LLM Ontology Induction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Novel ontology induction: LLM assigns classes to entities, structural
signatures verify assignments, disagreements are arbitrated by LLM
with enriched context. Produces 10-20 clean ontology classes.

Pre-requisite: Run zone2/pipeline.py first to populate Neo4j with Entity nodes.

Examples:
  python3 zone3/sv_loi.py
  python3 zone3/sv_loi.py --model qwen2.5:72b
  python3 zone3/sv_loi.py --model qwen2.5:72b --suffix zone3_svloi

After running, evaluate with:
  python3 baseline/eval.py --suffix zone3_svloi --riskine
        """,
    )
    parser.add_argument(
        "--model", default=config.OLLAMA_MODEL,
        help=f"Ollama model name (default: {config.OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--suffix", default="zone3_svloi",
        help="Suffix for result files (default: zone3_svloi)"
    )
    args = parser.parse_args()

    run_sv_loi(model=args.model, suffix=args.suffix)
