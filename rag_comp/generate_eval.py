import os
import sys
import json
import random
import argparse
from typing import Any, Dict, List

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from neo4j import GraphDatabase
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import config

# Manually curated questions for insurance data (fallback/seed)
SEED_QUESTIONS = [
    {
        "id": "seed_1",
        "category": "lookup",
        "question": "What is the policy number for the claim record 'CLM-94ca7cc26d29'?",
        "expected_answer": "The policy number for CLM-94ca7cc26d29 is REN285325500.",
        "keywords": ["CLM-94ca7cc26d29", "REN285325500"],
        "source_ids": ["CLM-94ca7cc26d29", "REN285325500"]
    },
    {
        "id": "seed_2",
        "category": "reasoning",
        "question": "Which person is associated with claim 'CLM-94ca7cc26d29' and what is their organization?",
        "expected_answer": "Claim CLM-94ca7cc26d29 belongs to person PER-bb3ac3cf357a, who is associated with 'Assurant Global Home'.",
        "keywords": ["PER-bb3ac3cf357a", "Assurant Global Home"],
        "source_ids": ["CLM-94ca7cc26d29", "PER-bb3ac3cf357a", "Assurant Global Home"]
    },
    {
        "id": "seed_3",
        "category": "global",
        "question": "What are the most common relationship types between claims and other entities in the graph?",
        "expected_answer": "Common relationships include BELONGS_TO, HAS_POLICY_NUMBER, HAS_CLAIM_NUMBER, and IS_A (to ClaimRecord).",
        "keywords": ["BELONGS_TO", "HAS_POLICY_NUMBER", "ClaimRecord"],
        "source_ids": ["ClaimRecord"]
    }
]


def _limit_sample(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if len(items) <= limit:
        return items
    return random.sample(items, limit)


def _entity_exists(session, entity_id: str) -> bool:
    record = session.run(
        """
        MATCH (n:Entity {id: $entity_id})
        RETURN n.id AS id
        LIMIT 1
        """,
        entity_id=entity_id,
    ).single()
    return record is not None


def _next_missing_claim_id(session, seed: int) -> str:
    candidate_num = seed
    while True:
        candidate = f"CLM-NOT-FOUND-{candidate_num:03d}"
        if not _entity_exists(session, candidate):
            return candidate
        candidate_num += 1


def _build_nonexistent_claim_task(claim_id: str, requested_fact: str) -> Dict[str, Any]:
    return {
        "category": "unanswerable",
        "question": f"What {requested_fact} is associated with claim {claim_id}?",
        "expected_answer": (
            f"Abstain: claim {claim_id} does not exist in the available knowledge graph, "
            f"so its {requested_fact} cannot be determined from this data."
        ),
        "keywords": [claim_id, "does not exist", "cannot be determined"],
        "source_ids": [claim_id],
        "support": {"abstention_reason": "claim_missing"},
    }


def _build_missing_fact_task(
    claim_id: str,
    requested_fact: str,
    missing_path_label: str,
) -> Dict[str, Any]:
    return {
        "category": "unanswerable",
        "question": f"What {requested_fact} is associated with claim {claim_id}?",
        "expected_answer": (
            f"Abstain: claim {claim_id} exists in the available knowledge graph, but no "
            f"{missing_path_label} path or fact is represented for it, so the requested "
            f"{requested_fact} cannot be determined from this data."
        ),
        "keywords": [claim_id, "exists", "cannot be determined"],
        "source_ids": [claim_id],
        "support": {"abstention_reason": "missing_fact_or_path", "missing_path": missing_path_label},
    }


def _build_organization_count_task(org_id: str, claim_count: int) -> Dict[str, Any]:
    return {
        "category": "aggregation",
        "question": "Which organization is connected to the most claims, and how many claims is it connected to?",
        "expected_answer": f"{org_id} is connected to the most claims, with {claim_count} claims.",
        "keywords": [org_id, str(claim_count), "most claims"],
        "source_ids": [org_id],
        "support": {"query_type": "top_count"},
    }


def _build_distinct_account_count_task(account_count: int, sample_accounts: List[str]) -> Dict[str, Any]:
    source_ids = sample_accounts[:3] if sample_accounts else [str(account_count)]
    return {
        "category": "aggregation",
        "question": "How many distinct ACCT_NAME values are associated with claim-linked persons?",
        "expected_answer": (
            f"There are {account_count} distinct ACCT_NAME values associated with claim-linked persons."
        ),
        "keywords": [str(account_count), "ACCT_NAME", "distinct"],
        "source_ids": source_ids,
        "support": {"query_type": "count_distinct"},
    }


def _build_claim_coverage_count_task(claim_count: int) -> Dict[str, Any]:
    return {
        "category": "aggregation",
        "question": "How many claims have both a linked person and a policy number in the knowledge graph?",
        "expected_answer": (
            f"There are {claim_count} claims that have both a linked person and a policy number."
        ),
        "keywords": [str(claim_count), "linked person", "policy number"],
        "source_ids": [str(claim_count)],
        "support": {"query_type": "count_with_required_relationships"},
    }


def _build_relationship_ranking_task(relationship_summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    relationship_names = [item["relationship"] for item in relationship_summary]
    summary_text = ", ".join(
        f"{item['relationship']} ({item['frequency']})" for item in relationship_summary
    )
    return {
        "category": "aggregation",
        "question": "What are the top 3 most common relationship types connected to claim nodes in the graph?",
        "expected_answer": f"The top 3 relationship types connected to claim nodes are {summary_text}.",
        "keywords": relationship_names[:3],
        "source_ids": relationship_names[:3],
        "support": {"query_type": "top_relationships"},
    }


def _build_constrained_person_org_account_task(
    claim_id: str,
    person_id: str,
    org_id: str,
    acct_id: str,
) -> Dict[str, Any]:
    return {
        "category": "constrained_multi_hop",
        "question": (
            f"For claim {claim_id}, which person is linked to the claim, and what organization "
            "and ACCT_NAME are associated with that same person?"
        ),
        "expected_answer": (
            f"Claim {claim_id} belongs to {person_id}, whose organization is {org_id} and "
            f"whose ACCT_NAME is {acct_id}."
        ),
        "keywords": [claim_id, person_id, org_id, acct_id],
        "source_ids": [claim_id, person_id, org_id, acct_id],
        "support": {"query_type": "claim_person_org_account"},
    }


def _build_constrained_policy_lob_task(
    claim_id: str,
    person_id: str,
    policy_id: str,
    lob_id: str,
) -> Dict[str, Any]:
    return {
        "category": "constrained_multi_hop",
        "question": (
            f"For claim {claim_id}, which person is linked to it, and what policy number and "
            "line of business are recorded for that same claim?"
        ),
        "expected_answer": (
            f"Claim {claim_id} is linked to {person_id}, with policy number {policy_id} and "
            f"line of business {lob_id}."
        ),
        "keywords": [claim_id, person_id, policy_id, lob_id],
        "source_ids": [claim_id, person_id, policy_id, lob_id],
        "support": {"query_type": "claim_person_policy_lob"},
    }

def load_chunks():
    path = os.path.join(_ROOT, config.CHUNKS_FILE)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def fetch_aggregation_tasks(session) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []

    record = session.run(
        """
        MATCH (c:Entity)-[:BELONGS_TO]->(p:Entity)-[:HAS_ACCT_NAME]->(acct:Entity)
        WHERE c.id STARTS WITH 'CLM' AND p.id STARTS WITH 'PER'
        RETURN count(DISTINCT acct.id) AS acct_count, collect(DISTINCT acct.id)[0..5] AS sample_accounts
        """
    ).single()
    if record and record["acct_count"] is not None:
        tasks.append(
            _build_distinct_account_count_task(
                int(record["acct_count"]),
                [str(value) for value in record["sample_accounts"] or []],
            )
        )

    record = session.run(
        """
        MATCH (c:Entity)-[:BELONGS_TO]->(:Entity)-[:HAS_ORGANIZATION_NAME]->(org:Entity)
        WHERE c.id STARTS WITH 'CLM'
        RETURN org.id AS org_id, count(DISTINCT c) AS claim_count
        ORDER BY claim_count DESC, org_id ASC
        LIMIT 1
        """
    ).single()
    if record and record["org_id"] is not None:
        tasks.append(
            _build_organization_count_task(
                str(record["org_id"]),
                int(record["claim_count"]),
            )
        )

    record = session.run(
        """
        MATCH (c:Entity)
        WHERE c.id STARTS WITH 'CLM'
          AND EXISTS { MATCH (c)-[:BELONGS_TO]->(:Entity) }
          AND EXISTS { MATCH (c)-[:HAS_POLICY_NUMBER]->(:Entity) }
        RETURN count(DISTINCT c) AS claim_count
        """
    ).single()
    if record and record["claim_count"] is not None:
        tasks.append(_build_claim_coverage_count_task(int(record["claim_count"])))

    relationship_rows = list(
        session.run(
            """
            MATCH (c:Entity)-[r]-()
            WHERE c.id STARTS WITH 'CLM'
            RETURN type(r) AS relationship, count(*) AS frequency
            ORDER BY frequency DESC, relationship ASC
            LIMIT 3
            """
        )
    )
    if relationship_rows:
        tasks.append(
            _build_relationship_ranking_task(
                [
                    {
                        "relationship": str(row["relationship"]),
                        "frequency": int(row["frequency"]),
                    }
                    for row in relationship_rows
                ]
            )
        )

    return tasks


def fetch_unanswerable_tasks(session) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []

    missing_fact_claims = list(
        session.run(
            """
            MATCH (c:Entity)
            WHERE c.id STARTS WITH 'CLM'
              AND EXISTS { MATCH (c)-[:BELONGS_TO]->(:Entity) }
              AND NOT EXISTS { MATCH (c)-[:BELONGS_TO]->(:Entity)-[:HAS_SURVEY_NAME]->(:Entity) }
            RETURN c.id AS claim_id
            ORDER BY claim_id ASC
            LIMIT 2
            """
        )
    )
    for row in missing_fact_claims:
        tasks.append(
            _build_missing_fact_task(
                str(row["claim_id"]),
                requested_fact="survey name",
                missing_path_label="BELONGS_TO -> HAS_SURVEY_NAME",
            )
        )

    missing_path_claims = list(
        session.run(
            """
            MATCH (c:Entity)
            WHERE c.id STARTS WITH 'CLM'
              AND EXISTS { MATCH (c)-[:BELONGS_TO]->(:Entity) }
              AND NOT EXISTS { MATCH (c)-[:BELONGS_TO]->(:Entity)-[:HAS_SURVEY_GATEWAY_NAME]->(:Entity) }
            RETURN c.id AS claim_id
            ORDER BY claim_id DESC
            LIMIT 2
            """
        )
    )
    for row in missing_path_claims:
        tasks.append(
            _build_missing_fact_task(
                str(row["claim_id"]),
                requested_fact="survey gateway name",
                missing_path_label="BELONGS_TO -> HAS_SURVEY_GATEWAY_NAME",
            )
        )

    for seed, fact_name in enumerate(["policy number", "NPS score", "organization"], start=1):
        missing_claim_id = _next_missing_claim_id(session, seed)
        tasks.append(_build_nonexistent_claim_task(missing_claim_id, fact_name))

    return tasks


def fetch_constrained_multi_hop_tasks(session) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []

    rows = list(
        session.run(
            """
            MATCH (c:Entity)-[:BELONGS_TO]->(p:Entity)
            MATCH (p)-[:HAS_ORGANIZATION_NAME]->(org:Entity)
            MATCH (p)-[:HAS_ACCT_NAME]->(acct:Entity)
            WHERE c.id STARTS WITH 'CLM' AND p.id STARTS WITH 'PER'
            RETURN DISTINCT c.id AS claim_id, p.id AS person_id, org.id AS org_id, acct.id AS acct_id
            ORDER BY claim_id ASC
            LIMIT 4
            """
        )
    )
    for row in rows:
        tasks.append(
            _build_constrained_person_org_account_task(
                str(row["claim_id"]),
                str(row["person_id"]),
                str(row["org_id"]),
                str(row["acct_id"]),
            )
        )

    rows = list(
        session.run(
            """
            MATCH (c:Entity)-[:BELONGS_TO]->(p:Entity)
            MATCH (c)-[:HAS_POLICY_NUMBER]->(policy:Entity)
            MATCH (c)-[:HAS_LOB]->(lob:Entity)
            WHERE c.id STARTS WITH 'CLM' AND p.id STARTS WITH 'PER'
            RETURN DISTINCT c.id AS claim_id, p.id AS person_id, policy.id AS policy_id, lob.id AS lob_id
            ORDER BY claim_id DESC
            LIMIT 4
            """
        )
    )
    for row in rows:
        tasks.append(
            _build_constrained_policy_lob_task(
                str(row["claim_id"]),
                str(row["person_id"]),
                str(row["policy_id"]),
                str(row["lob_id"]),
            )
        )

    return tasks


def fetch_motifs(session):
    motifs = {
        "1_hop": [],
        "2_hop": [],
        "cross_source": [],
        "aggregation": fetch_aggregation_tasks(session),
        "unanswerable": fetch_unanswerable_tasks(session),
        "constrained_multi_hop": fetch_constrained_multi_hop_tasks(session),
    }
    
    # 1-hop: Claim/Policy to Attribute
    res1 = session.run('''
        MATCH (n:Entity)-[r]->(m:Entity) 
        WHERE (n.id STARTS WITH 'CLM' OR n.id STARTS WITH 'POL' OR n.id STARTS WITH 'PER')
          AND NOT m.id STARTS WITH '202'  // Filter out dates for cleaner questions
        RETURN n.id AS s, type(r) AS rel, m.id AS o 
        LIMIT 100
    ''')
    for record in res1:
        motifs["1_hop"].append({
            "subgraph": f"({record['s']}) -[{record['rel']}]-> ({record['o']})",
            "source_ids": [record['s'], record['o']]
        })
        
    # 2-hop: Claim -> Person -> Organization/Survey
    res2 = session.run('''
        MATCH (n1:Entity)-[r1]->(n2:Entity)-[r2]->(n3:Entity) 
        WHERE n1.id STARTS WITH 'CLM' AND n2.id STARTS WITH 'PER'
        RETURN n1.id AS n1, type(r1) AS r1, n2.id AS n2, type(r2) AS r2, n3.id AS n3 
        ''')
    for record in res2:
        motifs["2_hop"].append({
            "subgraph": f"({record['n1']}) -[{record['r1']}]-> ({record['n2']}) -[{record['r2']}]-> ({record['n3']})",
            "source_ids": [record['n1'], record['n2'], record['n3']]
        })

    # Cross-source: Real Entity (KG) + Relevant PDF Chunk (Semantic Anchor)
    # We find claims and then find PDF chunks that actually mention the same organization/account.
    chunks = load_chunks()
    
    # Get entities with rich anchor properties
    res3 = session.run('''
        MATCH (n:Entity)
        WHERE n.id STARTS WITH 'CLM'
        OPTIONAL MATCH (n)-[:HAS_ORGANIZATION_NAME]->(org)
        OPTIONAL MATCH (n)-[:HAS_ACCT_NAME]->(acc)
        OPTIONAL MATCH (n)-[:HAS_LOB]->(lob)
        WITH n, 
             collect(DISTINCT org.id) + collect(DISTINCT acc.id) + collect(DISTINCT lob.id) as anchors,
             count(*) as degree
        WHERE size(anchors) > 0
        RETURN n.id as id, anchors
        LIMIT 50
    ''')
    
    kg_entities = list(res3)
    
    unique_pairs = set()
    for entity in kg_entities:
        anchors = [str(a) for a in entity['anchors'] if a and len(str(a)) > 2] # Skip short codes
        if not anchors: continue
        
        # Find a chunk that mentions any of these anchors
        potential_chunks = []
        for chunk in chunks:
            content = chunk.get('content', '').lower()
            if any(anchor.lower() in content for anchor in anchors):
                potential_chunks.append(chunk)
        
        if potential_chunks:
            # Pick a relevant chunk
            chunk = random.choice(potential_chunks)
            pair_key = (entity['id'], chunk['chunk_id'])
            if pair_key not in unique_pairs:
                unique_pairs.add(pair_key)
                motif_str = f"KG_ENTITY: {entity['id']} (Linked via Anchors: {', '.join(anchors)})\nDOCUMENT_CHUNK: {chunk['content'][:1000]}"
                motifs["cross_source"].append({
                    "subgraph": motif_str,
                    "source_ids": [entity['id'], chunk['chunk_id']]
                })
        
        if len(motifs["cross_source"]) >= 10:
            break

    return {
        "1_hop": _limit_sample(motifs["1_hop"], 15),
        "2_hop": _limit_sample(motifs["2_hop"], 15),
        "cross_source": motifs["cross_source"],
        "aggregation": _limit_sample(motifs["aggregation"], 8),
        "unanswerable": _limit_sample(motifs["unanswerable"], 8),
        "constrained_multi_hop": _limit_sample(motifs["constrained_multi_hop"], 8),
    }


def _log_category_summary(motifs: Dict[str, List[Dict[str, Any]]]) -> None:
    print("Question candidates by category:")
    for category, items in motifs.items():
        print(f"  - {category}: {len(items)}")


def generate_questions(motifs, llm, existing_questions=None):
    if existing_questions:
        questions = existing_questions
    else:
        questions = SEED_QUESTIONS.copy()
        
    task_id = len(questions) + 1

    print(f"Starting question generation from task ID {task_id}.")
    
    for category, subgraph_list in motifs.items():
        if not subgraph_list:
            print(f"Skipping {category}: no candidate items found.")
            continue

        print(f"\nProcessing category '{category}' with {len(subgraph_list)} item(s).")
        
        for index, item in enumerate(subgraph_list, start=1):
            if "question" in item and "expected_answer" in item:
                data = dict(item)
                data["id"] = f"gen_{task_id}"
                questions.append(data)
                print(
                    f"  [{category} {index}/{len(subgraph_list)}] "
                    f"Added deterministic question {data['id']}: {data['question'][:90]}"
                )
                task_id += 1
                continue

            subgraph = item["subgraph"]
            source_ids = item["source_ids"]
            
            if category == "cross_source":
                prompt = f"""You are an insurance knowledge auditor. You are given data from two different sources. 
Generate ONE evaluation question that requires synthesizing information from BOTH.

SOURCES PROVIDED:
{subgraph}

Task Category: {category}

Instructions:
1. The question should ask for something present in the Document Chunk (like a policy term or procedure) specifically as it applies to the KG_ENTITY (using the properties provided in the KG).
2. Example: "Given the dispute clause in the document, what is the required response time for claim CLM-XXXX?"
3. Provide a grounded expected_answer that combines facts from both sources.
4. List 3 crucial keywords (entity IDs or specific terms).
5. Output EXACTLY in JSON format:
{{
  "category": "{category}",
  "question": "...",
  "expected_answer": "...",
  "keywords": ["...", "...", "..."]
}}
"""
            else:
                prompt = f"""You are an insurance knowledge auditor. Given the following knowledge graph subgraph (triplets), generate ONE evaluation question.
            
Subgraph:
{subgraph}

Task Category: {category}

Instructions:
1. Generate a natural language question that requires information from the subgraph.
2. Provide a short, grounded expected_answer.
3. List 2-3 crucial keywords that MUST be present.
4. Output EXACTLY in JSON format:
{{
  "category": "{category}",
  "question": "...",
  "expected_answer": "...",
  "keywords": ["...", "..."]
}}
"""
            try:
                print(
                    f"  [{category} {index}/{len(subgraph_list)}] "
                    f"Generating LLM-backed question {task_id}..."
                )
                response = llm.invoke([HumanMessage(content=prompt)])
                content = response.content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                
                data = json.loads(content)
                data['id'] = f"gen_{task_id}"
                data['source_ids'] = source_ids
                questions.append(data)
                print(
                    f"     Saved {data['id']}: {data.get('question', '')[:90]}"
                )
                task_id += 1
            except Exception as e:
                print(
                    f"     Error while generating category '{category}' item "
                    f"{index}/{len(subgraph_list)}: {e}"
                )
                
    return questions

def run():
    parser = argparse.ArgumentParser(description="Generate Insurance RAG Evaluation Dataset")
    parser.add_argument("--mode", choices=["full", "cross_source_only"], default="full", 
                        help="Generate all or only replace cross-source questions")
    args = parser.parse_args()

    print("="*60)
    print(f"Generating Insurance RAG Eval Dataset (Mode: {args.mode})")
    print("="*60)
    
    out_dir = os.path.join(_ROOT, config.RESULTS_DIR, 'evaluation_datasets')
    out_path = os.path.join(out_dir, 'auto_gold_standard.json')
    
    existing_questions = []
    if args.mode == "cross_source_only":
        if os.path.exists(out_path):
            with open(out_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                # Keep all EXCEPT cross_source
                existing_questions = [q for q in existing if q.get('category') != 'cross_source']
                print(f"Loaded {len(existing_questions)} existing questions (seed/1-hop/2-hop).")
        else:
            print(f"Warning: {out_path} not found. Running full generation.")
            args.mode = "full"

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    motifs = {
        "1_hop": [],
        "2_hop": [],
        "cross_source": [],
        "aggregation": [],
        "unanswerable": [],
        "constrained_multi_hop": [],
    }
    
    try:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            all_motifs = fetch_motifs(session)
            if args.mode == "cross_source_only":
                motifs = {"cross_source": all_motifs["cross_source"]}
            else:
                motifs = all_motifs
            _log_category_summary(motifs)
    except Exception as e:
        print(f"Could not connect to Neo4j to mine motifs: {e}")
        if args.mode == "full":
            print("Continuing with seed questions only.")

    llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0.1, format="json")
    dataset = generate_questions(motifs, llm, existing_questions if args.mode == "cross_source_only" else None)
    
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"\n✓ Generated/Updated {len(dataset)} questions saved to {out_path}")

if __name__ == "__main__":
    run()
