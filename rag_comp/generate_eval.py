import os
import sys
import json
import random
import argparse
from typing import List, Dict

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

def load_chunks():
    path = os.path.join(_ROOT, config.CHUNKS_FILE)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def fetch_motifs(session):
    motifs = {
        "1_hop": [],
        "2_hop": [],
        "cross_source": []
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
        "1_hop": random.sample(motifs["1_hop"], min(15, len(motifs["1_hop"]))),
        "2_hop": random.sample(motifs["2_hop"], min(15, len(motifs["2_hop"]))),
        "cross_source": motifs["cross_source"]
    }

def generate_questions(motifs, llm, existing_questions=None):
    if existing_questions:
        questions = existing_questions
    else:
        questions = SEED_QUESTIONS.copy()
        
    task_id = len(questions) + 1
    
    for category, subgraph_list in motifs.items():
        if not subgraph_list: continue
        
        for item in subgraph_list:
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
                print(f"Generating for {category} [{task_id}]...")
                response = llm.invoke([HumanMessage(content=prompt)])
                content = response.content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                
                data = json.loads(content)
                data['id'] = f"gen_{task_id}"
                data['source_ids'] = source_ids
                questions.append(data)
                task_id += 1
            except Exception as e:
                print(f"  [Error] {e}")
                
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
    motifs = {"1_hop": [], "2_hop": [], "cross_source": []}
    
    try:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            all_motifs = fetch_motifs(session)
            if args.mode == "cross_source_only":
                motifs = {"cross_source": all_motifs["cross_source"]}
            else:
                motifs = all_motifs
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
