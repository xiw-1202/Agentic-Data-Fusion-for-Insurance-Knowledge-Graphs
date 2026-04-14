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
        "keywords": ["CLM-94ca7cc26d29", "REN285325500"]
    },
    {
        "id": "seed_2",
        "category": "reasoning",
        "question": "Which person is associated with claim 'CLM-94ca7cc26d29' and what is their organization?",
        "expected_answer": "Claim CLM-94ca7cc26d29 belongs to person PER-bb3ac3cf357a, who is associated with 'Assurant Global Home'.",
        "keywords": ["PER-bb3ac3cf357a", "Assurant Global Home"]
    },
    {
        "id": "seed_3",
        "category": "global",
        "question": "What are the most common relationship types between claims and other entities in the graph?",
        "expected_answer": "Common relationships include BELONGS_TO, HAS_POLICY_NUMBER, HAS_CLAIM_NUMBER, and IS_A (to ClaimRecord).",
        "keywords": ["BELONGS_TO", "HAS_POLICY_NUMBER", "ClaimRecord"]
    }
]

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
        motifs["1_hop"].append(f"({record['s']}) -[{record['rel']}]-> ({record['o']})")
        
    # 2-hop: Claim -> Person -> Organization/Survey
    res2 = session.run('''
        MATCH (n1:Entity)-[r1]->(n2:Entity)-[r2]->(n3:Entity) 
        WHERE n1.id STARTS WITH 'CLM' AND n2.id STARTS WITH 'PER'
        RETURN n1.id AS n1, type(r1) AS r1, n2.id AS n2, type(r2) AS r2, n3.id AS n3 
        LIMIT 100
    ''')
    for record in res2:
        motifs["2_hop"].append(f"({record['n1']}) -[{record['r1']}]-> ({record['n2']}) -[{record['r2']}]-> ({record['n3']})")

    # Cross-source: Shared attributes (like Organization ID)
    res3 = session.run('''
        MATCH (rec1:Entity)-[r1]->(m:Entity)<-[r2]-(rec2:Entity)
        WHERE rec1.id STARTS WITH 'CLM' AND rec2.id STARTS WITH 'REC'
          AND rec1 <> rec2
        RETURN rec1.id AS r1_id, m.id AS shared, rec2.id AS r2_id
        LIMIT 50
    ''')
    for record in res3:
        motifs["cross_source"].append(f"({record['r1_id']}) shared {record['shared']} with ({record['r2_id']})")

    return {
        "1_hop": random.sample(motifs["1_hop"], min(15, len(motifs["1_hop"]))),
        "2_hop": random.sample(motifs["2_hop"], min(15, len(motifs["2_hop"]))),
        "cross_source": random.sample(motifs["cross_source"], min(10, len(motifs["cross_source"])))
    }

def generate_questions(motifs, llm):
    questions = SEED_QUESTIONS.copy()
    task_id = len(questions) + 1
    
    for category, subgraph_list in motifs.items():
        for subgraph in subgraph_list:
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
                questions.append(data)
                task_id += 1
            except Exception as e:
                print(f"  [Error] {e}")
                
    return questions

def run():
    print("="*60)
    print("Generating Insurance RAG Evaluation Dataset")
    print("="*60)
    
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    motifs = {"1_hop": [], "2_hop": []}
    
    try:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            motifs = fetch_motifs(session)
    except Exception as e:
        print(f"Could not connect to Neo4j to mine motifs: {e}")
        print("Continuing with seed questions only.")

    llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0.1, format="json")
    dataset = generate_questions(motifs, llm)
    
    out_dir = os.path.join(_ROOT, config.RESULTS_DIR, 'evaluation_datasets')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'auto_gold_standard.json')
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"\n✓ Generated {len(dataset)} questions saved to {out_path}")

if __name__ == "__main__":
    run()
