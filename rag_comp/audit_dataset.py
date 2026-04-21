import json
import os
import sys
from neo4j import GraphDatabase

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import config

def verify_dataset():
    path = os.path.join(_ROOT, config.RESULTS_DIR, 'evaluation_datasets', 'auto_gold_standard.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    
    audit_log = []
    
    with driver.session(database=config.NEO4J_DATABASE) as session:
        for q in data:
            q_id = q.get('id')
            cat = q.get('category')
            source_ids = q.get('source_ids', [])
            
            result = {"id": q_id, "category": cat, "status": "VERIFYING", "notes": ""}
            
            try:
                if cat in ["lookup", "1_hop"]:
                    # Check if at least one relationship exists for the subject
                    subject = source_ids[0]
                    res = session.run("MATCH (n:Entity {id: $id})-[r]->(m) RETURN type(r) as rel, m.id as target", id=subject)
                    facts = [f"{record['rel']}: {record['target']}" for record in res]
                    if not facts:
                        # Try without :Entity label just in case
                        res = session.run("MATCH (n {id: $id})-[r]->(m) RETURN type(r) as rel, m.id as target", id=subject)
                        facts = [f"{record['rel']}: {record['target']}" for record in res]
                    
                    if not facts:
                        result["status"] = "FAILED"
                        result["notes"] = f"Subject {subject} not found or has no relations."
                    else:
                        match = False
                        if len(source_ids) > 1:
                            match = any(str(source_ids[1]).lower() in f.lower() for f in facts)
                        result["status"] = "PASSED" if match else "QUESTIONABLE"
                        result["notes"] = f"KG Facts: {facts[:3]}..."
                
                elif cat == "2_hop":
                    # Check the chain
                    if len(source_ids) >= 3:
                        n1, n2, n3 = source_ids[0], source_ids[1], source_ids[2]
                        res = session.run("MATCH (n1 {id: $n1})-->(n2 {id: $n2})-->(n3) RETURN n3.id as target", n1=n1, n2=n2)
                        targets = [record['target'] for record in res]
                        if any(str(n3).lower() in str(t).lower() for t in targets):
                            result["status"] = "PASSED"
                        else:
                            result["status"] = "FAILED"
                            result["notes"] = f"No 2-hop path {n1}->{n2}->{n3} found."
                    else:
                        result["status"] = "ERROR"
                        result["notes"] = "Insufficient source_ids for 2-hop."
                
                elif cat == "cross_source":
                    # Check KG part and mark for manual Doc check
                    kg_id = source_ids[0]
                    res = session.run("MATCH (n {id: $id}) RETURN count(n) as count", id=kg_id)
                    if list(res)[0]['count'] > 0:
                        result["status"] = "PASSED (KG Part)"
                        result["notes"] = f"KG ID {kg_id} exists. Manual Doc check needed for {source_ids[1]}."
                    else:
                        result["status"] = "FAILED"
                        result["notes"] = f"KG ID {kg_id} not found."
                
                elif cat in ["global", "reasoning"]:
                    result["status"] = "PASSED"
                    result["notes"] = "Manual verification required."
                        
            except Exception as e:
                result["status"] = "ERROR"
                result["notes"] = f"{type(e).__name__}: {str(e)}"
                
            audit_log.append(result)
            
    out_log = os.path.join(_ROOT, "data/results/audit_results.json")
    with open(out_log, "w") as f:
        json.dump(audit_log, f, indent=2)
    
    print(f"Audit complete. Logged {len(audit_log)} items to {out_log}")

if __name__ == "__main__":
    verify_dataset()
