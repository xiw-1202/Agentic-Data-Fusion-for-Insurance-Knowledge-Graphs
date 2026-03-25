import json
import os
import sys
from dotenv import load_dotenv

load_dotenv() # Explicitly load .env file

# Ensure config module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from zone2.pipeline import insert_to_neo4j, zone25_entity_resolution

def resume_zone2_insert():
    summary_path = os.path.join(config.RESULTS_DIR, "zone2_run_summary.json")
    
    print(f"Loading triples from {summary_path}...")
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {summary_path}: {e}")
        return
    
    triples = data.get("triples", [])
    if not triples:
        print("Error: No triples were found in the summary file.")
        return

    print(f"Successfully loaded {len(triples)} processed triples.")
    
    # Mock the Zone2State dictionary to pass to the pipeline stages
    state = {
        "triples": triples,
        "vocab": data.get("vocab", []),
        "entity_types": data.get("entity_types", [])
    }
    
    print("\n--- Resuming Step 4: Neo4j Insertion ---")
    neo4j_result = insert_to_neo4j(state)
    if "error" in neo4j_result.get("neo4j_stats", {}):
        print("\nNeo4j Insertion Failed. Did you fix your .env credentials and ensure Neo4j/AuraDB is running?")
        print("Error details:", neo4j_result["neo4j_stats"]["error"])
        return
        
    print("\n--- Resuming Step 4.5: Entity Resolution ---")
    resolution_result = zone25_entity_resolution(state)

    print("\n--- Final Recovery Stats ---")
    print("Neo4j Stats:", json.dumps(neo4j_result.get("neo4j_stats", {}), indent=2))
    print("Resolution Stats:", json.dumps(resolution_result.get("resolution_stats", {}), indent=2))
    
    print("\nRecovery Complete! You can now proceed to run evaluation or Zone 3.")

if __name__ == "__main__":
    resume_zone2_insert()
