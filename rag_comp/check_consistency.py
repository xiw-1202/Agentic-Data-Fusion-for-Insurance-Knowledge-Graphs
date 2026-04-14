import os
import sys
import pandas as pd
from neo4j import GraphDatabase
from typing import List, Dict

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config

def check_consistency():
    print("="*60)
    print("INSURANCE DATA CONSISTENCY AUDIT (CSV <-> NEO4J)")
    print("="*60)

    # 1. Load CSV Data
    csv_dir = os.path.join(_ROOT, "data", "auto", "csv")
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    
    if not csv_files:
        print(f"[Error] No CSV files found in {csv_dir}")
        return

    # Tracking metrics
    report = {
        "files_checked": len(csv_files),
        "total_csv_rows": 0,
        "total_neo4j_nodes": 0,
        "consistency_matches": [],
        "consistency_gaps": []
    }

    # Connect to Neo4j
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    
    try:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            
            # Global node count
            res = session.run("MATCH (n:Entity) RETURN count(n) as count")
            report["total_neo4j_nodes"] = res.single()["count"]
            print(f"Total Neo4j Entity Nodes: {report['total_neo4j_nodes']}")

            for csv_file in csv_files:
                path = os.path.join(csv_dir, csv_file)
                df = pd.read_csv(path)
                report["total_csv_rows"] += len(df)
                
                print(f"\nAnalyzing {csv_file} ({len(df)} rows)...")
                
                # Identify Key Columns
                id_cols = [c for c in df.columns if any(x in c.upper() for x in ["POLNO", "CLAIM", "ID", "INSURED"])]
                if not id_cols:
                    print(f"  [Skip] No clear ID columns found in {csv_file}")
                    continue
                
                # Sample Validation (Top 10)
                sample = df.head(10)
                found_count = 0
                
                for _, row in sample.iterrows():
                    # Try to find any ID from the row in Neo4j
                    found_in_row = False
                    for col in id_cols:
                        val = str(row[col])
                        if pd.isna(row[col]) or val == "nan": continue
                        
                        # Check as node ID or property
                        cypher = """
                            MATCH (n:Entity) 
                            WHERE n.id = $val OR n.id CONTAINS $val
                            RETURN n.id as id LIMIT 1
                        """
                        res = session.run(cypher, val=val)
                        if res.peek():
                            found_count += 1
                            found_in_row = True
                            break
                    
                match_rate = found_count / len(sample)
                print(f"  Sample Match Rate: {match_rate:.0%}")
                
                if match_rate > 0.8:
                    report["consistency_matches"].append(csv_file)
                else:
                    report["consistency_gaps"].append(csv_file)

            # Relationship check (Policy/Claim link)
            print("\n--- Relationship Audit ---")
            res = session.run("MATCH (n:Entity)-[r:HAS_POLICY_NUMBER|HAS_CLAIM_NUMBER]->(m) RETURN count(r) as count")
            rel_count = res.single()["count"]
            print(f"Record-to-ID relationships found: {rel_count}")
            
    except Exception as e:
        print(f"[Critical Error] {e}")
    finally:
        driver.close()

    # Final Summary
    print("\n" + "="*60)
    print("CONSISTENCY SUMMARY")
    print("="*60)
    print(f"Files with high consistency: {len(report['consistency_matches'])}")
    print(f"Files with GAPS: {len(report['consistency_gaps'])}")
    
    if report['consistency_gaps']:
        print("\nWARNING: The following files may not be fully ingested:")
        for f in report['consistency_gaps']:
            print(f"  - {f}")
        print("\nAction: Ensure rag_comp/data_prep.py has been run successfully.")
    else:
        print("\nSuccess: Graph structure appears to align with local CSV data.")

if __name__ == "__main__":
    check_consistency()
