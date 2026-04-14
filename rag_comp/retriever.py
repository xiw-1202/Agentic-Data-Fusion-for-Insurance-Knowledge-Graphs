import os
import sys
import re
from typing import List, Dict
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings
from neo4j_graphrag.retrievers import HybridRetriever

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config

class KGRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, 
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        self.embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )
        
        # Initialize HybridRetriever for node search
        self.retriever = HybridRetriever(
            driver=self.driver,
            vector_index_name=config.VECTOR_INDEX_NAME,
            fulltext_index_name=config.FULLTEXT_INDEX_NAME,
            embedder=self.embeddings,
            return_properties=["id", "ontology_class"],
        )

    def _sanitize_query(self, query: str) -> str:
        """Escape special characters that cause Lucene lexical errors."""
        for char in ['/', '(', ')', '[', ']', '{', '}', ':', '-', '+', '!', '^', '*', '?']:
            query = query.replace(char, f"\\{char}")
        return query

    def search_entities(self, query: str, top_k: int = 5) -> List[dict]:
        """Perform hybrid search for entities and return their IDs and classes."""
        sanitized_query = self._sanitize_query(query)
        results = self.retriever.search(query_text=sanitized_query, top_k=top_k)
        
        output = []
        for item in results.items:
            metadata = getattr(item, 'metadata', {})
            node_id = metadata.get("id")
            
            if not node_id:
                # Fallback extraction from content string if metadata is missing
                content_str = getattr(item, 'content', '')
                match = re.search(r"'id':\s*'([^']+)'|\"id\":\s*\"([^\"]+)\"", content_str)
                if match:
                    node_id = match.group(1) or match.group(2)
            
            if node_id:
                output.append({
                    "id": str(node_id), 
                    "class": metadata.get("ontology_class", "Entity"),
                    "score": getattr(item, 'score', 0.0)
                })
        return output

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    retriever = KGRetriever()
    hits = retriever.search_entities("What is the ICC limit?")
    for hit in hits:
        print(f"[{hit['score']:.4f}] {hit['id']} ({hit['class']})")
    retriever.close()
