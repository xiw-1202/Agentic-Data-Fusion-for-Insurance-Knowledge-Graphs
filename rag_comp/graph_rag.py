import os
import sys
import json
from typing import List, Dict, Any

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from rag_comp.retriever import KGRetriever

class GraphRAG:
    def __init__(self, model_name: str = None):
        self.retriever = KGRetriever()
        self.model_name = model_name or config.OLLAMA_MODEL
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json"
        )

    def get_enriched_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve nodes and their immediate relationships for context."""
        hits = self.retriever.search_entities(query, top_k=top_k)
        if not hits:
            return "No relevant entities found in Knowledge Graph."
        
        node_ids = [str(hit['id']) for hit in hits]
        
        # Enrichment Cypher query
        cypher = """
        // 1. Direct 1-hop bridges
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE n.id IN $node_ids AND m.id IN $node_ids AND n.id < m.id
        WITH collect({source: n.id, relationship: type(r), target: m.id}) as b1
        
        // 2. 2-hop bridges
        MATCH (n:Entity)-[r1]-(mid)-[r2]-(m:Entity)
        WHERE n.id IN $node_ids AND m.id IN $node_ids AND n.id < m.id
        WITH b1, collect({source: n.id, relationship: type(r1), target: mid.id}) + 
                 collect({source: mid.id, relationship: type(r2), target: m.id}) as b2
        
        // 3. Broad expansion
        MATCH (n:Entity)-[r]-(m)
        WHERE n.id IN $node_ids
        WITH b1 + b2 as bridges, n, r, m
        ORDER BY CASE 
            WHEN type(r) IN ['COVERS', 'DEFINED_AS', 'HAS_LIMIT'] THEN 1 
            ELSE 2 
        END ASC
        
        WITH bridges, collect({source: n.id, relationship: type(r), target: m.id})[0..10] as neighbors
        RETURN bridges + neighbors as combined_context
        """
        
        context_data = []
        try:
            with self.retriever.driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.run(cypher, node_ids=node_ids)
                record = result.single()
                if record:
                    context_data = record["combined_context"]
        except Exception as e:
            print(f"  [Cypher Error] {e}")
            
        if not context_data:
            return f"Found relevant entities but no relationships: {', '.join(node_ids)}"
            
        return json.dumps(context_data, indent=2)

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """Generate a grounded answer using the Knowledge Graph context."""
        context = self.get_enriched_context(question)
        
        prompt = ChatPromptTemplate.from_template("""
        You are an Abstract Data Extractor. Answer the question using ONLY the provided Knowledge Graph Context.
        
        Knowledge Graph Context:
        {context}
        
        Analytical Inquiry: {question}
        
        Respond ONLY in JSON format:
        {{
            "reasoning": "...",
            "answer": "..."
        }}
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({"question": question, "context": context})
        
        # Parse the JSON response
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            ans_dict = json.loads(content)
            answer = ans_dict.get("answer", "No answer found in JSON.")
            reasoning = ans_dict.get("reasoning", "")
        except:
            answer = response.content
            reasoning = "Failed to parse JSON reasoning."
            
        return {
            "answer": answer,
            "reasoning": reasoning,
            "context_used": context
        }

    def close(self):
        self.retriever.close()
