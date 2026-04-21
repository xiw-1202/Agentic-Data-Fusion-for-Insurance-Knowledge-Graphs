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


def _format_graph_context(context_data: List[Dict[str, Any]]) -> str:
    if not context_data:
        return "No relevant graph relationships found."
    return json.dumps(context_data, indent=2)

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

    def _is_relationship_summary_question(self, query: str) -> bool:
        lowered = query.lower()
        summary_markers = [
            "most common relationship",
            "common relationship types",
            "common relationships",
            "relationship types between claims",
        ]
        return any(marker in lowered for marker in summary_markers)

    def _get_relationship_summary_context(self, limit: int = 10) -> Dict[str, Any]:
        cypher = """
        MATCH (n:Entity)-[r]-(m)
        WHERE n.id STARTS WITH 'CLM'
        RETURN type(r) AS relationship, count(*) AS frequency
        ORDER BY frequency DESC, relationship ASC
        LIMIT $limit
        """

        context_data: List[Dict[str, Any]] = []
        try:
            with self.retriever.driver.session(database=config.NEO4J_DATABASE) as session:
                for record in session.run(cypher, limit=limit):
                    context_data.append(
                        {
                            "relationship": record["relationship"],
                            "frequency": record["frequency"],
                        }
                    )
        except Exception as e:
            print(f"  [Cypher Error] {e}")

        context_text = _format_graph_context(context_data)
        if not context_data:
            context_text = "No relationship summary data found in Knowledge Graph."

        return {
            "context_text": context_text,
            "context_data": context_data,
            "entity_hits": [],
            "strategy": "relationship_summary",
        }

    def _get_entity_neighborhood_context(
        self, query: str, top_k: int = 5
    ) -> Dict[str, Any]:
        hits = self.retriever.search_entities(query, top_k=top_k)
        if not hits:
            return {
                "context_text": "No relevant entities found in Knowledge Graph.",
                "context_data": [],
                "entity_hits": [],
                "strategy": "entity_neighborhood",
            }

        node_ids = [str(hit["id"]) for hit in hits]

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
            context_text = f"Found relevant entities but no relationships: {', '.join(node_ids)}"
        else:
            context_text = _format_graph_context(context_data)

        return {
            "context_text": context_text,
            "context_data": context_data,
            "entity_hits": hits,
            "strategy": "entity_neighborhood",
        }

    def retrieve_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve graph context without performing answer generation."""
        if self._is_relationship_summary_question(query):
            return self._get_relationship_summary_context()
        return self._get_entity_neighborhood_context(query, top_k=top_k)

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """Generate a grounded answer using the Knowledge Graph context."""
        retrieval = self.retrieve_context(question)
        context = retrieval["context_text"]
        
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
            "context_used": context,
            "entity_hits": retrieval["entity_hits"],
            "retrieval_strategy": retrieval["strategy"],
        }

    def close(self):
        self.retriever.close()
