import os
import sys
from typing import List, Dict, Any

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config
from rag_comp.document_rag import DocumentRAG
from rag_comp.graph_rag import GraphRAG
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class HybridRAG:
    def __init__(self, model_name: str = None):
        self.doc_rag = DocumentRAG(model_name=model_name)
        self.graph_rag = GraphRAG(model_name=model_name)
        self.model_name = model_name or config.OLLAMA_MODEL
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0
        )

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """Combine contexts from both Graph and Doc and generate total answer."""
        graph_retrieval = self.graph_rag.retrieve_context(question)
        graph_context = graph_retrieval.get("context_text", "No graph context.")

        doc_retrieval = self.doc_rag.retrieve_context(question)
        doc_context = doc_retrieval.get("context_text", "No document context.")
        
        prompt = ChatPromptTemplate.from_template("""
        You are an elite insurance investigator. You have two sources of information:
        1. Knowledge Graph (Triplets/Relationships)
        2. Document Chunks (Raw text)
        
        Knowledge Graph Context:
        {graph_context}
        
        Document Context:
        {doc_context}
        
        Analytical Inquiry: {question}
        
        Synthesize an answer using both sources. Prioritize Knowledge Graph for structural facts and Documents for narrative details. 
        If there is a conflict, mention BOTH.
        If neither source supports an answer, say you don't know based on the provided evidence.
        
        Answer:
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({
            "question": question,
            "graph_context": graph_context,
            "doc_context": doc_context
        })
        
        return {
            "answer": response.content,
            "graph_context": graph_context,
            "doc_context": doc_context,
            "graph_hits": graph_retrieval.get("entity_hits", []),
            "doc_chunks": doc_retrieval.get("chunks", []),
        }

    def close(self):
        self.graph_rag.close()
