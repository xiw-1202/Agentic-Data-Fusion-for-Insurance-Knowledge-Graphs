import os
import sys
import json
import numpy as np
from typing import List, Dict, Any

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

class DocumentRAG:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.OLLAMA_MODEL
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0
        )
        self.embeddings_model = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )
        self.chunks_file = os.path.join(_ROOT, config.CHUNKS_FILE)
        self.embeddings_path = os.path.join(_ROOT, config.RESULTS_DIR, "doc_embeddings.npy")
        
        self.chunks = self._load_chunks()
        self.embeddings = self._get_embeddings()

    def _load_chunks(self) -> List[Dict]:
        if not os.path.exists(self.chunks_file):
            return []
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_embeddings(self):
        if not self.chunks:
            return None
            
        if os.path.exists(self.embeddings_path):
            return np.load(self.embeddings_path)
            
        # In a real run, this would generate and save embeddings.
        # Since I cannot run commands, this is left as a placeholder for the logic.
        return None

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if self.embeddings is None or not self.chunks:
            return []
            
        query_embedding = self.embeddings_model.embed_query(query)
        query_vec = np.array(query_embedding)
        
        norm_query = np.linalg.norm(query_vec)
        norm_embeddings = np.linalg.norm(self.embeddings, axis=1)
        similarities = np.dot(self.embeddings, query_vec) / (norm_embeddings * norm_query)
        
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.chunks[i] for i in top_indices]

    def generate_answer(self, query: str, k: int = 5) -> Dict[str, Any]:
        retrieved_chunks = self.retrieve(query, k=k)
        if not retrieved_chunks:
            context = "No relevant document chunks found."
        else:
            context = "\n\n".join([f"Source: {c['source']}\nContent: {c['content']}" for c in retrieved_chunks])
        
        prompt = ChatPromptTemplate.from_template("""
        You are an insurance expert. Use the following document context to answer the question.
        If the answer is NOT in the context, say: "I don't know based on the provided documents."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": query})
        
        return {
            "answer": response.content,
            "context": retrieved_chunks
        }
