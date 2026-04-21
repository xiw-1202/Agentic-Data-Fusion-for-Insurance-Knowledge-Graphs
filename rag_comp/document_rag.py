import os
import sys
import json
import warnings
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate


def _format_document_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant document chunks found."
    return "\n\n".join(
        f"Source: {chunk['source']}\nContent: {chunk['content']}"
        for chunk in chunks
    )


def _is_context_length_error(exc: Exception) -> bool:
    return "input length exceeds the context length" in str(exc).lower()


def resolve_document_embeddings_path(root_dir: str) -> str:
    canonical = os.path.join(root_dir, config.DOCUMENT_EMBEDDINGS_FILE)
    legacy = os.path.join(root_dir, config.LEGACY_DOCUMENT_EMBEDDINGS_FILE)

    if os.path.exists(canonical):
        return canonical
    if os.path.exists(legacy):
        return legacy
    return canonical


def validate_document_embeddings(
    chunks: List[Dict[str, Any]], embeddings: np.ndarray
) -> np.ndarray:
    array = np.asarray(embeddings)
    if array.ndim != 2:
        raise ValueError(
            f"Document embeddings must be a 2D array, received shape {array.shape}."
        )
    if len(chunks) != array.shape[0]:
        raise ValueError(
            "Document embeddings row count does not match chunk count: "
            f"{array.shape[0]} embeddings for {len(chunks)} chunks."
        )
    return array

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
        self.embedding_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks_file = os.path.join(_ROOT, config.CHUNKS_FILE)
        self.embeddings_path = os.path.join(_ROOT, config.DOCUMENT_EMBEDDINGS_FILE)
        self.legacy_embeddings_path = os.path.join(
            _ROOT, config.LEGACY_DOCUMENT_EMBEDDINGS_FILE
        )
        
        self.chunks = self._load_chunks()
        self.embeddings = self._get_embeddings()

    def _load_chunks(self) -> List[Dict]:
        if not os.path.exists(self.chunks_file):
            return []
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_cached_embeddings(self) -> Optional[np.ndarray]:
        existing_path = resolve_document_embeddings_path(_ROOT)
        if not os.path.exists(existing_path):
            return None

        try:
            embeddings = validate_document_embeddings(
                self.chunks, np.load(existing_path, allow_pickle=False)
            )
        except (OSError, ValueError) as exc:
            warnings.warn(
                "Ignoring stale or unreadable document embeddings cache at "
                f"{existing_path}: {exc}. Regenerating embeddings from the current "
                "chunk set.",
                RuntimeWarning,
            )
            return None

        # Normalize older cache names into the canonical location.
        if existing_path != self.embeddings_path:
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
            np.save(self.embeddings_path, embeddings)

        return embeddings

    def _embed_long_text(self, text: str) -> List[float]:
        subchunks = self.embedding_splitter.split_text(text)
        if not subchunks:
            subchunks = [text]

        if len(subchunks) == 1:
            # As a last resort, trim to a manageable prefix rather than fail the run.
            truncated = text[: max(config.CHUNK_SIZE * 8, 1024)]
            warnings.warn(
                "A single document chunk exceeded the embedding context length. "
                "Using a truncated prefix for its embedding.",
                RuntimeWarning,
            )
            return self.embeddings_model.embed_documents([truncated])[0]

        warnings.warn(
            f"Splitting an oversized document chunk into {len(subchunks)} smaller "
            "segments for embedding and averaging the result.",
            RuntimeWarning,
        )
        subchunk_embeddings = np.array(self._embed_texts(subchunks))
        return np.mean(subchunk_embeddings, axis=0).tolist()

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        try:
            return self.embeddings_model.embed_documents(texts)
        except Exception as exc:
            if not _is_context_length_error(exc):
                raise

            if len(texts) == 1:
                return [self._embed_long_text(texts[0])]

            mid = max(1, len(texts) // 2)
            warnings.warn(
                f"Embedding batch of size {len(texts)} exceeded the model context "
                "limit. Retrying with smaller batches.",
                RuntimeWarning,
            )
            return self._embed_texts(texts[:mid]) + self._embed_texts(texts[mid:])

    def _generate_embeddings(self) -> np.ndarray:
        texts = [chunk["content"] for chunk in self.chunks]
        embeddings: List[List[float]] = []
        batch_size = 20

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings.extend(self._embed_texts(batch))

        embeddings_array = validate_document_embeddings(self.chunks, np.array(embeddings))
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        np.save(self.embeddings_path, embeddings_array)
        return embeddings_array

    def _get_embeddings(self):
        if not self.chunks:
            return None

        cached_embeddings = self._load_cached_embeddings()
        if cached_embeddings is not None:
            return cached_embeddings

        return self._generate_embeddings()

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

    def retrieve_context(self, query: str, k: int = 5) -> Dict[str, Any]:
        retrieved_chunks = self.retrieve(query, k=k)
        return {
            "chunks": retrieved_chunks,
            "context_text": _format_document_context(retrieved_chunks),
            "chunk_count": len(retrieved_chunks),
        }

    def generate_answer(self, query: str, k: int = 5) -> Dict[str, Any]:
        retrieval = self.retrieve_context(query, k=k)
        retrieved_chunks = retrieval["chunks"]
        context = retrieval["context_text"]
        
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
            "context": retrieved_chunks,
            "context_text": context,
        }
