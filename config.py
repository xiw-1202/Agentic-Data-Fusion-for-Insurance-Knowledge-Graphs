import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j AuraDB
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", NEO4J_USERNAME)  # AuraDB uses username as DB name

# Ollama
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",     "llama3.1:8b")
OLLAMA_MODEL_ALT = os.getenv("OLLAMA_MODEL_ALT", "qwen3:8b")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "mxbai-embed-large")

# Neo4j Indices
VECTOR_INDEX_NAME   = "entity_vector_index"
FULLTEXT_INDEX_NAME = "entity_fulltext_index"
VECTOR_DIMENSION    = 1024  # Dimension for mxbai-embed-large

# Chunking (matches baseline_pdf_loader.py)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Paths — relative to project root
AUTO_DATA_DIR     = "data/auto"
AUTO_CSV_DIR      = "data/auto/csv"
AUTO_PDF_DIR      = "data/auto/pdf"
CHUNKS_FILE       = "data/auto/processed/pdf_chunks.json"
ZONE1_CHUNKS_FILE = "data/auto/processed/zone1_chunks.json"
RESULTS_DIR       = "data/results"
DOCUMENT_EMBEDDINGS_FILE = "data/results/document_embeddings.npy"
LEGACY_DOCUMENT_EMBEDDINGS_FILE = "data/results/doc_embeddings.npy"
GOLD_STANDARD_FILE = "data/results/evaluation_datasets/gold_standard.json"
DOC_PRIMARY_EVAL_FILE = "data/results/evaluation_datasets/doc_primary_gold_standard.json"
GRAPH_PRIMARY_EVAL_FILE = "data/results/evaluation_datasets/auto_gold_standard.json"
