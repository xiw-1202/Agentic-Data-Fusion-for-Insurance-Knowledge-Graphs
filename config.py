import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j AuraDB
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", NEO4J_USERNAME)  # AuraDB uses username as DB name

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Chunking (matches baseline_pdf_loader.py)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Paths — relative to project root
CHUNKS_FILE       = "data/flood/processed/pdf_chunks.json"   # baseline 512-token chunks
ZONE1_CHUNKS_FILE = "data/flood/processed/zone1_chunks.json" # Zone 1 section-aware chunks
PDF_PATH          = "data/flood/raw/pdf/fema_F-123-general-property-SFIP_2021.pdf"
OPENFEMA_DIR      = "data/flood/raw/openfema"
RESULTS_DIR       = "data/results"
