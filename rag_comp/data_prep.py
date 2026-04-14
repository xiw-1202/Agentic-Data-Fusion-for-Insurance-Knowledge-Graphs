import os
import sys
import json
import pandas as pd
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config

def get_text_splitter():
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="gpt2",
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def process_pdfs(pdf_dir: str) -> List[Dict]:
    splitter = get_text_splitter()
    all_chunks = []
    
    if not os.path.exists(pdf_dir):
        print(f"Warning: PDF directory {pdf_dir} not found.")
        return []
        
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_dir, filename)
            try:
                loader = PyPDFLoader(path)
                pages = loader.load()
                chunks = splitter.split_documents(pages)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "chunk_id": f"{filename}_{i}",
                        "content": chunk.page_content,
                        "source": filename,
                        "page": chunk.metadata.get("page", -1)
                    })
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                
    return all_chunks

def process_csvs(csv_dir: str) -> List[Dict]:
    all_chunks = []
    
    if not os.path.exists(csv_dir):
        print(f"Warning: CSV directory {csv_dir} not found.")
        return []

    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            path = os.path.join(csv_dir, filename)
            try:
                df = pd.read_csv(path)
                for i, row in df.iterrows():
                    content = " | ".join([f"{col}: {val}" for col, val in row.items()])
                    all_chunks.append({
                        "chunk_id": f"{filename}_{i}",
                        "content": content,
                        "source": filename,
                        "page": -1
                    })
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                
    return all_chunks

def run_prep():
    print("Starting Data Preparation for Auto Dataset (rag_comp)...")
    
    pdf_chunks = process_pdfs(config.AUTO_PDF_DIR)
    csv_chunks = process_csvs(config.AUTO_CSV_DIR)
    
    combined_chunks = pdf_chunks + csv_chunks
    
    os.makedirs(os.path.dirname(config.CHUNKS_FILE), exist_ok=True)
    with open(config.CHUNKS_FILE, "w", encoding='utf-8') as f:
        json.dump(combined_chunks, f, indent=2)
        
    print(f"\n✓ Saved total {len(combined_chunks)} chunks to {config.CHUNKS_FILE}")

if __name__ == "__main__":
    run_prep()
