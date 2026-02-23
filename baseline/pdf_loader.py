import os
import sys

# Allow imports from project root (config.py lives there)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import config


def load_and_chunk_pdf(pdf_path, chunk_size=512, chunk_overlap=50):
    """
    Load PDF and chunk into 512-token segments with 50-token overlap
    """
    # Load PDF
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print(f"✓ Loaded {len(pages)} pages")
    print(f"✓ Total characters: {sum(len(p.page_content) for p in pages):,}")

    # Use tiktoken for accurate token counting (GPT-2 tokenizer as approximation)
    encoding = tiktoken.get_encoding("gpt2")

    # Create text splitter with token-based chunking
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="gpt2",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Split all pages into chunks
    chunks = text_splitter.split_documents(pages)

    print(f"✓ Created {len(chunks)} chunks")
    print(
        f"✓ Avg tokens per chunk: {sum(len(encoding.encode(c.page_content)) for c in chunks) / len(chunks):.1f}"
    )

    return chunks


def inspect_chunks(chunks, num_to_show=3):
    """
    Print first few chunks to verify quality
    """
    encoding = tiktoken.get_encoding("gpt2")

    print("\n" + "=" * 80)
    print(f"INSPECTING FIRST {num_to_show} CHUNKS")
    print("=" * 80)

    for i, chunk in enumerate(chunks[:num_to_show]):
        tokens = len(encoding.encode(chunk.page_content))
        print(f"\n--- CHUNK {i+1} ({tokens} tokens) ---")
        print(f"Page: {chunk.metadata.get('page', 'unknown')}")
        print(f"Content preview:\n{chunk.page_content[:300]}...")
        print("-" * 80)


if __name__ == "__main__":
    import json

    # Load and chunk using paths from config
    chunks = load_and_chunk_pdf(config.PDF_PATH, chunk_size=512, chunk_overlap=50)

    # Inspect results
    inspect_chunks(chunks, num_to_show=5)

    chunks_data = [
        {
            "chunk_id": i,
            "page": chunk.metadata.get("page", -1),
            "content": chunk.page_content,
            "source": chunk.metadata.get("source", ""),
        }
        for i, chunk in enumerate(chunks)
    ]

    os.makedirs(os.path.dirname(config.CHUNKS_FILE), exist_ok=True)
    with open(config.CHUNKS_FILE, "w") as f:
        json.dump(chunks_data, f, indent=2)

    print(f"\n✓ Saved {len(chunks)} chunks to {config.CHUNKS_FILE}")
