import os
import sys
import json
import random
import argparse
from typing import List, Dict

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import config

def load_chunks():
    path = os.path.join(_ROOT, config.CHUNKS_FILE)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def generate_doc_first_questions(chunks: List[Dict], llm, count: int = 20):
    # Filter for "meaty" chunks and group by type
    pdf_chunks = [c for c in chunks if c.get('source', '').endswith('.pdf') and len(c.get('content', '')) > 400]
    csv_chunks = [c for c in chunks if c.get('source', '').endswith('.csv') and len(c.get('content', '')) > 400]
    
    # Target 50/50 split if possible
    half = count // 2
    sampled_chunks = []
    
    if len(pdf_chunks) >= half:
        sampled_chunks.extend(random.sample(pdf_chunks, half))
    else:
        sampled_chunks.extend(pdf_chunks)
        
    remaining = count - len(sampled_chunks)
    if len(csv_chunks) >= remaining:
        sampled_chunks.extend(random.sample(csv_chunks, remaining))
    else:
        sampled_chunks.extend(csv_chunks)

    # Shuffle final mix
    random.shuffle(sampled_chunks)

    questions = []
    task_id = 1

    print(f"Generating balanced Doc-First questions (PDF: {len([c for c in sampled_chunks if c['source'].endswith('.pdf')])}, CSV: {len([c for c in sampled_chunks if c['source'].endswith('.csv')])})...")

    for chunk in sampled_chunks:
        content = chunk['content']
        source = chunk['source']
        chunk_id = chunk['chunk_id']

        prompt = f"""You are an insurance data auditor. Your goal is to generate a high-quality evaluation question based on a specific document chunk.

DOCUMENT CHUNK:
Source: {source}
Content: {content}

CRITICAL RULES FOR QUESTION QUALITY:
1. NO AMBIGUITY: Never use pronouns like "this claim", "this policy", "the customer", or "that record". To the RAG agent, these are meaningless without context.
2. INCLUDE IDENTIFIERS: If the text contains a specific ID (e.g., Claim Number like '64060460', Policy Number like 'REN538000900', or a specific Name), you MUST include that identifier in the question.
   - WRONG: "What is the status of this claim?"
   - RIGHT: "What is the current status of claim 64060460?"
3. SOURCE CONTEXT: If it's a general policy question, mention the source type.
   - RIGHT: "According to the Auto Service Contract, what is the required transfer fee?"
4. GROUNDED ANSWER: Provide a grounded "expected_answer" based strictly on the text.
5. KEYWORDS: List 3 crucial keywords (IDs, amounts, or terms).

Output EXACTLY in JSON format:
{{
  "category": "{source.split('.')[-1]}_fact",
  "question": "...",
  "expected_answer": "...",
  "keywords": ["...", "...", "..."]
}}
"""
        try:
            print(f"  [{task_id}/{len(sampled_chunks)}] Processing chunk {chunk_id}...")
            response = llm.invoke([HumanMessage(content=prompt)])
            raw_content = response.content.strip()
            
            # Clean JSON
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            
            data = json.loads(raw_content)
            data['id'] = f"doc_cov_{task_id}"
            data['source_ids'] = [chunk_id]
            questions.append(data)
            task_id += 1
        except Exception as e:
            print(f"    [Error] {e}")

    return questions

def run():
    print("="*60)
    print("Doc-First Evaluation Generator (Reverse Benchmark)")
    print("="*60)

    chunks = load_chunks()
    if not chunks:
        print("Error: No chunks found. Check your data/auto/processed directory.")
        return

    llm = ChatOllama(
        model=config.OLLAMA_MODEL, 
        base_url=config.OLLAMA_BASE_URL, 
        temperature=0.1, 
        format="json"
    )

    dataset = generate_doc_first_questions(chunks, llm, count=20)

    out_dir = os.path.join(_ROOT, config.RESULTS_DIR, 'evaluation_datasets')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'doc_primary_gold_standard.json')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✓ Generated {len(dataset)} doc-first questions.")
    print(f"✓ Saved to: {out_path}")

if __name__ == "__main__":
    run()
