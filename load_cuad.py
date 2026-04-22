"""
load_cuad.py — Convert CUADv1.json directly into chunks.json
-------------------------------------------------------------
Bypasses docling_extract.py entirely.
Reads the CUAD dataset JSON and converts contract text + Q&A
into the same chunks.json format your embed.py expects.

Usage:
    1. Put CUADv1.json in the same folder as this script
    2. Run: python load_cuad.py
    3. Then run: python embed.py
    4. Then run: python LLM_improved.py
"""

import json
import os

CUAD_FILE   = "CUADv1.json"   # path to your downloaded file
OUTPUT_FILE = "chunks.json"
MAX_CONTRACTS = 5              # how many contracts to load (keep small for speed)
CHUNK_SIZE    = 500            # characters per chunk

print(f"Loading {CUAD_FILE}...")
with open(CUAD_FILE, "r", encoding="utf-8") as f:
    cuad = json.load(f)

# CUADv1.json structure:
# { "data": [ { "title": "...", "paragraphs": [ { "context": "...", "qas": [...] } ] } ] }

contracts = cuad.get("data", [])
print(f"Found {len(contracts)} contracts in dataset.")
print(f"Loading first {MAX_CONTRACTS} contracts...\n")

chunks = []
chunk_idx = 0

for contract_num, contract in enumerate(contracts[:MAX_CONTRACTS]):
    title = contract.get("title", f"Contract_{contract_num}")
    print(f"  Processing: {title[:70]}...")

    for para in contract.get("paragraphs", []):
        context = para.get("context", "").strip()
        if not context:
            continue

        # Split long context into overlapping chunks of CHUNK_SIZE chars
        step = CHUNK_SIZE - 100  # 100 char overlap
        for i in range(0, len(context), step):
            chunk_text = context[i:i + CHUNK_SIZE].strip()
            if len(chunk_text) < 80:  # skip tiny trailing chunks
                continue
            chunks.append({
                "chunk_index": f"text_{chunk_idx}",
                "text": f"[Contract: {title[:50]}]\n\n{chunk_text}"
            })
            chunk_idx += 1

        # Also add each Q&A pair as a searchable chunk
        for qa in para.get("qas", []):
            question = qa.get("question", "").strip()
            answers  = qa.get("answers", [])
            if not question or not answers:
                continue
            answer_text = answers[0].get("text", "").strip()
            if not answer_text or len(answer_text) < 20:
                continue
            chunks.append({
                "chunk_index": f"qa_{chunk_idx}",
                "text": f"[Contract: {title[:50]}]\nClause Question: {question}\nAnswer: {answer_text}"
            })
            chunk_idx += 1

print(f"\n✅ Generated {len(chunks)} chunks from {MAX_CONTRACTS} contracts.")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"✅ Saved to {OUTPUT_FILE}")
print(f"\nNext steps:")
print(f"  1. python embed.py       ← builds the vector index")
print(f"  2. python LLM_improved.py ← start querying")
print(f"\nGood test queries to try:")
print(f"  - Who are the parties in this agreement?")
print(f"  - What are the termination conditions?")
print(f"  - What is the governing law?")
print(f"  - What are the payment terms?")
print(f"  - What are the confidentiality obligations?")
