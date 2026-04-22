"""
LLM.py — Improved RAG Query Engine
Enhancements:
  - Legal-specific prompt (no hallucination, cite clause)
  - MMR retrieval (diverse chunks, avoids duplicates)
  - Cross-encoder reranker for better top-k selection
  - k=5 retrieval with reranking to final top-2
  - Source document display with chunk metadata
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
import json

# ── Config ──────────────────────────────────────────────
PERSIST_DIR    = "chroma_db"
EMBED_MODEL    = "all-MiniLM-L6-v2"
LLM_MODEL      = "llama3.1:latest"
RETRIEVAL_K    = 5      # fetch top-5 via MMR
RERANK_TOP_N   = 2      # keep top-2 after reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Load vectorstore ─────────────────────────────────────
print("Loading vectorstore...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)

# ── MMR Retriever (diverse, non-redundant chunks) ────────
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ── Cross-encoder Reranker ───────────────────────────────
print("Loading reranker...")
reranker = CrossEncoder(RERANKER_MODEL)

def is_useful_chunk(doc):
    """Filter out image chunks with empty or garbage OCR text."""
    content = doc.page_content.strip()
    chunk_id = doc.metadata.get("chunk_index", "")
    if chunk_id.startswith("image_"):
        ocr_text = content.split("[OCR Text]")[-1].strip() if "[OCR Text]" in content else ""
        if len(ocr_text) < 50:
            return False
    return True

def rerank_docs(query, docs, top_n=RERANK_TOP_N):
    """Filter bad chunks, score against query, return top_n."""
    useful = [d for d in docs if is_useful_chunk(d)]
    if not useful:
        useful = docs  # fallback if everything got filtered
    pairs = [(query, doc.page_content) for doc in useful]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, useful), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]

# ── Legal-specific prompt ────────────────────────────────
LEGAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise legal document assistant.

STRICT RULES:
1. Answer ONLY using the provided context below.
2. If the answer is not in the context, respond exactly: "Not found in the provided documents."
3. Never infer, assume, or use outside legal knowledge.
4. Always cite the relevant section, clause, or page if identifiable.
5. Be concise and use plain language unless the user asks for legal terminology.

---
CONTEXT:
{context}
---

QUESTION: {question}

ANSWER (cite source if possible):"""
)

# ── LLM ─────────────────────────────────────────────────
llm = ChatOllama(model=LLM_MODEL, base_url="http://host.docker.internal:11434")


# ── Query loop ───────────────────────────────────────────
print("\n=== Legal RAG System Ready ===")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Query: ").strip()
    if user_input.lower() == "exit":
        break
    if not user_input:
        continue

    # Step 1: Retrieve with MMR
    raw_docs = retriever.invoke(user_input)

    # Step 2: Rerank
    reranked_docs = rerank_docs(user_input, raw_docs)

    # Step 3: Build context from reranked docs and query LLM
    context = "\n\n---\n\n".join([d.page_content for d in reranked_docs])
    response = llm.invoke(LEGAL_PROMPT.format(context=context, question=user_input)).content

    print(f"\n Answer:\n{response}")
    print(f"\n Retrieved & Reranked Sources ({len(reranked_docs)}):")
    for i, doc in enumerate(reranked_docs):
        chunk_id = doc.metadata.get("chunk_index", "unknown")
        print(f"  [{i+1}] chunk_index: {chunk_id}")
        print(f"       {doc.page_content[:200].strip()}...")
    print()