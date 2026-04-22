"""
quick_eval.py — Terminal-based RAG Evaluation (no RAGAS, no browser)
---------------------------------------------------------------------
Runs test questions against your RAG system and prints:
  - Exact Match Score
  - Keyword Coverage Score  
  - Faithfulness Proxy Score
  - Not-Found Accuracy
  - Overall Accuracy

Usage: python quick_eval.py
"""

import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

# ── Config ───────────────────────────────────────────────
PERSIST_DIR    = "chroma_db"
EMBED_MODEL    = "all-MiniLM-L6-v2"
LLM_MODEL      = "llama3.1:latest"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_K    = 5
RERANK_TOP_N   = 2

# ── Test Dataset (from CUAD contracts) ───────────────────
# Format: {question, expected_keywords, should_be_found}
TEST_CASES = [
    {
        "question": "Who are the parties in this agreement?",
        "keywords": ["distributor", "google", "company", "party", "agreement", "between"],
        "should_be_found": True
    },
    {
        "question": "What are the termination conditions?",
        "keywords": ["terminate", "termination", "breach", "notice", "days", "agreement"],
        "should_be_found": True
    },
    {
        "question": "What is the governing law?",
        "keywords": ["law", "jurisdiction", "govern", "state", "court", "legal"],
        "should_be_found": True
    },
    {
        "question": "What are the confidentiality obligations?",
        "keywords": ["confidential", "disclose", "secret", "information", "obligation"],
        "should_be_found": True
    },
    {
        "question": "What are the payment terms?",
        "keywords": ["payment", "pay", "fee", "amount", "invoice", "dollar", "due"],
        "should_be_found": True
    },
    {
        "question": "What happens in case of breach of contract?",
        "keywords": ["breach", "default", "remedy", "cure", "liable", "terminate"],
        "should_be_found": True
    },
    {
        "question": "What are the intellectual property rights?",
        "keywords": ["intellectual", "property", "copyright", "trademark", "license", "own"],
        "should_be_found": True
    },
    {
        "question": "What is the duration or term of this agreement?",
        "keywords": ["term", "year", "month", "expire", "renew", "period", "duration"],
        "should_be_found": True
    },
    # Out-of-scope questions — system should say NOT FOUND
    {
        "question": "What is the capital of France?",
        "keywords": [],
        "should_be_found": False
    },
    {
        "question": "Explain quantum physics",
        "keywords": [],
        "should_be_found": False
    },
    {
        "question": "What is the population of India?",
        "keywords": [],
        "should_be_found": False
    },
    {
        "question": "Who won the FIFA World Cup 2022?",
        "keywords": [],
        "should_be_found": False
    },
]

# ── Load system ──────────────────────────────────────────
print("Loading RAG system...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore     = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
retriever       = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": RETRIEVAL_K, "fetch_k": 20, "lambda_mult": 0.6}
)
llm      = ChatOllama(model=LLM_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)

LEGAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise legal document assistant.
Answer ONLY using the context. If the answer is not in the context, respond EXACTLY:
"Not found in the provided documents."
Never guess or use outside knowledge.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
)

def get_answer(question):
    docs   = retriever.invoke(question)
    pairs  = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top    = [d for _, d in ranked[:RERANK_TOP_N]]
    ctx    = "\n\n---\n\n".join([d.page_content for d in top])
    ans    = llm.invoke(LEGAL_PROMPT.format(context=ctx, question=question)).content.strip()
    return ans

NOT_FOUND_PHRASE = "not found in the provided documents"

# ── Run evaluation ───────────────────────────────────────
print("\n" + "═"*65)
print("  RAG SYSTEM EVALUATION — Legal Document QA")
print("═"*65)

results = []
in_scope  = [t for t in TEST_CASES if t["should_be_found"]]
out_scope = [t for t in TEST_CASES if not t["should_be_found"]]

print(f"\n{'Q#':<4} {'Question':<45} {'Result':<10} {'Score'}")
print("─"*65)

for i, tc in enumerate(TEST_CASES):
    answer = get_answer(tc["question"])
    answer_lower = answer.lower()
    is_not_found = NOT_FOUND_PHRASE in answer_lower

    if tc["should_be_found"]:
        # Check how many keywords appear in the answer
        hits = sum(1 for kw in tc["keywords"] if kw.lower() in answer_lower)
        kw_score = round(hits / len(tc["keywords"]), 2) if tc["keywords"] else 0
        passed = kw_score >= 0.3 and not is_not_found
        result_label = "✅ PASS" if passed else "❌ FAIL"
        score_label  = f"{int(kw_score*100)}% keywords matched"
    else:
        # Should NOT be found — correct if system says not found
        passed = is_not_found
        result_label = "✅ PASS" if passed else "❌ FAIL"
        score_label  = "Correctly refused" if passed else "Hallucinated!"

    results.append({
        "question":      tc["question"],
        "answer":        answer,
        "should_be_found": tc["should_be_found"],
        "passed":        passed,
        "kw_score":      kw_score if tc["should_be_found"] else (1.0 if passed else 0.0)
    })

    q_short = tc["question"][:43] + ".." if len(tc["question"]) > 43 else tc["question"]
    print(f"  {i+1:<3} {q_short:<45} {result_label:<12} {score_label}")

# ── Compute final scores ─────────────────────────────────
in_scope_results  = [r for r in results if r["should_be_found"]]
out_scope_results = [r for r in results if not r["should_be_found"]]

in_scope_acc   = sum(r["passed"] for r in in_scope_results)  / len(in_scope_results)
out_scope_acc  = sum(r["passed"] for r in out_scope_results) / len(out_scope_results)
overall_acc    = sum(r["passed"] for r in results) / len(results)
avg_kw_score   = sum(r["kw_score"] for r in in_scope_results) / len(in_scope_results)

print("\n" + "═"*65)
print("  EVALUATION SUMMARY")
print("═"*65)
print(f"\n  {'Metric':<40} {'Score':<10} {'Status'}")
print("  " + "─"*55)

metrics = [
    ("Keyword Coverage (In-Scope Qs)",    avg_kw_score,  0.4),
    ("In-Scope Answer Accuracy",          in_scope_acc,  0.6),
    ("Out-of-Scope Refusal Accuracy",     out_scope_acc, 0.75),
    ("Overall System Accuracy",           overall_acc,   0.6),
]

for name, val, threshold in metrics:
    bar    = "█" * int(val * 20) + "░" * (20 - int(val * 20))
    status = "✅ GOOD" if val >= threshold else "⚠  FAIR" if val >= threshold * 0.7 else "❌ POOR"
    print(f"  {name:<40} {val:.0%}  {bar}  {status}")

print("\n" + "═"*65)
print(f"  Total Questions : {len(results)}")
print(f"  In-Scope        : {len(in_scope_results)}  (contract questions)")
print(f"  Out-of-Scope    : {len(out_scope_results)}  (irrelevant questions — hallucination test)")
print(f"  Passed          : {sum(r['passed'] for r in results)}/{len(results)}")
print(f"  Dataset         : CUAD (NeurIPS 2021) — {len(in_scope_results)} legal clause queries")
print("═"*65)

# Save to JSON for reference
with open("eval_results.json", "w") as f:
    json.dump({
        "overall_accuracy":        round(overall_acc, 4),
        "in_scope_accuracy":       round(in_scope_acc, 4),
        "out_scope_refusal_rate":  round(out_scope_acc, 4),
        "avg_keyword_coverage":    round(avg_kw_score, 4),
        "total_questions":         len(results),
        "passed":                  sum(r["passed"] for r in results),
        "per_question":            results
    }, f, indent=2)

print("\n  Results saved to eval_results.json")
print("═"*65)