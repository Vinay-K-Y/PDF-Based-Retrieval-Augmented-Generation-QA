"""
app.py — Flask Backend API for DocuMind RAG Frontend
-----------------------------------------------------
Endpoints:
  POST /upload   — upload a PDF, run extraction + embedding
  POST /query    — query the RAG system
  GET  /status   — check if system is ready

FIX: Uses PyMuPDF direct text extraction (not OCR) as primary method.
     Slide-based PDFs (like IPCV) have embedded text that PyMuPDF reads perfectly.
     Falls back to OCR only when a page has no extractable text.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, shutil, traceback, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

app = Flask(__name__)
CORS(app)

# ── Config ───────────────────────────────────────────────
PERSIST_DIR    = "chroma_db"
EMBED_MODEL    = "all-MiniLM-L6-v2"
LLM_MODEL      = "llama3.1:latest"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNKS_FILE    = "chunks.json"
UPLOAD_FOLDER  = "uploaded_docs"
CHUNK_SIZE     = 600   # characters per chunk
CHUNK_OVERLAP  = 100   # overlap between chunks
RETRIEVAL_K    = 6
RERANK_TOP_N   = 3

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Prompt ───────────────────────────────────────────────
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable assistant. Answer the question using ONLY the context below.

RULES:
1. Answer using ONLY the provided context.
2. If the answer is not in the context, say: "Not found in the provided documents."
3. Be detailed and clear. Use bullet points if listing multiple items.
4. Mention the page number or section if visible in the context.

---
CONTEXT:
{context}
---

QUESTION: {question}

ANSWER:"""
)

# ── Lazy-load components ──────────────────────────────────
_embedding_model = None
_vectorstore     = None
_retriever       = None
_llm             = None
_reranker        = None

def get_components():
    global _embedding_model, _vectorstore, _retriever, _llm, _reranker
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if _reranker is None:
        print("Loading reranker...")
        _reranker = CrossEncoder(RERANKER_MODEL)
    if _llm is None:
        _llm = ChatOllama(model=LLM_MODEL, base_url="http://host.docker.internal:11434")
    if os.path.exists(PERSIST_DIR) and _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=_embedding_model
        )
        # Use similarity search to avoid simsimd MMR bug
        _retriever = _vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K}
        )
    return _embedding_model, _vectorstore, _retriever, _llm, _reranker

def reload_vectorstore():
    global _vectorstore, _retriever
    emb, _, _, _, _ = get_components()
    _vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=emb
    )
    _retriever = _vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K}
    )

# ── PDF Extraction ────────────────────────────────────────
def extract_pdf_text(pdf_path):
    """
    Extract text from PDF using PyMuPDF's direct text layer.
    This works perfectly for slide-based PDFs (PowerPoint exports, lecture notes).
    Falls back to OCR only for pages with no extractable text.
    Returns list of {page, text} dicts.
    """
    import fitz  # PyMuPDF

    pages_text = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # PRIMARY: Direct text extraction from PDF text layer
        text = page.get_text("text").strip()

        # FALLBACK: OCR if page has no text (scanned PDF)
        if len(text) < 30:
            try:
                import pytesseract
                from PIL import Image
                import io
                # Render page as image at 200 DPI for OCR
                mat = fitz.Matrix(200/72, 200/72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img).strip()
                print(f"  Page {page_num+1}: OCR fallback ({len(text)} chars)")
            except Exception as e:
                print(f"  Page {page_num+1}: OCR failed — {e}")
                text = ""

        if len(text) >= 20:
            pages_text.append({
                "page": page_num + 1,
                "text": text
            })
            print(f"  Page {page_num+1}: {len(text)} chars extracted")
        else:
            print(f"  Page {page_num+1}: skipped (too short)")

    doc.close()
    return pages_text


def chunk_pages(pages_text):
    """
    Split page texts into overlapping chunks for embedding.
    Keeps page number in metadata.
    """
    chunks = []
    chunk_idx = 0

    for page_data in pages_text:
        page_num = page_data["page"]
        text     = page_data["text"]

        # Clean up extra whitespace
        text = " ".join(text.split())

        # If page fits in one chunk, keep it whole
        if len(text) <= CHUNK_SIZE:
            if len(text) >= 50:
                chunks.append({
                    "chunk_index": f"p{page_num}_c{chunk_idx}",
                    "page": page_num,
                    "text": f"[Page {page_num}]\n{text}"
                })
                chunk_idx += 1
        else:
            # Sliding window chunking with overlap
            step = CHUNK_SIZE - CHUNK_OVERLAP
            for i in range(0, len(text), step):
                piece = text[i:i + CHUNK_SIZE].strip()
                if len(piece) >= 50:
                    chunks.append({
                        "chunk_index": f"p{page_num}_c{chunk_idx}",
                        "page": page_num,
                        "text": f"[Page {page_num}]\n{piece}"
                    })
                    chunk_idx += 1

    return chunks


def is_useful_chunk(doc):
    content = doc.page_content.strip()
    return len(content) >= 60


# ── Routes ───────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    ready  = os.path.exists(PERSIST_DIR) and os.path.exists(CHUNKS_FILE)
    chunks = 0
    if os.path.exists(CHUNKS_FILE):
        try:
            with open(CHUNKS_FILE) as f:
                chunks = len(json.load(f))
        except:
            pass
    return jsonify({"ready": ready, "chunk_count": chunks, "model": LLM_MODEL})


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported"}), 400

        # Save PDF
        pdf_path = os.path.join(UPLOAD_FOLDER, "document.pdf")
        file.save(pdf_path)
        print(f"\nUploaded: {file.filename}")

        # ── Step 1: Extract text from PDF ──
        print("Extracting text from PDF...")
        pages_text = extract_pdf_text(pdf_path)
        print(f"Extracted text from {len(pages_text)} pages")

        if not pages_text:
            return jsonify({"error": "No text could be extracted from this PDF"}), 400

        # ── Step 2: Chunk the text ──
        print("Chunking text...")
        chunks = chunk_pages(pages_text)
        print(f"Created {len(chunks)} chunks")

        if not chunks:
            return jsonify({"error": "No chunks created from PDF content"}), 400

        # Save chunks
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        # ── Step 3: Embed into ChromaDB ──
        print("Embedding chunks into ChromaDB...")
        from langchain.schema import Document as LCDoc

        emb, _, _, _, _ = get_components()

        # Clear old vectorstore
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)

        docs = [
            LCDoc(
                page_content=c["text"],
                metadata={"chunk_index": c["chunk_index"], "page": c["page"]}
            )
            for c in chunks
        ]

        vs = Chroma.from_documents(
            documents=docs,
            embedding=emb,
            persist_directory=PERSIST_DIR
        )
        vs.persist()
        reload_vectorstore()
        print(f"Done! {len(chunks)} chunks indexed.")

        return jsonify({
            "success":     True,
            "filename":    file.filename,
            "pages":       len(pages_text),
            "chunk_count": len(chunks),
            "message":     f"Processed {file.filename} — {len(pages_text)} pages, {len(chunks)} chunks indexed"
        })

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/query", methods=["POST"])
def query():
    try:
        data     = request.json
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "No question provided"}), 400

        emb, vs, retriever, llm, reranker = get_components()

        if retriever is None:
            return jsonify({"error": "No document indexed yet. Please upload a PDF first."}), 400

        # ── Step 1: Retrieve ──
        raw_docs = retriever.invoke(question)
        useful   = [d for d in raw_docs if is_useful_chunk(d)]

        if not useful:
            return jsonify({
                "answer":   "Not found in the provided documents.",
                "sources":  [],
                "question": question,
                "note":     "No relevant chunks found."
            })

        # ── Step 2: Rerank ──
        pairs  = [(question, d.page_content) for d in useful]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, useful), key=lambda x: x[0], reverse=True)

        best_score = float(ranked[0][0]) if ranked else -99
        print(f"Query: '{question}' | Best reranker score: {best_score:.3f}")

        # Only reject if truly unrelated (very low threshold)
        if best_score < -8.0:
            return jsonify({
                "answer":   "Not found in the provided documents.",
                "sources":  [],
                "question": question,
                "note":     f"No relevant content found (score: {best_score:.2f})"
            })

        top = [d for _, d in ranked[:RERANK_TOP_N]]

        # ── Step 3: Generate ──
        context = "\n\n---\n\n".join([d.page_content for d in top])
        answer  = llm.invoke(PROMPT.format(context=context, question=question)).content.strip()

        sources = [
            {
                "chunk_index": d.metadata.get("chunk_index", "unknown"),
                "page":        d.metadata.get("page", "?"),
                "preview":     d.page_content[:300].strip(),
                "score":       round(float(ranked[i][0]), 3)
            }
            for i, d in enumerate(top)
        ]

        return jsonify({
            "answer":   answer,
            "sources":  sources,
            "question": question
        })

    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/load_cuad", methods=["POST"])
def load_cuad():
    """Load from CUADv1.json instead of PDF upload."""
    try:
        cuad_path = "CUADv1.json"
        if not os.path.exists(cuad_path):
            return jsonify({"error": "CUADv1.json not found in project folder"}), 400

        with open(cuad_path, "r", encoding="utf-8") as f:
            cuad = json.load(f)

        contracts  = cuad.get("data", [])
        max_c      = int(request.json.get("max_contracts", 5))
        chunks     = []
        chunk_idx  = 0

        for contract in contracts[:max_c]:
            title = contract.get("title", "Contract")
            for para in contract.get("paragraphs", []):
                context = para.get("context", "").strip()
                step    = CHUNK_SIZE - CHUNK_OVERLAP
                for i in range(0, len(context), step):
                    piece = context[i:i + CHUNK_SIZE].strip()
                    if len(piece) < 80:
                        continue
                    chunks.append({
                        "chunk_index": f"text_{chunk_idx}",
                        "page": 0,
                        "text": f"[Contract: {title[:60]}]\n\n{piece}"
                    })
                    chunk_idx += 1

        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        from langchain.schema import Document as LCDoc
        emb, _, _, _, _ = get_components()
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)

        docs = [LCDoc(page_content=c["text"], metadata={"chunk_index": c["chunk_index"], "page": c["page"]}) for c in chunks]
        vs   = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=PERSIST_DIR)
        vs.persist()
        reload_vectorstore()

        return jsonify({
            "success":     True,
            "contracts":   max_c,
            "chunk_count": len(chunks),
            "message":     f"Loaded {max_c} CUAD contracts — {len(chunks)} chunks indexed"
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  DocuMind RAG API")
    print("  http://localhost:5000")
    print("  Open index.html in your browser")
    print("=" * 50)
    app.run(host='0.0.0.0', debug=True, port=5000)
