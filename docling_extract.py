from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import json
import os

# 🆕 New imports
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# --- Paths ---
DOC_SOURCE = "./context/test.pdf"
IMAGE_DIR = "./images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Step 1: Extract text chunks using docling ---
doc = DocumentConverter().convert(source=DOC_SOURCE).document
chunker = HybridChunker()
chunk_iter = list(chunker.chunk(dl_doc=doc))  # Convert to list to debug

chunks = []

if not chunk_iter:
    print("❌ No chunks were generated. Check the input document or chunker.")
else:
    for i, chunk in enumerate(chunk_iter):
        enriched_text = chunker.serialize(chunk=chunk)
        chunks.append({"chunk_index": f"text_{i}", "text": enriched_text})

# --- Step 2: Extract tables using pdfplumber ---
with pdfplumber.open(DOC_SOURCE) as pdf:
    for page_num, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        for idx, table in enumerate(tables):
            if table:
                table_text = "\n".join([", ".join(row) for row in table if row])
                chunks.append({
                    "chunk_index": f"table_p{page_num}_{idx}",
                    "text": f"[Table on Page {page_num + 1}]\n{table_text}"
                })

# --- Step 3: Extract images using PyMuPDF + OCR ---
doc_img = fitz.open(DOC_SOURCE)

for page_index in range(len(doc_img)):
    page = doc_img.load_page(page_index)
    images = page.get_images(full=True)

    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc_img.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_filename = f"image_p{page_index + 1}_{img_index}.{image_ext}"
        image_path = os.path.join(IMAGE_DIR, image_filename)

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        try:
            text = pytesseract.image_to_string(Image.open(image_path))
        except Exception:
            text = "[OCR failed or unsupported image]"

        chunks.append({
            "chunk_index": f"image_p{page_index}_{img_index}",
            "text": f"[Image on Page {page_index + 1}]: {image_filename}\n[OCR Text]\n{text}"
        })

# --- Final: Save all chunks ---
with open("chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)

print(f"Extracted and saved {len(chunks)} chunks (text + tables + images) to chunks.json")


