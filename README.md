# -PDF-Based-Retrieval-Augmented-Generation-QA

# Document Intelligence with LLM: Multi-Modal PDF Parsing and QA System

## Project Overview

This project is an intelligent document question-answering system that leverages Large Language Models (LLMs) to extract and interpret information from PDFs containing text, tables, and images. It is designed to help users query complex documents like technical manuals, datasheets, and reports through a conversational interface.

---

## Features

- **Text Extraction:** Parses and chunks PDF text for better context understanding.
- **Table Extraction:** Detects and cleans tables from PDFs, preserving structured data.
- **Image OCR:** Extracts text from images within PDFs using Tesseract OCR.
- **Semantic Search:** Uses HuggingFace embeddings to vectorize chunks and store them in a Chroma vector database.
- **LLM-powered QA:** Integrates with a local LLM (Ollama + LLaMA 3) for accurate question answering.
- **Source Attribution:** Returns source document snippets alongside answers for transparency.

---

## How It Works

1. The PDF is converted into document chunks (text, tables, images).
2. Tables are cleaned and serialized into readable text.
3. Images are processed with OCR to extract any embedded text.
4. All chunks are embedded into vector space using MiniLM.
5. A Chroma vector store indexes these embeddings for retrieval.
6. User queries are answered by retrieving relevant chunks and passing them to the LLM.
7. The LLM generates answers with references to the source chunks.

