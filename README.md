# RAG-PDF-QA: Retrieval-Augmented Generation over PDFs

**Short description**  
This is a compact demonstration of a Retrieval-Augmented Generation (RAG) pipeline over PDF documents.  
It ingests PDFs, splits text into chunks, builds embeddings (sentence-transformers), stores them in a FAISS vector index,  
and answers questions by retrieving context and generating an answer with either a Hugging Face model or OpenAI.

---

## What this demonstrates
- PDF extraction (PyMuPDF)
- Chunking and embedding (sentence-transformers)
- Vector search (FAISS)
- RAG: retrieval + LLM generation (Hugging Face / OpenAI)
- Minimal FastAPI endpoint to demo the pipeline

---

## Quick start (local)

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/rag-pdf-qa
   cd rag-pdf-qa
