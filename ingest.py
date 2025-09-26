"""
ingest.py
---------
Extract text from PDFs in sample_pdfs/, chunk the text, compute embeddings (sentence-transformers),
and save a FAISS index and metadata (meta.pkl).

Usage:
    python ingest.py
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import faiss

MODEL_NAME = "all-MiniLM-L6-v2"  # small and fast
SAMPLE_DIR = "sample_pdfs"
INDEX_FILE = "faiss.index"
META_FILE = "meta.pkl"

def extract_text_from_pdf(path):
    """Return concatenated text from all pages of a PDF file."""
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)

def chunk_text(text, chunk_size=250, chunk_overlap=50):
    """Chunk text into approximately chunk_size-word chunks with overlap."""
    tokens = text.split()
    if not tokens:
        return []
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - chunk_overlap
    return chunks

def build_index(pdf_paths, index_out=INDEX_FILE, meta_out=META_FILE, model_name=MODEL_NAME):
    """Build a FAISS index from PDF files and write metadata."""
    model = SentenceTransformer(model_name)
    all_embeddings = []
    metadata = []
    for p in pdf_paths:
        print("Processing:", p)
        text = extract_text_from_pdf(p)
        chunks = chunk_text(text)
        for idx, ch in enumerate(chunks):
            emb = model.encode(ch)
            all_embeddings.append(emb)
            metadata.append({"source": os.path.basename(p), "chunk_id": idx, "text": ch})
    if not all_embeddings:
        raise ValueError("No text chunks found. Put PDFs in sample_pdfs/")
    embeddings = np.vstack(all_embeddings).astype("float32")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, index_out)
    with open(meta_out, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Built index {index_out} with {len(metadata)} chunks.")

if __name__ == "__main__":
    pdfs = []
    if os.path.isdir(SAMPLE_DIR):
        pdfs = [os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("No PDFs found in sample_pdfs/. Please add a PDF and re-run.")
    else:
        build_index(pdfs)
