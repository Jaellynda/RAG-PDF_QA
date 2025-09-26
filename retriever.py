"""
retriever.py
------------
Load FAISS index and metadata, retrieve top-k most relevant chunks for a query.

Usage:
    from retriever import Retriever
    r = Retriever()
    results = r.query("Your question here", top_k=3)
"""

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "faiss.index"
META_FILE = "meta.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_file=INDEX_FILE, meta_file=META_FILE, model_name=MODEL_NAME):
        print("Loading FAISS index...")
        self.index = faiss.read_index(index_file)
        print("Loading metadata...")
        with open(meta_file, "rb") as f:
            self.metadata = pickle.load(f)
        self.model = SentenceTransformer(model_name)

    def query(self, text, top_k=3):
        """Return top_k most relevant chunks for the query text."""
        query_emb = self.model.encode(text).astype("float32").reshape(1, -1)
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results

# Quick test
if __name__ == "__main__":
    retriever = Retriever()
    res = retriever.query("test query")
    for r in res:
        print(r)
