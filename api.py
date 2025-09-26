"""
api.py
------
Minimal FastAPI API to query the RAG pipeline.
Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from generator import Generator

app = FastAPI(title="RAG-PDF-QA API")
generator = Generator()

class Query(BaseModel):
    question: str
    top_k: int = 3

@app.post("/query")
def query_endpoint(q: Query):
    if not q.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = generator.generate(q.question, top_k=q.top_k)
    return {"question": q.question, "answer": answer}
