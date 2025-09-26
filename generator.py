"""
generator.py
------------
Generate answers from retrieved chunks using OpenAI GPT or Hugging Face model.

Usage:
    from generator import Generator
    g = Generator()
    answer = g.generate(query="Your question", contexts=[list of retrieved chunks])
"""

import os
from retriever import Retriever

try:
    import openai
except ImportError:
    openai = None

MODEL = "gpt-3.5-turbo"  # OpenAI model, requires API key in .env

class Generator:
    def __init__(self, retriever=None):
        self.retriever = retriever or Retriever()
        if openai:
            key = os.getenv("OPENAI_API_KEY")
            if key:
                openai.api_key = key

    def generate(self, query, top_k=3):
        # Retrieve relevant chunks
        chunks = self.retriever.query(query, top_k=top_k)
        context_text = "\n\n".join([c["text"] for c in chunks])
        prompt = f"Answer the question based on the context below:\n\n{context_text}\n\nQuestion: {query}\nAnswer:"

        # Use OpenAI if API key exists
        if openai and openai.api_key:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        else:
            # Fallback: return concatenated top chunks
            return context_text[:500] + "..."  # simple fallback

# Quick test
if __name__ == "__main__":
    g = Generator()
    print(g.generate("What is in the PDF?", top_k=2))
