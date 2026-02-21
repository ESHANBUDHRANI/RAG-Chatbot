# Simple RAG Chatbot (Python)

## What this does
- Upload PDFs
- Extract text
- Chunk text
- Create embeddings
- Store in FAISS
- Retrieve relevant chunks
- Generate answer with an LLM

## Setup
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://127.0.0.1:8000

## RAG Pipeline (in simple words)
1. Read PDFs and get text
2. Break text into small chunks
3. Convert chunks to embeddings (numbers)
4. Store embeddings in a vector database (FAISS)
5. When a user asks a question, find the most similar chunks
6. Send the chunks + question to the language model
7. Model generates a context-aware answer

## Notes
- This is a minimal educational example.
- Everything is stored in memory.