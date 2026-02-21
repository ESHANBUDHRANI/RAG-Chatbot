import os
import logging
from typing import List
import tempfile
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq

logging.basicConfig(level=logging.INFO)

load_dotenv()
GROQ_API_KEY = os.getenv("GROK_API_KEY")
logging.info("Groq API Key loaded: %s", bool(GROQ_API_KEY))

app = FastAPI(title="Simple RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- In-memory stores ----
texts: List[str] = []
embeddings = None

# ---- Models ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Utils ----
def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    content = []
    for page in reader.pages:
        content.append(page.extract_text() or "")
    return "\n".join(content)

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

def build_index(chunks: List[str]):
    global embeddings, texts
    texts = chunks
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    logging.info("Index built with %d chunks", len(chunks))

def search(query: str, k: int = 3) -> List[str]:
    if embeddings is None or len(texts) == 0:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_k = sims.argsort()[-k:][::-1]
    return [texts[i] for i in top_k]

def generate_answer(query: str, contexts: List[str]) -> str:
    if not GROQ_API_KEY:
        return "Error: GROK_API_KEY is not set in your .env file."

    context_block = "\n".join(contexts)
    client = Groq(api_key=GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Answer using provided context."},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.exception("Groq API error")
        return f"Groq API Error: {e}"

# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
  <title>Simple RAG Chatbot</title>
  <style>
    body { font-family: Arial; max-width: 800px; margin: 40px auto; }
    textarea, input[type=text] { width: 100%; padding: 10px; margin: 8px 0; }
    button { padding: 10px 16px; }
    #chat { border: 1px solid #ddd; padding: 10px; min-height: 120px; }
  </style>
</head>
<body>
  <h2>Upload PDFs</h2>
  <form id="uploadForm">
    <input type="file" id="files" name="files" multiple />
    <button type="submit">Upload & Index</button>
  </form>

  <h2>Ask</h2>
  <div id="chat"></div>
  <input type="text" id="q" placeholder="Ask something..." />
  <button onclick="ask()">Send</button>

<script>
document.getElementById('uploadForm').onsubmit = async (e) => {
  e.preventDefault();
  const files = document.getElementById('files').files;
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  const res = await fetch('/upload', { method: 'POST', body: fd });
  alert(await res.text());
};

async function ask() {
  const q = document.getElementById('q').value;
  if (!q.trim()) return;
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ query: q })
  });
  let txt = await res.text();
  txt = txt.replace(/^"|"$/g, '');
  txt = txt.replace(/\\n/g, '<br>');
  const chat = document.getElementById('chat');
  chat.innerHTML += `<p><b>You:</b> ${q}</p><p><b>Bot:</b> ${txt}</p><hr>`;
  document.getElementById('q').value = '';
}
</script>
</body>
</html>
    """

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    try:
        all_text = []

        for f in files:
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, f.filename.replace(" ", "_"))
            tmp_path = os.path.normpath(tmp_path)

            contents = await f.read()
            with open(tmp_path, "wb") as tmp:
                tmp.write(contents)

            try:
                text = read_pdf(tmp_path)
                all_text.append(text)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        full_text = "\n".join(all_text)
        chunks = chunk_text(full_text)

        if not chunks:
            return "No text extracted."

        build_index(chunks)
        return f"Indexed {len(chunks)} chunks."

    except Exception as e:
        logging.exception("Upload error")
        return f"Error: {e}"

@app.post("/ask")
def ask(query: str = Form(...)):
    try:
        ctx = search(query, k=3)
        if not ctx:
            return "No documents indexed yet. Upload PDFs first."
        ans = generate_answer(query, ctx)
        return ans
    except Exception as e:
        logging.exception("Ask error")
        return f"Error: {e}"


