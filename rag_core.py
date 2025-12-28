import os
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline

DATA_PATH = "rag_docs.txt"
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"

def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

with open(DATA_PATH, encoding="utf-8") as f:
    text = f.read()

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
else:
    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1
)

def retrieve(query, k=3):
    q_emb = embedding_model.encode([query]).astype("float32")
    _, idx = index.search(q_emb, k)
    return [chunks[i] for i in idx[0]]

def answer_query(query):
    docs = retrieve(query)
    context = "\n".join(docs)
    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """
    answer = generator(prompt, max_length=200)[0]["generated_text"]
    return answer, docs
