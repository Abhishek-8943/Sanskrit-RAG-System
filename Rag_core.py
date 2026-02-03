import os
import pickle
import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DATA_PATH = "rag_docs.txt"
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"


def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


def generate_answer(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def build_index():
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks


def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    return build_index()


def answer_query(query):
    index, chunks = load_index()

    q_emb = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    _, idx = index.search(q_emb, 3)

    docs = [chunks[i] for i in idx[0]]
    context = "\n".join(docs)

    prompt = f"""
You are a medical assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "Answer not found in the document."

Context:
{context}

Question:
{query}
"""

    answer = generate_answer(prompt)
    return answer, docs
