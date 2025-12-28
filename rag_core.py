import os
import pickle
import faiss
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline

DATA_PATH = "data/rag_docs.txt"
INDEX_PATH = "artifacts/faiss_index.bin"
CHUNKS_PATH = "artifacts/chunks.pkl"

# ---------- CHUNKING ----------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    return splitter.split_text(text)

# ---------- BUILD INDEX ----------
def build_index():
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    embed_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = embed_model.encode(chunks).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs("artifacts", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return embed_model, index, chunks

# ---------- LOAD INDEX ----------
def load_index():
    embed_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        return build_index()

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return embed_model, index, chunks

# ---------- RETRIEVE ----------
def retrieve(query, model, index, chunks, k=3):
    q_emb = model.encode([query]).astype("float32")
    _, idx = index.search(q_emb, k)
    return [chunks[i] for i in idx[0]]

# ---------- GENERATOR ----------
def load_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1  # CPU only
    )

def answer_query(query):
    embed_model, index, chunks = load_index()
    generator = load_generator()

    docs = retrieve(query, embed_model, index, chunks)
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


# ---------- CLI TEST ----------
if __name__ == "__main__":
    print("Building / Loading RAG system...")
    embed_model, index, chunks = load_index()
    print("RAG system ready ✅")
