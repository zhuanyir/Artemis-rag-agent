"""
embed.py — Embed chunks and build a FAISS index for the Artemis II RAG pipeline.

Run this script ONCE to generate data/index.faiss.
Re-run tomorrow after swapping in the real get_embedding() function.

Usage:
    python scripts/embed.py

Inputs:
    data/chunks.json        ← produced by scripts/chunk.py

Outputs:
    data/index.faiss        ← FAISS index (vector search)
    data/chunks.json        ← overwritten in-place (unchanged, confirmed valid)
"""

from __future__ import annotations
 
import json

import os

import time
 
import faiss

import numpy as np

from dotenv import load_dotenv
 
load_dotenv()
 
# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
 
EMBEDDING_DIM = 1536   # text-embedding-3-small output dimension

BATCH_SIZE = 50        # max texts per OpenAI embedding API call
 
 
# ── Embedding function ────────────────────────────────────────────────────────
 
def get_embeddings(texts: list[str]) -> list[list[float]]:

    from openai import OpenAI
 
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(

        model="text-embedding-3-small",

        input=texts,

    )

    return [item.embedding for item in response.data]
 
 
# ── Build richer text for embedding ───────────────────────────────────────────
 
def build_embedding_text(chunk: dict) -> str:

    """

    Build the text that will actually be embedded.
 
    Even if your chunk has no section/title, including source and page still helps

    retrieval distinguish document context better than embedding raw text alone.

    """

    source = chunk.get("source", "")

    page = chunk.get("page", "")

    text = chunk.get("text", "").strip()
 
    header_parts = []

    if source:

        header_parts.append(f"Source: {source}")

    if page != "":

        header_parts.append(f"Page: {page}")
 
    header = "\n".join(header_parts)
 
    if header:

        return f"{header}\n\n{text}"

    return text
 
 
# ── Embed in batches ──────────────────────────────────────────────────────────
 
def embed_chunks(chunks: list[dict]) -> list[list[float]]:

    """

    Embed all chunks in batches of BATCH_SIZE.

    Prints progress so you can track cost and time.

    """

    texts = [build_embedding_text(c) for c in chunks]

    all_embeddings: list[list[float]] = []
 
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
 
    for i in range(0, len(texts), BATCH_SIZE):

        batch = texts[i : i + BATCH_SIZE]

        batch_num = i // BATCH_SIZE + 1

        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
 
        embeddings = get_embeddings(batch)

        all_embeddings.extend(embeddings)
 
        # Small delay to reduce rate-limit risk

        time.sleep(0.1)
 
    return all_embeddings
 
 
# ── Build and save FAISS index ────────────────────────────────────────────────
 
def build_and_save_index(chunks: list[dict], embeddings: list[list[float]]) -> None:

    """

    Build a FAISS IndexFlatIP (cosine similarity via L2-normalized vectors)

    and save it to disk alongside the chunks metadata.

    """

    vectors = np.array(embeddings, dtype="float32")

    faiss.normalize_L2(vectors)  # normalize -> inner product approximates cosine similarity
 
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    index.add(vectors)
 
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"\n[embed] FAISS index saved -> {FAISS_INDEX_PATH}")

    print(f"[embed] Total vectors in index: {index.ntotal}")
 
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:

        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[embed] Chunks confirmed -> {CHUNKS_PATH}")
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main() -> None:

    if not os.path.exists(CHUNKS_PATH):

        raise FileNotFoundError(

            f"chunks.json not found at {CHUNKS_PATH}.\n"

            "Run scripts/chunk.py first."

        )
 
    with open(CHUNKS_PATH, encoding="utf-8") as f:

        chunks = json.load(f)
 
    print(f"[embed] Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    print(f"[embed] Embedding dimension: {EMBEDDING_DIM}")

    print(f"[embed] Batch size: {BATCH_SIZE}")

    print()
 
    print("[embed] Starting embedding...")

    embeddings = embed_chunks(chunks)

    print(f"[embed] Done. {len(embeddings)} embeddings generated.\n")
 
    build_and_save_index(chunks, embeddings)
 
    print("\n[embed] All done. You can now run app/app.py.")
 
 
if __name__ == "__main__":

    main()
 