"""
retriever.py — Vector retrieval for the Artemis II RAG pipeline.

Follows the code reference from the crash course exactly.
Swap get_embedding() tomorrow when the API key is available.

Place this file at: app/retriever.py
"""

from __future__ import annotations

import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR         = os.path.join(os.path.dirname(__file__), "..", "data")
CHUNKS_PATH      = os.path.join(DATA_DIR, "chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")

EMBEDDING_DIM = 1536  # text-embedding-3-small


# ── Embedding function ────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


# ── Load index and chunks ─────────────────────────────────────────────────────

def load_index_and_chunks() -> tuple[faiss.Index, list[dict]]:
    """
    Load FAISS index and chunk metadata from disk.
    Run scripts/embed.py first to generate these files.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}.\n"
            "Run scripts/embed.py first."
        )
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}.\n"
            "Run scripts/chunk.py first."
        )

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[retriever] Loaded {index.ntotal} vectors and {len(chunks)} chunks.")
    return index, chunks


# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    k: int = 5,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query:  The user's question.
        index:  The loaded FAISS index.
        chunks: The list of chunk dicts (same order as index vectors).
        k:      Number of results to return.

    Returns:
        List of chunk dicts, each with an added "score" key.
    """
    query_vec = np.array(get_embedding(query), dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        result = dict(chunks[idx])
        result["score"] = round(float(score), 4)
        results.append(result)

    return results
