"""
retriever.py — Vector retrieval for the Artemis II RAG pipeline.

Place this file at: app/retriever.py
"""

from __future__ import annotations

import json
import os

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialise client ONCE at module level — not inside the function
# Avoids re-creating the client on every query (real speedup)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR         = os.path.join(os.path.dirname(__file__), "..", "data")
CHUNKS_PATH      = os.path.join(DATA_DIR, "chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")

EMBEDDING_DIM = 1536  # text-embedding-3-small


# ── Embedding function ────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Single embedding call — used for query embedding at search time."""
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


# ── Heuristic reranking ───────────────────────────────────────────────────────

def heuristic_rerank(query: str, results: list[dict]) -> list[dict]:
    """
    Adjust ranking using lightweight lexical heuristics.
    Runs in microseconds — no API call needed.
    """
    positive_terms = [
        "mission requirements", "requirements", "objectives",
        "mission objectives", "flight readiness", "crew safety",
        "lunar", "flyby", "orion", "sls", "hardware",
        "operations", "deep space", "safe return", "payloads",
    ]

    negative_terms = [
        "mission manager", "manager", "biography",
        "he began his career", "charged with helping",
        "in this role", "matt ramsey",
    ]

    reranked: list[dict] = []

    for item in results:
        text   = item.get("text", "").lower()
        source = str(item.get("source", "")).lower()

        bonus = 0.0
        for term in positive_terms:
            if term in text:
                bonus += 0.05
        for term in negative_terms:
            if term in text:
                bonus -= 0.12
        if "press kit" in source:
            bonus += 0.03
        if "reference guide" in source:
            bonus += 0.03

        new_item = dict(item)
        new_item["base_score"]   = item["score"]
        new_item["rerank_score"] = round(item["score"] + bonus, 4)
        reranked.append(new_item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    final_k: int = 5,
    candidate_k: int = 10,   # reduced from 20 → faster reranking
    use_rerank: bool = True,
    debug: bool = False,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Speed changes vs previous version:
    - Removed query rewriting (saved one full OpenAI API call per query)
    - Reduced candidate_k from 20 → 10 (fewer chunks to rerank)
    - OpenAI client initialised at module level (not per call)

    Args:
        query:       User question.
        index:       Loaded FAISS index.
        chunks:      List of chunk dicts.
        final_k:     Number of final chunks to return.
        candidate_k: Number of FAISS candidates before reranking.
        use_rerank:  Whether to apply heuristic reranking.
        debug:       Print retrieval diagnostics.

    Returns:
        List of chunk dicts with score fields added.
    """
    # Embed the query — this is the only API call in retrieval
    query_vec = np.array(get_embedding(query), dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query_vec)

    # FAISS search
    scores, indices = index.search(query_vec, candidate_k)

    candidates: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        result = dict(chunks[idx])
        result["score"] = round(float(score), 4)
        candidates.append(result)

    # Optional heuristic reranking (no API call — pure Python)
    if use_rerank:
        candidates = heuristic_rerank(query, candidates)

    results = candidates[:final_k]

    if debug:
        print("\n" + "=" * 80)
        print(f"[retrieve] Query: {query}")
        print(f"[retrieve] Candidates: {len(candidates)} → returning top {final_k}")
        print("=" * 80)
        for rank, item in enumerate(results, start=1):
            print(f"\n  [{rank}] chunk_id={item.get('chunk_id')} | "
                  f"score={item.get('score')} | "
                  f"rerank_score={item.get('rerank_score', 'N/A')} | "
                  f"source={item.get('source')} p.{item.get('page')}")
            print(f"       {item.get('text', '')[:200]}")

    return results