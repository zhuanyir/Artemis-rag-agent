"""
retriever.py — Vector retrieval for the Artemis II RAG pipeline.

Uses the new chunk format from chunk_corpus.py:
  id, document_id, document_title, paragraph_index, text, size, sentence_count

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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR         = os.path.join(os.path.dirname(__file__), "..", "data")
CHUNKS_PATH      = os.path.join(DATA_DIR, "chunks.json")
CORPUS_PATH      = os.path.join(DATA_DIR, "corpus.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")

EMBEDDING_DIM = 1536

# ── Embedding cache ───────────────────────────────────────────────────────────
_embedding_cache: dict[str, list[float]] = {}


def get_embedding(text: str) -> list[float]:
    cache_key = text.strip().lower()
    if cache_key in _embedding_cache:
        print(f"[retriever] Cache hit for query: {text[:50]}...")
        return _embedding_cache[cache_key]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    embedding = response.data[0].embedding
    _embedding_cache[cache_key] = embedding
    print(f"[retriever] Cache miss — embedded query ({len(_embedding_cache)} cached).")
    return embedding


# ── Load index and chunks ─────────────────────────────────────────────────────

def load_index_and_chunks() -> tuple[faiss.Index, list[dict]]:
    """
    Load FAISS index and chunk metadata from disk.
    Enriches chunks with source/page from corpus.json using document_id.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}.\n"
            "Run scripts/embed.py first."
        )
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}.\n"
            "Run scripts/chunk_corpus.py first."
        )

    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    # Enrich chunks with source + page from corpus.json
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, encoding="utf-8") as f:
            corpus = json.load(f)

        for chunk in chunks:
            doc_id = chunk.get("document_id", -1)
            if 0 <= doc_id < len(corpus):
                chunk["source"] = corpus[doc_id].get("source", "unknown")
                chunk["page"]   = corpus[doc_id].get("page", -1)
            else:
                chunk["source"] = chunk.get("document_title", "unknown")
                chunk["page"]   = -1

        print(f"[retriever] Enriched chunks with source/page from corpus.json")
    else:
        for chunk in chunks:
            chunk.setdefault("source", chunk.get("document_title", "unknown"))
            chunk.setdefault("page", -1)

    # Normalise chunk_id field
    for chunk in chunks:
        chunk.setdefault("chunk_id", chunk.get("id", -1))

    print(f"[retriever] Loaded {index.ntotal} vectors and {len(chunks)} chunks.")
    return index, chunks


# ── Heuristic reranking ───────────────────────────────────────────────────────

def heuristic_rerank(query: str, results: list[dict]) -> list[dict]:
    positive_terms = [
        "mission requirements", "requirements", "objectives",
        "mission objectives", "flight readiness", "crew safety",
        "lunar", "flyby", "orion", "sls", "hardware",
        "operations", "deep space", "safe return", "payloads",
        "322.4", "98.27", "block 1 by the numbers",
        "commander", "pilot", "mission specialist",
        "wiseman", "glover", "koch", "hansen",
        "assigned to be", "crew member",
        "10 days",        # mission duration
        "10-day",         # mission duration alternate
        "about 10 days",  # exact phrase in corpus
    ]

    negative_terms = [
        "mission manager", "manager", "biography",
        "he began his career", "charged with helping",
        "in this role", "matt ramsey",
    ]

    reranked: list[dict] = []

    for item in results:
        text   = item.get("text", "").lower()
        # CHANGED: 'source' → 'document_title'
        doc_title = str(item.get("document_title", "")).lower()

        bonus = 0.0
        for term in positive_terms:
            if term in text:
                bonus += 0.05
        for term in negative_terms:
            if term in text:
                bonus -= 0.12
        # CHANGED: Check document_title instead of source
        if "press kit" in doc_title:
            bonus += 0.03
        if "reference guide" in doc_title:
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
    candidate_k: int = 10,
    use_rerank: bool = True,
    debug: bool = False,
) -> list[dict]:
    query_vec = np.array(get_embedding(query), dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, candidate_k)

    candidates: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        result = dict(chunks[idx])
        result["score"] = round(float(score), 4)
        candidates.append(result)

    if use_rerank:
        candidates = heuristic_rerank(query, candidates)

    results = candidates[:final_k]

    if debug:
        print("\n" + "=" * 80)
        print(f"[retrieve] Query: {query}")
        print(f"[retrieve] Candidates: {len(candidates)} → returning top {final_k}")
        print("=" * 80)
        for rank, item in enumerate(results, start=1):
            # CHANGED: 'chunk_id' → 'id', 'source' → 'document_title'
            print(f"\n  [{rank}] id={item.get('id')} | "
                  f"score={item.get('score')} | "
                  f"rerank_score={item.get('rerank_score', 'N/A')} | "
                  f"document={item.get('document_title')} p.{item.get('page')}")
            print(f"       {item.get('text', '')[:200]}")

    return results