"""
generator.py — Answer generation for the Artemis II RAG pipeline.

Takes retrieved chunks and a user question, builds a prompt,
and returns an answer with source citations.

Usage:
    from generator import generate_answer

    answer = generate_answer(query, retrieved_chunks)
    print(answer)
"""

from __future__ import annotations

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for the Artemis II space mission.
Answer the question based ONLY on the provided context.

Rules:
- If the context contains the answer, provide it clearly and concisely.
- Always cite your sources using the format: (Source: filename, page X)
- If the context does NOT contain enough information to answer, respond with exactly:
  "I don't have information about this in the Artemis II documents."
- Do not make up information. Do not use knowledge outside the provided context.
- If multiple sources support the answer, cite all of them.
"""


# ── Cost tracker ──────────────────────────────────────────────────────────────

COST_FILE = os.path.join(os.path.dirname(__file__), "..", "cost_tracker.json")


def track_cost(response) -> None:
    """
    Track cumulative API cost to stay within the $5 team budget.
    Prints cost per call and running total.
    """
    usage = response.usage
    cost = (
        usage.prompt_tokens * 0.15 +
        usage.completion_tokens * 0.60
    ) / 1_000_000   # gpt-4o-mini pricing

    if os.path.exists(COST_FILE):
        with open(COST_FILE, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"total": 0.0, "calls": 0}

    data["total"] += cost
    data["calls"]  = data.get("calls", 0) + 1

    with open(COST_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[cost] This call: ${cost:.6f} | Total: ${data['total']:.4f} / $5.00 ({data['calls']} calls)")


# ── Build context string ──────────────────────────────────────────────────────

def build_context(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the prompt.
    Matches the reference code format exactly.
    """
    return "\n\n".join([
        f"[Source: {chunk['source']} p.{chunk['page']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])


# ── Generate answer ───────────────────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context = build_context(retrieved_chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0,
        max_tokens=400,
    )
    track_cost(response)
    return response.choices[0].message.content
