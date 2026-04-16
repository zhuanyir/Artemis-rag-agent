"""
generator.py — Answer generation for the Artemis II RAG pipeline.

Takes retrieved chunks and a user question, builds a prompt,
and returns a synthesized answer with source citations.

Supports smart conversation history — only injects previous turns
when the current question is detected as a follow-up.

Supports yes/no detection — answers ONLY "Yes" or "No" for
yes/no questions without any explanation or citations.

Usage:
    from generator import generate_answer
    answer = generate_answer(query, retrieved_chunks, history)
    print(answer)
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialise client ONCE at module level
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert assistant for the Artemis II space mission.
Answer questions using ONLY the provided context from official NASA documents.

Your goal is not to copy the text, but to synthesize and explain it clearly.

Rules:
- Always base your answer strictly on the provided context. Do not use outside knowledge.
- Identify the parts of the context that directly answer the question. Ignore irrelevant details (e.g. biographies or role descriptions unless explicitly required).
- Merge overlapping or similar points instead of repeating them.
- Summarize information in your own words rather than copying sentences verbatim.
- When the answer is a list (e.g. mission requirements), group related items into logical categories when possible.
- Start with a short summary (1–2 sentences), then provide a structured explanation or grouped bullet points.
- If multiple chunks contain similar information, consolidate them into fewer, clearer points.
- If the context is incomplete or only partially answers the question, briefly acknowledge that.

Special rule for yes/no questions:
- If the user's question can be answered with "Yes" or "No", respond with ONLY "Yes" or "No".
- Do NOT add any explanation, details, or citations in this case.
- Only answer "Yes" if the context clearly supports it. Otherwise answer "No".

Citations:
- Always cite sources inline using the format: (Source: filename, page X)
- If multiple sources support a point, cite them together.

Fallback:
- If the context does NOT contain enough information to answer, respond with exactly:
  "I don't have information about this in the Artemis II documents."

Style:
- Be concise but complete.
- Prefer clarity and synthesis over exhaustiveness.
"""


# ── Cost tracker ──────────────────────────────────────────────────────────────

COST_FILE = os.path.join(os.path.dirname(__file__), "..", "cost_tracker.json")


def track_cost(response) -> None:
    """Track cumulative API cost to stay within the $5 team budget."""
    usage = response.usage
    cost = (
        usage.prompt_tokens * 0.15 +
        usage.completion_tokens * 0.60
    ) / 1_000_000  # gpt-4o-mini pricing

    if os.path.exists(COST_FILE):
        with open(COST_FILE, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"total": 0.0, "calls": 0}

    data["total"] += cost
    data["calls"] = data.get("calls", 0) + 1

    with open(COST_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(
        f"[cost] This call: ${cost:.6f} | "
        f"Total: ${data['total']:.4f} / $5.00 ({data['calls']} calls)"
    )


# ── Follow-up detection ───────────────────────────────────────────────────────

FOLLOW_UP_INDICATORS = [
    "he", "she", "they", "it", "this", "that", "these", "those",
    "the same", "also", "what about", "and what", "but what",
    "how about", "what else", "tell me more", "explain more",
    "more about", "can you elaborate", "why", "when did",
    "who else", "where is", "what is his", "what is her",
    "what is their", "what does he", "what does she",
]


def is_followup(query: str, history: list[Any] | None) -> bool:
    """Returns True if the query looks like a follow-up to a previous turn."""
    if not history:
        return False

    query_lower = query.lower().strip()
    words = query_lower.split()

    if not words:
        return False

    if len(words) <= 4:
        print("[generator] Follow-up detected: short query.")
        return True

    first_word = words[0]
    if first_word in FOLLOW_UP_INDICATORS:
        print(f"[generator] Follow-up detected: starts with '{first_word}'.")
        return True

    for indicator in FOLLOW_UP_INDICATORS:
        if query_lower.startswith(indicator + " "):
            print(f"[generator] Follow-up detected: phrase '{indicator}'.")
            return True

    print("[generator] Standalone question — no history injected.")
    return False


# ── Yes/No detection ──────────────────────────────────────────────────────────

def is_yes_no_question(query: str) -> bool:
    """Return True if the question is likely a yes/no question."""
    q = query.lower().strip()
    yes_no_starts = (
        "is ", "are ", "was ", "were ",
        "does ", "do ", "did ",
        "can ", "could ",
        "will ", "would ",
        "has ", "have ", "had ",
    )
    return q.startswith(yes_no_starts)


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the prompt."""
    return "\n\n".join(
        f"[Source: {chunk['document_title']} p.{chunk['page']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )


# ── History conversion ────────────────────────────────────────────────────────

def clean_assistant_message(text: str) -> str:
    """Remove the appended sources block before reinjecting assistant replies."""
    if not isinstance(text, str):
        return ""
    return text.split("\n\n---\n**Sources retrieved:**")[0].strip()


def _extract_text(turn_content) -> str:
    """
    Extract plain text from a Gradio turn which may be:
    - A plain string
    - A list of {"text": "...", "type": "text"} dicts (Gradio 6.12 format)
    """
    if isinstance(turn_content, str):
        return turn_content
    if isinstance(turn_content, list):
        parts = []
        for item in turn_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts).strip()
    return ""


def extract_history_messages(
    history: list[Any] | None,
    max_turns: int = 3,
) -> list[dict[str, str]]:
    """
    Convert Gradio history into OpenAI chat messages.

    Supports:
    1. Gradio 6.x format: [[user_content, assistant_content], ...]
       where each content is either a string or list of {"text":..., "type":...} dicts
    2. Messages format: [{"role": "user", "content": ...}, ...]
    3. Old tuple format: [[str, str], ...]
    """
    if not history:
        return []

    history_messages: list[dict[str, str]] = []

    # Case 1: role/content dict format
    if all(isinstance(item, dict) and "role" in item and "content" in item for item in history):
        recent = history[-(max_turns * 2):]
        for msg in recent:
            role    = msg.get("role")
            content = _extract_text(msg.get("content", ""))
            if role not in {"user", "assistant"} or not content.strip():
                continue
            if role == "assistant":
                content = clean_assistant_message(content)
            history_messages.append({"role": role, "content": content})
        return history_messages

    # Case 2 & 3: list of [user_turn, assistant_turn] pairs
    for turn in history[-max_turns:]:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_raw, bot_raw = turn

            user_msg = _extract_text(user_raw)
            bot_msg  = _extract_text(bot_raw)

            if user_msg.strip():
                history_messages.append({"role": "user", "content": user_msg})
            if bot_msg.strip():
                history_messages.append({
                    "role":    "assistant",
                    "content": clean_assistant_message(bot_msg),
                })

    return history_messages


# ── Generate answer ───────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    history: list[Any] | None = None,
) -> str:
    """
    Generate a synthesized answer from retrieved chunks using gpt-4o-mini.

    Injects conversation history only when the query is detected as a follow-up.
    For yes/no questions, injects a strict extra instruction.
    """
    if not retrieved_chunks:
        return "I don't have information about this in the Artemis II documents."

    context = build_context(retrieved_chunks)
    history = history or []

    # ── History injection ─────────────────────────────────────────────────────
    history_messages: list[dict[str, str]] = []
    if is_followup(query, history):
        history_messages = extract_history_messages(history, max_turns=3)
        print(f"[generator] Injecting {len(history_messages)} history messages.")
    else:
        print("[generator] No history injected.")

    # ── Yes/No instruction ────────────────────────────────────────────────────
    extra_instruction = ""
    if is_yes_no_question(query):
        print("[generator] Yes/No question detected.")
        extra_instruction = (
            "Answer ONLY 'Yes' or 'No'. "
            "Do not add any explanation, details, or citations. "
            "Answer 'Yes' only if the provided context clearly supports it; "
            "otherwise answer 'No'.\n\n"
        )

    # ── Build prompt ──────────────────────────────────────────────────────────
    user_prompt = f"""{extra_instruction}Context:
{context}

Question: {query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *history_messages,
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=500,
    )

    track_cost(response)

    content = response.choices[0].message.content
    return content.strip() if content else "I don't have information about this in the Artemis II documents."