"""
app.py — Gradio web UI for the Artemis II RAG chatbot.

Wires together retriever.py and generator.py into a chat interface.

Usage:
    python app/app.py

Then open: http://localhost:7860
"""

from __future__ import annotations

import os
import sys

import gradio as gr

# Make sure app/ can import from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from retriever import load_index_and_chunks, retrieve
from generator import generate_answer

# ── Load index and chunks once at startup ─────────────────────────────────────
# These are kept in memory for the lifetime of the app.
# retrieve() reads from them on every query — no disk I/O per request.

print("[app] Loading FAISS index and chunks…")
index, chunks = load_index_and_chunks()
print("[app] Ready.\n")


# ── Core chat function ────────────────────────────────────────────────────────

def chat(message: str, history: list) -> str:
    """
    Called by Gradio on every user message.

    Args:
        message: The user's question.
        history: Previous turns (managed by Gradio, passed for context).

    Returns:
        The answer string shown in the chat bubble.
    """
    if not message.strip():
        return "Please enter a question."

    # 1. Retrieve relevant chunks
    retrieved = retrieve(message, index, chunks, k=5)

    # 2. Generate answer with citations
    answer = generate_answer(message, retrieved)

    # 3. Append source citations block below the answer
    if retrieved:
        sources_lines = []
        seen = set()
        for chunk in retrieved:
            key = f"{chunk['source']} — page {chunk['page']}"
            if key not in seen:
                seen.add(key)
                sources_lines.append(f"• {key}  (score: {chunk['score']})")

        sources_block = "\n\n---\n**Sources retrieved:**\n" + "\n".join(sources_lines)
        answer += sources_block

    return answer


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Artemis II RAG") as demo:

        gr.Markdown(
            """
            # 🚀 Artemis II Knowledge Assistant
            Ask any question about the Artemis II mission, SLS rocket, or Orion spacecraft.
            Answers are drawn exclusively from the official NASA documents in our corpus.
            """
        )

        chatbot = gr.ChatInterface(
            fn=chat,
            chatbot=gr.Chatbot(height=500),
            textbox=gr.Textbox(
                placeholder="e.g. Who are the Artemis II crew members?",
                container=False,
            ),
            examples=[
                "Who are the four crew members of Artemis II?",
                "What is the height of the SLS rocket?",
                "How long does the Artemis II mission last?",
                "What percentage of thrust do the solid rocket boosters provide?",
                "What happens if the answer is not in the documents?",
            ],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",   # accessible on local network, not just localhost
        server_port=7860,
        show_error=True,
    )