"""
app.py — Gradio web UI for the Artemis II RAG chatbot.
Fully compatible with Gradio 6.12.0

Features:
- Chat interface with conversation history
- Confidence threshold — warns if retrieval score too low
- Source citations below every answer
- Analytics dashboard tab
- Export conversation as .txt
- Built-in Like/Dislike feedback (native Gradio 6.x feature)

Wires together retriever.py and generator.py into a chat interface.

Usage:
    python app/app.py

Then open: http://localhost:7860
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import os
import sys
from typing import Any

import gradio as gr

# Make sure app/ can import from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from retriever import load_index_and_chunks, retrieve
from generator import generate_answer, is_followup

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
FEEDBACK_LOG = ROOT / "data" / "feedback_log.json"
COST_FILE    = ROOT / "cost_tracker.json"

# ── Config ────────────────────────────────────────────────────────────────────
DEBUG_HISTORY        = False  # set True to debug history format in terminal
CONFIDENCE_THRESHOLD = 0.35

from retriever import load_index_and_chunks, retrieve
from generator import generate_answer

# ── Debug toggle ──────────────────────────────────────────────────────────────

DEBUG_HISTORY = True

# ── Load index and chunks once at startup ─────────────────────────────────────

# ── Load index and chunks once at startup ─────────────────────────────────────
print("[app] Loading FAISS index and chunks...")
index, chunks = load_index_and_chunks()
print("[app] Ready.\n")

# ── In-memory analytics ───────────────────────────────────────────────────────
analytics: dict[str, Any] = {
    "total_questions": 0,
    "low_confidence":  0,
    "refusals":        0,
    "avg_score":       0.0,
    "score_sum":       0.0,
    "questions":       [],
}


# ── Feedback logging ──────────────────────────────────────────────────────────

def log_feedback(data: gr.LikeData) -> None:
    """Called by Gradio's native like/dislike event."""
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    if FEEDBACK_LOG.exists():
        with open(FEEDBACK_LOG, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    rating = "👍" if data.liked else "👎"
    existing.append({
        "timestamp": datetime.now().isoformat(),
        "message":   str(data.value)[:300],
        "rating":    rating,
    })

    with open(FEEDBACK_LOG, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"[feedback] {rating} logged.")


# ── Core chat function ────────────────────────────────────────────────────────

def chat(message: str, history: list) -> str:
    """Called by gr.ChatInterface on every message."""
    if not message or not message.strip():
        return "Please enter a question."

    if DEBUG_HISTORY:
        print(f"\n[app] ─── Incoming message: {message}")
        print(f"[app] History length: {len(history) if history else 0}")
        print(f"[app] History type: {type(history)}")
        if history:
            print(f"[app] First turn type: {type(history[0])}")
            print(f"[app] First turn value: {history[0]}")
            print(f"[app] Last turn value: {history[-1]}")
        print()

    # 1. Retrieve
    retrieved = retrieve(
        query=message,
        index=index,
        chunks=chunks,
        final_k=5,
        candidate_k=10,
    )

    # 2. Confidence check — suppress for follow-ups (expected low scores)
    top_score      = retrieved[0].get("score", 0.0) if retrieved else 0.0
    low_confidence = (
        top_score < CONFIDENCE_THRESHOLD
        and len(retrieved) > 0
        and not is_followup(message, history)
    )
    if low_confidence:
        analytics["low_confidence"] += 1

    # 3. Generate
    answer = generate_answer(message, retrieved, history)

    # 4. Track refusals
    if "i don't have information" in answer.lower():
        analytics["refusals"] += 1

    # 5. Update analytics
    analytics["total_questions"] += 1
    analytics["score_sum"] += top_score
    analytics["avg_score"] = analytics["score_sum"] / analytics["total_questions"]
    analytics["questions"].append({
        "question":  message[:100],
        "score":     round(top_score, 4),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    if len(analytics["questions"]) > 50:
        analytics["questions"].pop(0)

    # 6. Low confidence warning
    if low_confidence:
        answer = (
            f"⚠️ **Low confidence** (retrieval score: {top_score:.2f}) "
            f"— the following answer may be inaccurate.\n\n{answer}"
        )

    # 7. Source citations
    if retrieved:
        seen, lines = set(), []
        for chunk in retrieved:
            key = f"{chunk['source']} — page {chunk['page']}"
            if key not in seen:
                seen.add(key)
                lines.append(
                    f"• {key}  (chunk_id: {chunk.get('chunk_id', 'N/A')}, "
                    f"score: {chunk.get('score', 'N/A')})"
                )
        answer += "\n\n---\n**Sources retrieved:**\n" + "\n".join(lines)

    return answer


# ── Export ────────────────────────────────────────────────────────────────────

def export_conversation(history: list) -> str | None:
    if not history:
        return None

    export_dir  = ROOT / "data" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = export_dir / f"conversation_{timestamp}.txt"

    lines = [
        "Artemis II Knowledge Assistant — Conversation Export",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60, "",
    ]

    for turn in history:
        if isinstance(turn, dict):
            role  = turn.get("role", "")
            label = "USER" if role == "user" else "ASSISTANT"
            lines += [f"[{label}]", str(turn.get("content", "")), ""]
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            if turn[0]:
                lines += ["[USER]",      str(turn[0]), ""]
            if turn[1]:
                lines += ["[ASSISTANT]", str(turn[1]), ""]

    with open(export_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return str(export_path)


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_analytics_text() -> str:
    cost_total, cost_calls = 0.0, 0
    if COST_FILE.exists():
        with open(COST_FILE, encoding="utf-8") as f:
            d = json.load(f)
        cost_total = d.get("total", 0.0)
        cost_calls = d.get("calls", 0)

    pos, neg = 0, 0
    if FEEDBACK_LOG.exists():
        with open(FEEDBACK_LOG, encoding="utf-8") as f:
            fb = json.load(f)
        pos = sum(1 for x in fb if x.get("rating") == "👍")
        neg = sum(1 for x in fb if x.get("rating") == "👎")

    recent = analytics["questions"][-10:]
    recent_lines = "\n".join(
        f"  [{q['timestamp']}] score={q['score']}  {q['question']}"
        for q in reversed(recent)
    ) or "  No questions yet."

    return f"""## 📊 Session Analytics

| Metric | Value |
|--------|-------|
| Total questions | {analytics['total_questions']} |
| Low confidence answers | {analytics['low_confidence']} |
| Out-of-scope refusals | {analytics['refusals']} |
| Average retrieval score | {analytics['avg_score']:.3f} |
| 👍 Positive feedback | {pos} |
| 👎 Negative feedback | {neg} |
| API calls | {cost_calls} |
| Total API cost | ${cost_total:.4f} / $5.00 |

## 🕐 Recent Questions
{recent_lines}
"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Artemis II RAG") as demo:

        gr.Markdown("""
        # 🚀 Artemis II Knowledge Assistant
        Ask any question about the Artemis II mission, SLS rocket, or Orion spacecraft.
        Answers are drawn exclusively from official NASA documents.
        """)

        with gr.Tabs():

            # ── Tab 1: Chat ───────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):

                chatbot = gr.Chatbot(
                    height=500,
                    label="Artemis II Assistant",
                    feedback_options=("Like", "Dislike"),
                    placeholder="Ask me anything about Artemis II...",
                    layout="bubble",
                )

                chat_interface = gr.ChatInterface(
                    fn=chat,
                    chatbot=chatbot,
                    textbox=gr.Textbox(
                        placeholder="e.g. Who are the Artemis II crew members?",
                        container=False,
                    ),
                    examples=[
                        "Who are the four crew members of Artemis II?",
                        "What is the total height of the SLS Block 1 rocket?",
                        "How long is the Artemis II mission expected to last?",
                        "What percentage of thrust do the solid rocket boosters provide?",
                        "Did the Artemis II crew successfully return to Earth?",
                        "Tell me about the requirements for the mission.",
                    ],
                )

                # Native like/dislike
                chatbot.like(log_feedback, None, None)

                # Export button
                export_btn  = gr.Button("📥 Export conversation")
                export_file = gr.File(label="Download conversation", visible=False)

                def on_export(chatbot_history):
                    path = export_conversation(chatbot_history)
                    return gr.File(value=path, visible=True) if path else gr.File(visible=False)

                export_btn.click(on_export, inputs=[chatbot], outputs=[export_file])

            # ── Tab 2: Analytics ──────────────────────────────────────────────
            with gr.Tab("📊 Analytics"):

                analytics_display = gr.Markdown(get_analytics_text())
                refresh_btn       = gr.Button("🔄 Refresh")
                refresh_btn.click(
                    fn=lambda: get_analytics_text(),
                    outputs=analytics_display,
                )
    # 2. Generate answer — pass history so follow-ups work
    answer = generate_answer(message, retrieved, history)

    # 3. Append source citations block below the answer
    if retrieved:
        sources_lines = []
        seen = set()

        for chunk in retrieved:
            # CHANGED: 'source' → 'document_title'
            # CHANGED: 'chunk_id' → 'id'
            key = f"{chunk['document_title']} — page {chunk['page']}"

            if key not in seen:
                seen.add(key)
                sources_lines.append(
                    f"• {key}  (id: {chunk.get('id', 'N/A')}, "
                    f"score: {chunk.get('score', 'N/A')})"
                )

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

        gr.ChatInterface(
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
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )