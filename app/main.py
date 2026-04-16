"""
app/main.py — Human-in-the-Loop (HITL) interactive interface.

PURPOSE:
    This is the entry point for the full Agentic AI system.
    It implements "Human-in-the-Loop" (HITL) — a pattern where the AI
    does work but pauses to ask the human for approval before taking
    irreversible actions (like saving to the database or publishing a report).

WHY HUMAN-IN-THE-LOOP?
    From the slides: "Agents boost productivity only when reviewing their work
    is faster than doing the work yourself."
    The agent can make mistakes. HITL means:
      1. Agent does the research (fast)
      2. Human reviews the draft (quick check)
      3. Human says APPROVE / REJECT / REFINE
      4. Agent takes the approved action

    This is exactly the hospital/law firm pattern — the AI drafts, the expert
    approves. You get speed + accuracy + accountability.

HITL FLOW (from slides p.23):
    User asks question
        → Agent 1 (internal research)
        → Agent 2 (external fact-check via MCP)
        → Agent 3 drafts final answer
        → ⏸ HUMAN REVIEWS DRAFT
        → If APPROVE:  publish report + show answer
        → If REJECT:   discard, ask for a new question
        → If REFINE:   human types feedback → agents rerun with feedback
        → If ADD:      add new info to corpus via MCP (Step 3)

AVAILABLE COMMANDS AT THE PROMPT:
    /agentic  <question>   — run full 3-agent pipeline
    /rag      <question>   — run plain RAG (no agents, faster + cheaper)
    /add      <text>       — add new info to corpus directly (Step 3)
    /pdf      <path>       — load a PDF into corpus (Step 4)
    /cost                  — show current API cost
    /help                  — show all commands
    /quit                  — exit

HOW TO RUN:
    python app/main.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Add app/ to path ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from agents import run_agentic_pipeline, _call_mcp_tool
from retriever import load_index_and_chunks, retrieve
from generator import generate_answer

# ── Cost file ─────────────────────────────────────────────────────────────────
COST_FILE = Path(__file__).parent.parent / "cost_tracker.json"


def show_cost() -> None:
    """Print current API spend."""
    if COST_FILE.exists():
        with open(COST_FILE) as f:
            data = json.load(f)
        total = data.get("total", 0)
        calls = data.get("calls", 0)
        remaining = max(0, 5.0 - total)
        bar_filled = int((total / 5.0) * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"\n💰 Budget: [{bar}] ${total:.4f} / $5.00 "
              f"(${remaining:.4f} left, {calls} calls)")
    else:
        print("\n💰 No API calls made yet.")


def show_help() -> None:
    print("""
┌─────────────────────────────────────────────────────────────┐
│           Artemis II Agentic AI — Commands                  │
├─────────────────────────────────────────────────────────────┤
│  /agentic <question>   Full 3-agent pipeline (best quality) │
│  /rag     <question>   Plain RAG (faster, cheaper)          │
│  /add     <text>       Add text to knowledge base (Step 3)  │
│  /pdf     <filepath>   Load PDF into knowledge base (Step 4)│
│  /cost                 Show API spend vs $5 budget          │
│  /help                 Show this menu                       │
│  /quit                 Exit                                 │
│                                                             │
│  Or just type a question for plain RAG (default)           │
└─────────────────────────────────────────────────────────────┘
""")


def human_review_loop(query: str, draft: str) -> str:
    """
    HUMAN-IN-THE-LOOP review step.

    Shows the agent's draft answer and asks the human to:
      APPROVE  → use the draft as-is, save the report
      REJECT   → discard the draft
      REFINE   → provide feedback, re-run agents with that feedback

    WHY THIS MATTERS (from the slides):
        "Human in the loop workflow — the agent chains multiple steps to help
        you resolve a problem, getting your feedback at each step."

    Returns:
        The final approved answer (possibly refined).
    """
    print("\n" + "━" * 60)
    print("📋  AGENT DRAFT — Please review:")
    print("━" * 60)
    print(draft)
    print("━" * 60)

    while True:
        choice = input(
            "\n[HITL] Action: [A]pprove / [R]eject / [F]eedback? "
        ).strip().upper()

        if choice in ("A", "APPROVE", ""):
            print("[HITL] ✅ Approved. Answer will be shown to user.")
            return draft

        elif choice in ("R", "REJECT"):
            print("[HITL] ❌ Rejected. Discarding this answer.")
            return "Answer was rejected by the reviewer. Please ask a different question."

        elif choice in ("F", "FEEDBACK"):
            feedback = input("[HITL] Your feedback (what to improve): ").strip()
            if not feedback:
                print("[HITL] No feedback given. Try again.")
                continue

            print("[HITL] 🔄 Re-running agents with your feedback...")
            refined_query = f"{query}\n\nREVIEWER FEEDBACK: {feedback}"
            result = run_agentic_pipeline(
                refined_query,
                save_report=False,   # don't save mid-loop reports
                check_external=True,
            )
            draft = result["final_answer"]

            print("\n" + "━" * 60)
            print("📋  REFINED DRAFT:")
            print("━" * 60)
            print(draft)
            print("━" * 60)
            # Loop again for another review

        else:
            print("[HITL] Please type A (approve), R (reject), or F (feedback).")


def handle_add_command(text: str) -> None:
    """
    STEP 3: Add new text to the corpus via MCP.

    WHY HITL HERE?
        Writing to the database is irreversible (well, you'd have to manually
        edit corpus.json). We show the user what will be added before committing.
    """
    print(f"\n[HITL] You're about to add this to the corpus:")
    print(f"  '{text[:200]}{'...' if len(text) > 200 else ''}'")

    confirm = input("[HITL] Confirm add? [Y]es / [N]o: ").strip().upper()
    if confirm in ("Y", "YES"):
        result = _call_mcp_tool("add_to_database", text=text, source="user_added")
        print(f"[MCP] {result}")
    else:
        print("[HITL] Add cancelled.")


def handle_pdf_command(pdf_path: str) -> None:
    """STEP 4: Load a PDF into corpus via MCP."""
    pdf_path = pdf_path.strip()
    if not pdf_path:
        print("Usage: /pdf <path/to/file.pdf>")
        return

    print(f"\n[HITL] Loading PDF: {pdf_path}")
    confirm = input("[HITL] Confirm load PDF to corpus? [Y]es / [N]o: ").strip().upper()
    if confirm in ("Y", "YES"):
        result = _call_mcp_tool("load_pdf_to_database", pdf_path=pdf_path)
        print(f"[MCP] {result}")
    else:
        print("[HITL] PDF load cancelled.")


# ── Plain RAG (existing pipeline, faster + cheaper) ───────────────────────────
print("[main] Loading FAISS index for plain RAG...")
try:
    _rag_index, _rag_chunks = load_index_and_chunks()
    print(f"[main] Ready. {_rag_index.ntotal} vectors loaded.")
except FileNotFoundError as e:
    print(f"[main] WARNING: {e}")
    _rag_index, _rag_chunks = None, []


def handle_rag_query(query: str) -> str:
    """Run the existing plain RAG pipeline (retriever + generator, no agents)."""
    if _rag_index is None:
        return "FAISS index not loaded. Run scripts/embed.py first."

    chunks = retrieve(query=query, index=_rag_index, chunks=_rag_chunks,
                      final_k=5, candidate_k=10)
    return generate_answer(query, chunks, history=None)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN INTERACTIVE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Main interactive loop with Human-in-the-Loop support.

    STEP 3 (HITL) is implemented here:
        Every agentic answer goes through human_review_loop() before
        being finalized. The human can approve, reject, or refine.
    """
    print("\n" + "═" * 60)
    print("  🚀  Artemis II — Agentic AI Assistant")
    print("  Powered by: RAG + MCP + Multi-Agent Pipeline")
    print("═" * 60)
    show_help()
    show_cost()
    print()

    while True:
        try:
            user_input = input("You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[main] Goodbye! 👋")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("[main] Goodbye! 👋")
            break

        elif user_input.lower() == "/help":
            show_help()

        elif user_input.lower() == "/cost":
            show_cost()

        elif user_input.lower().startswith("/add "):
            text = user_input[5:].strip()
            if text:
                handle_add_command(text)
            else:
                print("Usage: /add <text to add to corpus>")

        elif user_input.lower().startswith("/pdf "):
            handle_pdf_command(user_input[5:])

        elif user_input.lower().startswith("/rag "):
            # Plain RAG — fast, no agents
            query = user_input[5:].strip()
            if query:
                print(f"\n[RAG] Answering: '{query}'")
                answer = handle_rag_query(query)
                print("\n" + "━" * 60)
                print("📖  RAG Answer:")
                print("━" * 60)
                print(answer)
                show_cost()
            else:
                print("Usage: /rag <your question>")

        elif user_input.lower().startswith("/agentic "):
            # Full 3-agent pipeline with HITL
            query = user_input[9:].strip()
            if query:
                print(f"\n[Agentic] Running 3-agent pipeline for: '{query}'")
                result = run_agentic_pipeline(
                    query,
                    save_report=False,    # don't save yet — wait for HITL approval
                    check_external=True,
                )
                draft = result["final_answer"]

                # ── STEP 3: Human-in-the-Loop review ─────────────────────────
                final = human_review_loop(query, draft)

                # Save report only after human approval
                if "rejected" not in final.lower():
                    save_result = _call_mcp_tool(
                        "create_markdown_report",
                        title=f"Approved: {query[:60]}",
                        content=final,
                    )
                    print(f"[main] 📄 {save_result}")

                print("\n" + "━" * 60)
                print("✅  FINAL ANSWER (approved by human):")
                print("━" * 60)
                print(final)
                show_cost()
            else:
                print("Usage: /agentic <your question>")

        else:
            # Default: treat plain input as a RAG question
            query = user_input
            print(f"\n[RAG] Answering: '{query}'  (tip: use /agentic for deeper research)")
            answer = handle_rag_query(query)
            print("\n" + "━" * 60)
            print("📖  Answer:")
            print("━" * 60)
            print(answer)
            show_cost()
            print()


if __name__ == "__main__":
    main()
