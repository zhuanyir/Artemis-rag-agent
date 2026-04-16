"""
eval_agentic.py — Agentic Pipeline Evaluation with LLM-as-Judge
================================================================
Evaluates the multi-agent system across four dimensions:

  1. Agent 1 Quality     — Did the internal researcher find relevant corpus info?
  2. Agent 3 Faithfulness — Did the synthesizer stick to what agents found?
  3. Hallucination Check — Did any agent invent facts not in the context?
  4. HITL Appropriateness — Is the draft clear enough for a human to review?

Uses the same questions.json test set as the RAG evaluator.
Runs only Agent 1 + Agent 3 (skips live web search to save cost & time).

Usage:
    python scripts/eval_agentic.py                    # full eval
    python scripts/eval_agentic.py --category factual # one category
    python scripts/eval_agentic.py --n 5              # first N questions only

Output:
    - Live results printed to console
    - Report saved to data/eval_agentic_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
QUESTIONS   = ROOT / "data" / "questions.json"
REPORT_OUT  = ROOT / "data" / "eval_agentic_report.json"
COST_FILE   = ROOT / "cost_tracker.json"

sys.path.insert(0, str(ROOT / "app"))

# ── Terminal colours ──────────────────────────────────────────────────────────
class C:
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"
    DIM     = "\033[2m"


# ══════════════════════════════════════════════════════════════════════════════
#  LLM-AS-JUDGE — Agentic-specific rubric
#  Evaluates four dimensions unique to multi-agent systems
# ══════════════════════════════════════════════════════════════════════════════

AGENTIC_JUDGE_PROMPT = """You are evaluating a multi-agent AI system for the Artemis II space mission.

The system has three agents:
  - Agent 1 (Internal Researcher): searches a NASA document corpus
  - Agent 2 (External Fact-Checker): searches the web (skipped in this eval)
  - Agent 3 (Synthesizer): writes the final answer from agent findings

You will receive:
  - The user question
  - The expected answer (ground truth)
  - Agent 1's internal corpus answer
  - Agent 3's final synthesized answer
  - The question category

Evaluate on FOUR dimensions. Respond ONLY with a valid JSON object:
{
  "agent1_quality": 0.0-1.0,
  "agent1_reasoning": "one sentence — did Agent 1 find the right info from corpus?",
  "faithfulness": 0.0-1.0,
  "faithfulness_reasoning": "one sentence — did Agent 3 stay faithful to what Agent 1 found?",
  "hallucination": "none | minor | major",
  "hallucination_reasoning": "one sentence — did any agent invent facts not in the context?",
  "hitl_appropriate": true or false,
  "hitl_reasoning": "one sentence — is the draft clear enough for a human to usefully review?",
  "overall_passed": true or false,
  "overall_score": 0.0-1.0
}

Scoring guidelines:
- agent1_quality: 1.0 = found correct answer with citation | 0.5 = found partial info | 0.0 = said no info when info exists
- faithfulness: 1.0 = Agent 3 only uses what Agent 1 found | 0.5 = minor additions | 0.0 = contradicts or ignores Agent 1
- hallucination: "none" = no invented facts | "minor" = small additions | "major" = fabricated key facts
- hitl_appropriate: true if draft is coherent and reviewable (even if wrong — human can correct it)
- overall_passed: true if agent1_quality >= 0.5 AND faithfulness >= 0.5 AND hallucination != "major"
"""


def llm_judge_agentic(
    question: dict,
    agent1_answer: str,
    final_answer: str,
) -> dict:
    """Run LLM-as-judge on the agentic pipeline output."""
    try:
        from openai import OpenAI
        client = OpenAI()

        user_content = f"""QUESTION: {question.get('question', '')}
CATEGORY: {question.get('category', 'factual')}
EXPECTED ANSWER: {question.get('expected_answer', 'N/A')}

AGENT 1 INTERNAL ANSWER:
{agent1_answer}

AGENT 3 FINAL ANSWER:
{final_answer}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            temperature=0,
            messages=[
                {"role": "system", "content": AGENTIC_JUDGE_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        verdict = json.loads(raw)

        # Track cost
        usage = response.usage
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000
        _track_cost(cost)

        return {**verdict, "judge": "llm", "error": None}

    except Exception as e:
        return {
            "agent1_quality": 0.0,
            "agent1_reasoning": f"Judge error: {e}",
            "faithfulness": 0.0,
            "faithfulness_reasoning": "",
            "hallucination": "unknown",
            "hallucination_reasoning": "",
            "hitl_appropriate": False,
            "hitl_reasoning": "",
            "overall_passed": False,
            "overall_score": 0.0,
            "judge": "fallback",
            "error": str(e),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  COST TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def _track_cost(cost: float) -> None:
    data = {"total": 0.0, "calls": 0}
    if COST_FILE.exists():
        with open(COST_FILE) as f:
            data = json.load(f)
    data["total"] = round(data.get("total", 0) + cost, 6)
    data["calls"] = data.get("calls", 0) + 1
    with open(COST_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 1 + AGENT 3 MINI-PIPELINE  (no web search — saves cost)
# ══════════════════════════════════════════════════════════════════════════════

def run_internal_pipeline(query: str, index, chunks) -> tuple[str, str, list[dict]]:
    """
    Run Agent 1 (internal research) + Agent 3 (synthesis) only.
    Skips Agent 2 web search to keep eval fast and cheap.

    Returns: (agent1_answer, final_answer, retrieved_chunks)
    """
    from retriever import retrieve
    from generator import generate_answer

    # Agent 1: retrieve + generate internal answer
    retrieved = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        final_k=5,
        candidate_k=10,
    )

    if not retrieved:
        return (
            "INTERNAL: No information found in corpus.",
            "I don't have information about this in the Artemis II documents.",
            [],
        )

    # Agent 1 answer (same as standalone RAG — the internal researcher)
    agent1_answer = generate_answer(query, retrieved, history=None)

    # Agent 3 synthesis (internal only — no external findings)
    # We pass a clear note that web search was skipped for this eval run
    from openai import OpenAI
    client = OpenAI()

    agent3_prompt = f"""You are the Synthesizer for the Artemis II mission assistant.
The external web search was skipped for this evaluation run.
Synthesize a final answer based ONLY on the internal corpus findings below.
Be concise, cite sources, do not invent facts.

Internal corpus answer:
{agent1_answer}

User question: {query}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": agent3_prompt}],
        temperature=0,
        max_tokens=400,
    )

    usage = response.usage
    cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000
    _track_cost(cost)

    final_answer = response.choices[0].message.content.strip()
    return agent1_answer, final_answer, retrieved


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EVAL LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(questions: list[dict], index, chunks) -> dict:
    print(f"\n{C.BOLD}{C.BLUE}{'='*70}{C.RESET}")
    print(f"{C.BOLD}  AGENTIC EVAL — {len(questions)} questions  (Agent 1 + Agent 3){C.RESET}")
    print(f"{C.BOLD}  LLM-as-Judge: Agent Quality · Faithfulness · Hallucination · HITL{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'='*70}{C.RESET}\n")

    results = []
    cost_before = _get_total_cost()

    for i, q in enumerate(questions, 1):
        q_text   = q.get("question", "")
        category = q.get("category", "unknown")

        print(f"{C.DIM}[{i}/{len(questions)}]{C.RESET} "
              f"{C.CYAN}{category.upper():15s}{C.RESET} {q_text[:55]}")

        start = time.time()

        # Run pipeline
        try:
            agent1_answer, final_answer, retrieved = run_internal_pipeline(
                q_text, index, chunks
            )
        except Exception as e:
            print(f"   {C.RED}Pipeline error: {e}{C.RESET}")
            agent1_answer = f"ERROR: {e}"
            final_answer  = f"ERROR: {e}"
            retrieved     = []

        # LLM-as-judge
        verdict = llm_judge_agentic(q, agent1_answer, final_answer)
        elapsed = time.time() - start

        # Display
        passed = verdict.get("overall_passed", False)
        score  = verdict.get("overall_score", 0.0)
        hall   = verdict.get("hallucination", "unknown")
        hitl   = verdict.get("hitl_appropriate", False)

        status_col = C.GREEN if passed else C.RED
        hall_col   = C.GREEN if hall == "none" else C.YELLOW if hall == "minor" else C.RED
        hitl_icon  = f"{C.GREEN}✅{C.RESET}" if hitl else f"{C.RED}❌{C.RESET}"

        print(f"   {status_col}{'PASS ✓' if passed else 'FAIL ✗'}{C.RESET}  "
              f"score={C.BOLD}{score:.2f}{C.RESET}  "
              f"halluc={hall_col}{hall}{C.RESET}  "
              f"hitl={hitl_icon}  ({elapsed:.1f}s)")

        if not passed:
            print(f"   {C.DIM}agent1: {verdict.get('agent1_reasoning','')[:70]}{C.RESET}")
            print(f"   {C.DIM}faith:  {verdict.get('faithfulness_reasoning','')[:70]}{C.RESET}")

        results.append({
            "question":         q_text,
            "category":         category,
            "expected_answer":  q.get("expected_answer"),
            "agent1_answer":    agent1_answer[:400],
            "final_answer":     final_answer[:400],
            "retrieved_sources": [
                {"source": r.get("source"), "page": r.get("page"), "score": r.get("score")}
                for r in retrieved
            ],
            "verdict":          verdict,
            "latency_s":        round(elapsed, 2),
        })

    cost_after = _get_total_cost()
    return _build_report(results, cost_after - cost_before)


def _get_total_cost() -> float:
    if COST_FILE.exists():
        with open(COST_FILE) as f:
            return json.load(f).get("total", 0.0)
    return 0.0


def _build_report(results: list[dict], eval_cost: float) -> dict:
    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    cat_summary = {}
    for cat, entries in by_cat.items():
        n       = len(entries)
        passed  = sum(1 for e in entries if e["verdict"].get("overall_passed"))
        avg_a1  = sum(e["verdict"].get("agent1_quality", 0) for e in entries) / n
        avg_fa  = sum(e["verdict"].get("faithfulness", 0) for e in entries) / n
        hall_counts = {"none": 0, "minor": 0, "major": 0, "unknown": 0}
        for e in entries:
            hall_counts[e["verdict"].get("hallucination", "unknown")] += 1
        hitl_ok = sum(1 for e in entries if e["verdict"].get("hitl_appropriate"))

        cat_summary[cat] = {
            "passed": passed, "total": n,
            "accuracy": round(passed / n, 3),
            "avg_agent1_quality": round(avg_a1, 3),
            "avg_faithfulness":   round(avg_fa, 3),
            "hallucination_counts": hall_counts,
            "hitl_appropriate_rate": round(hitl_ok / n, 3),
        }

    total = len(results)
    passed = sum(1 for r in results if r["verdict"].get("overall_passed"))

    return {
        "generated_at":    datetime.now().isoformat(),
        "eval_type":       "agentic (Agent 1 + Agent 3, no web search)",
        "total_questions": total,
        "eval_cost_usd":   round(eval_cost, 4),
        "overall": {
            "passed":   passed,
            "total":    total,
            "accuracy": round(passed / total, 3) if total else 0,
            "avg_agent1_quality": round(
                sum(r["verdict"].get("agent1_quality", 0) for r in results) / total, 3
            ) if total else 0,
            "avg_faithfulness": round(
                sum(r["verdict"].get("faithfulness", 0) for r in results) / total, 3
            ) if total else 0,
            "hallucination_major_rate": round(
                sum(1 for r in results if r["verdict"].get("hallucination") == "major") / total, 3
            ) if total else 0,
            "hitl_appropriate_rate": round(
                sum(1 for r in results if r["verdict"].get("hitl_appropriate")) / total, 3
            ) if total else 0,
        },
        "by_category": cat_summary,
        "details":     results,
    }


def print_report(report: dict) -> None:
    overall = report["overall"]

    print(f"\n{C.BOLD}{C.BLUE}{'='*70}{C.RESET}")
    print(f"{C.BOLD}  AGENTIC EVAL SCORECARD{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'='*70}{C.RESET}\n")

    acc   = overall["accuracy"]
    a1q   = overall["avg_agent1_quality"]
    faith = overall["avg_faithfulness"]
    hall  = overall["hallucination_major_rate"]
    hitl  = overall["hitl_appropriate_rate"]
    cost  = report["eval_cost_usd"]

    col = C.GREEN if acc >= 0.75 else C.YELLOW if acc >= 0.5 else C.RED

    print(f"  Overall Pass Rate       : {col}{C.BOLD}{acc*100:.1f}%{C.RESET}  "
          f"({overall['passed']}/{overall['total']})")
    print(f"  Avg Agent 1 Quality     : {C.CYAN}{C.BOLD}{a1q:.2f}/1.0{C.RESET}")
    print(f"  Avg Faithfulness        : {C.CYAN}{C.BOLD}{faith:.2f}/1.0{C.RESET}")
    print(f"  Major Hallucination Rate: "
          f"{(C.GREEN if hall == 0 else C.RED)}{C.BOLD}{hall*100:.1f}%{C.RESET}")
    print(f"  HITL Appropriate Rate   : {C.CYAN}{C.BOLD}{hitl*100:.1f}%{C.RESET}")
    print(f"  Eval Cost               : ${C.BOLD}{cost:.4f}{C.RESET}")

    print(f"\n  {'Category':<20} {'Pass':>5} {'Total':>6} {'Acc':>7} "
          f"{'A1 Qual':>8} {'Faith':>7} {'HITL':>6}")
    print(f"  {'-'*65}")

    for cat, s in report["by_category"].items():
        a = s["accuracy"]
        col = C.GREEN if a >= 0.75 else C.YELLOW if a >= 0.5 else C.RED
        print(f"  {cat:<20} {s['passed']:>5} {s['total']:>6}  "
              f"{col}{a*100:5.1f}%{C.RESET}  "
              f"{C.CYAN}{s['avg_agent1_quality']:.2f}{C.RESET}    "
              f"{C.CYAN}{s['avg_faithfulness']:.2f}{C.RESET}  "
              f"{C.CYAN}{s['hitl_appropriate_rate']*100:.0f}%{C.RESET}")

    print(f"\n  {C.GREEN}Report saved → data/eval_agentic_report.json{C.RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the agentic pipeline with LLM-as-judge.",
    )
    parser.add_argument("--category",
        choices=["factual", "cross-reference", "out-of-scope", "ambiguous"])
    parser.add_argument("--n", type=int, default=None,
        help="Limit to first N questions (useful for a quick smoke test)")
    args = parser.parse_args()

    # Load questions
    with open(QUESTIONS, encoding="utf-8") as f:
        questions = json.load(f)

    if args.category:
        questions = [q for q in questions if q.get("category") == args.category]
    if args.n:
        questions = questions[:args.n]

    print(f"{C.GREEN}✓ Loaded {len(questions)} questions{C.RESET}")

    # Load pipeline
    print(f"{C.DIM}Loading FAISS index and chunks...{C.RESET}")
    from retriever import load_index_and_chunks
    index, chunks = load_index_and_chunks()
    print(f"{C.GREEN}✓ Pipeline ready — {index.ntotal} vectors{C.RESET}")

    # Run
    report = run_eval(questions, index, chunks)

    # Print + save
    print_report(report)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
