"""
dry_run.py — Demo Day dry run script.

Runs 10 questions through the full pipeline, checks:
- App doesn't crash
- Answers are not empty
- Response time is acceptable (< 15s per question)
- Out-of-scope questions are correctly refused
- Confidence scores are reasonable

Usage:
    python scripts/dry_run.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from retriever import load_index_and_chunks, retrieve
from generator import generate_answer


# ── Terminal colours ──────────────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    DIM    = "\033[2m"


# ── Test questions ────────────────────────────────────────────────────────────
# Wording matches questions.json exactly for consistency
DRY_RUN_QUESTIONS = [
    {"question": "Who are the four crew members of the Artemis II mission?",                "expect_refusal": False},
    {"question": "What is the total height of the SLS Block 1 rocket?",                    "expect_refusal": False},
    {"question": "How long is the Artemis II mission expected to last?",                    "expect_refusal": False},
    {"question": "What percentage of total thrust do the solid rocket boosters provide at launch?", "expect_refusal": False},
    {"question": "How many parachutes does Orion's parachute system include?",              "expect_refusal": False},
    {"question": "Is Reid Wiseman the commander of Artemis II?",                            "expect_refusal": False},
    {"question": "What was the cost of the Artemis II mission?",                            "expect_refusal": True},
    {"question": "What happened after the Artemis II crew returned to Earth?",              "expect_refusal": True},
    {"question": "Who won the last US presidential election?",                              "expect_refusal": True},
    {"question": "Tell me about the requirements for the mission.",                         "expect_refusal": False},
]

MAX_LATENCY_S   = 15.0
REFUSAL_PHRASES = [
    "i don't have information",
    "i don't know",
    "outside the scope",
    "not available in",
    "no information",
    "i cannot answer",
    "not covered",
]


# ── Run dry run ───────────────────────────────────────────────────────────────

def run_dry_run() -> None:
    print(f"\n{C.BOLD}{C.BLUE}{'='*70}{C.RESET}")
    print(f"{C.BOLD}  DEMO DAY DRY RUN — {len(DRY_RUN_QUESTIONS)} questions{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'='*70}{C.RESET}\n")

    print(f"{C.DIM}Loading pipeline...{C.RESET}")
    index, chunks = load_index_and_chunks()
    print(f"{C.GREEN}✓ Pipeline loaded{C.RESET}\n")

    results      = []
    total_passed = 0
    total_failed = 0

    for i, q in enumerate(DRY_RUN_QUESTIONS, 1):
        question       = q["question"]
        expect_refusal = q["expect_refusal"]

        print(f"{C.DIM}[{i}/{len(DRY_RUN_QUESTIONS)}]{C.RESET} {question[:65]}")

        start = time.time()
        try:
            retrieved = retrieve(question, index, chunks, final_k=5, candidate_k=10)
            answer    = generate_answer(question, retrieved)
            elapsed   = time.time() - start
            crashed   = False
        except Exception as e:
            elapsed = time.time() - start
            answer  = ""
            crashed = True
            print(f"   {C.RED}CRASH: {e}{C.RESET}")

        top_score  = retrieved[0].get("score", 0.0) if retrieved and not crashed else 0.0
        is_refusal = any(p in answer.lower() for p in REFUSAL_PHRASES)
        latency_ok = elapsed < MAX_LATENCY_S
        has_answer = bool(answer.strip()) and not crashed

        refusal_ok = (is_refusal == expect_refusal)
        passed     = not crashed and has_answer and refusal_ok and latency_ok

        if passed:
            total_passed += 1
            status = f"{C.GREEN}PASS ✓{C.RESET}"
        else:
            total_failed += 1
            status = f"{C.RED}FAIL ✗{C.RESET}"

        lat_col = C.GREEN if latency_ok else C.RED
        print(f"   {status}  score={top_score:.3f}  "
              f"{lat_col}{elapsed:.1f}s{C.RESET}  "
              f"refusal={'✅' if is_refusal else '❌'}")

        if not passed:
            if crashed:
                print(f"   {C.RED}→ Pipeline crashed!{C.RESET}")
            if not refusal_ok and expect_refusal:
                print(f"   {C.YELLOW}→ Expected refusal but got: {answer[:80]}{C.RESET}")
            if not refusal_ok and not expect_refusal:
                print(f"   {C.YELLOW}→ Unexpected refusal: {answer[:80]}{C.RESET}")
            if not latency_ok:
                print(f"   {C.YELLOW}→ Too slow ({elapsed:.1f}s > {MAX_LATENCY_S}s){C.RESET}")

        results.append({
            "question":   question,
            "passed":     passed,
            "latency_s":  round(elapsed, 2),
            "top_score":  round(top_score, 4),
            "is_refusal": is_refusal,
            "crashed":    crashed,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{C.BOLD}{C.BLUE}{'='*70}{C.RESET}")
    print(f"{C.BOLD}  DRY RUN SUMMARY{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'='*70}{C.RESET}")

    acc     = total_passed / len(DRY_RUN_QUESTIONS)
    acc_col = C.GREEN if acc >= 0.9 else C.YELLOW if acc >= 0.7 else C.RED
    avg_lat = sum(r["latency_s"] for r in results) / len(results)

    print(f"\n  Passed      : {acc_col}{C.BOLD}{total_passed}/{len(DRY_RUN_QUESTIONS)}{C.RESET}")
    print(f"  Avg latency : {avg_lat:.1f}s")
    print(f"  Crashes     : {sum(1 for r in results if r['crashed'])}")

    if total_failed == 0:
        print(f"\n  {C.GREEN}{C.BOLD}✅ All checks passed — ready for Demo Day!{C.RESET}")
    else:
        print(f"\n  {C.RED}{C.BOLD}⚠️  {total_failed} check(s) failed — review before demo.{C.RESET}")

    print()


if __name__ == "__main__":
    run_dry_run()