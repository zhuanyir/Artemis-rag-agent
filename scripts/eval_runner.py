"""
eval_runner.py — Run all questions.json through the pipeline and produce a scorecard.

Simpler than evaluate.py — focused on retrieval hit rate and saving
results to data/eval_results.json for comparison across runs.

Hit detection uses keyword matching against the expected answer
rather than exact page numbers — robust to re-chunking.

Usage:
    python scripts/eval_runner.py
    python scripts/eval_runner.py --category factual
    python scripts/eval_runner.py --k 8

Output:
    - Live results printed to console
    - Scorecard saved to data/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
QUESTIONS   = ROOT / "data" / "questions.json"
RESULTS_OUT = ROOT / "data" / "eval_results.json"

sys.path.insert(0, str(ROOT / "app"))
from retriever import load_index_and_chunks, retrieve
from generator import generate_answer


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


# ── Stopwords ─────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "not", "with", "as", "by",
    "from", "this", "that", "it", "its", "they", "their", "each",
    "all", "both", "more", "than", "one", "two", "four", "five",
    "approximately", "about", "around", "roughly", "during",
}


# ── Keyword extraction ────────────────────────────────────────────────────────

def extract_keywords(text: str) -> list[str]:
    """
    Extract meaningful keywords from expected answer text.
    Prioritises numbers and technical terms.
    """
    if not text:
        return []
    numbers = re.findall(r'\b[\d,]+\.?\d*\b', text)
    words = [
        w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', text)
        if w.lower() not in STOPWORDS
    ]
    return numbers + words


# ── Retrieval hit check ───────────────────────────────────────────────────────

def check_retrieval_hit(results: list[dict], question: dict) -> bool:
    """
    Returns True if keywords from the expected answer appear in
    the retrieved chunks AND the correct source PDF was retrieved.

    Uses keyword matching instead of exact page numbers — robust
    to chunk/page mismatches after re-chunking.
    """
    expected_answer = question.get("expected_answer")
    expected_source = question.get("source")

    # out-of-scope / ambiguous — no expected answer
    if not expected_answer:
        return True

    keywords = extract_keywords(expected_answer)
    if not keywords:
        return True

    # Combine all retrieved text
    all_text = " ".join(r.get("text", "").lower() for r in results)
    all_sources = [r.get("source", "") for r in results]

    found = [kw for kw in keywords if kw.lower() in all_text]
    keyword_hit_rate = len(found) / len(keywords) if keywords else 0

    # Source check
    source_hit = any(
        expected_source in s or s in expected_source
        for s in all_sources
    ) if expected_source else True

    return keyword_hit_rate >= 0.5 and source_hit


# ── Refusal detection ─────────────────────────────────────────────────────────

def is_refusal(answer: str) -> bool:
    refusal_phrases = [
        "i don't have information",
        "i don't know",
        "outside the scope",
        "cannot find",
        "not available in",
        "no information",
        "i cannot answer",
        "not covered",
    ]
    return any(p in answer.lower() for p in refusal_phrases)


# ── Run evaluation ────────────────────────────────────────────────────────────

def run_eval(
    questions: list[dict],
    index,
    chunks: list[dict],
    category_filter: str | None,
    k: int,
) -> dict:

    questions_to_run = [
        q for q in questions
        if category_filter is None or q.get("category") == category_filter
    ]

    print(f"\n{C.BOLD}{C.BLUE}{'='*70}{C.RESET}")
    print(f"{C.BOLD}  EVAL RUNNER — {len(questions_to_run)} questions{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'='*70}{C.RESET}\n")

    all_results         = []
    retrieval_hits      = 0
    total_with_expected = 0

    for i, question in enumerate(questions_to_run, 1):
        q_text   = question.get("question", "")
        category = question.get("category", "unknown")

        print(f"{C.DIM}[{i}/{len(questions_to_run)}]{C.RESET} "
              f"{C.CYAN}{category.upper():15s}{C.RESET} {q_text[:60]}")

        start = time.time()

        # 1. Retrieve
        try:
            retrieved = retrieve(
                query=q_text,
                index=index,
                chunks=chunks,
                final_k=k,
                candidate_k=k * 2,
            )
        except Exception as e:
            retrieved = []
            print(f"   {C.RED}Retrieval error: {e}{C.RESET}")

        # 2. Check retrieval hit via keyword matching
        hit = check_retrieval_hit(retrieved, question)
        if question.get("expected_answer"):
            total_with_expected += 1
            if hit:
                retrieval_hits += 1

        hit_icon = f"{C.GREEN}✅{C.RESET}" if hit else f"{C.RED}❌{C.RESET}"

        # 3. Generate answer
        try:
            answer = generate_answer(q_text, retrieved)
        except Exception as e:
            answer = ""
            print(f"   {C.RED}Generation error: {e}{C.RESET}")

        elapsed = time.time() - start
        refused = is_refusal(answer)

        # 4. Determine pass/fail per category
        if category == "out-of-scope":
            passed = refused
        elif category == "ambiguous":
            passed = True  # manual check needed
        else:
            passed = hit and not refused

        status = f"{C.GREEN}PASS ✓{C.RESET}" if passed else f"{C.RED}FAIL ✗{C.RESET}"
        print(f"   {status}  retrieval: {hit_icon}  ({elapsed:.1f}s)")

        if not passed:
            print(f"   {C.DIM}→ answer: {answer[:100]}…{C.RESET}")

        # 5. Collect
        all_results.append({
            "question":        q_text,
            "category":        category,
            "expected_source": question.get("source"),
            "expected_page":   question.get("page"),
            "expected_answer": question.get("expected_answer"),
            "retrieval_hit":   hit,
            "passed":          passed,
            "refused":         refused,
            "latency_s":       round(elapsed, 2),
            "answer":          answer[:400],
            "retrieved_chunks": [
                {
                    "source":   r.get("source"),
                    "page":     r.get("page"),
                    "chunk_id": r.get("chunk_id"),
                    "score":    r.get("score"),
                }
                for r in retrieved
            ],
        })

    return _build_scorecard(all_results, retrieval_hits, total_with_expected)


# ── Build scorecard ───────────────────────────────────────────────────────────

def _build_scorecard(
    all_results: list[dict],
    retrieval_hits: int,
    total_with_expected: int,
) -> dict:

    by_category: dict[str, list] = {}
    for r in all_results:
        by_category.setdefault(r["category"], []).append(r)

    category_summary = {}
    for cat, entries in by_category.items():
        passed  = sum(1 for e in entries if e["passed"])
        hits    = sum(1 for e in entries if e["retrieval_hit"])
        total   = len(entries)
        avg_lat = sum(e["latency_s"] for e in entries) / total if total else 0
        failures = [
            {
                "question":      e["question"],
                "retrieval_hit": e["retrieval_hit"],
                "answer":        e["answer"][:150],
            }
            for e in entries if not e["passed"]
        ]
        category_summary[cat] = {
            "passed":             passed,
            "total":              total,
            "accuracy":           round(passed / total, 3) if total else 0,
            "retrieval_hit_rate": round(hits / total, 3) if total else 0,
            "avg_latency_s":      round(avg_lat, 2),
            "failures":           failures,
        }

    total_passed = sum(1 for r in all_results if r["passed"])
    total_all    = len(all_results)

    return {
        "generated_at":    datetime.now().isoformat(),
        "total_questions": total_all,
        "overall": {
            "passed":             total_passed,
            "total":              total_all,
            "accuracy":           round(total_passed / total_all, 3) if total_all else 0,
            "retrieval_hit_rate": round(
                retrieval_hits / total_with_expected, 3
            ) if total_with_expected else 0,
        },
        "by_category": category_summary,
        "details":     all_results,
    }


# ── Print scorecard ───────────────────────────────────────────────────────────

def print_scorecard(report: dict) -> None:
    overall = report["overall"]
    by_cat  = report["by_category"]

    print(f"\n{C.BOLD}{C.BLUE}{'='*70}{C.RESET}")
    print(f"{C.BOLD}  EVAL RESULTS SCORECARD{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'='*70}{C.RESET}")

    acc = overall["accuracy"]
    rhr = overall["retrieval_hit_rate"]
    col = C.GREEN if acc >= 0.8 else C.YELLOW if acc >= 0.6 else C.RED

    print(f"\n  Overall Accuracy     : {col}{C.BOLD}{acc*100:.1f}%{C.RESET}  "
          f"({overall['passed']}/{overall['total']} passed)")
    print(f"  Retrieval Hit Rate   : {C.CYAN}{C.BOLD}{rhr*100:.1f}%{C.RESET}  "
          f"(expected answer keywords found in top-k)\n")

    targets = {
        "factual":         0.80,
        "cross-reference": 0.70,
        "out-of-scope":    1.00,
        "ambiguous":       0.50,
    }

    print(f"  {'Category':<20} {'Pass':>5} {'Total':>6} {'Acc':>8} "
          f"{'Ret Hit':>9} {'Lat':>7}")
    print(f"  {'-'*60}")

    for cat, stats in by_cat.items():
        a   = stats["accuracy"]
        rh  = stats["retrieval_hit_rate"]
        tgt = targets.get(cat, 0.75)
        col = C.GREEN if a >= tgt else C.YELLOW if a >= tgt * 0.75 else C.RED
        bar = "█" * int(a * 10) + "░" * (10 - int(a * 10))

        print(f"  {cat:<20} {stats['passed']:>5} {stats['total']:>6}  "
              f"{col}{a*100:5.1f}%{C.RESET}  "
              f"{C.CYAN}{rh*100:5.1f}%{C.RESET}  "
              f"{stats['avg_latency_s']:>5.1f}s  {bar}")

        # Print failures
        for fail in stats.get("failures", []):
            ret_icon = "✅" if fail["retrieval_hit"] else "❌"
            print(f"    {C.RED}✗{C.RESET} {ret_icon} {fail['question'][:55]}")
            print(f"      {C.DIM}{fail['answer'][:80]}…{C.RESET}")

    print(f"\n  {C.DIM}Targets: factual ≥80%  "
          f"cross-ref ≥70%  out-of-scope 100%  ambiguous ≥50%{C.RESET}")
    print(f"\n  {C.GREEN}Results saved → data/eval_results.json{C.RESET}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all questions through the RAG pipeline and produce a scorecard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval_runner.py
  python scripts/eval_runner.py --category factual
  python scripts/eval_runner.py --k 8
        """
    )
    parser.add_argument(
        "--category",
        choices=["factual", "cross-reference", "out-of-scope", "ambiguous"],
        help="Run only one category.",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of chunks to retrieve per question (default: 5).",
    )
    parser.add_argument(
        "--questions",
        default=str(QUESTIONS),
        help=f"Path to questions.json (default: {QUESTIONS})",
    )
    args = parser.parse_args()

    # Load questions
    q_path = Path(args.questions)
    if not q_path.exists():
        print(f"{C.RED}✗ questions.json not found at {q_path}{C.RESET}")
        return

    with open(q_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"{C.GREEN}✓ Loaded {len(questions)} questions{C.RESET}")

    # Load pipeline
    print(f"{C.DIM}Loading FAISS index and chunks...{C.RESET}")
    index, chunks = load_index_and_chunks()

    # Run
    report = run_eval(questions, index, chunks, args.category, args.k)

    # Print + save
    print_scorecard(report)

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
