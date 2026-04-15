"""
retrieval_checker.py — Inspect retrieval for a single question.

The most important debugging tool in the pipeline.
When an answer is wrong, run this first — if the right chunks
weren't retrieved, the LLM never had a chance.

Hit detection uses keyword matching against the expected answer
rather than exact page numbers — this is robust to chunk/page
mismatches that occur after re-chunking.

Usage:
    python scripts/retrieval_checker.py "Who are the Artemis II crew members?"
    python scripts/retrieval_checker.py "What is the total height of the SLS Block 1 rocket?" --k 8
    python scripts/retrieval_checker.py "Tell me about the requirements." --debug

Output:
    - Top-k retrieved chunks (text, source, page, score)
    - Whether the expected answer keywords were found in retrieved chunks ✅ or missed ❌
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "questions.json"

sys.path.insert(0, str(ROOT / "app"))
from retriever import load_index_and_chunks, retrieve


# ── Terminal colours ──────────────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    DIM    = "\033[2m"


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


# ── Load question from questions.json ─────────────────────────────────────────

def load_question(query: str) -> dict | None:
    """Find the matching question in questions.json by exact text match."""
    if not QUESTIONS.exists():
        return None

    with open(QUESTIONS, encoding="utf-8") as f:
        questions = json.load(f)

    for q in questions:
        if q.get("question", "").lower().strip() == query.lower().strip():
            return q

    return None


# ── Extract keywords from expected answer ─────────────────────────────────────

def extract_keywords(text: str) -> list[str]:
    """
    Extract meaningful keywords from expected answer.
    Prioritises numbers and technical terms — these are most specific
    and don't depend on exact page numbers.
    """
    if not text:
        return []

    # Numbers and measurements first — most specific
    numbers = re.findall(r'\b[\d,]+\.?\d*\b', text)

    # Meaningful words — longer than 3 chars, not stopwords
    words = [
        w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', text)
        if w.lower() not in STOPWORDS
    ]

    return numbers + words


# ── Check retrieval hit via keyword matching ──────────────────────────────────

def check_retrieval_hit(
    results: list[dict],
    question: dict,
) -> tuple[bool, list[str], list[str]]:
    """
    Check if the retrieved chunks contain keywords from the expected answer.

    Uses keyword matching instead of exact page numbers — robust to
    re-chunking and page boundary differences.

    Returns:
        hit:           True if enough keywords were found
        found_kws:     Keywords that were found in retrieved chunks
        missing_kws:   Keywords that were NOT found
    """
    expected_answer = question.get("expected_answer")
    expected_source = question.get("source")

    # Out-of-scope / ambiguous — no expected answer to check
    if not expected_answer:
        return True, [], []

    keywords = extract_keywords(expected_answer)
    if not keywords:
        return True, [], []

    # Combine all retrieved text into one searchable blob
    all_retrieved_text = " ".join(
        r.get("text", "").lower() for r in results
    )
    all_retrieved_sources = [r.get("source", "") for r in results]

    found   = [kw for kw in keywords if kw.lower() in all_retrieved_text]
    missing = [kw for kw in keywords if kw.lower() not in all_retrieved_text]

    # Source check — did we retrieve from the right document?
    source_hit = any(
        expected_source in s or s in expected_source
        for s in all_retrieved_sources
    ) if expected_source else True

    # Pass if ≥50% of keywords found AND source matches
    keyword_hit_rate = len(found) / len(keywords) if keywords else 0
    hit = keyword_hit_rate >= 0.5 and source_hit

    return hit, found, missing


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_results(
    query: str,
    results: list[dict],
    question: dict | None,
    hit: bool,
    found_kws: list[str],
    missing_kws: list[str],
) -> None:

    print(f"\n{C.BOLD}{C.BLUE}{'='*70}{C.RESET}")
    print(f"{C.BOLD}  RETRIEVAL CHECKER{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'='*70}{C.RESET}")
    print(f"\n{C.BOLD}Query:{C.RESET}     {query}")
    print(f"{C.BOLD}Retrieved:{C.RESET} {len(results)} chunks\n")

    # ── Retrieved chunks ──────────────────────────────────────────────────────
    print(f"{C.CYAN}{C.BOLD}── Top Retrieved Chunks ──────────────────────────────────{C.RESET}")
    for i, chunk in enumerate(results, 1):
        score        = chunk.get("score", "N/A")
        rerank_score = chunk.get("rerank_score")
        source       = chunk.get("source", "unknown")
        page         = chunk.get("page", "?")
        chunk_id     = chunk.get("chunk_id", "?")
        text_preview = chunk.get("text", "")[:250].replace("\n", " ")

        score_str = f"{score}"
        if rerank_score is not None:
            score_str += f" → reranked: {rerank_score}"

        print(f"\n  {C.BOLD}[{i}]{C.RESET} {C.YELLOW}{source}{C.RESET} — page {page}")
        print(f"       chunk_id: {chunk_id}  |  score: {score_str}")
        print(f"       {C.DIM}{text_preview}…{C.RESET}")

    # ── Expected answer keyword check ─────────────────────────────────────────
    print(f"\n{C.CYAN}{C.BOLD}── Expected Answer Keyword Check ─────────────────────────{C.RESET}")

    if question is None:
        print(f"  {C.DIM}Question not found in questions.json — skipping keyword check.{C.RESET}")
        print(f"  {C.DIM}Tip: question must match exactly as written in questions.json.{C.RESET}")

    elif not question.get("expected_answer"):
        print(f"  {C.DIM}No expected answer (out-of-scope or ambiguous) — skipping check.{C.RESET}")

    else:
        expected_answer = question.get("expected_answer", "")
        expected_source = question.get("source", "")
        expected_page   = question.get("page")

        print(f"  Expected source : {expected_source} — page {expected_page}")
        print(f"  Expected answer : {expected_answer[:100]}…" if len(expected_answer) > 100
              else f"  Expected answer : {expected_answer}")
        print()

        if found_kws:
            print(f"  {C.GREEN}Keywords found   : {', '.join(found_kws[:10])}{C.RESET}")
        if missing_kws:
            print(f"  {C.RED}Keywords missing : {', '.join(missing_kws[:10])}{C.RESET}")

        print()
        if hit:
            print(f"  {C.GREEN}{C.BOLD}✅ RETRIEVAL HIT — expected answer keywords found in retrieved chunks.{C.RESET}")
        else:
            print(f"  {C.RED}{C.BOLD}❌ RETRIEVAL MISS — expected answer keywords NOT found.{C.RESET}")
            print(f"  {C.YELLOW}→ This is a retrieval problem. The LLM never saw the right context.{C.RESET}")
            print(f"  {C.YELLOW}→ Try: increase --k, adjust chunk size, or check embeddings.{C.RESET}")

    print(f"\n{C.BOLD}{C.BLUE}{'='*70}{C.RESET}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect retrieval for a single question.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/retrieval_checker.py "Who are the four crew members of the Artemis II mission?"
  python scripts/retrieval_checker.py "What is the total height of the SLS Block 1 rocket?" --k 8
  python scripts/retrieval_checker.py "Tell me about the requirements for the mission." --debug
        """
    )
    parser.add_argument("query", help="The question to check retrieval for.")
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of chunks to retrieve (default: 5)."
    )
    parser.add_argument(
        "--candidate-k", type=int, default=10,
        help="Number of FAISS candidates before reranking (default: 10)."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output from retriever."
    )
    args = parser.parse_args()

    # Load pipeline
    print(f"{C.DIM}[checker] Loading index and chunks...{C.RESET}")
    index, chunks = load_index_and_chunks()

    # Retrieve
    results = retrieve(
        query=args.query,
        index=index,
        chunks=chunks,
        final_k=args.k,
        candidate_k=args.candidate_k,
        debug=args.debug,
    )

    # Load question from questions.json
    question = load_question(args.query)

    # Check hit via keyword matching
    hit, found_kws, missing_kws = check_retrieval_hit(results, question or {})

    # Print
    print_results(args.query, results, question, hit, found_kws, missing_kws)


if __name__ == "__main__":
    main()
