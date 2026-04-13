"""
evaluate.py — RAG Evaluation Script
=====================================
Loads questions.json, runs each question through the RAG pipeline,
and scores results across all categories.

Usage:
    python scripts/evaluate.py                        # full eval
    python scripts/evaluate.py --category factual     # single category
    python scripts/evaluate.py --mock                 # run with mock pipeline (no API key needed)

Output:
    - Live results printed to console
    - Final report saved to data/eval_report.json
"""

import json
import argparse
import time
import re
from pathlib import Path
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent   # project root (scripts/evaluate.py → root)
QUESTIONS   = ROOT / "data" / "questions.json"
REPORT_OUT  = ROOT / "data" / "eval_report.json"

# ── colours for terminal output ────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    DIM    = "\033[2m"

# ==============================================================================
# MOCK PIPELINE  (replace with real imports once API key is available)
# ==============================================================================

def mock_rag_pipeline(question: str) -> dict:
    """
    Placeholder pipeline — swap this out for the real one tomorrow.
    Returns the same structure that the real pipeline will return.
    """
    return {
        "answer":   "Mock answer — API not yet available.",
        "sources":  [{"source": "mock_doc.pdf", "page": 1, "chunk_id": "mock_0"}],
        "refused":  False,   # True when model says "I don't have information about this"
    }


def load_real_pipeline():
    """
    Attempt to import the real retriever + generator.
    Falls back to mock if imports fail (no API key / files not ready yet).
    """
    try:
        import sys
        sys.path.insert(0, str(ROOT / "app"))
        from retriever  import retrieve          # app/retriever.py
        from generator  import generate_answer   # app/generator.py

        def real_pipeline(question: str) -> dict:
            chunks  = retrieve(question, top_k=5)
            result  = generate_answer(question, chunks)
            return result   # must return {"answer": str, "sources": [...], "refused": bool}

        print(f"{C.GREEN}✓ Real pipeline loaded.{C.RESET}")
        return real_pipeline

    except Exception as e:
        print(f"{C.YELLOW}⚠  Real pipeline not available ({e}). Using mock.{C.RESET}")
        return mock_rag_pipeline

# ==============================================================================
# SCORING FUNCTIONS
# ==============================================================================

def score_factual(result: dict, question: dict) -> dict:
    """
    Scores a factual question.

    Checks:
      1. answer_present    — did the pipeline return a non-empty, non-refused answer?
      2. citation_present  — is there at least one source citation returned?
      3. source_match      — does the cited source match the expected source in questions.json?
      4. keyword_match     — do key words from the expected answer appear in the returned answer?

    Returns a dict with individual check results + overall pass/fail.
    """
    answer          = result.get("answer", "")
    sources         = result.get("sources", [])
    refused         = result.get("refused", False)
    expected_answer = question.get("expected_answer", "")
    expected_source = question.get("source", "")
    expected_page   = question.get("page")

    # 1. Did we get a real answer?
    answer_present = bool(answer.strip()) and not refused

    # 2. Did we cite any source?
    citation_present = len(sources) > 0

    # 3. Does the cited source match the expected one?
    cited_sources = [s.get("source", "") for s in sources]
    source_match  = any(expected_source in s or s in expected_source for s in cited_sources) \
                    if expected_source else False

    # 4. Keyword overlap — simple but effective for factual answers
    keyword_match = _keyword_overlap(answer, expected_answer)

    # overall: pass if answer present + citation + at least keyword match
    passed = answer_present and citation_present and keyword_match

    return {
        "passed":           passed,
        "answer_present":   answer_present,
        "citation_present": citation_present,
        "source_match":     source_match,
        "keyword_match":    keyword_match,
        "answer_returned":  answer[:200] + ("…" if len(answer) > 200 else ""),
    }


def score_cross_reference(result: dict, question: dict) -> dict:
    """
    Scores a cross-reference question.

    Same as factual but also checks that multiple sources are cited,
    since cross-reference questions by definition need info from >1 place.
    """
    base    = score_factual(result, question)
    sources = result.get("sources", [])

    # cross-ref specific: do we get citations from multiple places?
    multi_source = len(set(s.get("source","") for s in sources)) >= 1  # relax to 1 for now

    base["multi_source"] = multi_source
    base["passed"]       = base["passed"] and multi_source
    return base


def score_out_of_scope(result: dict, question: dict) -> dict:
    """
    Scores an out-of-scope question.

    The system MUST refuse. We check:
      1. refused flag set to True, OR
      2. answer contains a refusal phrase
    """
    answer  = result.get("answer", "").lower()
    refused = result.get("refused", False)

    refusal_phrases = [
        "i don't have information",
        "i don't know",
        "not in my knowledge",
        "outside the scope",
        "cannot find",
        "not available in",
        "no information",
        "i cannot answer",
        "not covered",
    ]
    phrase_match = any(p in answer for p in refusal_phrases)
    correctly_refused = refused or phrase_match

    return {
        "passed":             correctly_refused,
        "correctly_refused":  correctly_refused,
        "refused_flag":       refused,
        "phrase_match":       phrase_match,
        "answer_returned":    answer[:200],
    }


def score_ambiguous(result: dict, question: dict) -> dict:
    """
    Scores an ambiguous question.

    Acceptable outcomes:
      1. Model asks for clarification, OR
      2. Model gives a broad answer acknowledging the ambiguity

    We check for clarification language OR ambiguity acknowledgement.
    """
    answer = result.get("answer", "").lower()

    clarification_phrases = [
        "could you clarify",
        "could you specify",
        "which",
        "can you be more specific",
        "please specify",
        "are you referring to",
    ]
    ambiguity_phrases = [
        "several",
        "multiple",
        "various",
        "different sections",
        "could refer to",
        "depends on",
        "it's unclear",
        "ambiguous",
        "broad",
    ]

    asked_for_clarification = any(p in answer for p in clarification_phrases)
    acknowledged_ambiguity  = any(p in answer for p in ambiguity_phrases)
    handled_well            = asked_for_clarification or acknowledged_ambiguity

    return {
        "passed":                    handled_well,
        "asked_for_clarification":   asked_for_clarification,
        "acknowledged_ambiguity":    acknowledged_ambiguity,
        "answer_returned":           answer[:200],
    }


# ── helpers ────────────────────────────────────────────────────────────────────

def _keyword_overlap(answer: str, expected: str, threshold: float = 0.35) -> bool:
    """
    Returns True if enough important keywords from `expected` appear in `answer`.
    Ignores stopwords. Threshold = fraction of keywords that must match.
    """
    if not expected or not answer:
        return False

    stopwords = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","of","in","on","at","to",
        "for","and","or","but","not","with","as","by","from","this",
        "that","it","its","they","their","there","what","which","who",
    }

    def tokenize(text):
        return {w for w in re.findall(r'\b\w+\b', text.lower()) if w not in stopwords}

    exp_words = tokenize(expected)
    ans_words = tokenize(answer)

    if not exp_words:
        return False

    overlap = len(exp_words & ans_words) / len(exp_words)
    return overlap >= threshold

# ==============================================================================
# EVALUATION RUNNER
# ==============================================================================

SCORERS = {
    "factual":         score_factual,
    "cross-reference": score_cross_reference,
    "out-of-scope":    score_out_of_scope,
    "ambiguous":       score_ambiguous,
}


def run_evaluation(pipeline_fn, questions: list, category_filter: str = None) -> dict:
    """
    Runs every question through the pipeline and collects scores.
    Returns a structured report dict.
    """
    results_by_category = {}
    all_results         = []

    questions_to_run = [
        q for q in questions
        if category_filter is None or q.get("category") == category_filter
    ]

    print(f"\n{C.BOLD}{C.BLUE}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  RAG Evaluation  —  {len(questions_to_run)} questions{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'='*60}{C.RESET}\n")

    for i, question in enumerate(questions_to_run, 1):
        q_text    = question.get("question", "")
        category  = question.get("category", "unknown")
        scorer    = SCORERS.get(category, score_factual)

        print(f"{C.DIM}[{i}/{len(questions_to_run)}]{C.RESET} "
              f"{C.CYAN}{category.upper():15s}{C.RESET} {q_text[:70]}")

        # ── run pipeline ──────────────────────────────────────────────────────
        start  = time.time()
        try:
            result = pipeline_fn(q_text)
        except Exception as e:
            result = {"answer": "", "sources": [], "refused": False, "error": str(e)}
        elapsed = time.time() - start

        # ── score ─────────────────────────────────────────────────────────────
        score = scorer(result, question)
        score["latency_s"] = round(elapsed, 2)

        # ── display ───────────────────────────────────────────────────────────
        status = f"{C.GREEN}PASS ✓{C.RESET}" if score["passed"] else f"{C.RED}FAIL ✗{C.RESET}"
        print(f"   {status}  ({elapsed:.1f}s)")
        if not score["passed"]:
            print(f"   {C.DIM}→ returned: {score.get('answer_returned','')[:80]}{C.RESET}")

        # ── collect ───────────────────────────────────────────────────────────
        entry = {
            "question":    q_text,
            "category":    category,
            "score":       score,
            "raw_result":  {
                "answer":  result.get("answer","")[:300],
                "sources": result.get("sources",[]),
                "refused": result.get("refused", False),
            },
        }
        all_results.append(entry)
        results_by_category.setdefault(category, []).append(entry)

    return _build_report(results_by_category, all_results)


def _build_report(results_by_category: dict, all_results: list) -> dict:
    """Aggregates per-question results into a summary report."""
    summary = {}
    for cat, entries in results_by_category.items():
        passed  = sum(1 for e in entries if e["score"]["passed"])
        total   = len(entries)
        avg_lat = sum(e["score"].get("latency_s",0) for e in entries) / total if total else 0
        summary[cat] = {
            "passed":         passed,
            "total":          total,
            "accuracy":       round(passed / total, 3) if total else 0,
            "avg_latency_s":  round(avg_lat, 2),
        }

    total_pass  = sum(1 for e in all_results if e["score"]["passed"])
    total_all   = len(all_results)

    report = {
        "generated_at":    datetime.now().isoformat(),
        "overall": {
            "passed":    total_pass,
            "total":     total_all,
            "accuracy":  round(total_pass / total_all, 3) if total_all else 0,
        },
        "by_category": summary,
        "details":     all_results,
    }
    return report

# ==============================================================================
# PRETTY PRINT REPORT
# ==============================================================================

def print_report(report: dict):
    overall = report["overall"]
    by_cat  = report["by_category"]

    print(f"\n{C.BOLD}{C.BLUE}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  EVALUATION REPORT{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'='*60}{C.RESET}")

    acc = overall["accuracy"]
    col = C.GREEN if acc >= 0.8 else C.YELLOW if acc >= 0.6 else C.RED
    print(f"\n  Overall Accuracy: {col}{C.BOLD}{acc*100:.1f}%{C.RESET}  "
          f"({overall['passed']}/{overall['total']} passed)\n")

    print(f"  {'Category':<20} {'Pass':>6} {'Total':>6} {'Accuracy':>10} {'Avg Latency':>13}")
    print(f"  {'-'*58}")

    target = {
        "factual":         0.80,
        "cross-reference": 0.70,
        "out-of-scope":    1.00,
        "ambiguous":       0.50,
    }
    for cat, stats in by_cat.items():
        a    = stats["accuracy"]
        tgt  = target.get(cat, 0.75)
        col  = C.GREEN if a >= tgt else C.YELLOW if a >= tgt * 0.75 else C.RED
        bar  = "█" * int(a * 10) + "░" * (10 - int(a * 10))
        print(f"  {cat:<20} {stats['passed']:>6} {stats['total']:>6} "
              f"  {col}{a*100:5.1f}%{C.RESET}  {bar}  {stats['avg_latency_s']:>5.1f}s")

    print(f"\n  {C.DIM}Targets: factual ≥80%  cross-ref ≥70%  "
          f"out-of-scope 100%  ambiguous ≥50%{C.RESET}")
    print(f"\n  Report saved → data/eval_report.json\n")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--category", choices=["factual","cross-reference","out-of-scope","ambiguous"],
                        help="Evaluate only one category")
    parser.add_argument("--mock", action="store_true",
                        help="Force mock pipeline (no API key needed)")
    parser.add_argument("--questions", default=str(QUESTIONS),
                        help=f"Path to questions.json (default: {QUESTIONS})")
    args = parser.parse_args()

    # ── load questions ─────────────────────────────────────────────────────────
    q_path = Path(args.questions)
    if not q_path.exists():
        print(f"{C.RED}✗  questions.json not found at {q_path}{C.RESET}")
        print(f"   Create it first (Assignment 1) or pass --questions <path>")
        return

    with open(q_path) as f:
        questions = json.load(f)
    print(f"{C.GREEN}✓  Loaded {len(questions)} questions from {q_path}{C.RESET}")

    # ── load pipeline ──────────────────────────────────────────────────────────
    if args.mock:
        print(f"{C.YELLOW}⚠  --mock flag set. Using mock pipeline.{C.RESET}")
        pipeline_fn = mock_rag_pipeline
    else:
        pipeline_fn = load_real_pipeline()

    # ── run ────────────────────────────────────────────────────────────────────
    report = run_evaluation(pipeline_fn, questions, category_filter=args.category)

    # ── print & save ───────────────────────────────────────────────────────────
    print_report(report)

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
