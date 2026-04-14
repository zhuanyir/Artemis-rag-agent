"""
evaluate.py — RAG Evaluation Script
=====================================
Loads questions.json, runs each question through the RAG pipeline,
and scores results across all categories.

Usage:
    python scripts/evaluate.py                        # full eval (rule-based)
    python scripts/evaluate.py --category factual     # single category only
    python scripts/evaluate.py --mock                 # mock pipeline (no API key)
    python scripts/evaluate.py --LLM                  # LLM-as-judge scoring (needs API key)
    python scripts/evaluate.py --mock --LLM           # mock pipeline + LLM judge

Output:
    - Live results printed to console
    - Final report saved to data/eval_report.json
"""

import json
import argparse
import time
import re
import os
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env before anything else runs

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
QUESTIONS   = ROOT / "data" / "questions.json"
REPORT_OUT  = ROOT / "data" / "eval_report.json"

# ── terminal colours ───────────────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    MAGENTA= "\033[95m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    DIM    = "\033[2m"

# ==============================================================================
# MOCK PIPELINE
# ==============================================================================

def mock_rag_pipeline(question: str) -> dict:
    """Placeholder — swap for the real pipeline once API key is available."""
    return {
        "answer":  "Mock answer — API not yet available.",
        "sources": [{"source": "mock_doc.pdf", "page": 1, "chunk_id": "mock_0"}],
        "refused": False,
    }


def load_real_pipeline():
    try:
        import sys
        sys.path.insert(0, str(ROOT / "app"))
        from retriever import load_index_and_chunks, retrieve
        from generator import generate_answer
 
        index, chunks = load_index_and_chunks()   # load once
 
        def real_pipeline(question: str) -> dict:
            retrieved = retrieve(question, index, chunks, final_k=5)
            answer = generate_answer(question, retrieved)
 
            # Detect refusal
            refused = "i don't have information" in answer.lower()
 
            # Format sources
            sources = [
                {"source": c["source"], "page": c["page"], "chunk_id": c.get("chunk_id")}
                for c in retrieved
            ]
 
            return {
                "answer":  answer,
                "sources": sources,
                "refused": refused,
            }
 
        print(f"{C.GREEN}✓ Real pipeline loaded.{C.RESET}")
        return real_pipeline
 
    except Exception as e:
        print(f"{C.YELLOW}⚠  Real pipeline not available ({e}). Using mock.{C.RESET}")
        return mock_rag_pipeline

# ==============================================================================
# LLM-AS-JUDGE  (activated only with --LLM flag)
# ==============================================================================

LLM_JUDGE_SYSTEM_PROMPT = """You are a strict but fair evaluator for a RAG (Retrieval-Augmented Generation) system.
Your job is to assess whether the system's answer is correct, faithful to the source, and appropriately handles edge cases.

You will be given:
- The question asked
- The expected answer (ground truth) — may be null for out-of-scope/ambiguous questions
- The system's actual answer
- The sources cited by the system
- The question category: factual | cross-reference | out-of-scope | ambiguous

Respond ONLY with a JSON object — no preamble, no markdown fences. Schema:
{
  "passed": true or false,
  "score": 0.0 to 1.0,
  "reasoning": "one or two sentences explaining the verdict",
  "faithfulness": "high | medium | low",
  "completeness": "high | medium | low"
}

Scoring guidelines per category:
- factual: passed=true only if answer is factually correct AND cites a source
- cross-reference: passed=true if answer synthesises info correctly AND cites sources
- out-of-scope: passed=true ONLY if system refuses (expected_answer is null — system must not fabricate)
- ambiguous: passed=true if system asks for clarification OR gives a broad answer acknowledging ambiguity"""


def llm_judge(question: dict, result: dict) -> dict:
    """
    Calls gpt-4o-mini to evaluate the RAG system's answer.
    Returns a dict with: passed, score, reasoning, faithfulness, completeness.
    Falls back to rule-based score on any API error.
    """
    try:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env

        q_text          = question.get("question", "")
        category        = question.get("category", "factual")
        expected_answer = question.get("expected_answer")
        answer          = result.get("answer", "")
        sources         = result.get("sources", [])
        refused         = result.get("refused", False)

        user_content = f"""QUESTION: {q_text}
CATEGORY: {category}
EXPECTED ANSWER: {expected_answer if expected_answer else "null (system must refuse — question is out of scope or too ambiguous)"}
SYSTEM ANSWER: {answer}
SYSTEM REFUSED: {refused}
SOURCES CITED: {json.dumps(sources, indent=2)}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            temperature=0,
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )

        raw = response.choices[0].message.content.strip()
        # strip markdown fences if model adds them anyway
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        verdict = json.loads(raw)

        return {
            "passed":        bool(verdict.get("passed", False)),
            "score":         float(verdict.get("score", 0.0)),
            "reasoning":     verdict.get("reasoning", ""),
            "faithfulness":  verdict.get("faithfulness", "unknown"),
            "completeness":  verdict.get("completeness", "unknown"),
            "judge":         "llm",
        }

    except Exception as e:
        print(f"   {C.YELLOW}⚠  LLM judge error: {e}. Falling back to rule-based.{C.RESET}")
        return {"passed": False, "score": 0.0, "reasoning": f"LLM judge failed: {e}", "judge": "fallback"}

# ==============================================================================
# RULE-BASED SCORING FUNCTIONS
# ==============================================================================

def score_factual(result: dict, question: dict) -> dict:
    answer          = result.get("answer", "")
    sources         = result.get("sources", [])
    refused         = result.get("refused", False)
    expected_answer = question.get("expected_answer", "")
    expected_source = question.get("source", "")

    answer_present   = bool(answer.strip()) and not refused
    citation_present = len(sources) > 0
    cited_sources    = [s.get("source", "") for s in sources]
    source_match     = any(expected_source in s or s in expected_source for s in cited_sources) \
                       if expected_source else False
    keyword_match    = _keyword_overlap(answer, expected_answer)
    passed           = answer_present and citation_present and keyword_match

    return {
        "passed":           passed,
        "answer_present":   answer_present,
        "citation_present": citation_present,
        "source_match":     source_match,
        "keyword_match":    keyword_match,
        "answer_returned":  answer[:200] + ("…" if len(answer) > 200 else ""),
        "judge":            "rule-based",
    }


def score_cross_reference(result: dict, question: dict) -> dict:
    base    = score_factual(result, question)
    sources = result.get("sources", [])
    multi_source     = len(set(s.get("source", "") for s in sources)) >= 1
    base["multi_source"] = multi_source
    base["passed"]       = base["passed"] and multi_source
    return base


def score_out_of_scope(result: dict, question: dict) -> dict:
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
    phrase_match      = any(p in answer for p in refusal_phrases)
    correctly_refused = refused or phrase_match

    return {
        "passed":            correctly_refused,
        "correctly_refused": correctly_refused,
        "refused_flag":      refused,
        "phrase_match":      phrase_match,
        "answer_returned":   answer[:200],
        "judge":             "rule-based",
    }


def score_ambiguous(result: dict, question: dict) -> dict:
    answer = result.get("answer", "").lower()

    clarification_phrases = [
        "could you clarify", "could you specify", "which",
        "can you be more specific", "please specify", "are you referring to",
    ]
    ambiguity_phrases = [
        "several", "multiple", "various", "different sections",
        "could refer to", "depends on", "it's unclear", "ambiguous", "broad",
    ]

    asked_for_clarification = any(p in answer for p in clarification_phrases)
    acknowledged_ambiguity  = any(p in answer for p in ambiguity_phrases)
    handled_well            = asked_for_clarification or acknowledged_ambiguity

    return {
        "passed":                  handled_well,
        "asked_for_clarification": asked_for_clarification,
        "acknowledged_ambiguity":  acknowledged_ambiguity,
        "answer_returned":         answer[:200],
        "judge":                   "rule-based",
    }


# ── keyword helper ─────────────────────────────────────────────────────────────

def _keyword_overlap(answer: str, expected: str, threshold: float = 0.35) -> bool:
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


def run_evaluation(
    pipeline_fn,
    questions: list,
    category_filter: str = None,
    use_llm_judge: bool = False,
) -> dict:
    results_by_category = {}
    all_results         = []

    questions_to_run = [
        q for q in questions
        if category_filter is None or q.get("category") == category_filter
    ]

    judge_label = f"{C.MAGENTA}[LLM Judge]{C.RESET}" if use_llm_judge else f"{C.DIM}[Rule-Based]{C.RESET}"

    print(f"\n{C.BOLD}{C.BLUE}{'='*65}{C.RESET}")
    print(f"{C.BOLD}  RAG Evaluation  —  {len(questions_to_run)} questions  {judge_label}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'='*65}{C.RESET}\n")

    for i, question in enumerate(questions_to_run, 1):
        q_text   = question.get("question", "")
        category = question.get("category", "unknown")
        scorer   = SCORERS.get(category, score_factual)

        print(f"{C.DIM}[{i}/{len(questions_to_run)}]{C.RESET} "
              f"{C.CYAN}{category.upper():15s}{C.RESET} {q_text[:65]}")

        # ── run pipeline ──────────────────────────────────────────────────────
        start = time.time()
        try:
            result = pipeline_fn(q_text)
        except Exception as e:
            result = {"answer": "", "sources": [], "refused": False, "error": str(e)}
        elapsed = time.time() - start

        # ── score: LLM judge or rule-based ────────────────────────────────────
        if use_llm_judge:
            score = llm_judge(question, result)
            # fill in fields the report expects
            score.setdefault("answer_returned", result.get("answer", "")[:200])
            score.setdefault("citation_present", len(result.get("sources", [])) > 0)
        else:
            score = scorer(result, question)

        score["latency_s"] = round(elapsed, 2)

        # ── display ───────────────────────────────────────────────────────────
        status = f"{C.GREEN}PASS ✓{C.RESET}" if score["passed"] else f"{C.RED}FAIL ✗{C.RESET}"
        extra  = ""
        if use_llm_judge:
            llm_score = score.get("score", 0.0)
            extra = f"  score={llm_score:.2f}  [{score.get('faithfulness','?')} faithfulness]"
        print(f"   {status}  ({elapsed:.1f}s){extra}")

        if not score["passed"]:
            reason = score.get("reasoning") or score.get("answer_returned", "")
            print(f"   {C.DIM}→ {reason[:90]}{C.RESET}")

        # ── collect ───────────────────────────────────────────────────────────
        entry = {
            "question":   q_text,
            "category":   category,
            "score":      score,
            "raw_result": {
                "answer":  result.get("answer", "")[:300],
                "sources": result.get("sources", []),
                "refused": result.get("refused", False),
            },
        }
        all_results.append(entry)
        results_by_category.setdefault(category, []).append(entry)

    return _build_report(results_by_category, all_results, use_llm_judge)


def _build_report(results_by_category: dict, all_results: list, use_llm_judge: bool) -> dict:
    summary = {}
    for cat, entries in results_by_category.items():
        passed  = sum(1 for e in entries if e["score"]["passed"])
        total   = len(entries)
        avg_lat = sum(e["score"].get("latency_s", 0) for e in entries) / total if total else 0
        avg_llm_score = (
            sum(e["score"].get("score", 0.0) for e in entries) / total
            if use_llm_judge and total else None
        )
        summary[cat] = {
            "passed":         passed,
            "total":          total,
            "accuracy":       round(passed / total, 3) if total else 0,
            "avg_latency_s":  round(avg_lat, 2),
        }
        if avg_llm_score is not None:
            summary[cat]["avg_llm_score"] = round(avg_llm_score, 3)

    total_pass = sum(1 for e in all_results if e["score"]["passed"])
    total_all  = len(all_results)

    return {
        "generated_at": datetime.now().isoformat(),
        "scoring_mode": "llm-judge" if use_llm_judge else "rule-based",
        "overall": {
            "passed":   total_pass,
            "total":    total_all,
            "accuracy": round(total_pass / total_all, 3) if total_all else 0,
        },
        "by_category": summary,
        "details":     all_results,
    }

# ==============================================================================
# PRETTY PRINT REPORT
# ==============================================================================

def print_report(report: dict):
    overall      = report["overall"]
    by_cat       = report["by_category"]
    scoring_mode = report.get("scoring_mode", "rule-based")

    mode_label = (
        f"{C.MAGENTA}{C.BOLD}LLM-as-Judge (gpt-4o-mini){C.RESET}"
        if scoring_mode == "llm-judge"
        else f"{C.DIM}Rule-Based (keyword/phrase matching){C.RESET}"
    )

    print(f"\n{C.BOLD}{C.BLUE}{'='*65}{C.RESET}")
    print(f"{C.BOLD}  EVALUATION REPORT  —  {mode_label}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'='*65}{C.RESET}")

    acc = overall["accuracy"]
    col = C.GREEN if acc >= 0.8 else C.YELLOW if acc >= 0.6 else C.RED
    print(f"\n  Overall Accuracy: {col}{C.BOLD}{acc*100:.1f}%{C.RESET}  "
          f"({overall['passed']}/{overall['total']} passed)\n")

    header_extra = "  Avg LLM Score" if scoring_mode == "llm-judge" else ""
    print(f"  {'Category':<20} {'Pass':>5} {'Total':>6} {'Accuracy':>10} {'Avg Lat':>9}{header_extra}")
    print(f"  {'-'*65}")

    targets = {
        "factual":         0.80,
        "cross-reference": 0.70,
        "out-of-scope":    1.00,
        "ambiguous":       0.50,
    }

    for cat, stats in by_cat.items():
        a    = stats["accuracy"]
        tgt  = targets.get(cat, 0.75)
        col  = C.GREEN if a >= tgt else C.YELLOW if a >= tgt * 0.75 else C.RED
        bar  = "█" * int(a * 10) + "░" * (10 - int(a * 10))
        llm_col = ""
        if scoring_mode == "llm-judge" and "avg_llm_score" in stats:
            ls = stats["avg_llm_score"]
            llm_col = f"  {C.MAGENTA}{ls:.2f}{C.RESET}"
        print(f"  {cat:<20} {stats['passed']:>5} {stats['total']:>6}  "
              f"  {col}{a*100:5.1f}%{C.RESET}  {bar}  {stats['avg_latency_s']:>5.1f}s{llm_col}")

    print(f"\n  {C.DIM}Targets: factual ≥80%  cross-ref ≥70%  out-of-scope 100%  ambiguous ≥50%{C.RESET}")

    if scoring_mode == "llm-judge":
        print(f"\n  {C.MAGENTA}ℹ  LLM judge also recorded faithfulness & completeness per question")
        print(f"     See data/eval_report.json → details[].score for full breakdown.{C.RESET}")

    print(f"\n  Report saved → data/eval_report.json\n")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py                   # rule-based scoring
  python scripts/evaluate.py --mock            # mock pipeline, no API key
  python scripts/evaluate.py --LLM             # LLM-as-judge (needs OPENAI_API_KEY)
  python scripts/evaluate.py --mock --LLM      # mock pipeline + LLM judge
  python scripts/evaluate.py --category factual --LLM
        """
    )
    parser.add_argument(
        "--category",
        choices=["factual", "cross-reference", "out-of-scope", "ambiguous"],
        help="Evaluate only one category",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock pipeline (no API key needed for retrieval/generation)",
    )
    parser.add_argument(
        "--LLM",
        action="store_true",
        help="Use LLM-as-judge (gpt-4o-mini) instead of rule-based scoring. Requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--questions",
        default=str(QUESTIONS),
        help=f"Path to questions.json (default: {QUESTIONS})",
    )
    args = parser.parse_args()

    # ── load questions ─────────────────────────────────────────────────────────
    q_path = Path(args.questions)
    if not q_path.exists():
        print(f"{C.RED}✗  questions.json not found at {q_path}{C.RESET}")
        return

    with open(q_path) as f:
        questions = json.load(f)
    print(f"{C.GREEN}✓  Loaded {len(questions)} questions from {q_path}{C.RESET}")

    # ── validate --LLM flag requires API key ──────────────────────────────────
    if args.LLM:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print(f"{C.RED}✗  --LLM flag requires OPENAI_API_KEY to be set in your environment.{C.RESET}")
            print(f"   export OPENAI_API_KEY=sk-...")
            return
        print(f"{C.MAGENTA}✓  LLM-as-judge mode enabled (gpt-4o-mini).{C.RESET}")
    else:
        print(f"{C.DIM}ℹ  Rule-based scoring. Pass --LLM to enable LLM-as-judge.{C.RESET}")

    # ── load pipeline ──────────────────────────────────────────────────────────
    if args.mock:
        print(f"{C.YELLOW}⚠  --mock flag set. Using mock pipeline.{C.RESET}")
        pipeline_fn = mock_rag_pipeline
    else:
        pipeline_fn = load_real_pipeline()

    # ── run ────────────────────────────────────────────────────────────────────
    report = run_evaluation(
        pipeline_fn,
        questions,
        category_filter=args.category,
        use_llm_judge=args.LLM,
    )

    # ── print & save ───────────────────────────────────────────────────────────
    print_report(report)

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
