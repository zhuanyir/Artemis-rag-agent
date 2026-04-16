"""
app/agents.py — The Multi-Agent Team for the Artemis II Agentic AI system.

PURPOSE:
    This file builds the three-agent pipeline shown in the workshop slides:

        User Query
            ↓
        Agent 1 (Internal Researcher)   ← searches your FAISS corpus
            ↓  query + internal answer
        Agent 2 (External Fact-Checker) ← uses MCP tools (web + full page fetch)
            ↓  query + internal + external
        Agent 3 (Synthesizer)           ← writes the final combined answer
            ↓
        User Gets Response + Markdown Report saved to disk

WHY THREE AGENTS INSTEAD OF ONE?
    A single agent trying to do everything gets confused and makes mistakes.
    The slides say: "The key is to decrease the amount of information an agent
    needs to process." Each agent has ONE job:
      - Agent 1: only looks at your documents
      - Agent 2: only looks at the internet (and fetches real page content)
      - Agent 3: only synthesizes what the other two found

KEY FIX vs PREVIOUS VERSION:
    Agent 2 previously only read DuckDuckGo snippets (1-2 sentence previews)
    and a 5-sentence Wikipedia summary. It never actually opened the URLs.

    Agent 2 now runs a proper 3-step external research loop:
      STEP A: web_search(query)
              → DuckDuckGo returns list of relevant URLs + short snippets.
                This tells us WHAT pages exist, but not their full content.

      STEP B: fetch_webpage(url)   ← THE MISSING STEP in the old version
              → Opens each top URL (e.g. https://en.wikipedia.org/wiki/Artemis_II)
                and extracts the full readable text from the page.
                This is the real content that enriches the RAG answer.

      STEP C: fetch_wikipedia_page(topic)
              → Fetches the complete Wikipedia article body for the topic
                (not just the 5-sentence intro — full crew details, mission
                timeline, spacecraft specs, launch history, etc.)

    The full page content from steps B and C is then:
      a) Fed to Agent 3 to produce a much richer synthesized answer
      b) Optionally saved to corpus via add_to_database() to keep RAG fresh

HOW TO RUN:
    Terminal 1:  python app/mcp_server.py     (keep running)
    Terminal 2:  python app/agents.py
    Or via:      python app/main.py           (full interactive loop)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from visualizer_agent import maybe_visualize

load_dotenv()

# ── Add app/ to path so we can import retriever/generator ────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from retriever import load_index_and_chunks, retrieve
from generator import build_context

# ── OpenAI client (shared across all agents) ─────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Cost tracker ─────────────────────────────────────────────────────────────
COST_FILE = Path(__file__).parent.parent / "cost_tracker.json"


def _track(response, label: str = "agent") -> None:
    """Log token cost for a single API call."""
    u = response.usage
    cost = (u.prompt_tokens * 0.15 + u.completion_tokens * 0.60) / 1_000_000

    data = {"total": 0.0, "calls": 0}
    if COST_FILE.exists():
        with open(COST_FILE) as f:
            data = json.load(f)
    data["total"] = round(data.get("total", 0) + cost, 6)
    data["calls"] = data.get("calls", 0) + 1
    with open(COST_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(
        f"[cost][{label}] ${cost:.6f} | "
        f"Total: ${data['total']:.4f} / $5.00 ({data['calls']} calls)"
    )


# ── MCP Tool caller ───────────────────────────────────────────────────────────
def _call_mcp_tool(tool_name: str, **kwargs) -> str:
    """
    Call an MCP tool function directly by importing from mcp_server.

    WHY DIRECT IMPORT INSTEAD OF JSON-RPC?
        The full MCP protocol uses JSON-RPC over stdio. For the hackathon,
        importing the functions directly is simpler — same tool logic, no
        network overhead. Swap for a proper MCP client in production.
    """
    try:
        import importlib.util
        server_path = Path(__file__).parent / "mcp_server.py"
        spec = importlib.util.spec_from_file_location("mcp_server", server_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tool_fn = getattr(mod, tool_name)
        return tool_fn(**kwargs)
    except Exception as e:
        return f"[MCP ERROR calling {tool_name}]: {e}"


# ── URL extractor helper ──────────────────────────────────────────────────────
def _extract_urls_from_search(search_result: str, max_urls: int = 3) -> list[str]:
    """
    Parse URLs from the formatted web_search() output string.

    The web_search tool returns blocks like:
        TITLE:   Some Title
        URL:     https://example.com/page
        SNIPPET: Short preview text...

    This helper extracts up to max_urls from that formatted string.

    Args:
        search_result: The raw string returned by web_search()
        max_urls:      Maximum number of URLs to extract (default 3)

    Returns:
        List of URL strings extracted from the search results.
    """
    import re
    urls = re.findall(r"URL:\s+(https?://[^\s\n]+)", search_result)
    # Skip Wikipedia redirect/search pages — prefer direct article URLs
    direct = [u for u in urls if "/w/index.php" not in u and "Special:" not in u]
    return direct[:max_urls]


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 1 — Internal Researcher
#  PURPOSE: Search your FAISS corpus for relevant information.
#  INPUT:   User question
#  OUTPUT:  Internal answer + which chunks were used
# ══════════════════════════════════════════════════════════════════════════════

print("[agents] Loading FAISS index for Agent 1...")
try:
    _faiss_index, _faiss_chunks = load_index_and_chunks()
    print(f"[agents] Index loaded: {_faiss_index.ntotal} vectors ready.")
except FileNotFoundError as e:
    print(f"[agents] WARNING: {e}")
    _faiss_index, _faiss_chunks = None, []


AGENT1_SYSTEM = """You are the Internal Researcher for the Artemis II mission.
Your ONLY job is to answer questions using the provided corpus chunks.
You have access to official NASA documents about Artemis II.

Rules:
- Answer ONLY from the provided context chunks.
- If the context doesn't answer the question, say exactly:
  "INTERNAL: No information found in corpus."
- Always cite: (Source: filename, page X)
- Be factual and concise. No opinions.
- Do NOT use any knowledge outside the provided context."""


def agent1_internal_researcher(query: str) -> dict[str, Any]:
    """
    Agent 1: Internal Researcher.

    WHAT IT DOES:
        1. Calls FAISS retrieve() to find the top-5 relevant chunks
        2. Sends those chunks + the question to GPT-4o-mini
        3. Returns the internal answer + retrieved chunks for Agent 2 to use

    Args:
        query: The user's original question

    Returns:
        dict with keys: "answer", "chunks", "context"
    """
    print(f"\n[Agent 1] 🔍 Searching internal corpus for: '{query}'")

    if _faiss_index is None:
        return {
            "answer":  "INTERNAL: FAISS index not loaded. Run scripts/embed.py first.",
            "chunks":  [],
            "context": "",
        }

    chunks = retrieve(
        query=query,
        index=_faiss_index,
        chunks=_faiss_chunks,
        final_k=5,
        candidate_k=10,
        use_rerank=True,
    )

    if not chunks:
        return {
            "answer":  "INTERNAL: No information found in corpus.",
            "chunks":  [],
            "context": "",
        }

    context = build_context(chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": AGENT1_SYSTEM},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0,
        max_tokens=400,
    )
    _track(response, "Agent1-InternalResearcher")

    answer = response.choices[0].message.content.strip()
    print(f"[Agent 1] ✅ Internal answer ready ({len(chunks)} chunks used).")

    return {"answer": answer, "chunks": chunks, "context": context}


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 2 — External Fact-Checker
#  PURPOSE: Verify and enrich Agent 1's answer with REAL live web content.
#  INPUT:   Original query + Agent 1's internal answer
#  OUTPUT:  External findings built from full page content (not snippets)
#
#  MCP TOOLS USED (in order):
#    Step A: web_search()           — find relevant URLs
#    Step B: fetch_webpage(url)     — read FULL content of top URLs  ← NEW
#    Step C: fetch_wikipedia_page() — read FULL Wikipedia article    ← NEW
#    Step D: add_to_database()      — store new content in corpus    ← NEW
#
#  THE KEY FIX:
#    Old version: only read DuckDuckGo snippets + 5-sentence Wikipedia intro
#    New version: actually opens and reads the full content of each URL,
#                 including the complete Wikipedia article body
# ══════════════════════════════════════════════════════════════════════════════

AGENT2_SYSTEM = """You are the External Fact-Checker for the Artemis II mission.
Your job is to verify and enrich the internal answer using FULL live web page content.

You have access to these tools (already called for you — results are below):
  - web_search()           — found relevant URLs
  - fetch_webpage(url)     — fetched FULL content from those URLs
  - fetch_wikipedia_page() — fetched FULL Wikipedia article

Process:
1. Read the internal answer and identify any claims to verify or enrich.
2. Compare against the FULL PAGE CONTENT provided (not just snippets).
3. Report: agreements, contradictions, and new information from the real pages.
4. Always cite the source URL for anything you report.

Output format:
  EXTERNAL FINDINGS:
  - [confirmed/contradicted/new fact] (Source: URL)
  - [any new information not in the internal answer] (Source: URL)

If internal answer is fully accurate and nothing new was found online, say:
  "EXTERNAL: Internal answer confirmed. No new information found online."

Do NOT invent facts. Only report what appears in the fetched page content below."""


def agent2_external_fact_checker(
    query: str,
    internal_answer: str,
    save_to_corpus: bool = True,
) -> str:
    """
    Agent 2: External Fact-Checker.

    WHAT IT DOES (NEW 3-STEP PROCESS):
        Step A — web_search(query)
                 Get a list of relevant URLs for the topic.
                 These are POINTERS to pages, not the content itself.

        Step B — fetch_webpage(url) for each top URL
                 Actually OPEN each URL and read the full page text.
                 This is the content we were missing before — e.g. the
                 full https://en.wikipedia.org/wiki/Artemis_II article
                 with crew biographies, mission timeline, launch history.

        Step C — fetch_wikipedia_page(topic)
                 Explicitly fetch the full Wikipedia article for the topic.
                 Wikipedia is the most structured and reliable free source.

        Step D — add_to_database() (optional)
                 Store the fetched web content into corpus.json so it
                 enriches future RAG queries without re-fetching.

    Args:
        query:           The user's original question
        internal_answer: The answer Agent 1 found in the corpus
        save_to_corpus:  If True, saves fetched web content to corpus.json

    Returns:
        A string of external findings grounded in full page content.
    """
    print(f"\n[Agent 2] 🌐 Starting external research for: '{query}'")

    # ── Step A: Ask LLM to generate optimal search query ─────────────────────
    search_query_prompt = (
        f"Given this question: '{query}'\n"
        f"And this internal answer: '{internal_answer[:300]}'\n\n"
        f"Write ONE concise web search query (max 8 words) to verify or "
        f"update the internal answer with current information. "
        f"Output ONLY the search query, nothing else."
    )

    sq_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": search_query_prompt}],
        temperature=0,
        max_tokens=30,
    )
    _track(sq_response, "Agent2-SearchQuery")
    search_query = sq_response.choices[0].message.content.strip().strip('"')
    print(f"[Agent 2] 🔎 Search query: '{search_query}'")

    # ── Step A: web_search — get URLs and snippets ────────────────────────────
    print(f"[Agent 2] 📡 Calling web_search...")
    web_search_results = _call_mcp_tool("web_search", query=search_query, max_results=5)

    # Extract the top URLs from the search results
    urls_to_fetch = _extract_urls_from_search(web_search_results, max_urls=3)
    print(f"[Agent 2] 🔗 Found {len(urls_to_fetch)} URLs to fetch: {urls_to_fetch}")

    # ── Step B: fetch_webpage — read FULL content of each URL ────────────────
    # This is the core fix: we now actually open and read each page.
    fetched_pages = []
    for i, url in enumerate(urls_to_fetch, 1):
        print(f"[Agent 2] 📄 Fetching page {i}/{len(urls_to_fetch)}: {url}")
        page_content = _call_mcp_tool("fetch_webpage", url=url, max_chars=3000)

        if not page_content.startswith("ERROR"):
            fetched_pages.append({
                "url":     url,
                "content": page_content,
            })
            print(f"[Agent 2] ✅ Page {i} fetched ({len(page_content)} chars)")

            # ── Step D: Save to corpus for future RAG queries ─────────────────
            if save_to_corpus:
                source_label = f"web_{url.split('//')[1].split('/')[0]}_{i}"
                save_result = _call_mcp_tool(
                    "add_to_database",
                    text=page_content,
                    source=source_label,
                    page=0,
                )
                print(f"[Agent 2] 💾 Saved to corpus: {save_result.split(chr(10))[0]}")
        else:
            print(f"[Agent 2] ⚠️  Could not fetch page {i}: {page_content[:80]}")

    # ── Step C: fetch_wikipedia_page — get FULL Wikipedia article ────────────
    # Derive a Wikipedia topic from the query (e.g. "Artemis II crew" → "Artemis_II")
    wiki_topic = query.split("?")[0].strip()
    # Try to make it a clean Wikipedia article title
    wiki_topic_clean = wiki_topic.replace(" ", "_")
    print(f"[Agent 2] 📖 Fetching Wikipedia article: '{wiki_topic_clean}'")

    wiki_content = _call_mcp_tool(
        "fetch_wikipedia_page",
        topic=wiki_topic_clean,
        max_chars=4000,
    )

    if not wiki_content.startswith("ERROR") and save_to_corpus:
        save_result = _call_mcp_tool(
            "add_to_database",
            text=wiki_content,
            source=f"wikipedia_{wiki_topic_clean}",
            page=0,
        )
        print(f"[Agent 2] 💾 Wikipedia saved to corpus: {save_result.split(chr(10))[0]}")

    # ── Build full external context for Agent 3 ───────────────────────────────
    # This now contains real full-page content, not just snippets
    pages_text = ""
    for p in fetched_pages:
        pages_text += f"\n\n--- SOURCE: {p['url']} ---\n{p['content'][:2000]}"

    external_context = (
        f"WEB SEARCH URLS FOUND:\n{web_search_results}\n\n"
        f"FULL PAGE CONTENT FETCHED FROM TOP URLS:{pages_text}\n\n"
        f"FULL WIKIPEDIA ARTICLE ({wiki_topic_clean}):\n{wiki_content[:3000]}"
    )

    # ── Ask Agent 2 LLM to synthesize findings from full content ─────────────
    print(f"[Agent 2] 🧠 Synthesizing findings from full page content...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": AGENT2_SYSTEM},
            {"role": "user", "content": (
                f"Original question: {query}\n\n"
                f"Internal answer from corpus:\n{internal_answer}\n\n"
                f"Full external content fetched from the web:\n{external_context}\n\n"
                f"Based on the FULL PAGE CONTENT above (not just snippets), "
                f"what do external sources confirm, contradict, or add to the internal answer?"
            )},
        ],
        temperature=0,
        max_tokens=600,
    )
    _track(response, "Agent2-ExternalFactChecker")

    findings = response.choices[0].message.content.strip()
    print("  [Agent 2] ✅ External findings ready (grounded in full page content).")
    return findings


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 3 — Synthesizer
#  PURPOSE: Combine internal + external findings into one coherent answer.
#  INPUT:   Query + Agent 1 output + Agent 2 output
#  OUTPUT:  Final answer + Markdown report saved to disk
# ══════════════════════════════════════════════════════════════════════════════

AGENT3_SYSTEM = """You are the Synthesizer for the Artemis II mission assistant.
You receive findings from two sources:
  1. INTERNAL: What the official NASA corpus says
  2. EXTERNAL: What live web pages say (FULL page content, not snippets)

Your job is to write ONE coherent, accurate final answer for the user.

Rules:
- Lead with what both sources agree on.
- If external sources add NEW information, include it clearly labeled:
  "🌐 Updated info: [new fact]"
- If external sources CONTRADICT the corpus, flag it:
  "⚠️ Note: The corpus says X, but recent sources suggest Y."
- If external sources confirm without adding anything new, use the internal answer.
- Always cite sources: (Source: filename, page X) for corpus, (Source: URL) for web.
- Be concise. Max 400 words.
- End with: "Report saved to: [filename]" if a report was saved."""


def agent3_synthesizer(
    query: str,
    internal_answer: str,
    external_findings: str,
    save_report: bool = True,
) -> str:
    """
    Agent 3: Synthesizer.

    WHAT IT DOES:
        1. Reads Agent 1's internal answer and Agent 2's full-page-grounded findings
        2. Writes one coherent final answer combining both
        3. Saves a Markdown report via MCP

    Args:
        query:             The user's original question
        internal_answer:   Agent 1's corpus-based answer
        external_findings: Agent 2's web-based findings (now from full pages)
        save_report:       Whether to save a .md report file

    Returns:
        The final synthesized answer string.
    """
    print(f"\n[Agent 3] ✍️  Synthesizing final answer...")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": AGENT3_SYSTEM},
            {"role": "user",   "content": (
                f"User Question: {query}\n\n"
                f"INTERNAL (corpus):\n{internal_answer}\n\n"
                f"EXTERNAL (full web page content):\n{external_findings}"
            )},
        ],
        temperature=0,
        max_tokens=500,
    )
    _track(response, "Agent3-Synthesizer")

    final_answer = response.choices[0].message.content.strip()

    if save_report:
        report_content = (
            f"## Question\n{query}\n\n"
            f"## Final Answer\n{final_answer}\n\n"
            f"---\n\n"
            f"## Internal Corpus Findings\n{internal_answer}\n\n"
            f"## External Web Findings (Full Page Content)\n{external_findings}"
        )
        save_result = _call_mcp_tool(
            "create_markdown_report",
            title=f"Artemis II: {query[:60]}",
            content=report_content,
        )
        print(f"[Agent 3] 📄 {save_result}")
        final_answer += f"\n\n*{save_result}*"

    print("[Agent 3] ✅ Final answer ready.")
    viz = maybe_visualize(query, final_answer)
    if viz:
        print(f"Visual saved: {viz['path']}")
    return final_answer



# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE — run all three agents in sequence
# ══════════════════════════════════════════════════════════════════════════════

def run_agentic_pipeline(
    query: str,
    save_report: bool = True,
    check_external: bool = True,
    save_web_to_corpus: bool = True,
) -> dict[str, Any]:
    """
    Run the full 3-agent pipeline for a user query.

    PIPELINE FLOW (slides p.21 / p.23 / p.28):
        User → Agent1 (internal FAISS)
             → Agent2 (web_search + fetch_webpage + fetch_wikipedia_page)
             → Agent3 (synthesize + save report)
             → User

    NEW PARAMETER:
        save_web_to_corpus: If True, Agent 2 saves fetched web content
                            into corpus.json via add_to_database(), keeping
                            the RAG knowledge base up to date automatically.

    Args:
        query:              The user's question
        save_report:        Save markdown report to disk (default True)
        check_external:     Run Agent 2 external check (default True)
        save_web_to_corpus: Persist fetched web content to corpus (default True)

    Returns:
        dict with: "final_answer", "internal_answer", "external_findings", "chunks_used"
    """
    print("\n" + "=" * 60)
    print(f"  AGENTIC PIPELINE  |  Query: {query[:60]}")
    print("=" * 60)

    # ── Agent 1: Internal Researcher ──────────────────────────────────────────
    agent1_result   = agent1_internal_researcher(query)
    internal_answer = agent1_result["answer"]

    # ── Agent 2: External Fact-Checker (now fetches full page content) ────────
    if check_external:
        external_findings = agent2_external_fact_checker(
            query,
            internal_answer,
            save_to_corpus=save_web_to_corpus,
        )
    else:
        external_findings = "External check skipped (check_external=False)."
        print("[Pipeline] Agent 2 skipped.")

    # ── Agent 3: Synthesizer ──────────────────────────────────────────────────
    final_answer = agent3_synthesizer(
        query=query,
        internal_answer=internal_answer,
        external_findings=external_findings,
        save_report=save_report,
    )

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    return {
        "final_answer":      final_answer,
        "internal_answer":   internal_answer,
        "external_findings": external_findings,
        "chunks_used":       agent1_result["chunks"],
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = "Who are the Artemis II crew members and what is their background?"
    result = run_agentic_pipeline(
        test_query,
        save_report=True,
        check_external=True,
        save_web_to_corpus=True,
    )
    print("\n" + "━" * 60)
    print("FINAL ANSWER:")
    print("━" * 60)
    print(result["final_answer"])
