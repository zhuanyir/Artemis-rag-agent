"""
app/mcp_server.py — The MCP Server for the Artemis II Agentic AI system.

PURPOSE:
    This is the "toolbox" that all agents can reach into.
    Instead of each agent calling APIs directly, they ask the MCP server.
    The MCP server is the single gatekeeper between the LLM agents
    and the outside world (internet, files, database).

    Think of it like a restaurant kitchen:
    - Agents are waiters who take orders
    - MCP Server is the kitchen that actually does the cooking
    - Tools are the individual chefs (web search, file writer, DB updater)

WHY MCP (Model Context Protocol)?
    Before MCP, every AI tool was a one-off custom integration (N x M problem).
    Each LLM × each tool needed separate glue code. Anthropic open-sourced MCP
    in 2024 to standardise this: one server, any LLM can connect via JSON-RPC.

PROJECT STEPS COVERED IN THIS FILE:
    ✅ Step 1  — web_search tool         (live internet lookup via DuckDuckGo)
    ✅ Step 1b — fetch_webpage tool       (fetch & extract FULL content of a URL)
    ✅ Step 1c — fetch_wikipedia_page     (fetch FULL Wikipedia article content)
    ✅ Step 2  — create_markdown_report   (save agent output to disk)
    ✅ Step 3  — add_to_database tool     (add new facts to corpus at runtime)
    ✅ Step 4  — load_pdf_to_database     (parse a PDF and add it to corpus)

KEY FIX vs PREVIOUS VERSION:
    The old code only used DuckDuckGo snippets (1-2 sentence previews) and
    Wikipedia summary sentences — it NEVER opened the actual URLs.
    This meant Agent 2 was fact-checking with incomplete, shallow data.

    This version adds:
      • fetch_webpage(url)         — opens any URL, strips HTML noise,
                                     returns up to max_chars of real content
      • fetch_wikipedia_page(topic) — fetches the FULL Wikipedia article body
                                     (not just the 5-sentence intro summary)

    Agent 2 now follows a 3-step external research process:
      1. web_search()            → find the best URLs
      2. fetch_webpage(url)      → read the actual content of top results
      3. fetch_wikipedia_page()  → get the full structured article

FOLDER STRUCTURE:
    project-root/
    ├── app/
    │   ├── mcp_server.py   ← THIS FILE
    │   ├── agents.py
    │   ├── main.py
    │   ├── retriever.py
    │   ├── generator.py
    │   └── app.py
    ├── scripts/
    │   ├── chunk.py
    │   ├── embed.py
    │   └── evaluate.py
    └── data/
        ├── corpus.json
        ├── chunks.json
        └── index.faiss

HOW TO RUN:
    pip install fastmcp duckduckgo-search pymupdf wikipedia-api requests beautifulsoup4
    python app/mcp_server.py
    (keep it running — agents.py connects to it)
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ── fastMCP import ────────────────────────────────────────────────────────────
try:
    from fastmcp import FastMCP
except ImportError:
    print("Install fastmcp:  pip install fastmcp")
    sys.exit(1)

# ── Paths (relative to project root) ─────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus.json"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ── Create the MCP server instance ───────────────────────────────────────────
mcp = FastMCP("ArtemisRAG-MCP-Server")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1a — TOOL: web_search
# PURPOSE: Search the internet and return a list of relevant URLs + snippets.
#          This is STEP 1 of Agent 2's research — it finds WHICH pages to read.
#          Agent 2 then calls fetch_webpage() on the top URLs to read the
#          ACTUAL content (not just the snippet preview).
# API USED: DuckDuckGo (free, no API key needed)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the live internet for up-to-date information.

    WHY THIS EXISTS:
        Your corpus may be outdated. This tool finds which web pages are
        relevant to a query. The URLs it returns should then be passed to
        fetch_webpage() to retrieve the full page content.

    HOW IT WORKS:
        DuckDuckGo search → returns title + URL + snippet for each result.
        NOTE: Snippets are only 1-2 sentence previews. Always follow up
        with fetch_webpage(url) to get the full content of important URLs.

    Args:
        query:       The search query string (e.g. "Artemis II launch date 2026")
        max_results: How many results to return (default 5, max 10)

    Returns:
        A formatted string with title, URL, and snippet for each result.
        Use the URLs with fetch_webpage() to get full page content.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "ERROR: Install duckduckgo-search:  pip install duckduckgo-search"

    max_results = min(max_results, 10)

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"TITLE:   {r.get('title', 'N/A')}\n"
                    f"URL:     {r.get('href', 'N/A')}\n"
                    f"SNIPPET: {r.get('body', 'N/A')}\n"
                )

        if not results:
            return f"No results found for query: '{query}'"

        header = (
            f"=== Web Search Results for: '{query}' ({len(results)} results) ===\n"
            f"⚠️  These are SNIPPETS only. Call fetch_webpage(url) on the most\n"
            f"    relevant URLs below to read the actual full page content.\n\n"
        )
        return header + "\n---\n".join(results)

    except Exception as e:
        return f"Web search failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1b — TOOL: fetch_webpage  ← NEW / KEY FIX
# PURPOSE: Open a URL and extract its full readable text content.
#          This is the missing piece from the previous version — we now
#          actually READ the pages that web_search finds, instead of only
#          using the short DuckDuckGo snippet preview.
#
# EXAMPLE USE CASE:
#   web_search("Artemis II crew 2026") returns a URL like:
#     https://en.wikipedia.org/wiki/Artemis_II
#   fetch_webpage("https://en.wikipedia.org/wiki/Artemis_II") then opens
#   that page and returns the real article text — crew names, mission details,
#   launch dates, etc. — which can be fed into the RAG corpus or used by Agent 3.
#
# LIBRARIES: requests (HTTP), beautifulsoup4 (HTML parsing / noise removal)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def fetch_webpage(url: str, max_chars: int = 4000) -> str:
    """
    Fetch the full text content of a web page by URL.

    WHY THIS EXISTS:
        web_search() only returns short snippet previews (1-2 sentences).
        To actually enrich the RAG with real information — like the full
        Artemis II Wikipedia article — we must open the URL and extract
        the complete readable text.

        Without this tool, Agent 2 was only reading DuckDuckGo previews,
        not the actual source pages. This is the core fix.

    HOW IT WORKS:
        1. requests.get(url) — downloads the raw HTML
        2. BeautifulSoup — parses and cleans the HTML
        3. Removes noise tags: <script>, <style>, <nav>, <footer>, <header>,
           <aside>, cookie banners, ads, etc.
        4. Extracts clean paragraph text
        5. Returns up to max_chars characters of readable content

    TYPICAL USE:
        urls = web_search("Artemis II crew")   # get URLs
        content = fetch_webpage(urls[0])        # read the best one

    Args:
        url:       The full URL to fetch (e.g. "https://en.wikipedia.org/wiki/Artemis_II")
        max_chars: Max characters of text to return (default 4000, ~800 words)

    Returns:
        Cleaned readable text from the page, or an error message.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return "ERROR: Install dependencies:  pip install requests beautifulsoup4"

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; ArtemisRAG-Agent/1.0; "
                "+https://github.com/artemis-rag)"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # ── Remove noise elements ─────────────────────────────────────────────
        # These tags add no readable content but inflate the text
        noise_tags = [
            "script", "style", "nav", "footer", "header", "aside",
            "noscript", "iframe", "form", "button", "input", "meta",
            "link", "figure", "figcaption", "table",   # tables often garble text
        ]
        for tag in soup(noise_tags):
            tag.decompose()

        # ── Also remove common class-based noise (ads, banners, menus) ───────
        noise_classes = [
            "cookie", "banner", "advertisement", "ad-", "sidebar",
            "menu", "breadcrumb", "social", "share", "related",
            "comment", "footer", "header", "navigation",
        ]
        for element in soup.find_all(True):
            classes = " ".join(element.get("class", [])).lower()
            el_id   = (element.get("id") or "").lower()
            if any(nc in classes or nc in el_id for nc in noise_classes):
                element.decompose()

        # ── Extract clean paragraphs ──────────────────────────────────────────
        # Prefer <p> tags for structured pages (Wikipedia, NASA, news sites)
        paragraphs = soup.find_all("p")
        if paragraphs:
            text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        else:
            # Fallback: get all text if no <p> tags found
            text = soup.get_text(separator="\n", strip=True)

        # ── Clean up whitespace ───────────────────────────────────────────────
        text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excessive blank lines
        text = re.sub(r" {2,}", " ", text)        # collapse multiple spaces
        text = text.strip()

        if not text:
            return f"No readable text content found at: {url}"

        # ── Truncate to max_chars ─────────────────────────────────────────────
        truncated = text[:max_chars]
        suffix = f"\n\n[... content truncated at {max_chars} chars. Full page has ~{len(text)} chars]" \
                 if len(text) > max_chars else ""

        return (
            f"=== Page Content: {url} ===\n\n"
            f"{truncated}"
            f"{suffix}"
        )

    except requests.exceptions.Timeout:
        return f"ERROR: Timeout fetching {url} (>15s). Try a different URL."
    except requests.exceptions.HTTPError as e:
        return f"ERROR: HTTP {e.response.status_code} fetching {url}"
    except requests.exceptions.ConnectionError:
        return f"ERROR: Could not connect to {url}. Check network or try another URL."
    except Exception as e:
        return f"ERROR fetching {url}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1c — TOOL: fetch_wikipedia_page  ← NEW / KEY FIX
# PURPOSE: Fetch the FULL body of a Wikipedia article — not just the 5-sentence
#          intro summary that the old search_wikipedia() returned.
#
# WHY REPLACE search_wikipedia()?
#   The old tool used wikipedia-api's page.summary which gives only the
#   introduction paragraph. For Artemis II, that's ~3 sentences.
#   The full article has: crew bios, mission timeline, spacecraft details,
#   launch history, scientific objectives — all missed by the old approach.
#
#   This tool fetches the real Wikipedia page via requests + BeautifulSoup,
#   extracts all article sections, and returns up to max_chars of content.
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def fetch_wikipedia_page(topic: str, max_chars: int = 5000) -> str:
    """
    Fetch the full content of a Wikipedia article by topic name.

    WHY THIS EXISTS:
        The previous search_wikipedia() tool only returned 5 intro sentences.
        For rich topics like Artemis II, the full article body contains:
          - Complete crew biographies
          - Detailed mission timeline and objectives
          - Spacecraft specifications
          - Launch date history and delays
          - References and external links

        This tool fetches the real Wikipedia page and extracts all of it.

    HOW IT WORKS:
        1. Constructs the Wikipedia URL: https://en.wikipedia.org/wiki/{topic}
        2. Calls fetch_webpage() to download and clean the HTML
        3. Returns up to max_chars of the article body

    Args:
        topic:     Wikipedia article title (e.g. "Artemis_II", "Victor_Glover")
                   Use underscores for spaces (Wikipedia URL format)
        max_chars: Max characters to return (default 5000, ~1000 words)

    Returns:
        Full Wikipedia article text, or an error if not found.
    """
    # Normalise topic → Wikipedia URL format (spaces → underscores)
    topic_clean = topic.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{topic_clean}"

    print(f"[MCP] fetch_wikipedia_page → {url}")
    result = fetch_webpage(url, max_chars=max_chars)

    # If direct title lookup fails, try a search redirect
    if result.startswith("ERROR"):
        # Try Wikipedia's search endpoint as fallback
        search_url = f"https://en.wikipedia.org/w/index.php?search={topic_clean}&ns0=1"
        result = fetch_webpage(search_url, max_chars=max_chars)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TOOL: create_markdown_report
# PURPOSE: Save the agent's synthesized answer to a real .md file on disk.
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def create_markdown_report(
    title: str,
    content: str,
    filename: str = "",
) -> str:
    """
    Save the agent's final synthesized answer as a Markdown file.

    WHY THIS EXISTS:
        Agents produce rich answers combining internal corpus + web results.
        This tool persists that answer as a .md file so the human can
        read, share, or archive it.

    Args:
        title:    The report title (shown as H1 heading in markdown)
        content:  The full report body (markdown-formatted text)
        filename: Optional filename without extension (auto-generated if empty)

    Returns:
        The full path where the file was saved.
    """
    if not filename:
        safe = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"{safe}_{timestamp}"

    filepath = REPORTS_DIR / f"{filename}.md"

    report_content = f"""# {title}

*Generated by Artemis II Agentic AI System*
*Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

---

{content}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[MCP] Report saved → {filepath}")
    return f"Report saved successfully to: {filepath}"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TOOL: add_to_database
# PURPOSE: Let the agent update the knowledge base at runtime with new content.
#
# NEW IN THIS VERSION:
#   Agent 2 now fetches full web page content via fetch_webpage(). That rich
#   content should be passed here to be chunked and stored in corpus.json,
#   making it available to future queries without re-fetching the web.
#
# CHUNKING FORMULA:
#   chunk_size = 500 chars, overlap = 100 chars, step = 400 chars
#   positions:  0, 400, 800, 1200, ...
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def add_to_database(
    text: str,
    source: str = "agent_added",
    page: int = 0,
) -> str:
    """
    Add a new piece of information to the corpus (knowledge base).

    WHY THIS EXISTS:
        After Agent 2 fetches live web content (e.g. the full Artemis II
        Wikipedia article), that content can be added here so the RAG
        corpus stays up to date for future queries.

    HOW IT WORKS:
        1. Reads current corpus.json
        2. Splits new text into 500-char chunks with 100-char overlap
        3. Appends each chunk with source/page metadata
        4. Writes corpus.json back to disk
        ⚠️  Re-run scripts/embed.py to update the FAISS index.

    Args:
        text:   The new text content to add
        source: Label for where this came from (e.g. "wikipedia_artemis_ii_2026")
        page:   Page number (0 for web sources)

    Returns:
        Confirmation message with chunk count.
    """
    if not text or not text.strip():
        return "ERROR: text cannot be empty."

    if CORPUS_PATH.exists():
        with open(CORPUS_PATH, encoding="utf-8") as f:
            corpus = json.load(f)
    else:
        corpus = []

    chunk_size = 500
    overlap    = 100
    step       = chunk_size - overlap   # 400

    new_chunks = []
    start = 0
    chunk_index = len(corpus)

    while start < len(text):
        chunk_text = text[start : start + chunk_size].strip()
        if chunk_text:
            new_chunks.append({
                "chunk_id":   f"agent_{chunk_index:04d}",
                "source":     source,
                "page":       page,
                "char_count": len(chunk_text),
                "text":       chunk_text,
                "added_by":   "agent",
                "added_at":   datetime.now().isoformat(),
            })
            chunk_index += 1
        start += step

    corpus.extend(new_chunks)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    msg = (
        f"✅ Added {len(new_chunks)} chunk(s) from source '{source}' to corpus.json.\n"
        f"⚠️  Run scripts/embed.py to update the FAISS index with these new chunks."
    )
    print(f"[MCP] {msg}")
    return msg


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TOOL: load_pdf_to_database
# PURPOSE: Parse a PDF file and add all its text to the corpus.
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def load_pdf_to_database(pdf_path: str) -> str:
    """
    Parse a PDF file and add all its text to the corpus database.

    WHY THIS EXISTS:
        Instead of manually running extract.py for every new document,
        an agent can call this tool directly with a file path.

    HOW IT WORKS:
        1. Open PDF with PyMuPDF (fitz)
        2. Extract text from each page
        3. Call add_to_database() for each page's text
        ⚠️  Re-run scripts/embed.py after this to update FAISS.

    Args:
        pdf_path: Absolute or relative path to the PDF file

    Returns:
        Summary of pages processed and chunks added.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "ERROR: Install PyMuPDF:  pip install pymupdf"

    pdf_path = os.path.expanduser(pdf_path)

    if not os.path.exists(pdf_path):
        return f"ERROR: PDF file not found at: {pdf_path}"

    pdf_name = os.path.basename(pdf_path)
    pages_processed = 0
    total_chunks = 0

    try:
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if not text or len(text) < 50:
                continue

            result = add_to_database(
                text=text,
                source=pdf_name,
                page=page_num + 1,
            )
            pages_processed += 1

            if "Added" in result:
                try:
                    n = int(result.split("Added")[1].strip().split(" ")[0])
                    total_chunks += n
                except Exception:
                    pass

        doc.close()

        return (
            f"✅ PDF '{pdf_name}' processed: {pages_processed} pages → "
            f"~{total_chunks} chunks added to corpus.json.\n"
            f"⚠️  Run scripts/embed.py to update the FAISS index."
        )

    except Exception as e:
        return f"ERROR processing PDF '{pdf_path}': {e}"


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Artemis II MCP Server starting...")
    print("  Tools available:")
    print("    ✅ web_search             (Step 1a — find URLs + snippets)")
    print("    ✅ fetch_webpage          (Step 1b — read full page content) [NEW]")
    print("    ✅ fetch_wikipedia_page   (Step 1c — full Wikipedia article) [NEW]")
    print("    ✅ create_markdown_report (Step 2 — save report to disk)")
    print("    ✅ add_to_database        (Step 3 — update corpus at runtime)")
    print("    ✅ load_pdf_to_database   (Step 4 — load PDF into corpus)")
    print("=" * 60)
    mcp.run()
