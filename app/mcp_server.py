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

    HOW IT WORKS:
        Primary:  DuckDuckGo (DDGS) → title + URL + snippet per result.
        Fallback: Wikipedia OpenSearch API → used if DuckDuckGo fails or
                  returns no results (rate-limit, network block, etc.)

    Args:
        query:       The search query string (e.g. "Artemis II launch date 2026")
        max_results: How many results to return (default 5, max 10)

    Returns:
        A formatted string with title, URL, and snippet for each result.
        Use the URLs with fetch_webpage() to get full page content.
    """
    import requests as _req

    max_results = min(max_results, 10)
    ddgs_error  = None

    # ── Primary: DuckDuckGo ───────────────────────────────────────────────────
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"TITLE:   {r.get('title', 'N/A')}\n"
                    f"URL:     {r.get('href', 'N/A')}\n"
                    f"SNIPPET: {r.get('body', 'N/A')}\n"
                )
        if results:
            header = (
                f"=== Web Search Results for: '{query}' ({len(results)} results) ===\n"
                f"⚠️  These are SNIPPETS only. Call fetch_webpage(url) on the most\n"
                f"    relevant URLs below to read the actual full page content.\n\n"
            )
            return header + "\n---\n".join(results)
        ddgs_error = "DuckDuckGo returned no results"
    except ImportError:
        ddgs_error = "duckduckgo_search not installed"
    except Exception as e:
        ddgs_error = str(e)

    print(f"[MCP] web_search: DuckDuckGo failed ({ddgs_error}), trying Wikipedia fallback...")

    # ── Fallback A: Wikipedia Full-text Search API ────────────────────────────
    try:
        resp = _req.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":   "query",
                "list":     "search",
                "srsearch": query,
                "srlimit":  max_results,
                "format":   "json",
            },
            timeout=10,
            headers={"User-Agent": "ArtemisRAG-Agent/1.0"},
        )
        resp.raise_for_status()
        data   = resp.json()
        hits   = data.get("query", {}).get("search", [])

        if hits:
            results = []
            for h in hits:
                title   = h.get("title", "N/A")
                snippet = re.sub(r"<[^>]+>", "", h.get("snippet", "N/A"))  # strip HTML tags
                url     = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                results.append(
                    f"TITLE:   {title}\n"
                    f"URL:     {url}\n"
                    f"SNIPPET: {snippet}\n"
                )
            header = (
                f"=== Web Search Results for: '{query}' ({len(results)} results) ===\n"
                f"[Source: Wikipedia Search — DuckDuckGo unavailable: {ddgs_error}]\n\n"
            )
            return header + "\n---\n".join(results)
    except Exception as e2:
        ddgs_error += f" | Wikipedia search also failed: {e2}"

    # ── Fallback B: Return a synthetic result pointing to Wikipedia directly ──
    # Even if search APIs fail, we can return a direct Wikipedia URL
    # so that fetch_wikipedia_page() can still retrieve the article.
    topic_guess = query.split()[:3]  # first 3 words as best-guess article title
    wiki_title  = "_".join(w.capitalize() for w in topic_guess)
    fallback_url = f"https://en.wikipedia.org/wiki/{wiki_title}"
    return (
        f"=== Web Search (limited — {ddgs_error}) ===\n\n"
        f"TITLE:   {' '.join(topic_guess)} (Wikipedia best-guess)\n"
        f"URL:     {fallback_url}\n"
        f"SNIPPET: Direct Wikipedia URL for the topic — fetch with fetch_wikipedia_page()\n"
    )


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

    HOW IT WORKS:
        1. Wikipedia REST API /page/summary/{title} → fast intro paragraph
        2. Wikipedia REST API /page/sections/{title}  → full article sections
        3. Falls back to fetch_webpage scraping if REST API fails
        4. Falls back to Wikipedia OpenSearch redirect if title not exact

    Args:
        topic:     Wikipedia article title (e.g. "Artemis_II", "Victor_Glover")
                   Spaces or underscores both work.
        max_chars: Max characters to return (default 5000, ~1000 words)

    Returns:
        Full Wikipedia article text, or an error if not found.
    """
    import requests as _req

    # Normalise topic: spaces → underscores for URL, but also spaces for display
    topic_clean = topic.strip().replace(" ", "_")
    # Remove common question words if they crept in (e.g. "What_is_Artemis_II")
    for prefix in ["What_is_", "What_are_", "Who_are_", "How_does_", "Why_does_",
                   "When_was_", "Where_is_", "Which_", "What_"]:
        if topic_clean.startswith(prefix):
            topic_clean = topic_clean[len(prefix):]
            break

    headers = {"User-Agent": "ArtemisRAG-Agent/1.0 (educational hackathon project)"}

    print(f"[MCP] fetch_wikipedia_page → topic='{topic_clean}'")

    # ── Method 1: Wikipedia REST API summary ─────────────────────────────────
    # Gives the intro section (extract) and confirms the article exists
    try:
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic_clean}"
        r = _req.get(summary_url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            intro = data.get("extract", "")
            article_title = data.get("title", topic_clean)
            canonical_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

            # Try to get more content via the actual wiki page
            if canonical_url:
                full = fetch_webpage(canonical_url, max_chars=max_chars)
                if not full.startswith("ERROR"):
                    return full

            # Fallback: return at least the intro + whatever we have
            if intro:
                return (
                    f"=== Wikipedia: {article_title} ===\n\n"
                    f"{intro}\n\n"
                    f"[Full article: {canonical_url or 'https://en.wikipedia.org/wiki/' + topic_clean}]"
                )[:max_chars]
    except Exception as e1:
        print(f"[MCP] Wikipedia REST API failed: {e1}")

    # ── Method 2: Scrape the wiki page directly ───────────────────────────────
    url = f"https://en.wikipedia.org/wiki/{topic_clean}"
    result = fetch_webpage(url, max_chars=max_chars)
    if not result.startswith("ERROR"):
        return result

    # ── Method 3: Wikipedia OpenSearch → find closest article title ──────────
    try:
        search_resp = _req.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":   "opensearch",
                "search":   topic.replace("_", " "),
                "limit":    3,
                "format":   "json",
                "profile":  "fuzzy",
            },
            headers=headers,
            timeout=10,
        )
        search_data = search_resp.json()
        titles = search_data[1] if len(search_data) > 1 else []
        urls   = search_data[3] if len(search_data) > 3 else []
        if titles and urls:
            best_url = urls[0]
            print(f"[MCP] Wikipedia OpenSearch matched: '{titles[0]}' → {best_url}")
            result2 = fetch_webpage(best_url, max_chars=max_chars)
            if not result2.startswith("ERROR"):
                return result2
    except Exception as e3:
        print(f"[MCP] Wikipedia OpenSearch failed: {e3}")

    return f"ERROR: Could not retrieve Wikipedia article for '{topic}'. Original error: {result}"


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

    # ── Load existing chunks.json to get next chunk_id ────────────────────────
    CHUNKS_PATH = DATA_DIR / "chunks.json"
    existing_chunks: list[dict] = []
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            raw = f.read().rstrip("\x00").rstrip()  # guard against null-byte corruption
        try:
            existing_chunks = json.loads(raw)
        except json.JSONDecodeError:
            existing_chunks = []

    chunk_index = len(existing_chunks)
    new_chunks  = []
    start       = 0

    while start < len(text):
        chunk_text = text[start : start + chunk_size].strip()
        if chunk_text:
            new_chunks.append({
                "chunk_id":   chunk_index,
                "source":     source,
                "page":       page,
                "char_count": len(chunk_text),
                "text":       chunk_text,
                "added_by":   "agent",
                "added_at":   datetime.now().isoformat(),
            })
            chunk_index += 1
        start += step

    # ── Write to BOTH corpus.json (record) AND chunks.json (new format) ───────
    corpus.extend(new_chunks)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    existing_chunks.extend(new_chunks)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_chunks, f, indent=2, ensure_ascii=False)

    msg = (
        f"✅ Added {len(new_chunks)} chunk(s) from source '{source}' to corpus.json + chunks.json.\n"
        f"⚠️  Run scripts/embed.py to update the FAISS index.  (No need for chunk_corpus.py)"
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
    chunk_size = 500
    overlap    = 100
    step       = chunk_size - overlap

    try:
        doc = fitz.open(pdf_path)

        # ── Load existing chunks ONCE ─────────────────────────────────────────
        CHUNKS_PATH = DATA_DIR / "chunks.json"
        existing_chunks: list[dict] = []
        if CHUNKS_PATH.exists():
            with open(CHUNKS_PATH, encoding="utf-8") as f:
                raw = f.read().rstrip("\x00").rstrip()
            try:
                existing_chunks = json.loads(raw)
            except json.JSONDecodeError:
                existing_chunks = []

        chunk_index   = len(existing_chunks)
        new_chunks    = []
        pages_processed = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if not text or len(text) < 50:
                continue

            pages_processed += 1
            start = 0
            while start < len(text):
                chunk_text = text[start : start + chunk_size].strip()
                if chunk_text:
                    new_chunks.append({
                        "chunk_id":   chunk_index,
                        "source":     pdf_name,
                        "page":       page_num + 1,
                        "char_count": len(chunk_text),
                        "text":       chunk_text,
                        "added_by":   "load_pdf",
                        "added_at":   datetime.now().isoformat(),
                    })
                    chunk_index += 1
                start += step

        doc.close()

        # ── Single write for all pages at once ────────────────────────────────
        all_chunks = existing_chunks + new_chunks
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        # Also update corpus.json
        if CORPUS_PATH.exists():
            with open(CORPUS_PATH, encoding="utf-8") as f:
                corpus = json.load(f)
        else:
            corpus = []
        corpus.extend(new_chunks)
        with open(CORPUS_PATH, "w", encoding="utf-8") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)

        return (
            f"✅ PDF '{pdf_name}' processed: {pages_processed} pages → "
            f"{len(new_chunks)} chunks added in a single write.\n"
            f"⚠️  Run scripts/embed.py to update the FAISS index."
        )

    except Exception as e:
        return f"ERROR processing PDF '{pdf_path}': {e}"


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

MCP_HOST = "localhost"
MCP_PORT = 8080

if __name__ == "__main__":
    print("=" * 60)
    print("  Artemis II MCP Server starting...")
    print(f"  HTTP endpoint: http://{MCP_HOST}:{MCP_PORT}/mcp")
    print("  Tools available:")
    print("    ✅ web_search             (Step 1a — find URLs + snippets)")
    print("    ✅ fetch_webpage          (Step 1b — read full page content)")
    print("    ✅ fetch_wikipedia_page   (Step 1c — full Wikipedia article)")
    print("    ✅ create_markdown_report (Step 2 — save report to disk)")
    print("    ✅ add_to_database        (Step 3 — update corpus at runtime)")
    print("    ✅ load_pdf_to_database   (Step 4 — load PDF into corpus)")
    print("=" * 60)
    mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)
