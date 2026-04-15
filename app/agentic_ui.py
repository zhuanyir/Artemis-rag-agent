"""
app/agentic_ui.py — Agentic AI Web UI for the Artemis II project.
Port: 7861  (RAG stays on 7860)

KEY DESIGN DECISION — No chunk.py / embed.py needed at startup:
    You already have index.faiss in data/.
    This file auto-generates chunks.json from corpus.json at first launch
    (pure Python, zero API calls, zero cost) if chunks.json is missing.
    After that, both files are cached on disk — so every restart is instant.

Run:
    python app/agentic_ui.py
    → http://localhost:7861
"""

from __future__ import annotations

import json
import os
import sys
import tiktoken
from datetime import datetime
from pathlib import Path

import gradio as gr

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR     = Path(__file__).resolve().parent
ROOT_DIR    = APP_DIR.parent
DATA_DIR    = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

CORPUS_PATH      = DATA_DIR / "corpus.json"
CHUNKS_PATH      = DATA_DIR / "chunks.json"
FAISS_INDEX_PATH = DATA_DIR / "index.faiss"
COST_FILE        = ROOT_DIR / "cost_tracker.json"

sys.path.insert(0, str(APP_DIR))

# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-GENERATE chunks.json IF MISSING
#  Mirrors chunk_corpus.py logic exactly — same token sizes, same chunk_id
#  format. Zero cost, zero API calls. Runs once, cached forever.
# ══════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE_TOKENS    = 130
CHUNK_OVERLAP_TOKENS = 26
MIN_TEXT_LENGTH      = 50


def _build_chunks_from_corpus() -> list[dict]:
    """
    Rebuild chunks.json from corpus.json using the same token-chunking
    logic as scripts/chunk_corpus.py.

    WHY THIS IS SAFE (no cost):
        tiktoken is a local tokenizer — it runs entirely on your machine.
        No API call. No internet. No money spent.
        It just splits text into overlapping 130-token windows.

    FORMULA:
        step = chunk_size - overlap = 130 - 26 = 104 tokens
        chunk_1 = tokens[0   : 130]
        chunk_2 = tokens[104 : 234]
        chunk_3 = tokens[208 : 338]  ...
    """
    print("[startup] chunks.json not found — auto-generating from corpus.json (free, local)...")
    enc = tiktoken.encoding_for_model("gpt-4o-mini")

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    chunks   = []
    chunk_id = 0
    step     = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS  # = 104 tokens

    for entry in corpus:
        text   = entry.get("text", "").strip()
        source = entry.get("source", "unknown")
        page   = entry.get("page", 0)

        if not text or len(text) < MIN_TEXT_LENGTH:
            continue

        tokens = enc.encode(text)
        start  = 0

        while start < len(tokens):
            chunk_tokens = tokens[start : start + CHUNK_SIZE_TOKENS]
            chunk_text   = enc.decode(chunk_tokens).strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "source":   source,
                    "page":     page,
                    "text":     chunk_text,
                })
                chunk_id += 1
            start += step

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[startup] ✅ Generated {len(chunks)} chunks → saved to data/chunks.json")
    return chunks


def _ensure_chunks() -> list[dict]:
    """Load chunks.json, auto-generating it first if it doesn't exist."""
    if not CHUNKS_PATH.exists():
        return _build_chunks_from_corpus()
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[startup] ✅ Loaded {len(chunks)} chunks from data/chunks.json (cached)")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD PIPELINE (index.faiss + chunks — no embed.py needed)
# ══════════════════════════════════════════════════════════════════════════════

PIPELINE_READY = False
_index, _chunks_list = None, []

try:
    import faiss
    import numpy as np
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    # ── Step 1: chunks (auto-build if missing) ────────────────────────────────
    _chunks_list = _ensure_chunks()

    # ── Step 2: FAISS index (already on disk — just load it) ─────────────────
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"index.faiss not found at {FAISS_INDEX_PATH}.\n"
            "Copy your index.faiss into the data/ folder."
        )

    _index = faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"[startup] ✅ FAISS index loaded — {_index.ntotal} vectors")

    # ── Step 3: verify alignment ──────────────────────────────────────────────
    # The number of chunks must match the number of vectors in the index.
    # If they don't, the index was built from a different chunking run.
    if _index.ntotal != len(_chunks_list):
        print(
            f"[startup] ⚠ Mismatch: index has {_index.ntotal} vectors "
            f"but chunks.json has {len(_chunks_list)} entries.\n"
            f"          Regenerating chunks.json to match..."
        )
        # Force-regenerate chunks from scratch
        CHUNKS_PATH.unlink(missing_ok=True)
        _chunks_list = _build_chunks_from_corpus()

        # If still mismatched after regeneration, warn but continue
        if _index.ntotal != len(_chunks_list):
            print(
                f"[startup] ⚠ Still mismatched after regeneration "
                f"({_index.ntotal} vs {len(_chunks_list)}).\n"
                f"          Retrieval will work but chunk text may be off by a few entries."
            )

    # ── Step 4: load app modules ──────────────────────────────────────────────
    from generator import generate_answer, build_context
    from agents    import run_agentic_pipeline, _call_mcp_tool
    from mcp_server import (
        add_to_database    as mcp_add_to_database,
        load_pdf_to_database as mcp_load_pdf,
        create_markdown_report as mcp_save_report,
        web_search         as mcp_web_search,
    )

    _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    PIPELINE_READY = True
    print("[startup] ✅ Full pipeline ready. No scripts needed.")

except Exception as _err:
    print(f"[startup] ⚠ Pipeline not ready: {_err}")
    print("[startup]   Running in demo mode (UI loads, results are mocked).")


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL (local to this file — doesn't depend on retriever.py globals)
# ══════════════════════════════════════════════════════════════════════════════

def _embed_query(text: str) -> "np.ndarray":
    """Embed a single query string. This is the ONLY API call in retrieval."""
    resp = _openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def _retrieve(query: str, final_k: int = 5, candidate_k: int = 10) -> list[dict]:
    """
    Embed the query and search the FAISS index.
    Returns top final_k chunks with score metadata.

    COST: exactly 1 embedding API call per query (~$0.000002).
    """
    if not PIPELINE_READY:
        return []

    vec = _embed_query(query)
    scores, indices = _index.search(vec, candidate_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_chunks_list):
            continue
        chunk = dict(_chunks_list[idx])
        chunk["score"] = round(float(score), 4)
        results.append(chunk)

    # Heuristic rerank (free — pure Python, no API)
    pos = ["lunar","orion","sls","crew","mission","flyby","artemis","requirements"]
    neg = ["manager","biography","career"]
    for c in results:
        t = c.get("text","").lower()
        b = sum(0.04 for w in pos if w in t) - sum(0.10 for w in neg if w in t)
        c["rerank_score"] = round(c["score"] + b, 4)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:final_k]


# ══════════════════════════════════════════════════════════════════════════════
#  PLAIN RAG (wraps retrieval + generator)
# ══════════════════════════════════════════════════════════════════════════════

def run_plain_rag(query: str) -> tuple[str, list]:
    if not PIPELINE_READY:
        return ("⚠️  Pipeline not ready. Check terminal for startup errors.", [])
    chunks = _retrieve(query)
    answer = generate_answer(query, chunks, history=None)
    return answer, chunks


# ══════════════════════════════════════════════════════════════════════════════
#  AGENTIC PIPELINE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def run_agentic(query: str, save_report: bool = False) -> dict:
    if not PIPELINE_READY:
        return {
            "final_answer":      "⚠️  Pipeline not ready. Check terminal.",
            "internal_answer":   "",
            "external_findings": "",
            "chunks_used":       [],
        }
    return run_agentic_pipeline(query, save_report=save_report, check_external=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_cost_data() -> dict:
    if COST_FILE.exists():
        with open(COST_FILE) as f:
            return json.load(f)
    return {"total": 0.0, "calls": 0}


def format_cost_md() -> str:
    d         = get_cost_data()
    total     = d.get("total", 0.0)
    calls     = d.get("calls", 0)
    remaining = max(0.0, 5.0 - total)
    pct       = min(100, int((total / 5.0) * 100))
    bar       = "█" * (pct // 5) + "░" * (20 - pct // 5)
    return (
        f"**Spent:** ${total:.4f}  |  "
        f"**Remaining:** ${remaining:.4f}  |  "
        f"**Calls:** {calls}\n\n"
        f"`[{bar}]` {pct}% of $5.00 budget used"
    )


def format_sources(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    seen, lines = set(), []
    for c in chunks:
        key = f"{c['source']} — page {c['page']}"
        if key not in seen:
            seen.add(key)
            score = c.get("rerank_score", c.get("score", "?"))
            lines.append(f"• {key}  (score: {score})")
    return "**Sources retrieved:**\n" + "\n".join(lines)


def get_saved_reports() -> list[str]:
    return [r.name for r in sorted(
        REPORTS_DIR.glob("*.md"), key=os.path.getmtime, reverse=True
    )[:20]]


def read_report(name: str) -> str:
    if not name:
        return ""
    p = REPORTS_DIR / name
    return p.read_text(encoding="utf-8") if p.exists() else "Report not found."


def pipeline_info() -> str:
    if PIPELINE_READY:
        vecs   = _index.ntotal if _index else 0
        chunks = len(_chunks_list)
        return (
            f"🟢 **Pipeline Ready** — "
            f"{vecs} vectors in FAISS · {chunks} chunks loaded · "
            f"No scripts needed at startup"
        )
    return "🔴 **Demo Mode** — check terminal for errors"


# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0a0e1a;
    --bg2:      #111827;
    --bg3:      #1a2235;
    --border:   #1e3a5f;
    --accent:   #00d4ff;
    --purple:   #7c3aed;
    --green:    #10b981;
    --amber:    #f59e0b;
    --red:      #ef4444;
    --text:     #e2e8f0;
    --muted:    #64748b;
}

/* ── Global ─────────────────────────────────────────────────────────────── */
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Header ─────────────────────────────────────────────────────────────── */
.hdr {
    background: linear-gradient(135deg,#0a0e1a,#0d1b2e,#0a0e1a);
    border-bottom: 1px solid var(--border);
    padding: 18px 24px 14px;
}
.hdr-title {
    font-family:'Space Mono',monospace;
    font-size:20px; font-weight:700;
    color:var(--accent); letter-spacing:2px;
    text-transform:uppercase; margin:0;
}
.hdr-sub {
    font-family:'Space Mono',monospace;
    font-size:11px; color:var(--muted);
    margin-top:4px; letter-spacing:1px;
}
.dot {
    display:inline-block; width:8px; height:8px;
    border-radius:50%; margin-right:6px;
}
.dot-on  { background:var(--green); animation:pulse 2s infinite; }
.dot-off { background:var(--red); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ── Pipeline strip ─────────────────────────────────────────────────────── */
.pipeline-strip {
    display:flex; align-items:center; gap:6px; flex-wrap:wrap;
    padding:10px 14px; margin:10px 0;
    background:var(--bg3); border:1px solid var(--border); border-radius:8px;
    font-family:'Space Mono',monospace; font-size:10px; color:var(--muted);
}
.pill {
    padding:3px 11px; border-radius:20px; font-weight:700;
    white-space:nowrap; font-size:10px;
}
.pill-a1  { background:rgba(0,212,255,.12);  color:var(--accent); border:1px solid var(--accent); }
.pill-a2  { background:rgba(124,58,237,.12); color:var(--purple); border:1px solid var(--purple); }
.pill-a3  { background:rgba(16,185,129,.12); color:var(--green);  border:1px solid var(--green); }
.pill-mcp { background:rgba(245,158,11,.12); color:var(--amber);  border:1px dashed var(--amber); }
.pill-hi  { background:rgba(239,68,68,.12);  color:var(--red);    border:1px solid var(--red); }
.arr { color:var(--muted); }

/* ── Step tags ──────────────────────────────────────────────────────────── */
.tag {
    font-family:'Space Mono',monospace; font-size:10px;
    padding:2px 7px; border-radius:4px; font-weight:700;
    display:inline-block; margin-right:5px;
}
.tag-1 { background:rgba(0,212,255,.12);  color:var(--accent); }
.tag-2 { background:rgba(16,185,129,.12); color:var(--green);  }
.tag-3 { background:rgba(124,58,237,.12); color:var(--purple); }
.tag-4 { background:rgba(245,158,11,.12); color:var(--amber);  }
.tag-h { background:rgba(239,68,68,.12);  color:var(--red);    }

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.tab-nav { border-bottom:1px solid var(--border) !important; }
.tab-nav button {
    font-family:'Space Mono',monospace !important;
    font-size:10px !important; letter-spacing:1px !important;
    text-transform:uppercase !important;
    color:var(--muted) !important;
    border:none !important; background:transparent !important;
    padding:9px 16px !important;
}
.tab-nav button.selected {
    color:var(--accent) !important;
    border-bottom:2px solid var(--accent) !important;
}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
textarea, input[type="text"] {
    background:var(--bg3) !important;
    border:1px solid var(--border) !important;
    color:var(--text) !important;
    font-family:'DM Sans',sans-serif !important;
    border-radius:6px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color:var(--accent) !important;
    box-shadow:0 0 0 2px rgba(0,212,255,.1) !important;
    outline:none !important;
}
label span {
    font-family:'Space Mono',monospace !important;
    font-size:10px !important; letter-spacing:.5px !important;
    color:var(--muted) !important; text-transform:uppercase !important;
}

/* ── HITL box ───────────────────────────────────────────────────────────── */
.hitl-box {
    background:rgba(245,158,11,.07);
    border:1px solid var(--amber); border-radius:8px;
    padding:12px 16px; margin-bottom:10px;
    font-family:'Space Mono',monospace; font-size:11px;
}

/* ── Chatbot bubbles ────────────────────────────────────────────────────── */
.message.user { background:var(--bg3) !important; border:1px solid var(--border) !important; }
.message.bot  { background:rgba(0,212,255,.04) !important; border:1px solid rgba(0,212,255,.12) !important; }

/* ── Info bar ───────────────────────────────────────────────────────────── */
.info-bar {
    background:var(--bg2); border:1px solid var(--border);
    border-radius:6px; padding:10px 14px; margin:8px 0;
    font-family:'Space Mono',monospace; font-size:11px; color:var(--muted);
}
"""

# ══════════════════════════════════════════════════════════════════════════════
#  BUILD UI
# ══════════════════════════════════════════════════════════════════════════════

def build_ui() -> gr.Blocks:

    dot_cls = "dot-on" if PIPELINE_READY else "dot-off"
    dot_lbl = "PIPELINE READY · index.faiss loaded — no embed.py needed" if PIPELINE_READY \
              else "DEMO MODE — check terminal"

    with gr.Blocks(css=CSS, title="Artemis II — Agentic AI · Port 7861") as demo:

        # ── State for HITL ────────────────────────────────────────────────────
        st_draft    = gr.State(None)
        st_query    = gr.State(None)
        st_internal = gr.State(None)
        st_external = gr.State(None)

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div class="hdr">
          <div class="hdr-title">🚀 Artemis II — Agentic AI</div>
          <div class="hdr-sub">
            <span class="dot {dot_cls}"></span>{dot_lbl}
            &nbsp;·&nbsp;PORT 7861&nbsp;·&nbsp;gpt-4o-mini · FAISS · MCP · 3-Agent Pipeline
          </div>
        </div>
        """)

        # ── Pipeline strip ────────────────────────────────────────────────────
        gr.HTML("""
        <div class="pipeline-strip">
          USER QUERY
          <span class="arr">→</span>
          <span class="pill pill-a1">Agent 1 · Internal Researcher</span>
          <span class="arr">→</span>
          <span class="pill pill-mcp">MCP · web_search · wikipedia</span>
          <span class="arr">→</span>
          <span class="pill pill-a2">Agent 2 · Fact-Checker</span>
          <span class="arr">→</span>
          <span class="pill pill-a3">Agent 3 · Synthesizer</span>
          <span class="arr">→</span>
          <span class="pill pill-hi">HITL Review</span>
          <span class="arr">→</span>
          FINAL OUTPUT
        </div>
        """)

        # ── Index info bar ────────────────────────────────────────────────────
        gr.Markdown(pipeline_info())

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ═══════════════════════════════════════════════════════════════
            # TAB 1 — Agentic (HITL)
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("🤖  Agentic  (3-Agent + HITL)"):

                gr.HTML("""
                <div style="padding:8px 0;font-size:11px;color:#64748b;font-family:'Space Mono',monospace">
                  <span class="tag tag-1">STEP 1</span>web_search via MCP &nbsp;
                  <span class="tag tag-2">STEP 2</span>save markdown report &nbsp;
                  <span class="tag tag-3">STEP 3</span>add to corpus &nbsp;
                  <span class="tag tag-h">HITL</span>human approves before publish
                </div>""")

                with gr.Row():
                    with gr.Column(scale=4):
                        q_agentic = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g. Who are the Artemis II crew and what are their roles?",
                            lines=2,
                        )
                    with gr.Column(scale=1, min_width=130):
                        btn_run = gr.Button("▶  Run Agents", variant="primary")

                # ── HITL Checkpoint ───────────────────────────────────────────
                gr.HTML("""<div class="hitl-box">
                  <b style="color:#f59e0b">⏸ HITL CHECKPOINT</b> — The agent produces a draft.
                  You review it, then Approve, Reject, or give Feedback to refine.
                </div>""")

                draft_out = gr.Textbox(
                    label="Agent Draft  (review below before approving)",
                    lines=9, interactive=False,
                    placeholder="Draft answer appears here after clicking Run Agents...",
                )

                with gr.Row():
                    btn_approve = gr.Button("✅  Approve & Save Report", variant="primary")
                    btn_reject  = gr.Button("❌  Reject", variant="stop")

                feedback_in = gr.Textbox(
                    label="💬  Feedback — type here then click Refine",
                    placeholder="e.g. Focus more on crew backgrounds, skip rocket details.",
                    lines=2,
                )
                btn_refine = gr.Button("🔄  Refine with Feedback")

                # ── Final approved answer ─────────────────────────────────────
                final_out = gr.Textbox(
                    label="✅  Final Approved Answer",
                    lines=9, interactive=False,
                    placeholder="Approved answer appears here...",
                )

                # ── Agent breakdown ───────────────────────────────────────────
                with gr.Row():
                    with gr.Column():
                        internal_out = gr.Textbox(
                            label="🗂  Agent 1 — Internal Corpus Findings",
                            lines=5, interactive=False,
                        )
                    with gr.Column():
                        external_out = gr.Textbox(
                            label="🌐  Agent 2 — External Web Findings",
                            lines=5, interactive=False,
                        )

                sources_out = gr.Markdown(label="Sources")
                status_out  = gr.Textbox(
                    label="Pipeline Status", interactive=False,
                    value="Ready. Enter a question and click Run Agents.",
                )

            # ═══════════════════════════════════════════════════════════════
            # TAB 2 — Plain RAG
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("📖  Plain RAG  (fast · cheap)"):

                gr.HTML("""<div style="padding:8px 0 4px;font-size:11px;color:#64748b;
                font-family:'Space Mono',monospace">
                  Single-step · no agents · no web search ·
                  uses pre-built index.faiss (no API call until you ask a question)
                </div>""")

                def rag_chat(msg, hist):
                    ans, chunks = run_plain_rag(msg)
                    src = format_sources(chunks)
                    return f"{ans}\n\n---\n{src}" if src else ans

                gr.ChatInterface(
                    fn=rag_chat,
                    chatbot=gr.Chatbot(height=440, label="Artemis II RAG Chat"),
                    textbox=gr.Textbox(
                        placeholder="Ask about Artemis II...", container=False,
                    ),
                    examples=[
                        "Who are the four Artemis II crew members?",
                        "What is the height of the SLS rocket?",
                        "How many parachutes does Orion have?",
                        "How long does the Artemis II mission last?",
                        "What percentage of thrust do the boosters provide?",
                    ],
                )

            # ═══════════════════════════════════════════════════════════════
            # TAB 3 — Corpus Tools  (Steps 3 & 4)
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("🛠  Corpus Tools  (Steps 3 & 4)"):

                gr.HTML("""<div style="padding:8px 0 4px;font-size:11px;color:#64748b;
                font-family:'Space Mono',monospace">
                  <span class="tag tag-3">STEP 3</span>Add new text to corpus.json via MCP &nbsp;
                  <span class="tag tag-4">STEP 4</span>Load a PDF directly into corpus
                </div>""")

                gr.HTML("""<div class="info-bar">
                  ⚠ After adding text or a PDF, run <code>scripts/embed.py</code>
                  once to rebuild index.faiss. Then restart this UI to load the new index.
                </div>""")

                with gr.Row():

                    # Step 3 — Add text
                    with gr.Column():
                        gr.HTML("<b style='color:#7c3aed;font-size:13px'>Step 3 — Add Text</b>")
                        add_text = gr.Textbox(
                            label="New information",
                            placeholder="e.g. Artemis II launch rescheduled to June 2026.",
                            lines=4,
                        )
                        add_src  = gr.Textbox(
                            label="Source label",
                            placeholder="nasa_update_2026",
                            value="user_added",
                        )
                        btn_add  = gr.Button("➕  Add to Corpus", variant="primary")
                        res_add  = gr.Textbox(label="Result", interactive=False, lines=3)

                    # Step 4 — Load PDF
                    with gr.Column():
                        gr.HTML("<b style='color:#f59e0b;font-size:13px'>Step 4 — Load PDF</b>")
                        pdf_path = gr.Textbox(
                            label="PDF file path (absolute)",
                            placeholder="/absolute/path/to/new_document.pdf",
                        )
                        btn_pdf  = gr.Button("📄  Load PDF into Corpus", variant="primary")
                        res_pdf  = gr.Textbox(label="Result", interactive=False, lines=3)

            # ═══════════════════════════════════════════════════════════════
            # TAB 4 — Saved Reports  (Step 2)
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("📁  Reports  (Step 2)"):

                gr.HTML("""<div style="padding:8px 0 4px;font-size:11px;color:#64748b;
                font-family:'Space Mono',monospace">
                  <span class="tag tag-2">STEP 2</span>
                  Markdown reports saved by Agent 3 after HITL approval → reports/ folder
                </div>""")

                with gr.Row():
                    btn_refresh = gr.Button("🔄  Refresh List")
                    dd_reports  = gr.Dropdown(
                        choices=get_saved_reports(),
                        label="Select Report",
                        interactive=True,
                    )

                report_view = gr.Textbox(
                    label="Report Contents",
                    lines=22, interactive=False,
                    placeholder="Select a report above to preview...",
                )

            # ═══════════════════════════════════════════════════════════════
            # TAB 5 — Budget
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("💰  Budget"):

                cost_md      = gr.Markdown(format_cost_md())
                btn_cost_ref = gr.Button("🔄  Refresh")

                gr.HTML("""<div class="info-bar" style="margin-top:16px;line-height:2">
                  Cost per operation:<br>
                  &nbsp;• Plain RAG query &nbsp;&nbsp;— ~$0.0002 &nbsp;(1 embed + 1 LLM)<br>
                  &nbsp;• Agentic query &nbsp;&nbsp;&nbsp;— ~$0.0010 &nbsp;(4 LLM calls + 1 embed)<br>
                  &nbsp;• Startup (index load) — $0.0000 &nbsp;(free — local FAISS file)<br>
                  &nbsp;• chunks.json rebuild — $0.0000 &nbsp;(free — local tiktoken)<br>
                  <br>
                  <b>Index loaded from disk at startup = $0 every time you restart.</b>
                </div>""")

        # ══════════════════════════════════════════════════════════════════
        #  EVENT HANDLERS
        # ══════════════════════════════════════════════════════════════════

        # ── Run Agents ────────────────────────────────────────────────────
        def on_run(query):
            if not query.strip():
                empty = ("", "", "", "", "⚠ Enter a question first.", None, None, None, None)
                return empty

            ts = datetime.now().strftime("%H:%M:%S")
            result   = run_agentic(query, save_report=False)
            draft    = result["final_answer"]
            internal = result["internal_answer"]
            external = result["external_findings"]
            chunks   = result.get("chunks_used", [])
            src_md   = format_sources(chunks)

            return (
                draft,          # draft_out
                "",             # final_out (cleared until approved)
                internal,       # internal_out
                external,       # external_out
                src_md,         # sources_out
                f"[{ts}] ✅ Agents done. Review the draft, then Approve or Refine.",
                draft, query, internal, external,   # state updates
            )

        btn_run.click(
            fn=on_run,
            inputs=[q_agentic],
            outputs=[
                draft_out, final_out, internal_out, external_out,
                sources_out, status_out,
                st_draft, st_query, st_internal, st_external,
            ],
        )

        # ── Approve ───────────────────────────────────────────────────────
        def on_approve(draft, query, internal, external):
            if not draft:
                return "", "⚠ Nothing to approve yet."

            save_note = ""
            if PIPELINE_READY:
                body = (
                    f"## Question\n{query}\n\n"
                    f"## Final Answer\n{draft}\n\n---\n\n"
                    f"## Internal Corpus Findings\n{internal}\n\n"
                    f"## External Web Findings\n{external}"
                )
                mcp_save_report(
                    title=f"Artemis II: {(query or 'Report')[:60]}",
                    content=body,
                )
                save_note = "\n\n*✅ Report saved → see Reports tab.*"
            else:
                save_note = "\n\n*(Demo mode — report not saved.)*"

            ts = datetime.now().strftime("%H:%M:%S")
            return draft + save_note, f"[{ts}] ✅ Approved & saved."

        btn_approve.click(
            fn=on_approve,
            inputs=[st_draft, st_query, st_internal, st_external],
            outputs=[final_out, status_out],
        )

        # ── Reject ────────────────────────────────────────────────────────
        def on_reject():
            return "", "❌ Draft rejected. Enter a new question.", None, None, None, None

        btn_reject.click(
            fn=on_reject,
            outputs=[draft_out, status_out, st_draft, st_query, st_internal, st_external],
        )

        # ── Refine ────────────────────────────────────────────────────────
        def on_refine(query, feedback, internal, external):
            if not query:
                return "", "", "", "⚠ Run agents first.", None, None, None, None
            if not feedback.strip():
                return "", "", "", "⚠ Enter feedback before refining.", None, None, None, None

            refined = f"{query}\n\nREVIEWER FEEDBACK: {feedback}"
            result  = run_agentic(refined, save_report=False)
            draft   = result["final_answer"]
            intern2 = result["internal_answer"]
            extern2 = result["external_findings"]

            ts = datetime.now().strftime("%H:%M:%S")
            return (
                draft, intern2, extern2,
                f"[{ts}] 🔄 Refined. Review new draft.",
                draft, query, intern2, extern2,
            )

        btn_refine.click(
            fn=on_refine,
            inputs=[st_query, feedback_in, st_internal, st_external],
            outputs=[
                draft_out, internal_out, external_out, status_out,
                st_draft, st_query, st_internal, st_external,
            ],
        )

        # ── Add text (Step 3) ─────────────────────────────────────────────
        def on_add(text, src):
            if not text.strip():
                return "⚠ Please enter text to add."
            if PIPELINE_READY:
                return mcp_add_to_database(text=text, source=src or "user_added")
            return f"(Demo) Would add {len(text)} chars from source '{src}' to corpus.json."

        btn_add.click(fn=on_add, inputs=[add_text, add_src], outputs=[res_add])

        # ── Load PDF (Step 4) ─────────────────────────────────────────────
        def on_pdf(path):
            if not path.strip():
                return "⚠ Please enter a PDF file path."
            if PIPELINE_READY:
                return mcp_load_pdf(pdf_path=path)
            return f"(Demo) Would load PDF: {path}"

        btn_pdf.click(fn=on_pdf, inputs=[pdf_path], outputs=[res_pdf])

        # ── Reports ───────────────────────────────────────────────────────
        btn_refresh.click(
            fn=lambda: gr.Dropdown(choices=get_saved_reports()),
            outputs=[dd_reports],
        )
        dd_reports.change(fn=read_report, inputs=[dd_reports], outputs=[report_view])

        # ── Budget ────────────────────────────────────────────────────────
        btn_cost_ref.click(fn=format_cost_md, outputs=[cost_md])

    return demo


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("  Artemis II — Agentic AI Web UI")
    print("  URL  : http://localhost:7861")
    print(f"  Index: {'LOADED ✅' if PIPELINE_READY else 'NOT READY ❌'}")
    print(f"  Chunks: {len(_chunks_list)} entries")
    print("  Cost at startup: $0.00  (FAISS loaded from disk)")
    print("=" * 62)

    build_ui().launch(
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True,
    )