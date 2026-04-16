"""
app/visualizer_agent.py  —  "Go Even Further": Visualizer Agent
================================================================

WHAT THIS DOES:
    A dedicated agent that generates visual outputs from RAG answers:
      1. Matplotlib charts  (.png files)  — bar, line, pie charts
      2. Mermaid diagrams   (.md files)   — flowcharts, sequence diagrams
      3. Data summary tables (.md)        — structured comparison tables

    It is triggered automatically when a question looks like it needs
    a visual (detected by keyword patterns), or can be called directly.

WHY A SEPARATE VISUALIZER AGENT?
    From the slides: "Generate graphs and specific files using MCP,
    matplotlib, mermaid — Visualizer Agent"

    Text answers are great, but some questions need visuals:
      "Compare the thrust of each engine"      → bar chart
      "Show the mission timeline"              → mermaid sequence diagram
      "What is the mission flow from launch?"  → flowchart
      "Show crew roles"                        → comparison table

    The Visualizer Agent reads Agent 3's synthesized text answer,
    detects if a visual would help, and generates the right output.
    The image/diagram file is then:
      a) Saved to reports/ folder
      b) Shown in the agentic_ui.py (Tab 1 — after approval)
      c) Referenced in the Markdown report

FILE PATH:
    project-root/
    └── app/
        └── visualizer_agent.py   ← THIS FILE

HOW TO USE:
    # Standalone
    python app/visualizer_agent.py

    # From agents.py (automatic detection)
    from visualizer_agent import maybe_visualize
    viz = maybe_visualize(query, final_answer)
    # viz = {"type": "chart"|"mermaid"|"table"|None, "path": "...", "content": "..."}

    # From agentic_ui.py (display image in Gradio)
    if viz and viz["type"] == "chart":
        gr.Image(viz["path"])
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

APP_DIR     = Path(__file__).resolve().parent
ROOT_DIR    = APP_DIR.parent
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(APP_DIR))

# ── OpenAI client ─────────────────────────────────────────────────────────────
from openai import OpenAI
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Cost tracker ──────────────────────────────────────────────────────────────
COST_FILE = ROOT_DIR / "cost_tracker.json"

def _track(response, label="visualizer") -> None:
    u    = response.usage
    cost = (u.prompt_tokens * 0.15 + u.completion_tokens * 0.60) / 1_000_000
    data = {"total": 0.0, "calls": 0}
    if COST_FILE.exists():
        with open(COST_FILE) as f:
            data = json.load(f)
    data["total"] = round(data.get("total", 0) + cost, 6)
    data["calls"] = data.get("calls", 0) + 1
    with open(COST_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[cost][{label}] ${cost:.6f} | Total: ${data['total']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — DETECT if a visual is needed
# ══════════════════════════════════════════════════════════════════════════════

# Keywords that strongly suggest a chart would be useful
CHART_TRIGGERS = [
    "compare", "comparison", "versus", "vs",
    "how much", "how many", "percentage", "percent", "%",
    "thrust", "height", "weight", "speed", "distance", "capacity",
    "number of", "count", "total",
]

# Keywords that suggest a flowchart / sequence diagram
FLOW_TRIGGERS = [
    "timeline", "sequence", "steps", "phases", "process",
    "flow", "order", "stages", "procedure", "mission flow",
    "after", "then", "next", "finally", "launch",
]

# Keywords that suggest a comparison table
TABLE_TRIGGERS = [
    "difference", "differ", "differences between",
    "crew", "roles", "each member", "members",
    "compare", "artemis i vs", "artemis ii vs",
]


def detect_viz_type(query: str, answer: str) -> str | None:
    """
    Decide what kind of visual to generate (if any).

    Logic:
        chart  → numbers/comparisons present
        mermaid → timeline/flow present
        table  → comparisons of named items (crew, missions)
        None   → no visual needed

    Returns:
        "chart" | "mermaid" | "table" | None
    """
    q_lower = (query + " " + answer[:500]).lower()

    chart_score  = sum(1 for kw in CHART_TRIGGERS  if kw in q_lower)
    flow_score   = sum(1 for kw in FLOW_TRIGGERS   if kw in q_lower)
    table_score  = sum(1 for kw in TABLE_TRIGGERS  if kw in q_lower)

    # Numbers in the answer strongly suggest a chart
    number_count = len(re.findall(r'\b\d+[\.,]?\d*\s*(%|mph|ft|m|lbs|kg|miles|km|°)\b', answer))
    chart_score += number_count * 2

    scores = {"chart": chart_score, "mermaid": flow_score, "table": table_score}
    best   = max(scores, key=scores.get)

    if scores[best] >= 2:
        return best
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2a — MATPLOTLIB CHART GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

CHART_SYSTEM = """You are a data extraction assistant.
Read the provided RAG answer and extract numerical data for a chart.
Return ONLY valid JSON in this exact format — no markdown, no explanation:

{
  "chart_type": "bar" | "horizontal_bar" | "pie" | "line",
  "title": "Chart title here",
  "x_label": "X axis label",
  "y_label": "Y axis label",
  "data": [
    {"label": "Name 1", "value": 123.4},
    {"label": "Name 2", "value": 456.7}
  ]
}

Rules:
- chart_type: use "bar" for comparisons, "pie" for percentages, "line" for over time
- Extract ALL numerical comparisons you can find
- Round values to 2 decimal places
- If no chart-worthy data exists, return: {"chart_type": null}"""


def generate_chart(query: str, answer: str) -> dict[str, Any]:
    """
    Ask the LLM to extract data from the answer, then plot it with matplotlib.

    FORMULA:
        answer text → LLM extracts numbers → JSON data → matplotlib figure
        Saves as .png to reports/ folder.

    Returns:
        dict with keys: path (str), title (str), success (bool)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — works without display
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return {"success": False, "error": "Install matplotlib: pip install matplotlib"}

    # ── Ask LLM to extract chart data ─────────────────────────────────────────
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",  "content": CHART_SYSTEM},
            {"role": "user",    "content": f"Question: {query}\n\nRAG Answer:\n{answer}"},
        ],
        temperature=0,
        max_tokens=400,
    )
    _track(resp, "visualizer-chart-extract")

    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"success": False, "error": f"LLM returned invalid JSON: {raw[:200]}"}

    if not data.get("chart_type"):
        return {"success": False, "error": "LLM found no chart-worthy data."}

    # ── Plot ──────────────────────────────────────────────────────────────────
    labels  = [d["label"]  for d in data["data"]]
    values  = [d["value"]  for d in data["data"]]
    c_type  = data["chart_type"]
    title   = data.get("title", query[:60])
    x_label = data.get("x_label", "")
    y_label = data.get("y_label", "Value")

    # Dark NASA-style theme
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#111827")

    colors = ["#00d4ff", "#7c3aed", "#10b981", "#f59e0b", "#ef4444",
              "#06b6d4", "#8b5cf6", "#34d399", "#fbbf24", "#f87171"]

    if c_type in ("bar", "horizontal_bar"):
        if c_type == "horizontal_bar":
            bars = ax.barh(labels, values, color=colors[:len(labels)], edgecolor="none")
            ax.set_xlabel(y_label, color="#94a3b8", fontsize=11)
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                        f"{val:,.1f}", va="center", color="#e2e8f0", fontsize=9)
        else:
            bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor="none",
                          width=0.6)
            ax.set_xlabel(x_label, color="#94a3b8", fontsize=11)
            ax.set_ylabel(y_label, color="#94a3b8", fontsize=11)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f"{val:,.1f}", ha="center", color="#e2e8f0", fontsize=9)

    elif c_type == "pie":
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct="%1.1f%%",
            colors=colors[:len(labels)], startangle=90,
            textprops={"color": "#e2e8f0", "fontsize": 10},
        )
        for at in autotexts:
            at.set_color("#0a0e1a")
            at.set_fontweight("bold")

    elif c_type == "line":
        ax.plot(labels, values, color="#00d4ff", linewidth=2.5,
                marker="o", markersize=6, markerfacecolor="#7c3aed")
        ax.fill_between(range(len(values)), values, alpha=0.15, color="#00d4ff")
        ax.set_xlabel(x_label, color="#94a3b8", fontsize=11)
        ax.set_ylabel(y_label, color="#94a3b8", fontsize=11)

    ax.set_title(title, color="#e2e8f0", fontsize=13, fontweight="bold", pad=16)
    ax.tick_params(colors="#64748b", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")

    # Watermark
    fig.text(0.99, 0.01, "Artemis II RAG · Auto-generated",
             ha="right", color="#1e3a5f", fontsize=8)

    plt.tight_layout()

    # Save
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chart_{ts}.png"
    path     = REPORTS_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"[visualizer] 📊 Chart saved → {path}")
    return {"success": True, "path": str(path), "title": title, "type": "chart"}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2b — MERMAID DIAGRAM GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

MERMAID_SYSTEM = """You are a Mermaid diagram expert.
Read the provided question and RAG answer, then write a Mermaid diagram
that visually represents the key information.

Choose the most appropriate diagram type:
- flowchart TD  — for processes, mission phases, decision flows
- sequenceDiagram — for timelines or sequential events
- pie title X   — for percentage breakdowns

Output ONLY the raw Mermaid code (no markdown fences, no explanation).

Example flowchart:
flowchart TD
    A[Launch] --> B[Booster Separation]
    B --> C[ICPS Burn]
    C --> D[Lunar Flyby]
    D --> E[Trans-Earth Return]
    E --> F[Splashdown]

Keep it clean, max 15 nodes."""


def generate_mermaid(query: str, answer: str) -> dict[str, Any]:
    """
    Ask the LLM to write a Mermaid diagram from the answer.
    Saves the .md file to reports/. Also saves the raw .mmd file.

    Mermaid files can be rendered at: https://mermaid.live

    Returns:
        dict with keys: path (str), mermaid_code (str), success (bool)
    """
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",  "content": MERMAID_SYSTEM},
            {"role": "user",    "content": f"Question: {query}\n\nRAG Answer:\n{answer[:1000]}"},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    _track(resp, "visualizer-mermaid")

    mermaid_code = resp.choices[0].message.content.strip()
    # Strip any accidental markdown fences
    mermaid_code = re.sub(r"```mermaid|```", "", mermaid_code).strip()

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diagram_{ts}.md"
    path     = REPORTS_DIR / filename

    md_content = f"""# Artemis II — Diagram
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Question: {query}*

---

```mermaid
{mermaid_code}
```

---
*Render at: https://mermaid.live  — paste the code above*
"""
    path.write_text(md_content, encoding="utf-8")

    print(f"[visualizer] 🔷 Mermaid diagram saved → {path}")
    return {
        "success":       True,
        "path":          str(path),
        "mermaid_code":  mermaid_code,
        "type":          "mermaid",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2c — COMPARISON TABLE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

TABLE_SYSTEM = """You are a data extraction assistant.
Read the question and RAG answer, then extract structured comparison data.
Return ONLY valid JSON:

{
  "title": "Table title",
  "columns": ["Column 1", "Column 2", "Column 3"],
  "rows": [
    ["Row 1 Label", "Value A", "Value B"],
    ["Row 2 Label", "Value C", "Value D"]
  ]
}

Keep the table concise — max 8 rows, max 5 columns.
If no table-worthy data exists, return: {"title": null}"""


def generate_table(query: str, answer: str) -> dict[str, Any]:
    """
    Ask the LLM to extract comparison data and format it as a markdown table.
    Saves to reports/ as a .md file.

    Returns:
        dict with keys: path (str), markdown (str), success (bool)
    """
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",  "content": TABLE_SYSTEM},
            {"role": "user",    "content": f"Question: {query}\n\nRAG Answer:\n{answer}"},
        ],
        temperature=0,
        max_tokens=400,
    )
    _track(resp, "visualizer-table")

    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"success": False, "error": "LLM returned invalid JSON for table."}

    if not data.get("title"):
        return {"success": False, "error": "No table data found."}

    # Build markdown table
    cols = data["columns"]
    rows = data["rows"]

    header    = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_rows = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]

    md_table = "\n".join([header, separator] + body_rows)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"table_{ts}.md"
    path     = REPORTS_DIR / filename

    md_content = f"""# {data['title']}
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Question: {query}*

---

{md_table}

---
*Source: Artemis II RAG Assistant*
"""
    path.write_text(md_content, encoding="utf-8")

    print(f"[visualizer] 📋 Table saved → {path}")
    return {
        "success":  True,
        "path":     str(path),
        "markdown": md_table,
        "title":    data["title"],
        "type":     "table",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — called from agents.py and agentic_ui.py
# ══════════════════════════════════════════════════════════════════════════════

def maybe_visualize(
    query: str,
    answer: str,
    force_type: str | None = None,
) -> dict[str, Any] | None:
    """
    Main entry point. Detects if a visual is needed and generates it.

    Called from agents.py after Agent 3 finishes synthesizing.
    Called from agentic_ui.py to display output in the UI.

    Args:
        query:      The user's original question
        answer:     Agent 3's synthesized text answer
        force_type: "chart" | "mermaid" | "table" — skip detection

    Returns:
        dict with keys: type, path, success, (and type-specific extras)
        OR None if no visual is needed / generation failed.
    """
    viz_type = force_type or detect_viz_type(query, answer)

    if not viz_type:
        print("[visualizer] No visual needed for this query.")
        return None

    print(f"[visualizer] Generating {viz_type} for: '{query[:60]}'")

    if viz_type == "chart":
        result = generate_chart(query, answer)
    elif viz_type == "mermaid":
        result = generate_mermaid(query, answer)
    elif viz_type == "table":
        result = generate_table(query, answer)
    else:
        return None

    if not result.get("success"):
        print(f"[visualizer] ⚠ Generation failed: {result.get('error')}")
        return None

    return result


def visualize_all(query: str, answer: str) -> list[dict]:
    """
    Generate ALL applicable visuals for a query (chart + mermaid + table).
    Used for the "Go Further" demo mode where we want to show everything.

    Returns:
        List of result dicts for each visual that was successfully generated.
    """
    results = []
    for viz_type in ("chart", "mermaid", "table"):
        r = maybe_visualize(query, answer, force_type=viz_type)
        if r:
            results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MCP TOOL WRAPPERS  (register these in mcp_server.py)
# ══════════════════════════════════════════════════════════════════════════════

def mcp_generate_chart(query: str, answer: str) -> str:
    """MCP tool: generate a matplotlib chart from a RAG answer."""
    r = generate_chart(query, answer)
    if r.get("success"):
        return f"✅ Chart saved: {r['path']}"
    return f"❌ Chart failed: {r.get('error')}"


def mcp_generate_mermaid(query: str, answer: str) -> str:
    """MCP tool: generate a Mermaid diagram from a RAG answer."""
    r = generate_mermaid(query, answer)
    if r.get("success"):
        return f"✅ Mermaid diagram saved: {r['path']}\n\nCode:\n{r['mermaid_code']}"
    return f"❌ Mermaid failed: {r.get('error')}"


def mcp_generate_table(query: str, answer: str) -> str:
    """MCP tool: generate a markdown comparison table from a RAG answer."""
    r = generate_table(query, answer)
    if r.get("success"):
        return f"✅ Table saved: {r['path']}\n\n{r['markdown']}"
    return f"❌ Table failed: {r.get('error')}"


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("  Artemis II — Visualizer Agent (Go Further)")
    print("  Generates: matplotlib charts · Mermaid diagrams · Tables")
    print("=" * 62)

    # Test query
    test_q = "Compare the thrust of SLS engines and solid rocket boosters"
    test_a = (
        "The SLS Block 1 generates a maximum thrust of 8.8 million pounds (39,144 kN). "
        "The four RS-25 engines each produce 418,000 lbs at launch, totalling 1,672,000 lbs. "
        "The two solid rocket boosters produce approximately 3,300,000 lbs each, "
        "totalling 6,600,000 lbs — about 75% of total thrust. "
        "The RS-25 engines contribute the remaining 25%."
    )

    print(f"\nTest Query: {test_q}")
    print(f"Answer: {test_a[:100]}...\n")

    detected = detect_viz_type(test_q, test_a)
    print(f"Detected viz type: {detected}\n")

    # Generate all 3 types for demo
    print("Generating all 3 visual types for demo...\n")
    results = visualize_all(test_q, test_a)

    for r in results:
        if r.get("success"):
            print(f"✅ {r['type'].upper()} → {r['path']}")
        else:
            print(f"❌ {r['type'].upper()} → {r.get('error')}")
