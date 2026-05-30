# Artemis II Knowledge Navigator: RAG & Agentic AI Assistant

Artemis II Knowledge Navigator is a Python RAG and agentic AI project for answering questions about NASA's Artemis II mission. It turns official NASA PDF documents into a searchable knowledge base, retrieves relevant passages with FAISS, generates grounded answers with OpenAI models, and optionally runs a multi-agent workflow that verifies answers against live web sources before saving a report.

The project is built as a practical end-to-end AI assistant: document extraction, chunking, embeddings, retrieval, answer generation, web UI, command-line HITL workflow, MCP tools, evaluation scripts, feedback logging, and cost tracking are all included.

## What This Project Does

- Extracts text from NASA Artemis II PDFs and builds a JSON corpus.
- Splits the corpus into sentence-aware chunks with source and page metadata.
- Embeds chunks with `text-embedding-3-small` and stores them in a FAISS vector index.
- Answers user questions using retrieved Artemis II context only, with inline citations.
- Supports a Gradio chatbot UI with source display, feedback, analytics, and conversation export.
- Provides a command-line interactive assistant with plain RAG mode and a deeper agentic mode.
- Implements a three-agent pipeline:
  - Internal Researcher: searches the local Artemis II FAISS corpus.
  - External Fact-Checker: searches the web, fetches full pages, and retrieves Wikipedia article content through MCP tools.
  - Synthesizer: combines internal and external findings into a concise final answer.
- Adds Human-in-the-Loop review for agentic answers before publishing reports or updating the corpus.
- Tracks OpenAI API cost against a small project budget.
- Includes evaluation and debugging scripts for retrieval quality, chunk validation, prompt experiments, and failure analysis.

## Project Highlights

### Core Implementation

- Built an end-to-end RAG assistant over official NASA Artemis II documents, including PDF extraction, sentence-aware chunking, OpenAI embeddings, FAISS retrieval, and GPT-based answer generation.
- Improved retrieval relevance with a two-stage retrieval flow: FAISS candidate search followed by heuristic reranking to select the final top-5 context chunks.
- Engineered prompts for synthesized structured answers, strict yes/no response control, grounded citations, and selective chat-history injection for follow-up questions.

### Grounded Artemis II Q&A

The assistant is designed to answer from official Artemis II documents rather than general model knowledge. Retrieved chunks include document titles and page numbers, and generated answers cite the source pages used.

### RAG Plus Agentic Verification

The project includes both a fast plain RAG path and a higher-quality agentic path. The agentic path first checks the local corpus, then uses MCP tools to read external pages, and finally synthesizes the two sources.

### Real Page Fetching, Not Just Snippets

The external fact-checker does not stop at search snippets. It calls `fetch_webpage()` and `fetch_wikipedia_page()` to extract readable full-page content before asking the synthesizer to compare or enrich the corpus answer.

### Human-in-the-Loop Control

The CLI workflow pauses before irreversible actions. A reviewer can approve, reject, or refine an agent draft before a markdown report is saved or new knowledge is added.

### Retrieval Quality Work

The repository contains a failure log and evaluation scripts showing iterative improvements: yes/no handling, ambiguous question handling, follow-up detection, heuristic reranking, low-confidence warnings, and out-of-scope refusal behavior.

## Architecture

```text
NASA PDFs
   |
   v
scripts/extract.py
   |
   v
data/corpus.json
   |
   v
scripts/chunk_corpus.py
   |
   v
data/chunks.json
   |
   v
scripts/embed.py
   |
   v
data/index.faiss
   |
   +--> app/app.py       Gradio chatbot UI
   |
   +--> app/main.py      CLI assistant with RAG, agents, and HITL
   |
   +--> app/agents.py    Internal researcher, external checker, synthesizer
          |
          v
      app/mcp_server.py  Web search, full-page fetch, reports, corpus updates
```

## Repository Structure

```text
app/
  app.py                 Gradio web UI for the RAG chatbot
  main.py                Interactive CLI with RAG, agentic mode, and HITL review
  agents.py              Three-agent research and synthesis pipeline
  retriever.py           FAISS loading, query embedding, retrieval, reranking
  generator.py           Prompting, answer generation, citations, follow-up support
  mcp_server.py          MCP tools for web search, page fetching, reports, corpus updates
  visualizer_agent.py    Optional visualization helper
  email_bridge.py        Email-related integration helper
  agentic_ui.py          Agentic UI experiment

scripts/
  extract.py             Extract text from PDFs into corpus JSON
  chunk_corpus.py        Split corpus into retrieval chunks
  embed.py               Generate embeddings and build FAISS index
  evaluate.py            Evaluate answer and retrieval behavior
  eval_runner.py         Run evaluation batches
  eval_agentic.py        Evaluate the agentic pipeline
  retrieval_checker.py   Inspect retrieval results
  validate_chunks.py     Validate chunk quality
  check_corpus.py        Inspect corpus health
  dry_run.py             Development smoke test helper
  EXTRACTION_REPORT.md   Notes from the PDF extraction run
  FAILURE_LOG.md         Known failures, fixes, and limitations

prompts/
  v1_baseline.txt        Baseline prompt
  v2_synthesis.txt       Synthesis-oriented prompt
  v3_yes_no.txt          Yes/no handling prompt
```

The generated `data/`, `reports/`, and `cost_tracker.json` files may be created locally during use and are not necessarily present in a fresh clone.

## Requirements

- Python 3.12 recommended
- OpenAI API key
- Dependencies listed in `requirements.txt`

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

## Build the Knowledge Base

If you have the source PDFs under `data/pdfs/`, run the full preparation pipeline:

```bash
python scripts/extract.py data/pdfs
python scripts/chunk_corpus.py
python scripts/embed.py
```

Expected generated files:

```text
data/corpus.json
data/chunks.json
data/index.faiss
```

According to `scripts/EXTRACTION_REPORT.md`, the original extraction run processed 3 documents, 253 pages, and about 494k characters.

## Run the Gradio Chatbot

```bash
python app/app.py
```

Open:

```text
http://localhost:7860
```

The web UI provides:

- Chat over Artemis II documents
- Source citations for retrieved chunks
- Low-confidence warning when retrieval score is weak
- Like/dislike feedback logging
- Session analytics
- Conversation export

## Run the CLI Assistant

```bash
python app/main.py
```

Useful commands:

```text
/rag <question>       Run plain RAG
/agentic <question>   Run the three-agent pipeline
/add <text>           Add text to the corpus after confirmation
/pdf <path>           Load a PDF into the corpus after confirmation
/cost                 Show API cost tracking
/help                 Show commands
/quit                 Exit
```

Plain input without a command is treated as a RAG question.

## Run the MCP Server

The MCP server exposes tools used by the agentic workflow:

```bash
python app/mcp_server.py
```

Available tools include:

- `web_search`
- `fetch_webpage`
- `fetch_wikipedia_page`
- `create_markdown_report`
- `add_to_database`
- `load_pdf_to_database`

Note: `agents.py` currently imports tool functions directly for reliability, while preserving the MCP server architecture and tool definitions.

## Evaluation and Debugging

The repository includes utilities for checking retrieval and answer behavior:

```bash
python scripts/retrieval_checker.py
python scripts/validate_chunks.py
python scripts/evaluate.py
python scripts/eval_runner.py
python scripts/eval_agentic.py
```

See `scripts/FAILURE_LOG.md` for the main issues discovered during development and how they were addressed. Improvements include strict yes/no output, follow-up history injection, prompt synthesis, heuristic reranking, and Gradio compatibility fixes.

## Known Limitations

- Follow-up pronoun questions can still retrieve weak chunks because FAISS cannot resolve pronouns by itself; history injection mitigates this at generation time.
- Very short chunks from tables of contents, captions, or sparse pages can produce weak embeddings.
- The local corpus only knows what has been extracted and embedded. After adding web text or PDFs through MCP tools, re-run `scripts/embed.py` to refresh the FAISS index.
- Live web search depends on network access and external services such as DuckDuckGo or Wikipedia.
- Some code comments contain display artifacts from earlier encoding issues, but the Python logic is still readable and functional.

## Tech Stack

- Python
- OpenAI API
- FAISS
- NumPy
- Gradio
- PyMuPDF
- FastMCP
- DuckDuckGo Search
- Requests and BeautifulSoup
- Wikipedia API helpers
- Matplotlib

## Suggested Demo Questions

- Who are the four crew members of Artemis II?
- What is the total height of the SLS Block 1 rocket?
- How long is the Artemis II mission expected to last?
- What percentage of thrust do the solid rocket boosters provide?
- Tell me about the mission requirements.
- Did the Artemis II crew successfully return to Earth?

## Project Status

This is a working educational prototype of a domain-specific RAG and agentic AI assistant. It demonstrates how to move from raw PDFs to a cited chatbot, then extend that chatbot with MCP tools, external verification, runtime corpus updates, reporting, evaluation, and human review.
