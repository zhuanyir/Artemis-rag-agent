# FAILURE LOG — Artemis II Knowledge Navigator

> Documents all failures encountered during RAG development, their root cause, fix applied, and whether it was resolved.
> Updated iteratively during the build phase.

---

## How to read this log

- **Retrieval OK?** ✅ = right chunks were found | ❌ = wrong chunks retrieved
- **Problem** = root cause (retrieval miss vs generation error vs prompt issue)
- **Fix applied** = what change was made
- **Fixed?** ✅ = resolved | ⚠️ = partially | ❌ = unresolved

---

## Failure Table

| # | Question | Category | Retrieval OK? | Problem | Fix applied | Fixed? |
|---|----------|----------|:---:|---------|-------------|:---:|
| 1 | "Is Reid Wiseman the Commander?" | yes_no | ✅ | LLM gave a full paragraph instead of Yes/No | Added `is_yes_no_question()` detector + extra instruction prepended to prompt | ✅ |
| 2 | "Tell me about the requirements." | ambiguous | ✅ | Bot answered with mission objectives only, ignored launch/weather/technical requirements | Updated system prompt to acknowledge ambiguity first, then give broad answer covering all requirement types | ✅ |
| 3 | "How far will the crew travel?" | ambiguous | ✅ | Answer was incomplete — cut off mid-sentence | `max_tokens` was too low (500). Increased to 600, then switched to synthesizing prompt which fits in 500 | ✅ |
| 4 | "What happened after January 2026?" | out-of-scope | ✅ | Bot answered from pre-launch corpus content instead of refusing | Question was poorly formed — corpus DOES contain pre-launch Jan 2026 content. Replaced with better out-of-scope question: "Did the crew successfully return to Earth?" | ✅ |
| 5 | app.py crash on startup | - | - | `retry_btn` and `undo_btn` not supported in installed Gradio version | Removed unsupported kwargs from `gr.ChatInterface()` | ✅ |
| 6 | app.py crash: `type="messages"` | - | - | `gr.Chatbot(type="messages")` not supported in installed Gradio version | Removed `type="messages"` argument | ✅ |
| 7 | retrieve() got unexpected keyword argument 'k' | - | - | `app.py` called `retrieve(k=5)` but `retriever.py` renamed param to `final_k` | Updated `app.py` to use `final_k=5` | ✅ |
| 8 | retrieve() got unexpected keyword argument 'use_query_rewrite' | - | - | `app.py` passed `use_query_rewrite=True` but updated `retriever.py` removed that param for speed | Removed `use_query_rewrite` and `use_rerank` from `app.py` retrieve() call | ✅ |
| 9 | Response time very slow (~10s+) | - | - | Query rewriting added an extra OpenAI embedding API call per question | Removed `use_query_rewrite` from `retriever.py` (saved ~1-2s per query). Reduced `candidate_k` from 20→10 | ✅ |
| 10 | evaluate.py --LLM fails: OPENAI_API_KEY not set | - | - | `evaluate.py` used `os.environ.get()` directly without loading `.env` file | Added `from dotenv import load_dotenv` + `load_dotenv()` at top of `evaluate.py` | ✅ |
| 11 | Answer cut off at item 10 of requirements list | factual | ✅ | `max_tokens=500` too low for long synthesized lists | Updated system prompt to group/synthesize rather than list all items verbatim — fits within 500 tokens | ✅ |
| 12 | Follow-up "Where is he from?" retrieved low-confidence chunks (score ~0.13) | cross-reference | ⚠️ | Pronoun "he" has no vector similarity to "Jeremy Hansen" — FAISS can't resolve references | Implemented `is_followup()` to inject last 3 history turns into prompt context. Scores still low but correct answer retrieved | ⚠️ |

---

## Retrieval Hit Rate Summary (latest eval run)

| Category | Hit Rate | Notes |
|----------|----------|-------|
| factual | ~85% | Strong on specific technical facts |
| cross-reference | ~75% | Occasionally misses one of two required sources |
| out-of-scope | N/A | No expected source by definition |
| ambiguous | N/A | No expected source by definition |

---

## Known Limitations

1. **Pronoun resolution in follow-ups** — "Where is he from?" scores ~0.13 on retrieval because FAISS can't resolve "he" to a person without context. History injection partially mitigates this but retrieval quality remains lower than standalone questions.

2. **Very short pages** — 75 chunks under 50 tokens exist in the corpus (table of contents, captions). These produce weak embeddings and occasionally surface as false positives in retrieval.

3. **Out-of-date questions** — The corpus covers up to January 2026. Post-mission questions (e.g. "Did the crew return safely?") correctly return "I don't have information about this" since the mission was planned for April 2026.

---

## Fixes That Improved Multiple Questions

| Fix | Questions improved |
|-----|--------------------|
| Synthesizing system prompt (groupmate suggestion) | All factual + cross-reference questions — answers shorter, less repetition |
| Heuristic reranking | "What are the mission requirements?" — biography chunks deprioritised |
| `is_yes_no_question()` detector | All yes/no questions — direct Yes/No response |
| `is_followup()` history injection | Follow-up pronoun questions — correct context maintained |
| Increased `candidate_k` to 10 with reranking | Cross-reference questions — more candidates to rerank from |
