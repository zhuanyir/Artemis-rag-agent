# Extraction Report

## Overview
- Date: 2026-04-10
- Script: `scripts/extract.py`
- Engine: PyMuPDF (`fitz`)
- Input: `data/pdfs`
- Output: `data/corpus.json`, `data/sample.json`

## Corpus Stats
- Documents: 3
- Total pages: 253
- Total characters: 493963
- Avg chars/page: 1952.42
- Empty or near-empty pages (`< 50` chars): 7

## Main Issues and Fixes
- Repeated headers/footers:
  - Problem: same lines appeared on many pages.
  - Fix: detect repeated top/bottom lines and remove them.

- Broken line wraps in paragraphs:
  - Problem: many sentences were split by PDF line breaks.
  - Fix: merge single line breaks into spaces.

- Too much whitespace:
  - Problem: extra spaces and blank lines made text noisy.
  - Fix: normalize spaces and reduce blank lines.

- Table-like pages:
  - Problem: some tables became flat text.
  - Fix: kept PyMuPDF output and cleaned it; for important tables, use `pdfplumber`.

- Near-empty/scanned pages:
  - Problem: some pages had little machine-readable text.
  - Fix: count pages with `< 50` chars and report them (7 pages in this run).

## Reproducibility
```bash
python scripts/extract.py data/pdfs
```
