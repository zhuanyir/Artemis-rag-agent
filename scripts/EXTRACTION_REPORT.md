# EXTRACTION_REPORT

## 1. Overview
This report summarizes the PDF text extraction run for the Artemis II knowledge corpus.

- Extraction date: 2026-04-10
- Project: Artemis_II_Knowledge_Navigator
- Extraction script: `scripts/extract.py`
- Extraction engine: PyMuPDF (`fitz`)
- Input folder: `data/pdfs`
- Output files:
  - `data/corpus.json`
  - `data/sample.json`

## 2. Output Format
Both output files follow the same flat page-level schema:

```json
[
  {
    "source": "example.pdf",
    "page": 1,
    "char_count": 1234,
    "text": "Extracted and cleaned page text..."
  }
]
```

## 3. Corpus Statistics
Run-level statistics printed by the script:

- Number of documents (PDFs): 3
- Total pages: 253
- Total characters: 493963
- Average characters per page: 1952.42
- Number of empty or near-empty pages (< 50 chars): 7

## 4. Text Cleaning Applied
The extraction pipeline includes baseline cleanup to improve downstream RAG quality:

- Whitespace normalization:
  - Collapses repeated spaces/tabs
  - Normalizes line endings
  - Reduces excessive blank lines
- Repeated header/footer removal:
  - Detects repeated top/bottom lines across pages
  - Removes detected repeated lines from each page
- Paragraph line-break repair:
  - Converts single newlines (mid-paragraph PDF wraps) into spaces
  - Keeps paragraph boundaries where double newlines exist

## 5. Sampling Strategy
`sample.json` contains the first 10 extracted pages from the full corpus in the same schema as `corpus.json` for quick manual inspection.

## 6. Notes and Limitations
- Some PDFs may include OCR artifacts, ligatures, or layout-induced text order issues.
- Table-heavy pages may still require `pdfplumber` for better structure preservation.
- Header/footer detection is heuristic-based and may miss edge cases in highly variable templates.

## 7. Reproducibility
Run command used:

```bash
python scripts/extract.py data/pdfs
```

Optional custom output directory:

```bash
python scripts/extract.py data/pdfs --output-dir data
```
