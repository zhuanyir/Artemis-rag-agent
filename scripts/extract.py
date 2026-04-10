from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import fitz


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CORPUS_PATH = DATA_DIR / "corpus.json"
SAMPLE_PATH = DATA_DIR / "sample.json"

EMPTY_PAGE_THRESHOLD = 50


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.strip() for line in text.split("\n")]
    lines = [re.sub(r"[ \t]+", " ", line) for line in lines]

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def detect_repeated_headers_footers(
    raw_page_texts: list[str],
    top_n: int = 2,
    bottom_n: int = 2,
    min_repeat: int = 3,
) -> tuple[set[str], set[str]]:
    header_counts: dict[str, int] = {}
    footer_counts: dict[str, int] = {}

    for text in raw_page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue

        for line in lines[:top_n]:
            header_counts[line] = header_counts.get(line, 0) + 1

        for line in lines[-bottom_n:]:
            footer_counts[line] = footer_counts.get(line, 0) + 1

    repeated_headers = {line for line, count in header_counts.items() if count >= min_repeat}
    repeated_footers = {line for line, count in footer_counts.items() if count >= min_repeat}

    return repeated_headers, repeated_footers


def remove_headers_footers(
    raw_text: str,
    repeated_headers: set[str],
    repeated_footers: set[str],
    top_n: int = 2,
    bottom_n: int = 2,
) -> str:
    lines = raw_text.splitlines()
    stripped = [line.strip() for line in lines]

    for i in range(min(top_n, len(lines))):
        if stripped[i] in repeated_headers:
            lines[i] = ""

    start = max(0, len(lines) - bottom_n)
    for i in range(start, len(lines)):
        if stripped[i] in repeated_footers:
            lines[i] = ""

    return "\n".join(lines)


def extract_one_pdf(pdf_path: Path) -> dict[str, Any]:
    doc = fitz.open(pdf_path)

    raw_page_texts: list[str] = []
    for page in doc:
        raw_page_texts.append(page.get_text("text"))

    repeated_headers, repeated_footers = detect_repeated_headers_footers(raw_page_texts)

    pages: list[dict[str, Any]] = []
    total_characters = 0
    near_empty_pages = 0

    for page_number, raw_text in enumerate(raw_page_texts, start=1):
        cleaned = remove_headers_footers(raw_text, repeated_headers, repeated_footers)
        cleaned = normalize_text(cleaned)

        char_count = len(cleaned)
        is_near_empty = char_count < EMPTY_PAGE_THRESHOLD

        if is_near_empty:
            near_empty_pages += 1

        total_characters += char_count

        pages.append(
            {
                "page_number": page_number,
                "text": cleaned,
                "char_count": char_count,
                "is_near_empty": is_near_empty,
            }
        )

    doc.close()

    return {
        "document_id": pdf_path.stem,
        "filename": pdf_path.name,
        "source_path": str(pdf_path),
        "page_count": len(pages),
        "total_characters": total_characters,
        "empty_or_near_empty_pages": near_empty_pages,
        "pages": pages,
    }


def save_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_sample(documents: list[dict[str, Any]], limit: int = 10) -> dict[str, Any]:
    sample_pages: list[dict[str, Any]] = []

    for document in documents:
        for page in document["pages"]:
            sample_pages.append(
                {
                    "document_id": document["document_id"],
                    "filename": document["filename"],
                    "page_number": page["page_number"],
                    "char_count": page["char_count"],
                    "text": page["text"],
                }
            )
            if len(sample_pages) >= limit:
                return {
                    "sample_size": len(sample_pages),
                    "pages": sample_pages,
                }

    return {
        "sample_size": len(sample_pages),
        "pages": sample_pages,
    }


def main() -> None:
    if not PDF_DIR.exists() or not PDF_DIR.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {PDF_DIR}")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    documents: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        documents.append(extract_one_pdf(pdf_path))

    document_count = len(documents)
    total_pages = sum(doc["page_count"] for doc in documents)
    total_characters = sum(doc["total_characters"] for doc in documents)
    empty_pages = sum(doc["empty_or_near_empty_pages"] for doc in documents)
    avg_characters_per_page = total_characters / total_pages if total_pages else 0.0

    corpus = {
        "metadata": {
            "source_folder": str(PDF_DIR),
            "document_count": document_count,
            "total_pages": total_pages,
            "total_characters": total_characters,
            "average_characters_per_page": round(avg_characters_per_page, 2),
            "empty_or_near_empty_pages": empty_pages,
            "empty_page_threshold": EMPTY_PAGE_THRESHOLD,
        },
        "documents": documents,
    }

    sample = build_sample(documents, limit=10)

    save_json(corpus, CORPUS_PATH)
    save_json(sample, SAMPLE_PATH)

    print("\nCorpus stats")
    print("-" * 40)
    print(f"Number of documents (PDFs): {document_count}")
    print(f"Total pages: {total_pages}")
    print(f"Total characters: {total_characters}")
    print(f"Average characters per page: {round(avg_characters_per_page, 2)}")
    print(f"Number of empty or near-empty pages (< 50 characters): {empty_pages}")

    print(f"\nSaved: {CORPUS_PATH}")
    print(f"Saved: {SAMPLE_PATH}")


if __name__ == "__main__":
    main()