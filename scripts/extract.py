from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import fitz

EMPTY_PAGE_THRESHOLD = 50
SAMPLE_SIZE = 10


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def detect_repeated_headers_footers(
    raw_page_texts: list[str],
    top_n: int = 2,
    bottom_n: int = 2,
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

    min_repeat = max(2, int(len(raw_page_texts) * 0.5))
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


def extract_one_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    with fitz.open(pdf_path) as doc:
        raw_page_texts = [page.get_text("text") for page in doc]

    repeated_headers, repeated_footers = detect_repeated_headers_footers(raw_page_texts)
    rows: list[dict[str, Any]] = []

    for page_number, raw_text in enumerate(raw_page_texts, start=1):
        cleaned = remove_headers_footers(raw_text, repeated_headers, repeated_footers)
        cleaned = normalize_text(cleaned)
        rows.append(
            {
                "source": pdf_path.name,
                "page": page_number,
                "char_count": len(cleaned),
                "text": cleaned,
            }
        )

    return rows


def save_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PDF text into corpus/sample JSON files.")
    parser.add_argument("input_dir", type=Path, help="Folder containing PDF files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output folder for corpus.json and sample.json (default: data).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    corpus_path = output_dir / "corpus.json"
    sample_path = output_dir / "sample.json"

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {input_dir}")

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    corpus_rows: list[dict[str, Any]] = []
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        corpus_rows.extend(extract_one_pdf(pdf_path))

    sample_rows = corpus_rows[:SAMPLE_SIZE]
    save_json(corpus_rows, corpus_path)
    save_json(sample_rows, sample_path)

    document_count = len(pdf_files)
    total_pages = len(corpus_rows)
    total_characters = sum(row["char_count"] for row in corpus_rows)
    avg_characters_per_page = total_characters / total_pages if total_pages else 0.0
    near_empty_pages = sum(1 for row in corpus_rows if row["char_count"] < EMPTY_PAGE_THRESHOLD)

    print("\nCorpus stats")
    print("-" * 40)
    print(f"Number of documents (PDFs): {document_count}")
    print(f"Total pages: {total_pages}")
    print(f"Total characters: {total_characters}")
    print(f"Average characters per page: {avg_characters_per_page:.2f}")
    print(f"Number of empty or near-empty pages (< 50 characters): {near_empty_pages}")
    print(f"\nSaved: {corpus_path}")
    print(f"Saved: {sample_path}")


if __name__ == "__main__":
    main()
