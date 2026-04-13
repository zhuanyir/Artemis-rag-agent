import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

CORPUS_PATH = DATA_DIR / "corpus.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
MIN_TEXT_LENGTH = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def main():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    chunks = []
    chunk_id = 0
    skipped_empty = 0
    skipped_short = 0

    for entry in corpus:
        source = entry.get("source", "unknown")
        page = entry.get("page", -1)
        text = entry.get("text", "").strip()

        if not text:
            skipped_empty += 1
            continue

        if len(text) < MIN_TEXT_LENGTH:
            skipped_short += 1
            continue

        text_chunks = chunk_text(text)

        for chunk in text_chunks:
            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "text": chunk
            })
            chunk_id += 1

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Original corpus entries: {len(corpus)}")
    print(f"Skipped empty entries: {skipped_empty}")
    print(f"Skipped short entries: {skipped_short}")
    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        print("\nFirst 3 chunks preview:\n")
        for chunk in chunks[:3]:
            print(f"chunk_id: {chunk['chunk_id']}")
            print(f"source: {chunk['source']}")
            print(f"page: {chunk['page']}")
            print(f"text preview: {chunk['text'][:200]!r}")
            print()


if __name__ == "__main__":
    main()