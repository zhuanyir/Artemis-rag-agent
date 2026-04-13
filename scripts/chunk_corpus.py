import json
from pathlib import Path
import tiktoken

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

CORPUS_PATH = DATA_DIR / "corpus.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"

# =========================
# Config
# =========================
MODEL_NAME = "gpt-4o-mini"

# Token-based chunking settings
# Start with these values to stay near your previous chunk count
CHUNK_SIZE_TOKENS = 220
CHUNK_OVERLAP_TOKENS = 40

# Skip entries that are empty or too short
MIN_TEXT_LENGTH = 50

# Tokenizer
enc = tiktoken.encoding_for_model(MODEL_NAME)


def chunk_text_by_tokens(
    text: str,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS
) -> list[str]:
    """
    Split text into overlapping chunks based on token count.
    """
    text = text.strip()
    if not text:
        return []

    tokens = enc.encode(text)
    if not tokens:
        return []

    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    chunks = []
    step = chunk_size_tokens - overlap_tokens
    start = 0

    while start < len(tokens):
        chunk_tokens = tokens[start:start + chunk_size_tokens]
        chunk_text = enc.decode(chunk_tokens).strip()

        if chunk_text:
            chunks.append(chunk_text)

        start += step

    return chunks


def load_corpus(path: Path) -> list[dict]:
    """
    Load corpus.json
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chunks(chunks: list[dict], path: Path) -> None:
    """
    Save chunks.json
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main():
    corpus = load_corpus(CORPUS_PATH)

    chunks = []
    chunk_id = 0
    skipped_empty = 0
    skipped_short = 0

    for entry in corpus:
        source = entry.get("source", "unknown")
        page = entry.get("page", -1)
        text = entry.get("text", "").strip()

        # Skip empty entries
        if not text:
            skipped_empty += 1
            continue

        # Skip very short entries
        if len(text) < MIN_TEXT_LENGTH:
            skipped_short += 1
            continue

        text_chunks = chunk_text_by_tokens(text)

        for chunk in text_chunks:
            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "text": chunk
            })
            chunk_id += 1

    save_chunks(chunks, CHUNKS_PATH)

    print(f"Original corpus entries: {len(corpus)}")
    print(f"Skipped empty entries: {skipped_empty}")
    print(f"Skipped short entries: {skipped_short}")
    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        print("\nFirst 3 chunks preview:\n")
        for chunk in chunks[:3]:
            token_count = len(enc.encode(chunk["text"]))
            print(f"chunk_id: {chunk['chunk_id']}")
            print(f"source: {chunk['source']}")
            print(f"page: {chunk['page']}")
            print(f"token_count: {token_count}")
            print(f"text preview: {chunk['text'][:200]!r}")
            print()


if __name__ == "__main__":
    main()