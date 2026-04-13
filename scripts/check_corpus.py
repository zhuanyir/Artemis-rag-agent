import json
from pathlib import Path

CORPUS_PATH = Path("../data/corpus.json")  # 如果你的 corpus.json 不在 data/，就改路径

def main():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Total entries: {len(corpus)}")

    if not corpus:
        print("Corpus is empty.")
        return

    print("\nFirst 3 entries preview:\n")
    for i, entry in enumerate(corpus[:3]):
        print(f"Entry {i}:")
        print(f"  keys: {list(entry.keys())}")
        print(f"  source: {entry.get('source')}")
        print(f"  page: {entry.get('page')}")
        text = entry.get("text", "")
        print(f"  text preview: {text[:200]!r}")
        print()

    missing_source = 0
    missing_page = 0
    missing_text = 0
    short_text = 0

    for entry in corpus:
        if "source" not in entry:
            missing_source += 1
        if "page" not in entry:
            missing_page += 1
        if "text" not in entry or not entry["text"].strip():
            missing_text += 1
        elif len(entry["text"].strip()) < 50:
            short_text += 1

    print("Summary:")
    print(f"  Missing source: {missing_source}")
    print(f"  Missing page: {missing_page}")
    print(f"  Missing/empty text: {missing_text}")
    print(f"  Very short text (<50 chars): {short_text}")

if __name__ == "__main__":
    main()