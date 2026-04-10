# Artemis II Knowledge Navigator

This project extracts text from a collection of NASA Artemis II PDF documents and builds a JSON corpus for a chatbot knowledge base.

## Project Structure

- `scripts/extract.py`: extracts text from PDF files
- `data/pdfs/`: input PDF files
- `data/corpus.json`: full extracted corpus
- `data/sample.json`: sample output with the first 10 pages
- `scripts/EXTRACTION_REPORT.md`: report about extraction choices and observations

## Environment

This project uses Python 3.12 and `uv`.

### Create and activate the environment

On Windows Git Bash:

```bash
python -m uv venv --python 3.12
source .venv/Scripts/activate
python -m uv pip install --python .venv/Scripts/python.exe pymupdf
```
