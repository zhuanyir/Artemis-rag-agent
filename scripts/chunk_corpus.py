#!/usr/bin/env python3
"""
Chunk corpus for Artemis II Knowledge Navigator
Splits documents into chunks for embedding generation
"""

import os
import re
import json
from pathlib import Path

def safe_text(value):
    """Safely convert input to UTF-8 string"""
    if value is None:
        return ""
    
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return ""
    
    try:
        value = value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return ""
    
    return value

def normalize_text(text):
    """Normalize text by removing extra whitespace"""
    if text is None:
        return ""
    
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    
    text = safe_text(text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def get_document_title(document):
    """Extract readable title from document - uses 'source' field as primary"""
    # Priority 1: Use 'source' filename (e.g., "a2-reference-guide-012825.pdf")
    if 'source' in document and document['source']:
        source = document['source']
        # Remove file extension
        title = source.replace('.pdf', '').replace('.txt', '').replace('.md', '')
        # Convert hyphens and underscores to spaces
        title = title.replace('-', ' ').replace('_', ' ')
        # Convert to title case
        title = title.title()
        return safe_text(title)
    
    # Priority 2: Use 'title' field if available
    if 'title' in document and document['title']:
        return safe_text(document['title'])
    
    return "Untitled"

def get_document_content(document):
    """Extract content from document - uses 'text' field as primary"""
    # Priority 1: Use 'text' field
    if 'text' in document and document['text']:
        content = document['text']
        if isinstance(content, str) and content.strip():
            return content
    
    # Priority 2: Use 'content' field
    if 'content' in document and document['content']:
        content = document['content']
        if isinstance(content, str) and content.strip():
            return content
    
    return ""

def get_document_page(document):
    """Extract page number from document"""
    if 'page' in document:
        return document['page']
    return None

def chunk_paragraph_by_sentences(paragraph, max_chunk_size=800, min_chunk_size=300, overlap_sentences=1):
    """
    Chunk a paragraph by sentences with overlap
    - max_chunk_size: Maximum characters per chunk
    - min_chunk_size: Minimum size before creating a chunk (for merging)
    - overlap_sentences: Number of sentences to overlap between chunks
    """
    if paragraph is None:
        return []
    
    if not isinstance(paragraph, str):
        paragraph = str(paragraph)
    
    if not paragraph.strip():
        return []
    
    # Split into sentences (detects .!? followed by space and capital letter)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # If no sentences found, treat as one chunk
    if not sentences:
        return [{'text': normalize_text(paragraph), 'size': len(paragraph), 'sentence_count': 1}]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    i = 0
    while i < len(sentences):
        sentence = normalize_text(sentences[i])
        if not sentence:
            i += 1
            continue
            
        sentence_size = len(sentence)
        
        # Check if adding this sentence would exceed max size
        will_exceed = current_size + sentence_size > max_chunk_size
        
        # Save current chunk if adding next sentence would exceed max size
        if will_exceed and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'size': current_size,
                'sentence_count': len(current_chunk)
            })
            
            # Start new chunk with overlap (carry over previous sentences)
            overlap_start = max(0, len(current_chunk) - overlap_sentences)
            current_chunk = current_chunk[overlap_start:]
            current_size = sum(len(s) for s in current_chunk)
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_size += sentence_size
        i += 1
    
    # Add the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'size': current_size,
            'sentence_count': len(current_chunk)
        })
    
    return chunks

def main():
    """Main function to chunk the corpus"""
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    input_file = data_dir / 'corpus.json'
    output_file = data_dir / 'chunks.json'
    
    print(f"Looking for input file: {input_file}")
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        return
    
    # Load corpus
    print(f"Loading corpus from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return
    
    print(f"Loaded {len(corpus)} documents")
    
    # Debug: Show structure of first document
    if corpus:
        print(f"\nFirst document keys: {list(corpus[0].keys())}")
    
    # Chunking parameters
    MAX_CHUNK_SIZE = 800      # Maximum characters per chunk
    MIN_CHUNK_SIZE = 300      # Minimum target size
    OVERLAP_SENTENCES = 1     # Number of sentences to overlap
    
    print(f"\nChunking parameters:")
    print(f"  Max chunk size: {MAX_CHUNK_SIZE} characters")
    print(f"  Min chunk size: {MIN_CHUNK_SIZE} characters")
    print(f"  Overlap: {OVERLAP_SENTENCES} sentence(s)")
    
    # ── Preserve new-format chunks already in chunks.json ────────────────────
    # New-format chunks (added via add_to_database or load_pdf) have a "source"
    # key. We must keep them or they will be lost every time this script runs.
    preserved_chunks = []
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                raw = f.read().rstrip('\x00').rstrip()
            existing = json.loads(raw)
            preserved_chunks = [c for c in existing if c.get("source")]
            if preserved_chunks:
                print(f"Preserving {len(preserved_chunks)} new-format chunks (user-added/web).")
        except Exception as e:
            print(f"Warning: could not load existing chunks.json ({e}), starting fresh.")

    all_chunks = []
    chunk_id = 0
    documents_with_content = 0

    # Process each document
    for doc_id, document in enumerate(corpus):
        # Extract metadata
        title = get_document_title(document)      # Extract title from 'source' field
        content = get_document_content(document)  # Extract content from 'text' field
        page = get_document_page(document)        # Extract page number
        
        if not content or not content.strip():
            continue
        
        documents_with_content += 1
        
        # Progress update every 50 documents
        if documents_with_content % 50 == 0:
            print(f"Processing document {documents_with_content}/{len(corpus)}: {title} (Page {page})")
        
        # Split into paragraphs (by double newlines or double line breaks)
        paragraphs = content.split('\n\n')
        
        # Process each paragraph
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph or not paragraph.strip():
                continue
            
            # Chunk the paragraph
            para_chunks = chunk_paragraph_by_sentences(
                paragraph, 
                max_chunk_size=MAX_CHUNK_SIZE,
                min_chunk_size=MIN_CHUNK_SIZE,
                overlap_sentences=OVERLAP_SENTENCES
            )
            
            # Add metadata to each chunk
            for chunk in para_chunks:
                chunk['id'] = chunk_id                         # Unique chunk identifier
                chunk['document_id'] = doc_id                  # Reference to original document index
                chunk['document_title'] = title                # Readable title for display
                chunk['page'] = page                           # KEPT: Original PDF page number
                # REMOVED: 'paragraph_index' - not needed for retrieval
                # REMOVED: 'source' - redundant with document_title
                chunk_id += 1
                all_chunks.append(chunk)
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Total documents: {len(corpus)}")
    print(f"  Documents with content: {documents_with_content}")
    print(f"  Total chunks generated: {len(all_chunks)}")
    
    if len(all_chunks) == 0:
        print("\n⚠️  WARNING: No chunks were generated!")
        return
    
    # Merge: old-format re-chunked + preserved new-format chunks
    combined_chunks = all_chunks + preserved_chunks

    # Save chunks to file
    print(f"\nSaving {len(combined_chunks)} chunks to {output_file} "
          f"({len(all_chunks)} re-chunked + {len(preserved_chunks)} preserved)...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_chunks, f, indent=2, ensure_ascii=False)
        print("✅ Successfully saved chunks!")
    except Exception as e:
        print(f"Error saving chunks: {e}")
        return
    
    # Print statistics
    if all_chunks:
        sizes = [c['size'] for c in all_chunks]
        avg_size = sum(sizes) / len(sizes)
        avg_sentences = sum(c['sentence_count'] for c in all_chunks) / len(all_chunks)
        
        print(f"\nStatistics:")
        print(f"  Total chunks: {len(all_chunks)}")
        print(f"  Average chunk size: {avg_size:.1f} characters")
        print(f"  Average sentences per chunk: {avg_sentences:.1f}")
        print(f"  Max chunk size: {max(sizes)} characters")
        print(f"  Min chunk size: {min(sizes)} characters")
        
        # Show sample chunk with new metadata structure
        print(f"\nSample chunk output (with page number):")
        sample = all_chunks[0]
        print(f"  {{")
        print(f"    'id': {sample.get('id')},")
        print(f"    'document_id': {sample.get('document_id')},")
        print(f"    'document_title': '{sample.get('document_title')}',")
        print(f"    'page': {sample.get('page')},")
        print(f"    'size': {sample.get('size')},")
        print(f"    'sentence_count': {sample.get('sentence_count')},")
        print(f"    'text': '{sample.get('text', '')[:100]}...'")
        print(f"  }}")

if __name__ == "__main__":
    main()