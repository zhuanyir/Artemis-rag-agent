#!/usr/bin/env python3
"""
Chunk corpus for Artemis II Knowledge Navigator
Optimized for larger chunks (target 500-800 characters)
"""

import os
import re
import json
from typing import List, Dict, Any
from pathlib import Path

def safe_text(value):
    """安全地将输入转换为UTF-8字符串"""
    if value is None:
        return ""
    
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception as e:
            print(f"Warning: Could not convert {type(value)} to string: {e}")
            return ""
    
    try:
        value = value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Warning: Could not normalize text: {e}")
        return ""
    
    return value

def normalize_text(text):
    """Normalize text by removing extra whitespace and special characters"""
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
    
    # Keep basic punctuation and alphanumeric characters
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]\{\}]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def get_document_content(document):
    """Extract content from document, trying different possible field names"""
    content_fields = ['content', 'text', 'body', 'article', 'description', 'full_text']
    
    for field in content_fields:
        if field in document and document[field]:
            content = document[field]
            if isinstance(content, str) and content.strip():
                return content
    
    for key, value in document.items():
        if isinstance(value, str) and len(value) > 100:
            return value
    
    return ""

def get_document_title(document):
    """Extract title from document"""
    title_fields = ['title', 'name', 'heading', 'header', 'subject']
    
    for field in title_fields:
        if field in document and document[field]:
            title = document[field]
            if isinstance(title, str) and title.strip():
                return safe_text(title)
    
    return "Untitled"

def chunk_paragraph_by_sentences(paragraph, max_chunk_size=800, min_chunk_size=300, overlap_sentences=1):
    """
    Chunk a paragraph by sentences with improved size targeting
    
    Args:
        paragraph: Paragraph text to chunk
        max_chunk_size: Maximum characters per chunk (increased to 800)
        min_chunk_size: Minimum characters before creating a chunk (new)
        overlap_sentences: Number of sentences to overlap between chunks
    """
    if paragraph is None:
        return []
    
    if not isinstance(paragraph, str):
        paragraph = str(paragraph)
    
    if not paragraph.strip():
        return []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
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
        
        # Save current chunk if:
        # 1. Adding next sentence would exceed max size AND we have some content
        # 2. Current chunk already meets minimum size requirement
        if will_exceed and current_chunk:
            # Only save if current chunk meets minimum size
            if current_size >= min_chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'size': current_size,
                    'sentence_count': len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - overlap_sentences)
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(s) for s in current_chunk)
            else:
                # Current chunk is too small, keep building it
                # (don't save yet, try to add more sentences)
                pass
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_size += sentence_size
        i += 1
    
    # Add the last chunk (even if smaller than min_chunk_size)
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'size': current_size,
            'sentence_count': len(current_chunk)
        })
    
    # Post-process: merge very small chunks with previous/next
    chunks = merge_small_chunks(chunks, min_chunk_size)
    
    return chunks

def merge_small_chunks(chunks, min_size=300):
    """Merge small chunks with neighboring chunks"""
    if len(chunks) <= 1:
        return chunks
    
    merged = []
    i = 0
    
    while i < len(chunks):
        current = chunks[i]
        
        # If current chunk is too small and there's a next chunk
        if current['size'] < min_size and i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            
            # Merge with next chunk
            merged_text = current['text'] + ' ' + next_chunk['text']
            merged_size = current['size'] + next_chunk['size'] + 1  # +1 for space
            merged_sentences = current['sentence_count'] + next_chunk['sentence_count']
            
            merged.append({
                'text': merged_text,
                'size': merged_size,
                'sentence_count': merged_sentences
            })
            
            # Skip the next chunk since it's merged
            i += 2
        else:
            merged.append(current)
            i += 1
    
    return merged

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
    
    # Parameters for chunking
    MAX_CHUNK_SIZE = 800  # Increased from 500 to 800
    MIN_CHUNK_SIZE = 400  # Target minimum size
    OVERLAP_SENTENCES = 1
    
    print(f"Chunking parameters:")
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
        title = get_document_title(document)
        content = get_document_content(document)
        
        if not content or not content.strip():
            continue
        
        documents_with_content += 1
        
        if doc_id % 50 == 0:  # Progress update every 50 docs
            print(f"Processing document {doc_id + 1}/{len(corpus)}: {title}")
        
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        
        # Process each paragraph
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph or not paragraph.strip():
                continue
            
            # Chunk the paragraph with new parameters
            para_chunks = chunk_paragraph_by_sentences(
                paragraph, 
                max_chunk_size=MAX_CHUNK_SIZE,
                min_chunk_size=MIN_CHUNK_SIZE,
                overlap_sentences=OVERLAP_SENTENCES
            )
            
            # Add metadata to chunks
            for chunk in para_chunks:
                chunk['id'] = chunk_id
                chunk['document_id'] = doc_id
                chunk['document_title'] = title
                chunk['paragraph_index'] = para_idx
                chunk_id += 1
                all_chunks.append(chunk)
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Total documents: {len(corpus)}")
    print(f"  Documents with content: {documents_with_content}")
    print(f"  Documents without content: {len(corpus) - documents_with_content}")
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
        print(f"  Median chunk size: {sorted(sizes)[len(sizes)//2]} characters")
        
        # Size distribution
        print(f"\nSize Distribution:")
        ranges = [
            (0, 200, "Very Small"),
            (201, 400, "Small"),
            (401, 600, "Medium (Target)"),
            (601, 800, "Large"),
            (801, 1000, "Very Large"),
            (1001, 9999, "Huge")
        ]
        
        for min_s, max_s, label in ranges:
            count = sum(1 for s in sizes if min_s <= s <= max_s)
            pct = (count / len(sizes)) * 100
            bar = "█" * int(pct / 2)
            print(f"  {label:20} : {count:4} chunks ({pct:5.1f}%) {bar}")

if __name__ == "__main__":
    main()