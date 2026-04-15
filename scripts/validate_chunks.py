#!/usr/bin/env python3
"""
Validate and inspect generated chunks
"""

import json
from pathlib import Path

def validate_chunks():
    """验证并显示chunks.json的内容"""
    
    base_dir = Path(__file__).parent.parent
    chunks_file = base_dir / 'data' / 'chunks.json'
    
    if not chunks_file.exists():
        print(f"❌ Error: chunks.json not found at {chunks_file}")
        print("Please run chunk_corpus.py first")
        return
    
    print(f"📂 Loading chunks from: {chunks_file}\n")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"{'='*60}")
    print(f"📊 TOTAL CHUNKS: {len(chunks)}")
    print(f"{'='*60}\n")
    
    # 统计信息
    documents = set(chunk.get('document_title') for chunk in chunks)
    print(f"📚 Documents covered: {len(documents)}")
    
    # 计算平均大小
    avg_size = sum(c.get('size', 0) for c in chunks) / len(chunks)
    avg_sentences = sum(c.get('sentence_count', 0) for c in chunks) / len(chunks)
    
    print(f"📏 Average chunk size: {avg_size:.1f} characters")
    print(f"📝 Average sentences per chunk: {avg_sentences:.1f}\n")
    
    # 显示前3个块的示例
    print(f"{'='*60}")
    print("🔍 FIRST 3 CHUNKS PREVIEW:")
    print(f"{'='*60}")
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"🆔 ID: {chunk.get('id')}")
        print(f"📄 Document: {chunk.get('document_title')}")
        print(f"📏 Size: {chunk.get('size')} chars, {chunk.get('sentence_count')} sentences")
        print(f"📍 Paragraph index: {chunk.get('paragraph_index')}")
        print(f"📝 Text preview:")
        text = chunk.get('text', '')
        # 显示前200个字符，并适当换行
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"   {preview}")
    
    # 可选：显示一些统计图表
    print(f"\n{'='*60}")
    print("📈 CHUNK SIZE DISTRIBUTION:")
    print(f"{'='*60}")
    
    size_ranges = [
        (0, 100, "Very Small (0-100)"),
        (101, 300, "Small (101-300)"),
        (301, 500, "Medium (301-500)"),
        (501, 800, "Large (501-800)"),
        (801, 1000, "Very Large (801-1000)"),
        (1001, 9999, "Huge (1000+)")
    ]
    
    for min_size, max_size, label in size_ranges:
        count = sum(1 for c in chunks if min_size <= c.get('size', 0) <= max_size)
        percentage = (count / len(chunks)) * 100
        bar = "█" * int(percentage / 2)
        print(f"{label:20} : {count:4} chunks ({percentage:5.1f}%) {bar}")
    
    # 保存到文本文件（可选）
    save_report = input(f"\n💾 Save full report to file? (y/n): ").lower().strip()
    if save_report == 'y':
        report_file = base_dir / 'data' / 'chunks_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Chunks Report\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write(f"Documents: {len(documents)}\n")
            f.write(f"Average size: {avg_size:.1f} chars\n")
            f.write(f"Average sentences: {avg_sentences:.1f}\n\n")
            
            for chunk in chunks:
                f.write(f"Chunk {chunk.get('id')}:\n")
                f.write(f"  Document: {chunk.get('document_title')}\n")
                f.write(f"  Size: {chunk.get('size')} chars\n")
                f.write(f"  Text: {chunk.get('text')}\n")
                f.write("-" * 40 + "\n")
        
        print(f"✅ Report saved to: {report_file}")

if __name__ == "__main__":
    validate_chunks()