#!/usr/bin/env python3
"""
Chunk Browser - Interactive web interface for browsing and searching chunks
Run: python chunk_browser.py
Then open http://localhost:7860
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import gradio as gr

class ChunkBrowser:
    def __init__(self, chunks_file=None):
        if chunks_file is None:
            base_dir = Path(__file__).parent.parent
            chunks_file = base_dir / 'data' / 'chunks.json'
        
        self.chunks_file = Path(chunks_file)
        self.chunks = []
        self.documents = set()
        self.load_chunks()
    
    def load_chunks(self):
        """Load chunks from JSON file"""
        if not self.chunks_file.exists():
            print(f"❌ Error: {self.chunks_file} not found")
            return
        
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Extract unique document titles
        self.documents = sorted(set(
            chunk.get('document_title', 'Untitled') for chunk in self.chunks
        ))
        
        print(f"✅ Loaded {len(self.chunks)} chunks from {len(self.documents)} documents")
    
    def search_chunks(self, keyword: str, selected_document: str, min_size: int, max_size: int):
        """Search chunks by keyword, document, and size range"""
        
        if not self.chunks:
            return [], "No chunks loaded"
        
        results = []
        
        for chunk in self.chunks:
            # Filter by document
            if selected_document != "All Documents":
                if chunk.get('document_title', 'Untitled') != selected_document:
                    continue
            
            # Filter by size
            chunk_size = chunk.get('size', 0)
            if chunk_size < min_size or chunk_size > max_size:
                continue
            
            # Filter by keyword
            if keyword and keyword.strip():
                text = chunk.get('text', '').lower()
                keyword_lower = keyword.lower().strip()
                
                # Support exact phrase search with quotes
                if keyword_lower.startswith('"') and keyword_lower.endswith('"'):
                    # Exact phrase search
                    phrase = keyword_lower[1:-1]
                    if phrase not in text:
                        continue
                else:
                    # Keyword search (any word)
                    keywords = keyword_lower.split()
                    if not all(kw in text for kw in keywords):
                        continue
            
            results.append(chunk)
        
        # Sort by size (descending)
        results.sort(key=lambda x: x.get('size', 0), reverse=True)
        
        # Format results for display
        if not results:
            return ["<div style='text-align: center; padding: 40px; color: #666;'>🔍 No chunks found. Try different search terms or adjust filters.</div>"], "Found 0 chunks"
        
        formatted_results = []
        for i, chunk in enumerate(results[:100]):  # Limit to 100 results
            formatted = self.format_chunk(chunk, i+1, keyword)
            formatted_results.append(formatted)
        
        summary = f"📊 Found {len(results)} chunks (showing first {min(100, len(results))})"
        
        # Join all formatted results
        return formatted_results, summary
    
    def format_chunk(self, chunk: Dict, index: int, highlight_keyword: str = ""):
        """Format a single chunk for display with highlighting"""
        
        chunk_id = chunk.get('id', 'N/A')
        title = chunk.get('document_title', 'Untitled')
        size = chunk.get('size', 0)
        sentences = chunk.get('sentence_count', 0)
        text = chunk.get('text', '')
        
        # Highlight keywords
        if highlight_keyword and highlight_keyword.strip() and highlight_keyword != '""':
            # Remove quotes for exact phrase highlighting
            search_term = highlight_keyword.strip()
            if search_term.startswith('"') and search_term.endswith('"'):
                search_term = search_term[1:-1]
            
            # Simple highlighting (case-insensitive)
            escaped_term = re.escape(search_term)
            pattern = re.compile(f'({escaped_term})', re.IGNORECASE)
            text = pattern.sub(r'<mark style="background-color: #ffeb3b; padding: 0 2px; border-radius: 3px;">\1</mark>', text)
        
        # Create HTML card
        html = f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px; flex-wrap: wrap;">
                <div>
                    <span style="font-weight: bold; color: #2c3e50; font-size: 14px;">📄 {title}</span>
                </div>
                <div>
                    <span style="background-color: #3498db; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px;">
                        ID: {chunk_id}
                    </span>
                </div>
            </div>
            
            <div style="margin-bottom: 12px;">
                <span style="background-color: #ecf0f1; padding: 3px 8px; border-radius: 12px; font-size: 11px; margin-right: 8px;">
                    📏 {size} chars
                </span>
                <span style="background-color: #ecf0f1; padding: 3px 8px; border-radius: 12px; font-size: 11px;">
                    📝 {sentences} sentences
                </span>
            </div>
            
            <div style="background-color: white; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; line-height: 1.6; font-size: 14px;">
                {text}
            </div>
        </div>
        """
        return html
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        if not self.chunks:
            return "No data loaded"
        
        sizes = [c.get('size', 0) for c in self.chunks]
        sentences = [c.get('sentence_count', 0) for c in self.chunks]
        
        stats = f"""
### 📊 Chunk Statistics

| Metric | Value |
|--------|-------|
| **Total Chunks** | {len(self.chunks)} |
| **Total Documents** | {len(self.documents)} |
| **Average Size** | {sum(sizes)/len(sizes):.1f} chars |
| **Median Size** | {sorted(sizes)[len(sizes)//2]} chars |
| **Min Size** | {min(sizes)} chars |
| **Max Size** | {max(sizes)} chars |
| **Average Sentences** | {sum(sentences)/len(sentences):.1f} |

### 📏 Size Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| 0-200 | {sum(1 for s in sizes if s <= 200)} | {sum(1 for s in sizes if s <= 200)/len(sizes)*100:.1f}% |
| 201-400 | {sum(1 for s in sizes if 201 <= s <= 400)} | {sum(1 for s in sizes if 201 <= s <= 400)/len(sizes)*100:.1f}% |
| 401-600 | {sum(1 for s in sizes if 401 <= s <= 600)} | {sum(1 for s in sizes if 401 <= s <= 600)/len(sizes)*100:.1f}% |
| 601-800 | {sum(1 for s in sizes if 601 <= s <= 800)} | {sum(1 for s in sizes if 601 <= s <= 800)/len(sizes)*100:.1f}% |
| 801-1000 | {sum(1 for s in sizes if 801 <= s <= 1000)} | {sum(1 for s in sizes if 801 <= s <= 1000)/len(sizes)*100:.1f}% |
| 1000+ | {sum(1 for s in sizes if s > 1000)} | {sum(1 for s in sizes if s > 1000)/len(sizes)*100:.1f}% |
"""
        
        return stats

def create_interface():
    """Create Gradio interface"""
    
    browser = ChunkBrowser()
    
    if not browser.chunks:
        return gr.Interface(
            fn=lambda: "Error: No chunks loaded. Please run chunk_corpus.py first.",
            inputs=[],
            outputs="text"
        )
    
    with gr.Blocks(title="Chunk Browser - Artemis II Knowledge Navigator", css="""
        .gradio-container { max-width: 1200px; margin: auto; }
        .markdown-text { font-size: 14px; }
    """) as demo:
        
        gr.Markdown("""
        # 🔍 Chunk Browser
        ### Browse, search, and analyze text chunks from the Artemis II Knowledge Navigator
        """)
        
        with gr.Tabs():
            with gr.TabItem("🔍 Search & Browse"):
                # Search inputs
                with gr.Row():
                    with gr.Column(scale=2):
                        keyword_input = gr.Textbox(
                            label="🔎 Search Keywords",
                            placeholder="Enter keywords (e.g., 'Orion SLS') or use quotes for exact phrase: 'exact phrase'",
                            lines=1
                        )
                    
                    with gr.Column(scale=1):
                        doc_filter = gr.Dropdown(
                            label="📚 Filter by Document",
                            choices=["All Documents"] + browser.documents,
                            value="All Documents"
                        )
                
                with gr.Row():
                    with gr.Column():
                        min_size_input = gr.Slider(
                            label="Minimum Size (characters)",
                            minimum=0,
                            maximum=2000,
                            value=0,
                            step=50
                        )
                    
                    with gr.Column():
                        max_size_input = gr.Slider(
                            label="Maximum Size (characters)",
                            minimum=0,
                            maximum=2000,
                            value=2000,
                            step=50
                        )
                
                # Search button
                search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                
                # Status and results
                status_text = gr.Textbox(label="Status", interactive=False)
                results_html = gr.HTML(label="Results")
                
                # Example searches as buttons
                gr.Markdown("### 💡 Example Searches (Click to try):")
                
                with gr.Row():
                    btn1 = gr.Button("Artemis", size="sm")
                    btn2 = gr.Button("Orion spacecraft", size="sm")
                    btn3 = gr.Button('"Space Launch System"', size="sm")
                    btn4 = gr.Button("NASA Kennedy", size="sm")
                    btn5 = gr.Button("moon", size="sm")
                
                # Connect example buttons to search
                def set_search_example_1():
                    return "Artemis", "All Documents", 0, 2000
                
                def set_search_example_2():
                    return "Orion spacecraft", "All Documents", 0, 2000
                
                def set_search_example_3():
                    return '"Space Launch System"', "All Documents", 300, 800
                
                def set_search_example_4():
                    return "NASA Kennedy", "All Documents", 0, 2000
                
                def set_search_example_5():
                    return "moon", "All Documents", 400, 1000
                
                btn1.click(set_search_example_1, outputs=[keyword_input, doc_filter, min_size_input, max_size_input])
                btn2.click(set_search_example_2, outputs=[keyword_input, doc_filter, min_size_input, max_size_input])
                btn3.click(set_search_example_3, outputs=[keyword_input, doc_filter, min_size_input, max_size_input])
                btn4.click(set_search_example_4, outputs=[keyword_input, doc_filter, min_size_input, max_size_input])
                btn5.click(set_search_example_5, outputs=[keyword_input, doc_filter, min_size_input, max_size_input])
                
                # Perform search when button clicked
                search_btn.click(
                    browser.search_chunks,
                    inputs=[keyword_input, doc_filter, min_size_input, max_size_input],
                    outputs=[results_html, status_text]
                )
                
                # Also search when Enter is pressed in textbox
                keyword_input.submit(
                    browser.search_chunks,
                    inputs=[keyword_input, doc_filter, min_size_input, max_size_input],
                    outputs=[results_html, status_text]
                )
                
                # Auto-search when document filter changes
                doc_filter.change(
                    browser.search_chunks,
                    inputs=[keyword_input, doc_filter, min_size_input, max_size_input],
                    outputs=[results_html, status_text]
                )
                
                # Auto-search when size sliders change
                min_size_input.change(
                    browser.search_chunks,
                    inputs=[keyword_input, doc_filter, min_size_input, max_size_input],
                    outputs=[results_html, status_text]
                )
                
                max_size_input.change(
                    browser.search_chunks,
                    inputs=[keyword_input, doc_filter, min_size_input, max_size_input],
                    outputs=[results_html, status_text]
                )
            
            with gr.TabItem("📊 Statistics"):
                stats_text = gr.Markdown(browser.get_statistics())
                refresh_stats = gr.Button("🔄 Refresh Statistics")
                refresh_stats.click(
                    lambda: browser.get_statistics(),
                    outputs=stats_text
                )
            
            with gr.TabItem("📄 Raw Data"):
                raw_data = gr.JSON(label="Raw Chunks Data (first 100)")
                show_raw = gr.Button("📋 Show Raw Data")
                
                def show_raw_data():
                    return browser.chunks[:100]
                
                show_raw.click(show_raw_data, outputs=raw_data)
        
        gr.Markdown("""
        ---
        ### ℹ️ Tips:
        - **Click the example buttons** above to try different searches
        - Use **quotes** for exact phrase search: `"exact phrase"`
        - Adjust **size sliders** to find chunks of specific lengths
        - Use **document filter** to focus on specific sources
        - Results show **keyword highlighting** in yellow
        - **Auto-search**: Results update automatically when you change filters
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)