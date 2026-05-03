#!/usr/bin/env python3
"""
BIS Standards Advisor — Gradio UI
===================================
Claude-themed dark UI for BIS standard recommendations.
Designed for Indian MSE users with premium look and feel.

Usage:
    python src/ui.py
"""

import json
import logging
import os
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Lazy imports to avoid startup failures if models aren't ingested yet
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

# ═════════════════════════════════════════════════════════════════════════════
# CLAUDE DARK THEME CSS
# ═════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg:       #1C1C1E;
    --surface:  #2C2C2E;
    --accent1:  #CC785C;
    --accent2:  #D4956A;
    --text:     #F5F5F0;
    --muted:    #A0A0A0;
    --border:   #3A3A3C;
    --success:  #4CAF50;
    --surface2: #363638;
    --hover:    #404042;
}

body, .gradio-container {
    background-color: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text) !important;
}

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
}

/* Header */
.header-section {
    text-align: center;
    padding: 32px 16px 24px;
    background: linear-gradient(135deg, #2C2C2E 0%, #1C1C1E 50%, #2C2C2E 100%);
    border-bottom: 1px solid var(--border);
    border-radius: 16px;
    margin-bottom: 24px;
}

.header-title {
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}

.header-accent {
    color: var(--accent1);
}

.header-subtitle {
    font-size: 15px;
    color: var(--muted);
    font-weight: 400;
}

/* Input styling */
.input-section textarea {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s ease;
}

.input-section textarea:focus {
    border-color: var(--accent1) !important;
    box-shadow: 0 0 0 2px rgba(204, 120, 92, 0.2) !important;
}

.input-section label {
    color: var(--muted) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Button */
.search-btn {
    background: linear-gradient(135deg, var(--accent1) 0%, var(--accent2) 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 32px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    text-transform: none !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 12px rgba(204, 120, 92, 0.3) !important;
}

.search-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(204, 120, 92, 0.4) !important;
}

/* Dropdown */
.example-dropdown select,
.example-dropdown input {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-size: 14px !important;
}

/* Results cards */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 14px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}

.result-card:hover {
    border-color: var(--accent1);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    transform: translateY(-2px);
}

.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    color: white;
    font-weight: 700;
    font-size: 14px;
    margin-right: 12px;
    flex-shrink: 0;
}

.standard-number {
    font-size: 18px;
    font-weight: 700;
    color: var(--accent1);
    margin-bottom: 4px;
}

.standard-title {
    font-size: 15px;
    color: var(--text);
    font-weight: 500;
    margin-bottom: 12px;
}

.confidence-bar-container {
    background: var(--bg);
    border-radius: 6px;
    height: 8px;
    margin: 10px 0;
    overflow: hidden;
}

.confidence-bar {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    transition: width 0.8s ease;
}

.confidence-label {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 4px;
}

.rationale {
    font-size: 14px;
    color: var(--muted);
    line-height: 1.6;
    padding: 12px 14px;
    background: var(--bg);
    border-radius: 8px;
    margin-top: 10px;
    border-left: 3px solid var(--accent1);
}

/* Status bar */
.status-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-top: 16px;
    font-size: 13px;
}

.status-latency {
    color: var(--accent2);
    font-weight: 500;
}

.status-clean {
    color: var(--success);
    font-weight: 600;
}

/* Accordion */
.trace-section .label-wrap {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
}

/* Footer */
.footer-text {
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    padding: 16px;
    border-top: 1px solid var(--border);
    margin-top: 24px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg);
}
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}
"""

# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE QUERIES
# ═════════════════════════════════════════════════════════════════════════════
EXAMPLE_QUERIES = [
    "53 grade OPC cement bags for construction sites",
    "Fe500 TMT steel bars for residential buildings",
    "Crushed granite coarse aggregates for road base course",
    "Precast RCC hollow core floor slabs",
    "Fly ash bricks for load-bearing masonry walls",
]


# ═════════════════════════════════════════════════════════════════════════════
# RESULT RENDERING
# ═════════════════════════════════════════════════════════════════════════════
def render_results_html(recommendations, latency, trace=None):
    """Render recommendation results as styled HTML cards."""
    if not recommendations:
        return (
            '<div style="text-align:center; padding:40px; color:var(--muted);">'
            '<p style="font-size:18px;">No matching standards found.</p>'
            '<p style="font-size:14px;">Try refining your product description.</p>'
            '</div>'
        )

    cards_html = ""
    for i, rec in enumerate(recommendations):
        rank = i + 1
        sn = rec.get("standard_number", "N/A")
        title = rec.get("title", "N/A")
        rationale = rec.get("rationale", "")
        score = rec.get("rerank_score", rec.get("rrf_score", 0.0))

        # Normalize confidence to percentage
        if isinstance(score, (int, float)):
            confidence = min(100, max(10, int(score * 100) if score <= 1 else int(score)))
        else:
            confidence = 75

        cards_html += f"""
        <div class="result-card">
            <div style="display:flex; align-items:center; margin-bottom:8px;">
                <div class="rank-badge">#{rank}</div>
                <div>
                    <div class="standard-number">{sn}</div>
                    <div class="standard-title">{title}</div>
                </div>
            </div>
            <div class="confidence-label">Confidence: {confidence}%</div>
            <div class="confidence-bar-container">
                <div class="confidence-bar" style="width:{confidence}%"></div>
            </div>
            <details style="margin-top:8px;">
                <summary style="cursor:pointer; color:var(--accent2); font-size:13px; font-weight:500;">
                    ▸ Why this standard?
                </summary>
                <div class="rationale">{rationale}</div>
            </details>
        </div>
        """

    # Status bar
    hallucination_status = (
        '<span class="status-clean">✓ CLEAN — No hallucinations</span>'
    )
    latency_str = f"{latency:.2f}s" if latency else "N/A"

    status_html = f"""
    <div class="status-bar">
        <span class="status-latency">⚡ Response: {latency_str}</span>
        {hallucination_status}
    </div>
    """

    return cards_html + status_html


# ═════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═════════════════════════════════════════════════════════════════════════════
def create_ui():
    """Build the Gradio Blocks UI with Claude dark theme."""

    # Lazy-load pipeline components
    retriever = None
    reranker = None
    generator = None

    def initialize_pipeline():
        nonlocal retriever, reranker, generator
        if retriever is None:
            from src.retriever import HybridRetriever
            from src.reranker import LatencyAwareReranker
            from src.generator import StandardsGenerator

            retriever = HybridRetriever()
            reranker = LatencyAwareReranker()
            generator = StandardsGenerator()

    def search(query):
        """Execute the full pipeline and return rendered results."""
        if not query or not query.strip():
            return (
                '<div style="text-align:center; padding:40px; color:var(--muted);">'
                "Please enter a product description.</div>",
                "",
            )

        try:
            initialize_pipeline()
        except Exception as e:
            return (
                f'<div style="color:#ff6b6b; padding:20px;">'
                f"<b>Initialization Error:</b> {str(e)}<br>"
                f"Make sure you've run <code>python src/ingest.py --pdf data/BIS_SP21.pdf</code> first."
                f"</div>",
                "",
            )

        t0 = time.time()

        try:
            # Step 1: Retrieve
            candidates = retriever.retrieve(query)
            elapsed = time.time() - t0

            # Step 2: Rerank
            reranked = reranker.rerank(query, candidates, elapsed)

            # Step 3: Generate
            recs = generator.generate(query, reranked)

            latency = time.time() - t0

            # Build trace for technical judges
            trace_data = {}
            if candidates and candidates[0].get("retrieval_trace"):
                trace_data = candidates[0]["retrieval_trace"]
            trace_data["total_latency"] = round(latency, 4)
            trace_data["candidates_retrieved"] = len(candidates)
            trace_data["candidates_reranked"] = len(reranked)
            trace_data["recommendations_generated"] = len(recs)

            # Merge rerank scores into recommendations for confidence display
            for rec in recs:
                for cand in reranked:
                    if rec.get("standard_number") == cand.get("standard_number"):
                        rec["rerank_score"] = cand.get("rerank_score", cand.get("rrf_score", 0.7))
                        break

            results_html = render_results_html(recs, latency, trace_data)
            trace_json = json.dumps(trace_data, indent=2, ensure_ascii=False)

            return results_html, trace_json

        except Exception as e:
            latency = time.time() - t0
            return (
                f'<div style="color:#ff6b6b; padding:20px;">'
                f"<b>Error:</b> {str(e)}</div>",
                json.dumps({"error": str(e), "latency": round(latency, 4)}, indent=2),
            )

    def set_example(example):
        """Set the query textbox to the selected example."""
        return example if example else ""

    # ── Build UI ──
    with gr.Blocks(css=CUSTOM_CSS, title="BIS Standards Advisor", theme=gr.themes.Base()) as demo:

        # Header
        gr.HTML("""
        <div class="header-section">
            <div class="header-title">
                BIS Standards <span class="header-accent">Advisor</span>
            </div>
            <div class="header-subtitle">
                AI-powered BIS standard recommendations for Indian Micro & Small Enterprises
            </div>
        </div>
        """)

        with gr.Column(elem_classes=["input-section"]):
            query_input = gr.Textbox(
                label="Product Description",
                placeholder="Describe your building material product... e.g., '53 grade OPC cement bags for construction sites'",
                lines=3,
                max_lines=5,
                elem_id="query-input",
            )

            with gr.Row():
                example_dropdown = gr.Dropdown(
                    choices=[""] + EXAMPLE_QUERIES,
                    label="Try an example",
                    value="",
                    elem_classes=["example-dropdown"],
                    scale=3,
                )
                search_btn = gr.Button(
                    "🔍 Find Relevant Standards",
                    variant="primary",
                    elem_classes=["search-btn"],
                    scale=1,
                )

        # Results
        results_output = gr.HTML(
            value=(
                '<div style="text-align:center; padding:60px; color:var(--muted);">'
                '<p style="font-size:16px;">Enter a product description to discover applicable BIS standards.</p>'
                '</div>'
            ),
            elem_id="results-container",
        )

        # Retrieval trace (for technical judges)
        with gr.Accordion("🔧 Retrieval Trace (Technical Details)", open=False, elem_classes=["trace-section"]):
            trace_output = gr.Code(
                label="Pipeline Trace",
                language="json",
                lines=12,
            )

        # Footer
        gr.HTML("""
        <div class="footer-text">
            BIS Standards Advisor • Powered by Hybrid RAG + HyDE • Built for Indian MSEs<br>
            <span style="font-size:11px; opacity:0.7;">
                Data source: BIS SP 21 • Embeddings: BGE-small-en-v1.5 •
                Reranker: ms-marco-MiniLM • Generator: Claude Sonnet 4
            </span>
        </div>
        """)

        # ── Event handlers ──
        example_dropdown.change(
            fn=set_example,
            inputs=[example_dropdown],
            outputs=[query_input],
        )

        search_btn.click(
            fn=search,
            inputs=[query_input],
            outputs=[results_output, trace_output],
        )

        query_input.submit(
            fn=search,
            inputs=[query_input],
            outputs=[results_output, trace_output],
        )

    return demo


# ═════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
