# BIS Standards Recommendation Engine

> AI-powered BIS standard recommendations for Indian Micro & Small Enterprises  
> Built with Hybrid RAG + Dual-Index HyDE + Cross-Encoder Reranking

---

## Quick Start (3 commands)

```bash
pip install -r requirements.txt
python src/ingest.py --pdf data/BIS_SP21.pdf
python inference.py --input data/public_test_set.json --output data/public_test_results.json
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Query                                   │
│               "Fe500 TMT steel bars for buildings"                  │
└────────────────────────┬────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  QUERY UNDERSTANDING ENGINE                                         │
│  • Abbreviation expansion (TMT → Thermo Mechanically Treated)       │
│  • Category detection → Steel                                       │
│  • Grade extraction → Fe500                                         │
│  • Query type classification → product_description                  │
└────────────────────────┬────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  HYBRID RETRIEVAL (ThreadPoolExecutor — parallel)                   │
│  ┌──────────────────┐ ┌──────────────────┐ ┌────────────────────┐  │
│  │  Dense Search     │ │  Synthetic Search │ │  BM25 Sparse       │  │
│  │  (ChromaDB text)  │ │  (ChromaDB synth) │ │  (rank_bm25)       │  │
│  │  BGE-small-en-v1.5│ │  HyDE augmented  │ │  std + title + kw  │  │
│  └────────┬─────────┘ └────────┬─────────┘ └────────┬───────────┘  │
│           └─────────┬──────────┘                     │              │
│                     ▼                                ▼              │
│          Deduplicate by std_number     Metadata-Aware Boosting      │
│              keep highest score        category×1.15 grade×1.10     │
│                     │                  IS-number×1.25               │
│                     └────────────┬───────────────────┘              │
│                                  ▼                                  │
│                     WEIGHTED RRF FUSION                             │
│              0.65×dense_rrf + 0.35×sparse_rrf                       │
│                     → top-10 candidates                             │
└────────────────────────┬────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  HyDE RESCUE (if top_score < 0.6)                                   │
│  Claude generates hypothetical standard summary                     │
│  → embed → second dense pass → merge into RRF pool                 │
└────────────────────────┬────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CROSS-ENCODER RERANKER                                             │
│  ms-marco-MiniLM-L-6-v2 (latency budget: 3.5s)                     │
│  top-10 → top-5                                                     │
└────────────────────────┬────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CLAUDE GENERATOR + HALLUCINATION FILTER                            │
│  Strict system prompt — only recommend from context                 │
│  Post-processing whitelist validation                               │
│  Deterministic fallback on API failure                              │
└────────────────────────┬────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STRICT JSON OUTPUT                                                  │
│  {"id": "...", "retrieved_standards": [...], "latency_seconds": X}  │
└─────────────────────────────────────────────────────────────────────┘
```

## Innovation: Dual-Index HyDE + Query-Time HyDE Rescue

### 1. Ingestion-Time HyDE (Synthetic Query Augmentation)
During ingestion, for **every** BIS standard chunk, we generate **7 diverse synthetic queries** via Claude API — simulating how different user personas (MSE owners, contractors, engineers, compliance officers) would search for that standard. These synthetic queries are embedded into a **separate ChromaDB collection** (`standards_synthetic`), creating a dual-index system that dramatically improves recall by bridging the vocabulary gap between user queries and technical standard language.

### 2. Query-Time HyDE Rescue
When the top RRF fusion score is below 0.6 (low confidence), we activate a **query-time HyDE rescue**:
- Claude generates a 2-sentence hypothetical BIS standard summary matching the user's query
- This hypothetical document is embedded and used for a second dense retrieval pass
- Results are merged back into the RRF pool
- This recovers relevant standards that were missed by both dense and sparse retrieval

This **dual-HyDE strategy** (ingestion-time + query-time) is our key innovation differentiator.

## External APIs & Transparency Disclosure

| Service              | Purpose                          | Setup                      |
|---------------------|----------------------------------|----------------------------|
| Anthropic Claude    | Synthetic queries + rationale    | ANTHROPIC_API_KEY env var  |
| bge-small-en-v1.5   | Embeddings (local, no key)       | Auto-downloaded            |
| ms-marco reranker   | Cross-encoder reranking (local)  | Auto-downloaded            |

## Data Sources

**BIS SP 21 PDF** (organizer-provided) — **SOLE dataset**.  
No external data scraped or used.

## Evaluation Results (Public Test Set)

| Metric          | Score   | Target   |
|----------------|---------|----------|
| Hit Rate @3    | XX%     | >80%     |
| MRR @5         | X.XX    | >0.7     |
| Avg Latency    | X.Xs    | <5s      |
| Hallucinations | 0       | 0        |

*(Update after running eval_script.py on public test set)*

## Environment Variables

```bash
cp .env.example .env
# Fill in your ANTHROPIC_API_KEY
```

## Running the UI

```bash
python src/ui.py
# Open http://localhost:7860
```

## Project Structure

```
/
├── inference.py                   ← ROOT — judges run this
├── eval_script.py                 ← Organizer-provided, DO NOT MODIFY
├── requirements.txt               ← All deps, pinned versions
├── README.md
├── hallucination_log.json         ← Auto-generated at runtime
├── .env.example                   ← ANTHROPIC_API_KEY template
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── ingest.py                  ← PDF parsing + indexing pipeline
│   ├── retriever.py               ← QueryPreprocessor + HybridRetriever + HyDERescue
│   ├── reranker.py                ← LatencyAwareReranker (cross-encoder)
│   ├── generator.py               ← StandardsGenerator (Claude + whitelist)
│   └── ui.py                      ← Gradio Claude-themed UI
│
└── data/
    ├── bm25_index.pkl             ← Serialized BM25 index
    ├── standard_whitelist.json    ← All valid standard numbers
    ├── standards_metadata.json    ← Extracted metadata per standard
    ├── chroma_db/                 ← ChromaDB persistent store
    └── public_test_results.json   ← Output on public test set (committed)
```

## Reproducibility

```bash
# Full pipeline from scratch:
git clone <repo>
cd BISxRAG
pip install -r requirements.txt
cp .env.example .env  # fill in ANTHROPIC_API_KEY

# Ingest (one-time, ~10 min with Claude API, ~2 min without):
python src/ingest.py --pdf data/BIS_SP21.pdf

# Inference:
python inference.py --input data/public_test_set.json --output data/public_test_results.json

# Eval:
python eval_script.py --input data/public_test_results.json

# UI:
python src/ui.py
```
