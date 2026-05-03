#!/usr/bin/env python3
"""
BIS Standards Recommendation Engine
=====================================
Production inference entrypoint for the BIS Standards RAG pipeline.

Usage:
    python inference.py --input hidden_private_dataset.json --output team_results.json

Output format (strict — eval_script.py depends on these exact keys):
    [
        {
            "id": "...",
            "retrieved_standards": ["IS 456", "IS 383", ...],
            "latency_seconds": 1.23
        }
    ]
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

assert os.getenv("ANTHROPIC_API_KEY"), (
    "ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill in your key."
)

from src.retriever import HybridRetriever
from src.reranker import LatencyAwareReranker
from src.generator import StandardsGenerator

logger = logging.getLogger(__name__)


def run_pipeline(
    query: str,
    retriever: HybridRetriever,
    reranker: LatencyAwareReranker,
    generator: StandardsGenerator,
) -> dict:
    """
    Run the full retrieval → rerank → generate pipeline for a single query.
    Returns dict with retrieved_standards and latency_seconds.
    Never raises — returns fallback on any error.
    """
    t0 = time.time()

    try:
        # Step 1: Hybrid retrieval
        candidates = retriever.retrieve(query)
        elapsed = time.time() - t0

        # Step 2: Latency-aware reranking
        reranked = reranker.rerank(query, candidates, elapsed)

        # Step 3: Claude generation with hallucination filtering
        recs = generator.generate(query, reranked)

        # Extract standard numbers
        retrieved = [r["standard_number"] for r in recs if r.get("standard_number")]

        # Ensure we have at least some results
        if not retrieved and reranked:
            retrieved = [
                r["standard_number"]
                for r in reranked[:3]
                if r.get("standard_number")
            ]

        return {
            "retrieved_standards": retrieved,
            "latency_seconds": round(time.time() - t0, 4),
        }

    except Exception as e:
        logger.error(f"Pipeline error for query '{query[:50]}...': {e}")
        logger.error(traceback.format_exc())

        # Deterministic fallback — never crash
        elapsed = round(time.time() - t0, 4)
        return {
            "retrieved_standards": [],
            "latency_seconds": elapsed,
        }


def main():
    parser = argparse.ArgumentParser(
        description="BIS Standards Recommendation Engine — Inference"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file with queries",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file for results",
    )
    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # ── Load ALL models ONCE before the query loop ──
    logger.info("=" * 60)
    logger.info("INITIALIZING BIS STANDARDS RECOMMENDATION ENGINE")
    logger.info("=" * 60)

    init_start = time.time()
    retriever = HybridRetriever()
    reranker = LatencyAwareReranker()
    generator = StandardsGenerator()
    init_time = time.time() - init_start
    logger.info(f"All models loaded in {init_time:.2f}s")

    # ── Load queries ──
    with open(args.input, "r", encoding="utf-8") as f:
        queries = json.load(f)

    logger.info(f"Processing {len(queries)} queries...")

    # ── Process each query ──
    results = []
    total_latency = 0

    for i, item in enumerate(queries):
        query_id = item.get("id", f"query_{i}")
        query_text = item.get("query", "")

        logger.info(f"[{i+1}/{len(queries)}] Processing: {query_text[:60]}...")

        result = run_pipeline(query_text, retriever, reranker, generator)

        output_item = {
            "id": query_id,
            "query": query_text,
            "expected_standards": item.get("expected_standards", []),
            "retrieved_standards": result["retrieved_standards"],
            "latency_seconds": result["latency_seconds"],
        }
        results.append(output_item)
        total_latency += result["latency_seconds"]

        logger.info(
            f"  → {len(result['retrieved_standards'])} standards, "
            f"latency={result['latency_seconds']}s"
        )

    # ── Write output ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    avg_latency = total_latency / len(results) if results else 0
    total_standards = sum(len(r["retrieved_standards"]) for r in results)

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"  Queries processed: {len(results)}")
    print(f"  Total standards:   {total_standards}")
    print(f"  Avg latency:       {avg_latency:.2f}s")
    print(f"  Output saved to:   {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
