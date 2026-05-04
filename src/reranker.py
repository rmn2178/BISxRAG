#!/usr/bin/env python3
"""
Latency-Aware Cross-Encoder Reranker
======================================
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 for semantic reranking.
Respects a latency budget — skips reranking if elapsed time exceeds threshold.
"""

import logging
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LATENCY_BUDGET = 1.0  # seconds


class LatencyAwareReranker:
    """
    Cross-encoder reranker with latency budget awareness.
    If elapsed time exceeds budget, returns candidates as-is (top-5).
    """

    def __init__(self):
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        self.model = CrossEncoder(RERANKER_MODEL, max_length=512)
        self.latency_budget = LATENCY_BUDGET
        logger.info("Reranker model loaded successfully")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        elapsed: float,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: The user's search query
            candidates: List of candidate documents from retrieval
            elapsed: Time elapsed so far in the pipeline (seconds)

        Returns:
            Top-5 reranked candidates
        """
        if not candidates:
            return []

        # ── Latency budget check ──
        if elapsed > self.latency_budget:
            logger.warning(
                f"Latency budget exceeded ({elapsed:.2f}s > {self.latency_budget}s) "
                f"— skipping reranker, returning RRF top-5"
            )
            return candidates[:5]

        # ── Build query-document pairs for cross-encoder ──
        pairs = []
        for c in candidates:
            doc_text = (
                f"{c.get('standard_number', '')}: {c.get('title', '')}. "
                f"{c.get('scope_text', '')[:300]}"
            )
            pairs.append((query, doc_text))

        try:
            # ── Score with cross-encoder ──
            scores = self.model.predict(pairs)

            # ── Sort candidates by cross-encoder score ──
            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for candidate, score in scored[:5]:
                candidate = candidate.copy()
                candidate["rerank_score"] = float(score)
                reranked.append(candidate)

            logger.info(
                f"Reranked {len(candidates)} candidates → top-5 "
                f"(best score: {scored[0][1]:.4f})"
            )
            return reranked

        except Exception as e:
            logger.error(f"Reranker failed: {e} — falling back to RRF top-5")
            return candidates[:5]
