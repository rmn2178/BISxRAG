#!/usr/bin/env python3
"""
Hallucination-Safe Claude Standards Generator
================================================
Generates BIS standard recommendations using Claude with strict guardrails.
Implements whitelist validation and deterministic fallbacks.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WHITELIST_PATH = DATA_DIR / "standard_whitelist.json"
HALLUCINATION_LOG_PATH = BASE_DIR / "hallucination_log.json"

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a BIS (Bureau of Indian Standards) compliance expert helping
Indian Micro and Small Enterprises identify applicable standards.

RULES — NEVER VIOLATE:
1. ONLY recommend standards explicitly present in the CONTEXT below.
2. NEVER invent, fabricate, or extrapolate any standard number or title.
3. If context is insufficient, return an empty recommendations array.
4. Every standard_number in your output MUST appear verbatim in the context.
5. Output ONLY valid JSON. Zero preamble. Zero explanation outside JSON.

OUTPUT FORMAT (strict):
{
  "recommendations": [
    {
      "standard_number": "IS XXXX",
      "title": "...",
      "rationale": "One sentence explaining applicability to this product."
    }
  ]
}"""

USER_PROMPT_TEMPLATE = """PRODUCT DESCRIPTION: {query}

RELEVANT BIS STANDARDS FROM SP21 (use ONLY these):
{context}

Return the top 3–5 most relevant standards in the exact JSON format specified."""


class StandardsGenerator:
    """
    Generates BIS standard recommendations using Claude API
    with strict hallucination prevention.
    """

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = None

        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("Claude API client initialized")
        else:
            logger.warning("No ANTHROPIC_API_KEY — generator will use fallback mode")

        # Load whitelist at startup
        self.whitelist = set()
        if WHITELIST_PATH.exists():
            with open(WHITELIST_PATH, "r") as f:
                self.whitelist = set(json.load(f))
            logger.info(f"Loaded {len(self.whitelist)} standards in whitelist")
        else:
            logger.warning(f"Whitelist not found at {WHITELIST_PATH}")

        # Load existing hallucination log
        self.hallucination_log = []
        if HALLUCINATION_LOG_PATH.exists():
            try:
                with open(HALLUCINATION_LOG_PATH, "r") as f:
                    self.hallucination_log = json.load(f)
            except (json.JSONDecodeError, Exception):
                self.hallucination_log = []

    def generate(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations from reranked candidates.

        1. Format context from candidates
        2. Call Claude API
        3. Parse and validate response
        4. Post-process with whitelist
        5. Fallback on any error
        """
        if not candidates:
            return []

        # ── Format context ──
        context = self._format_context(candidates)

        # ── Call Claude ──
        if self.client:
            try:
                recommendations = self._call_claude(query, context)
                if recommendations is not None:
                    # Post-process: whitelist validation
                    clean_recs = self._validate_whitelist(recommendations, query)
                    return clean_recs
            except Exception as e:
                logger.error(f"Claude generation failed: {e}")

        # ── Deterministic fallback ──
        logger.info("Using deterministic fallback for recommendations")
        return self._deterministic_fallback(candidates)

    def _format_context(
        self, candidates: List[Dict[str, Any]], max_tokens: int = 1500
    ) -> str:
        """Format candidates into context string for Claude."""
        context_parts = []
        total_chars = 0
        # Approximate 1 token ≈ 4 chars
        char_limit = max_tokens * 4

        for c in candidates:
            # Determine how much scope text we can include
            scope_limit = min(400, max(100, (char_limit - total_chars) // 2))

            entry = (
                f"Standard: {c.get('standard_number', 'N/A')} | "
                f"{c.get('title', 'N/A')}\n"
                f"Category: {c.get('material_category', 'N/A')}\n"
                f"Keywords: {', '.join(c.get('keywords', [])[:8])}\n"
                f"Scope: {c.get('scope_text', '')[:scope_limit]}\n"
                f"---"
            )

            if total_chars + len(entry) > char_limit:
                # Trim scope to fit
                remaining = max(0, char_limit - total_chars - 100)
                if remaining > 50:
                    entry = (
                        f"Standard: {c.get('standard_number', 'N/A')} | "
                        f"{c.get('title', 'N/A')}\n"
                        f"Category: {c.get('material_category', 'N/A')}\n"
                        f"---"
                    )
                    context_parts.append(entry)
                break

            context_parts.append(entry)
            total_chars += len(entry)

        return "\n".join(context_parts)

    def _call_claude(
        self, query: str, context: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Call Claude API and parse the response."""
        user_prompt = USER_PROMPT_TEMPLATE.format(query=query, context=context)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text = response.content[0].text.strip()

        # Parse JSON from response
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                logger.error(f"Failed to parse Claude response: {raw_text[:200]}")
                return None

        recommendations = parsed.get("recommendations", [])
        return recommendations

    def _validate_whitelist(
        self, recommendations: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Validate all standard_numbers against the whitelist."""
        clean = []
        removed = []

        for rec in recommendations:
            sn = rec.get("standard_number", "")

            if sn in self.whitelist:
                clean.append(rec)
            else:
                # Try fuzzy matching (e.g., "IS 456" vs "IS  456")
                import re
                normalized = re.sub(r"\s+", " ", sn.strip())
                if normalized in self.whitelist:
                    rec["standard_number"] = normalized
                    clean.append(rec)
                else:
                    removed.append(rec)
                    logger.warning(
                        f"Hallucination detected: '{sn}' not in whitelist — removed"
                    )

        # Log removals
        if removed:
            for r in removed:
                self.hallucination_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "removed_standard": r.get("standard_number", ""),
                    "removed_title": r.get("title", ""),
                })

            # Save log
            try:
                with open(HALLUCINATION_LOG_PATH, "w") as f:
                    json.dump(self.hallucination_log, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save hallucination log: {e}")

        return clean

    def _deterministic_fallback(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deterministic fallback when Claude API is unavailable.
        Returns retrieval-only results with template rationale.
        """
        fallback = []
        for c in candidates[:3]:
            fallback.append({
                "standard_number": c.get("standard_number", ""),
                "title": c.get("title", ""),
                "rationale": (
                    f"Highly relevant to your product based on scope match "
                    f"in {c.get('material_category', 'the relevant')} category."
                ),
            })
        return fallback
