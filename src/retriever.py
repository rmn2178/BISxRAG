#!/usr/bin/env python3
"""
BIS Standards Hybrid Retriever
================================
QueryPreprocessor: abbreviation expansion, category detection, grade extraction
HybridRetriever:   parallel dense + sparse search, metadata boosting, weighted RRF
HyDERescue:        query-time hypothetical document generation for low-confidence results
"""

import json
import logging
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
import requests
import chromadb
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
WHITELIST_PATH = DATA_DIR / "standard_whitelist.json"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = "nomic-embed-text:v1.5"

class OllamaEmbedder:
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        url = f"{OLLAMA_URL}/api/embed"
        payload = {"model": OLLAMA_EMBED_MODEL, "input": texts}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return np.array(resp.json()["embeddings"])
        except Exception as e:
            logger.error(f"Ollama embed failed: {e}")
            return np.zeros((len(texts), 768))

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# QUERY PREPROCESSOR
# ═════════════════════════════════════════════════════════════════════════════
class QueryPreprocessor:
    """Expands abbreviations, detects categories, extracts grades."""

    ABBREVIATION_MAP = {
        "RCC":   "Reinforced Cement Concrete",
        "PCC":   "Plain Cement Concrete",
        "TMT":   "Thermo Mechanically Treated",
        "OPC":   "Ordinary Portland Cement",
        "PPC":   "Portland Pozzolana Cement",
        "AAC":   "Autoclaved Aerated Concrete",
        "GGBS":  "Ground Granulated Blast Furnace Slag",
        "FA":    "Fly Ash",
        "MS":    "Mild Steel",
        "HSD":   "High Strength Deformed",
        "Fe500": "Fe 500 grade reinforcement steel",
        "OPC53": "53 Grade Ordinary Portland Cement",
        "SRC":   "Sulphate Resisting Cement",
        "HAC":   "High Alumina Cement",
    }

    CATEGORY_KEYWORDS = {
        "Cement": [
            "cement", "opc", "ppc", "clinker", "portland", "pozzolana",
            "slag cement", "sulphate resisting", "hydraulic",
        ],
        "Steel": [
            "steel", "tmt", "fe500", "fe 500", "fe415", "fe 415",
            "rebar", "reinforcement", "bar", "rod", "wire",
            "mild steel", "deformed", "high tensile",
        ],
        "Concrete": [
            "concrete", "rcc", "pcc", "mix", "grade", "m20", "m25", "m30",
            "ready mixed", "admixture", "precast", "prestressed",
        ],
        "Aggregates": [
            "aggregate", "sand", "gravel", "crushed stone", "coarse",
            "fine aggregate", "stone dust", "crusher",
        ],
        "Masonry": [
            "brick", "block", "masonry", "wall", "aac", "fly ash brick",
            "hollow block", "solid block", "paving",
        ],
    }

    GRADE_PATTERNS = [
        r"Fe\s*\d{3}[A-Z]?",
        r"M\s*\d{2,3}",
        r"\d{2,3}\s*[Gg]rade",
        r"Grade\s*\d+",
    ]

    IS_NUMBER_PATTERN = re.compile(r"\bIS\s+\d{1,5}(?:\s*\(\s*Part\s+\d+\s*\))?(?:\s*:\s*\d{4})?\b", re.IGNORECASE)

    def preprocess(self, query: str) -> Dict[str, Any]:
        """Full query preprocessing: expansion, detection, extraction."""
        expanded_query = self._expand_abbreviations(query)
        categories = self._detect_categories(expanded_query)
        grades = self._extract_grades(expanded_query)
        is_numbers = self._extract_is_numbers(expanded_query)
        query_type = self._classify_query(expanded_query)

        result = {
            "original_query": query,
            "expanded_query": expanded_query,
            "detected_categories": categories,
            "detected_grades": grades,
            "detected_is_numbers": is_numbers,
            "query_type": query_type,
        }
        logger.debug(f"Preprocessed query: {result}")
        return result

    def _expand_abbreviations(self, query: str) -> str:
        """Expand known abbreviations in the query."""
        expanded = query
        for abbr, full in self.ABBREVIATION_MAP.items():
            # Use word boundary matching for abbreviation replacement
            pattern = re.compile(r"\b" + re.escape(abbr) + r"\b", re.IGNORECASE)
            if pattern.search(expanded):
                expanded = pattern.sub(f"{abbr} ({full})", expanded, count=1)
        return expanded

    def _detect_categories(self, query: str) -> List[str]:
        """Detect material categories from query text."""
        query_lower = query.lower()
        detected = []
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in query_lower:
                    if category not in detected:
                        detected.append(category)
                    break
        return detected

    def _extract_grades(self, query: str) -> List[str]:
        """Extract grade specifications from query."""
        grades = set()
        for pattern in self.GRADE_PATTERNS:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for m in matches:
                grades.add(m.strip())
        return sorted(grades)

    def _extract_is_numbers(self, query: str) -> List[str]:
        """Extract any IS standard numbers mentioned in query."""
        matches = self.IS_NUMBER_PATTERN.findall(query)
        return [re.sub(r"\s+", " ", m.strip()) for m in matches]

    def _classify_query(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        if any(w in query_lower for w in ["test", "testing", "method", "sampling"]):
            return "testing"
        if any(w in query_lower for w in ["compliance", "certification", "mandatory"]):
            return "compliance"
        if any(w in query_lower for w in ["specification", "requirement", "property"]):
            return "specification"
        if any(w in query_lower for w in ["is ", "code", "standard"]):
            return "standard_lookup"
        return "product_description"


# ═════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVER
# ═════════════════════════════════════════════════════════════════════════════
class HybridRetriever:
    """
    Parallel hybrid retrieval combining:
      - Dense search (ChromaDB standards_text + standards_synthetic)
      - Sparse search (BM25)
      - Metadata-aware score boosting
      - Weighted RRF fusion
      - HyDE rescue for low-confidence results
    """

    def __init__(self):
        logger.info("Initializing HybridRetriever...")

        # Load embedder
        self.embedder = OllamaEmbedder()

        # Load ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.text_collection = self.chroma_client.get_collection("standards_text")
        self.synth_collection = self.chroma_client.get_collection("standards_synthetic")

        # Load BM25
        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_doc_map = bm25_data["doc_map"]

        # Load whitelist
        with open(WHITELIST_PATH, "r") as f:
            self.whitelist = set(json.load(f))

        # Preprocessor
        self.preprocessor = QueryPreprocessor()

        # HyDE rescue module
        self.hyde_rescue = HyDERescue(self.embedder, self.text_collection)

        logger.info(
            f"HybridRetriever ready. "
            f"Text docs: {self.text_collection.count()}, "
            f"Synthetic docs: {self.synth_collection.count()}, "
            f"BM25 docs: {len(self.bm25_doc_map)}, "
            f"Whitelist: {len(self.whitelist)} standards"
        )

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Main retrieval pipeline:
        1. Preprocess query
        2. Parallel dense + sparse search
        3. Metadata-aware boosting
        4. Weighted RRF fusion
        5. HyDE rescue if top score < 0.6
        """
        preprocessed = self.preprocessor.preprocess(query)
        expanded_query = preprocessed["expanded_query"]

        # ── Parallel retrieval ──
        dense_results = []
        sparse_results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._dense_search, expanded_query): "dense",
                executor.submit(self._sparse_search, expanded_query): "sparse",
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    if source == "dense":
                        dense_results = future.result()
                    else:
                        sparse_results = future.result()
                except Exception as e:
                    logger.error(f"{source} search failed: {e}")

        # ── Apply metadata-aware boosts ──
        dense_results = self._apply_boosts(dense_results, preprocessed)
        sparse_results = self._apply_boosts(sparse_results, preprocessed)

        # ── Weighted RRF fusion ──
        fused = self._weighted_rrf(dense_results, sparse_results)

        # ── HyDE rescue if top score is low ──
        retrieval_trace = {
            "expanded_query": expanded_query,
            "detected_categories": preprocessed["detected_categories"],
            "detected_grades": preprocessed["detected_grades"],
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "hyde_activated": False,
        }

        if fused and fused[0].get("rrf_score", 0) < 0.6:
            logger.info("Low confidence — activating HyDE rescue")
            hyde_results = self.hyde_rescue.rescue(query, fused)
            if hyde_results:
                fused = self._weighted_rrf(
                    fused + hyde_results,
                    sparse_results,
                    dense_weight=0.70,
                    sparse_weight=0.30,
                )
                retrieval_trace["hyde_activated"] = True

        # Attach trace to top results
        for candidate in fused[:10]:
            candidate["retrieval_trace"] = retrieval_trace

        return fused[:10]

    def _dense_search(self, query: str, n_results: int = 15) -> List[Dict[str, Any]]:
        """Search both ChromaDB collections and merge results."""
        query_embedding = self.embedder.encode(query).tolist()

        results = []

        # Search standards_text
        try:
            text_results = self.text_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.text_collection.count()),
                include=["metadatas", "distances", "documents"],
            )
            results.extend(self._parse_chroma_results(text_results, "text"))
        except Exception as e:
            logger.error(f"standards_text search failed: {e}")

        # Search standards_synthetic
        try:
            synth_results = self.synth_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.synth_collection.count()),
                include=["metadatas", "distances", "documents"],
            )
            results.extend(self._parse_chroma_results(synth_results, "synthetic"))
        except Exception as e:
            logger.error(f"standards_synthetic search failed: {e}")

        # Deduplicate by standard_number, keep highest score
        deduped = {}
        for r in results:
            sn = r["standard_number"]
            if sn not in deduped or r["dense_score"] > deduped[sn]["dense_score"]:
                deduped[sn] = r

        return sorted(deduped.values(), key=lambda x: x["dense_score"], reverse=True)

    def _parse_chroma_results(
        self, results: dict, source: str
    ) -> List[Dict[str, Any]]:
        """Parse ChromaDB query results into standard format."""
        parsed = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return parsed

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, (doc_id, meta, dist) in enumerate(zip(ids, metadatas, distances)):
            # ChromaDB cosine distance: lower = more similar
            # Convert to similarity score (0 to 1)
            score = max(0, 1.0 - dist)

            parsed.append({
                "standard_number": meta.get("standard_number", ""),
                "title": meta.get("title", ""),
                "material_category": meta.get("material_category", ""),
                "scope_text": meta.get("scope_text", ""),
                "keywords": json.loads(meta.get("keywords", "[]")),
                "grades": json.loads(meta.get("grades", "[]")),
                "applications": json.loads(meta.get("applications", "[]")),
                "test_methods": json.loads(meta.get("test_methods", "[]")),
                "engineering_terms": json.loads(meta.get("engineering_terms", "[]")),
                "dense_score": score,
                "source": source,
                "rank": i + 1,
            })

        return parsed

    def _sparse_search(self, query: str, n_results: int = 15) -> List[Dict[str, Any]]:
        """BM25 sparse search over combined text fields."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top n_results
        top_indices = np.argsort(scores)[::-1][:n_results]

        results = []
        max_score = max(scores) if max(scores) > 0 else 1.0

        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            doc = self.bm25_doc_map[idx]
            results.append({
                "standard_number": doc["standard_number"],
                "title": doc["title"],
                "material_category": doc["material_category"],
                "scope_text": doc["scope_text"],
                "keywords": doc.get("keywords", []),
                "grades": doc.get("grades", []),
                "applications": doc.get("applications", []),
                "test_methods": doc.get("test_methods", []),
                "engineering_terms": doc.get("engineering_terms", []),
                "sparse_score": scores[idx] / max_score,  # Normalize to 0-1
                "source": "bm25",
                "rank": rank + 1,
            })

        return results

    def _apply_boosts(
        self, results: List[Dict[str, Any]], preprocessed: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply metadata-aware score boosting."""
        categories = set(preprocessed["detected_categories"])
        grades = set(g.lower() for g in preprocessed["detected_grades"])
        is_numbers = set(preprocessed["detected_is_numbers"])

        for r in results:
            boost = 1.0

            # Category match → ×1.15
            if r.get("material_category") in categories:
                boost *= 1.15

            # Grade match → ×1.10
            doc_grades = set(g.lower() for g in r.get("grades", []))
            if grades & doc_grades:
                boost *= 1.10

            # Exact IS-number match → ×1.25
            if r.get("standard_number") in is_numbers:
                boost *= 1.25

            # Apply boost to whichever score exists
            if "dense_score" in r:
                r["dense_score"] *= boost
            if "sparse_score" in r:
                r["sparse_score"] *= boost

            r["boost_applied"] = boost

        return results

    def _weighted_rrf(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Weighted Reciprocal Rank Fusion.
        final_score = dense_weight × dense_rrf + sparse_weight × sparse_rrf + boosts
        """
        # Build RRF scores
        rrf_scores: Dict[str, float] = {}
        candidate_data: Dict[str, Dict[str, Any]] = {}

        # Dense RRF
        for rank, r in enumerate(dense_results):
            sn = r["standard_number"]
            rrf = 1.0 / (k + rank + 1)
            rrf_scores[sn] = rrf_scores.get(sn, 0) + dense_weight * rrf
            if sn not in candidate_data:
                candidate_data[sn] = r.copy()

        # Sparse RRF
        for rank, r in enumerate(sparse_results):
            sn = r["standard_number"]
            rrf = 1.0 / (k + rank + 1)
            rrf_scores[sn] = rrf_scores.get(sn, 0) + sparse_weight * rrf
            if sn not in candidate_data:
                candidate_data[sn] = r.copy()

        # Sort by RRF score
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for sn, score in sorted_candidates:
            candidate = candidate_data[sn]
            candidate["rrf_score"] = score
            fused.append(candidate)

        return fused


# ═════════════════════════════════════════════════════════════════════════════
# HYDE RESCUE
# ═════════════════════════════════════════════════════════════════════════════
class HyDERescue:
    """
    Query-time Hypothetical Document Embeddings rescue.
    Activated when top RRF score < 0.6 to improve recall on hard queries.
    """

    def __init__(
        self,
        embedder: OllamaEmbedder,
        text_collection: chromadb.Collection,
    ):
        self.embedder = embedder
        self.text_collection = text_collection
        self.client = None

        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel("gemini-2.5-flash")

    def rescue(
        self, query: str, existing_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate a hypothetical BIS standard summary, embed it,
        and do a second dense pass to find missed standards.
        """
        if self.client is None:
            logger.warning("No API key — HyDE rescue skipped")
            return []

        try:
            prompt = (
                f"Write a 2-sentence BIS SP21 standard summary that would match "
                f"this product description: {query}\n"
                f"Return only the summary text, nothing else."
            )
            response = self.client.generate_content(
                prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 200},
            )
            hypothetical_text = response.text.strip()

            logger.info(f"HyDE generated: {hypothetical_text[:100]}...")

            # Embed the hypothetical document
            hypothetical_embedding = self.embedder.encode(hypothetical_text).tolist()

            # Second dense pass
            second_pass = self.text_collection.query(
                query_embeddings=[hypothetical_embedding],
                n_results=min(10, self.text_collection.count()),
                include=["metadatas", "distances"],
            )

            results = []
            if second_pass and second_pass["ids"] and second_pass["ids"][0]:
                for i, (doc_id, meta, dist) in enumerate(zip(
                    second_pass["ids"][0],
                    second_pass["metadatas"][0],
                    second_pass["distances"][0],
                )):
                    score = max(0, 1.0 - dist)
                    results.append({
                        "standard_number": meta.get("standard_number", ""),
                        "title": meta.get("title", ""),
                        "material_category": meta.get("material_category", ""),
                        "scope_text": meta.get("scope_text", ""),
                        "keywords": json.loads(meta.get("keywords", "[]")),
                        "grades": json.loads(meta.get("grades", "[]")),
                        "applications": json.loads(meta.get("applications", "[]")),
                        "test_methods": json.loads(meta.get("test_methods", "[]")),
                        "engineering_terms": json.loads(
                            meta.get("engineering_terms", "[]")
                        ),
                        "dense_score": score,
                        "source": "hyde_rescue",
                        "rank": i + 1,
                    })

            return results

        except Exception as e:
            logger.error(f"HyDE rescue failed: {e}")
            return []
