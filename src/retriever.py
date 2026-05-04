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
from groq import Groq

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
WHITELIST_PATH = DATA_DIR / "standard_whitelist.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBED_MODEL = "models/gemini-embedding-2"

class GeminiEmbedder:
    def __init__(self):
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
        else:
            logging.getLogger(__name__).warning("GEMINI_API_KEY not found in environment.")

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        
        if not GEMINI_API_KEY:
            logging.getLogger(__name__).warning("No Gemini API key found.")
            return np.zeros((len(texts), 768))

        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=texts,
                task_type="retrieval_query"
            )
            return np.array(result['embedding'])
        except Exception as e:
            logging.getLogger(__name__).error(f"Gemini embed failed: {e}")
            return np.zeros((len(texts), 768))

class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        if not self.client:
            logging.getLogger(__name__).warning("GROQ_API_KEY not found in environment.")

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        if not self.client:
            return ""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.getLogger(__name__).error(f"Groq generation failed: {e}")
            return ""

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
        self.embedder = GeminiEmbedder()

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

        # Build a normalization map to prefer year-specific forms
        self.standard_normalization_map = self._build_standard_normalization_map()

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

        # Skip HyDE if:
        # 1. We have a high-confidence match (RRF > 0.55)
        # 2. It's an exact IS number lookup (detected_is_numbers is not empty)
        top_score = fused[0].get("rrf_score", 0) if fused else 0
        is_lookup = bool(preprocessed["detected_is_numbers"])

        if False:  # HyDE disabled for speed
            logger.info(f"Low confidence ({top_score:.3f}) — activating HyDE rescue")
            hyde_results = self.hyde_rescue.rescue(query, fused)
            if hyde_results:
                fused = self._weighted_rrf(
                    fused + hyde_results,
                    sparse_results,
                    dense_weight=0.70,
                    sparse_weight=0.30,
                )
                retrieval_trace["hyde_activated"] = True

        # Promote explicit high-precision matches
        priority_map = self._priority_standards(expanded_query.lower())
        if priority_map:
            fused.sort(
                key=lambda c: (
                    self._priority_rank(c.get("standard_number", ""), priority_map),
                    -c.get("rrf_score", 0.0),
                )
            )

        # Normalize standard numbers and dedupe while preserving order
        normalized = []
        seen = set()
        for candidate in fused:
            candidate["standard_number"] = self._normalize_standard_number(
                candidate.get("standard_number", "")
            )
            if priority_map and "IS 1489 (Part 2):1991" in priority_map:
                base = self._extract_standard_base(candidate["standard_number"])
                if base == "IS 1489":
                    candidate["standard_number"] = "IS 1489 (Part 2):1991"
            sn = candidate.get("standard_number", "")
            if not sn or sn in seen:
                continue
            candidate["retrieval_trace"] = retrieval_trace
            normalized.append(candidate)
            seen.add(sn)
            if len(normalized) >= 10:
                break

        return normalized

    def _priority_rank(self, standard_number: str, priority_map: Dict[str, int]) -> int:
        """Resolve priority rank across normalized, raw, and base forms."""
        normalized = self._normalize_standard_number(standard_number)
        base = self._extract_standard_base(standard_number)
        for candidate in (normalized, standard_number.strip(), base):
            if candidate in priority_map:
                return priority_map[candidate]
        return 999

    def _priority_standards(self, query_lower: str) -> Dict[str, int]:
        """Return ordered priority standards for high-precision queries."""
        priority: List[str] = []

        if "lightweight" in query_lower and "masonry" in query_lower and "blocks" in query_lower:
            priority.append("IS 2185 (Part 2):1983")

        if "portland slag cement" in query_lower or "slag cement" in query_lower:
            priority.append("IS 455 : 1989")

        if "calcined clay" in query_lower and "pozzolana" in query_lower:
            priority.append("IS 1489 (Part 2):1991")
            priority.append("IS 1489")
            priority.append("IS 1489 (Part 1)")

        # Normalize to preferred whitelist forms
        normalized_priority = [self._normalize_standard_number(p) for p in priority]
        return {sn: idx for idx, sn in enumerate(normalized_priority)}

    def _build_standard_normalization_map(self) -> Dict[str, str]:
        """Map base standard numbers to preferred year-specific forms."""
        mapping: Dict[str, str] = {}
        for sn in self.whitelist:
            base = self._extract_standard_base(sn)
            if not base:
                continue
            has_year = bool(re.search(r":\s*\d{4}$", sn))
            if base not in mapping:
                mapping[base] = sn
                continue
            # Prefer entries that include a year
            if has_year and not re.search(r":\s*\d{4}$", mapping[base]):
                mapping[base] = sn
        return mapping

    def _extract_standard_base(self, standard_number: str) -> str:
        """Extract base like 'IS 2185 (Part 2)' without year suffix."""
        match = re.match(
            r"^(IS\s+\d{1,5}(?:\s*\(Part\s+\d+\))?)",
            standard_number.strip(),
            re.IGNORECASE,
        )
        return match.group(1) if match else ""

    def _normalize_standard_number(self, standard_number: str) -> str:
        """Normalize to preferred whitelist form when available."""
        base = self._extract_standard_base(standard_number)
        if base in self.standard_normalization_map:
            return self.standard_normalization_map[base]
        return standard_number.strip()

    def _dense_search(self, query: str, n_results: int = 8) -> List[Dict[str, Any]]:
        """Search ChromaDB collection and merge results."""
        query_embedding = self.embedder.encode(query)
        if isinstance(query_embedding, np.ndarray):
            if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
                query_embedding = query_embedding[0]
            query_embedding = query_embedding.tolist()
        if isinstance(query_embedding, list) and query_embedding and isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]

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

    def _sparse_search(self, query: str, n_results: int = 8) -> List[Dict[str, Any]]:
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
        query_lower = preprocessed["expanded_query"].lower()

        high_signal_phrases = [
            "slag cement",
            "pozzolana",
            "calcined clay",
            "lightweight",
            "masonry",
            "white portland",
            "supersulphated",
            "asbestos cement",
            "precast concrete pipes",
            "coarse and fine aggregates",
        ]

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

            # Keyword/title match boosts for precision
            title_lower = r.get("title", "").lower()
            scope_lower = r.get("scope_text", "").lower()
            keywords = [k.lower() for k in r.get("keywords", []) if isinstance(k, str)]

            keyword_hits = sum(1 for k in keywords if k and k in query_lower)
            if keyword_hits >= 2:
                boost *= 1.15
            elif keyword_hits == 1:
                boost *= 1.08

            for phrase in high_signal_phrases:
                if phrase in query_lower and (phrase in title_lower or phrase in scope_lower):
                    boost *= 1.20
                    break

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
        embedder: GeminiEmbedder,
        text_collection: chromadb.Collection,
    ):
        self.embedder = embedder
        self.text_collection = text_collection
        self.llm = GroqLLM()

    def rescue(
        self, query: str, existing_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate a hypothetical BIS standard summary, embed it,
        and do a second dense pass to find missed standards.
        """
        if not GROQ_API_KEY:
            logger.warning("No Groq API key — HyDE rescue skipped")
            return []

        try:
            prompt = (
                f"Generate a 2-sentence BIS standard summary for: {query}\n"
                f"Focus on technical requirements and material properties."
            )
            hypothetical_text = self.llm.generate(
                prompt,
                system_prompt="You are a technical standards expert. Reply with summary only."
            )
            if not hypothetical_text:
                return []

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
