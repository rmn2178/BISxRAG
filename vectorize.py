#!/usr/bin/env python3
"""
BIS Standards Vectorization Pipeline (Groq Edition)
======================================================
Replaces Ollama with Groq API for embeddings.
Produces the SAME outputs as ingest.py so retriever.py works unchanged:
  - data/chroma_db/          (ChromaDB: standards_text + standards_synthetic)
  - data/bm25_index.pkl      (BM25Okapi sparse index)
  - data/standard_whitelist.json
  - data/standards_metadata.json

Usage:
    python vectorize.py --pdf pdf/BIS.pdf
    python vectorize.py --pdf pdf/BIS.pdf --skip-synthetic
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import pdfplumber
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from groq import Groq

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
WHITELIST_PATH = DATA_DIR / "standard_whitelist.json"
METADATA_PATH = DATA_DIR / "standards_metadata.json"

# Load Gemini API keys for rotation
GEMINI_KEYS = [k for k in [os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2")] if k]
if not GEMINI_KEYS and os.getenv("GEMINI_API_KEY"):
    GEMINI_KEYS = [os.getenv("GEMINI_API_KEY")]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBED_MODEL = "models/gemini-embedding-2"
EMBED_DIM = 768  # gemini-embedding-2 is 768.

BATCH_SIZE = 32  # how many texts to embed per call

logger = logging.getLogger(__name__)

# ─── Embedder ──────────────────────────────────────────────────────────

class GeminiRotator:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.current_idx = 0
        self._configure_current()

    def _configure_current(self):
        if self.keys:
            genai.configure(api_key=self.keys[self.current_idx])
            logger.info(f"Configured Gemini with key index {self.current_idx}")

    def rotate(self):
        if len(self.keys) > 1:
            self.current_idx = (self.current_idx + 1) % len(self.keys)
            self._configure_current()

    def embed(self, texts: List[str], task_type: str = "retrieval_document"):
        if not self.keys:
            logger.error("No Gemini API keys found.")
            raise ValueError("GEMINI_API_KEY_1/2 is required for embeddings.")
        
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            # Try up to len(self.keys) times if we hit rate limits
            for _ in range(len(self.keys)):
                try:
                    result = genai.embed_content(
                        model=EMBED_MODEL,
                        content=texts,
                        task_type=task_type
                    )
                    return result['embedding']
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "quota" in err_str or "limit" in err_str:
                        logger.warning(f"Rate limit hit for key {self.current_idx}. Rotating...")
                        self.rotate()
                        time.sleep(2) # Small delay after rotation
                    else:
                        logger.error(f"Gemini embed failed: {e}")
                        raise
            
            logger.warning(f"All keys exhausted. Waiting 60 seconds (Attempt {attempt+1}/{max_retries})...")
            time.sleep(60) # Wait a minute before trying again
            
        raise Exception("All Gemini API keys hit rate limits after multiple retries.")

gemini_rotator = GeminiRotator(GEMINI_KEYS)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Call Gemini API to get embeddings for a list of texts (with rotation).
    """
    return gemini_rotator.embed(texts)


def embed_texts_batched(texts: List[str], desc: str = "Embedding") -> List[List[float]]:
    """Embed in batches with a progress bar."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)
    return all_embeddings


def check_api_keys() -> None:
    """Verify API keys are present."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is missing in .env file.")
        sys.exit(1)
    if not GEMINI_KEYS:
        logger.error("GEMINI_API_KEY_1/2 is missing in .env file (needed for embeddings).")
        sys.exit(1)
    logger.info(f"API keys found. Using Groq for LLM and {len(GEMINI_KEYS)} Gemini key(s) for embeddings.")



# ─── PDF Parsing ──────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    logger.info(f"Extracting text from: {pdf_path}")
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="Reading PDF pages"):
            text = page.extract_text()
            if text:
                pages.append(text)
    combined = "\n\n".join(pages)
    logger.info(f"Extracted {len(combined):,} characters from {len(pages)} pages")
    return combined


def normalize_standard_number(sn: str) -> str:
    return re.sub(r"\s+", " ", sn.strip())


def split_into_standards(text: str) -> List[Dict[str, str]]:
    boundary_pattern = r"(?=\bIS\s+\d{1,5}(?:\s*[-:(\s]))"
    chunks = re.split(boundary_pattern, text)
    standards = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        is_match = re.match(
            r"(IS\s+\d{1,5}(?:\s*\(\s*Part\s+\d+\s*\))?(?:\s*:\s*\d{4})?)", chunk
        )
        if is_match:
            sn = normalize_standard_number(is_match.group(1))
            standards.append({"standard_number": sn, "raw_text": chunk})
        elif standards:
            standards[-1]["raw_text"] += "\n" + chunk

    # Merge duplicates
    merged: Dict[str, Any] = {}
    for s in standards:
        key = s["standard_number"]
        if key in merged:
            merged[key]["raw_text"] += "\n" + s["raw_text"]
        else:
            merged[key] = s

    result = list(merged.values())
    logger.info(f"Found {len(result)} unique standards after merging")
    return result


# ─── Metadata Extraction ──────────────────────────────────────────────────────

MATERIAL_CATEGORIES = {
    "Cement": ["cement", "portland", "clinker", "opc", "ppc", "pozzolana",
               "sulphate resisting", "slag cement", "masonry cement", "white cement"],
    "Concrete": ["concrete", "rcc", "pcc", "ready mixed", "admixture",
                 "precast", "prestressed", "reinforced concrete"],
    "Steel": ["steel", "tmt", "rebar", "reinforcement", "bar", "wire",
              "structural steel", "mild steel", "fe 500", "fe 415"],
    "Aggregates": ["aggregate", "sand", "gravel", "crushed stone",
                   "coarse aggregate", "fine aggregate", "lightweight aggregate"],
    "Masonry": ["brick", "block", "masonry", "wall", "aac",
                "fly ash brick", "concrete block", "hollow block"],
    "Testing": ["test", "testing", "sampling", "method of test",
                "determination", "analysis", "chemical analysis"],
    "Structural": ["structural", "design", "code of practice", "earthquake",
                   "seismic", "wind load", "loading", "foundation"],
    "Misc": [],
}

GRADE_PATTERNS = [
    r"Fe\s*\d{3}[A-Z]?", r"M\s*\d{2,3}",
    r"\d{2,3}\s*[Gg]rade", r"Grade\s*\d+",
]

DOMAIN_TERMS = [
    "compressive strength", "tensile strength", "flexural strength",
    "water absorption", "fineness", "setting time", "soundness",
    "workability", "slump", "durability", "chemical composition",
    "physical requirements", "grading", "sampling", "acceptance criteria",
    "dimensions", "mix design", "curing", "quality control",
]

APP_TERMS = [
    "construction", "building", "road", "bridge", "foundation",
    "residential", "commercial", "industrial", "infrastructure",
    "roofing", "plastering", "water supply", "drainage",
]

TEST_TERMS = [
    "compressive strength test", "tensile test", "flexural test",
    "water absorption test", "fineness test", "setting time test",
    "soundness test", "sieve analysis", "slump test",
    "specific gravity", "bulk density", "chemical analysis",
]

ENG_TERMS = [
    "portland cement", "pozzolana", "fly ash", "slag", "silica fume",
    "aggregate", "admixture", "reinforcement", "prestressing",
    "water-cement ratio", "concrete mix", "steel grade", "yield strength",
]


def _detect_category(text: str) -> str:
    text_lower = text.lower()
    scores = {
        cat: sum(text_lower.count(kw) for kw in kws)
        for cat, kws in MATERIAL_CATEGORIES.items()
        if cat != "Misc"
    }
    active = {k: v for k, v in scores.items() if v > 0}
    return max(active, key=active.get) if active else "Misc"


def _extract_title(raw_text: str, sn: str) -> str:
    lines = raw_text.split("\n")
    parts = []
    started = False
    for line in lines[:15]:
        line = line.strip()
        if not line:
            continue
        if sn in line:
            after = re.sub(
                r"^IS\s+\d{1,5}(?:\s*\(\s*Part\s+\d+\s*\))?[\s:—–-]*", "", line
            ).strip()
            if after:
                parts.append(after)
                started = True
        elif started:
            if re.match(r"^(Scope|1\.|References|This standard)", line, re.I):
                break
            if len(line) > 5:
                parts.append(line)
                if len(parts) >= 3:
                    break
    title = re.sub(r"\s+", " ", " ".join(parts)).rstrip(".:;,")
    return title if title else f"BIS Standard {sn}"


def _extract_scope(raw_text: str) -> str:
    m = re.search(
        r"(?:Scope|SCOPE|1\s+Scope)[:\s]*(.+?)(?=\n\s*(?:2\s+|References|Terminology|Definitions))",
        raw_text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip())[:800]
    return raw_text[:400].strip()


def _extract_list(text: str, terms: List[str]) -> List[str]:
    text_lower = text.lower()
    return sorted(t for t in terms if t in text_lower)


def _extract_grades(text: str) -> List[str]:
    grades: set = set()
    for pat in GRADE_PATTERNS:
        for m in re.findall(pat, text, re.IGNORECASE):
            grades.add(m.strip())
    return sorted(grades)


def _extract_keywords(text: str, title: str) -> List[str]:
    stop = {"this", "that", "with", "from", "have", "been", "part",
            "standard", "code", "practice", "specification"}
    kws: set = set()
    for word in title.split():
        w = word.strip("().,;:-").lower()
        if len(w) > 3 and w not in stop:
            kws.add(w)
    for term in DOMAIN_TERMS:
        if term in text.lower():
            kws.add(term)
    return sorted(kws)[:20]


def extract_metadata(standard: Dict[str, str]) -> Dict[str, Any]:
    raw = standard["raw_text"]
    sn = standard["standard_number"]
    title = _extract_title(raw, sn)
    return {
        "standard_number": sn,
        "title": title,
        "material_category": _detect_category(raw),
        "scope_text": _extract_scope(raw),
        "keywords": _extract_keywords(raw, title),
        "grades": _extract_grades(raw),
        "applications": _extract_list(raw, APP_TERMS),
        "test_methods": _extract_list(raw, TEST_TERMS),
        "engineering_terms": _extract_list(raw, ENG_TERMS),
        "chunk_id": str(uuid.uuid4()),
        "raw_text": raw,
    }


# ─── Synthetic Query Generation (rule-based, no API needed) ──────────────────

def generate_queries(m: Dict[str, Any]) -> List[str]:
    sn, title, cat = m["standard_number"], m["title"], m["material_category"]
    kws, grades, apps = m["keywords"], m["grades"], m["applications"]
    q = [
        f"What BIS standard covers {title.lower()}?",
        f"Which IS code should I follow for {cat.lower()} in construction?",
        f"{sn} specification requirements",
        f"BIS standard for {' '.join(kws[:3])} compliance",
        f"Indian standard for {cat.lower()} quality testing",
    ]
    q.append(f"BIS code for {grades[0]} {cat.lower()}" if grades
              else f"IS code for {cat.lower()} material properties")
    q.append(f"Standard for {apps[0]} {cat.lower()} specification" if apps
              else f"Which standard covers {title.lower()} for building?")
    return q[:7]


# ─── Indexing ─────────────────────────────────────────────────────────────────

def build_chroma_index(
    all_metadata: List[Dict[str, Any]],
    all_queries: Dict[str, List[str]],
) -> None:
    logger.info("Building ChromaDB collections with Groq embeddings...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    for name in ["standards_text", "standards_synthetic"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    # ── Collection 1: standards_text ─────────────────────────────────────────
    text_col = client.get_or_create_collection(
        "standards_text", metadata={"hnsw:space": "cosine"}
    )

    text_docs, text_ids, text_metas = [], [], []
    for m in all_metadata:
        doc = (
            f"{m['standard_number']}: {m['title']}. "
            f"Category: {m['material_category']}. "
            f"Keywords: {', '.join(m['keywords'][:10])}. "
            f"Scope: {m['scope_text'][:500]}"
        )
        text_docs.append(doc)
        text_ids.append(m["chunk_id"])
        text_metas.append({
            "standard_number": m["standard_number"],
            "title": m["title"],
            "material_category": m["material_category"],
            "scope_text": m["scope_text"][:500],
            "keywords": json.dumps(m["keywords"]),
            "grades": json.dumps(m["grades"]),
            "applications": json.dumps(m["applications"]),
            "test_methods": json.dumps(m["test_methods"]),
            "engineering_terms": json.dumps(m["engineering_terms"]),
        })

    logger.info(f"Embedding {len(text_docs)} standard documents via Groq...")
    text_embeddings = embed_texts_batched(text_docs, desc="Embedding standards_text")

    for i in range(0, len(text_docs), 100):
        end = min(i + 100, len(text_docs))
        text_col.add(
            ids=text_ids[i:end],
            documents=text_docs[i:end],
            embeddings=text_embeddings[i:end],
            metadatas=text_metas[i:end],
        )
    logger.info(f"standards_text: {len(text_docs)} documents indexed")

    # ── Collection 2: standards_synthetic ────────────────────────────────────
    synth_col = client.get_or_create_collection(
        "standards_synthetic", metadata={"hnsw:space": "cosine"}
    )

    synth_docs, synth_ids, synth_metas = [], [], []
    for m in all_metadata:
        sn = m["standard_number"]
        for idx, q in enumerate(all_queries.get(sn, [])):
            synth_docs.append(q)
            synth_ids.append(f"{m['chunk_id']}_synth_{idx}")
            synth_metas.append({
                "standard_number": sn,
                "title": m["title"],
                "material_category": m["material_category"],
                "scope_text": m["scope_text"][:500],
                "keywords": json.dumps(m["keywords"]),
                "grades": json.dumps(m["grades"]),
                "applications": json.dumps(m["applications"]),
                "test_methods": json.dumps(m["test_methods"]),
                "engineering_terms": json.dumps(m["engineering_terms"]),
            })

    if synth_docs:
        logger.info(f"Embedding {len(synth_docs)} synthetic queries via Groq...")
        synth_embeddings = embed_texts_batched(synth_docs, desc="Embedding standards_synthetic")
        for i in range(0, len(synth_docs), 100):
            end = min(i + 100, len(synth_docs))
            synth_col.add(
                ids=synth_ids[i:end],
                documents=synth_docs[i:end],
                embeddings=synth_embeddings[i:end],
                metadatas=synth_metas[i:end],
            )
    logger.info(f"standards_synthetic: {len(synth_docs)} documents indexed")


def build_bm25_index(all_metadata: List[Dict[str, Any]]) -> None:
    logger.info("Building BM25 sparse index...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    corpus, doc_map = [], []
    for m in all_metadata:
        combined = (
            f"{m['standard_number']} {m['title']} "
            f"{' '.join(m['keywords'])} {' '.join(m['grades'])} "
            f"{m['scope_text'][:300]} {m['material_category']} "
            f"{' '.join(m['applications'])} {' '.join(m['engineering_terms'])}"
        )
        corpus.append(combined.lower().split())
        doc_map.append({k: v for k, v in m.items() if k != "raw_text"})

    bm25 = BM25Okapi(corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_map": doc_map, "corpus": corpus}, f)
    logger.info(f"BM25 index saved → {BM25_PATH}  ({len(corpus)} docs)")


def save_whitelist_and_metadata(all_metadata: List[Dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    whitelist = sorted(set(m["standard_number"] for m in all_metadata))
    with open(WHITELIST_PATH, "w") as f:
        json.dump(whitelist, f, indent=2)
    logger.info(f"Whitelist saved → {WHITELIST_PATH}  ({len(whitelist)} standards)")

    clean = [{k: v for k, v in m.items() if k != "raw_text"} for m in all_metadata]
    with open(METADATA_PATH, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"Metadata saved  → {METADATA_PATH}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vectorize BIS PDF using Ollama nomic-embed-text:v1.5"
    )
    parser.add_argument("--pdf", required=True, help="Path to BIS PDF (e.g. pdf/BIS.pdf)")
    parser.add_argument(
        "--skip-synthetic", action="store_true",
        help="Skip synthetic query generation (saves ~30%% of time)"
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        sys.exit(1)

    # 0. Verify API Keys
    logger.info("=" * 60)
    logger.info("STEP 0: Checking API connections")
    logger.info("=" * 60)
    check_api_keys()

    # 1. Parse PDF
    logger.info("=" * 60)
    logger.info("STEP 1: PDF Parsing")
    logger.info("=" * 60)
    full_text = extract_text_from_pdf(str(pdf_path))
    standards = split_into_standards(full_text)
    if not standards:
        logger.error("No standards found. Check PDF format.")
        sys.exit(1)

    # 2. Extract metadata
    logger.info("=" * 60)
    logger.info("STEP 2: Metadata Extraction")
    logger.info("=" * 60)
    all_metadata = []
    for s in tqdm(standards, desc="Extracting metadata"):
        all_metadata.append(extract_metadata(s))
    logger.info(f"Extracted metadata for {len(all_metadata)} standards")

    # 3. Synthetic queries
    logger.info("=" * 60)
    logger.info("STEP 3: Synthetic Query Generation (rule-based)")
    logger.info("=" * 60)
    all_queries: Dict[str, List[str]] = {}
    if not args.skip_synthetic:
        for m in tqdm(all_metadata, desc="Generating queries"):
            all_queries[m["standard_number"]] = generate_queries(m)
        logger.info(f"Generated queries for {len(all_queries)} standards")
    else:
        logger.info("Skipped (--skip-synthetic)")

    # 4. ChromaDB vector index
    logger.info("=" * 60)
    logger.info("STEP 4: Building ChromaDB vector index")
    logger.info("=" * 60)
    build_chroma_index(all_metadata, all_queries)

    # 5. BM25 sparse index
    logger.info("=" * 60)
    logger.info("STEP 5: Building BM25 sparse index")
    logger.info("=" * 60)
    build_bm25_index(all_metadata)

    # 6. Whitelist + metadata JSON
    logger.info("=" * 60)
    logger.info("STEP 6: Saving whitelist & metadata")
    logger.info("=" * 60)
    save_whitelist_and_metadata(all_metadata)

    logger.info("=" * 60)
    logger.info("VECTORIZATION COMPLETE")
    logger.info(f"  Standards indexed : {len(all_metadata)}")
    logger.info(f"  ChromaDB path     : {CHROMA_DIR}")
    logger.info(f"  BM25 index        : {BM25_PATH}")
    logger.info(f"  Whitelist         : {WHITELIST_PATH}")
    logger.info(f"  Metadata          : {METADATA_PATH}")
    logger.info("  All outputs are in data/ — commit that folder to GitHub")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
