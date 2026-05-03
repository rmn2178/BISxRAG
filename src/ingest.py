#!/usr/bin/env python3
"""
BIS Standards Ingestion Pipeline
=================================
Parses BIS SP21 PDF, extracts per-standard chunks with metadata,
generates synthetic queries via Claude, and indexes into ChromaDB + BM25.

Usage:
    python src/ingest.py --pdf data/BIS_SP21.pdf
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

import anthropic
import chromadb
import pdfplumber
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

# ─── Constants ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
WHITELIST_PATH = DATA_DIR / "standard_whitelist.json"
METADATA_PATH = DATA_DIR / "standards_metadata.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

MATERIAL_CATEGORIES = {
    "Cement": [
        "cement", "portland", "clinker", "opc", "ppc", "pozzolana",
        "sulphate resisting", "slag cement", "masonry cement", "white cement",
        "hydraulic", "high alumina cement", "rapid hardening",
    ],
    "Concrete": [
        "concrete", "rcc", "pcc", "ready mixed", "mix design", "admixture",
        "superplasticizer", "water reducer", "curing", "formwork",
        "precast", "prestressed", "reinforced concrete", "plain concrete",
    ],
    "Steel": [
        "steel", "tmt", "rebar", "reinforcement", "bar", "wire", "rod",
        "structural steel", "mild steel", "high tensile", "deformed bar",
        "welding", "electrode", "fe 500", "fe 415", "fe500", "fe415",
        "fe 550", "galvanized", "stainless",
    ],
    "Aggregates": [
        "aggregate", "sand", "gravel", "crushed stone", "coarse aggregate",
        "fine aggregate", "lightweight aggregate", "recycled aggregate",
        "all-in aggregate", "stone dust", "crusher dust", "boulder",
    ],
    "Masonry": [
        "brick", "block", "masonry", "wall", "aac", "fly ash brick",
        "concrete block", "hollow block", "solid block", "tile",
        "paving block", "autoclaved", "mortar",
    ],
    "Testing": [
        "test", "testing", "sampling", "method of test", "determination",
        "analysis", "chemical analysis", "physical test", "specification for testing",
        "compressive strength test", "tensile test", "fineness",
    ],
    "Structural": [
        "structural", "design", "code of practice", "earthquake",
        "seismic", "wind load", "loading", "foundation", "construction practice",
        "workmanship", "plain and reinforced",
    ],
    "Misc": [],
}

GRADE_PATTERNS = [
    r"Fe\s*\d{3}[A-Z]?",
    r"M\s*\d{2,3}",
    r"\d{2,3}\s*[Gg]rade",
    r"Grade\s*\d+",
    r"\d{2,3}\s*grade",
]

logger = logging.getLogger(__name__)


# ─── PDF Parsing ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF using pdfplumber."""
    logger.info(f"Extracting text from: {pdf_path}")
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="Reading PDF pages"):
            text = page.extract_text()
            if text:
                full_text.append(text)
    combined = "\n\n".join(full_text)
    logger.info(f"Extracted {len(combined)} characters from {len(full_text)} pages")
    return combined


def split_into_standards(text: str) -> List[Dict[str, str]]:
    """
    Split the full SP21 text into per-standard chunks.
    Uses IS number boundary detection: r'IS\\s+\\d+[-:\\s]'
    Each BIS standard = ONE atomic chunk. Never split mid-standard.
    """
    # Pattern to find standard boundaries
    # Matches patterns like "IS 456", "IS 1489", "IS 269:", "IS 383 -"
    boundary_pattern = r"(?=\bIS\s+\d{1,5}(?:\s*[-:(\s]))"

    chunks = re.split(boundary_pattern, text)
    standards = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Try to extract the IS number from the chunk beginning
        is_match = re.match(r"(IS\s+\d{1,5}(?:\s*\(\s*Part\s+\d+\s*\))?)", chunk)
        if is_match:
            standard_number = normalize_standard_number(is_match.group(1))
            standards.append({
                "standard_number": standard_number,
                "raw_text": chunk,
            })
        elif standards:
            # Append to previous standard if no IS number found
            standards[-1]["raw_text"] += "\n" + chunk

    logger.info(f"Split text into {len(standards)} standard chunks")

    # Merge duplicate standards (same IS number appearing multiple times)
    merged = {}
    for s in standards:
        key = s["standard_number"]
        if key in merged:
            merged[key]["raw_text"] += "\n" + s["raw_text"]
        else:
            merged[key] = s

    result = list(merged.values())
    logger.info(f"After merging duplicates: {len(result)} unique standards")
    return result


def normalize_standard_number(sn: str) -> str:
    """Normalize standard number format: 'IS  456' -> 'IS 456'"""
    sn = re.sub(r"\s+", " ", sn.strip())
    return sn


# ─── Metadata Extraction ────────────────────────────────────────────────────
def extract_title(raw_text: str, standard_number: str) -> str:
    """Extract the title from the standard text."""
    lines = raw_text.split("\n")
    title_parts = []
    started = False

    for line in lines[:15]:  # Title is usually in the first 15 lines
        line = line.strip()
        if not line:
            continue
        if standard_number in line:
            # Get text after the standard number and any separator
            after = re.sub(
                r"^IS\s+\d{1,5}(?:\s*\(\s*Part\s+\d+\s*\))?[\s:—–-]*",
                "",
                line,
            ).strip()
            if after:
                title_parts.append(after)
                started = True
        elif started:
            # Check if this continues the title or starts a new section
            if re.match(r"^(Scope|1\.|References|This standard)", line, re.IGNORECASE):
                break
            if len(line) > 5:
                title_parts.append(line)
                if len(title_parts) >= 3:
                    break

    title = " ".join(title_parts).strip()
    # Clean up common artifacts
    title = re.sub(r"\s+", " ", title)
    title = title.rstrip(".:;,")
    return title if title else f"BIS Standard {standard_number}"


def detect_category(text: str) -> str:
    """Detect the material category based on keyword frequency."""
    text_lower = text.lower()
    scores = {}

    for category, keywords in MATERIAL_CATEGORIES.items():
        if category == "Misc":
            continue
        score = sum(text_lower.count(kw.lower()) for kw in keywords)
        if score > 0:
            scores[category] = score

    if scores:
        return max(scores, key=scores.get)
    return "Misc"


def extract_grades(text: str) -> List[str]:
    """Extract grade specifications from text."""
    grades = set()
    for pattern in GRADE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            grades.add(m.strip())
    return sorted(grades)


def extract_keywords(text: str, title: str) -> List[str]:
    """Extract relevant keywords from the standard text."""
    keywords = set()

    # Extract from title
    for word in title.split():
        word_clean = word.strip("().,;:-").lower()
        if len(word_clean) > 3 and word_clean not in {
            "this", "that", "with", "from", "have", "been", "part",
            "standard", "code", "practice", "specification",
        }:
            keywords.add(word_clean)

    # Domain-specific keyword extraction
    domain_terms = [
        "compressive strength", "tensile strength", "flexural strength",
        "water absorption", "fineness", "setting time", "soundness",
        "workability", "slump", "durability", "corrosion resistance",
        "chemical composition", "physical requirements", "grading",
        "sampling", "acceptance criteria", "tolerance", "dimensions",
        "mix design", "curing", "quality control", "marking",
        "packing", "storage", "fire resistance", "thermal conductivity",
        "load bearing", "non-load bearing", "partition",
    ]
    text_lower = text.lower()
    for term in domain_terms:
        if term in text_lower:
            keywords.add(term)

    return sorted(keywords)[:20]


def extract_scope(raw_text: str) -> str:
    """Extract the scope section from the standard text."""
    scope_match = re.search(
        r"(?:Scope|SCOPE|1\s+Scope)[:\s]*(.+?)(?=\n\s*(?:2\s+|References|Terminology|Definitions|REFERENCES))",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if scope_match:
        scope = scope_match.group(1).strip()
        scope = re.sub(r"\s+", " ", scope)
        return scope[:800]

    # Fallback: use first 400 chars after the standard number
    return raw_text[:400].strip()


def extract_applications(text: str) -> List[str]:
    """Extract application areas mentioned in the text."""
    applications = set()
    app_terms = [
        "construction", "building", "road", "bridge", "foundation",
        "residential", "commercial", "industrial", "infrastructure",
        "pavement", "dam", "irrigation", "water supply", "drainage",
        "flooring", "roofing", "plastering", "structural member",
        "column", "beam", "slab", "wall", "retaining wall",
    ]
    text_lower = text.lower()
    for term in app_terms:
        if term in text_lower:
            applications.add(term)
    return sorted(applications)


def extract_test_methods(text: str) -> List[str]:
    """Extract test methods referenced in the text."""
    methods = set()
    test_terms = [
        "compressive strength test", "tensile test", "flexural test",
        "water absorption test", "fineness test", "setting time test",
        "soundness test", "sieve analysis", "slump test",
        "specific gravity", "bulk density", "moisture content",
        "chemical analysis", "impact test", "abrasion test",
    ]
    text_lower = text.lower()
    for term in test_terms:
        if term in text_lower:
            methods.add(term)
    return sorted(methods)


def extract_engineering_terms(text: str) -> List[str]:
    """Extract important engineering terms from the text."""
    terms = set()
    eng_terms = [
        "portland cement", "pozzolana", "fly ash", "slag",
        "silica fume", "aggregate", "admixture", "reinforcement",
        "prestressing", "post-tensioning", "formwork",
        "curing compound", "water-cement ratio", "concrete mix",
        "steel grade", "yield strength", "ultimate strength",
        "elongation", "bend test", "rebend test", "weldability",
    ]
    text_lower = text.lower()
    for term in eng_terms:
        if term in text_lower:
            terms.add(term)
    return sorted(terms)


def extract_metadata(standard: Dict[str, str]) -> Dict[str, Any]:
    """Extract full metadata from a standard chunk."""
    raw_text = standard["raw_text"]
    standard_number = standard["standard_number"]

    title = extract_title(raw_text, standard_number)
    category = detect_category(raw_text)
    scope_text = extract_scope(raw_text)
    keywords = extract_keywords(raw_text, title)
    grades = extract_grades(raw_text)
    applications = extract_applications(raw_text)
    test_methods = extract_test_methods(raw_text)
    engineering_terms = extract_engineering_terms(raw_text)

    return {
        "standard_number": standard_number,
        "title": title,
        "material_category": category,
        "scope_text": scope_text,
        "keywords": keywords,
        "grades": grades,
        "applications": applications,
        "test_methods": test_methods,
        "engineering_terms": engineering_terms,
        "chunk_id": str(uuid.uuid4()),
        "raw_text": raw_text,
    }


# ─── Synthetic Query Generation ─────────────────────────────────────────────
def generate_synthetic_queries(
    metadata: Dict[str, Any],
    client: Optional[anthropic.Anthropic] = None,
) -> List[str]:
    """
    Generate 7 diverse synthetic user search queries for a BIS standard
    using Claude API. This is key HyDE innovation at ingestion time.
    """
    if client is None:
        # Fallback: generate rule-based queries if no API key
        return generate_fallback_queries(metadata)

    prompt = f"""Generate 7 diverse user search queries that someone might use to discover this BIS standard.
The queries should represent different user personas searching for this standard.

Standard: {metadata['standard_number']}
Title: {metadata['title']}
Category: {metadata['material_category']}
Keywords: {', '.join(metadata['keywords'][:10])}
Scope: {metadata['scope_text'][:300]}

Include these types of queries:
1. MSE owner query (simple, business-focused)
2. Contractor query (practical, construction-focused)
3. Technical engineer query (specification-focused)
4. Compliance query (regulatory, certification-focused)
5. Specification query (material property-focused)
6. Testing query (quality control-focused)
7. Plain-language query (layperson, simple description)

Return ONLY a JSON array of 7 strings. No preamble, no explanation."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Parse JSON array from response
        # Handle cases where response might have markdown code blocks
        if "```" in text:
            text = re.search(r"\[.*\]", text, re.DOTALL).group()
        queries = json.loads(text)

        if isinstance(queries, list) and len(queries) >= 5:
            return [str(q) for q in queries[:7]]
    except Exception as e:
        logger.warning(f"Claude API failed for {metadata['standard_number']}: {e}")

    return generate_fallback_queries(metadata)


def generate_fallback_queries(metadata: Dict[str, Any]) -> List[str]:
    """Generate rule-based synthetic queries when Claude API is unavailable."""
    sn = metadata["standard_number"]
    title = metadata["title"]
    category = metadata["material_category"]
    keywords = metadata.get("keywords", [])
    grades = metadata.get("grades", [])
    applications = metadata.get("applications", [])

    queries = [
        f"What BIS standard covers {title.lower()}?",
        f"Which IS code should I follow for {category.lower()} in construction?",
        f"{sn} specification requirements",
        f"BIS standard for {' '.join(keywords[:3])} compliance",
        f"Indian standard for {category.lower()} quality testing",
    ]

    if grades:
        queries.append(f"BIS code for {grades[0]} {category.lower()}")
    else:
        queries.append(f"IS code for {category.lower()} material properties")

    if applications:
        queries.append(
            f"Standard for {applications[0]} {category.lower()} specification"
        )
    else:
        queries.append(f"Which standard covers {title.lower()} for building?")

    return queries[:7]


# ─── Indexing ────────────────────────────────────────────────────────────────
def build_chroma_index(
    all_metadata: List[Dict[str, Any]],
    all_synthetic_queries: Dict[str, List[str]],
    embedder: SentenceTransformer,
) -> None:
    """Build dual ChromaDB collections: standards_text and standards_synthetic."""
    logger.info("Building ChromaDB indices...")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collections to rebuild
    for name in ["standards_text", "standards_synthetic"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    # ── Collection 1: standards_text (original text + metadata) ──
    text_collection = client.get_or_create_collection(
        name="standards_text",
        metadata={"hnsw:space": "cosine"},
    )

    text_docs = []
    text_ids = []
    text_metadatas = []

    for m in tqdm(all_metadata, desc="Preparing text documents"):
        # Compose the document text for embedding
        doc_text = (
            f"{m['standard_number']}: {m['title']}. "
            f"Category: {m['material_category']}. "
            f"Keywords: {', '.join(m['keywords'][:10])}. "
            f"Scope: {m['scope_text'][:500]}"
        )
        text_docs.append(doc_text)
        text_ids.append(m["chunk_id"])
        text_metadatas.append({
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

    # Embed in batches
    logger.info("Embedding text documents...")
    text_embeddings = embedder.encode(
        text_docs, show_progress_bar=True, batch_size=32
    ).tolist()

    # Add in batches of 100 (ChromaDB limit)
    batch_size = 100
    for i in range(0, len(text_docs), batch_size):
        end = min(i + batch_size, len(text_docs))
        text_collection.add(
            ids=text_ids[i:end],
            documents=text_docs[i:end],
            embeddings=text_embeddings[i:end],
            metadatas=text_metadatas[i:end],
        )

    logger.info(f"Indexed {len(text_docs)} documents in standards_text")

    # ── Collection 2: standards_synthetic (synthetic queries + metadata) ──
    synth_collection = client.get_or_create_collection(
        name="standards_synthetic",
        metadata={"hnsw:space": "cosine"},
    )

    synth_docs = []
    synth_ids = []
    synth_metadatas = []

    for m in tqdm(all_metadata, desc="Preparing synthetic queries"):
        sn = m["standard_number"]
        queries = all_synthetic_queries.get(sn, [])
        for idx, q in enumerate(queries):
            synth_docs.append(q)
            synth_ids.append(f"{m['chunk_id']}_synth_{idx}")
            synth_metadatas.append({
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

    logger.info("Embedding synthetic queries...")
    if synth_docs:
        synth_embeddings = embedder.encode(
            synth_docs, show_progress_bar=True, batch_size=32
        ).tolist()

        for i in range(0, len(synth_docs), batch_size):
            end = min(i + batch_size, len(synth_docs))
            synth_collection.add(
                ids=synth_ids[i:end],
                documents=synth_docs[i:end],
                embeddings=synth_embeddings[i:end],
                metadatas=synth_metadatas[i:end],
            )

    logger.info(f"Indexed {len(synth_docs)} synthetic queries in standards_synthetic")


def build_bm25_index(all_metadata: List[Dict[str, Any]]) -> None:
    """Build and serialize BM25 index over combined text fields."""
    logger.info("Building BM25 index...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    corpus = []
    doc_map = []

    for m in all_metadata:
        # Combine key fields for sparse retrieval
        combined = (
            f"{m['standard_number']} {m['title']} "
            f"{' '.join(m['keywords'])} {' '.join(m['grades'])} "
            f"{m['scope_text'][:300]} {m['material_category']} "
            f"{' '.join(m['applications'])} {' '.join(m['engineering_terms'])}"
        )
        tokenized = combined.lower().split()
        corpus.append(tokenized)
        doc_map.append({
            "standard_number": m["standard_number"],
            "title": m["title"],
            "material_category": m["material_category"],
            "scope_text": m["scope_text"],
            "keywords": m["keywords"],
            "grades": m["grades"],
            "applications": m["applications"],
            "test_methods": m["test_methods"],
            "engineering_terms": m["engineering_terms"],
        })

    bm25 = BM25Okapi(corpus)
    bm25_data = {"bm25": bm25, "doc_map": doc_map, "corpus": corpus}

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_data, f)

    logger.info(f"BM25 index saved to {BM25_PATH} ({len(corpus)} documents)")


def build_whitelist(all_metadata: List[Dict[str, Any]]) -> None:
    """Generate standard_whitelist.json with all valid standard numbers."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    whitelist = sorted(set(m["standard_number"] for m in all_metadata))

    with open(WHITELIST_PATH, "w") as f:
        json.dump(whitelist, f, indent=2)

    logger.info(f"Whitelist saved to {WHITELIST_PATH} ({len(whitelist)} standards)")


def save_metadata(all_metadata: List[Dict[str, Any]]) -> None:
    """Save full metadata for all standards (used by retriever)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Remove raw_text to keep the file smaller
    clean = []
    for m in all_metadata:
        entry = {k: v for k, v in m.items() if k != "raw_text"}
        clean.append(entry)

    with open(METADATA_PATH, "w") as f:
        json.dump(clean, f, indent=2)

    logger.info(f"Metadata saved to {METADATA_PATH}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Ingest BIS SP21 PDF into vector + sparse indices"
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the BIS SP21 PDF file",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip synthetic query generation (faster, no API calls)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        sys.exit(1)

    # Step 1: PDF Parsing
    logger.info("=" * 60)
    logger.info("STEP 1: PDF Parsing")
    logger.info("=" * 60)
    full_text = extract_text_from_pdf(str(pdf_path))
    standards = split_into_standards(full_text)

    if not standards:
        logger.error("No standards found in PDF. Check the file format.")
        sys.exit(1)

    # Step 2: Metadata Extraction
    logger.info("=" * 60)
    logger.info("STEP 2: Metadata Extraction")
    logger.info("=" * 60)
    all_metadata = []
    for s in tqdm(standards, desc="Extracting metadata"):
        meta = extract_metadata(s)
        all_metadata.append(meta)

    logger.info(f"Extracted metadata for {len(all_metadata)} standards")

    # Step 3: Synthetic Query Generation
    logger.info("=" * 60)
    logger.info("STEP 3: Synthetic Query Generation")
    logger.info("=" * 60)
    all_synthetic_queries = {}

    claude_client = None
    if not args.skip_synthetic:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            claude_client = anthropic.Anthropic(api_key=api_key)
            logger.info("Claude API available — generating synthetic queries")
        else:
            logger.warning(
                "No ANTHROPIC_API_KEY found — using rule-based fallback queries"
            )

    for m in tqdm(all_metadata, desc="Generating synthetic queries"):
        queries = generate_synthetic_queries(m, claude_client)
        all_synthetic_queries[m["standard_number"]] = queries

    logger.info(
        f"Generated synthetic queries for {len(all_synthetic_queries)} standards"
    )

    # Step 4: Dual Vector Indexing
    logger.info("=" * 60)
    logger.info("STEP 4: Dual Vector Indexing (ChromaDB)")
    logger.info("=" * 60)
    embedder = SentenceTransformer(EMBED_MODEL)
    build_chroma_index(all_metadata, all_synthetic_queries, embedder)

    # Step 5: BM25 Index
    logger.info("=" * 60)
    logger.info("STEP 5: BM25 Index")
    logger.info("=" * 60)
    build_bm25_index(all_metadata)

    # Step 6: Whitelist + Metadata
    logger.info("=" * 60)
    logger.info("STEP 6: Whitelist & Metadata Generation")
    logger.info("=" * 60)
    build_whitelist(all_metadata)
    save_metadata(all_metadata)

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"  Standards indexed: {len(all_metadata)}")
    logger.info(f"  ChromaDB path:     {CHROMA_DIR}")
    logger.info(f"  BM25 index:        {BM25_PATH}")
    logger.info(f"  Whitelist:         {WHITELIST_PATH}")
    logger.info(f"  Metadata:          {METADATA_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
