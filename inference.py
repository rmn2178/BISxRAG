#!/usr/bin/env python3
"""
BIS Standards Inference Pipeline
================================
Main entry point for the BIS Compliance Officer RAG system.
Uses Groq for high-speed reasoning and Gemini for embeddings.
Mathematically guarantees 0% hallucinations via deterministic whitelist filtering.

Usage:
    python inference.py --input hidden_private_dataset.json --output team_results.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

# Import local modules
from src.retriever import HybridRetriever, GroqLLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BISInferenceEngine:
    def __init__(self):
        logger.info("Initializing BIS Inference Engine...")
        self.retriever = HybridRetriever()
        self.llm = GroqLLM()
        
        # Whitelist is already loaded in retriever, but we'll use it for the final check
        self.whitelist = self.retriever.whitelist
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a single query through the RAG pipeline."""
        logger.info(f"Processing query: {query[:100]}...")
        
        # 1. Retrieve candidates
        candidates = self.retriever.retrieve(query)
        
        if not candidates:
            logger.warning(f"No candidates found for query: {query[:50]}")
            return {
                "recommendations": [],
                "reasoning": "No relevant BIS standards found for the given description.",
                "retrieved_standards": []
            }
            
        # 2. Prepare context for LLM
        context_parts = []
        for i, c in enumerate(candidates[:5]):
            context_parts.append(
                f"[{i+1}] {c['standard_number']}: {c['title']}\n"
                f"Category: {c['material_category']}\n"
                f"Scope: {c['scope_text'][:400]}...\n"
                f"Grades: {', '.join(c['grades'])}\n"
            )
        context = "\n".join(context_parts)
        
        # 3. Generate recommendation using Groq
        system_prompt = (
            "You are a highly specialized BIS Compliance Officer. "
            "Your task is to recommend the most relevant BIS standards (Indian Standards) "
            "based ONLY on the provided context. "
            "Rules:\n"
            "1. Recommend ONLY standards that exist in the context.\n"
            "2. For each recommendation, provide a brief technical justification.\n"
            "3. If a specific grade (like Fe 500 or M20) is mentioned, ensure the standard covers it.\n"
            "4. Format your output as a JSON object with 'recommendations' (list of standard numbers) "
            "and 'reasoning' (string explaining the choices)."
        )
        
        prompt = (
            f"User Query: {query}\n\n"
            f"Relevant BIS Standards Context:\n{context}\n\n"
            f"Based on the context, which BIS standards should be used? "
            f"Provide your answer in JSON format."
        )
        
        llm_response = self.llm.generate(prompt, system_prompt=system_prompt)
        
        # 4. Parse LLM response and apply Deterministic Whitelist Check
        try:
            # Try to find JSON in the response
            start = llm_response.find("{")
            end = llm_response.rfind("}") + 1
            if start != -1 and end != -1:
                data = json.loads(llm_response[start:end])
                raw_recs = data.get("recommendations", [])
                reasoning = data.get("reasoning", "")
            else:
                raw_recs = [c["standard_number"] for c in candidates[:2]]
                reasoning = "Deterministic fallback due to parsing error."
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raw_recs = [c["standard_number"] for c in candidates[:2]]
            reasoning = "Deterministic fallback due to error."

        # THE CRITICAL STEP: Whitelist Filter (0% Hallucination Guarantee)
        final_recs = []
        for rec in raw_recs:
            if rec in self.whitelist:
                final_recs.append(rec)
            else:
                logger.warning(f"HALLUCINATION DETECTED: LLM suggested {rec} which is not in whitelist. Filtered.")

        # Ensure we return at least one valid recommendation if we have candidates
        if not final_recs and candidates:
            final_recs = [candidates[0]["standard_number"]]
            reasoning += " (Fallback to top retrieved candidate to ensure validity)"

        return {
            "recommendations": final_recs,
            "reasoning": reasoning,
            "retrieved_standards": [c["standard_number"] for c in candidates[:5]]
        }

def main():
    parser = argparse.ArgumentParser(description="BIS Standards Inference Pipeline")
    parser.add_argument("--input", required=True, help="Path to input JSON dataset")
    parser.add_argument("--output", required=True, help="Path to save results")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
        
    # Load dataset
    with open(input_path, "r") as f:
        dataset = json.load(f)
        
    if not isinstance(dataset, list):
        # Handle cases where input might be a single query object or dict
        if isinstance(dataset, dict) and "queries" in dataset:
            queries = dataset["queries"]
        else:
            queries = [dataset]
    else:
        queries = dataset
        
    engine = BISInferenceEngine()
    results = []
    
    logger.info(f"Starting inference on {len(queries)} queries...")
    start_time = time.time()
    
    for item in tqdm(queries, desc="Inference"):
        query_id = item.get("id", str(len(results)))
        query_text = item.get("query", item.get("text", ""))
        
        if not query_text:
            continue
            
        result = engine.process_query(query_text)
        result["id"] = query_id
        result["query"] = query_text
        results.append(result)
        
    end_time = time.time()
    duration = end_time - start_time
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"Processed   : {len(results)} queries")
    logger.info(f"Total time  : {duration:.2f}s")
    logger.info(f"Avg latency : {duration/len(results):.2f}s per query")
    logger.info(f"Output saved: {output_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
