"""
ci/run_evals.py - CI evaluation gate

Runs Ragas evaluation and exits with code 1 if any metric fails.
Called by GitHub Actions on every push to main.

Usage:
    python ci/run_evals.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.indexer import load_vector_index, load_bm25_index
from src.generation.rag_chain import RAGChain
from src.evaluation.eval_pipeline import run_evaluation, check_thresholds

CHROMA_DIR   = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
BM25_PATH    = os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl")
EVAL_DATASET = "tests/eval_dataset.json"

THRESHOLDS = {
    "faithfulness":      float(os.getenv("RAGAS_FAITHFULNESS_THRESHOLD",    0.85)),
    "answer_relevancy":  float(os.getenv("RAGAS_ANSWER_RELEVANCY_THRESHOLD", 0.80)),
    "context_precision": float(os.getenv("RAGAS_CONTEXT_PRECISION_THRESHOLD",0.75)),
    "context_recall":    float(os.getenv("RAGAS_CONTEXT_RECALL_THRESHOLD",   0.75)),
}


def main():
    print("=" * 60)
    print("  Production RAG -- CI Evaluation Gate")
    print("=" * 60)

    print("\nLoading indexes...")
    vectorstore  = load_vector_index(CHROMA_DIR)
    bm25, chunks = load_bm25_index(BM25_PATH)
    print(f"  Vector index: {CHROMA_DIR}")
    print(f"  BM25 index:   {BM25_PATH} ({len(chunks)} chunks)")

    chain  = RAGChain(bm25=bm25, chunks=chunks, vectorstore=vectorstore)
    scores = run_evaluation(chain, EVAL_DATASET)

    print("\n" + "=" * 60)
    print("  Ragas Evaluation Results")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Score':>8}  {'Threshold':>10}  {'Status':>8}")
    print(f"  {'-'*55}")
    for metric, threshold in THRESHOLDS.items():
        score  = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  {metric:<25} {score:>8.3f}  {threshold:>10.3f}  {status:>8}")

    passed, failures = check_thresholds(scores, THRESHOLDS)
    print("\n" + "=" * 60)

    if passed:
        print("  All metrics passed -- CI gate APPROVED")
        sys.exit(0)
    else:
        print("  CI gate FAILED:")
        for f in failures:
            print(f)
        sys.exit(1)


if __name__ == "__main__":
    main()
