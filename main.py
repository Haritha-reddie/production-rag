"""
main.py - CLI entry point

Commands:
  python main.py ingest           -- Load docs, build indexes
  python main.py query "..."      -- Ask a question
  python main.py eval             -- Run Ragas evaluation suite
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR  = "data/docs"
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
BM25_PATH  = os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl")


def cmd_ingest():
    from src.ingestion.loader import load_documents, chunk_documents
    from src.ingestion.indexer import build_vector_index, build_bm25_index

    print("\nIngestion pipeline starting...")
    print(f"  Source: {DOCS_DIR}\n")

    docs   = load_documents(DOCS_DIR)
    chunks = chunk_documents(docs)

    print(f"\nBuilding indexes...")
    build_vector_index(chunks, CHROMA_DIR)
    build_bm25_index(chunks, BM25_PATH)

    print("\nIngestion complete. Run: python main.py query \"your question\"")


def cmd_query(question: str):
    from src.ingestion.indexer import load_vector_index, load_bm25_index
    from src.generation.rag_chain import RAGChain

    if not Path(CHROMA_DIR).exists() or not Path(BM25_PATH).exists():
        print("Indexes not found. Run: python main.py ingest")
        sys.exit(1)

    vectorstore = load_vector_index(CHROMA_DIR)
    bm25, chunks = load_bm25_index(BM25_PATH)

    chain = RAGChain(
        bm25=bm25,
        chunks=chunks,
        vectorstore=vectorstore,
        hybrid_top_k=20,
        rerank_top_n=3,
    )

    response = chain.run(question)
    print("\n" + "=" * 60)
    print(response.display())
    print("=" * 60)


def cmd_eval():
    from src.ingestion.indexer import load_vector_index, load_bm25_index
    from src.generation.rag_chain import RAGChain
    from src.evaluation.eval_pipeline import run_evaluation, check_thresholds

    vectorstore = load_vector_index(CHROMA_DIR)
    bm25, chunks = load_bm25_index(BM25_PATH)

    chain = RAGChain(bm25=bm25, chunks=chunks, vectorstore=vectorstore)

    thresholds = {
        "faithfulness":      float(os.getenv("RAGAS_FAITHFULNESS_THRESHOLD",   0.85)),
        "answer_relevancy":  float(os.getenv("RAGAS_ANSWER_RELEVANCY_THRESHOLD",0.80)),
        "context_precision": float(os.getenv("RAGAS_CONTEXT_PRECISION_THRESHOLD",0.75)),
        "context_recall":    float(os.getenv("RAGAS_CONTEXT_RECALL_THRESHOLD",  0.75)),
    }

    scores = run_evaluation(chain, "tests/eval_dataset.json")

    print("\n" + "=" * 60)
    print("  Ragas Evaluation Results")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Score':>8}  {'Threshold':>10}  {'Status':>8}")
    print(f"  {'-'*55}")
    for metric, threshold in thresholds.items():
        score  = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  {metric:<25} {score:>8.3f}  {threshold:>10.3f}  {status:>8}")

    passed, failures = check_thresholds(scores, thresholds)
    print("\n" + "=" * 60)
    if passed:
        print("  All metrics passed.")
    else:
        print("  Some metrics failed:")
        for f in failures:
            print(f)


def print_help():
    print("""
Production RAG

Usage:
  python main.py ingest               Build indexes from data/docs/
  python main.py query "question"     Query the RAG system
  python main.py eval                 Run Ragas evaluation suite
""")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print_help()
    elif args[0] == "ingest":
        cmd_ingest()
    elif args[0] == "query":
        if len(args) < 2:
            print("Provide a question: python main.py query \"your question\"")
            sys.exit(1)
        cmd_query(args[1])
    elif args[0] == "eval":
        cmd_eval()
    else:
        print(f"Unknown command: {args[0]}")
        print_help()
        sys.exit(1)
