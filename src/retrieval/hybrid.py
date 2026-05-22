"""
src/retrieval/hybrid.py
------------------------
Hybrid retrieval: BM25 (keyword) + Vector (semantic) fused via
Reciprocal Rank Fusion (RRF).

Why RRF?
  - Simple, parameter-free, robust to score scale differences
  - Each result's contribution = 1 / (rank + k) where k=60 (standard)
  - A doc ranked #1 by BM25 and #5 by vector beats one ranked #3 by both
"""

from typing import List, Dict
from langchain.schema import Document
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi


RRF_K = 60  # Standard RRF constant — higher k reduces the impact of top ranks


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    chunks: List[Document],
    top_k: int = 20,
) -> List[Document]:
    """Return top_k chunks by BM25 score."""
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Get indices sorted by score descending
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = ranked_indices[:top_k]

    results = []
    for idx in top_indices:
        doc = chunks[idx]
        doc.metadata["bm25_score"] = float(scores[idx])
        results.append(doc)

    return results


def vector_search(
    query: str,
    vectorstore: Chroma,
    top_k: int = 20,
) -> List[Document]:
    """Return top_k chunks by cosine similarity."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    docs = []
    for doc, score in results:
        doc.metadata["vector_score"] = float(score)
        docs.append(doc)

    return docs


def reciprocal_rank_fusion(
    bm25_results: List[Document],
    vector_results: List[Document],
    top_k: int = 20,
) -> List[Document]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score = sum over methods of: 1 / (rank + k)

    A document's chunk_id is its identity key — same chunk from both
    methods gets its scores summed, not duplicated.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Document] = {}

    # Score from BM25 ranks
    for rank, doc in enumerate(bm25_results, start=1):
        cid = doc.metadata["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + RRF_K)
        chunk_map[cid] = doc

    # Score from vector ranks
    for rank, doc in enumerate(vector_results, start=1):
        cid = doc.metadata["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + RRF_K)
        chunk_map[cid] = doc

    # Sort by fused RRF score
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, score in sorted_chunks[:top_k]:
        doc = chunk_map[cid]
        doc.metadata["rrf_score"] = score
        results.append(doc)

    return results


def hybrid_search(
    query: str,
    bm25: BM25Okapi,
    chunks: List[Document],
    vectorstore: Chroma,
    top_k: int = 20,
) -> List[Document]:
    """
    Full hybrid search pipeline:
      1. BM25 keyword retrieval
      2. Vector semantic retrieval
      3. RRF fusion
    Returns top_k fused and ranked chunks.
    """
    bm25_results = bm25_search(query, bm25, chunks, top_k=top_k)
    vector_results = vector_search(query, vectorstore, top_k=top_k)
    fused = reciprocal_rank_fusion(bm25_results, vector_results, top_k=top_k)

    return fused
