"""
src/retrieval/reranker.py
--------------------------
Cross-encoder reranking via Cohere Rerank API.

Why a cross-encoder after hybrid retrieval?
  - Bi-encoder models (embeddings) trade accuracy for speed — they encode
    query and doc independently, so they miss fine-grained relevance.
  - A cross-encoder sees (query, document) together and scores their
    joint relevance, which is much more accurate but expensive.
  - Strategy: fetch 20 candidates cheaply, rerank to top 3 accurately.
    This gives you the best of both worlds.
"""

import os
from typing import List
import cohere
from langchain.schema import Document


def rerank(
    query: str,
    candidates: List[Document],
    top_n: int = 3,
    model: str = "rerank-english-v3.0",
) -> List[Document]:
    """
    Rerank a list of candidate documents using Cohere's cross-encoder.

    Args:
        query:      The user's original question
        candidates: Up to ~20 documents from hybrid retrieval
        top_n:      How many to return after reranking (typically 3–5)
        model:      Cohere rerank model name

    Returns:
        top_n documents, sorted by rerank score descending
    """
    if not candidates:
        return []

    client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

    # Cohere rerank takes plain text strings
    texts = [doc.page_content for doc in candidates]

    response = client.rerank(
        query=query,
        documents=texts,
        top_n=top_n,
        model=model,
    )

    # Map rerank results back to original Document objects
    reranked: List[Document] = []
    for result in response.results:
        doc = candidates[result.index]
        doc.metadata["rerank_score"] = result.relevance_score
        reranked.append(doc)

    return reranked
