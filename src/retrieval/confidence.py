"""
src/retrieval/confidence.py
----------------------------
Source confidence scoring for retrieved chunks.

Every chunk gets a composite confidence score based on:
  1. Rerank relevance  (40%) — Cohere cross-encoder score
  2. Keyword overlap   (30%) — exact word match with query
  3. Source freshness  (20%) — newer docs score higher
  4. Chunk completeness(10%) — longer chunks = more context

Why this matters:
  Low-confidence chunks should NEVER heavily influence generation.
  A chunk with confidence < 0.3 is flagged and either:
  - Excluded from the prompt
  - Marked as LOW CONFIDENCE in the citation
  This is a key hallucination prevention mechanism.
"""

import time
import re
from typing import List, Tuple
from langchain_core.documents import Document


# ── Confidence thresholds ──────────────────────────────────────
HIGH_CONFIDENCE   = 0.7   # Use fully — strong evidence
MEDIUM_CONFIDENCE = 0.4   # Use with caution note
LOW_CONFIDENCE    = 0.2   # Flag heavily, consider excluding
EXCLUDE_THRESHOLD = 0.1   # Do not use — too risky


def score_chunk_confidence(
    chunk:        Document,
    query:        str,
    rerank_score: float,
) -> float:
    """
    Compute a composite confidence score for a retrieved chunk.

    Args:
        chunk:        The retrieved document chunk
        query:        The user's original question
        rerank_score: Cohere reranker score (0-1)

    Returns:
        Composite confidence score (0.0 - 1.0)
    """

    # ── Component 1: Rerank relevance (40%) ────────────────────
    relevance = min(1.0, max(0.0, rerank_score))

    # ── Component 2: Keyword overlap (30%) ─────────────────────
    query_tokens = set(re.findall(r'\b\w{3,}\b', query.lower()))
    chunk_tokens = set(re.findall(r'\b\w{3,}\b', chunk.page_content.lower()))

    if query_tokens:
        overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
    else:
        overlap = 0.0
    overlap = min(1.0, overlap)

    # ── Component 3: Source freshness (20%) ────────────────────
    # Newer documents are generally more trustworthy
    created_at = chunk.metadata.get("created_at", None)
    if created_at:
        age_days  = (time.time() - created_at) / 86400
        freshness = max(0.0, 1.0 - (age_days / 365))
    else:
        freshness = 0.5  # Unknown age → neutral score

    # ── Component 4: Chunk completeness (10%) ──────────────────
    # Very short chunks may be truncated and lack context
    content_len  = len(chunk.page_content)
    completeness = min(1.0, content_len / 300)  # 300 chars = full score

    # ── Composite score ─────────────────────────────────────────
    confidence = (
        relevance    * 0.40 +
        overlap      * 0.30 +
        freshness    * 0.20 +
        completeness * 0.10
    )

    return round(min(1.0, max(0.0, confidence)), 3)


def score_chunks(
    chunks: List[Document],
    query:  str,
) -> List[Tuple[Document, float]]:
    """
    Score all chunks and return (chunk, confidence) tuples sorted by confidence.

    Args:
        chunks: Retrieved and reranked document chunks
        query:  The user's original question

    Returns:
        List of (chunk, confidence_score) tuples, sorted descending
    """
    scored = []
    for chunk in chunks:
        rerank_score = chunk.metadata.get("rerank_score", 0.0)
        confidence   = score_chunk_confidence(chunk, query, rerank_score)

        # Stamp confidence onto the chunk metadata
        chunk.metadata["confidence_score"] = confidence
        chunk.metadata["confidence_level"] = _confidence_level(confidence)

        scored.append((chunk, confidence))

    # Sort by confidence descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def filter_by_confidence(
    chunks:    List[Document],
    query:     str,
    threshold: float = EXCLUDE_THRESHOLD,
) -> List[Document]:
    """
    Filter out chunks below the confidence threshold.
    Returns only chunks safe to use for generation.
    """
    scored   = score_chunks(chunks, query)
    filtered = [chunk for chunk, score in scored if score >= threshold]

    if len(filtered) < len(chunks):
        excluded = len(chunks) - len(filtered)
        print(f"  🛡️  Confidence filter: excluded {excluded} low-confidence chunk(s)")

    return filtered


def _confidence_level(score: float) -> str:
    """Human-readable confidence label."""
    if score >= HIGH_CONFIDENCE:   return "HIGH"
    if score >= MEDIUM_CONFIDENCE: return "MEDIUM"
    if score >= LOW_CONFIDENCE:    return "LOW"
    return "VERY_LOW"


def confidence_report(chunks: List[Document], query: str) -> dict:
    """
    Generate a confidence report for a set of retrieved chunks.
    Useful for the monitoring dashboard.
    """
    scored = score_chunks(chunks, query)

    return {
        "total_chunks":  len(scored),
        "high":          sum(1 for _, s in scored if s >= HIGH_CONFIDENCE),
        "medium":        sum(1 for _, s in scored if MEDIUM_CONFIDENCE <= s < HIGH_CONFIDENCE),
        "low":           sum(1 for _, s in scored if LOW_CONFIDENCE <= s < MEDIUM_CONFIDENCE),
        "very_low":      sum(1 for _, s in scored if s < LOW_CONFIDENCE),
        "avg_confidence":round(sum(s for _, s in scored) / len(scored), 3) if scored else 0,
        "min_confidence":round(min(s for _, s in scored), 3) if scored else 0,
        "max_confidence":round(max(s for _, s in scored), 3) if scored else 0,
        "scores":        [(c.metadata.get("chunk_id"), round(s, 3)) for c, s in scored],
    }
