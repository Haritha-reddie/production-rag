"""
tests/test_retrieval.py
-----------------------
Unit tests for hybrid retrieval and RRF fusion.
These tests run WITHOUT API keys — they use mock data.
"""

import pytest
from langchain.schema import Document
from src.retrieval.hybrid import reciprocal_rank_fusion, bm25_search
from rank_bm25 import BM25Okapi


def make_doc(chunk_id: str, text: str) -> Document:
    return Document(page_content=text, metadata={"chunk_id": chunk_id})


# ──────────────────────────────────────────────────────────────
# RRF Fusion tests
# ──────────────────────────────────────────────────────────────

def test_rrf_deduplicates_same_chunk():
    """A chunk appearing in both lists should appear once in the output."""
    bm25_results = [make_doc("chunk_001", "return policy 30 days")]
    vector_results = [make_doc("chunk_001", "return policy 30 days")]

    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    chunk_ids = [d.metadata["chunk_id"] for d in fused]

    assert chunk_ids.count("chunk_001") == 1, "Duplicate chunk in RRF output"


def test_rrf_boosts_chunk_in_both_lists():
    """A chunk ranked #1 by both methods should outscore chunks in only one."""
    bm25_results = [
        make_doc("chunk_top", "best match for everything"),
        make_doc("chunk_bm25_only", "keyword match only"),
    ]
    vector_results = [
        make_doc("chunk_top", "best match for everything"),
        make_doc("chunk_vec_only", "semantic match only"),
    ]

    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    top_chunk = fused[0].metadata["chunk_id"]

    assert top_chunk == "chunk_top", f"Expected 'chunk_top' at top, got '{top_chunk}'"


def test_rrf_score_attached():
    """RRF scores should be present in the output metadata."""
    docs_a = [make_doc("a", "hello world")]
    docs_b = [make_doc("b", "goodbye world")]

    fused = reciprocal_rank_fusion(docs_a, docs_b)
    for doc in fused:
        assert "rrf_score" in doc.metadata, f"Missing rrf_score on {doc.metadata['chunk_id']}"


def test_rrf_respects_top_k():
    """Output should not exceed top_k documents."""
    docs_a = [make_doc(f"a{i}", f"doc {i}") for i in range(10)]
    docs_b = [make_doc(f"b{i}", f"doc {i}") for i in range(10)]

    fused = reciprocal_rank_fusion(docs_a, docs_b, top_k=5)
    assert len(fused) <= 5


# ──────────────────────────────────────────────────────────────
# BM25 tests
# ──────────────────────────────────────────────────────────────

def test_bm25_returns_relevant_chunk():
    """BM25 should rank chunks with matching keywords higher."""
    chunks = [
        make_doc("chunk_001", "our return policy allows refunds within 30 days"),
        make_doc("chunk_002", "the weather in San Francisco is often foggy"),
        make_doc("chunk_003", "you can return items if you have the receipt"),
    ]
    texts = [c.page_content.lower().split() for c in chunks]
    bm25 = BM25Okapi(texts)

    results = bm25_search("return policy refund", bm25, chunks, top_k=3)
    top_ids = [r.metadata["chunk_id"] for r in results]

    # The weather chunk should NOT be top
    assert top_ids[0] != "chunk_002", "Irrelevant chunk ranked first"
    # Both return-related chunks should appear
    assert "chunk_001" in top_ids
    assert "chunk_003" in top_ids


def test_bm25_respects_top_k():
    """BM25 should return at most top_k results."""
    chunks = [make_doc(f"c{i}", f"word{i} content here") for i in range(20)]
    texts = [c.page_content.lower().split() for c in chunks]
    bm25 = BM25Okapi(texts)

    results = bm25_search("word content", bm25, chunks, top_k=5)
    assert len(results) <= 5
