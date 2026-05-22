"""
tests/test_generation.py
-------------------------
Unit tests for prompt building and citation enforcement.
No API keys required — these test the prompt structure only.
"""

import pytest
from langchain.schema import Document
from src.generation.prompt import build_rag_prompt


def make_doc(chunk_id: str, source: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "source": source},
    )


def test_prompt_contains_chunk_ids():
    """Each chunk_id should appear in the prompt so the LLM can cite it."""
    docs = [
        make_doc("chunk_001", "policy.pdf", "Return within 30 days."),
        make_doc("chunk_002", "faq.txt", "Contact support@example.com."),
    ]
    prompt = build_rag_prompt("How do I return an item?", docs)

    assert "chunk_001" in prompt
    assert "chunk_002" in prompt


def test_prompt_contains_source_filenames():
    """Source filenames should appear in the prompt for traceability."""
    docs = [make_doc("chunk_001", "returns_policy.pdf", "30-day returns.")]
    prompt = build_rag_prompt("Returns?", docs)
    assert "returns_policy.pdf" in prompt


def test_prompt_contains_query():
    """The user's question should be included verbatim in the prompt."""
    docs = [make_doc("chunk_001", "doc.txt", "Some content.")]
    query = "What is the cancellation policy?"
    prompt = build_rag_prompt(query, docs)
    assert query in prompt


def test_prompt_instructs_citations():
    """The prompt must contain citation instructions to enforce grounded answers."""
    docs = [make_doc("chunk_001", "doc.txt", "Some content.")]
    prompt = build_rag_prompt("Any question?", docs)

    assert "Source:" in prompt or "citation" in prompt.lower(), \
        "Prompt must instruct the LLM to include citations"


def test_prompt_handles_empty_context():
    """Prompt should still be valid with zero context docs."""
    prompt = build_rag_prompt("Any question?", [])
    assert "Any question?" in prompt
    assert isinstance(prompt, str)
    assert len(prompt) > 0
