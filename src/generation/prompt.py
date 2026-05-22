"""
src/generation/prompt.py
"""

from typing import List
from langchain_core.documents import Document

GREETING_WORDS = {
    "hi", "hello", "hey", "howdy", "hiya", "greetings",
    "good morning", "good afternoon", "good evening",
    "what's up", "sup", "yo"
}

def is_greeting(text: str) -> bool:
    cleaned = text.lower().strip().rstrip("!?,.")
    return cleaned in GREETING_WORDS or any(
        cleaned.startswith(g) for g in GREETING_WORDS
    )

SYSTEM_PROMPT = """You are an expert document assistant. Answer questions using only
the provided context documents.

RULES:
1. Only use information from the provided context. Do NOT add outside knowledge.
2. Every factual claim must be followed by a citation: [Source: chunk_id]
3. If multiple chunks support a claim, cite all of them: [Source: chunk_001, chunk_002]
4. If the context does not contain the answer, respond with:
   "I don't have enough information in the provided documents to answer this question."
5. Keep answers clear, detailed and well-structured.
6. Never fabricate information or citations.
"""

def build_rag_prompt(
    query: str,
    context_docs: List[Document],
    doc_names: List[str] = [],
) -> str:

    if is_greeting(query):
        docs_list = ", ".join(doc_names) if doc_names else "uploaded documents"
        return f"""{SYSTEM_PROMPT}

The user said hello. Respond warmly and:
1. Tell them what documents are available: {docs_list}
2. Suggest 3 specific example questions they could ask
Be friendly and helpful."""

    if not context_docs:
        docs_list = ", ".join(doc_names) if doc_names else "the uploaded documents"
        return f"""{SYSTEM_PROMPT}

Available documents: {docs_list}
The retrieval system found no relevant chunks for this query.

User question: {query}

Respond by:
1. Explaining what topics the available documents cover
2. Suggesting 3 related questions the user COULD ask
3. Being honest that this specific question is not covered"""

    context_blocks = []
    for doc in context_docs:
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        source   = doc.metadata.get("source", "unknown")
        block    = f"[{chunk_id}] (source: {source})\n{doc.page_content}"
        context_blocks.append(block)

    context_text = "\n\n---\n\n".join(context_blocks)
    docs_list    = ", ".join(doc_names) if doc_names else "uploaded documents"

    return f"""{SYSTEM_PROMPT}

=== AVAILABLE DOCUMENTS ===
{docs_list}

=== RETRIEVED CONTEXT ===

{context_text}

=== END OF CONTEXT ===

User Question: {query}

Provide a detailed, well-structured answer using the context above.
Cite every fact as [Source: chunk_id].

Answer:"""
