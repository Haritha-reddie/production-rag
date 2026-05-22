"""
src/ingestion/indexer.py
------------------------
Builds and persists two complementary indexes:
  1. ChromaDB — dense vector index for semantic search
  2. BM25     — sparse keyword index for exact-match search
"""

import os
import pickle
from typing import List, Tuple

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def build_vector_index(
    chunks: List[Document],
    persist_dir: str,
) -> Chroma:
    """
    Embed chunks with OpenAI ada-002 and store in ChromaDB.
    Persists to disk so it survives restarts.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"  Embedding {len(chunks)} chunks → ChromaDB at {persist_dir}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="production_rag",
    )
    print(f"✅ Vector index built ({len(chunks)} chunks)")
    return vectorstore


def load_vector_index(persist_dir: str) -> Chroma:
    """Load an existing ChromaDB index from disk."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="production_rag",
    )


def build_bm25_index(
    chunks: List[Document],
    index_path: str,
) -> Tuple[BM25Okapi, List[Document]]:
    """
    Build a BM25 index over the chunk texts.
    Returns the BM25 object and the original chunk list (needed for lookup).
    Persists both to disk as a pickle.
    """
    tokenized = [_tokenize(chunk.page_content) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"✅ BM25 index built ({len(chunks)} chunks) → {index_path}")
    return bm25, chunks


def load_bm25_index(index_path: str) -> Tuple[BM25Okapi, List[Document]]:
    """Load an existing BM25 index from disk."""
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]
