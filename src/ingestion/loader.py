"""
src/ingestion/loader.py
-----------------------
Loads documents from a directory and splits them into chunks.
Supports .txt, .md, and .pdf files using only built-in LangChain loaders
(no unstructured dependency needed).
"""

from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader


def load_documents(docs_dir: str) -> List[Document]:
    """
    Load all supported documents from a directory.
    .txt and .md files use TextLoader; .pdf uses PyPDFLoader.
    """
    docs_path = Path(docs_dir)
    all_docs: List[Document] = []

    for file_path in docs_path.rglob("*"):
        suffix = file_path.suffix.lower()

        if suffix in (".txt", ".md"):
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            continue

        print(f"  Loading: {file_path.name}")
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file_path.name

        all_docs.extend(docs)

    print(f"✅ Loaded {len(all_docs)} raw document pages from {docs_dir}")
    return all_docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents into overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i:05d}"

    print(f"✅ Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
