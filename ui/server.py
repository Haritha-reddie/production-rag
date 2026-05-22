"""
ui/server.py - FastAPI backend

Endpoints:
  POST /upload    -- Upload and ingest documents
  POST /query     -- Ask a question
  GET  /status    -- Check system status
  DELETE /documents/{filename} -- Remove a document
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import load_documents, chunk_documents
from src.ingestion.indexer import (
    build_vector_index, build_bm25_index,
    load_vector_index, load_bm25_index
)
from src.generation.rag_chain import RAGChain

app = FastAPI(title="Production RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_DIR   = Path("data/docs")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
BM25_PATH  = os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

_chain = None


def get_chain():
    global _chain
    if _chain is None:
        if not Path(CHROMA_DIR).exists() or not Path(BM25_PATH).exists():
            return None
        vectorstore = load_vector_index(CHROMA_DIR)
        bm25, chunks = load_bm25_index(BM25_PATH)
        _chain = RAGChain(bm25=bm25, chunks=chunks, vectorstore=vectorstore)
    return _chain


def reset_chain():
    global _chain
    _chain = None


class QueryRequest(BaseModel):
    question: str

class SourceDoc(BaseModel):
    chunk_id: str
    source: str
    preview: str
    rerank_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]


@app.get("/status")
def status():
    ready = Path(CHROMA_DIR).exists() and Path(BM25_PATH).exists()
    docs  = [f.name for f in DOCS_DIR.iterdir() if f.suffix in (".txt", ".pdf", ".md")]
    return {"ready": ready, "document_count": len(docs), "documents": docs}


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    allowed = {".txt", ".pdf", ".md"}
    saved   = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in allowed:
            raise HTTPException(400, f"File type not supported: {file.filename}")
        dest = DOCS_DIR / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)

    docs   = load_documents(str(DOCS_DIR))
    chunks = chunk_documents(docs)
    build_vector_index(chunks, CHROMA_DIR)
    build_bm25_index(chunks, BM25_PATH)
    reset_chain()

    return {"uploaded": saved, "message": f"{len(saved)} file(s) uploaded and indexed."}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    chain = get_chain()
    if chain is None:
        raise HTTPException(503, "No documents indexed yet. Please upload documents first.")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    response = chain.run(req.question)

    sources = [
        SourceDoc(
            chunk_id=doc.metadata.get("chunk_id", "?"),
            source=doc.metadata.get("source", "?"),
            preview=doc.page_content[:200].replace("\n", " "),
            rerank_score=round(doc.metadata.get("rerank_score", 0.0), 3),
        )
        for doc in response.sources
    ]

    return QueryResponse(answer=response.answer, sources=sources)


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    target = DOCS_DIR / filename
    if not target.exists():
        raise HTTPException(404, "File not found.")
    target.unlink()

    remaining = list(DOCS_DIR.iterdir())
    if remaining:
        docs   = load_documents(str(DOCS_DIR))
        chunks = chunk_documents(docs)
        build_vector_index(chunks, CHROMA_DIR)
        build_bm25_index(chunks, BM25_PATH)
    else:
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        Path(BM25_PATH).unlink(missing_ok=True)

    reset_chain()
    return {"deleted": filename}


ui_dir = Path(__file__).parent
app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="static")
