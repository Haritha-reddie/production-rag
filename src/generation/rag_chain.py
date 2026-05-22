"""
src/generation/rag_chain.py
-----------------------------
Production RAG chain without monitoring.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma

from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.generation.prompt import build_rag_prompt, SYSTEM_PROMPT, is_greeting


@dataclass
class RAGResponse:
    query:   str
    answer:  str
    sources: List[Document]
    response_type: str = "answer"

    def display(self) -> str:
        lines = [
            f"Question: {self.query}",
            "",
            f"Answer:\n{self.answer}",
        ]
        if self.sources:
            lines.append("\nSources used:")
            for doc in self.sources:
                cid   = doc.metadata.get("chunk_id", "?")
                src   = doc.metadata.get("source", "?")
                score = doc.metadata.get("rerank_score", 0.0)
                preview = doc.page_content[:100].replace("\n", " ")
                lines.append(f"  [{cid}] {src} (rerank={score:.3f}) -- {preview}...")
        return "\n".join(lines)


class RAGChain:
    def __init__(
        self,
        bm25: BM25Okapi,
        chunks: List[Document],
        vectorstore: Chroma,
        docs_dir: str = "data/docs",
        hybrid_top_k: int = 20,
        rerank_top_n: int = 3,
        temperature: float = 0.3,
    ):
        self.bm25          = bm25
        self.chunks        = chunks
        self.vectorstore   = vectorstore
        self.docs_dir      = docs_dir
        self.hybrid_top_k  = hybrid_top_k
        self.rerank_top_n  = rerank_top_n

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=os.environ["GROQ_API_KEY"],
        )

    def _get_doc_names(self) -> List[str]:
        path = Path(self.docs_dir)
        return [f.name for f in path.iterdir()
                if f.suffix in (".txt", ".pdf", ".md")]

    def run(self, query: str) -> RAGResponse:
        print(f"\nQuery: {query}")
        doc_names = self._get_doc_names()

        # Greeting
        if is_greeting(query):
            prompt   = build_rag_prompt(query, [], doc_names)
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            return RAGResponse(query=query, answer=response.content,
                               sources=[], response_type="greeting")

        # Retrieval
        print(f"  Step 1: Hybrid retrieval (top {self.hybrid_top_k})...")
        candidates = hybrid_search(
            query=query, bm25=self.bm25, chunks=self.chunks,
            vectorstore=self.vectorstore, top_k=self.hybrid_top_k,
        )
        print(f"    -> {len(candidates)} candidates retrieved")

        # Reranking
        print(f"  Step 2: Cohere reranking -> top {self.rerank_top_n}...")
        reranked = rerank(query=query, candidates=candidates, top_n=self.rerank_top_n)
        for doc in reranked:
            print(f"       [{doc.metadata['chunk_id']}] rerank={doc.metadata.get('rerank_score', 0):.3f}")

        top_score     = reranked[0].metadata.get("rerank_score", 0.0) if reranked else 0.0
        response_type = "answer" if top_score >= 0.01 else "out_of_context"

        # Generation
        print("  Step 3: Building prompt and generating answer...")
        prompt_text = build_rag_prompt(query, reranked, doc_names)
        response    = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ])

        return RAGResponse(
            query=query,
            answer=response.content,
            sources=reranked if response_type == "answer" else [],
            response_type=response_type,
        )
