# Production RAG Application

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) [![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://www.langchain.com/) [![Groq](https://img.shields.io/badge/Groq-Llama3.3-orange.svg)](https://groq.com/) [![Cohere](https://img.shields.io/badge/Cohere-Rerank-blue.svg)](https://cohere.com/) [![Ragas](https://img.shields.io/badge/Ragas-Evaluation-purple.svg)](https://docs.ragas.io/)

A production-grade document Q&A system that answers questions from your own files using hybrid retrieval, cross-encoder reranking, citation enforcement, and automated quality evaluation.

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [How It Works](#how-it-works)
  - [Step 1: Ingestion](#step-1-ingestion)
  - [Step 2: Hybrid Retrieval](#step-2-hybrid-retrieval)
  - [Step 3: Reranking](#step-3-reranking)
  - [Step 4: Generation with Citations](#step-4-generation-with-citations)
  - [Step 5: Evaluation and CI Gate](#step-5-evaluation-and-ci-gate)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Stack](#stack)

---

## What This Project Does

You drop any document into the system, a PDF, a text file, or a markdown file, and then ask questions about it in plain English. The system finds the most relevant parts of your document and generates an answer that cites exactly which chunk it used.

The key difference from a basic chatbot is that this system is built around a core constraint: the model is not allowed to answer from its own training knowledge. Every answer must come from the documents you provide, and every claim must point back to a specific source chunk. If the system cannot find relevant content, it says so instead of guessing.

---

## How It Works

### Step 1: Ingestion

Before the system can answer questions, it needs to process and index the documents. The ingestion pipeline handles three tasks: loading documents, splitting them into chunks, and building two separate search indexes.

**Loading documents**

The loader reads `.txt`, `.pdf`, and `.md` files from the `data/docs/` directory and assigns a source filename to each one.

```python
from src.ingestion.loader import load_documents, chunk_documents

docs = load_documents("data/docs")
# OUTPUT: Loaded 1 raw document pages from data/docs
```

**Chunking**

Each document is split into chunks of 512 tokens with a 50-token overlap. The overlap ensures that sentences at the boundary of two chunks are not cut off and lost.

```python
chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=50)
# OUTPUT: Created 6 chunks (size=512, overlap=50)
```

Each chunk gets a stable `chunk_id` stamped into its metadata. This ID is what powers the citation system later.

**Building two indexes**

The system builds two completely different indexes from the same chunks.

```python
from src.ingestion.indexer import build_vector_index, build_bm25_index

# Vector index: embeddings stored in ChromaDB
vectorstore = build_vector_index(chunks, persist_dir="./chroma_db")

# Keyword index: BM25 stored as a pickle file
bm25, chunks = build_bm25_index(chunks, index_path="./bm25_index.pkl")
```

The vector index captures the meaning of each chunk. The BM25 index captures the exact words. Both are needed because they are good at different things.

---

### Step 2: Hybrid Retrieval

When a user asks a question, both indexes are queried at the same time and their results are merged.

**Why two indexes?**

Vector search alone fails on exact terminology. If a user asks "what does GDPR article 17 require", vector search may dilute the meaning across the embedding space and miss the right chunk. BM25 finds "GDPR" and "article 17" exactly.

BM25 alone fails on semantic queries. If a user asks "can I get my money back", BM25 finds nothing because those words do not appear in a document that talks about "return policy". Vector search finds it because the meaning is similar.

Using both covers both failure modes.

**BM25 search**

```python
from src.retrieval.hybrid import bm25_search

bm25_results = bm25_search(
    query="What is the return policy?",
    bm25=bm25,
    chunks=chunks,
    top_k=20
)
```

**Vector search**

```python
from src.retrieval.hybrid import vector_search

vector_results = vector_search(
    query="What is the return policy?",
    vectorstore=vectorstore,
    top_k=20
)
```

**RRF Fusion**

Reciprocal Rank Fusion merges the two ranked lists into one. A chunk that appears in both lists scores higher than a chunk that appears in only one. The formula is:

```
score = sum of 1 / (rank + 60) for each retrieval method
```

```python
from src.retrieval.hybrid import reciprocal_rank_fusion

fused_results = reciprocal_rank_fusion(
    bm25_results=bm25_results,
    vector_results=vector_results,
    top_k=20
)
# OUTPUT: 20 unique deduplicated candidates
```

---

### Step 3: Reranking

The 20 candidates from hybrid retrieval are passed to Cohere's reranker, which reduces them to the top 3.

**Why rerank after retrieval?**

Embeddings encode the query and the document independently. They compare two separate vectors. A cross-encoder, which is what Cohere uses, reads the query and the document together in a single pass. This is much more accurate because the model sees how the query and document relate to each other directly.

The tradeoff is speed. Running a cross-encoder on thousands of chunks would be too slow. Running it on 20 candidates is fast, and it gives you close to the accuracy of running it on everything.

```python
from src.retrieval.reranker import rerank

reranked = rerank(
    query="What is the return policy?",
    candidates=fused_results,
    top_n=3
)

# OUTPUT:
# [chunk_00000] rerank=0.916
# [chunk_00001] rerank=0.089
# [chunk_00003] rerank=0.005
```

The reranker assigns a relevance score to each chunk. Chunks with a score below a threshold are excluded from generation. If no chunk scores above the threshold, the system returns a fallback message instead of generating an answer from weak context.

---

### Step 4: Generation with Citations

The top 3 reranked chunks are formatted into a prompt and sent to the LLM.

**The system prompt**

The prompt enforces three rules:

```python
SYSTEM_PROMPT = """You are a precise document assistant.

RULES:
1. Only use information from the provided context. Do NOT add outside knowledge.
2. Every factual claim must be followed by a citation: [Source: chunk_id]
3. If the context does not contain the answer, say:
   "I don't have enough information in the provided documents to answer this."
4. Never fabricate information or citations.
"""
```

**Building the prompt**

Each chunk is formatted with its ID and source file so the model knows what to cite.

```python
from src.generation.prompt import build_rag_prompt

prompt = build_rag_prompt(
    query="What is the return policy?",
    context_docs=reranked,
    doc_names=["acme_policy.txt"]
)
```

The prompt includes blocks like this for each chunk:

```
[chunk_00000] (source: acme_policy.txt)
RETURN POLICY — Items purchased from ACME Corp may be returned within 30 days...
```

**The response**

```
Items purchased from ACME Corp may be returned within 30 days of the
original purchase date [Source: chunk_00000]. To be eligible for a return,
the item must be unused and in its original packaging [Source: chunk_00000].
Refunds are processed within 5 to 7 business days [Source: chunk_00000].
```

Every claim links back to a real chunk. Any claim without a citation is immediately suspicious and gets flagged during evaluation.

---

### Step 5: Evaluation and CI Gate

After every push to the main branch, GitHub Actions runs a Ragas evaluation suite against a set of ground truth Q&A pairs.

**The evaluation dataset**

The dataset lives in `tests/eval_dataset.json` and contains questions with known correct answers.

```json
[
  {
    "question": "What is the return policy?",
    "ground_truth": "Items can be returned within 30 days of purchase with a receipt."
  },
  {
    "question": "How do I contact customer support?",
    "ground_truth": "Customer support can be reached at support@example.com."
  }
]
```

**The four metrics**

```python
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

metrics = [
    Faithfulness(),      # does the answer use only the retrieved context?
    AnswerRelevancy(),   # does the answer address the question?
    ContextPrecision(),  # were the retrieved chunks actually relevant?
    ContextRecall(),     # did retrieval find all the needed information?
]
```

Each metric returns a score between 0.0 and 1.0.

**The CI gate**

The gate script checks each metric against a threshold and exits with code 1 if any metric fails. This causes the GitHub Actions build to fail.

```python
THRESHOLDS = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.80,
    "context_precision": 0.75,
    "context_recall":    0.75,
}
```

Sample output from a passing run:

```
Metric                    Score     Threshold   Status
faithfulness              0.923     0.850       PASS
answer_relevancy          0.887     0.800       PASS
context_precision         0.812     0.750       PASS
context_recall            0.798     0.750       PASS

All metrics passed. CI gate APPROVED.
```

---

## Project Structure

```
production-rag/
├── main.py                        CLI runner
├── requirements.txt
├── .env.example
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py              load files, chunk at 512 tokens, stamp chunk_id
│   │   └── indexer.py             build ChromaDB vector index and BM25 index
│   ├── retrieval/
│   │   ├── hybrid.py              BM25 search, vector search, RRF fusion
│   │   └── reranker.py            Cohere cross-encoder, top 20 to top 3
│   ├── generation/
│   │   ├── prompt.py              citation-enforced system prompt
│   │   └── rag_chain.py           full pipeline orchestration
│   └── evaluation/
│       └── eval_pipeline.py       Ragas evaluation runner
│
├── ui/
│   ├── server.py                  FastAPI server
│   └── index.html                 drag-drop upload and chat interface
│
├── ci/
│   └── run_evals.py               CI gate script
│
├── tests/
│   ├── test_retrieval.py          unit tests for RRF and BM25
│   ├── test_generation.py         unit tests for prompt citation enforcement
│   └── eval_dataset.json          ground truth Q&A pairs
│
├── data/docs/
│   └── acme_policy.txt            sample document
│
└── .github/
    └── workflows/
        └── rag_eval.yml           GitHub Actions pipeline
```

---

## Setup and Installation

**Prerequisites**

- Python 3.11
- Conda or a virtual environment

**Step 1: Clone the repository**

```bash
git clone https://github.com/Haritha-reddie/production-rag.git
cd production-rag
```

**Step 2: Create and activate an environment**

```bash
conda activate ragenv
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Get your API keys**

All services have free tiers. No credit card required.

| Service | Purpose | Get it at |
|---|---|---|
| Groq | LLM inference (Llama 3.3 70B) | console.groq.com |
| Cohere | Cross-encoder reranking | dashboard.cohere.com |

**Step 5: Configure environment variables**

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...
```

---

## Running the Application

**Ingest documents**

```bash
set -a && source .env && set +a
python main.py ingest
```

Output:

```
Loading: acme_policy.txt
Loaded 1 raw document pages from data/docs
Created 6 chunks (size=512, overlap=50)
Vector index built (6 chunks)
BM25 index built (6 chunks)
Ingestion complete.
```

**Query from the terminal**

```bash
python main.py query "What is the return policy?"
```

Output:

```
Answer:
Items purchased from ACME Corp may be returned within 30 days [Source: chunk_00000].
Refunds are processed within 5 to 7 business days [Source: chunk_00000].

Sources used:
  [chunk_00000] acme_policy.txt (rerank=0.916)
  [chunk_00001] acme_policy.txt (rerank=0.089)
```

**Start the web UI**

```bash
python -m uvicorn ui.server:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your browser.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | /status | System health and indexed document list |
| POST | /upload | Upload and auto-ingest documents |
| POST | /query | Ask a question, returns answer and citations |
| DELETE | /documents/{filename} | Remove a document and re-index |

**Example request**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'
```

**Example response**

```json
{
  "answer": "Items can be returned within 30 days [Source: chunk_00000].",
  "sources": [
    {
      "chunk_id": "chunk_00000",
      "source": "acme_policy.txt",
      "preview": "RETURN POLICY - Items purchased from ACME Corp...",
      "rerank_score": 0.916
    }
  ]
}
```

---

## Evaluation

**Run the evaluation suite**

```bash
python main.py eval
```

**Run the CI gate manually**

```bash
python ci/run_evals.py
```

**Add your own test cases**

Edit `tests/eval_dataset.json` and add Q&A pairs in this format:

```json
{
  "question": "What payment methods are accepted?",
  "ground_truth": "We accept Visa, MasterCard, PayPal, and bank transfers."
}
```

---

## Stack

| Component | Tool |
|---|---|
| LLM | Groq, Llama 3.3 70B |
| Embeddings | HuggingFace all-MiniLM-L6-v2, runs locally |
| Vector store | ChromaDB, persisted to disk |
| Keyword search | rank-bm25, BM25Okapi |
| Reranking | Cohere rerank-english-v3.0 |
| Evaluation | Ragas 0.2.x |
| Backend | FastAPI |
| CI | GitHub Actions |

---

## Author

Haritha Gurram, Data Scientist and AI Engineer based in Dallas, TX.

harithagurram5@gmail.com | [github.com/Haritha-reddie](https://github.com/Haritha-reddie)
