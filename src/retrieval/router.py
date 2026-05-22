"""
src/retrieval/router.py
------------------------
Logical routing — classifies incoming queries and routes them
to the correct datasource or handling strategy.

Routes:
  vectorstore   — answer from uploaded documents
  direct_answer — greetings, chitchat, simple facts
  out_of_scope  — questions clearly outside document scope

Why this matters:
  Not every query should hit the vector store.
  Routing saves cost, reduces latency, and gives better responses
  for queries the documents can't answer.
"""

import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ── Output schema ──────────────────────────────────────────────
class RouteDecision(BaseModel):
    """Structured routing decision from the LLM."""

    datasource: Literal["vectorstore", "direct_answer", "out_of_scope"] = Field(
        description=(
            "vectorstore: question about uploaded documents. "
            "direct_answer: greeting, chitchat, simple math, or general knowledge. "
            "out_of_scope: question clearly outside what any document could answer."
        )
    )
    confidence: float = Field(
        description="Confidence in this routing decision, 0.0 to 1.0",
        ge=0.0, le=1.0,
    )
    reasoning: str = Field(
        description="One sentence explaining why this route was chosen."
    )


# ── Router chain ───────────────────────────────────────────────
ROUTER_SYSTEM = """You are an expert at routing user questions to the correct handler.

You have access to:
1. vectorstore — a collection of uploaded company/domain documents
2. direct_answer — for greetings, chitchat, or questions you can answer directly
3. out_of_scope — for questions that clearly can't be answered from any document

Available documents: {doc_names}

Route the question to the most appropriate handler.
"""

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM),
    ("human", "{question}"),
])


class QueryRouter:
    """
    Routes incoming queries to the correct handler.

    Usage:
        router = QueryRouter()
        decision = router.route("What is the return policy?", doc_names=["policy.pdf"])
        # decision.datasource == "vectorstore"
    """

    def __init__(self):
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            api_key=os.environ["GROQ_API_KEY"],
        )
        self._chain = ROUTER_PROMPT | llm.with_structured_output(RouteDecision)

    def route(self, question: str, doc_names: list = None) -> RouteDecision:
        """
        Classify the query and return a routing decision.

        Args:
            question:  The user's question
            doc_names: List of available document names for context

        Returns:
            RouteDecision with datasource, confidence, and reasoning
        """
        docs_str = ", ".join(doc_names) if doc_names else "general documents"

        decision = self._chain.invoke({
            "question":  question,
            "doc_names": docs_str,
        })

        print(f"  🔀 Router: {decision.datasource} "
              f"(confidence={decision.confidence:.2f}) — {decision.reasoning}")

        return decision

    def should_use_vectorstore(self, question: str, doc_names: list = None) -> bool:
        """Quick helper — returns True if the query should hit the vectorstore."""
        decision = self.route(question, doc_names)
        return decision.datasource == "vectorstore"
