"""
src/retrieval/query_transformer.py
------------------------------------
Query transformation techniques that improve retrieval accuracy:

1. Multi-Query    — generate 3 versions of the question
2. Step-Back      — make question more general for broader retrieval
3. HyDE           — generate hypothetical answer, use it as search query
4. Decomposition  — break complex question into sub-questions

Why this matters:
  A user's original query might use different words than the documents.
  By transforming the query multiple ways, we cast a wider, more accurate net.
"""

import os
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── LLM ───────────────────────────────────────────────────────
def _get_llm(temperature: float = 0.3):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=os.environ["GROQ_API_KEY"],
    )


# ── 1. Multi-Query ─────────────────────────────────────────────
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template("""
You are an AI assistant helping improve document retrieval.
Generate exactly 3 different versions of the user's question.
Each version should use different words or angles but ask the same thing.
This helps retrieve documents that use different vocabulary.

Original question: {question}

Output exactly 3 questions, one per line, no numbering or bullets:
""")

def multi_query(question: str) -> List[str]:
    """
    Generate 3 alternative phrasings of the question.
    Returns the original + 3 alternatives (4 total).
    """
    llm    = _get_llm(temperature=0.4)
    chain  = MULTI_QUERY_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question})

    alternatives = [q.strip() for q in result.strip().split("\n") if q.strip()]
    # Always include the original + up to 3 alternatives
    all_queries = [question] + alternatives[:3]
    return all_queries


# ── 2. Step-Back ───────────────────────────────────────────────
STEPBACK_PROMPT = ChatPromptTemplate.from_template("""
You are an expert at rewriting specific questions into more general ones.
A "step-back" question is a broader, more general version of the original.

Examples:
- Specific: "What was the return policy change in Q3 2024?"
- Step-back: "What is the return policy?"

- Specific: "What does GDPR say about cookie consent for EU users under 16?"
- Step-back: "What does GDPR say about user consent?"

Now rewrite this question as a step-back question:
Original: {question}
Step-back question (one line only):
""")

def step_back(question: str) -> str:
    """
    Generate a more general version of the question.
    Useful when the original is too specific to match documents.
    """
    llm   = _get_llm(temperature=0.2)
    chain = STEPBACK_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question}).strip()


# ── 3. HyDE ────────────────────────────────────────────────────
HYDE_PROMPT = ChatPromptTemplate.from_template("""
Write a short passage (3-4 sentences) that would be a perfect answer to this question.
This passage will be used to search a document database, so use relevant terminology.
Do not say "I don't know" — write the most plausible answer even if uncertain.

Question: {question}

Passage:
""")

def hyde(question: str) -> str:
    """
    Generate a Hypothetical Document Embedding (HyDE).

    Instead of searching with the question, we generate a hypothetical
    answer and search with that. This bridges the vocabulary gap between
    questions and documents.
    """
    llm   = _get_llm(temperature=0.5)
    chain = HYDE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question}).strip()


# ── 4. Decomposition ───────────────────────────────────────────
DECOMPOSE_PROMPT = ChatPromptTemplate.from_template("""
Break down this complex question into 2-3 simpler sub-questions.
Each sub-question should be answerable on its own.
Only decompose if the question is genuinely complex — if it's simple, return just the original.

Question: {question}

Output one sub-question per line, no numbering:
""")

def decompose(question: str) -> List[str]:
    """
    Break a complex question into simpler sub-questions.
    Each sub-question is answered individually, then synthesized.
    """
    llm    = _get_llm(temperature=0.3)
    chain  = DECOMPOSE_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question})

    sub_questions = [q.strip() for q in result.strip().split("\n") if q.strip()]
    return sub_questions if len(sub_questions) > 1 else [question]


# ── Combined transformer ────────────────────────────────────────
class QueryTransformer:
    """
    Applies multiple query transformation strategies and returns
    all generated queries for use in retrieval.

    Strategy selection:
    - Short questions (<6 words)  → multi-query + step-back
    - Long/complex questions       → decomposition + multi-query
    - All questions                → HyDE always included
    """

    def transform(self, question: str) -> dict:
        """
        Transform a query using all applicable strategies.

        Returns:
            {
                "original":    str,
                "multi_query": List[str],
                "step_back":   str,
                "hyde":        str,
                "decomposed":  List[str],
                "all_queries": List[str],  # deduplicated union for retrieval
            }
        """
        word_count = len(question.split())

        multi   = multi_query(question)
        sb      = step_back(question) if word_count < 12 else question
        hyd     = hyde(question)
        decomp  = decompose(question) if word_count >= 8 else [question]

        # Union of all queries for retrieval
        all_queries = list(dict.fromkeys(
            multi + [sb, hyd] + decomp
        ))

        return {
            "original":    question,
            "multi_query": multi,
            "step_back":   sb,
            "hyde":        hyd,
            "decomposed":  decomp,
            "all_queries": all_queries,
        }
