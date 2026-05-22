"""
src/evaluation/eval_pipeline.py
---------------------------------
Ragas evaluation pipeline (compatible with ragas>=0.4.x).

Ragas 0.4.x API changes vs 0.1.x:
  - Metrics are now classes, not module-level objects
  - evaluate() takes a list of metric instances
  - EvaluationDataset replaces HuggingFace Dataset directly
  - Results are accessed via result['metric_name']
"""

import os
import json
from typing import List, Dict, Any

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain.schema import Document


def load_eval_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation Q&A pairs from a JSON file.

    Expected format:
    [
      {
        "question": "What is the return policy?",
        "ground_truth": "Items can be returned within 30 days."
      },
      ...
    ]
    """
    with open(path, "r") as f:
        return json.load(f)


def run_evaluation(
    rag_chain,
    eval_dataset_path: str,
) -> Dict[str, float]:
    """
    Run the RAG chain over the eval dataset and score with Ragas 0.4.x.

    Returns a dict of metric_name → score (0.0–1.0).
    """
    eval_items = load_eval_dataset(eval_dataset_path)
    print(f"\n📊 Running Ragas evaluation on {len(eval_items)} questions...")

    samples = []

    for item in eval_items:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Run RAG chain
        response = rag_chain.run(question)

        sample = SingleTurnSample(
            user_input=question,
            response=response.answer,
            retrieved_contexts=[doc.page_content for doc in response.sources],
            reference=ground_truth,
        )
        samples.append(sample)

    # Build Ragas EvaluationDataset
    ragas_dataset = EvaluationDataset(samples=samples)

    # Instantiate metrics (0.4.x style)
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    result = evaluate(dataset=ragas_dataset, metrics=metrics)

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
    }

    return scores


def check_thresholds(
    scores: Dict[str, float],
    thresholds: Dict[str, float],
) -> tuple[bool, List[str]]:
    """
    Check whether all scores meet their thresholds.

    Returns:
        (passed, failures) where failures lists which metrics failed.
    """
    failures = []
    for metric, threshold in thresholds.items():
        score = scores.get(metric, 0.0)
        if score < threshold:
            failures.append(
                f"  ❌ {metric}: {score:.3f} < threshold {threshold:.3f}"
            )

    return len(failures) == 0, failures
