# ============================================================
# src/evals/evaluator.py - RAG Evaluation with RAGAS
#
# Learning Note:
#   HOW DO YOU KNOW IF YOUR RAG IS GOOD?
#   You need quantitative metrics. RAGAS provides 4 key metrics:
#
#   1. FAITHFULNESS (0-1):
#      "Is the answer factually consistent with the context?"
#      Low score = LLM hallucinated facts not in the documents.
#      Calculation: # claims in answer supported by context / total claims
#
#   2. ANSWER RELEVANCY (0-1):
#      "Does the answer actually address the question?"
#      Low score = LLM gave a correct but off-topic answer.
#      Calculation: cosine similarity between generated question
#                   from answer vs original question.
#
#   3. CONTEXT PRECISION (0-1):
#      "Are the retrieved chunks actually relevant to the question?"
#      Low score = retriever bringing in too much noisy data.
#      Calculation: relevant chunks in top-K / total chunks in top-K
#
#   4. CONTEXT RECALL (0-1):
#      "Did we retrieve all the information needed to answer?"
#      Low score = retriever missing important chunks.
#      Calculation: ground truth statements found in context / total statements
#
#   These metrics together tell you WHERE your RAG is failing.
# ============================================================

import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.config import settings
from src.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalMetrics:
    """Container for RAG evaluation scores."""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    # Computed average across all metrics
    overall_score: float = 0.0
    # Whether evaluation actually ran (False if sampled out or failed)
    evaluated: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "overall_score": round(self.overall_score, 3),
            "evaluated": self.evaluated,
            "error": self.error,
        }


class RAGEvaluator:
    """
    Inline RAG evaluation using RAGAS.
    Runs after every response (or sampled subset) to track quality.

    Learning Note:
        RAGAS uses the LLM itself as a judge in some metrics (LLM-as-judge).
        This means evaluations also consume LLM tokens.
        Use eval_sample_rate < 1.0 in production to control cost.
        e.g. eval_sample_rate=0.1 means evaluate 10% of queries.
    """

    def __init__(self):
        self._ragas_available = self._check_ragas()

    def _check_ragas(self) -> bool:
        """Check if RAGAS is installed."""
        try:
            import ragas
            logger.info("ragas_available")
            return True
        except ImportError:
            logger.warning("ragas_not_installed", msg="pip install ragas to enable evaluations")
            return False

    def _should_evaluate(self) -> bool:
        """
        Decide whether to evaluate this request based on sample rate.
        Learning Note: This is a stochastic sampling approach.
        random.random() returns 0.0-1.0 uniformly.
        If sample_rate=0.1, roughly 10% of calls will evaluate.
        """
        if not settings.enable_inline_evals:
            return False
        return random.random() < settings.eval_sample_rate

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvalMetrics:
        """
        Run RAGAS evaluation on a single Q&A exchange.

        Args:
            question: The user's original question
            answer: The LLM's generated answer
            contexts: The document chunks used for generation
            ground_truth: Optional reference answer (for context_recall)

        Returns:
            EvalMetrics with scores for each metric
        """
        if not self._should_evaluate():
            return EvalMetrics(evaluated=False)

        if not self._ragas_available:
            return self._fallback_heuristic_eval(question, answer, contexts)

        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset

            # RAGAS expects a HuggingFace Dataset format
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }

            metrics_to_run = [faithfulness, answer_relevancy, context_precision]

            # context_recall requires ground_truth
            if ground_truth:
                data["ground_truth"] = [ground_truth]
                metrics_to_run.append(context_recall)

            dataset = Dataset.from_dict(data)
            result = evaluate(dataset, metrics=metrics_to_run)

            scores = result.to_pandas().iloc[0].to_dict()

            faith = float(scores.get("faithfulness", 0.0))
            relevancy = float(scores.get("answer_relevancy", 0.0))
            precision = float(scores.get("context_precision", 0.0))
            recall = float(scores.get("context_recall", 0.0)) if ground_truth else 0.0

            overall = (faith + relevancy + precision) / 3 if not ground_truth else (faith + relevancy + precision + recall) / 4

            metrics = EvalMetrics(
                faithfulness=faith,
                answer_relevancy=relevancy,
                context_precision=precision,
                context_recall=recall,
                overall_score=overall,
                evaluated=True,
            )

            logger.info(
                "ragas_evaluation_complete",
                faithfulness=round(faith, 3),
                answer_relevancy=round(relevancy, 3),
                context_precision=round(precision, 3),
                overall=round(overall, 3),
            )

            return metrics

        except Exception as exc:
            logger.error("ragas_evaluation_failed", error=str(exc))
            return EvalMetrics(evaluated=False, error=str(exc))

    def _fallback_heuristic_eval(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> EvalMetrics:
        """
        Simple heuristic evaluation when RAGAS is not available.
        Not as accurate as RAGAS but gives a rough quality signal.

        Learning Note:
            This is a simplified version. In production always use RAGAS.
            But heuristics are good for understanding what the metrics mean:
            - Faithfulness: how much of the answer appears in context
            - Relevancy: does the answer contain question keywords
        """
        if not answer or not contexts:
            return EvalMetrics(evaluated=False)

        full_context = " ".join(contexts).lower()
        answer_lower = answer.lower()
        question_lower = question.lower()

        # Heuristic faithfulness: phrase overlap
        words = answer_lower.split()
        phrases = [" ".join(words[i:i+3]) for i in range(max(0, len(words)-2))]
        if phrases:
            grounded = sum(1 for p in phrases if p in full_context)
            faithfulness = min(1.0, grounded / len(phrases) * 2)  # scale up
        else:
            faithfulness = 0.5

        # Heuristic relevancy: question keyword overlap in answer
        question_words = set(question_lower.split()) - {"what", "how", "why", "is", "the", "a", "an"}
        answer_words = set(answer_lower.split())
        if question_words:
            relevancy = len(question_words & answer_words) / len(question_words)
        else:
            relevancy = 0.5

        # Simple context precision: assume retrieval was decent
        context_precision = 0.7

        overall = (faithfulness + relevancy + context_precision) / 3

        metrics = EvalMetrics(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=context_precision,
            context_recall=0.0,
            overall_score=overall,
            evaluated=True,
        )

        logger.info("heuristic_evaluation_complete", overall=round(overall, 3))
        return metrics


# Singleton instance
rag_evaluator = RAGEvaluator()
