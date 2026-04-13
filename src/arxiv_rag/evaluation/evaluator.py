"""
RAG Evaluator — оценка качества RAG-системы с помощью RAGAS.

Пайплайн:
    1. Загрузить QA пары из бенчмарка
    2. Для каждого вопроса прогнать через ArxivQueryEngine
    3. Собрать: question, answer, contexts (source nodes), ground_truth
    4. Вычислить RAGAS метрики: faithfulness, answer_relevancy, etc.
    5. Сохранить результаты в JSON

Использование:
    from src.evaluation.evaluator import RAGEvaluator

    evaluator = RAGEvaluator(index, model="qwen2.5:14b", top_k=5)
    results = evaluator.run(qa_pairs)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.arxiv_rag.evaluation.dataset_loader import QAPair
from src.arxiv_rag.rag.query_engine import ArxivQueryEngine, QueryResult

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    """Результат обработки одного вопроса."""

    num: int
    paper_title: str
    category: str
    question: str
    ground_truth: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None

    # RAGAS метрики (заполняются после evaluate)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


@dataclass
class BenchmarkReport:
    """Итоговый отчёт бенчмарка."""

    total: int = 0
    successful: int = 0
    failed: int = 0
    avg_latency_ms: float = 0.0

    # Средние метрики
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    # По категориям
    category_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    # Детальные результаты
    results: list[QuestionResult] = field(default_factory=list)


class RAGEvaluator:
    """Оценка RAG-системы по бенчмарк-датасету."""

    def __init__(
        self,
        index,
        *,
        model: str = "qwen2.5:14b",
        top_k: int = 5,
        temperature: float = 0.1,
        request_timeout: float = 180.0,
    ) -> None:
        self.engine = ArxivQueryEngine(
            index,
            model=model,
            top_k=top_k,
            temperature=temperature,
            request_timeout=request_timeout,
        )
        self.model = model
        self.top_k = top_k

    def run(
        self,
        qa_pairs: list[QAPair],
        *,
        max_questions: int | None = None,
        categories: list[str] | None = None,
    ) -> BenchmarkReport:
        """Запустить оценку на датасете.

        Args:
            qa_pairs: Список QA пар.
            max_questions: Ограничить число вопросов (для быстрого теста).
            categories: Фильтр по категориям (None = все).

        Returns:
            BenchmarkReport с результатами.
        """
        # Фильтрация
        filtered = qa_pairs
        if categories:
            filtered = [q for q in filtered if q.category in categories]
        if max_questions:
            filtered = filtered[:max_questions]

        logger.info(
            "Running benchmark: %d questions (model=%s, top_k=%d)",
            len(filtered), self.model, self.top_k,
        )

        # ── Шаг 1: Прогнать RAG ────────────────────────────────────
        results: list[QuestionResult] = []

        for i, qa in enumerate(filtered):
            logger.info(
                "[%d/%d] Q%d: %s", i + 1, len(filtered), qa.num, qa.question[:80]
            )

            qr = self._run_single(qa)
            results.append(qr)

        successful = [r for r in results if not r.error]
        failed = [r for r in results if r.error]

        logger.info(
            "RAG pass complete: %d success, %d failed", len(successful), len(failed)
        )

        if not successful:
            logger.warning("No successful queries — skipping RAGAS evaluation")
            return self._build_report(results)

        # ── Шаг 2: RAGAS ────────────────────────────────────────────
        logger.info("Computing RAGAS metrics...")
        ragas_results = self._compute_ragas(successful)

        # Записать метрики в каждый результат
        for qr in successful:
            if qr.question in ragas_results:
                metrics = ragas_results[qr.question]
                qr.faithfulness = metrics.get("faithfulness")
                qr.answer_relevancy = metrics.get("answer_relevancy")
                qr.context_precision = metrics.get("context_precision")
                qr.context_recall = metrics.get("context_recall")

        return self._build_report(results)

    def _run_single(self, qa: QAPair) -> QuestionResult:
        """Прогнать один вопрос через RAG."""
        try:
            t0 = time.monotonic()
            result: QueryResult = self.engine.query(qa.question)
            latency = (time.monotonic() - t0) * 1000

            # Собрать контексты из source_nodes
            contexts = []
            # Получаем source_nodes через внутренний engine
            response_obj = self.engine.engine.query(qa.question)
            if hasattr(response_obj, "source_nodes") and response_obj.source_nodes:
                contexts = [
                    node.node.content
                    if hasattr(node, "node") else str(node)
                    for node in response_obj.source_nodes
                ]

            return QuestionResult(
                num=qa.num,
                paper_title=qa.paper_title,
                category=qa.benchmark_category,
                question=qa.question,
                ground_truth=qa.ground_truth,
                answer=result.answer,
                contexts=contexts,
                latency_ms=latency,
            )

        except Exception as e:
            logger.error("Failed on Q%d: %s", qa.num, e)
            return QuestionResult(
                num=qa.num,
                paper_title=qa.paper_title,
                category=qa.benchmark_category,
                question=qa.question,
                ground_truth=qa.ground_truth,
                answer="",
                error=str(e),
            )

    def _compute_ragas(
        self, results: list[QuestionResult]
    ) -> dict[str, dict[str, float]]:
        """Вычислить RAGAS метрики."""
        try:
            data = {
                "question": [r.question for r in results],
                "answer": [r.answer for r in results],
                "contexts": [r.contexts for r in results],
                "ground_truth": [r.ground_truth for r in results],
            }

            ds = Dataset.from_dict(data)

            eval_result = evaluate(
                ds,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )

            # Маппинг: вопрос → метрики
            metric_map = {}
            for i, question in enumerate(data["question"]):
                metric_map[question] = {
                    "faithfulness": self._safe_float(eval_result["faithfulness"][i])
                    if "faithfulness" in eval_result else None,
                    "answer_relevancy": self._safe_float(eval_result["answer_relevancy"][i])
                    if "answer_relevancy" in eval_result else None,
                    "context_precision": self._safe_float(eval_result["context_precision"][i])
                    if "context_precision" in eval_result else None,
                    "context_recall": self._safe_float(eval_result["context_recall"][i])
                    if "context_recall" in eval_result else None,
                }

            return metric_map

        except Exception as e:
            logger.error("RAGAS evaluation failed: %s", e)
            return {}

    def _build_report(self, results: list[QuestionResult]) -> BenchmarkReport:
        """Собрать итоговый отчёт."""
        successful = [r for r in results if not r.error]
        failed = [r for r in results if r.error]

        report = BenchmarkReport(
            total=len(results),
            successful=len(successful),
            failed=len(failed),
            avg_latency_ms=(
                sum(r.latency_ms for r in successful) / len(successful)
                if successful else 0.0
            ),
        )

        if successful:
            def avg(key: str) -> float | None:
                vals = [getattr(r, key) for r in successful if getattr(r, key) is not None]
                return sum(vals) / len(vals) if vals else None

            report.faithfulness = avg("faithfulness")
            report.answer_relevancy = avg("answer_relevancy")
            report.context_precision = avg("context_precision")
            report.context_recall = avg("context_recall")

            # По категориям
            from collections import defaultdict
            cat_groups: dict[str, list[QuestionResult]] = defaultdict(list)
            for r in successful:
                cat_groups[r.category].append(r)

            for cat, cat_results in cat_groups.items():
                def cat_avg(key: str) -> float | None:
                    vals = [getattr(r, key) for r in cat_results if getattr(r, key) is not None]
                    return sum(vals) / len(vals) if vals else None

                report.category_scores[cat] = {
                    "count": len(cat_results),
                    "faithfulness": cat_avg("faithfulness"),
                    "answer_relevancy": cat_avg("answer_relevancy"),
                    "context_precision": cat_avg("context_precision"),
                    "context_recall": cat_avg("context_recall"),
                }

        report.results = results
        return report

    @staticmethod
    def _safe_float(val: Any) -> float | None:
        """Безопасно конвертировать в float."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return None


def save_report(report: BenchmarkReport, path: str | Path) -> Path:
    """Сохранить отчёт в JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total": report.total,
        "successful": report.successful,
        "failed": report.failed,
        "avg_latency_ms": round(report.avg_latency_ms, 1),
        "metrics": {
            "faithfulness": _r(report.faithfulness),
            "answer_relevancy": _r(report.answer_relevancy),
            "context_precision": _r(report.context_precision),
            "context_recall": _r(report.context_recall),
        },
        "category_scores": {
            cat: {
                k: _r(v) if isinstance(v, float) else v
                for k, v in scores.items()
            }
            for cat, scores in report.category_scores.items()
        },
        "results": [
            {
                "num": r.num,
                "paper_title": r.paper_title,
                "category": r.category,
                "question": r.question,
                "ground_truth": r.ground_truth[:200],
                "answer": r.answer[:200],
                "latency_ms": round(r.latency_ms, 1),
                "faithfulness": _r(r.faithfulness),
                "answer_relevancy": _r(r.answer_relevancy),
                "context_precision": _r(r.context_precision),
                "context_recall": _r(r.context_recall),
                "error": r.error,
            }
            for r in report.results
        ],
    }

    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Report saved to %s", path)
    return path


def _r(val: float | None) -> float | None:
    """Round to 4 decimal places, keep None as None."""
    return round(val, 4) if val is not None else None