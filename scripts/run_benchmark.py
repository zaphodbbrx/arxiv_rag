#!/usr/bin/env python3
"""
ArXiv RAG Benchmark — оценка качества RAG-системы по бенчмарк-датасету.

Использование:
    # Полный запуск
    uv run python scripts/run_benchmark.py benchmarks/3d_object_detection.xlsx

    # Быстрый тест (5 вопросов)
    uv run python scripts/run_benchmark.py benchmarks/3d_object_detection.xlsx --max 5

    # Только определённая категория
    uv run python scripts/run_benchmark.py benchmarks/3d_object_detection.xlsx --category Point-based

    # Сохранить в конкретный файл
    uv run python scripts/run_benchmark.py benchmarks/3d_object_detection.xlsx --output results/bench.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.arxiv_rag.data_ingestion.models import ArxivPaper
from src.arxiv_rag.evaluation.dataset_loader import load_benchmark
from src.arxiv_rag.evaluation.evaluator import RAGEvaluator, save_report
from src.arxiv_rag.rag.index_builder import build_index, load_index
from src.arxiv_rag.utils.io_utils import load_jsonl


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ArXiv RAG Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("benchmark", help="Path to benchmark Excel file (.xlsx)")
    parser.add_argument("--papers", default="src/data/processed/papers.jsonl",
                        help="Path to papers.jsonl")
    parser.add_argument("--persist-dir", default="src/data/indexes/llama",
                        help="Path to FAISS index")
    parser.add_argument("--model", default="qwen2.5:14b",
                        help="Ollama model")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max", type=int, default=None,
                        help="Max questions to evaluate")
    parser.add_argument("--category", action="append", default=None,
                        help="Filter by category (can be repeated)")
    parser.add_argument("--output", default="src/data/evaluation/benchmark_results.json",
                        help="Output JSON path")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild index")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # ── Setup ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ArXiv RAG Benchmark")
    print("=" * 60)

    t0 = time.monotonic()

    # ── Step 1: Load benchmark ─────────────────────────────────────
    print(f"\n[1/3] Loading benchmark: {args.benchmark}")
    records = load_jsonl(args.papers)
    papers = [ArxivPaper.from_dict(r) for r in records]

    # Передаём в загрузчик
    qa_pairs = load_benchmark(args.benchmark, papers=papers)

    unmatched = [q for q in qa_pairs if q.paper is None]
    if unmatched:
        print(f"  WARNING: {len(unmatched)} QA pairs unmatched (paper not in index)")
    print(f"  Loaded {len(qa_pairs)} QA pairs")

    if args.category:
        qa_pairs = [q for q in qa_pairs if q.category in args.category]
        print(f"  Filtered to {len(qa_pairs)} pairs (categories: {args.category})")

    if args.max:
        qa_pairs = qa_pairs[:args.max]
        print(f"  Limited to {len(qa_pairs)} pairs")

    # ── Step 2: Load index ─────────────────────────────────────────
    persist_dir = Path(args.persist_dir)
    index_exists = (persist_dir / "docstore.json").exists()

    if args.rebuild or not index_exists:
        print(f"\n[2/3] Building index...")
        papers_path = Path(args.papers)
        if not papers_path.exists():
            print(f"ERROR: {papers_path} not found")
            sys.exit(1)
        records = load_jsonl(papers_path)
        papers = [ArxivPaper.from_dict(r) for r in records]
        index = build_index(
            papers, persist_dir=persist_dir,
            embedding_model=args.embedding_model,
        )
    else:
        print(f"\n[2/3] Loading index from {persist_dir}...")
        index = load_index(
            persist_dir=persist_dir,
            embedding_model=args.embedding_model,
        )

    # ── Step 3: Run benchmark ──────────────────────────────────────
    print(f"\n[3/3] Running benchmark (model={args.model}, top_k={args.top_k})...")
    print(f"  Make sure Ollama is running: ollama serve")
    print()

    evaluator = RAGEvaluator(
        index,
        model=args.model,
        top_k=args.top_k,
    )

    report = evaluator.run(qa_pairs)

    # ── Results ────────────────────────────────────────────────────
    setup_ms = (time.monotonic() - t0) * 1000

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Total:    {report.total}")
    print(f"  Success:  {report.successful}")
    print(f"  Failed:   {report.failed}")
    print(f"  Avg time: {report.avg_latency_ms:.0f} ms/query")
    print(f"  Total:    {setup_ms / 1000:.1f}s")
    print()

    if report.faithfulness is not None:
        print("  RAGAS Metrics:")
        print(f"    Faithfulness:      {report.faithfulness:.4f}")
        print(f"    Answer Relevancy:  {report.answer_relevancy:.4f}")
        print(f"    Context Precision:  {report.context_precision:.4f}")
        print(f"    Context Recall:    {report.context_recall:.4f}")
        print()

    if report.category_scores:
        print("  By Category:")
        print(f"    {'Category':<20} {'N':>4} {'Faith':>6} {'Rel':>6} {'Prec':>6} {'Rec':>6}")
        print(f"    {'-'*56}")
        for cat, scores in report.category_scores.items():
            print(
                f"    {cat:<20} "
                f"{scores['count']:>4} "
                f"{scores['faithfulness'] or 0:>6.3f} "
                f"{scores['answer_relevancy'] or 0:>6.3f} "
                f"{scores['context_precision'] or 0:>6.3f} "
                f"{scores['context_recall'] or 0:>6.3f}"
            )

    # ── Save ───────────────────────────────────────────────────────
    output_path = Path(args.output)
    save_report(report, output_path)
    print(f"\n  Report saved: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()