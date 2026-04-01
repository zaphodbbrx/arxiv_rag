"""
CLI-скрипт для загрузки и обработки научных статей с arXiv.

Примеры:
    # Поиск по запросу
    python scripts/ingest.py --query "retrieval augmented generation" --max 20

    # По категориям
    python scripts/ingest.py --category cs.CL --category cs.AI --max 50

    # С фильтром по дате
    python scripts/ingest.py --query "large language models" --date-from 2024-01-01 --date-to 2025-01-01

    # Только метаданные (без скачивания PDF)
    python scripts/ingest.py --query "RAG" --skip-download

    # Использовать только PyMuPDF (без Grobid)
    python scripts/ingest.py --query "transformer" --parser pymupdf --max 10

    # Принудительно перепарсить все
    python scripts/ingest.py --query "knowledge graph" --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

# Добавляем корень проекта в sys.path (для запуска как скрипта)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.arxiv_rag.config.settings import get_settings
from src.arxiv_rag.data_ingestion.pipeline import IngestionPipeline


def setup_logging(level: str) -> None:
    """Настраивает логирование для CLI."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    p = argparse.ArgumentParser(
        description="Download and process research papers from arXiv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --query "retrieval augmented generation" --max 20
  %(prog)s --category cs.CL --category cs.AI --max 50
  %(prog)s --query "LLM" --date-from 2024-06-01 --skip-download
        """,
    )

    # -- Поиск ---------------------------------------------------------------
    p.add_argument(
        "--query", "-q",
        type=str, default=None,
        help="Free-text search query (searches title + abstract).",
    )
    p.add_argument(
        "--category", "-c",
        type=str, action="append", dest="categories", default=None,
        help="arXiv category filter (e.g. cs.CL). Can be repeated.",
    )
    p.add_argument(
        "--max", "-n",
        type=int, default=None,
        help="Maximum number of papers to fetch.",
    )
    p.add_argument(
        "--date-from",
        type=str, default=None,
        help="Start date filter (YYYY-MM-DD).",
    )
    p.add_argument(
        "--date-to",
        type=str, default=None,
        help="End date filter (YYYY-MM-DD).",
    )

    # -- Обработка -----------------------------------------------------------
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip PDF download (use existing files).",
    )
    p.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip PDF parsing (metadata only).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-process already downloaded/parsed papers.",
    )
    p.add_argument(
        "--parser",
        type=str, choices=["auto", "grobid", "pymupdf"], default=None,
        help="PDF parser strategy: auto (default), grobid, or pymupdf.",
    )

    # -- Вывод ---------------------------------------------------------------
    p.add_argument(
        "--config",
        type=str, default=None,
        help="Path to YAML config file.",
    )
    p.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Output JSONL file path (default: data/processed/papers.jsonl).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it.",
    )

    return p.parse_args()


def main() -> None:
    """Точка входа."""
    args = parse_args()

    # Логирование
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Настройки
    settings = get_settings(config_path=args.config)

    # Переопределяем настройки из CLI
    if args.parser:
        settings.pdf_parser = args.parser
    if args.max:
        settings.arxiv_max_results = args.max

    # Проверяем, что хотя бы один критерий поиска задан
    if not args.query and not args.categories:
        logger.error(
            "Provide at least --query or --category. "
            "Example: python scripts/ingest.py --query 'RAG' --max 20"
        )
        sys.exit(1)

    # Парсим даты
    date_from = None
    date_to = None
    if args.date_from:
        try:
            date_from = datetime.strptime(args.date_from, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid --date-from format. Expected YYYY-MM-DD.")
            sys.exit(1)
    if args.date_to:
        try:
            date_to = datetime.strptime(args.date_to, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid --date-to format. Expected YYYY-MM-DD.")
            sys.exit(1)

    # Dry run
    if args.dry_run:
        logger.info("[DRY RUN] Would run ingestion with:")
        logger.info("  Query:      %s", args.query or "(none)")
        logger.info("  Categories: %s", args.categories or settings.arxiv_categories)
        logger.info("  Max:        %d", settings.arxiv_max_results)
        logger.info("  Date:       %s → %s", args.date_from, args.date_to)
        logger.info("  Parser:     %s", settings.pdf_parser)
        logger.info("  Output:     %s", settings.processed_dir / "papers.jsonl")
        return

    # Запуск пайплайна
    pipeline = IngestionPipeline(
        settings=settings,
        output_path=Path(args.output) if args.output else None,
    )

    papers = pipeline.run(
        query=args.query,
        categories=args.categories,
        max_results=args.max,
        date_from=date_from,
        date_to=date_to,
        skip_download=args.skip_download,
        skip_parse=args.skip_parse,
        force_reparse=args.force,
    )

    # Итоги
    parsed = [p for p in papers if p.is_parsed]
    failed = [p for p in papers if p.parse_status == "failed"]

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Total fetched:    {len(papers)}")
    print(f"  Successfully parsed: {len(parsed)}")
    print(f"  Failed:           {len(failed)}")

    if parsed:
        sections_count = sum(len(p.sections) for p in parsed)
        words_count = sum(
            len(p.full_text.split()) for p in parsed if p.full_text
        )
        print(f"  Total sections:   {sections_count}")
        print(f"  Total words:      {words_count:,}")

    if failed:
        print("\n  Failed papers:")
        for p in failed[:10]:
            print(f"    - {p.arxiv_id}: {p.parse_error}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")

    print("=" * 60)


if __name__ == "__main__":
    main()