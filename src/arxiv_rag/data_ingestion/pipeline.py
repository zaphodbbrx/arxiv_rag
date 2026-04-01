"""
Ingestion Pipeline — оркестрация полного процесса загрузки и обработки статей.

Пайплайн:
  1. Поиск статей на arXiv (query / categories)
  2. Скачивание PDF
  3. Парсинг PDF (Grobid / PyMuPDF)
  4. Сохранение результатов в JSONL

Использование:
    pipeline = IngestionPipeline()
    papers = pipeline.run(query="retrieval augmented generation", max_results=50)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.arxiv_rag.config.settings import Settings, get_settings
from src.arxiv_rag.data_ingestion.arxiv_client import ArxivClient
from src.arxiv_rag.data_ingestion.models import ArxivPaper
from src.arxiv_rag.data_ingestion.pdf_parser import PDFParser
from src.arxiv_rag.utils.io_utils import load_jsonl, save_jsonl

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Полный пайплайн загрузки и обработки научных статей с arXiv.

    Поддерживает инкрементальную загрузку: уже обработанные статьи
    (по arxiv_id) пропускаются при повторном запуске.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        output_path: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.output_path = output_path or (self.settings.processed_dir / "papers.jsonl")

        self.client = ArxivClient()
        self.parser = PDFParser(
            grobid_url=self.settings.grobid_url,
            strategy=self.settings.pdf_parser,
        )

        # Множество уже обработанных arxiv_id (для инкрементальной загрузки)
        self._existing_ids: set[str] = set()
        if self.output_path.exists():
            for record in load_jsonl(self.output_path):
                self._existing_ids.add(record.get("arxiv_id", ""))

        if self._existing_ids:
            logger.info(
                "Found %d already processed papers in %s",
                len(self._existing_ids),
                self.output_path,
            )

        # Проверяем статус парсеров
        status = self.parser.parse_status()
        for name, available in status.items():
            logger.info("Parser '%s': %s", name, "available" if available else "not available")

    def run(
        self,
        *,
        query: str | None = None,
        categories: list[str] | None = None,
        max_results: int | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        skip_download: bool = False,
        skip_parse: bool = False,
        force_reparse: bool = False,
    ) -> list[ArxivPaper]:
        """
        Запускает полный ingestion pipeline.

        Args:
            query:          Текстовый запрос для поиска на arXiv.
            categories:     Список категорий (например ["cs.CL", "cs.AI"]).
            max_results:    Максимум статей для загрузки.
            date_from:      Начальная дата публикации.
            date_to:        Конечная дата публикации.
            skip_download:  Пропустить скачивание PDF (использовать существующие).
            skip_parse:     Пропустить парсинг PDF (только метаданные).
            force_reparse:  Перепарсить уже обработанные статьи.

        Returns:
            Список обработанных ArxivPaper.
        """
        max_results = max_results or self.settings.arxiv_max_results
        date_from_str = date_from.strftime("%Y-%m-%d") if date_from else None
        date_to_str = date_to.strftime("%Y-%m-%d") if date_to else None

        logger.info("=" * 60)
        logger.info("Ingestion Pipeline")
        logger.info("  Query:       %s", query or "(categories only)")
        logger.info("  Categories:  %s", categories or self.settings.arxiv_categories)
        logger.info("  Max results: %d", max_results)
        logger.info("  Parser:      %s", self.settings.pdf_parser)
        logger.info("  Output:      %s", self.output_path)
        logger.info("=" * 60)

        # ------------------------------------------------------------------
        #  Шаг 1: Поиск на arXiv
        # ------------------------------------------------------------------
        all_papers: list[ArxivPaper] = []

        if query:
            all_papers.extend(
                self.client.search_by_query(
                    query=query,
                    max_results=max_results,
                    date_from=date_from_str,
                    date_to=date_to_str,
                )
            )

        if categories:
            # max_results делим поровну между категориями
            per_cat = max(1, max_results // len(categories))
            for cat in categories:
                all_papers.extend(
                    self.client.search_by_category(
                        category=cat,
                        max_results=per_cat,
                        date_from=date_from_str,
                        date_to=date_to_str,
                    )
                )

        # Убираем дубликаты по arxiv_id
        seen_ids: set[str] = set()
        unique_papers: list[ArxivPaper] = []
        for paper in all_papers:
            if paper.arxiv_id not in seen_ids:
                seen_ids.add(paper.arxiv_id)
                unique_papers.append(paper)

        # Инкрементальная загрузка — пропускаем уже обработанные
        papers: list[ArxivPaper] = []
        skipped_count = 0
        for paper in unique_papers:
            if paper.arxiv_id in self._existing_ids and not force_reparse:
                skipped_count += 1
                continue
            papers.append(paper)

        logger.info(
            "Search complete: %d new, %d skipped (already processed), %d total found",
            len(papers), skipped_count, len(unique_papers),
        )

        if not papers:
            logger.info("No new papers to process.")
            return self._load_existing_papers()

        # ------------------------------------------------------------------
        #  Шаг 2: Скачивание PDF
        # ------------------------------------------------------------------
        if not skip_download:
            logger.info("Step 2: Downloading PDFs...")
            downloaded_paths = self.client.download_pdfs(
                papers,
                output_dir=self.settings.raw_dir,
            )

            # Фильтруем — оставляем только те, у которых скачался PDF
            papers = [p for p in papers if p.pdf_path is not None]

            logger.info(
                "Downloaded: %d/%d PDFs",
                len(downloaded_paths), len(papers) + (len(papers) - len(downloaded_paths)),
            )

            if len(papers) < len(downloaded_paths):
                logger.warning(
                    "%d papers without PDF were dropped",
                    len(downloaded_paths) - len(papers),
                )

        # ------------------------------------------------------------------
        #  Шаг 3: Парсинг PDF
        # ------------------------------------------------------------------
        if not skip_parse and papers:
            logger.info("Step 3: Parsing PDFs...")
            parse_success = 0
            parse_failed = 0

            for i, paper in enumerate(papers, start=1):
                if not paper.pdf_path:
                    paper.parse_status = "failed"
                    paper.parse_error = "No PDF file"
                    parse_failed += 1
                    continue

                try:
                    parsed_doc = self.parser.parse(paper.pdf_path)
                    paper.apply_parsed_document(parsed_doc)
                    parse_success += 1
                except Exception as e:
                    paper.parse_status = "failed"
                    paper.parse_error = str(e)
                    parse_failed += 1
                    logger.error(
                        "Failed to parse %s: %s",
                        paper.arxiv_id, e,
                    )

                if i % 10 == 0:
                    logger.info(
                        "Parsed %d/%d (success=%d, failed=%d)",
                        i, len(papers), parse_success, parse_failed,
                    )

            logger.info(
                "Parsing complete: %d success, %d failed out of %d",
                parse_success, parse_failed, len(papers),
            )

        # ------------------------------------------------------------------
        #  Шаг 4: Сохранение
        # ------------------------------------------------------------------
        self._save_results(papers)
        logger.info(
            "Saved %d papers to %s",
            len(papers), self.output_path,
        )

        return papers

    # ------------------------------------------------------------------
    #  Вспомогательные методы
    # ------------------------------------------------------------------

    def _save_results(self, papers: list[ArxivPaper]) -> None:
        """Сохраняет результаты в JSONL. Если файл существует — добавляет новые."""
        records = [p.to_dict() for p in papers if p.parse_status == "parsed"]
        if not records:
            logger.warning("No successfully parsed papers to save.")
            return

        # Добавляем к существующим или создаём новый файл
        existing_records: list[dict] = []
        if self.output_path.exists():
            existing_records = load_jsonl(self.output_path)

        # Удаляем дубликаты по arxiv_id
        existing_ids = {r["arxiv_id"] for r in existing_records}
        new_records = [r for r in records if r["arxiv_id"] not in existing_ids]

        all_records = existing_records + new_records
        save_jsonl(all_records, self.output_path)

        logger.info(
            "Saved: %d new records, %d total in %s",
            len(new_records), len(all_records), self.output_path,
        )

    def _load_existing_papers(self) -> list[ArxivPaper]:
        """Загружает ранее сохранённые статьи из JSONL."""
        if not self.output_path.exists():
            return []
        records = load_jsonl(self.output_path)
        return [ArxivPaper.from_dict(r) for r in records]