"""
Dataset Loader — чтение бенчмарк-датасета из Excel-файла.

Поддерживает формат:
    Sheet "RAG Benchmark QA":
        Col B (#), C (Paper Title), D (Authors), E (Venue),
        F (Category), G (Question), H (Answer)

Использование:
    from src.evaluation.dataset_loader import load_benchmark

    qa_pairs = load_benchmark("benchmarks/3d_object_detection.xlsx")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import openpyxl

from src.arxiv_rag.data_ingestion.models import ArxivPaper

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """Один вопрос-ответ из бенчмарк-датасета."""

    num: int
    question: str
    ground_truth: str
    paper: ArxivPaper | None = None  # ← ссылка на статью из papers.jsonl
    venue: str = ""
    benchmark_category: str = ""     # категория из бенчмарка (Point-based, Voxel-based...)

    # Доступ к полям статьи через paper (если замэтчена)
    @property
    def paper_title(self) -> str:
        return self.paper.title if self.paper else "(unmatched)"

    @property
    def arxiv_id(self) -> str | None:
        return self.paper.arxiv_id if self.paper else None

def load_benchmark(
    path: str | Path,
    papers: list[ArxivPaper] | None = None,
) -> list[QAPair]:
    """Загрузить QA пары из Excel-файла бенчмарка.

    Читает лист 'RAG Benchmark QA'. Данные начинаются с row 5.
    Пропускает строки, где вопрос или ответ пустые.

    Args:
        path: Путь к .xlsx файлу.

    Returns:
        Список QAPair.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    wb = openpyxl.load_workbook(path, read_only=True)
    sheet_name = "RAG Benchmark QA"
    if sheet_name not in wb.sheetnames:
        raise ValueError(f"Sheet '{sheet_name}' not found. Available: {wb.sheetnames}")

    ws = wb[sheet_name]

    papers_by_title: dict[str, ArxivPaper] = {}
    if papers:
        for p in papers:
            key = p.title.strip().lower()
            papers_by_title[key] = p

    pairs: list[QAPair] = []
    for row in ws.iter_rows(min_row=6, max_col=8, values_only=True):

        title = str(row[2] or "").strip()
        matched_paper = papers_by_title.get(title.lower())

        pairs.append(QAPair(
            num=int(row[1]) if row[1] else len(pairs) + 1,
            question=str(row[6]).strip(),
            ground_truth=str(row[7]).strip(),
            paper=matched_paper,
            venue=str(row[4] or "").strip(),
            benchmark_category=str(row[5] or "").strip(),
        ))

    wb.close()

    logger.info("Loaded %d QA pairs from %s", len(pairs), path)
    return pairs