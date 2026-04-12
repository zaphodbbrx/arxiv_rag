"""
Парсинг PDF научных статей.

Две стратегии:
  1. GrobidParser (основная) — через GROBID REST API (Docker).
     Извлекает секции, формулы, таблицы, метаданные.
     Золотой стандарт для академических PDF.

  2. PyMuPDFParser (фоллбэк) — быстрый локальный парсинг.
     Извлекает текст по страницам с эвристическим определением секций.
     Работает без Docker, но хуже с математикой и таблицами.

Использование:
    parser = PDFParser(grobid_url="http://localhost:8070")
    doc = parser.parse(pdf_path)
    # doc: ParsedDocument с секциями, таблицами, формулами
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pymupdf  # PyMuPDF

from src.arxiv_rag.data_ingestion.models import ParsedDocument, Section
from src.arxiv_rag.utils.text_processing import SectionType, clean_latex_math, normalize_whitespace

logger = logging.getLogger(__name__)


# ============================================================================
#  Базовый класс
# ============================================================================

class BasePDFParser(ABC):
    """Абстрактный парсер PDF."""

    @abstractmethod
    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Парсит PDF-файл и возвращает структурированный документ."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Проверяет, доступен ли парсер (сервис запущен, зависимости установлены)."""
        ...


# ============================================================================
#  Grobid Parser
# ============================================================================

@dataclass
class GrobidParser(BasePDFParser):
    """
    Парсер через GROBID REST API.

    GROBID — индустриальный standard для парсинга академических PDF.
    Извлекает структурированный TEI XML с разметкой секций,
    формул, таблиц, авторов, DOI и т.д.

    Запуск:
        docker run -d --init --name grobid -p 8070:8070 lfoppiano/grobid:0.8.1
    """

    grobid_url: str = "http://localhost:8070"
    timeout: float = 120.0  # секунд (Grobid бывает медленным на больших PDF)
    _is_available: bool | None = field(default=None, repr=False)

    def is_available(self) -> bool:
        """Проверяет, что GROBID сервис доступен."""
        if self._is_available is not None:
            return self._is_available
        try:
            resp = httpx.get(
                f"{self.grobid_url}/api/isalive",
                timeout=10.0,
            )
            self._is_available = resp.status_code == 200
        except httpx.ConnectError:
            self._is_available = False
            logger.warning(
                "GROBID is not available at %s. "
                "Start it with: docker run -d --init --name grobid "
                "-p 8070:8070 lfoppiano/grobid:0.8.1",
                self.grobid_url,
            )
        except Exception as e:
            self._is_available = False
            logger.warning("GROBID health check failed: %s", e)
        return self._is_available

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Парсит PDF через GROBID API и конвертирует TEI XML в ParsedDocument."""
        if not self.is_available():
            raise RuntimeError(
                f"GROBID not available at {self.grobid_url}. "
                "Cannot parse PDF."
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Parsing %s with GROBID...", pdf_path.name)

        # Отправляем PDF в GROBID
        with open(pdf_path, "rb") as f:
            resp = httpx.post(
                f"{self.grobid_url}/api/processFulltextDocument",
                files={"input": (pdf_path.name, f, "application/pdf")},
                data={
                    "consolidateHeader": "1",
                    "consolidateCitations": "0",
                    "consolidateSections": "1",
                },
                timeout=self.timeout,
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"GROBID returned status {resp.status_code} for {pdf_path.name}"
            )

        # Парсим TEI XML
        return self._parse_tei(resp.text)

    def _parse_tei(self, tei_xml: str) -> ParsedDocument:
        """Конвертирует TEI XML от GROBID в ParsedDocument."""
        doc = ParsedDocument()

        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as e:
            logger.error("Failed to parse TEI XML: %s", e)
            return doc

        # XML namespace GROBID
        ns = {
            "tei": "http://www.tei-c.org/ns/1.0",
            "w": "http://www.tei-c.org/ns/1.0",
        }

        # -- Извлекаем секции текста ----------------------------------------
        body = root.find(".//tei:body", ns)
        if body is None:
            logger.warning("No <body> found in TEI XML")
            return doc

        all_text_parts: list[str] = []

        for div in body.findall(".//tei:div", ns):
            heading_el = div.find("tei:head", ns)
            heading = ""
            if heading_el is not None and heading_el.text:
                heading = heading_el.text.strip()

            section_type = SectionType.from_heading(heading)

            # Собираем текст из всех <p> внутри div
            paragraphs: list[str] = []
            for p in div.findall(".//tei:p", ns):
                p_text = self._extract_text_from_element(p)
                if p_text.strip():
                    paragraphs.append(p_text.strip())

            content = "\n\n".join(paragraphs)

            if content.strip():
                section = Section(
                    heading=heading,
                    content=content,
                    section_type=section_type.name,
                )
                doc.sections.append(section)
                all_text_parts.append(content)

        doc.full_text = "\n\n".join(all_text_parts)

        # -- Извлекаем формулы (из <formula> элементов) ---------------------
        for formula_el in root.findall(".//tei:formula", ns):
            if formula_el.text and formula_el.text.strip():
                formula_text = formula_el.text.strip()
                if formula_text.startswith("$$") or formula_text.startswith("\\"):
                    doc.equations.append(formula_text)

        logger.info(
            "GROBID parsed: %d sections, %d equations, %d words",
            len(doc.sections),
            len(doc.equations),
            doc.word_count,
        )

        return doc

    @staticmethod
    def _extract_text_from_element(element: ET.Element) -> str:
        """
        Рекурсивно извлекает текст из XML-элемента,
        включая вложенные теги (горизонталы, формулы и т.д.).
        """
        if element.text:
            text = element.text
        else:
            text = ""

        for child in element:
            child_text = GrobidParser._extract_text_from_element(child)
            text += child_text
            if child.tail:
                text += child.tail

        return text


# ============================================================================
#  PyMuPDF Parser (фоллбэк)
# ============================================================================

@dataclass
class PyMuPDFParser(BasePDFParser):
    """
    Локальный парсер PDF через PyMuPDF (fitz).

    Не требует Docker, работает быстро.
    Эвристически определяет секции по заголовкам в верхнем регистре
    или по определённым паттернам.
    """

    # Паттерны для определения заголовков секций
    # Захватывает строки, которые выглядят как заголовки (заглавные / пронумерованные)
    section_pattern: re.Pattern = re.compile(
        r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Za-z\s,:\-–—]{5,80}$",
        re.MULTILINE,
    )

    def is_available(self) -> bool:
        """PyMuPDF всегда доступен (локальная библиотека)."""
        return True

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Парсит PDF через PyMuPDF с эвристическим разделением на секции."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Parsing %s with PyMuPDF...", pdf_path.name)
        doc = pymupdf.open(str(pdf_path))
        parsed = ParsedDocument()

        all_text_parts: list[str] = []
        current_heading = "Introduction"
        current_paragraphs: list[str] = []
        current_pages: list[int] = []

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            for block in blocks:
                # block[6] = block type (0=text, 1=image)
                if block[6] != 0:
                    continue

                text = block[4].strip()
                if not text:
                    continue

                # Проверяем, является ли блок заголовком
                if self._is_heading(text, page.rect, block):
                    # Сохраняем предыдущую секцию
                    if current_paragraphs:
                        content = "\n\n".join(current_paragraphs)
                        section_type = SectionType.from_heading(current_heading)
                        section = Section(
                            heading=current_heading,
                            content=content,
                            page_numbers=list(set(current_pages)),
                            section_type=section_type.name,
                        )
                        parsed.sections.append(section)
                        all_text_parts.append(content)

                    current_heading = text
                    current_paragraphs = []
                    current_pages = [page_num]
                else:
                    # Очищаем текст (убираем LaTeX-мусор)
                    cleaned = clean_latex_math(text)
                    cleaned = normalize_whitespace(cleaned)
                    if cleaned:
                        current_paragraphs.append(cleaned)
                        current_pages.append(page_num)

        # Не забываем последнюю секцию
        if current_paragraphs:
            content = "\n\n".join(current_paragraphs)
            section_type = SectionType.from_heading(current_heading)
            section = Section(
                heading=current_heading,
                content=content,
                page_numbers=list(set(current_pages)),
                section_type=section_type.name,
            )
            parsed.sections.append(section)
            all_text_parts.append(content)

        parsed.full_text = "\n\n".join(all_text_parts)

        # Фильтруем пустые секции
        parsed.sections = [s for s in parsed.sections if not s.is_empty()]

        doc.close()

        logger.info(
            "PyMuPDF parsed: %d sections, %d words",
            len(parsed.sections),
            parsed.word_count,
        )

        return parsed

    def _is_heading(
        self,
        text: str,
        page_rect: pymupdf.Rect,
        block: tuple,
    ) -> bool:
        """
        Эвристика для определения заголовка секции.

        Заголовок если:
          - Текст короткий (< 100 символов)
          - Начинается с числа (нумерация секции)
          - Или весь текст в верхнем регистре
          - И размер шрифта >= 11pt (обычно заголовки крупнее)
        """
        # Проверяем длину
        if len(text) > 100 or len(text) < 3:
            return False

        # Проверяем номерацию: "1.", "2.3", "1 Introduction"
        if re.match(r"^\d+(?:\.\d+)?\s", text):
            return True

        # Проверяем, весь ли текст в верхнем регистре
        words = text.split()
        if len(words) >= 2 and all(w.isupper() for w in words if len(w) > 2):
            return True

        # Проверяем размер шрифта (если доступен)
        # block содержит: (x0, y0, x1, y1, text, block_no, block_type, ...)
        # Для более точного определения нужно использовать spans
        return False


# ============================================================================
#  Фабрика / Универсальный парсер
# ============================================================================

@dataclass
class PDFParser:
    """
    Универсальный парсер с автоматическим выбором стратегии.

    Если strategy="auto" — пробует GROBID, при недоступности падает на PyMuPDF.

    Пример:
        parser = PDFParser(grobid_url="http://localhost:8070")
        doc = parser.parse(pdf_path)
    """

    grobid_url: str = "http://localhost:8070"
    strategy: str = "auto"  # "auto" | "grobid" | "pymupdf"
    grobid_timeout: float = 120.0

    def __post_init__(self) -> None:
        self._grobid = GrobidParser(grobid_url=self.grobid_url, timeout=self.grobid_timeout)
        self._pymupdf = PyMuPDFParser()

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """
        Парсит PDF, автоматически выбирая стратегию.

        Args:
            pdf_path: Путь к PDF-файлу.

        Returns:
            ParsedDocument с секциями и текстом.

        Raises:
            FileNotFoundError: если PDF не найден.
            RuntimeError: если не удалось распарсить.
        """
        pdf_path = Path(pdf_path)

        if self.strategy == "pymupdf":
            return self._pymupdf.parse(pdf_path)

        if self.strategy == "grobid":
            return self._grobid.parse(pdf_path)

        # strategy == "auto": пробуем GROBID, потом PyMuPDF
        if self._grobid.is_available():
            try:
                return self._grobid.parse(pdf_path)
            except Exception as e:
                logger.warning(
                    "GROBID failed for %s: %s. Falling back to PyMuPDF.",
                    pdf_path.name, e,
                )

        logger.info("Using PyMuPDF parser for %s", pdf_path.name)
        return self._pymupdf.parse(pdf_path)

    def parse_status(self) -> dict[str, bool]:
        """Возвращает статус доступности каждого парсера."""
        return {
            "grobid": self._grobid.is_available(),
            "pymupdf": self._pymupdf.is_available(),
        }