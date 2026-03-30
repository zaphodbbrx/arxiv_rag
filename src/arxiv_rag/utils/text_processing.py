"""
Утилиты для работы с текстом: нормализация, секции, токенизация.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---- Маппинг заголовков секций → нормализованный тип ------------------------

SECTION_TYPE_MAP: dict[str, str] = {
    # Основные
    "introduction": "introduction",
    "related work": "related_work",
    "related works": "related_work",
    "background": "background",
    "background and related work": "related_work",
    "preliminaries": "background",
    "methodology": "method",
    "method": "method",
    "methods": "method",
    "approach": "method",
    "model": "method",
    "models": "method",
    "framework": "method",
    "proposed method": "method",
    "our method": "method",
    "our approach": "method",
    # Эксперименты
    "experiments": "experiments",
    "experiment": "experiments",
    "experimental setup": "experiments",
    "experimental results": "results",
    "setup": "experiments",
    "results": "results",
    "results and discussion": "results",
    # Анализ
    "discussion": "discussion",
    "analysis": "discussion",
    # Заключение
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "conclusion and future work": "conclusion",
    "summary": "conclusion",
    "future work": "future_work",
    "limitations": "limitations",
    "limitations and future work": "limitations",
    # Пропускаемые
    "references": "skip",
    "bibliography": "skip",
    "acknowledgments": "skip",
    "acknowledgements": "skip",
    "appendix": "appendix",
    "supplementary": "appendix",
    "supplementary material": "appendix",
}


@dataclass(frozen=True)
class SectionType:
    """Нормализованный тип секции с флагами обработки."""
    name: str
    is_skippable: bool
    is_main_body: bool

    @staticmethod
    def from_heading(heading: str) -> SectionType:
        """
        Определяет тип секции по заголовку.
        Нечувствительна к регистру, номерам, лишним пробелам.
        """
        normalized = _normalize_heading(heading)
        section_type = SECTION_TYPE_MAP.get(normalized, "other")

        return SectionType(
            name=section_type,
            is_skippable=section_type == "skip",
            is_main_body=section_type in {
                "introduction", "related_work", "background",
                "method", "experiments", "results",
                "discussion", "conclusion", "limitations", "future_work",
            },
        )


def _normalize_heading(heading: str) -> str:
    """
    Нормализует заголовок секции:
      - Убирает номерацию ("1. Introduction" → "introduction")
      - Нижний регистр
      - Убирает лишние пробелы
      - Убирает LaTeX-маркеры
    """
    # Убираем номерацию: "1.", "1.1", "1.1.1", "A.", "A.1"
    text = re.sub(r"^(?:\d+\.?\s*|[A-Z]\.?\s*)+", "", heading)
    # Убираем LaTeX-команды
    text = re.sub(r"\\(?:textbf|textit|emph|section)\{([^}]*)\}", r"\1", text)
    # Нижний регистр + нормализация пробелов
    text = " ".join(text.lower().split())
    return text


# ---- Текстовые операции -----------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Заменяет множественные пробелы/newlines на одиночные."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_latex_math(text: str) -> str:
    """
    Очищает текст от inline/display LaTeX-формул.
    Заменяет на читаемые описания, сохраняя смысл.
    """
    # Display math: $$...$$ или \[...\]
    text = re.sub(r"\$\$(.+?)\$\$", r"[EQUATION]", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.+?\\\]", r"[EQUATION]", text, flags=re.DOTALL)
    # Inline math: $...$
    text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", r"[FORMULA]", text)
    # \begin{equation}...\end{equation}
    text = re.sub(
        r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}",
        r"[EQUATION]", text, flags=re.DOTALL,
    )
    return text


def extract_section_number(heading: str) -> str | None:
    """Извлекает номер секции из заголовка: '2.3 Methodology' → '2.3'."""
    match = re.match(r"^(\d+(?:\.\d+)*)", heading.strip())
    return match.group(1) if match else None