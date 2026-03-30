"""
Глобальная конфигурация проекта.

Загружает настройки из:
  1. Переменных окружения (.env)
  2. YAML-файлов из configs/
  3. Значений по умолчанию

Использование:
    from src.config.settings import get_settings
    settings = get_settings()
    print(settings.arxiv_categories)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# --- YAML-конфиги -----------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Загружает YAML-файл, возвращает пустой dict если файл не найден."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge(base: dict, override: dict) -> dict:
    """Рекурсивное слияние двух словарей (override побеждает)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


# --- Settings ---------------------------------------------------------------

class Settings(BaseSettings):
    """
    Единая точка доступа ко всем настройкам проекта.
    Все поля можно переопределить через переменные окружения.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # ARXIV__MAX_RESULTS → arxiv.max_results
        extra="ignore",
    )

    # -- Проект --------------------------------------------------------------
    project_name: str = "arxiv-rag"
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    log_level: str = "INFO"

    # -- arXiv ---------------------------------------------------------------
    arxiv_categories: list[str] = ["cs.CL", "cs.AI", "cs.LG"]
    arxiv_max_results: int = 100
    arxiv_date_from: str | None = None   # "2024-01-01"
    arxiv_date_to: str | None = None     # "2025-01-01"
    arxiv_delay_seconds: float = 3.0     # Rate-limiting между запросами

    # -- Данные --------------------------------------------------------------
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data"
    )
    raw_dir: Path | None = None
    processed_dir: Path | None = None
    chunks_dir: Path | None = None
    indexes_dir: Path | None = None
    eval_dir: Path | None = None

    # -- PDF парсинг ---------------------------------------------------------
    grobid_url: str = "http://localhost:8070"
    pdf_parser: str = "auto"  # "auto" | "grobid" | "pymupdf"

    # -- Чанкинг -------------------------------------------------------------
    chunking_strategy: str = "semantic"
    chunking_max_tokens: int = 512
    chunking_overlap_tokens: int = 64
    chunking_min_tokens: int = 50

    # -- Embeddings ----------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    embedding_normalize: bool = True

    # -- Retrieval -----------------------------------------------------------
    dense_top_k: int = 20
    sparse_top_k: int = 50
    fusion_alpha: float = 0.5
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5

    # -- Generation ----------------------------------------------------------
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    openai_api_key: str = ""

    @field_validator("arxiv_categories", mode="before")
    @classmethod
    def parse_categories(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [c.strip() for c in v.split(",") if c.strip()]
        return v

    def model_post_init(self, __context: Any) -> None:
        # Вычисляем пути относительно data_dir
        if self.raw_dir is None:
            self.raw_dir = self.data_dir / "raw"
        if self.processed_dir is None:
            self.processed_dir = self.data_dir / "processed"
        if self.chunks_dir is None:
            self.chunks_dir = self.data_dir / "chunks"
        if self.indexes_dir is None:
            self.indexes_dir = self.data_dir / "indexes"
        if self.eval_dir is None:
            self.eval_dir = self.data_dir / "eval"

    def ensure_dirs(self) -> None:
        """Создаёт все необходимые директории."""
        for d in [
            self.data_dir, self.raw_dir, self.processed_dir,
            self.chunks_dir, self.indexes_dir, self.eval_dir,
            self.eval_dir / "results",
        ]:
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> Settings:
    """
    Возвращает кэшированный экземпляр Settings.
    Опционально подгружает YAML-конфиг и мёржит его.
    """
    settings = Settings()

    # Подгрузка YAML (если передан путь)
    if config_path:
        yaml_data = _load_yaml(Path(config_path))
        merged = _merge(settings.model_dump(), yaml_data)
        settings = Settings(**merged)

    settings.ensure_dirs()
    return settings