"""
Утилиты для ввода-вывода: сохранение/загрузка JSON, JSONL, Parquet, хеширование.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq

T = TypeVar("T")


# --- JSON / JSONL -----------------------------------------------------------

def save_json(data: Any, path: Path | str, *, indent: int = 2) -> Path:
    """Сохраняет объект в JSON-файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
    return path


def load_json(path: Path | str) -> Any:
    """Загружает объект из JSON-файла."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: list[dict[str, Any]], path: Path | str) -> Path:
    """Сохраняет список словарей в JSONL (один JSON на строку)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Загружает список словарей из JSONL-файла."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --- Parquet ----------------------------------------------------------------

def save_parquet(
    records: list[dict[str, Any]],
    path: Path | str,
    *,
    schema: pa.Schema | None = None,
) -> Path:
    """Сохраняет список словарей в Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path)
    return path


def load_parquet(path: Path | str) -> list[dict[str, Any]]:
    """Загружает Parquet-файл как список словарей."""
    path = Path(path)
    table = pq.read_table(path)
    return table.to_pylist()


# --- Хеширование ------------------------------------------------------------

def hash_file(path: Path | str, algorithm: str = "sha256") -> str:
    """Вычисляет хеш файла для проверки повторной обработки."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_string(text: str, algorithm: str = "sha256") -> str:
    """Вычисляет хеш строки."""
    return hashlib.new(algorithm, text.encode("utf-8")).hexdigest()


# --- Пути -------------------------------------------------------------------

def ensure_dir(path: Path | str) -> Path:
    """Создаёт директорию, если не существует."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path