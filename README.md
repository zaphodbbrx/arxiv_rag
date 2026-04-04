# ArXiv RAG

RAG-система для поиска и ответа на вопросы по научным статьям с arXiv.
Курсовой проект по курсу **Advanced NLP**.

## Архитектура

```
arXiv API ──► ArxivClient ──► PDF ──► PDFParser (Grobid / PyMuPDF)
                                            │
                                    ParsedDocument (секции)
                                            │
                                    IngestionPipeline
                                            │
                                      papers.jsonl
                                            │
                              ┌─────────────┴──────────────┐
                              │  LlamaIndex VectorStoreIndex │
                              │  ├─ SentenceSplitter (512)  │
                              │  ├─ HuggingFaceEmbedding    │
                              │  └─ FAISS (IndexFlatIP)     │
                              └─────────────┬──────────────┘
                                            │
                                 ArxivQueryEngine
                                 └─ Ollama (qwen2.5:14b)
                                            │
                                    QueryResult + цитаты
```

## Стек технологий

| Компонент | Технология |
|---|---|
| RAG-фреймворк | [LlamaIndex](https://github.com/run-llama/llama_index) |
| Векторное хранилище | [FAISS](https://github.com/facebookresearch/faiss) (IndexFlatIP) |
| Эмбеддинги | `sentence-transformers/all-MiniLM-L6-v2` через [llama-index-embeddings-huggingface](https://pypi.org/project/llama-index-embeddings-huggingface/) |
| LLM | [Ollama](https://ollama.com/) с `qwen2.5:14b` (локально, через [llama-index-llms-ollama](https://pypi.org/project/llama-index-llms-ollama/)) |
| Парсинг PDF | [GROBID](https://github.com/kermitt2/grobid) (основной) + [PyMuPDF](https://pymupdf.readthedocs.io/) (fallback) |
| Поиск по arXiv | [arxiv](https://pypi.org/project/arxiv/) Python-пакет |
| Конфигурация | [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) + YAML |
| Пакетный менеджер | [uv](https://docs.astral.sh/uv/) |

## Структура проекта

```
arxiv_rag/
├── pyproject.toml                # Зависимости и метаданные
├── .python-version               # Python 3.10
│
├── configs/                      # YAML-конфиги (пока пусто)
│
├── data/                         # Генерируемые данные (.gitignore)
│   ├── raw/                      # Скачанные PDF
│   ├── processed/                # papers.jsonl — результат ingest
│   └── indexes/llama/            # Сохранённый FAISS-индекс
│
├── scripts/
│   ├── ingest.py                 # CLI: поиск → скачивание → парсинг → JSONL
│   └── demo.py                   # CLI: построение индекса + запросы к RAG
│
└── src/
    └── arxiv_rag/
        ├── config/
        │   └── settings.py       # Pydantic Settings — вся конфигурация проекта
        │
        ├── data_ingestion/
        │   ├── arxiv_client.py   # Поиск и скачивание статей с arXiv
        │   ├── models.py         # ArxivPaper, Section, ParsedDocument
        │   ├── pdf_parser.py     # GrobidParser + PyMuPDFParser + фабрика
        │   └── pipeline.py       # IngestionPipeline: оркестрация всего пайплайна
        │
        ├── rag/
        │   ├── index_builder.py  # Papers → LlamaIndex Documents → FAISS Index
        │   └── query_engine.py   # ArxivQueryEngine: запросы с академическими цитатами
        │
        ├── evaluation/           # (запланировано) RAGAS-оценка
        ├── api/                  # (запланировано) FastAPI REST API
        ├── ui/                   # (запланировано) Streamlit-интерфейс
        │
        └── utils/
            ├── io_utils.py       # JSON/JSONL/Parquet I/O
            └── text_processing.py # Нормализация текста, очистка LaTeX
```

## Быстрый старт

### 1. Установка зависимостей

```bash
# Клонировать репозиторий
git clone https://github.com/zaphodbbrx/arxiv_rag.git
cd arxiv_rag
git checkout mvp

# Создать виртуальное окружение и установить зависимости
uv sync
```

### 2. Скачивание и парсинг статей

```bash
# Поиск по ключевому слову (скачает до 10 статей)
uv run python scripts/ingest.py --query "retrieval augmented generation" --max 10

# Поиск по категории arXiv
uv run python scripts/ingest.py --category cs.CL --max 20

# Комбинированный поиск с датами
uv run python scripts/ingest.py --query "RAG" --category cs.CL --date-from 2024-01-01 --max 30

# Пропустить скачивание (использовать уже скачанные PDF)
uv run python scripts/ingest.py --query "RAG" --skip-download
```

Результат: `data/processed/papers.jsonl` — по одной JSON-записи на статью с распарсенными секциями.

> **Grobid** (опционально): для качественного парсинга структуры статьи запустите
> `docker run -d -p 8070:8070 lfoppiano/grobid:latest`. Без него используется PyMuPDF
> (быстрее, но менее точное разбиение на секции).

### 3. Запуск RAG-демо

```bash
# Установить и запустить Ollama с моделью
ollama pull qwen2.5:14b
ollama serve

# Один запрос
uv run python scripts/demo.py "What are recent approaches to improving RAG?"

# Интерактивный режим
uv run python scripts/demo.py --interactive

# С опциями
uv run python scripts/demo.py --interactive --top-k 10 --model qwen2.5:7b -v
```

При первом запуске `demo.py` автоматически:
1. Загружает `papers.jsonl`
2. Строит FAISS-индекс (сохраняется в `data/indexes/llama/`)
3. При последующих запусках — загружает существующий индекс (используйте `--rebuild` для пересборки)

### Аргументы CLI

#### `ingest.py`

| Аргумент | Описание | По умолчанию |
|---|---|---|
| `--query` | Поисковый запрос по заголовкам/аннотациям | — |
| `--category` | Категория arXiv (напр. `cs.CL`, `cs.AI`) | — |
| `--max` | Максимальное число статей | 10 |
| `--date-from` | Начальная дата (`YYYY-MM-DD`) | — |
| `--date-to` | Конечная дата (`YYYY-MM-DD`) | — |
| `--skip-download` | Не скачивать PDF повторно | `false` |
| `--skip-parse` | Не парсить PDF повторно | `false` |
| `--force` | Переписать существующие записи | `false` |
| `--parser` | Парсер: `auto`, `grobid`, `pymupdf` | `auto` |
| `--dry-run` | Показать что будет скачано, без实际行动 | `false` |
| `-v`, `--verbose` | Детальное логирование | `false` |

#### `demo.py`

| Аргумент | Описание | По умолчанию |
|---|---|---|
| `query` | Вопрос к RAG (позиционный) | — |
| `-i`, `--interactive` | Интерактивный режим | `false` |
| `--rebuild` | Пересобрать индекс | `false` |
| `--top-k` | Число извлекаемых чанков | 5 |
| `--model` | Ollama модель | `qwen2.5:14b` |
| `--chunk-size` | Размер чанка (токены) | 512 |
| `--papers` | Путь к `papers.jsonl` | `data/processed/papers.jsonl` |
| `--persist-dir` | Путь к индексу | `data/indexes/llama` |
| `-v`, `--verbose` | Детальное логирование | `false` |

## Ключевые модули

### `data_ingestion/` — Сбор и парсинг статей

**`IngestionPipeline`** — центральный класс, который:
1. Ищет статьи на arXiv (по запросу и/или категории)
2. Скачивает PDF с rate limiting (3 сек между запросами)
3. Парсит каждый PDF через Grobid (при наличии Docker) или PyMuPDF
4. Сохраняет результат в JSONL с инкрементальным обновлением (не перепарсивает уже обработанные `arxiv_id`)

**`ArxivPaper`** — основная модель данных:
- Метаданные: `arxiv_id`, `title`, `authors`, `published`, `categories`, `abstract`
- Распарсенный контент: `sections[]` (с типом секции), `full_text`
- Сериализация: `to_dict()` / `from_dict()` для JSONL

**`PDFParser`** — фабрика с двумя стратегиями:
- **GrobidParser**: извлекает структуру из TEI XML (секции, формулы, таблицы). Требует запущенный GROBID Docker-контейнер
- **PyMuPDFParser**: локальный fallback с эвристическим определением секций (нумерованные заголовки, ALL CAPS)

### `rag/` — Индексация и поиск

**`index_builder.py`** — строит FAISS-индекс через LlamaIndex:
1. `papers_to_documents()` — конвертирует `ArxivPaper` в LlamaIndex `Document` (один документ на секцию, секции типа SKIP исключаются)
2. `SentenceSplitter` разбивает на чанки (512 токенов, 64 overlap)
3. `HuggingFaceEmbedding` кодирует чанки (384-мерные векторы)
4. `FAISS IndexFlatIP` хранит векторы (inner product similarity)
5. Индекс сохраняется на диск: FAISS + docstore + metadata

**`query_engine.py`** — обёртка над `index.as_query_engine()`:
- Настройка локальной LLM через Ollama (`qwen2.5:14b` по умолчанию)
- `response_mode="compact"` — сжатие контекста перед генерацией
- Автоматическое извлечение цитат из `source_nodes` в формате `[Author et al., Year, arXiv:XXXX.XXXXX]`
- `QueryResult` с `answer`, `citations[]`, `latency_ms`

### `config/settings.py` — Централизованная конфигурация

Pydantic Settings с поддержкой:
- Переменных окружения (с префиксом `ARXIV_RAG__`)
- YAML-файлов конфигурации
- Вычисляемых путей через `model_post_init`
- `ensure_dirs()` для создания структуры директорий

## Запланированные фичи

- [ ] **Hybrid retrieval**: BM25 (sparse) + dense retrieval + Reciprocal Rank Fusion
- [ ] **Cross-encoder reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- [ ] **Оценка качества**: RAGAS (faithfulness, answer relevancy, context precision/recall)
- [ ] **REST API**: FastAPI эндпоинты для запросов
- [ ] **Web UI**: Streamlit-интерфейс
- [ ] **Docker Compose**: GROBID + приложение
- [ ] **Тесты**: pytest для всех модулей

## Требования к окружению

- Python >= 3.10
- GPU: NVIDIA с 16+ GB VRAM (для `qwen2.5:14b`; альтернатива — `qwen2.5:7b` на 8 GB)
- [Ollama](https://ollama.com/) — для локального запуска LLM
- Docker (опционально) — для GROBID парсер