# ArXiv RAG

RAG-система для семантического поиска и генерации ответов с цитированием по актуальным научным статьям с arXiv. Курсовой проект по курсу Advanced NLP.

---

## Архитектура

```
┌──────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                         │
│               (Streamlit UI + FastAPI Backend)                │
├──────────────────────────────────────────────────────────────┤
│                     ORCHESTRATION LAYER                       │
│            (query routing, hybrid search, re-ranking)         │
├──────────┬──────────┬───────────────┬────────────────────────┤
│ RETRIEVER│ RERANKER │    LLM (GEN)  │     EVALUATION         │
│ (BM25 +  │(Cross-   │  (OpenAI API) │  (RAGAS / custom       │
│  dense)  │ encoder) │               │   metrics)             │
├──────────┴──────────┴───────────────┴────────────────────────┤
│                     VECTOR STORE + INDEX                      │
│           (FAISS + BM25 index)                                │
├──────────────────────────────────────────────────────────────┤
│                    DOCUMENT PIPELINE                          │
│  arXiv API → PDF extraction (Grobid) → chunking → embeddings │
└──────────────────────────────────────────────────────────────┘
```

**Ключевые принципы:**

- **Hybrid retrieval** — плотный (embeddings) + разреженный (BM25) поиск с fusion через Reciprocal Rank Fusion
- **Semantic chunking** — разбиение статей по смысловым секциям (Introduction, Method, Results, ...) с сохранением структуры
- **Reranking** — Cross-Encoder для переупорядочивания кандидатов
- **Citation-aware generation** — LLM отвечает с автоматическим цитированием источников в формате `[Author, Year, arXiv:XXXX.XXXXX]`
- **Оценка качества** — RAGAS-метрики (faithfulness, answer relevance) + retrieval-метрики (Recall@K, MRR, NDCG) + ablation study

---

## Структура репозитория

```
arxiv-rag/
│
├── configs/                        # Конфигурация (YAML)
│   ├── base.yaml                   #   Глобальные настройки: data_dir, log_level, seed, device
│   ├── ingestion.yaml              #   Сбор данных: arxiv_categories, max_results, date_range
│   ├── chunking.yaml               #   Чанкинг: strategy, max_chunk_tokens, overlap_tokens
│   ├── embedding.yaml              #   Эмбеддинги: model_name, batch_size, normalize
│   ├── retrieval.yaml              #   Поиск: top_k, fusion_alpha, rerank_model
│   ├── generation.yaml             #   Генерация: LLM model, temperature, prompts
│   └── evaluation.yaml             #   Оценка: eval_dataset_path, metrics
│
├── src/
│   ├── config/
│   │   └── settings.py             # Pydantic Settings — единая точка доступа к конфигам
│   │
│   ├── data_ingestion/             # Загрузка и парсинг статей
│   │   ├── arxiv_client.py         #   Обёртка над arXiv API: поиск, скачивание PDF
│   │   ├── pdf_parser.py           #   Извлечение текста: Grobid (основной) + PyMuPDF (фоллбэк)
│   │   ├── models.py               #   ArxivPaper, Section, ParsedDocument
│   │   └── pipeline.py             #   IngestionPipeline: arXiv → PDF → текст → JSONL
│   │
│   ├── chunking/                   # Стратегии разбиения документов на чанки
│   │   ├── base.py                 #   BaseChunker (абстрактный класс)
│   │   ├── models.py               #   Chunk — атомарная единица с метаданными
│   │   ├── fixed_chunker.py        #   Baseline: фиксированный размер + перекрытие
│   │   ├── semantic_chunker.py     #   Основной: разбиение по секциям статьи
│   │   ├── hierarchical_chunker.py #   Parent-child чанки (для ablation)
│   │   └── factory.py              #   get_chunker(strategy) — фабрика
│   │
│   ├── retrieval/                  # Поиск по корпусу
│   │   ├── dense_retriever.py      #   Vector search: SentenceTransformer + FAISS
│   │   ├── sparse_retriever.py     #   BM25 (Okapi TF-IDF)
│   │   ├── hybrid_retriever.py     #   Fusion dense + sparse через RRF
│   │   ├── reranker.py             #   Cross-Encoder reranking
│   │   └── query_transform.py      #   HyDE, multi-query expansion
│   │
│   ├── generation/                 # LLM-генерация ответов
│   │   ├── llm_client.py           #   Единый интерфейс для OpenAI / Anthropic / Ollama
│   │   ├── rag_pipeline.py         #   Оркестратор: retrieve → rerank → generate → cite
│   │   ├── citation_handler.py     #   Парсинг и валидация цитат в ответе
│   │   └── prompts/                #   Шаблоны промптов
│   │       ├── system.txt          #     Системный промпт
│   │       ├── answer_with_citations.txt  # Ответ с цитатами
│   │       ├── compare_papers.txt  #     Сравнение статей
│   │       └── summarise_paper.txt #     Саммари статьи
│   │
│   ├── evaluation/                 # Оценка качества системы
│   │   ├── dataset_generator.py    #   LLM-генерация Q&A пар из статей
│   │   ├── retrieval_eval.py       #   Recall@K, MRR, NDCG@10
│   │   ├── ragas_eval.py           #   RAGAS: faithfulness, answer_relevance
│   │   └── ablation.py             #   Сравнение конфигураций (ablation study)
│   │
│   ├── api/                        # REST API (FastAPI)
│   │   ├── main.py                 #   FastAPI app, lifespan, CORS
│   │   ├── dependencies.py         #   DI: get_settings(), get_rag_pipeline()
│   │   ├── schemas.py              #   Pydantic модели: SearchRequest, SearchResponse
│   │   └── routes/
│   │       ├── search.py           #   POST /search, POST /search/stream
│   │       ├── papers.py           #   POST /papers/ingest, GET /papers
│   │       └── eval.py             #   POST /eval/run, GET /eval/results
│   │
│   ├── ui/
│   │   └── app.py                  # Streamlit: чат-интерфейс с sidebar-фильтрами
│   │
│   └── utils/                      # Общие утилиты
│       ├── tokenization.py         #   count_tokens(), split_sentences() (tiktoken)
│       ├── text_processing.py      #   normalize_whitespace(), SECTION_TYPE_MAP
│       ├── logging_config.py       #   structlog
│       └── io_utils.py             #   save/load JSON, Parquet, hash_file
│
├── data/                           # Данные (не коммитятся в git)
│   ├── raw/                        #   Сырые PDF
│   ├── processed/                  #   Парсированные JSONL
│   ├── chunks/                     #   Чанки в Parquet
│   ├── indexes/                    #   FAISS + BM25 индексы
│   └── eval/
│       ├── qa_pairs.json           #   Q&A тестовый датасет
│       └── results/                #   Результаты оценки
│
├── scripts/                        # CLI-скрипты для запуска пайплайнов
│   ├── ingest.py                   #   Загрузка статей: python scripts/ingest.py --query "RAG" --max 50
│   ├── build_index.py              #   Построение индексов: python scripts/build_index.py --strategy semantic
│   ├── run_eval.py                 #   Запуск оценки: python scripts/run_eval.py
│   ├── generate_qa.py              #   Генерация Q&A: python scripts/generate_qa.py --n-questions 5
│   └── demo.py                     #   Быстрый демо: python scripts/demo.py "query..."
│
├── tests/
│   ├── conftest.py                 #   Общие фикстуры: sample_paper, sample_chunks
│   ├── unit/                       #   Модульные тесты
│   │   ├── test_chunking.py
│   │   ├── test_retrieval.py
│   │   ├── test_query_transform.py
│   │   └── test_citation_handler.py
│   └── integration/                #   Интеграционные тесты
│       ├── test_ingestion_pipeline.py
│       ├── test_rag_pipeline.py
│       └── test_api.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   #   Исследование качества парсинга
│   ├── 02_chunking_analysis.ipynb  #   Сравнение стратегий чанкинга
│   └── 03_eval_visualization.ipynb #   Визуализация результатов ablation
│
├── docs/
│   ├── architecture.md             #   Подробное описание архитектуры
│   ├── data_collection.md          #   Протокол сбора данных
│   ├── evaluation_protocol.md      #   Протокол оценки (для курсовой)
│   └── api_reference.md            #   Документация API эндпоинтов
│
├── pyproject.toml                  # Зависимости (Poetry / uv)
├── Makefile                        # Быстрые команды: make install, make ingest, make serve
├── docker-compose.yml              # Grobid + API + UI
├── .env.example                    # Шаблон переменных окружения
├── .pre-commit-config.yaml         # Линтеры: ruff, mypy
└── .gitignore
```

---

## Технологический стек

| Слой | Технологии |
|------|-----------|
| **Data ingestion** | `arxiv` (API), `PyMuPDF` (PDF fallback), `Grobid` (PDF parsing, Docker) |
| **Chunking** | `tiktoken` (токенизация), `nltk`/`spacy` (split sentences) |
| **Embeddings** | `sentence-transformers`: all-MiniLM-L6-v2, SciBERT, bge-large |
| **Vector store** | `faiss-cpu` (или `Qdrant` для масштабирования) |
| **Sparse search** | `rank-bm25` (BM25Okapi) |
| **Reranking** | `sentence-transformers` CrossEncoder (ms-marco, bge-reranker) |
| **LLM generation** | `openai` API (GPT-4o-mini), streaming |
| **Evaluation** | `ragas` (faithfulness, relevance), custom metrics |
| **API** | `FastAPI`, `uvicorn`, `pydantic` |
| **UI** | `Streamlit` |
| **Infrastructure** | `Docker`, `docker-compose`, `Make` |
| **Quality** | `pytest`, `ruff`, `mypy`, `pre-commit` |

---

## Лицензия

MIT — только для учебных целей.