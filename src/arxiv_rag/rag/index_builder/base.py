"""
RAG index builder — converts ArxivPapers into a VectorStoreIndex LlamaIndex VectorStoreIndex.

Pipeline:
  ArxivPaper (JSONL)
    → LlamaIndex Documents (one per section)
    → SentenceSplitter (chunking)
    → HuggingFaceEmbedding (encoding)
    → FAISS VectorStore (index)

Usage:
    from src.rag.index_builder import build_index, load_index

    # Build from scratch
    index, node_count = build_index(papers, persist_dir="data/indexes/llama")

    # Load existing
    index = load_index(persist_dir="data/indexes/llama")
"""


from abc import ABC, abstractmethod

from src.arxiv_rag.data_ingestion.models import ArxivPaper



class BaseIndexBuilder(ABC):
    """Абстрактный индекс для векторной БД."""

    @abstractmethod
    def load_index():
        raise NotImplementedError()
    
    @abstractmethod
    def build_index(self, papers: list[ArxivPaper]):
        raise NotImplementedError()

