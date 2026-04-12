"""
RAG index builder — converts ArxivPapers into a LlamaIndex VectorStoreIndex.

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

from __future__ import annotations

import logging
from pathlib import Path
import faiss
from src.arxiv_rag.rag.index_builder.base import BaseIndexBuilder

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from src.arxiv_rag.data_ingestion.models import ArxivPaper, SectionType

logger = logging.getLogger(__name__)


class LlamaIndexBuilder(BaseIndexBuilder):
    def papers_to_documents(self, papers: list[ArxivPaper]) -> list[Document]:
        """Convert parsed ArxivPapers into LlamaIndex Documents | Neo4j documents.

        One Document per section (not per paper) — this preserves section
        boundaries and attaches section-level metadata to every chunk.

        Sections with type SKIP (references, acknowledgments) are excluded.
        """
        documents: list[Document] = []
        skipped = 0

        for paper in papers:
            if not paper.is_parsed or not paper.sections:
                continue

            for section in paper.sections:
                stype = (
                    section.section_type.value
                    if isinstance(section.section_type, SectionType)
                    else str(section.section_type)
                )

                # Skip non-content sections
                if stype == "skip":
                    skipped += 1
                    continue

                text = section.content.strip()
                if not text or len(text) < 20:
                    continue

                doc = Document(
                    text=text,
                    metadata={
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "authors": ", ".join(paper.authors),
                        "first_author": paper.authors[0].split()[-1] if paper.authors else "Unknown",
                        "published": paper.published,
                        "year": paper.published[:4] if paper.published and len(paper.published) >= 4 else None,
                        "url": paper.entry_url or f"https://arxiv.org/abs/{paper.arxiv_id}",
                        "section": section.heading,
                        "section_type": stype,
                        "page_numbers": section.page_numbers,
                    },
                )
                documents.append(doc)

        logger.info(
            "Converted %d papers → %d documents (%d sections skipped)",
            len(papers), len(documents), skipped,
        )
        return documents


    def build_index(self,
        papers: list[ArxivPaper],
        *,
        persist_dir: str | Path = "data/indexes/llama",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        show_progress: bool = True,
    ) -> VectorStoreIndex:
        """Build a LlamaIndex VectorStoreIndex from ArxivPapers.

        Args:
            papers: List of parsed ArxivPaper objects.
            persist_dir: Directory to save the index (FAISS + docstore + metadata).
            embedding_model: HuggingFace model name for embeddings.
            chunk_size: Max tokens per chunk.
            chunk_overlap: Overlap between chunks.
            show_progress: Show progress bar for embedding.

        Returns:
            Built VectorStoreIndex (also persisted to disk).
        """
        persist_dir = Path(persist_dir)

        # 1. Convert papers → LlamaIndex Documents
        logger.info("Converting papers to LlamaIndex Documents...")
        documents = self.papers_to_documents(papers)

        if not documents:
            raise ValueError("No documents to index. Check that papers are parsed.")

        # 2. Configure LlamaIndex global settings
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
        )
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        # 3. Node parser (chunking)
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # 4. Parse documents into nodes
        logger.info("Parsing documents into nodes (chunk_size=%d, overlap=%d)...", chunk_size, chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=show_progress)
        logger.info("Created %d nodes (chunks)", len(nodes))

        # 5. Build index
        logger.info("Building FAISS vector index with %s...", embedding_model)

        # faiss_store = FaissVectorStore()
        test_embedding = Settings.embed_model.get_text_embedding("dimension probe")
        d = len(test_embedding)
        faiss_index = faiss.IndexFlatIP(d)        # создаём пустой FAISS-индекс
        faiss_store = FaissVectorStore(faiss_index=faiss_index)    
        storage_context = StorageContext.from_defaults(vector_store=faiss_store)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=show_progress,
        )

        # 6. Persist
        persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_dir))
        logger.info("Index persisted to %s", persist_dir)

        return index


    def load_index(self,
        persist_dir: str | Path = "data/indexes/llama",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> VectorStoreIndex:
        """Load a previously saved LlamaIndex index."""
        persist_dir = Path(persist_dir)

        if not (persist_dir / "docstore.json").exists():
            raise FileNotFoundError(
                f"No index found in {persist_dir}. Run build_index() first."
            )

        # Configure embedding model (must match the one used during build)
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        # Determine embedding dimension via a test embedding
        d = len(Settings.embed_model.get_text_embedding("probe"))

        # Create empty FAISS index — load_index_from_storage will repopulate it
        faiss_index = faiss.IndexFlatIP(d)
        faiss_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(
            vector_store=faiss_store,
            persist_dir=str(persist_dir),
        )

        index = load_index_from_storage(storage_context=storage_context)

        logger.info("Loaded index from %s", persist_dir)
        return index