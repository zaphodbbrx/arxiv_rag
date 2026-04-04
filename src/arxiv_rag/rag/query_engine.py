"""
RAG Query Engine — query a LlamaIndex with citation formatting.

Wraps LlamaIndex's VectorQueryEngine and formats the response
with academic-style citations from source node metadata.

Usage:
    from src.rag.query_engine import ArxivQueryEngine

    engine = ArxivQueryEngine(index, model="qwen2.5:14b", top_k=5)
    result = engine.query("What are recent approaches to improving RAG?")
    print(result.answer)
    print(result.citations)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a scientific research assistant specializing in computer science papers from arXiv.

Rules:
1. Answer ONLY based on the provided context passages. If the context is insufficient, say: "I could not find enough information in the indexed papers to answer this."
2. Always cite your sources using the format: [Author et al., Year, arXiv:XXXX.XXXXX]
3. If multiple papers support the same point, cite all of them.
4. Do NOT hallucinate paper titles, authors, numbers, or findings.
5. Use technical terminology accurately.
6. Structure your answer clearly, starting with a direct response."""


@dataclass
class Citation:
    """A single citation extracted from a source node."""

    arxiv_id: str
    title: str
    authors: str
    year: Optional[str]
    url: str
    section: str
    score: float

    def format(self) -> str:
        """Format as academic citation."""
        author = self.authors.split(",")[0].strip().split()[-1] if self.authors else "Unknown"
        year = self.year or "n.d."
        return f"[{author} et al., {year}, arXiv:{self.arxiv_id}]"

    def __str__(self) -> str:
        return self.format()


@dataclass
class QueryResult:
    """Result of a RAG query."""

    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0

    def print_rich(self) -> None:
        """Pretty-print the result to console."""
        print(f"\n{'─' * 60}")
        print(f"A: {self.answer}")
        if self.citations:
            print(f"\n📚 Sources ({len(self.citations)}):")
            for c in self.citations:
                print(f"  {c.format()}")
                print(f"    {c.title[:80]}...")
                print(f"    Section: {c.section or '(general)'}")
                print(f"    Score: {c.score:.3f} | {c.url}")
        print(f"\n⏱  {self.latency_ms:.0f} ms ({self.model})")


class ArxivQueryEngine:
    """Query engine with academic citation formatting over arXiv papers.

    Uses a local LLM via Ollama (e.g. Qwen2.5-14B) by default.
    """

    def __init__(
        self,
        index,
        *,
        model: str = "qwen2.5:14b",
        top_k: int = 5,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        system_prompt: str = SYSTEM_PROMPT,
        request_timeout: float = 120.0,
    ) -> None:
        # Configure local LLM via Ollama
        Settings.llm = Ollama(
            model=model,
            temperature=temperature,
            request_timeout=request_timeout,
            system_prompt=SYSTEM_PROMPT,   # ← system prompt здесь
        )
        self.model = model
        self.top_k = top_k

        # Create query engine
        self.engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
        )

    def query(self, question: str) -> QueryResult:
        """Ask a question and get an answer with citations.

        Args:
            question: User's question.

        Returns:
            QueryResult with answer and formatted citations.
        """
        t0 = time.monotonic()

        logger.info("Querying: %s (top_k=%d, model=%s)", question[:100], self.top_k, self.model)

        # LlamaIndex query
        response = self.engine.query(question)

        latency = (time.monotonic() - t0) * 1000

        # Extract citations from source nodes
        citations = self._extract_citations(response.source_nodes) if response.source_nodes else []

        logger.info(
            "Response: %d chars, %d sources, %.0f ms",
            len(response.response or ""), len(citations), latency,
        )

        return QueryResult(
            query=question,
            answer=response.response or "",
            citations=citations,
            model=self.model,
            latency_ms=latency,
        )

    @staticmethod
    def _extract_citations(source_nodes) -> list[Citation]:
        """Convert LlamaIndex source nodes into Citation objects."""
        citations: list[Citation] = []

        for node in source_nodes:
            meta = node.metadata

            # Skip nodes without arxiv_id
            arxiv_id = meta.get("arxiv_id", "")
            if not arxiv_id:
                continue

            score = node.score if hasattr(node, "score") and node.score is not None else 0.0

            citations.append(Citation(
                arxiv_id=arxiv_id,
                title=meta.get("title", ""),
                authors=meta.get("authors", ""),
                year=meta.get("year"),
                url=meta.get("url", f"https://arxiv.org/abs/{arxiv_id}"),
                section=meta.get("section", ""),
                score=float(score),
            ))

        # Sort by score descending
        citations.sort(key=lambda c: c.score, reverse=True)
        return citations