"""
ArXiv RAG Demo — ask questions about research papers using LlamaIndex + Ollama.

Usage:
    # Single query
    uv run python scripts/demo.py "What are recent approaches to improving RAG?"

    # With options
    uv run python scripts/demo.py "query..." --top-k 3 --model qwen2.5:14b -v

    # Rebuild index from scratch
    uv run python scripts/demo.py "query..." --rebuild

    # Interactive mode
    uv run python scripts/demo.py --interactive

Prerequisites:
    1. Ingest papers:    uv run python scripts/ingest.py --query "RAG" --max 10
    2. Install Ollama:  curl -fsSL https://ollama.com/install.sh | sh
    3. Pull model:      ollama pull qwen2.5:14b
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.arxiv_rag.data_ingestion.models import ArxivPaper
from src.arxiv_rag.rag.index_builder import build_index, load_index
from src.arxiv_rag.rag.query_engine import ArxivQueryEngine
from src.arxiv_rag.utils.io_utils import load_jsonl


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def load_papers(papers_path: Path) -> list[ArxivPaper]:
    """Load parsed papers from JSONL."""
    if not papers_path.exists():
        print(f"ERROR: No papers found at {papers_path}")
        print("Run ingestion first: uv run python scripts/ingest.py --query 'RAG' --max 10")
        sys.exit(1)

    records = load_jsonl(papers_path)
    papers = [ArxivPaper.from_dict(r) for r in records]
    parsed = [p for p in papers if p.is_parsed]
    print(f"  Loaded {len(parsed)} parsed papers (out of {len(papers)} total)")
    return parsed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ArXiv RAG Demo (LlamaIndex + Qwen via Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--papers", type=str, default="src/data/processed/papers.jsonl")
    parser.add_argument("--persist-dir", type=str, default="src/data/indexes/llama")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model", type=str, default="qwen2.5:14b", help="Ollama model (qwen2.5:7b, qwen2.5:14b)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)

    papers_path = Path(args.papers)
    persist_dir = Path(args.persist_dir)

    # ── Setup ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ArXiv RAG — LlamaIndex + Qwen (Ollama)")
    print("=" * 60)

    t0 = time.monotonic()

    # ── Step 1: Load papers ──────────────────────────────────────────
    print(f"\n[1/3] Loading papers from {papers_path}...")
    papers = load_papers(papers_path)

    # ── Step 2: Build or load index ──────────────────────────────────
    index_exists = (persist_dir / "docstore.json").exists()

    if args.rebuild or not index_exists:
        print(f"\n[2/3] Building index...")
        print(f"  Embedding: {args.embedding_model}")
        print(f"  Chunk size: {args.chunk_size}")

        index = build_index(
            papers,
            persist_dir=persist_dir,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
        )
    else:
        print(f"\n[2/3] Loading index from {persist_dir}...")
        index = load_index(
            persist_dir=persist_dir,
            embedding_model=args.embedding_model,
        )

    # ── Step 3: Create query engine ─────────────────────────────────
    print(f"\n[3/3] Initializing query engine (model={args.model}, top_k={args.top_k})...")
    print(f"  Make sure Ollama is running: ollama serve")
    engine = ArxivQueryEngine(
        index,
        model=args.model,
        top_k=args.top_k,
    )

    setup_ms = (time.monotonic() - t0) * 1000
    print(f"\n  Ready! Setup took {setup_ms:.0f} ms")
    print("=" * 60)

    # ── Query ────────────────────────────────────────────────────────

    def ask(question: str) -> None:
        result = engine.query(question)
        result.print_rich()

    if args.interactive:
        print("\n💬 Interactive mode (Ctrl+D or 'quit' to exit)\n")
        while True:
            try:
                question = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not question or question.lower() in ("quit", "exit", "q"):
                break
            ask(question)
    else:
        ask(args.query)


if __name__ == "__main__":
    main()