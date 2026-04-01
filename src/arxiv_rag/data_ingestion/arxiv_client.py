"""Client for fetching papers from the arXiv API."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import arxiv

from src.arxiv_rag.data_ingestion.models import ArxivPaper, Section, SectionType, classify_section

logger = logging.getLogger(__name__)

# arXiv API rate limit: ~1 request per 3 seconds
ARXIV_RATE_LIMIT_SECONDS = 3.0
MAX_RETRIES = 3


def _format_authors(raw: list) -> list[str]:
    """Convert arxiv.Author objects to plain name strings."""
    return [str(a) for a in raw]


def _iso_from_arxiv(dt) -> Optional[str]:
    """Convert arxiv datetime to ISO 8601 string."""
    if dt is None:
        return None
    return dt.isoformat()


class ArxivClient:
    """Wrapper around the arXiv API for searching and downloading papers.

    Usage::

        client = ArxivClient()
        papers = client.search_by_query("retrieval augmented generation", max_results=10)
        client.download_pdfs(papers, output_dir="data/raw")
    """

    def __init__(
        self,
        rate_limit_seconds: float = ARXIV_RATE_LIMIT_SECONDS,
        max_retries: int = MAX_RETRIES,
        raw_dir: str | Path | None = None,
    ):
        self._client = arxiv.Client(
            page_size=100,
            delay_seconds=rate_limit_seconds,
            num_retries=max_retries,
        )
        self._rate_limit = rate_limit_seconds
        self._raw_dir = Path(raw_dir) if raw_dir else Path("data/raw")

    # -- Search methods ---------------------------------------------------

    def search_by_query(
        self,
        query: str,
        max_results: int = 50,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sort_by: str = "submittedDate",
    ) -> list[ArxivPaper]:
        """Search arXiv by a free-text query.

        Args:
            query: Free-text search query (e.g. "retrieval augmented generation").
            max_results: Maximum number of papers to return.
            date_from: Start date filter (YYYY-MM-DD).
            date_to: End date filter (YYYY-MM-DD).
            sort_by: "relevance", "submittedDate", or "lastUpdatedDate".

        Returns:
            List of ArxivPaper objects with metadata.
        """
        logger.info(
            "Searching arXiv: query=%r, max_results=%d, date=%s..%s",
            query, max_results, date_from or "*", date_to or "*",
        )

        sort_map = {
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        }
        sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.SubmittedDate)

        search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_criterion)
        results = list(self._client.results(search))
        papers = self._results_to_papers(results)

        if date_from or date_to:
            papers = self._filter_by_date(papers, date_from, date_to)

        logger.info("Found %d papers for query %r", len(papers), query)
        return papers

    def search_by_category(
        self,
        category: str,
        max_results: int = 100,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> list[ArxivPaper]:
        """Search arXiv by a category (e.g. "cs.CL", "cs.AI").

        Args:
            category: arXiv category identifier.
            max_results: Maximum number of papers to return.
            date_from: Start date filter (YYYY-MM-DD).
            date_to: End date filter (YYYY-MM-DD).

        Returns:
            List of ArxivPaper objects.
        """
        logger.info("Searching arXiv: category=%r, max_results=%d", category, max_results)

        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        results = list(self._client.results(search))
        papers = self._results_to_papers(results)

        if date_from or date_to:
            papers = self._filter_by_date(papers, date_from, date_to)

        logger.info("Found %d papers for category %r", len(papers), category)
        return papers

    def search_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Fetch a single paper by its arXiv ID (e.g. "2401.12345").

        Args:
            arxiv_id: The arXiv paper identifier.

        Returns:
            ArxivPaper if found, None otherwise.
        """
        logger.info("Fetching paper by ID: %s", arxiv_id)
        clean_id = arxiv_id.replace("arxiv:", "").split("v")[0]

        search = arxiv.Search(id_list=[clean_id])
        results = list(self._client.results(search))

        if not results:
            logger.warning("Paper %s not found", arxiv_id)
            return None

        return self._results_to_papers([results[0]])[0]

    # -- PDF download ----------------------------------------------------

    def download_pdfs(
        self,
        papers: list[ArxivPaper],
        output_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        """Download PDF files for the given papers.

        Args:
            papers: List of ArxivPaper objects.
            output_dir: Directory to save PDFs to. Defaults to the
                raw_dir set in __init__ ("data/raw" if not specified).
            overwrite: Whether to re-download existing PDFs.

        Returns:
            List of paths to downloaded PDF files.
        """
        output_dir = Path(output_dir) if output_dir else self._raw_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[Path] = []
        failed: list[str] = []

        for paper in papers:
            pdf_filename = f"{paper.arxiv_id.replace('/', '_')}.pdf"
            pdf_path = output_dir / pdf_filename

            if pdf_path.exists() and not overwrite:
                logger.info("PDF already exists, skipping: %s", pdf_filename)
                paper.pdf_path = str(pdf_path)
                downloaded.append(pdf_path)
                continue

            if not paper.pdf_url:
                logger.warning("No PDF URL for paper %s, skipping", paper.arxiv_id)
                failed.append(paper.arxiv_id)
                continue

            try:
                logger.info("Downloading PDF: %s -> %s", paper.arxiv_id, pdf_filename)
                result = self._get_result_by_id(paper.arxiv_id)
                if result:
                    result.download_pdf(dirpath=str(output_dir), filename=pdf_filename)
                    paper.pdf_path = str(pdf_path)
                    downloaded.append(pdf_path)
                else:
                    failed.append(paper.arxiv_id)
                time.sleep(self._rate_limit)

            except Exception as e:
                logger.error("Failed to download PDF for %s: %s", paper.arxiv_id, e)
                failed.append(paper.arxiv_id)

        logger.info(
            "Downloaded %d PDFs, %d failed out of %d total",
            len(downloaded), len(failed), len(papers),
        )
        if failed:
            logger.warning("Failed papers: %s", failed)

        return downloaded

    # -- ar5iv text extraction (lightweight, no PDF) ---------------------

    def get_full_text_from_api(self, paper: ArxivPaper) -> Optional[str]:
        """Attempt to extract full text from the arXiv HTML version (ar5iv).

        This is a lightweight alternative to PDF parsing. It fetches the
        HTML version and strips tags to get plain text.

        Args:
            paper: Paper to fetch text for.

        Returns:
            Extracted text or None if unavailable.
        """
        arxiv_id = paper.arxiv_id.replace("/", "_")
        url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

        try:
            response = httpx.get(url, follow_redirects=True, timeout=30.0)
            if response.status_code != 200:
                logger.warning("ar5iv returned %d for %s", response.status_code, arxiv_id)
                return None

            html = response.text
            text = _strip_html_tags(html)

            if len(text) < 200:
                logger.warning("ar5iv returned very short text for %s", arxiv_id)
                return None

            return text

        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning("Failed to fetch ar5iv for %s: %s", arxiv_id, e)
            return None

    # -- Private helpers -------------------------------------------------

    def _results_to_papers(self, results: list) -> list[ArxivPaper]:
        """Convert a list of arxiv.Result objects to ArxivPaper dataclasses."""
        papers: list[ArxivPaper] = []

        for r in results:
            categories = list(r.categories) if r.categories else []
            paper = ArxivPaper(
                arxiv_id=r.entry_id.split("/abs/")[-1],
                title=r.title.replace("\n", " ").strip(),
                authors=_format_authors(r.authors),
                abstract=r.summary.replace("\n", " ").strip(),
                categories=categories,
                primary_category=r.primary_category,
                published=_iso_from_arxiv(r.published),
                updated=_iso_from_arxiv(r.updated),
                pdf_url=r.pdf_url or "",
                entry_url=r.entry_id,
                doi=r.doi,
            )
            papers.append(paper)

        return papers

    def _filter_by_date(
        self,
        papers: list[ArxivPaper],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> list[ArxivPaper]:
        """Filter papers by publication date range (YYYY-MM-DD)."""
        filtered = []
        for paper in papers:
            if not paper.published:
                filtered.append(paper)
                continue
            pub_date = paper.published[:10]
            if date_from and pub_date < date_from:
                continue
            if date_to and pub_date > date_to:
                continue
            filtered.append(paper)
        return filtered

    def _get_result_by_id(self, arxiv_id: str):
        """Get the raw arxiv.Result for a paper ID (for PDF download)."""
        clean_id = arxiv_id.replace("arxiv:", "").split("v")[0]
        search = arxiv.Search(id_list=[clean_id])
        results = list(self._client.results(search))
        return results[0] if results else None


def _strip_html_tags(html: str) -> str:
    """Remove HTML tags and clean up whitespace.

    Uses html.parser which is more tolerant of malformed HTML than ElementTree.
    """
    from html.parser import HTMLParser

    class _TagStripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts: list[str] = []

        def handle_data(self, data: str) -> None:
            self._parts.append(data)

        def get_text(self) -> str:
            return "".join(self._parts)

    stripper = _TagStripper()
    stripper.feed(html)
    text = stripper.get_text()

    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text