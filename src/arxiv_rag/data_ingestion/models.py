"""Data models for arXiv paper ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class SectionType(str, Enum):
    """Normalized section types for scientific papers."""

    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    BACKGROUND = "background"
    METHOD = "method"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    FUTURE_WORK = "future_work"
    LIMITATIONS = "limitations"
    APPENDIX = "appendix"
    ABSTRACT = "abstract"

    # Sections to skip during indexing
    SKIP = "skip"

    # Fallback for unrecognized sections
    OTHER = "other"


# Mapping from raw section headings to normalized types.
# Order matters -- first match wins. Keys are lowered.
SECTION_TYPE_MAP: list[tuple[str, SectionType]] = [
    ("abstract", SectionType.ABSTRACT),
    ("introduction", SectionType.INTRODUCTION),
    ("related work", SectionType.RELATED_WORK),
    ("background", SectionType.BACKGROUND),
    ("preliminar", SectionType.BACKGROUND),
    ("problem formulation", SectionType.METHOD),
    ("methodology", SectionType.METHOD),
    ("method", SectionType.METHOD),
    ("approach", SectionType.METHOD),
    ("model", SectionType.METHOD),
    ("framework", SectionType.METHOD),
    ("architecture", SectionType.METHOD),
    ("experimental setup", SectionType.EXPERIMENTS),
    ("setup", SectionType.EXPERIMENTS),
    ("experiments", SectionType.EXPERIMENTS),
    ("experiment", SectionType.EXPERIMENTS),
    ("results", SectionType.RESULTS),
    ("experimental results", SectionType.RESULTS),
    ("analysis", SectionType.DISCUSSION),
    ("discussion", SectionType.DISCUSSION),
    ("conclusion", SectionType.CONCLUSION),
    ("conclusions", SectionType.CONCLUSION),
    ("future work", SectionType.FUTURE_WORK),
    ("limitations", SectionType.LIMITATIONS),
    ("appendix", SectionType.APPENDIX),
    ("supplementary", SectionType.APPENDIX),
    # Sections to skip
    ("references", SectionType.SKIP),
    ("bibliography", SectionType.SKIP),
    ("acknowledgment", SectionType.SKIP),
    ("acknowledgements", SectionType.SKIP),
]


def classify_section(heading: str) -> SectionType:
    """Classify a section heading into a normalized SectionType.

    Uses fuzzy matching -- checks if any known keyword appears in the heading.
    Returns SectionType.METHOD as fallback for unknown headings.
    """
    heading_lower = heading.lower().strip()

    for keyword, section_type in SECTION_TYPE_MAP:
        if keyword in heading_lower:
            return section_type

    # Unknown section -- keep as method candidate (most content-rich sections)
    return SectionType.METHOD


@dataclass
class Section:
    """A single section of a parsed paper."""

    heading: str
    content: str
    section_type: SectionType = field(default_factory=lambda: SectionType.METHOD)
    page_numbers: list[int] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.section_type, str):
            try:
                self.section_type = SectionType(self.section_type)
            except ValueError:
                self.section_type = SectionType.OTHER
        if self.section_type == SectionType.METHOD:
            self.section_type = classify_section(self.heading)

    def is_empty(self) -> bool:
        """True if the section has no meaningful content."""
        return not self.content.strip()


@dataclass
class ParsedDocument:
    """Structured result of PDF parsing."""

    full_text: str = ""
    sections: list[Section] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def total_sections(self) -> int:
        return len(self.sections)

    @property
    def main_sections(self) -> list[Section]:
        """Sections of the main body (skip references, acknowledgments, etc.)."""
        return [s for s in self.sections if s.section_type != SectionType.SKIP]

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


@dataclass
class ArxivPaper:
    """Metadata and content for a single arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    categories: list[str] = field(default_factory=list)
    published: Optional[str] = None  # ISO format string
    updated: Optional[str] = None
    pdf_url: str = ""
    entry_url: str = ""
    doi: Optional[str] = None
    primary_category: str = ""

    # Populated after PDF parsing
    pdf_path: Optional[str] = None
    full_text: Optional[str] = None
    sections: list[Section] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)

    # Parsing status
    parse_status: str = "pending"  # pending | parsed | failed
    parse_error: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dict (JSON-safe). Handles Section dataclasses."""
        data = asdict(self)
        # Convert sections' SectionType enums to strings
        for section in data.get("sections", []):
            if "section_type" in section and isinstance(
                section["section_type"], SectionType
            ):
                section["section_type"] = section["section_type"].value
        return data

    @property
    def published_year(self) -> Optional[int]:
        """Extract publication year from published date."""
        if self.published:
            try:
                return datetime.fromisoformat(self.published).year
            except (ValueError, TypeError):
                return None
        return None

    @property
    def first_author(self) -> str:
        """Return the first author's last name for citation."""
        if self.authors:
            return self.authors[0].split()[-1]
        return "Unknown"

    @property
    def is_parsed(self) -> bool:
        """True if the paper was successfully parsed."""
        return self.parse_status == "parsed" and bool(self.full_text)

    def apply_parsed_document(self, doc: ParsedDocument) -> None:
        """Fill text fields from a ParsedDocument result."""
        self.full_text = doc.full_text
        self.sections = doc.sections
        self.tables = doc.tables
        self.equations = doc.equations
        self.parse_status = "parsed"

    @classmethod
    def from_dict(cls, data: dict) -> ArxivPaper:
        """Deserialize from a dict (inverse of to_dict)."""
        # Restore Section objects with proper SectionType
        raw_sections = data.pop("sections", [])
        sections = []
        for s in raw_sections:
            s = dict(s)
            st = s.get("section_type")
            if isinstance(st, str):
                s["section_type"] = SectionType(st)
            sections.append(Section(**s))
        data["sections"] = sections
        return cls(**data)

    def __str__(self) -> str:
        return f"[{self.arxiv_id}] {self.title} ({self.first_author} et al., {self.published_year or '?'})"