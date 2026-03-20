"""
PDF Extractor
-------------
Extracts text page-by-page from PDFs using PyMuPDF (fitz).
Falls back to pdfplumber for pages where PyMuPDF yields poor results (e.g. scanned/table-heavy pages).
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import fitz  # PyMuPDF
import pdfplumber
from loguru import logger


@dataclass
class PageContent:
    page_num: int          # 1-based
    text: str
    char_count: int
    source_file: str
    extraction_method: str  # "pymupdf" | "pdfplumber"


@dataclass
class DocumentContent:
    source_file: str
    total_pages: int
    pages: List[PageContent] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


# Minimum chars on a page to trust PyMuPDF extraction
_MIN_CHARS_THRESHOLD = 100


def _clean_text(text: str) -> str:
    """Basic cleaning: fix hyphenation, collapse whitespace, strip control chars."""
    # Rejoin words broken across lines (e.g. "semiconduc-\ntor" → "semiconductor")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse multiple blank lines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove non-printable control characters (keep newlines/tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
    return text.strip()


def _extract_page_pymupdf(page: fitz.Page) -> str:
    return page.get_text("text")


def _extract_page_pdfplumber(pdf_path: str, page_num: int) -> str:
    """page_num is 0-based for pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        text = page.extract_text() or ""
        # Also extract tables and append as tab-separated rows
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                row_text = "\t".join(cell or "" for cell in row)
                if row_text.strip():
                    text += "\n" + row_text
        return text


def extract_pdf(pdf_path: str | Path) -> DocumentContent:
    """
    Extract all pages from a PDF.
    Uses PyMuPDF first; falls back to pdfplumber for sparse pages.
    """
    pdf_path = str(pdf_path)
    logger.info(f"Extracting: {pdf_path}")

    doc_content = DocumentContent(
        source_file=Path(pdf_path).name,
        total_pages=0,
    )

    with fitz.open(pdf_path) as doc:
        doc_content.total_pages = len(doc)
        for i, page in enumerate(doc):
            pymupdf_text = _extract_page_pymupdf(page)

            if len(pymupdf_text.strip()) >= _MIN_CHARS_THRESHOLD:
                text = pymupdf_text
                method = "pymupdf"
            else:
                logger.debug(f"  Page {i+1}: sparse via PyMuPDF, falling back to pdfplumber")
                text = _extract_page_pdfplumber(pdf_path, i)
                method = "pdfplumber"

            cleaned = _clean_text(text)
            doc_content.pages.append(
                PageContent(
                    page_num=i + 1,
                    text=cleaned,
                    char_count=len(cleaned),
                    source_file=Path(pdf_path).name,
                    extraction_method=method,
                )
            )

    total_chars = sum(p.char_count for p in doc_content.pages)
    logger.success(
        f"Extracted {doc_content.total_pages} pages, {total_chars:,} chars from {Path(pdf_path).name}"
    )
    return doc_content


def extract_all_pdfs(raw_dir: str | Path) -> List[DocumentContent]:
    """Extract all PDFs found in raw_dir."""
    raw_dir = Path(raw_dir)
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {raw_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s) in {raw_dir}")
    return [extract_pdf(p) for p in pdf_files]
