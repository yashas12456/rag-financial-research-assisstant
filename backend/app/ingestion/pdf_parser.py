"""
PDF ingestion: page-level text extraction with metadata preservation.

WHAT: Extracts raw text from each page of a PDF using PyMuPDF (fitz),
      which is fast and handles most real-world financial-report PDFs
      (multi-column layout, tables-as-text, headers/footers) better
      than pure-Python alternatives.
WHY:  Every downstream component (chunking, citations) depends on
      accurate page numbers. We extract page-by-page, never as one
      giant blob, so that a citation like "Page 24" is always traceable
      to real PDF extraction -- never invented (see docs/ml_concepts.md,
      section on anti-hallucination design).
HOW:  fitz.open() gives us a Document; iterating page-by-page and calling
      page.get_text() returns that page's text. Pages with no extractable
      text (e.g. scanned images with no OCR layer) are flagged as empty
      rather than silently dropped or crashing the pipeline.
"""
import hashlib
from dataclasses import dataclass

import fitz  # PyMuPDF

from app.core.errors import EmptyDocumentError, InvalidFileError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedPage:
    page_number: int  # 1-indexed, matches what a human reading the PDF would see
    text: str
    is_empty: bool


@dataclass
class ExtractedDocument:
    filename: str
    content_hash: str
    file_size_bytes: int
    pages: list[ExtractedPage]

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def has_any_text(self) -> bool:
        return any(not p.is_empty for p in self.pages)


def compute_content_hash(file_bytes: bytes) -> str:
    """SHA-256 of the raw file bytes, used for duplicate-document detection."""
    return hashlib.sha256(file_bytes).hexdigest()


def extract_pdf(filename: str, file_bytes: bytes) -> ExtractedDocument:
    """
    Extract page-level text from PDF bytes.

    Raises:
        InvalidFileError: if the bytes are not a parseable PDF.
        EmptyDocumentError: if no page in the document contains extractable text
            (e.g. a fully scanned PDF with no OCR layer). We surface this clearly
            rather than silently indexing an empty document.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises various low-level errors on corrupt files
        logger.warning("Failed to open PDF %s: %s", filename, exc)
        raise InvalidFileError(
            f"'{filename}' could not be parsed as a PDF. It may be corrupted or not a real PDF."
        ) from exc

    pages: list[ExtractedPage] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text").strip()
            pages.append(
                ExtractedPage(page_number=i + 1, text=text, is_empty=(len(text) == 0))
            )
    finally:
        doc.close()

    extracted = ExtractedDocument(
        filename=filename,
        content_hash=compute_content_hash(file_bytes),
        file_size_bytes=len(file_bytes),
        pages=pages,
    )

    if not extracted.has_any_text:
        raise EmptyDocumentError(
            f"'{filename}' has no extractable text on any page. It may be a scanned "
            f"document without an OCR text layer. Try running OCR first, then re-upload."
        )

    empty_count = sum(1 for p in pages if p.is_empty)
    if empty_count:
        logger.info(
            "%s: %d/%d pages had no extractable text (likely images/blank pages)",
            filename, empty_count, len(pages),
        )

    return extracted
