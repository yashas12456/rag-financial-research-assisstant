"""Tests for PDF text extraction (app/ingestion/pdf_parser.py)."""
import pytest

from app.core.errors import EmptyDocumentError, InvalidFileError
from app.ingestion.pdf_parser import compute_content_hash, extract_pdf


def test_extract_pdf_returns_pages_with_text(sample_pdf_bytes):
    result = extract_pdf("sample.pdf", sample_pdf_bytes)
    assert result.num_pages == 2
    assert "revenue increased" in result.pages[0].text.lower()
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert result.has_any_text is True


def test_extract_pdf_invalid_bytes_raises():
    with pytest.raises(InvalidFileError):
        extract_pdf("not_a_pdf.pdf", b"this is not a pdf file")


def test_extract_pdf_empty_document_raises():
    import fitz
    doc = fitz.open()
    doc.new_page()  # blank page, no text
    pdf_bytes = doc.tobytes()
    doc.close()
    with pytest.raises(EmptyDocumentError):
        extract_pdf("blank.pdf", pdf_bytes)


def test_content_hash_is_deterministic():
    h1 = compute_content_hash(b"hello world")
    h2 = compute_content_hash(b"hello world")
    h3 = compute_content_hash(b"different content")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # sha256 hex digest length
