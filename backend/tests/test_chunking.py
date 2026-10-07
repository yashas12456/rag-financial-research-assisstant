"""Tests for text chunking (app/ingestion/chunking.py)."""
from app.ingestion.chunking import chunk_document, chunk_text
from app.ingestion.pdf_parser import ExtractedDocument, ExtractedPage


def test_chunk_text_respects_size_and_overlap():
    text = "word " * 1000  # 5000 chars
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 540  # allow small lookahead slack for word boundaries


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_overlap_must_be_smaller_than_size():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_chunk_document_preserves_metadata():
    doc = ExtractedDocument(
        filename="acme_2025.pdf",
        content_hash="abc123",
        file_size_bytes=1000,
        pages=[
            ExtractedPage(page_number=1, text="Item 1. Business\n\n" + "revenue data " * 100, is_empty=False),
            ExtractedPage(page_number=2, text="", is_empty=True),
        ],
    )
    chunks = chunk_document(doc, document_id="doc-123")
    assert len(chunks) > 0
    for c in chunks:
        assert c.document_id == "doc-123"
        assert c.filename == "acme_2025.pdf"
        assert c.page_number == 1  # only page 1 had text
        assert c.chunk_id  # non-empty uuid string
