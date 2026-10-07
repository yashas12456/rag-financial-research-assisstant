"""Tests for BM25 keyword retrieval (app/retrieval/bm25_retriever.py)."""
from app.retrieval.bm25_retriever import BM25Index


def _build_index():
    texts = {
        "c1": "Apple reported strong iPhone revenue growth in fiscal year 2025.",
        "c2": "Tesla delivered a record number of vehicles this quarter.",
        "c3": "NVIDIA gross margin expanded due to data center demand.",
    }
    docs = {"c1": "doc-apple", "c2": "doc-tesla", "c3": "doc-nvidia"}
    idx = BM25Index()
    idx.build(texts, docs)
    return idx


def test_bm25_finds_exact_keyword_match():
    idx = _build_index()
    results = idx.search("iPhone revenue", top_k=3)
    assert results[0].chunk_id == "c1"


def test_bm25_respects_document_filter():
    idx = _build_index()
    results = idx.search("revenue", top_k=3, document_ids=["doc-tesla"])
    assert all(r.chunk_id == "c2" for r in results) or results == []


def test_bm25_empty_index_returns_empty():
    idx = BM25Index()
    assert idx.is_empty
    assert idx.search("anything", top_k=5) == []
