"""Tests for Reciprocal Rank Fusion (app/retrieval/hybrid.py)."""
from app.retrieval.bm25_retriever import BM25Hit
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.vector_store import VectorHit


def test_rrf_favors_items_ranked_high_in_both_lists():
    vector_hits = [VectorHit(chunk_id="a", score=0.9), VectorHit(chunk_id="b", score=0.8)]
    bm25_hits = [BM25Hit(chunk_id="a", score=5.0), BM25Hit(chunk_id="c", score=4.0)]

    fused = reciprocal_rank_fusion(vector_hits, bm25_hits, k=60)
    # "a" appears at rank 1 in both lists -> should be first
    assert fused[0].chunk_id == "a"
    assert {f.chunk_id for f in fused} == {"a", "b", "c"}


def test_rrf_includes_items_found_by_only_one_method():
    vector_hits = [VectorHit(chunk_id="x", score=0.5)]
    bm25_hits = [BM25Hit(chunk_id="y", score=1.0)]
    fused = reciprocal_rank_fusion(vector_hits, bm25_hits, k=60)
    ids = {f.chunk_id for f in fused}
    assert "x" in ids and "y" in ids  # neither retrieval method's unique hits are dropped


def test_rrf_score_formula():
    vector_hits = [VectorHit(chunk_id="a", score=0.9)]
    bm25_hits = []
    fused = reciprocal_rank_fusion(vector_hits, bm25_hits, k=60)
    assert abs(fused[0].rrf_score - (1.0 / (60 + 1))) < 1e-9
