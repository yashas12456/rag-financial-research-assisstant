"""
Hybrid retrieval: fusing dense vector search and BM25 with RRF.

ML CONCEPT: Reciprocal Rank Fusion (RRF)
WHAT: A rank-based (not score-based) fusion method for combining two or
      more ranked lists into one. For each item, RRF sums:

          RRF_score(d) = sum_over_lists( 1 / (k + rank_in_that_list(d)) )

      where rank_in_that_list is 1-indexed and `k` is a damping constant
      (we use k=60, the value from the original Cormack et al. 2009
      paper and the de-facto standard in hybrid-search implementations).
WHY:  Vector similarity scores and BM25 scores are on completely
      different, incomparable scales (cosine similarity is bounded
      [-1, 1]; BM25 is an unbounded, corpus-dependent magnitude). Naively
      averaging or weighting raw scores is fragile and requires constant
      re-tuning. RRF sidesteps this entirely by only using each list's
      *rank order*, which is far more stable across different
      queries/documents.
HOW:  A chunk that appears near the top of *either* list gets a high RRF
      score; a chunk that appears near the top of *both* lists gets an
      even higher one. A chunk found only by BM25 (e.g. an exact ticker
      match vector search missed) still surfaces into the candidate set
      that gets passed to the reranker -- this is precisely why hybrid
      retrieval outperforms either method alone (see docs/ml_concepts.md
      and the baseline comparison in scripts/benchmark.py).
"""
from dataclasses import dataclass

from app.core.config import get_settings
from app.retrieval.bm25_retriever import BM25Hit
from app.retrieval.vector_store import VectorHit

settings = get_settings()


@dataclass
class FusedHit:
    chunk_id: str
    rrf_score: float
    vector_score: float | None
    bm25_score: float | None


def reciprocal_rank_fusion(
    vector_hits: list[VectorHit],
    bm25_hits: list[BM25Hit],
    k: int | None = None,
) -> list[FusedHit]:
    k = k or settings.RRF_K

    vector_rank = {hit.chunk_id: i + 1 for i, hit in enumerate(vector_hits)}
    bm25_rank = {hit.chunk_id: i + 1 for i, hit in enumerate(bm25_hits)}
    vector_score_map = {hit.chunk_id: hit.score for hit in vector_hits}
    bm25_score_map = {hit.chunk_id: hit.score for hit in bm25_hits}

    all_chunk_ids = set(vector_rank) | set(bm25_rank)

    fused: list[FusedHit] = []
    for cid in all_chunk_ids:
        score = 0.0
        if cid in vector_rank:
            score += 1.0 / (k + vector_rank[cid])
        if cid in bm25_rank:
            score += 1.0 / (k + bm25_rank[cid])
        fused.append(
            FusedHit(
                chunk_id=cid,
                rrf_score=score,
                vector_score=vector_score_map.get(cid),
                bm25_score=bm25_score_map.get(cid),
            )
        )

    fused.sort(key=lambda h: h.rrf_score, reverse=True)
    return fused
