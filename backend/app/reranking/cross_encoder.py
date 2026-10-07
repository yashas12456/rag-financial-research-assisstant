"""
Cross-encoder reranking.

ML CONCEPT: Cross-encoder reranking (second-stage retrieval)
WHAT: Unlike the bi-encoder used for embeddings (which encodes the query
      and each chunk *separately* into vectors, then compares them), a
      cross-encoder feeds the (query, chunk) pair *together* into a
      single transformer forward pass and outputs one relevance score
      directly. No vectors, no similarity function -- one score per pair.
WHY:  Cross-encoders are far more accurate at judging fine-grained
      relevance because the model can attend between the query and the
      passage token-by-token, but that accuracy comes at the cost of
      needing one full forward pass per (query, chunk) pair -- it cannot
      be pre-computed or indexed like bi-encoder embeddings can. That
      makes it too slow to run over an entire corpus, so we run it only
      over the small candidate set (20-50 chunks) that hybrid retrieval
      already narrowed things down to.
HOW:  First-stage retrieval (BM25 + vector search, fused with RRF) casts
      a wide, cheap net over the whole corpus. The cross-encoder then
      re-scores just those candidates and we keep only the strongest
      RERANK_TOP_K, which become the actual evidence sent to the LLM.
      This two-stage "retrieve cheap, rerank precise" pattern is standard
      in production search/RAG systems because it gets both the speed of
      bi-encoders and the accuracy of cross-encoders.
"""
from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    logger.info("Loading cross-encoder reranker: %s", settings.RERANKER_MODEL)
    return CrossEncoder(settings.RERANKER_MODEL)


@dataclass
class RerankedHit:
    chunk_id: str
    rerank_score: float


class Reranker:
    def __init__(self):
        self._model = _get_model()

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_k: int | None = None
    ) -> list[RerankedHit]:
        """
        candidates: list of (chunk_id, chunk_text) tuples -- the fused
                    candidate set from hybrid retrieval.
        """
        top_k = top_k or settings.RERANK_TOP_K
        if not candidates:
            return []

        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip((cid for cid, _ in candidates), scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [RerankedHit(chunk_id=cid, rerank_score=float(s)) for cid, s in ranked[:top_k]]
