"""
Query service: orchestrates the full RAG pipeline for a single question.

Pipeline (see docs/architecture.md for the diagram):

  question
    -> vector retrieval (top_k) + BM25 retrieval (top_k)
    -> RRF fusion
    -> cross-encoder reranking -> top N
    -> context builder (evidence passages with citations)
    -> LLM
    -> answer + citations
    -> persist Query + RetrievalResult rows for evaluation/audit
"""
import time
import uuid

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import InvalidQueryError
from app.core.logging import get_logger
from app.llm.client import EvidencePassage, get_llm_client
from app.models.models import DocumentChunk, Query, RetrievalResult
from app.reranking.cross_encoder import Reranker
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.services.ingestion_service import get_bm25_index, get_embedder, get_vector_store

logger = get_logger(__name__)
settings = get_settings()

_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def run_rag_pipeline(
    db: Session,
    question: str,
    document_ids: list[str] | None = None,
    top_k: int | None = None,
    use_bm25: bool = True,
    use_rerank: bool = True,
    use_vector: bool = True,
) -> dict:
    """
    Runs the pipeline and returns a dict with all intermediate results, so
    both the API layer and the evaluation scripts can reuse this single
    implementation (no duplicated pipeline logic -- see scripts/evaluate.py).
    """
    question = question.strip()
    if not question:
        raise InvalidQueryError("Question must not be empty.")

    start = time.perf_counter()

    vector_hits = []
    if use_vector:
        embedder = get_embedder()
        query_vector = embedder.embed_query(question)
        vector_hits = get_vector_store().search(
            query_vector=query_vector,
            top_k=top_k or settings.VECTOR_TOP_K,
            document_ids=document_ids,
            score_threshold=settings.SIMILARITY_THRESHOLD,
        )

    bm25_hits = []
    if use_bm25 and not get_bm25_index().is_empty:
        bm25_hits = get_bm25_index().search(
            question, top_k=top_k or settings.BM25_TOP_K, document_ids=document_ids
        )

    fused = reciprocal_rank_fusion(vector_hits, bm25_hits)
    candidate_ids = [f.chunk_id for f in fused[: max(settings.VECTOR_TOP_K, settings.BM25_TOP_K)]]

    if not candidate_ids:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "answer": "The provided documents do not contain enough information to answer "
                      "this question.",
            "insufficient_evidence": True,
            "citations": [],
            "retrieved_chunks": [],
            "latency_ms": latency_ms,
        }

    chunks_by_id = {
        str(c.id): c
        for c in db.query(DocumentChunk).filter(DocumentChunk.id.in_(candidate_ids)).all()
    }
    fused_score_map = {f.chunk_id: f for f in fused}

    if use_rerank and settings.RERANK_ENABLED:
        candidates = [(cid, chunks_by_id[cid].text) for cid in candidate_ids if cid in chunks_by_id]
        reranked = get_reranker().rerank(question, candidates, top_k=settings.RERANK_TOP_K)
        final_order = [(r.chunk_id, r.rerank_score) for r in reranked]
    else:
        # Baseline mode (no reranker): keep RRF order, no rerank_score.
        final_order = [(cid, None) for cid in candidate_ids[: settings.RERANK_TOP_K]]

    retrieved_chunks = []
    evidence: list[EvidencePassage] = []
    for rank, (cid, rerank_score) in enumerate(final_order, start=1):
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        fused_hit = fused_score_map.get(cid)
        marker = f"[{rank}]"
        evidence.append(
            EvidencePassage(
                marker=marker,
                filename=chunk.document.filename,
                page_number=chunk.page_number,
                text=chunk.text,
            )
        )
        retrieved_chunks.append(
            {
                "chunk_id": cid,
                "document_id": str(chunk.document_id),
                "filename": chunk.document.filename,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "text": chunk.text,
                "vector_score": fused_hit.vector_score if fused_hit else None,
                "bm25_score": fused_hit.bm25_score if fused_hit else None,
                "rrf_score": fused_hit.rrf_score if fused_hit else None,
                "rerank_score": rerank_score,
                "rank": rank,
            }
        )

    llm_response = get_llm_client().answer_question(question, evidence)

    citations = [
        {
            "marker": e.marker,
            "filename": e.filename,
            "page_number": e.page_number,
            "chunk_id": rc["chunk_id"],
            "rerank_score": rc["rerank_score"],
        }
        for e, rc in zip(evidence, retrieved_chunks)
    ]

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "answer": llm_response.answer,
        "insufficient_evidence": llm_response.insufficient_evidence,
        "citations": citations,
        "retrieved_chunks": retrieved_chunks,
        "latency_ms": latency_ms,
    }


def persist_query(db: Session, question: str, document_ids: list[str] | None, result: dict) -> Query:
    query = Query(
        question=question,
        answer=result["answer"],
        document_ids=document_ids or [],
        insufficient_evidence=result["insufficient_evidence"],
        latency_ms=result["latency_ms"],
    )
    db.add(query)
    db.flush()

    for rc in result["retrieved_chunks"]:
        db.add(
            RetrievalResult(
                query_id=query.id,
                chunk_id=uuid.UUID(rc["chunk_id"]),
                rank=rc["rank"],
                vector_score=rc["vector_score"],
                bm25_score=rc["bm25_score"],
                rrf_score=rc["rrf_score"],
                rerank_score=rc["rerank_score"],
                used_in_answer=True,
            )
        )
    db.flush()
    return query
