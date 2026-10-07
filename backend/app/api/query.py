"""RAG query / chat endpoint."""
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.errors import AppError
from app.db.session import get_db
from app.models.models import Query as QueryModel
from app.schemas.schemas import Citation, QueryRequest, QueryResponse, RetrievedChunk
from app.services.query_service import persist_query, run_rag_pipeline

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def ask_question(payload: QueryRequest, db: Session = Depends(get_db)):
    document_ids = [str(d) for d in payload.document_ids] if payload.document_ids else None
    result = run_rag_pipeline(
        db, question=payload.question, document_ids=document_ids, top_k=payload.top_k
    )
    query = persist_query(db, payload.question, document_ids, result)

    return QueryResponse(
        query_id=query.id,
        question=payload.question,
        answer=result["answer"],
        insufficient_evidence=result["insufficient_evidence"],
        citations=[Citation(**c) for c in result["citations"]],
        retrieved_chunks=[RetrievedChunk(**rc) for rc in result["retrieved_chunks"]],
        latency_ms=result["latency_ms"],
    )


@router.get("/{query_id}")
def get_query(query_id: UUID, db: Session = Depends(get_db)):
    query = db.query(QueryModel).filter(QueryModel.id == query_id).first()
    if not query:
        raise AppError(f"Query {query_id} not found.")
    return {
        "id": query.id,
        "question": query.question,
        "answer": query.answer,
        "insufficient_evidence": query.insufficient_evidence,
        "latency_ms": query.latency_ms,
        "created_at": query.created_at,
    }
