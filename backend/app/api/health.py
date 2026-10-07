"""Health check endpoint -- verifies DB, vector store, and config are reachable/valid."""
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.schemas.schemas import HealthResponse

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"error: {exc}"

    vector_status = "ok"
    try:
        from app.services.ingestion_service import get_vector_store
        get_vector_store()
    except Exception as exc:
        vector_status = f"error: {exc}"

    return HealthResponse(
        status="ok" if db_status == "ok" and vector_status == "ok" else "degraded",
        database=db_status,
        vector_store=vector_status,
        embedding_model=settings.EMBEDDING_MODEL,
        reranker_model=settings.RERANKER_MODEL,
        llm_provider=settings.LLM_PROVIDER,
    )
