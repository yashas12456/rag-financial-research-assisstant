"""
Database models.

Tables:
  documents          - one row per uploaded PDF
  document_pages     - one row per extracted page (raw text + page number)
  document_chunks     - one row per chunk fed into embeddings/BM25 index
  queries             - one row per user question asked
  retrieval_results   - one row per (query, chunk) retrieved, with scores
  evaluations         - one row per evaluation run (aggregate metrics)
"""
import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _uuid_col():
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_col()
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    # SHA-256 of file bytes -> used for duplicate detection (see ingestion/pdf_parser.py)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    num_pages: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus), default=DocumentStatus.UPLOADED, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    company: Mapped[str | None] = mapped_column(String(255), nullable=True)
    fiscal_year: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_documents_status", "status"),)


class DocumentPage(Base):
    __tablename__ = "document_pages"

    id: Mapped[uuid.UUID] = _uuid_col()
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_empty: Mapped[bool] = mapped_column(Boolean, default=False)  # e.g. scanned/no extractable text

    document: Mapped["Document"] = relationship(back_populates="pages")

    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_document_page"),
        Index("ix_document_pages_document_id", "document_id"),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = _uuid_col()
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)  # order within document
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Qdrant point id (string form of a uuid) - the vector itself lives in Qdrant, not Postgres.
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_document_chunks_document_id", "document_id"),
        Index("ix_document_chunks_page_number", "page_number"),
    )


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = _uuid_col()
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_ids: Mapped[list] = mapped_column(JSON, default=list)  # scope of the query, if filtered
    insufficient_evidence: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    retrieval_results: Mapped[list["RetrievalResult"]] = relationship(
        back_populates="query", cascade="all, delete-orphan"
    )


class RetrievalResult(Base):
    """One retrieved-and-scored chunk for a given query, kept for citations + evaluation."""

    __tablename__ = "retrieval_results"

    id: Mapped[uuid.UUID] = _uuid_col()
    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("queries.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)  # final rank after reranking
    vector_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    bm25_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rrf_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rerank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    used_in_answer: Mapped[bool] = mapped_column(Boolean, default=False)

    query: Mapped["Query"] = relationship(back_populates="retrieval_results")
    chunk: Mapped["DocumentChunk"] = relationship()

    __table_args__ = (Index("ix_retrieval_results_query_id", "query_id"),)


class EvaluationRun(Base):
    """Aggregate results of one evaluation/benchmark script execution."""

    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = _uuid_col()
    method_name: Mapped[str] = mapped_column(String(128), nullable=False)  # e.g. "hybrid+rerank"
    num_questions: Mapped[int] = mapped_column(Integer, nullable=False)
    recall_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    mrr: Mapped[float | None] = mapped_column(Float, nullable=True)
    hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    hallucination_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_results: Mapped[dict] = mapped_column(JSON, default=dict)  # per-question breakdown
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
