"""
Pydantic request/response schemas used by the API layer.

These are intentionally decoupled from the SQLAlchemy models (app/models)
so the API contract can evolve independently of the DB schema.
"""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# ---------- Documents ----------

class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    status: str
    num_pages: int
    file_size_bytes: int
    company: str | None = None
    fiscal_year: str | None = None
    error_message: str | None = None
    created_at: datetime
    indexed_at: datetime | None = None

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    document: DocumentResponse
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class IndexDocumentResponse(BaseModel):
    document_id: UUID
    num_chunks: int
    status: str


# ---------- Query / Chat ----------

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[UUID] | None = Field(
        default=None, description="Optional subset of documents to restrict retrieval to"
    )
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    page_number: int
    section: str | None
    text: str
    vector_score: float | None
    bm25_score: float | None
    rrf_score: float | None
    rerank_score: float | None
    rank: int


class Citation(BaseModel):
    marker: str  # e.g. "[1]"
    filename: str
    page_number: int
    chunk_id: UUID
    rerank_score: float | None


class QueryResponse(BaseModel):
    query_id: UUID
    question: str
    answer: str
    insufficient_evidence: bool
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    latency_ms: float


# ---------- Evaluation ----------

class EvaluationRunRequest(BaseModel):
    methods: list[str] = Field(
        default_factory=lambda: ["vector_only", "vector_rerank", "hybrid_rerank"],
        description="Which pipeline configurations to benchmark",
    )
    dataset_path: str = Field(default="data/evaluation/questions.json")
    k: int = Field(default=5, ge=1, le=50)


class EvaluationMethodResult(BaseModel):
    method_name: str
    num_questions: int
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr: float | None = None
    hit_rate: float | None = None
    answer_accuracy: float | None = None
    citation_accuracy: float | None = None
    faithfulness: float | None = None
    hallucination_rate: float | None = None
    avg_latency_ms: float | None = None


class EvaluationRunResponse(BaseModel):
    id: UUID
    results: list[EvaluationMethodResult]
    created_at: datetime


# ---------- Health ----------

class HealthResponse(BaseModel):
    status: str
    database: str
    vector_store: str
    embedding_model: str
    reranker_model: str
    llm_provider: str
