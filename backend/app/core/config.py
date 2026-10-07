"""
Centralized application configuration.

WHAT: Uses pydantic-settings to load configuration from environment
      variables (and a local .env file) into a typed, validated object.
WHY:  Hardcoding config (DB URLs, model names, retrieval parameters) is a
      production anti-pattern. Every tunable knob in the RAG pipeline
      (chunk size, top_k, rerank_top_k, thresholds) lives here so it can
      be changed without touching code, and so experiments (baseline vs
      hybrid vs reranked) are reproducible and documented.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    APP_NAME: str = "RAG Financial Research Assistant"
    ENV: Literal["development", "test", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    API_V1_PREFIX: str = "/api"

    # --- Database ---
    DATABASE_URL: str = Field(
        default="postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db"
    )

    # --- Vector store (Qdrant) ---
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "document_chunks"

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384  # must match EMBEDDING_MODEL's output dimension

    # --- Reranking ---
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_ENABLED: bool = True

    # --- LLM ---
    LLM_PROVIDER: Literal["anthropic", "openai"] = "anthropic"
    LLM_API_KEY: str | None = None
    LLM_MODEL: str = "claude-sonnet-4-6"
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.0

    # --- Chunking ---
    CHUNK_SIZE: int = 800          # characters per chunk (config'd, see docs/ml_concepts.md)
    CHUNK_OVERLAP: int = 150       # characters of overlap between consecutive chunks

    # --- Retrieval ---
    VECTOR_TOP_K: int = 20
    BM25_TOP_K: int = 20
    RRF_K: int = 60                # RRF damping constant (standard default from literature)
    RERANK_TOP_K: int = 5          # chunks sent to the LLM after reranking
    SIMILARITY_THRESHOLD: float = 0.0  # min cosine similarity to keep a vector hit (0 disables filtering)

    # --- Uploads ---
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_UPLOAD_EXTENSIONS: tuple[str, ...] = (".pdf",)
    UPLOAD_DIR: str = "./data/sample_documents"

    # --- Rate limiting (extension point, see docs/deployment.md) ---
    RATE_LIMIT_PER_MINUTE: int = 60


@lru_cache
def get_settings() -> Settings:
    """Settings are cached so the .env file is parsed only once per process."""
    return Settings()
