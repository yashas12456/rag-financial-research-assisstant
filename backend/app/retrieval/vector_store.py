"""
Vector store: Qdrant.

WHY QDRANT OVER pgvector (see docs/architecture.md for full comparison):
  - Qdrant is a purpose-built vector database with native filtering
    (by document_id, page_number, etc.) alongside ANN search, payload
    storage co-located with vectors, and a straightforward Python client.
  - pgvector is a fine choice too (one fewer moving part, everything in
    Postgres) but at portfolio/demo scale we prioritized showing a
    dedicated vector-DB integration since that's a very common
    production pattern and a common interview topic ("why not just use
    Postgres for everything?").
  - The retrieval interface below (`VectorStore`) is small and abstract
    enough that swapping in a pgvector-backed implementation later would
    only require changing this one file.

ML CONCEPT: Vector search / cosine similarity / nearest-neighbor search
WHAT: Given a query vector, find the K stored vectors with the highest
      cosine similarity (semantic closeness) to it.
WHY:  This is the "semantic" half of retrieval -- it finds chunks that
      mean the same thing as the question even without shared keywords.
HOW:  Qdrant indexes vectors with an HNSW (approximate nearest neighbor)
      graph so search is sub-linear in the number of stored vectors.
      Because embeddings are L2-normalized (see embedder.py), cosine
      similarity is equivalent to a dot product, which is what Qdrant's
      "Cosine" distance metric computes directly.
"""
import uuid
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings
from app.core.errors import VectorStoreError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class VectorHit:
    chunk_id: str
    score: float  # cosine similarity, higher is better


class VectorStore:
    def __init__(self):
        self._client = self._init_client()
        self._collection = settings.QDRANT_COLLECTION
        self._ensure_collection()

    def _init_client(self) -> QdrantClient:
        from pathlib import Path
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        default_storage = str(root_dir / "storage")

        if settings.QDRANT_URL and (settings.QDRANT_URL.startswith("http://") or settings.QDRANT_URL.startswith("https://")):
            try:
                client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=2.0)
                client.get_collections()
                return client
            except Exception as exc:
                logger.warning(
                    "Could not connect to Qdrant server at %s (%s). Falling back to embedded local storage at %s",
                    settings.QDRANT_URL, exc, default_storage
                )
                return QdrantClient(path=default_storage)
        elif settings.QDRANT_URL:
            return QdrantClient(path=settings.QDRANT_URL)
        return QdrantClient(path=default_storage)

    def _ensure_collection(self) -> None:
        try:
            existing = [c.name for c in self._client.get_collections().collections]
            if self._collection not in existing:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=qmodels.VectorParams(
                        size=settings.EMBEDDING_DIM, distance=qmodels.Distance.COSINE
                    ),
                )
                logger.info("Created Qdrant collection '%s'", self._collection)
        except Exception as exc:
            raise VectorStoreError(f"Could not connect to / initialize Qdrant: {exc}") from exc

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict],
    ) -> None:
        if len(chunk_ids) == 0:
            return
        try:
            points = [
                qmodels.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_ids[i])),
                    vector=vectors[i].tolist(),
                    payload={**payloads[i], "chunk_id": chunk_ids[i]},
                )
                for i in range(len(chunk_ids))
            ]
            self._client.upsert(collection_name=self._collection, points=points, wait=True)
        except Exception as exc:
            logger.exception("Failed to upsert %d vectors", len(chunk_ids))
            raise VectorStoreError(f"Failed to store embeddings: {exc}") from exc

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        document_ids: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorHit]:
        query_filter = None
        if document_ids:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id", match=qmodels.MatchAny(any=document_ids)
                    )
                ]
            )
        try:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector.tolist(),
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold if score_threshold and score_threshold > 0 else None,
            )
        except Exception as exc:
            logger.exception("Vector search failed")
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        return [VectorHit(chunk_id=r.payload["chunk_id"], score=r.score) for r in results]

    def delete_document(self, document_id: str) -> None:
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id", match=qmodels.MatchValue(value=document_id)
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            logger.exception("Failed to delete vectors for document %s", document_id)
            raise VectorStoreError(f"Failed to delete document vectors: {exc}") from exc
