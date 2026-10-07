"""
Text embeddings.

ML CONCEPT: Sentence embeddings / semantic similarity
WHAT: Converts a piece of text into a fixed-length dense vector (e.g. 384
      floating point numbers for all-MiniLM-L6-v2) using a Sentence
      Transformers bi-encoder model.
WHY:  Two sentences that mean the same thing but use different words
      ("revenue grew" vs "sales increased") produce vectors that are
      close together in the embedding space, even with zero shared
      keywords. This is what lets vector search find semantically
      relevant chunks that a keyword search (BM25) would miss entirely.
HOW:  The model is trained (via contrastive/triplet losses on sentence
      pairs) so that cosine similarity between two vectors approximates
      how semantically similar the two texts are. We embed once at
      ingestion time (documents) and once per request (the query), then
      compare with cosine similarity in the vector store (see
      app/retrieval/vector_store.py).

ADVANTAGES: captures meaning/paraphrase, language-agnostic to some extent,
            fast approximate search at scale.
LIMITATIONS: can miss exact identifiers (ticker symbols, exact figures,
             rare proper nouns) that a plain keyword match would catch --
             this is exactly why we also run BM25 and fuse the two
             (see app/retrieval/hybrid.py).
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.errors import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
    return SentenceTransformer(settings.EMBEDDING_MODEL)


class Embedder:
    """Thin wrapper so the rest of the app depends on an interface, not a library."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = _get_model() if self.model_name == settings.EMBEDDING_MODEL else SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)
        try:
            vectors = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,  # so dot product == cosine similarity
                convert_to_numpy=True,
            )
            return vectors.astype(np.float32)
        except Exception as exc:
            logger.exception("Embedding failed for batch of %d texts", len(texts))
            raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
