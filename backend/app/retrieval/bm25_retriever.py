"""
BM25 keyword retrieval.

ML CONCEPT: BM25 (Best Matching 25)
WHAT: A classical, non-neural ranking function that scores documents by
      term-frequency / inverse-document-frequency (TF-IDF-like), with
      length normalization. Given a query, it scores every document by
      how much and how distinctively the query's exact terms appear.
WHY:  Dense vector search learns "meaning" but can genuinely miss exact
      matches -- a query like "what was NVDA's Q3 FY2025 gross margin"
      needs the literal tokens "Q3", "FY2025", "gross margin" to line up
      with a specific table row, and embeddings sometimes rank a
      thematically-similar-but-wrong paragraph above the exact one. BM25
      guarantees exact-term matches are never invisible to retrieval.
HOW:  We build one in-memory BM25 index per running process (rebuilt
      whenever a document is (re)indexed) over all chunk texts, tokenized
      with a simple lowercase + regex word-splitter. `rank_bm25`'s
      BM25Okapi implements the standard Okapi BM25 formula:

          score(D, Q) = sum_i IDF(q_i) * ( f(q_i, D) * (k1 + 1) )
                              / ( f(q_i, D) + k1 * (1 - b + b * |D| / avgdl) )

      where f(q_i, D) is term frequency, |D| is document length, avgdl is
      average document length, and k1/b are tuning constants (rank_bm25
      defaults: k1=1.5, b=0.75).

LIMITATIONS: no understanding of synonyms/paraphrase (misses "sales grew"
             when queried with "revenue increased") -- this is exactly
             what vector search complements it with. See hybrid.py.
"""
import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Hit:
    chunk_id: str
    score: float


@dataclass
class BM25Index:
    """
    In-memory BM25 index over all indexed chunks.

    NOTE: For a portfolio-scale project (dozens to low-hundreds of
    documents) an in-memory index rebuilt on ingestion is simple and fast
    enough. docs/deployment.md discusses how this would be swapped for a
    persistent inverted index (e.g. Elasticsearch/OpenSearch/Tantivy) to
    scale past that -- see also interview_questions.md, "how would you
    handle 1 million documents?".
    """
    chunk_ids: list[str] = field(default_factory=list)
    _document_map: dict[str, set[str]] = field(default_factory=dict)  # document_id -> chunk_ids
    _bm25: BM25Okapi | None = field(default=None, repr=False)

    def build(self, chunk_id_to_text: dict[str, str], chunk_id_to_document: dict[str, str]) -> None:
        self.chunk_ids = list(chunk_id_to_text.keys())
        corpus = [_tokenize(chunk_id_to_text[cid]) for cid in self.chunk_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._document_map = {}
        for cid in self.chunk_ids:
            doc_id = chunk_id_to_document[cid]
            self._document_map.setdefault(doc_id, set()).add(cid)
        logger.info("BM25 index built over %d chunks", len(self.chunk_ids))

    def search(
        self, query: str, top_k: int, document_ids: list[str] | None = None
    ) -> list[BM25Hit]:
        if not self._bm25 or not self.chunk_ids:
            return []

        scores = self._bm25.get_scores(_tokenize(query))

        allowed: set[str] | None = None
        if document_ids:
            allowed = set()
            for d in document_ids:
                allowed |= self._document_map.get(d, set())

        scored = [
            (self.chunk_ids[i], float(scores[i]))
            for i in range(len(self.chunk_ids))
            if scores[i] > 0 and (allowed is None or self.chunk_ids[i] in allowed)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [BM25Hit(chunk_id=cid, score=s) for cid, s in scored[:top_k]]

    @property
    def is_empty(self) -> bool:
        return self._bm25 is None
