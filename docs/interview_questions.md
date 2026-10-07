# Interview Preparation — Q&A

30+ questions an interviewer might ask about this project, with answers
pitched at a level understandable to an engineering student while still
being technically accurate.

---

**1. Why did you use embeddings instead of just keyword search?**
Keyword search (like BM25) only matches literal words. Embeddings capture
*meaning* — "revenue grew" and "sales increased" produce similar vectors
even with zero shared words, so vector search finds paraphrased or
conceptually related content that keyword search would completely miss.

**2. Why use hybrid search instead of picking one method?**
Each method has a distinct failure mode. Vector search can miss exact
identifiers (ticker symbols, precise dollar figures) buried in a
semantically-different-sounding passage. BM25 can miss meaning-equivalent
phrasing. Running both and fusing the results means a chunk only needs to
be found by *either* method to enter the candidate pool — you get the
recall benefits of both.

**3. Why BM25 specifically, and not just add more keyword filters?**
BM25 isn't a naive keyword filter — it's a principled ranking function
that weighs term frequency against how distinctive/rare a term is across
the whole corpus (IDF) and normalizes for document length. That makes it
much better at ranking "which chunk is most about this term" than a
boolean AND/OR filter would be.

**4. Why reranking? Isn't the initial retrieval enough?**
Bi-encoder embeddings and BM25 both score the query and each chunk
somewhat independently — they're fast but approximate. A cross-encoder
looks at the query and a candidate chunk *together* in one model pass,
letting it reason about fine-grained relevance far more accurately. It's
too slow to run over the whole corpus, so we use it as a second pass over
only the ~20-40 candidates the first stage already narrowed down.

**5. What is cosine similarity, concretely?**
It measures the angle between two vectors, ignoring their magnitude:
`cos(θ) = (A·B)/(||A||·||B||)`. Two vectors pointing in the same direction
score close to 1 (very similar); orthogonal vectors score 0; opposite
vectors score -1. Because our embeddings are pre-normalized to unit
length, cosine similarity is just a dot product.

**6. Why not send the entire PDF to the LLM instead of retrieving chunks?**
Three reasons: (1) context window limits — a 200-page annual report may
exceed what fits or what's affordable per query; (2) cost — sending the
whole document on every question is far more expensive than sending 5
relevant chunks; (3) accuracy — LLMs are measurably worse at pinpointing a
specific fact buried in a huge context ("needle in a haystack") than when
given a small, pre-filtered, highly relevant context.

**7. How do you reduce hallucinations?**
Multiple layers: a system prompt that explicitly forbids using outside
knowledge and requires every claim to cite retrieved evidence; an explicit
instruction to say "insufficient information" rather than guess; a
short-circuit in code that returns an "insufficient evidence" response
whenever retrieval returns nothing (never even calling the LLM to guess);
and a faithfulness/groundedness evaluation metric to measure how often
the model still violates this after all of that.

**8. How do you evaluate a RAG system?**
Two separate things need evaluating: retrieval quality (did we find the
right evidence — Recall@K, Precision@K, MRR, Hit Rate) and answer quality
(did the LLM use that evidence correctly — correctness, faithfulness,
citation accuracy). This project's `scripts/evaluate.py` computes the
first set automatically against a labeled question set, and provides a
grading hook for the second set, which requires human or LLM-as-judge
evaluation since "correct" is a judgment call.

**9. What is Recall@K?**
Of all the relevant chunks that exist for a question, what fraction did
you retrieve in your top K results? With one labeled-relevant chunk per
question, it's simply: did the right chunk appear anywhere in the top K,
averaged over all questions.

**10. What is MRR (Mean Reciprocal Rank)?**
The average of `1/rank` of the first relevant result, across all
questions. It differs from Recall@K by rewarding *where* in the ranking
the right answer landed — finding it at rank 1 scores 1.0; finding it at
rank 5 scores 0.2, even though both would count as a "hit" under
Recall@5.

**11. What is RRF (Reciprocal Rank Fusion)?**
A way to merge two ranked lists (from vector search and BM25) using only
each item's *rank position* in each list, not its raw score. This avoids
the problem that cosine similarity and BM25 scores live on totally
different, incomparable numeric scales — rank position is comparable
across any ranking method.

**12. Why use a cross-encoder specifically for reranking, not just a bigger embedding model?**
A bigger bi-encoder still encodes query and passage separately and
compares vectors — it can't model interactions between specific words in
the query and specific words in the passage. A cross-encoder processes
them jointly through self-attention, which is strictly more expressive for
judging relevance, at the cost of not being pre-computable/indexable.

**13. What happens if retrieval fails (no relevant chunks found)?**
The pipeline detects an empty candidate set after fusion and returns an
explicit "the provided documents do not contain enough information"
response without even calling the LLM — see
`run_rag_pipeline` in `app/services/query_service.py`.

**14. How would you scale this system to more documents?**
The main bottlenecks at scale are (a) the in-memory BM25 index, which
would need to move to a persistent inverted-index engine like
Elasticsearch/OpenSearch, and (b) embedding/reranking compute, which
should move off the API server into a separate worker pool (ideally
GPU-backed) so ingestion spikes don't slow down query latency. Qdrant
itself already supports horizontal sharding for the vector side.

**15. How would you reduce latency?**
Cache embeddings for repeated/similar queries; batch embedding calls;
reduce `RERANK_TOP_K`/candidate-set size (fewer cross-encoder forward
passes, the slowest step); use a smaller/faster reranker model; run
vector search and BM25 concurrently instead of sequentially (they're
independent); consider streaming the LLM response to the frontend instead
of waiting for the full generation.

**16. How would you handle 1 million documents?**
Move BM25 to a dedicated search engine; shard the Qdrant collection;
introduce a metadata pre-filter (e.g., company/fiscal year) before vector
search to shrink the search space per query when the user's question
implies scope; move embedding/reranking to async batch workers with a
job queue; add caching for common queries; consider a coarser first-stage
retrieval (e.g., approximate search with lower recall but much higher
throughput) before the existing pipeline.

**17. Why chunk by characters instead of by tokens?**
Simplicity and zero extra dependency — token-based chunking would require
loading a tokenizer just for chunking. Character-based chunking with a
generous margin below the embedding model's token limit is a common,
pragmatic choice; the trade-off is slightly less precise control over
exactly how many tokens end up in a chunk.

**18. Why store the raw page text in Postgres in addition to chunks?**
It preserves an audit trail back to the original extraction — you can
always verify a citation traces back to real, unmodified page text, and
you could re-chunk with different parameters later without re-parsing the
PDF.

**19. How do you know a citation's page number is accurate and not hallucinated?**
Page numbers are attached to chunks at ingestion time directly from
PyMuPDF's page-by-page extraction (`app/ingestion/pdf_parser.py`) — never
inferred or generated by the LLM. The LLM only ever sees pre-labeled
evidence passages and is instructed to cite by the bracketed marker we
assigned, not to invent its own page numbers.

**20. What's the difference between Precision@K and Recall@K here?**
With one relevant chunk per question, Precision@K = (1 if found else 0)/K,
while Recall@K = (1 if found else 0)/1. They move together in this
single-relevant-item setup but diverge if a question has multiple
relevant chunks — Precision@K would then also penalize retrieving
*additional* irrelevant chunks alongside the right one.

**21. Why is `RERANK_TOP_K` typically much smaller than `VECTOR_TOP_K`/`BM25_TOP_K`?**
The first stage's job is recall (cast a wide net, don't miss the right
answer). The second stage's job is precision (pick the strongest handful
from that net). A small final K keeps the LLM's context focused on the
strongest evidence and reduces cost/latency.

**22. What would you do differently with more time/budget?**
Add persistent BM25 (Elasticsearch), an LLM-as-judge grading pipeline for
answer accuracy/faithfulness wired into CI, streaming answer generation
in the UI, a feedback loop where users can flag wrong answers/citations to
build a fine-tuning or few-shot example set, and semantic caching of
recent queries.

**23. Why Qdrant instead of pgvector, given you're already running Postgres?**
Mainly to demonstrate a dedicated vector-database integration pattern
(native payload filtering + HNSW indexing purpose-built for this),
common in production RAG systems. pgvector is an equally valid choice if
minimizing infrastructure footprint matters more than that — see
docs/architecture.md for the full trade-off discussion.

**24. How do you prevent duplicate documents from being indexed twice?**
Every upload is hashed (SHA-256 of the raw bytes); the hash is stored with
a unique DB constraint, so uploading the exact same file twice is rejected
with a clear "already uploaded" error before any parsing/embedding work
happens.

**25. What happens if a PDF is scanned (no extractable text)?**
`extract_pdf` detects that no page has extractable text and raises an
explicit `EmptyDocumentError` with a message suggesting OCR, rather than
silently indexing an empty/useless document or crashing with an unclear
error.

**26. How is the system prompt designed to reduce hallucination, specifically?**
It explicitly: (1) restricts answers to only the provided evidence, (2)
requires bracketed citations for every claim, (3) instructs the model to
say "insufficient information" when evidence doesn't support an answer,
(4) tells it to distinguish stated fact from inference, and (5) forbids
inventing citations/page numbers beyond what's given.

**27. What's the actual mathematical difference between BM25 and TF-IDF?**
TF-IDF scores grow roughly linearly with term frequency; BM25 saturates
term frequency (diminishing returns for repeated terms via the `k1`
parameter) and normalizes for document length via the `b` parameter,
which empirically ranks better for retrieval tasks than raw TF-IDF.

**28. Why use `k=60` for RRF specifically?**
It's the constant from the original Cormack et al. (2009) paper that
introduced RRF, and has become the de-facto standard default in most
hybrid-search implementations. It controls how much rank-1 vs. rank-10
differ in contribution — smaller `k` more sharply rewards top ranks.

**29. How would you extend this to support tables (common in financial reports)?**
PyMuPDF's plain-text extraction linearizes tables into text, which can
lose column alignment. A more robust approach would add a
table-detection/extraction library (e.g. `pdfplumber`'s table extraction
or `camelot`) as a parallel extraction path, storing detected tables as
structured chunks (e.g. markdown tables) so numeric lookups are more
reliable.

**30. How do you test this system?**
Unit tests cover each pipeline stage in isolation (PDF parsing, chunking,
BM25 retrieval, RRF fusion, evaluation metric math) with synthetic,
fast, dependency-light fixtures; API integration tests mock the
heavy ML dependencies (embedding model, vector store) to test
request validation and error handling without needing a live Qdrant/GPU.
End-to-end evaluation against real documents is a separate, explicit step
(`scripts/evaluate.py`) rather than part of the fast unit test suite.

**31. Why separate `document_chunks` (Postgres) from the vectors (Qdrant)?**
Postgres holds the source of truth for chunk text/metadata plus
relational integrity (foreign keys, cascade deletes); Qdrant holds only
what it needs for fast similarity search. This separation means you can
rebuild/re-embed vectors from Postgres at any time (e.g. after upgrading
the embedding model) without re-parsing PDFs.
