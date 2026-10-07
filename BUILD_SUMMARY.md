# BUILD SUMMARY

## What was built

A complete, production-style domain-specific RAG (Retrieval-Augmented
Generation) research assistant for financial documents:

- **Backend**: FastAPI + PostgreSQL + Qdrant, implementing PDF ingestion,
  configurable chunking, Sentence-Transformer embeddings, BM25 keyword
  retrieval, hybrid fusion (Reciprocal Rank Fusion), cross-encoder
  reranking, and LLM answer generation (Anthropic/OpenAI, pluggable) with
  anti-hallucination prompting and citation grounding.
- **Frontend**: Next.js/TypeScript dashboard with Dashboard, Documents,
  Upload, Research Chat (with expandable per-source score inspection),
  Evaluation, and Settings pages.
- **Evaluation system**: a real evaluation harness
  (`scripts/evaluate.py`, `scripts/benchmark.py`, and the underlying
  metrics in `app/services/evaluation_service.py`) computing Recall@K,
  Precision@K, MRR, Hit Rate, and latency against a labeled question set,
  with three/five-way baseline comparisons (vector-only vs. vector+rerank
  vs. BM25-only vs. hybrid vs. hybrid+rerank).
- **DevOps**: Dockerfiles for backend and frontend, `docker-compose.yml`
  wiring Postgres + Qdrant + backend + frontend with health checks and
  correct startup ordering, `.env.example`, `.gitignore`.
- **Tests**: `backend/tests/` — unit tests for PDF parsing, chunking,
  BM25 retrieval, RRF fusion, evaluation metric math, and API integration
  tests (with heavy ML dependencies mocked for fast, dependency-light
  execution).
- **Documentation**: `docs/architecture.md` (Mermaid diagrams + design
  rationale), `docs/ml_concepts.md` (every ML/NLP concept explained in
  depth), `docs/evaluation.md` (how to actually run evaluation on your own
  data), `docs/deployment.md` (local/Docker setup, scaling, security),
  `docs/interview_questions.md` (31 Q&A).

## Folder structure

```
project-2-rag-financial-research-assistant/
├── backend/
│   ├── app/
│   │   ├── api/            # documents.py, query.py, evaluation.py, health.py
│   │   ├── core/           # config.py, logging.py, errors.py
│   │   ├── db/             # session.py
│   │   ├── models/         # models.py (SQLAlchemy ORM)
│   │   ├── schemas/        # schemas.py (Pydantic)
│   │   ├── services/       # ingestion_service.py, query_service.py, evaluation_service.py
│   │   ├── retrieval/      # vector_store.py, bm25_retriever.py, hybrid.py
│   │   ├── embeddings/     # embedder.py
│   │   ├── reranking/      # cross_encoder.py
│   │   ├── llm/            # client.py
│   │   ├── ingestion/      # pdf_parser.py, chunking.py
│   │   └── main.py
│   ├── tests/               # 7 test files
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/                 # dashboard, documents, upload, chat, evaluation, settings
│   ├── components/          # Sidebar, Card, StatusBadge, CitationPanel
│   ├── services/            # api.ts
│   ├── types/                # index.ts
│   ├── package.json
│   └── Dockerfile
├── data/
│   ├── sample_documents/    # put your PDFs here
│   └── evaluation/          # questions.json (labeled eval set)
├── scripts/
│   ├── ingest.py
│   ├── evaluate.py
│   └── benchmark.py
├── docs/
│   ├── architecture.md
│   ├── ml_concepts.md
│   ├── evaluation.md
│   ├── deployment.md
│   └── interview_questions.md
├── docker-compose.yml
├── .env.example
├── README.md
└── BUILD_SUMMARY.md (this file)
```

## How to install

```bash
cp .env.example .env
# Edit .env: set LLM_API_KEY and LLM_PROVIDER
docker compose up --build
```

## How to run

- UI: http://localhost:3000
- API docs: http://localhost:8000/docs
- Upload a PDF via the Upload page (or `python scripts/ingest.py`), click
  Index, then ask questions in Research Chat.

## Environment variables

See `.env.example` — chunking (`CHUNK_SIZE`/`CHUNK_OVERLAP`), retrieval
(`VECTOR_TOP_K`/`BM25_TOP_K`/`RRF_K`/`RERANK_TOP_K`/`SIMILARITY_THRESHOLD`),
models (`EMBEDDING_MODEL`/`RERANKER_MODEL`/`LLM_MODEL`), and infra
(`DATABASE_URL`/`QDRANT_URL`/`LLM_API_KEY`) are all configurable there.

## ML concepts used — and exactly where

See `docs/ml_concepts.md` for full explanations. Quick index:

```
NLP              → app/ingestion/, app/embeddings/, app/retrieval/, app/llm/ (general)
Text Chunking    → backend/app/ingestion/chunking.py
Embeddings       → backend/app/embeddings/embedder.py
Cosine Similarity→ backend/app/embeddings/embedder.py (normalization) + app/retrieval/vector_store.py (Qdrant COSINE metric)
Vector Search    → backend/app/retrieval/vector_store.py
BM25             → backend/app/retrieval/bm25_retriever.py
Hybrid Search    → backend/app/services/query_service.py (orchestration)
RRF              → backend/app/retrieval/hybrid.py
Reranking        → backend/app/reranking/cross_encoder.py
RAG (full loop)  → backend/app/services/query_service.py::run_rag_pipeline
LLM              → backend/app/llm/client.py
Evaluation       → backend/app/services/evaluation_service.py + scripts/evaluate.py + scripts/benchmark.py
```

## Testing procedure

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

**Note on this build environment specifically**: this codebase was
written and syntax-checked (`python -m py_compile` on every file) in a
sandbox with no network access, so `pip install`/`npm install` could not
be run here and the test suite has **not** been executed end-to-end in
this environment. Every file passed static syntax checking. You should
run `pytest tests/ -v` yourself once dependencies are installed to confirm
runtime behavior, especially around library-version-specific APIs
(Qdrant client, Sentence Transformers, SQLAlchemy 2.0 style).

## Evaluation procedure

See `docs/evaluation.md`. Summary: index real documents → build a labeled
`data/evaluation/questions.json` with the real filename/page for each
question's answer → run `python scripts/evaluate.py` or
`python scripts/benchmark.py`.

## Docker instructions

See `docs/deployment.md` and `docker-compose.yml`. All four services
(Postgres, Qdrant, backend, frontend) have health checks; the backend
waits for Postgres and Qdrant to report healthy before starting.

## Known limitations

- BM25 index is in-memory, rebuilt per ingestion (fine at portfolio
  scale, not at millions of documents — see docs/deployment.md).
- PDF table extraction is linearized text, not structured tables.
- Answer-quality metrics (accuracy/citation accuracy/faithfulness)
  require manual or LLM-as-judge grading — not auto-computed.
- No multi-tenant auth; single-user demo/portfolio scope as shipped.
- This build environment had no network access, so `npm install` /
  `pip install` / Docker builds / the pytest suite / a live LLM call have
  **not** been executed here. Code was written carefully and syntax
  checked, but you should validate the runtime behavior yourself before
  treating this as fully verified — see "Testing procedure" above.

## Future improvements

See README.md's "Future Improvements" section: persistent search index at
scale, structured table extraction, LLM-as-judge grading in CI, streaming
responses, multi-tenant auth, semantic query caching.

## Interview explanation

See `docs/interview_questions.md` for 31 Q&A covering every major design
decision (why hybrid search, why RRF, why cross-encoder reranking, how
citations are guaranteed non-hallucinated, how this would scale, etc.).

## ML Concept → File → Function/Class → Purpose table (mandatory)

| ML Concept | File | Function/Class | Purpose |
|---|---|---|---|
| PDF text extraction | `backend/app/ingestion/pdf_parser.py` | `extract_pdf` | Page-level text extraction with metadata, powering all downstream citations |
| Text Chunking | `backend/app/ingestion/chunking.py` | `chunk_document`, `chunk_text` | Splits page text into overlapping, metadata-tagged chunks for embedding |
| Embeddings | `backend/app/embeddings/embedder.py` | `Embedder.embed_texts` / `embed_query` | Converts text to dense vectors via Sentence Transformers |
| Cosine Similarity | `backend/app/embeddings/embedder.py` + `backend/app/retrieval/vector_store.py` | `normalize_embeddings=True`; `Distance.COSINE` | Normalizes vectors so dot product = cosine similarity; Qdrant's similarity metric |
| Vector Search | `backend/app/retrieval/vector_store.py` | `VectorStore.search` | Dense semantic nearest-neighbor retrieval via Qdrant/HNSW |
| BM25 | `backend/app/retrieval/bm25_retriever.py` | `BM25Index.build` / `search` | Keyword-based ranked retrieval via Okapi BM25 |
| Hybrid Search | `backend/app/services/query_service.py` | `run_rag_pipeline` | Orchestrates running vector + BM25 retrieval together |
| RRF | `backend/app/retrieval/hybrid.py` | `reciprocal_rank_fusion` | Merges vector + BM25 ranked lists into one fused ranking |
| Reranking | `backend/app/reranking/cross_encoder.py` | `Reranker.rerank` | Cross-encoder second-stage precision reranking of fused candidates |
| RAG (end-to-end) | `backend/app/services/query_service.py` | `run_rag_pipeline` | Full retrieve → fuse → rerank → generate → cite pipeline |
| LLM / Prompt Engineering / Anti-hallucination | `backend/app/llm/client.py` | `LLMClient.answer_question`, `SYSTEM_PROMPT` | Grounded answer generation with citation and insufficient-evidence handling |
| Evaluation (Retrieval) | `backend/app/services/evaluation_service.py` | `recall_at_k`, `precision_at_k`, `mean_reciprocal_rank`, `hit_rate_at_k` | Computes standard IR metrics against labeled ground truth |
| Evaluation (Orchestration) | `scripts/evaluate.py`, `scripts/benchmark.py` | `run_all_methods`, `benchmark_one` | Runs baseline comparisons and prints/saves real measured results |
