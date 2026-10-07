# RAG Financial Research Assistant

A production-style, domain-specific Retrieval-Augmented Generation (RAG)
system that lets you upload financial documents (annual reports, earnings
releases, investor presentations) and ask questions about them, with every
answer grounded in cited, page-accurate evidence.

```
Documents → Parser → Chunking → Embeddings → Vector DB → Hybrid Retrieval
          → Reranker → LLM → Answer + Citations
```

## Problem Statement

Financial analysts spend hours manually searching through hundred-page PDF
reports to answer specific questions ("what was the gross margin trend?",
"what risk factors did they disclose about supply chains?"). Generic LLM
chatbots either don't have access to the specific document, or — worse —
confidently answer from memorized (and possibly outdated or wrong) training
data instead of the actual filing in front of them.

## Solution

This system builds a proper retrieval pipeline over your uploaded
documents — combining dense vector search (semantic meaning) with BM25
keyword search (exact terms), fusing both with Reciprocal Rank Fusion, and
reranking the fused candidates with a cross-encoder before ever calling the
LLM. The LLM is instructed, and structurally constrained, to answer only
from that retrieved evidence and to say so explicitly when the documents
don't contain enough information — with every claim traceable to a real
filename and page number.

## Why This Is Different From a Basic "Chat with PDF" App

| Basic PDF chatbot | This project |
|---|---|
| Single retrieval method (usually vector-only) | Hybrid: BM25 + vector search fused with RRF |
| No reranking | Cross-encoder reranking as a dedicated second stage |
| Vague or absent citations | Every claim traces to filename + exact page number, from real PDF extraction metadata |
| No measured performance | A real evaluation harness (Recall@K, Precision@K, MRR, Hit Rate, latency) run against your own labeled question set — no fabricated numbers |
| One "does it work" test | Baseline comparison: vector-only vs. vector+rerank vs. hybrid+rerank |
| Minimal error handling | Explicit handling of corrupt PDFs, scanned/empty PDFs, duplicates, oversized files, LLM/vector-store failures |

## Features

- PDF upload with page-level text extraction, duplicate detection (SHA-256
  content hashing), and clear error messages for corrupt/empty/scanned PDFs
- Configurable chunking (size, overlap) with document/page/section
  metadata preserved end-to-end
- Sentence-Transformer embeddings + Qdrant vector search
- BM25 keyword retrieval running alongside vector search
- Hybrid fusion via Reciprocal Rank Fusion
- Cross-encoder reranking as a dedicated second retrieval stage
- LLM answer generation (Anthropic or OpenAI, pluggable) with an
  anti-hallucination system prompt and explicit "insufficient evidence"
  handling
- Full source citations (document + page + retrieved text + retrieval and
  rerank scores), clickable/expandable in the UI
- A real evaluation system: Recall@K, Precision@K, MRR, Hit Rate, latency,
  computed against a labeled question set you build for your own documents
- Baseline comparison scripts (`evaluate.py`, `benchmark.py`)
- Professional dashboard: document management, upload, research chat,
  citation inspection, evaluation dashboard, system settings
- Dockerized, with health checks on every service

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for detailed Mermaid
diagrams of the ingestion and query pipelines, and the reasoning behind
using Qdrant + hybrid retrieval + reranking.

## ML Concepts Used

See [`docs/ml_concepts.md`](docs/ml_concepts.md) for a full explanation
(what/why/where/input/output/math/limitations) of every ML/NLP concept:
tokenization, chunking, embeddings, cosine similarity, vector search,
BM25, hybrid search, RRF, cross-encoder reranking, RAG, prompt engineering,
hallucination mitigation, and retrieval evaluation metrics.

## Technology Stack

**Backend:** Python, FastAPI, Pydantic, SQLAlchemy, PostgreSQL, Qdrant,
Sentence Transformers, `rank_bm25`, PyMuPDF, Anthropic/OpenAI SDKs, pytest

**Frontend:** Next.js (App Router), React, TypeScript, Tailwind CSS

**DevOps:** Docker, Docker Compose, health checks

## Installation

```bash
git clone <this-repo>
cd project-2-rag-financial-research-assistant
cp .env.example .env
# Edit .env: set LLM_API_KEY and LLM_PROVIDER (anthropic or openai)
```

## Environment Variables

See [`.env.example`](.env.example) for the full list. At minimum, set
`LLM_API_KEY` and `LLM_PROVIDER`.

## Database Setup

Handled automatically on backend startup (`Base.metadata.create_all()`)
for local/demo use. For production, switch to Alembic migrations — see
[`docs/deployment.md`](docs/deployment.md).

## Running Locally

**With Docker (recommended):**
```bash
docker compose up --build
```
- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard

**Without Docker:** see [`docs/deployment.md`](docs/deployment.md) for
step-by-step backend/frontend setup.

## Docker Setup

See [`docker-compose.yml`](docker-compose.yml): Postgres, Qdrant, backend,
and frontend, each with health checks and correct startup ordering via
`depends_on: condition: service_healthy`.

## API Documentation

FastAPI auto-generates interactive docs at `/docs` (Swagger UI) and
`/redoc` once the backend is running. Key endpoints:

```
POST   /api/documents/upload
GET    /api/documents
GET    /api/documents/{id}
DELETE /api/documents/{id}
POST   /api/documents/{id}/index
POST   /api/query
GET    /api/query/{id}
POST   /api/evaluation/run
GET    /api/evaluation/results
GET    /api/health
```

## Evaluation

See [`docs/evaluation.md`](docs/evaluation.md) for the full walkthrough:
index real documents, build a labeled question set, then run
`python scripts/evaluate.py` or `python scripts/benchmark.py`. **This
project does not ship with precomputed benchmark numbers** — see below.

## Benchmark Results

**Not included by design.** Per the project requirements, we never
fabricate accuracy, Recall@K, MRR, or latency numbers. This repository
ships with a placeholder `data/evaluation/questions.json` and a fully
working evaluation harness — running it against your own uploaded
documents and your own labeled questions will produce real numbers
specific to your data and your `LLM_API_KEY`. See
[`docs/evaluation.md`](docs/evaluation.md) for exactly how to do this, and
[`BUILD_SUMMARY.md`](BUILD_SUMMARY.md) for the current status of what has
and hasn't been run in this environment.

## Screenshots

_Add screenshots here after running the app locally — e.g. the dashboard,
upload flow, research chat with expanded citations, and evaluation
dashboard._

## Project Demo

_Add a link to a recorded walkthrough or hosted demo here._

## Limitations

- In-memory BM25 index is rebuilt on every ingestion — fine for
  dozens/hundreds of documents, not millions (see docs/deployment.md).
- PDF table extraction is linearized text, not structured table data —
  numeric lookups inside dense tables may be less reliable than prose.
- Answer-quality metrics (accuracy, citation accuracy, faithfulness)
  require manual or LLM-as-judge grading; they are not auto-computed out
  of the box (see docs/evaluation.md).
- No authentication/multi-tenancy — this is a single-user portfolio/demo
  system, not a multi-tenant SaaS product as shipped.
- Rate limiting is defined as a config value but not yet enforced in
  middleware (documented extension point).

## Future Improvements

- Persistent search index (Elasticsearch/OpenSearch) for BM25 at scale
- Structured table extraction for financial statements
- LLM-as-judge grading wired into CI for continuous answer-quality tracking
- Streaming LLM responses in the chat UI
- Multi-tenant auth and per-user document scoping
- Semantic caching of repeated/similar queries

## Interview Explanation

See [`docs/interview_questions.md`](docs/interview_questions.md) for 30+
Q&A covering every design decision in this project, pitched at a level
useful for technical interviews.

## Resume Bullet Points

- Built a production-style hybrid RAG system (BM25 + dense vector search
  fused via Reciprocal Rank Fusion, cross-encoder reranking) over
  financial PDF documents, with FastAPI, PostgreSQL, and Qdrant
- Designed an anti-hallucination LLM pipeline with citation-grounded
  answer generation, explicit insufficient-evidence handling, and
  page-accurate source attribution traced to real PDF extraction metadata
- Implemented a quantitative evaluation harness (Recall@K, Precision@K,
  MRR, Hit Rate, latency) comparing baseline vector search against hybrid
  retrieval and reranking configurations
- Built a full-stack Next.js/TypeScript dashboard for document management,
  research chat with expandable citation/score inspection, and an
  evaluation results dashboard
- Containerized the full stack (FastAPI, Next.js, PostgreSQL, Qdrant) with
  Docker Compose, health checks, and environment-based configuration

## License

MIT — see [`LICENSE`](LICENSE).
