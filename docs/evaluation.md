# Evaluation Guide

This project ships with a **working evaluation harness**, not precomputed
results. Per the project requirements, we never fabricate accuracy,
Recall@K, MRR, latency, or any other metric — every number this system
reports is computed live against whatever documents and questions you
actually provide.

## Why this matters

It's easy to build a RAG demo that *looks* impressive on a couple of
manually-checked examples. It's much more valuable — and much more
representative of real engineering work — to have an evaluation harness
that tells you, quantitatively, whether hybrid retrieval and reranking
are actually helping, and by how much, on YOUR documents.

## Step 1: Index real documents

```bash
# Option A: via the UI (Upload page), or
# Option B: bulk-ingest a folder of PDFs
python scripts/ingest.py --folder data/sample_documents
```

Add real financial PDFs (annual reports, 10-Ks, earnings releases) to
`data/sample_documents/` first. See docs/deployment.md for where to find
public ones (SEC EDGAR, investor relations pages).

## Step 2: Build a real evaluation dataset

Edit `data/evaluation/questions.json`. Replace the shipped placeholder
entries with real questions about your indexed documents:

```json
[
  {
    "question": "What was total net revenue in fiscal year 2025?",
    "expected_answer": "Net revenue was $X billion, an increase of Y% year over year.",
    "document": "acme_corp_annual_report_2025.pdf",
    "page": 34,
    "expected_evidence_substring": "net revenue"
  }
]
```

`document` and `page` must be the **real filename and real page number**
where you personally verified the answer appears — this is your labeled
ground truth for computing Recall@K / MRR / Hit Rate. The spec calls for
50–100 such questions across several documents for a thorough evaluation;
start with 15–20 to sanity-check the pipeline, then expand.

## Step 3: Run the evaluation

```bash
python scripts/evaluate.py --dataset data/evaluation/questions.json --k 5
```

This runs three baselines against your real dataset:

| Method | What it tests |
|---|---|
| `vector_only` | Dense vector search alone (Baseline 1) |
| `vector_rerank` | Vector search + cross-encoder reranking (Baseline 2) |
| `hybrid_rerank` | BM25 + vector fused with RRF, then reranked (Baseline 3 — full pipeline) |

Output: a printed comparison table plus
`data/evaluation/last_run_results.json` with the full per-question
breakdown (useful for spotting exactly which questions each method got
wrong).

For a finer-grained breakdown (isolating BM25-alone and the effect of
reranking on top of hybrid), run:

```bash
python scripts/benchmark.py --dataset data/evaluation/questions.json --k 5
```

which additionally reports `bm25_only` and `hybrid` (no rerank).

## Step 4 (optional): grade answer quality

Recall@K/MRR/Hit Rate/latency are computed automatically because they only
require comparing retrieved page numbers to your ground truth. **Answer
accuracy, citation accuracy, and faithfulness/groundedness require
judgment** — either:

1. **Manual grading**: read each generated answer in
   `data/evaluation/last_run_results.json` against your `expected_answer`,
   and record correct/incorrect yourself.
2. **LLM-as-judge**: implement
   `scripts/evaluate.py::grade_answer_with_llm` (a `NotImplementedError`
   stub is provided) to call an LLM that compares the generated answer +
   citations against your expected answer, using your own `LLM_API_KEY`.

We deliberately did not wire up a default LLM-judge implementation and
call it "faithfulness: 0.94" without you running it — that would be
exactly the kind of fabricated metric the project spec prohibits.

## Interpreting results

- If `hybrid_rerank`'s Recall@K/MRR is *not* clearly better than
  `vector_only`'s on your dataset, that's a legitimate and useful finding
  — it likely means your questions don't have much exact-keyword
  dependency (BM25 helps most on rare terms, tickers, exact figures) or
  your document set is small enough that vector search alone already
  finds everything. Try adding questions that reference specific numbers,
  dates, or named entities verbatim to see hybrid retrieval's advantage
  more clearly.
- Latency differences between methods mostly come from the reranker
  (`vector_rerank`/`hybrid_rerank` run a cross-encoder forward pass per
  candidate) — expect reranked methods to be slower but more accurate.
