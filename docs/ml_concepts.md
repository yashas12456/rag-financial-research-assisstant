# ML / NLP Concepts Used in This Project

Every concept below states: what it is, why it's used, where it's
implemented, its input/output, the mathematical intuition where useful,
advantages, limitations, and how it affects overall system performance.

---

## 1. NLP (Natural Language Processing)

**What:** The general field of processing and understanding human language
computationally. This project applies it at multiple stages: extracting
structured text from unstructured PDFs, splitting text into semantically
coherent units, converting text to numeric representations, and generating
natural-language answers grounded in retrieved evidence.

**Where:** Threaded through `app/ingestion/`, `app/embeddings/`,
`app/retrieval/`, `app/llm/`.

---

## 2. Tokenization

**What:** Splitting text into discrete units (words, subwords, or
characters) that a model can process.

**Where implemented:**
- `app/retrieval/bm25_retriever.py::_tokenize` — a simple lowercase +
  regex word-splitter (`[a-z0-9]+`) used to build the BM25 inverted index.
- Internally inside the Sentence Transformers and cross-encoder models
  (subword/WordPiece tokenization), which we don't re-implement — the
  `sentence-transformers` library handles this.

**Why:** BM25 needs discrete terms to count frequencies; transformer models
need subword tokens to bound vocabulary size while still handling rare
words. **Limitation:** our BM25 tokenizer is a basic heuristic (no
stemming/lemmatization) — "revenue" and "revenues" are treated as different
tokens, which is a known, documented limitation (see docs/deployment.md
for how to extend this).

---

## 3. Text Chunking

**What:** Splitting each page's text into overlapping, fixed-size windows.

**Where:** `app/ingestion/chunking.py::chunk_text` / `chunk_document`.

**Input:** raw page text + `CHUNK_SIZE` (default 800 chars) and
`CHUNK_OVERLAP` (default 150 chars).
**Output:** list of `Chunk` objects with `document_id`, `filename`,
`page_number`, `section`, `chunk_id`, `text`.

**Why it's required:** Embedding models produce ONE vector per input. A
100-page document embedded as a single vector would average together
completely unrelated facts (revenue figures, risk disclosures, executive
bios) into a nearly useless vector that can't discriminate between topics.
Chunking lets each vector represent one coherent idea, enabling precise
retrieval instead of "the whole document is vaguely related."

**Trade-offs:**
- *Smaller chunks* → sharper retrieval precision (a matched chunk is
  more likely to be exactly on-topic), but less surrounding context per
  chunk for the LLM to reason with, and more chunks/vectors to store and
  search.
- *Larger chunks* → more context per retrieved unit, but the embedding
  blurs multiple topics together, hurting retrieval precision.
- *Overlap* prevents a sentence that would answer the question from being
  split exactly at a chunk boundary and losing coherence in both halves.

**How poor chunking hurts retrieval:** if chunks are too large, a query
about "gross margin" might retrieve a chunk whose vector is dominated by
an unrelated adjacent paragraph about litigation, burying the actually
relevant sentence and giving the reranker/LLM less to work with. If chunks
are too small with no overlap, a fact split across a chunk boundary
("...margin expanded" | "to 45% due to...") can become unretrievable by
either half.

---

## 4. Embeddings (Sentence Embeddings)

**What:** Dense vector representations of text (384 floats for
`all-MiniLM-L6-v2`) such that semantically similar texts produce vectors
that are close together, even without shared vocabulary.

**Where:** `app/embeddings/embedder.py::Embedder`.

**Input:** raw text (a chunk at ingestion time, a question at query time).
**Output:** a normalized float32 vector (`np.ndarray`).

**Why:** Enables *semantic* similarity search — "sales grew" and "revenue
increased" should be found as related even with zero shared keywords.

**Advantages:** captures paraphrase/meaning; fast approximate search at
scale via HNSW indexes.
**Limitations:** can miss exact identifiers (ticker symbols, precise
figures, rare proper nouns) that a literal keyword match would catch — this
is exactly why BM25 is run alongside it (see #6, #7).

---

## 5. Semantic Similarity / Cosine Similarity

**What:** A measure of how similar two vectors are, based on the angle
between them rather than their magnitude:

```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

Because embeddings are L2-normalized at generation time
(`normalize_embeddings=True` in `embedder.py`), `||A|| = ||B|| = 1`, so
cosine similarity reduces to a plain dot product — which is what Qdrant's
`Distance.COSINE` metric computes directly.

**Where:** `app/embeddings/embedder.py` (normalization),
`app/retrieval/vector_store.py` (Qdrant's COSINE distance metric).

**Range:** -1 (opposite meaning) to 1 (identical meaning); in practice,
semantically related sentence pairs typically score 0.3–0.8.

---

## 6. Vector Search / Nearest-Neighbor Search

**What:** Given a query vector, find the K stored vectors with highest
cosine similarity.

**Where:** `app/retrieval/vector_store.py::VectorStore.search`.

**How:** Qdrant indexes vectors with HNSW (Hierarchical Navigable Small
World graphs), an approximate-nearest-neighbor algorithm that makes search
sub-linear in the number of stored vectors (avoiding an exhaustive O(n)
comparison against every stored vector).

---

## 7. BM25 (Best Matching 25)

**What:** A classical (non-neural) term-frequency/inverse-document-frequency
ranking function with document-length normalization.

**Where:** `app/retrieval/bm25_retriever.py` (using `rank_bm25`'s
`BM25Okapi`).

**Formula:**

```
score(D, Q) = Σ IDF(q_i) * f(q_i, D) * (k1 + 1)
                ─────────────────────────────────────
              f(q_i, D) + k1 * (1 - b + b * |D| / avgdl)
```

where `f(q_i, D)` is term frequency of query term `q_i` in document `D`,
`|D|` is document length, `avgdl` is average document length across the
corpus, and `k1` (default 1.5) / `b` (default 0.75) are tuning constants.

**Why:** Dense vector search learns "meaning" but can genuinely miss exact
matches — a query for "Q3 FY2025 gross margin" needs those literal tokens
to line up with a specific table row; embeddings sometimes rank a
thematically-similar-but-wrong paragraph above the exact one. BM25
guarantees exact-term matches are never invisible to retrieval.

**Limitation:** no understanding of synonyms or paraphrase — this is
exactly what vector search complements it with.

---

## 8. Hybrid Search

**What:** Running both BM25 and vector search, then combining their ranked
result lists into one.

**Where:** orchestrated in `app/services/query_service.py::run_rag_pipeline`,
fused in `app/retrieval/hybrid.py`.

**Why:** Vector search can miss exact terms; BM25 can miss semantic
meaning. Running both and fusing catches cases either method alone would
miss, which is why hybrid retrieval consistently outperforms either method
alone in IR literature and in this project's own benchmark
(`scripts/benchmark.py`).

---

## 9. Reciprocal Rank Fusion (RRF)

**What:** A rank-based (not score-based) method for merging multiple
ranked lists:

```
RRF_score(d) = Σ_lists 1 / (k + rank_in_that_list(d))
```

`k=60` (the standard damping constant from Cormack et al., 2009) is used
here (`RRF_K` in config).

**Where:** `app/retrieval/hybrid.py::reciprocal_rank_fusion`.

**Why RRF instead of averaging raw scores:** cosine similarity (bounded
[-1,1]) and BM25 scores (unbounded, corpus-dependent magnitude) are on
completely different, incomparable scales. Naive score averaging is
fragile and needs constant re-tuning per corpus. RRF only uses *rank
order* within each list, which is far more stable across different
queries and document sets. A chunk near the top of either list scores
well; a chunk near the top of *both* lists scores even higher; a chunk
found only by one method (e.g. an exact match BM25 caught that vector
search missed) still enters the fused candidate set.

---

## 10. Cross-Encoder Reranking

**What:** Unlike the bi-encoder used for embeddings (which encodes the
query and each chunk *separately*, then compares vectors), a cross-encoder
feeds the `(query, chunk)` pair *together* into one transformer forward
pass and outputs a single relevance score directly — no vectors, no
similarity function.

**Where:** `app/reranking/cross_encoder.py` (using
`cross-encoder/ms-marco-MiniLM-L-6-v2`).

**Why two stages, not just a cross-encoder everywhere:** cross-encoders are
far more accurate at fine-grained relevance judgment because the model
attends between query and passage token-by-token — but that accuracy costs
one full forward pass per `(query, chunk)` pair, which cannot be
pre-computed/indexed like bi-encoder embeddings can. Running it over an
entire corpus of thousands of chunks per query would be too slow. So
first-stage retrieval (hybrid BM25+vector, fused with RRF) cheaply narrows
the corpus down to ~20–40 candidates; the cross-encoder then re-scores only
those, and we keep the strongest `RERANK_TOP_K` (default 5) as the actual
evidence sent to the LLM.

**Effect on evidence quality:** this "retrieve cheap, rerank precise"
pattern gets both the speed of bi-encoders and the accuracy of
cross-encoders — see the measured Recall@K/MRR difference between
`vector_only`, `vector_rerank`, and `hybrid_rerank` in
`scripts/benchmark.py`'s output once you run it against real documents.

---

## 11. Information Retrieval / Top-K Retrieval

**What:** Returning the K most relevant items from a larger collection
rather than everything, or a single "best" match. Applied at every
retrieval stage (`VECTOR_TOP_K`, `BM25_TOP_K`, `RERANK_TOP_K` — all
configurable in `.env`).

**Why:** Balances relevance coverage (larger K catches more potentially
relevant evidence) against noise and LLM context budget (too large a K
means the LLM has to sift through weak evidence, increasing hallucination
risk and cost).

---

## 12. RAG (Retrieval-Augmented Generation)

**What:** Instead of asking an LLM to answer purely from its trained
knowledge (which is fixed at training time and can't know about your
uploaded documents), RAG retrieves relevant evidence at query time and
gives it to the LLM as context, asking it to answer *using that evidence*.

**Where:** the full pipeline in `app/services/query_service.py`.

**Why:** LLMs cannot know about documents uploaded after their training
cutoff, and even when they "know" a public company's numbers generally,
those numbers may not match the specific fiscal year/report the user
uploaded. RAG grounds every answer in retrievable, citable, real text.

---

## 13. Prompt Engineering

**What:** Deliberately designing the system prompt and context format sent
to the LLM to shape its behavior.

**Where:** `app/llm/client.py::SYSTEM_PROMPT` and `_build_user_prompt`.

**How:** the system prompt explicitly instructs the model to (1) answer
only from provided evidence, (2) cite evidence by bracketed number, (3)
say so explicitly when evidence is insufficient rather than guess, (4)
distinguish stated facts from inference, (5) never invent citations. See
`docs/evaluation.md` for how faithfulness to these instructions is
measured.

---

## 14. Hallucination & Anti-Hallucination Design

**What:** "Hallucination" is when an LLM generates a plausible-sounding but
unsupported or fabricated claim.

**Where mitigated:**
- The system prompt (`app/llm/client.py`) forbids using outside knowledge
  and requires citing evidence for every claim.
- `LLMClient.answer_question` explicitly short-circuits to an "insufficient
  evidence" response when no evidence was retrieved at all, rather than
  letting the LLM attempt an answer from nothing.
- A lightweight post-hoc check
  (`"do not contain enough information" in raw.lower()`) flags responses
  where the model itself reported insufficient evidence, surfaced to the
  frontend as a visible warning banner.

**Measured via:** faithfulness/groundedness grading (see
`app/services/evaluation_service.py::faithfulness_rate` /
`hallucination_rate`) — see docs/evaluation.md for how to run this.

---

## 15. Retrieval Evaluation Metrics

Implemented in `app/services/evaluation_service.py`. With a single
labeled-relevant chunk per evaluation question (the common case for this
dataset format):

- **Precision@K** = (# relevant chunks in top K) / K
- **Recall@K** = (# relevant chunks in top K) / (# relevant chunks that
  exist) — with one relevant chunk per question, this is 1.0 if found
  anywhere in the top K, else 0.0, averaged over all questions.
- **MRR (Mean Reciprocal Rank)** = mean(1 / rank of first relevant result)
  — rewards ranking the right evidence *first*, not just somewhere in the
  top K.
- **Hit Rate@K** = fraction of questions where the relevant chunk appears
  anywhere in the top K. Numerically identical to Recall@K under
  single-relevant-item ground truth (kept as a separate named metric
  because they diverge under multi-label ground truth).
- **Latency** = wall-clock time (ms) for the full pipeline run, measured
  directly around `run_rag_pipeline` in `scripts/evaluate.py` /
  `scripts/benchmark.py`.
- **Groundedness / Faithfulness** = whether every claim in a generated
  answer is actually supported by the retrieved evidence (vs. invented) —
  requires grading (manual or LLM-as-judge), see docs/evaluation.md.

## How each concept affects system performance (summary)

| Concept | Primary effect if misconfigured |
|---|---|
| Chunk size/overlap | Too large → blurry embeddings, weak retrieval precision. Too small → lost context, fragmented facts. |
| Embedding model | Weaker models → worse semantic recall, especially on paraphrased questions. |
| BM25 | Disabling it → misses exact-term/rare-identifier queries. |
| RRF k | Very small k → over-rewards rank-1 items; very large k → nearly ignores rank differences. |
| Reranker top-k | Too small → may exclude the correct evidence before the LLM sees it. Too large → dilutes LLM context with weak evidence. |
| LLM temperature | Higher → more fluent but less deterministic/faithful answers; kept at 0.0 by default for factual consistency. |
