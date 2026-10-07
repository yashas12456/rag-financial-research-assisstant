"""
Evaluation metrics.

ML CONCEPT: Retrieval evaluation (Precision@K, Recall@K, MRR, Hit Rate)
and answer-quality evaluation (correctness, faithfulness/groundedness,
hallucination rate).

WHY: A RAG system that "seems to work" on a few manual spot-checks is not
     the same as one with measured performance. This module computes the
     standard information-retrieval metrics against a labeled evaluation
     set (data/evaluation/questions.json), so the improvement from
     baseline -> hybrid -> reranked can be demonstrated with real numbers
     instead of asserted.

Definitions used here (k = number of chunks returned to the LLM):

  Recall@K    = (# relevant chunks retrieved in top K) / (# relevant chunks that exist)
                For this project each question is labeled with ONE relevant
                page/chunk, so Recall@K is 1.0 if that page is anywhere in
                the top K, else 0.0, averaged over all questions.
  Precision@K = (# relevant chunks retrieved in top K) / K
                With a single relevant chunk per question this is
                (1/K) if found else 0, averaged over all questions --
                included for completeness/transparency even though with a
                single-relevant-item ground truth, Recall@K and Hit Rate
                will be numerically identical here; Precision@K is what
                differentiates them.
  MRR         = mean(1 / rank_of_first_relevant_result), 0 if not found.
                Rewards ranking the right evidence *first*, not just
                somewhere in the top K.
  Hit Rate@K  = fraction of questions where the relevant page appears
                anywhere in the top K (same as Recall@K for single-label
                ground truth, kept separate because with multi-label
                ground truth they diverge).

Answer-quality metrics (require an LLM-graded or human-graded judgment,
see scripts/evaluate.py for how these are computed against real answers):
  - answer_accuracy: does the generated answer match the expected answer's
    key facts?
  - citation_accuracy: does the answer cite the correct document/page?
  - faithfulness/groundedness: is every claim in the answer actually
    supported by the retrieved evidence (vs. invented)?
  - hallucination_rate: 1 - faithfulness, fraction of answers containing
    at least one unsupported claim.
"""
from dataclasses import dataclass


@dataclass
class RetrievalJudgement:
    """One evaluation question's retrieval outcome."""
    question: str
    expected_document: str
    expected_page: int
    retrieved_pages: list[tuple[str, int]]  # ordered (filename, page_number) as retrieved


def _is_relevant(expected_doc: str, expected_page: int, filename: str, page_number: int) -> bool:
    return filename == expected_doc and page_number == expected_page


def recall_at_k(judgements: list[RetrievalJudgement], k: int) -> float:
    if not judgements:
        return 0.0
    hits = 0
    for j in judgements:
        top_k = j.retrieved_pages[:k]
        if any(_is_relevant(j.expected_document, j.expected_page, f, p) for f, p in top_k):
            hits += 1
    return hits / len(judgements)


def precision_at_k(judgements: list[RetrievalJudgement], k: int) -> float:
    if not judgements:
        return 0.0
    total_precision = 0.0
    for j in judgements:
        top_k = j.retrieved_pages[:k]
        relevant_count = sum(
            1 for f, p in top_k if _is_relevant(j.expected_document, j.expected_page, f, p)
        )
        total_precision += relevant_count / k if k > 0 else 0.0
    return total_precision / len(judgements)


def mean_reciprocal_rank(judgements: list[RetrievalJudgement]) -> float:
    if not judgements:
        return 0.0
    total = 0.0
    for j in judgements:
        for rank, (f, p) in enumerate(j.retrieved_pages, start=1):
            if _is_relevant(j.expected_document, j.expected_page, f, p):
                total += 1.0 / rank
                break
    return total / len(judgements)


def hit_rate_at_k(judgements: list[RetrievalJudgement], k: int) -> float:
    # Identical formula to recall_at_k under single-relevant-item ground truth;
    # kept as a distinct named metric per the spec and for clarity in reports.
    return recall_at_k(judgements, k)


@dataclass
class AnswerJudgement:
    question: str
    generated_answer: str
    expected_answer: str
    expected_evidence_substring: str  # a short string expected to appear in the retrieved evidence
    retrieved_texts: list[str]
    is_correct: bool | None = None       # filled in by an LLM-graded or manual judge
    is_faithful: bool | None = None      # filled in by an LLM-graded or manual judge
    cited_correct_source: bool | None = None


def answer_accuracy(judgements: list[AnswerJudgement]) -> float | None:
    graded = [j for j in judgements if j.is_correct is not None]
    if not graded:
        return None
    return sum(1 for j in graded if j.is_correct) / len(graded)


def citation_accuracy(judgements: list[AnswerJudgement]) -> float | None:
    graded = [j for j in judgements if j.cited_correct_source is not None]
    if not graded:
        return None
    return sum(1 for j in graded if j.cited_correct_source) / len(graded)


def faithfulness_rate(judgements: list[AnswerJudgement]) -> float | None:
    graded = [j for j in judgements if j.is_faithful is not None]
    if not graded:
        return None
    return sum(1 for j in graded if j.is_faithful) / len(graded)


def hallucination_rate(judgements: list[AnswerJudgement]) -> float | None:
    f = faithfulness_rate(judgements)
    return None if f is None else round(1.0 - f, 4)
