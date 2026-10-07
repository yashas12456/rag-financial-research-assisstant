"""Tests for retrieval/answer evaluation metrics (app/services/evaluation_service.py)."""
from app.services.evaluation_service import (
    RetrievalJudgement,
    hit_rate_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_hit_and_miss():
    judgements = [
        RetrievalJudgement(
            question="q1", expected_document="a.pdf", expected_page=1,
            retrieved_pages=[("a.pdf", 1), ("b.pdf", 2)],
        ),
        RetrievalJudgement(
            question="q2", expected_document="a.pdf", expected_page=5,
            retrieved_pages=[("b.pdf", 2), ("c.pdf", 3)],
        ),
    ]
    assert recall_at_k(judgements, k=2) == 0.5  # only q1 found its page


def test_mrr_rewards_earlier_rank():
    judgements = [
        RetrievalJudgement(
            question="q1", expected_document="a.pdf", expected_page=1,
            retrieved_pages=[("a.pdf", 1), ("b.pdf", 2)],  # found at rank 1
        ),
        RetrievalJudgement(
            question="q2", expected_document="a.pdf", expected_page=1,
            retrieved_pages=[("b.pdf", 2), ("a.pdf", 1)],  # found at rank 2
        ),
    ]
    mrr = mean_reciprocal_rank(judgements)
    assert abs(mrr - ((1.0 + 0.5) / 2)) < 1e-9


def test_precision_at_k():
    judgements = [
        RetrievalJudgement(
            question="q1", expected_document="a.pdf", expected_page=1,
            retrieved_pages=[("a.pdf", 1), ("b.pdf", 2)],
        ),
    ]
    assert precision_at_k(judgements, k=2) == 0.5  # 1 relevant out of 2 returned


def test_hit_rate_matches_recall_for_single_label_ground_truth():
    judgements = [
        RetrievalJudgement(
            question="q1", expected_document="a.pdf", expected_page=1,
            retrieved_pages=[("a.pdf", 1)],
        ),
    ]
    assert hit_rate_at_k(judgements, k=1) == recall_at_k(judgements, k=1)


def test_empty_judgements_return_zero():
    assert recall_at_k([], k=5) == 0.0
    assert mean_reciprocal_rank([]) == 0.0
    assert precision_at_k([], k=5) == 0.0
