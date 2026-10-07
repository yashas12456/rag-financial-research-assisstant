#!/usr/bin/env python3
"""
Evaluate the RAG pipeline against data/evaluation/questions.json, comparing
the three baselines called for in the spec:

  BASELINE 1 "vector_only"    : vector search only, no BM25, no reranking
  BASELINE 2 "vector_rerank"  : vector search + cross-encoder reranking
  BASELINE 3 "hybrid_rerank"  : BM25 + vector (RRF fusion) + reranking

This module is imported both by the CLI (`python scripts/evaluate.py`) and by
the API endpoint (POST /api/evaluation/run in app/api/evaluation.py) so there
is exactly one implementation of "how do we score the pipeline" -- the API
and the CLI can never silently diverge.

IMPORTANT (see main spec section 29 "do not fake results"): this script
computes real Recall@K / Precision@K / MRR / Hit Rate / latency by actually
running retrieval against whatever documents are currently indexed in your
database. It will NOT fabricate numbers. If you have not indexed any
documents, or `questions.json` still contains the shipped placeholder
entries, results will legitimately be empty/zero -- that is correct
behavior, not a bug. See data/evaluation/questions.json.README.md.

Answer-quality metrics (answer_accuracy, citation_accuracy, faithfulness) are
optional and require either manual grading or an LLM-as-judge call (see
`grade_answer_with_llm` below) -- they are left as None if not computed, and
are NEVER filled in with guessed values.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy.orm import Session

from app.core.logging import configure_logging, get_logger
from app.db.session import SessionLocal
from app.models.models import Document, DocumentStatus
from app.services.evaluation_service import (
    RetrievalJudgement,
    hit_rate_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from app.services.ingestion_service import rebuild_bm25_index
from app.services.query_service import run_rag_pipeline

configure_logging()
logger = get_logger("scripts.evaluate")

METHOD_CONFIGS = {
    "vector_only":   {"use_bm25": False, "use_rerank": False},
    "vector_rerank": {"use_bm25": False, "use_rerank": True},
    "hybrid_rerank": {"use_bm25": True,  "use_rerank": True},
}


def load_questions(dataset_path: str) -> list[dict]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    with open(dataset_path) as f:
        return json.load(f)


def _has_indexed_documents(db: Session) -> bool:
    return db.query(Document).filter(Document.status == DocumentStatus.INDEXED).count() > 0


def run_all_methods(
    db: Session,
    dataset_path: str = "data/evaluation/questions.json",
    k: int = 5,
    methods: list[str] | None = None,
) -> list[dict]:
    methods = methods or list(METHOD_CONFIGS.keys())
    questions = load_questions(dataset_path)

    if not _has_indexed_documents(db):
        logger.warning(
            "No indexed documents found. Upload and index documents before running "
            "evaluation -- results below will be empty/zero, which is expected, not fabricated."
        )

    # BM25 index is process-local and in-memory; make sure it reflects the DB.
    rebuild_bm25_index(db)

    results = []
    for method_name in methods:
        if method_name not in METHOD_CONFIGS:
            logger.warning("Unknown method '%s', skipping", method_name)
            continue
        config = METHOD_CONFIGS[method_name]

        judgements: list[RetrievalJudgement] = []
        latencies: list[float] = []
        raw_per_question = []

        for q in questions:
            start = time.perf_counter()
            try:
                pipeline_result = run_rag_pipeline(
                    db,
                    question=q["question"],
                    top_k=k,
                    use_bm25=config["use_bm25"],
                    use_rerank=config["use_rerank"],
                )
            except Exception as exc:
                logger.exception("Pipeline failed for question: %s", q["question"])
                raw_per_question.append({"question": q["question"], "error": str(exc)})
                continue
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            retrieved_pages = [
                (rc["filename"], rc["page_number"]) for rc in pipeline_result["retrieved_chunks"]
            ]
            judgements.append(
                RetrievalJudgement(
                    question=q["question"],
                    expected_document=q["document"],
                    expected_page=q["page"],
                    retrieved_pages=retrieved_pages,
                )
            )
            raw_per_question.append(
                {
                    "question": q["question"],
                    "expected_document": q["document"],
                    "expected_page": q["page"],
                    "retrieved_pages": retrieved_pages,
                    "answer": pipeline_result["answer"],
                    "latency_ms": latency_ms,
                }
            )

        results.append(
            {
                "method_name": method_name,
                "num_questions": len(questions),
                "recall_at_k": recall_at_k(judgements, k) if judgements else None,
                "precision_at_k": precision_at_k(judgements, k) if judgements else None,
                "mrr": mean_reciprocal_rank(judgements) if judgements else None,
                "hit_rate": hit_rate_at_k(judgements, k) if judgements else None,
                # Answer-quality metrics require grading -- see grade_answer_with_llm().
                # Left as None (never fabricated) unless a grading pass is run separately.
                "answer_accuracy": None,
                "citation_accuracy": None,
                "faithfulness": None,
                "hallucination_rate": None,
                "avg_latency_ms": statistics.mean(latencies) if latencies else None,
                "raw_results": {"per_question": raw_per_question},
            }
        )

    return results


def grade_answer_with_llm(question: str, generated_answer: str, expected_answer: str) -> dict:
    """
    OPTIONAL: LLM-as-judge grading for answer_accuracy / faithfulness.

    Not called by default (requires your own LLM_API_KEY and judgement on
    what "correct" means for your dataset). Wire this into run_all_methods
    above if you want automated grading; otherwise grade a sample of
    generated answers manually against your questions.json expected_answer
    field and record the accuracy yourself.
    """
    raise NotImplementedError(
        "Answer grading is not wired up in this template -- see docstring. "
        "This is intentional: automated 'correctness' grading requires a judgment "
        "call (exact match? key-fact overlap? LLM judge?) that depends on your "
        "evaluation dataset, and should not be fabricated."
    )


def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("RAG PIPELINE EVALUATION REPORT")
    print("=" * 100)
    header = f"{'Method':<16}{'N':>5}{'Recall@K':>12}{'Precision@K':>14}{'MRR':>10}{'HitRate':>10}{'AvgLatency(ms)':>18}"
    print(header)
    print("-" * 100)
    for r in results:
        def f(v, pct=True):
            if v is None:
                return "—"
            return f"{v*100:.1f}%" if pct else f"{v:.1f}"
        print(
            f"{r['method_name']:<16}{r['num_questions']:>5}"
            f"{f(r['recall_at_k']):>12}{f(r['precision_at_k']):>14}"
            f"{f(r['mrr'], pct=False):>10}{f(r['hit_rate']):>10}"
            f"{f(r['avg_latency_ms'], pct=False):>18}"
        )
    print("=" * 100)
    print(
        "Answer-quality metrics (accuracy/citation/faithfulness) are not shown here -- "
        "they require grading (see grade_answer_with_llm). This report only reflects "
        "retrieval-quality metrics computed directly from real pipeline runs.\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--dataset", default="data/evaluation/questions.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--methods", nargs="+", default=list(METHOD_CONFIGS.keys()), choices=list(METHOD_CONFIGS.keys())
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        results = run_all_methods(db, dataset_path=args.dataset, k=args.k, methods=args.methods)
        print_report(results)

        os.makedirs("data/evaluation", exist_ok=True)
        out_path = "data/evaluation/last_run_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Full results (including per-question breakdown) written to {out_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
