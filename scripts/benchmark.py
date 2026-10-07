#!/usr/bin/env python3
"""
Benchmark script: compares 5 retrieval configurations (a finer-grained
breakdown than scripts/evaluate.py's 3 baselines, isolating BM25-alone and
the effect of reranking on top of hybrid retrieval):

  1. vector_only        - dense vector search alone
  2. vector_rerank       - dense vector search + cross-encoder reranking
  3. bm25_only           - BM25 keyword search alone
  4. hybrid              - BM25 + vector fused with RRF, no reranking
  5. hybrid_rerank       - BM25 + vector fused with RRF, then reranked

Measures Recall@K, MRR, Hit Rate, and latency for each, and prints a
readable comparison table. Like scripts/evaluate.py, this computes real
numbers against your currently-indexed documents and questions.json --
it does not fabricate results, and will show honest zeros if nothing is
indexed yet.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.logging import configure_logging, get_logger
from app.db.session import SessionLocal
from app.services.evaluation_service import (
    RetrievalJudgement,
    hit_rate_at_k,
    mean_reciprocal_rank,
    recall_at_k,
)
from app.services.ingestion_service import rebuild_bm25_index
from app.services.query_service import run_rag_pipeline
from scripts.evaluate import load_questions

configure_logging()
logger = get_logger("scripts.benchmark")

CONFIGS = {
    "vector_only":   {"use_bm25": False, "use_rerank": False, "use_vector": True},
    "vector_rerank": {"use_bm25": False, "use_rerank": True,  "use_vector": True},
    "bm25_only":     {"use_bm25": True,  "use_rerank": False, "use_vector": False},
    "hybrid":        {"use_bm25": True,  "use_rerank": False, "use_vector": True},
    "hybrid_rerank": {"use_bm25": True,  "use_rerank": True,  "use_vector": True},
}


def benchmark_one(db, method_name: str, config: dict, questions: list[dict], k: int) -> dict:
    judgements = []
    latencies = []

    for q in questions:
        start = time.perf_counter()
        result = run_rag_pipeline(
            db,
            question=q["question"],
            top_k=k,
            use_bm25=config["use_bm25"],
            use_rerank=config["use_rerank"],
            use_vector=config.get("use_vector", True),
        )
        latencies.append((time.perf_counter() - start) * 1000)

        retrieved_pages = [
            (rc["filename"], rc["page_number"]) for rc in result["retrieved_chunks"]
        ]
        judgements.append(
            RetrievalJudgement(
                question=q["question"],
                expected_document=q["document"],
                expected_page=q["page"],
                retrieved_pages=retrieved_pages,
            )
        )

    return {
        "method": method_name,
        "recall_at_k": recall_at_k(judgements, k) if judgements else None,
        "mrr": mean_reciprocal_rank(judgements) if judgements else None,
        "hit_rate": hit_rate_at_k(judgements, k) if judgements else None,
        "avg_latency_ms": statistics.mean(latencies) if latencies else None,
        "p95_latency_ms": (
            statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=None)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval configurations")
    parser.add_argument("--dataset", default="data/evaluation/questions.json")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    questions = load_questions(args.dataset)

    db = SessionLocal()
    try:
        rebuild_bm25_index(db)
        rows = [benchmark_one(db, name, cfg, questions, args.k) for name, cfg in CONFIGS.items()]
    finally:
        db.close()

    print("\n" + "=" * 90)
    print("RETRIEVAL BENCHMARK")
    print("=" * 90)
    print(f"{'Method':<16}{'Recall@K':>12}{'MRR':>10}{'HitRate':>10}{'AvgLatency':>14}{'P95Latency':>14}")
    print("-" * 90)
    for r in rows:
        def f(v):
            return "—" if v is None else f"{v:.3f}"
        print(
            f"{r['method']:<16}{f(r['recall_at_k']):>12}{f(r['mrr']):>10}"
            f"{f(r['hit_rate']):>10}{f(r['avg_latency_ms']):>14}{f(r['p95_latency_ms']):>14}"
        )
    print("=" * 90 + "\n")

    os.makedirs("data/evaluation", exist_ok=True)
    with open("data/evaluation/last_benchmark_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("Results written to data/evaluation/last_benchmark_results.json")


if __name__ == "__main__":
    main()
