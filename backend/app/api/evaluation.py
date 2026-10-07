"""Evaluation endpoints -- trigger the same logic as scripts/evaluate.py via API."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.models import EvaluationRun
from app.schemas.schemas import EvaluationMethodResult, EvaluationRunRequest, EvaluationRunResponse

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationRunResponse)
def run_evaluation(payload: EvaluationRunRequest, db: Session = Depends(get_db)):
    # Delegates to the shared evaluation runner so the API and the CLI script
    # (scripts/evaluate.py) never diverge in behavior.
    from scripts.evaluate import run_all_methods

    results = run_all_methods(db, dataset_path=payload.dataset_path, k=payload.k, methods=payload.methods)

    saved = []
    for r in results:
        run = EvaluationRun(
            method_name=r["method_name"],
            num_questions=r["num_questions"],
            recall_at_k=r.get("recall_at_k"),
            precision_at_k=r.get("precision_at_k"),
            mrr=r.get("mrr"),
            hit_rate=r.get("hit_rate"),
            answer_accuracy=r.get("answer_accuracy"),
            citation_accuracy=r.get("citation_accuracy"),
            faithfulness=r.get("faithfulness"),
            hallucination_rate=r.get("hallucination_rate"),
            avg_latency_ms=r.get("avg_latency_ms"),
            raw_results=r.get("raw_results", {}),
        )
        db.add(run)
        db.flush()
        saved.append(run)

    return EvaluationRunResponse(
        id=saved[0].id if saved else None,
        results=[EvaluationMethodResult(**r) for r in results],
        created_at=saved[0].created_at if saved else None,
    )


@router.get("/results")
def get_results(db: Session = Depends(get_db)):
    runs = db.query(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(50).all()
    return [
        {
            "id": r.id,
            "method_name": r.method_name,
            "num_questions": r.num_questions,
            "recall_at_k": r.recall_at_k,
            "precision_at_k": r.precision_at_k,
            "mrr": r.mrr,
            "hit_rate": r.hit_rate,
            "answer_accuracy": r.answer_accuracy,
            "citation_accuracy": r.citation_accuracy,
            "faithfulness": r.faithfulness,
            "hallucination_rate": r.hallucination_rate,
            "avg_latency_ms": r.avg_latency_ms,
            "created_at": r.created_at,
        }
        for r in runs
    ]
