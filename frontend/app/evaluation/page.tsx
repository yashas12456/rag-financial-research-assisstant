"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import { api } from "@/services/api";
import type { EvaluationMethodResult } from "@/types";
import { PlayCircle, Info } from "lucide-react";

const METRIC_COLUMNS: { key: keyof EvaluationMethodResult; label: string; pct?: boolean }[] = [
  { key: "recall_at_k", label: "Recall@K", pct: true },
  { key: "precision_at_k", label: "Precision@K", pct: true },
  { key: "mrr", label: "MRR" },
  { key: "hit_rate", label: "Hit Rate", pct: true },
  { key: "answer_accuracy", label: "Answer Accuracy", pct: true },
  { key: "citation_accuracy", label: "Citation Accuracy", pct: true },
  { key: "faithfulness", label: "Faithfulness", pct: true },
  { key: "hallucination_rate", label: "Hallucination Rate", pct: true },
  { key: "avg_latency_ms", label: "Avg Latency (ms)" },
];

function fmt(value: number | null | undefined, pct?: boolean) {
  if (value === null || value === undefined) return "—";
  return pct ? `${(value * 100).toFixed(1)}%` : value.toFixed(1);
}

export default function EvaluationPage() {
  const [results, setResults] = useState<EvaluationMethodResult[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = () => api.getEvaluationResults().then(setResults).catch(() => {});

  useEffect(() => {
    load();
  }, []);

  const runEvaluation = async () => {
    setRunning(true);
    setError(null);
    try {
      const r = await api.runEvaluation(["vector_only", "vector_rerank", "hybrid_rerank"]);
      setResults(r.results);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Evaluation Dashboard</h2>
          <p className="text-slate-500 mt-1">
            Compare retrieval and answer quality across pipeline configurations.
          </p>
        </div>
        <button
          onClick={runEvaluation}
          disabled={running}
          className="flex items-center gap-2 bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
        >
          <PlayCircle size={16} /> {running ? "Running..." : "Run Evaluation"}
        </button>
      </div>

      <Card className="bg-blue-50 border-blue-200 text-blue-800 text-sm flex items-start gap-2">
        <Info size={16} className="mt-0.5 shrink-0" />
        Metrics are computed live against <code>data/evaluation/questions.json</code> and your
        currently indexed documents — nothing here is precomputed or fabricated. Results will
        show "—" for any question set you haven't run yet, or for answer-quality metrics that
        require LLM/manual grading (see docs/evaluation.md).
      </Card>

      {error && <Card className="border-red-200 bg-red-50 text-red-700 text-sm">{error}</Card>}

      <Card>
        {results.length === 0 ? (
          <p className="text-sm text-slate-500">
            No evaluation runs yet. Click "Run Evaluation" after uploading and indexing some
            documents and populating data/evaluation/questions.json with real questions about them.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-slate-500 border-b border-slate-100">
                  <th className="pb-2 pr-4">Method</th>
                  {METRIC_COLUMNS.map((c) => (
                    <th key={c.key as string} className="pb-2 pr-4 whitespace-nowrap">{c.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.method_name} className="border-b border-slate-50">
                    <td className="py-3 pr-4 font-medium text-slate-800">{r.method_name}</td>
                    {METRIC_COLUMNS.map((c) => (
                      <td key={c.key as string} className="py-3 pr-4 text-slate-600">
                        {fmt(r[c.key] as number | null, c.pct)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
