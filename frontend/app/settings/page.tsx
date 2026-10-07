"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import { api } from "@/services/api";
import type { HealthStatus } from "@/types";

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => {});
  }, []);

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">System Settings</h2>
        <p className="text-slate-500 mt-1">
          Retrieval and model configuration is set via backend environment variables
          (see .env.example) and requires a backend restart to change.
        </p>
      </div>

      <Card>
        <h3 className="font-semibold text-slate-900 mb-3">Current Configuration</h3>
        {health ? (
          <dl className="space-y-3 text-sm">
            <Row label="Status" value={health.status} />
            <Row label="Database" value={health.database} />
            <Row label="Vector store" value={health.vector_store} />
            <Row label="Embedding model" value={health.embedding_model} />
            <Row label="Reranker model" value={health.reranker_model} />
            <Row label="LLM provider" value={health.llm_provider} />
          </dl>
        ) : (
          <p className="text-sm text-slate-500">Could not reach the backend.</p>
        )}
      </Card>

      <Card>
        <h3 className="font-semibold text-slate-900 mb-2">Tunable pipeline parameters</h3>
        <p className="text-sm text-slate-500 mb-3">
          Set these in your root <code>.env</code> file — see docs/ml_concepts.md for what each
          one controls and its trade-offs.
        </p>
        <ul className="text-sm text-slate-600 space-y-1 font-mono">
          <li>CHUNK_SIZE, CHUNK_OVERLAP</li>
          <li>VECTOR_TOP_K, BM25_TOP_K, RRF_K</li>
          <li>RERANK_TOP_K, RERANK_ENABLED</li>
          <li>SIMILARITY_THRESHOLD</li>
          <li>LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE</li>
        </ul>
      </Card>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between border-b border-slate-50 pb-2">
      <dt className="text-slate-500">{label}</dt>
      <dd className="font-medium text-slate-800">{value}</dd>
    </div>
  );
}
