"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import StatusBadge from "@/components/StatusBadge";
import { api } from "@/services/api";
import type { Document, HealthStatus } from "@/types";
import { FileStack, MessagesSquare, Activity, Database } from "lucide-react";

export default function DashboardPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listDocuments().then((r) => setDocuments(r.documents)).catch((e) => setError(e.message));
    api.health().then(setHealth).catch(() => {});
  }, []);

  const indexed = documents.filter((d) => d.status === "indexed").length;
  const totalPages = documents.reduce((sum, d) => sum + (d.num_pages || 0), 0);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Dashboard</h2>
        <p className="text-slate-500 mt-1">
          Overview of your document library and system status.
        </p>
      </div>

      {error && (
        <Card className="border-red-200 bg-red-50 text-red-700 text-sm">
          Could not reach the backend API: {error}. Is the backend running (see README)?
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <div className="flex items-center gap-3">
            <FileStack className="text-brand-600" size={22} />
            <div>
              <p className="text-2xl font-bold">{documents.length}</p>
              <p className="text-xs text-slate-500">Documents uploaded</p>
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex items-center gap-3">
            <Database className="text-brand-600" size={22} />
            <div>
              <p className="text-2xl font-bold">{indexed}</p>
              <p className="text-xs text-slate-500">Documents indexed</p>
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex items-center gap-3">
            <MessagesSquare className="text-brand-600" size={22} />
            <div>
              <p className="text-2xl font-bold">{totalPages}</p>
              <p className="text-xs text-slate-500">Pages processed</p>
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex items-center gap-3">
            <Activity className={health?.status === "ok" ? "text-emerald-600" : "text-amber-600"} size={22} />
            <div>
              <p className="text-2xl font-bold capitalize">{health?.status || "unknown"}</p>
              <p className="text-xs text-slate-500">System status</p>
            </div>
          </div>
        </Card>
      </div>

      <Card>
        <h3 className="font-semibold text-slate-900 mb-4">Recent Documents</h3>
        {documents.length === 0 ? (
          <p className="text-sm text-slate-500">
            No documents yet. Head to Upload to add an annual report or earnings PDF.
          </p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-500 border-b border-slate-100">
                <th className="pb-2">Filename</th>
                <th className="pb-2">Pages</th>
                <th className="pb-2">Status</th>
                <th className="pb-2">Uploaded</th>
              </tr>
            </thead>
            <tbody>
              {documents.slice(0, 8).map((d) => (
                <tr key={d.id} className="border-b border-slate-50">
                  <td className="py-2 font-medium text-slate-800">{d.filename}</td>
                  <td className="py-2 text-slate-600">{d.num_pages}</td>
                  <td className="py-2"><StatusBadge status={d.status} /></td>
                  <td className="py-2 text-slate-500">
                    {new Date(d.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      {health && (
        <Card>
          <h3 className="font-semibold text-slate-900 mb-3">System Configuration</h3>
          <dl className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <dt className="text-slate-500">Embedding model</dt>
              <dd className="font-medium">{health.embedding_model}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Reranker model</dt>
              <dd className="font-medium">{health.reranker_model}</dd>
            </div>
            <div>
              <dt className="text-slate-500">LLM provider</dt>
              <dd className="font-medium capitalize">{health.llm_provider}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Vector store</dt>
              <dd className="font-medium">{health.vector_store}</dd>
            </div>
          </dl>
        </Card>
      )}
    </div>
  );
}
