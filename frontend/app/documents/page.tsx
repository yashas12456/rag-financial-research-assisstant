"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import StatusBadge from "@/components/StatusBadge";
import { api } from "@/services/api";
import type { Document } from "@/types";
import { Trash2, PlayCircle, RefreshCw } from "lucide-react";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () =>
    api.listDocuments().then((r) => setDocuments(r.documents)).catch((e) => setError(e.message));

  useEffect(() => {
    load();
  }, []);

  const handleIndex = async (id: string) => {
    setBusyId(id);
    try {
      await api.indexDocument(id);
      await load();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this document and all its indexed chunks?")) return;
    setBusyId(id);
    try {
      await api.deleteDocument(id);
      await load();
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Documents</h2>
          <p className="text-slate-500 mt-1">Manage uploaded reports and their indexing status.</p>
        </div>
        <button
          onClick={load}
          className="flex items-center gap-2 text-sm px-3 py-2 rounded-lg border border-slate-200 hover:bg-slate-50"
        >
          <RefreshCw size={16} /> Refresh
        </button>
      </div>

      {error && (
        <Card className="border-red-200 bg-red-50 text-red-700 text-sm">{error}</Card>
      )}

      <Card>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-500 border-b border-slate-100">
              <th className="pb-2">Filename</th>
              <th className="pb-2">Size</th>
              <th className="pb-2">Pages</th>
              <th className="pb-2">Status</th>
              <th className="pb-2">Uploaded</th>
              <th className="pb-2 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {documents.map((d) => (
              <tr key={d.id} className="border-b border-slate-50">
                <td className="py-3 font-medium text-slate-800">{d.filename}</td>
                <td className="py-3 text-slate-600">{(d.file_size_bytes / 1024 / 1024).toFixed(2)} MB</td>
                <td className="py-3 text-slate-600">{d.num_pages}</td>
                <td className="py-3">
                  <StatusBadge status={d.status} />
                  {d.status === "failed" && d.error_message && (
                    <p className="text-xs text-red-600 mt-1 max-w-xs">{d.error_message}</p>
                  )}
                </td>
                <td className="py-3 text-slate-500">{new Date(d.created_at).toLocaleString()}</td>
                <td className="py-3">
                  <div className="flex justify-end gap-2">
                    {d.status !== "indexed" && (
                      <button
                        disabled={busyId === d.id}
                        onClick={() => handleIndex(d.id)}
                        className="flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-brand-50 text-brand-700 hover:bg-brand-100 disabled:opacity-50"
                      >
                        <PlayCircle size={14} /> Index
                      </button>
                    )}
                    <button
                      disabled={busyId === d.id}
                      onClick={() => handleDelete(d.id)}
                      className="flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-red-50 text-red-700 hover:bg-red-100 disabled:opacity-50"
                    >
                      <Trash2 size={14} /> Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {documents.length === 0 && (
          <p className="text-sm text-slate-500 py-4">No documents uploaded yet.</p>
        )}
      </Card>
    </div>
  );
}
