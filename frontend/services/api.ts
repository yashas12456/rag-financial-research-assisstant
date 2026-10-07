import type {
  Document,
  EvaluationMethodResult,
  HealthStatus,
  QueryResponse,
} from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: { ...(options?.headers || {}) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ message: res.statusText }));
    throw new Error(body.message || `Request to ${path} failed (${res.status})`);
  }
  return res.json();
}

export const api = {
  health: () => request<HealthStatus>("/health"),

  listDocuments: () =>
    request<{ documents: Document[]; total: number }>("/documents"),

  getDocument: (id: string) => request<Document>(`/documents/${id}`),

  uploadDocument: async (file: File): Promise<{ document: Document; message: string }> => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_URL}/documents/upload`, { method: "POST", body: formData });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ message: res.statusText }));
      throw new Error(body.message || "Upload failed");
    }
    return res.json();
  },

  indexDocument: (id: string) =>
    request<{ document_id: string; num_chunks: number; status: string }>(
      `/documents/${id}/index`,
      { method: "POST" }
    ),

  deleteDocument: (id: string) =>
    fetch(`${API_URL}/documents/${id}`, { method: "DELETE" }),

  askQuestion: (question: string, documentIds?: string[]) =>
    request<QueryResponse>("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, document_ids: documentIds }),
    }),

  runEvaluation: (methods: string[], k = 5) =>
    request<{ id: string; results: EvaluationMethodResult[]; created_at: string }>(
      "/evaluation/run",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ methods, k }),
      }
    ),

  getEvaluationResults: () => request<EvaluationMethodResult[]>("/evaluation/results"),
};
