export interface Document {
  id: string;
  filename: string;
  status: "uploaded" | "processing" | "indexed" | "failed";
  num_pages: number;
  file_size_bytes: number;
  company?: string | null;
  fiscal_year?: string | null;
  error_message?: string | null;
  created_at: string;
  indexed_at?: string | null;
}

export interface RetrievedChunk {
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  section: string | null;
  text: string;
  vector_score: number | null;
  bm25_score: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
  rank: number;
}

export interface Citation {
  marker: string;
  filename: string;
  page_number: number;
  chunk_id: string;
  rerank_score: number | null;
}

export interface QueryResponse {
  query_id: string;
  question: string;
  answer: string;
  insufficient_evidence: boolean;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
  latency_ms: number;
}

export interface EvaluationMethodResult {
  method_name: string;
  num_questions: number;
  recall_at_k: number | null;
  precision_at_k: number | null;
  mrr: number | null;
  hit_rate: number | null;
  answer_accuracy: number | null;
  citation_accuracy: number | null;
  faithfulness: number | null;
  hallucination_rate: number | null;
  avg_latency_ms: number | null;
}

export interface HealthStatus {
  status: string;
  database: string;
  vector_store: string;
  embedding_model: string;
  reranker_model: string;
  llm_provider: string;
}
