import type { Citation, RetrievedChunk } from "@/types";

export default function CitationPanel({
  citations,
  chunks,
}: {
  citations: Citation[];
  chunks: RetrievedChunk[];
}) {
  if (citations.length === 0) return null;

  return (
    <div className="mt-4 border-t border-slate-100 pt-4">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
        Sources
      </h4>
      <div className="space-y-3">
        {citations.map((c) => {
          const chunk = chunks.find((rc) => rc.chunk_id === c.chunk_id);
          return (
            <details key={c.chunk_id} className="group bg-slate-50 rounded-lg border border-slate-200">
              <summary className="cursor-pointer list-none px-3 py-2 flex items-center justify-between text-sm">
                <span className="font-medium text-slate-800">
                  {c.marker} {c.filename} — Page {c.page_number}
                </span>
                <div className="flex items-center gap-3 text-xs text-slate-500">
                  {c.rerank_score !== null && (
                    <span>rerank: {c.rerank_score.toFixed(3)}</span>
                  )}
                  <span className="group-open:rotate-180 transition-transform">▾</span>
                </div>
              </summary>
              {chunk && (
                <div className="px-3 pb-3 text-sm text-slate-600 space-y-2">
                  <p className="whitespace-pre-wrap leading-relaxed">{chunk.text}</p>
                  <div className="flex gap-4 text-xs text-slate-400 pt-1 border-t border-slate-200">
                    {chunk.vector_score !== null && (
                      <span>vector: {chunk.vector_score?.toFixed(3)}</span>
                    )}
                    {chunk.bm25_score !== null && (
                      <span>bm25: {chunk.bm25_score?.toFixed(3)}</span>
                    )}
                    {chunk.rrf_score !== null && (
                      <span>rrf: {chunk.rrf_score?.toFixed(4)}</span>
                    )}
                    {chunk.section && <span>section: {chunk.section}</span>}
                  </div>
                </div>
              )}
            </details>
          );
        })}
      </div>
    </div>
  );
}
