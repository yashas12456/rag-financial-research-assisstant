"use client";

import { useState } from "react";
import Card from "@/components/Card";
import CitationPanel from "@/components/CitationPanel";
import { api } from "@/services/api";
import type { QueryResponse } from "@/types";
import { Send, AlertTriangle } from "lucide-react";

interface Turn {
  question: string;
  response?: QueryResponse;
  error?: string;
  loading?: boolean;
}

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);

  const ask = async () => {
    const question = input.trim();
    if (!question) return;
    setInput("");
    setTurns((t) => [...t, { question, loading: true }]);
    try {
      const response = await api.askQuestion(question);
      setTurns((t) => {
        const copy = [...t];
        copy[copy.length - 1] = { question, response, loading: false };
        return copy;
      });
    } catch (e: any) {
      setTurns((t) => {
        const copy = [...t];
        copy[copy.length - 1] = { question, error: e.message, loading: false };
        return copy;
      });
    }
  };

  return (
    <div className="space-y-6 flex flex-col h-[calc(100vh-4rem)]">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Research Chat</h2>
        <p className="text-slate-500 mt-1">
          Ask questions grounded in your uploaded financial documents.
        </p>
      </div>

      <div className="flex-1 overflow-y-auto space-y-6 pr-1">
        {turns.length === 0 && (
          <Card className="text-sm text-slate-500">
            Try asking something like "What was revenue growth in the most recent fiscal year?"
            Answers are generated only from your uploaded, indexed documents.
          </Card>
        )}

        {turns.map((turn, i) => (
          <div key={i} className="space-y-3">
            <div className="flex justify-end">
              <div className="bg-brand-600 text-white rounded-2xl rounded-tr-sm px-4 py-2 max-w-lg text-sm">
                {turn.question}
              </div>
            </div>

            {turn.loading && (
              <Card className="text-sm text-slate-500">Retrieving evidence and generating an answer...</Card>
            )}

            {turn.error && (
              <Card className="border-red-200 bg-red-50 text-red-700 text-sm flex items-start gap-2">
                <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                {turn.error}
              </Card>
            )}

            {turn.response && (
              <Card>
                {turn.response.insufficient_evidence && (
                  <div className="mb-3 flex items-center gap-2 text-amber-700 bg-amber-50 rounded-lg px-3 py-2 text-xs">
                    <AlertTriangle size={14} /> Insufficient evidence in the indexed documents
                  </div>
                )}
                <p className="text-slate-800 whitespace-pre-wrap leading-relaxed">
                  {turn.response.answer}
                </p>
                <p className="text-xs text-slate-400 mt-3">
                  {turn.response.latency_ms.toFixed(0)} ms · {turn.response.retrieved_chunks.length} chunks retrieved
                </p>
                <CitationPanel
                  citations={turn.response.citations}
                  chunks={turn.response.retrieved_chunks}
                />
              </Card>
            )}
          </div>
        ))}
      </div>

      <div className="flex gap-3">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="Ask a question about your financial documents..."
          className="flex-1 rounded-xl border border-slate-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
        />
        <button
          onClick={ask}
          className="flex items-center gap-2 bg-brand-600 hover:bg-brand-700 text-white px-5 rounded-xl text-sm font-medium"
        >
          <Send size={16} /> Ask
        </button>
      </div>
    </div>
  );
}
