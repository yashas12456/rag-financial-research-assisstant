"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import Card from "@/components/Card";
import { api } from "@/services/api";
import { UploadCloud, CheckCircle2, XCircle } from "lucide-react";

export default function UploadPage() {
  const router = useRouter();
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
  const [message, setMessage] = useState<string>("");

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setStatus("error");
      setMessage("Only PDF files are supported.");
      return;
    }
    setStatus("uploading");
    try {
      const result = await api.uploadDocument(file);
      setStatus("success");
      setMessage(`Uploaded "${result.document.filename}". Indexing it now...`);
      await api.indexDocument(result.document.id);
      setMessage(`"${result.document.filename}" uploaded and indexed successfully.`);
      setTimeout(() => router.push("/documents"), 1200);
    } catch (e: any) {
      setStatus("error");
      setMessage(e.message || "Upload failed.");
    }
  }, [router]);

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Upload Document</h2>
        <p className="text-slate-500 mt-1">
          Upload an annual report, earnings release, or investor presentation (PDF).
        </p>
      </div>

      <Card>
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files?.[0];
            if (file) handleFile(file);
          }}
          className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors ${
            dragging ? "border-brand-500 bg-brand-50" : "border-slate-300"
          }`}
        >
          <UploadCloud className="mx-auto text-slate-400 mb-3" size={40} />
          <p className="text-slate-600 mb-2">Drag & drop a PDF here, or</p>
          <label className="inline-block cursor-pointer text-brand-600 font-medium hover:text-brand-700">
            browse files
            <input
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
              }}
            />
          </label>
          <p className="text-xs text-slate-400 mt-3">PDF only. Max 50MB.</p>
        </div>

        {status !== "idle" && (
          <div
            className={`mt-4 flex items-center gap-2 text-sm p-3 rounded-lg ${
              status === "error"
                ? "bg-red-50 text-red-700"
                : status === "success"
                ? "bg-emerald-50 text-emerald-700"
                : "bg-slate-50 text-slate-600"
            }`}
          >
            {status === "success" && <CheckCircle2 size={16} />}
            {status === "error" && <XCircle size={16} />}
            {message || "Uploading..."}
          </div>
        )}
      </Card>
    </div>
  );
}
