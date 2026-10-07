import clsx from "clsx";

const STYLES: Record<string, string> = {
  indexed: "bg-emerald-100 text-emerald-700",
  processing: "bg-amber-100 text-amber-700",
  uploaded: "bg-slate-100 text-slate-700",
  failed: "bg-red-100 text-red-700",
};

export default function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={clsx(
        "px-2 py-0.5 rounded-full text-xs font-medium capitalize",
        STYLES[status] || "bg-slate-100 text-slate-700"
      )}
    >
      {status}
    </span>
  );
}
