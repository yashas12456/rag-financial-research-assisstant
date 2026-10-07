"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import {
  LayoutDashboard,
  FileStack,
  UploadCloud,
  MessagesSquare,
  BarChart3,
  Settings as SettingsIcon,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/documents", label: "Documents", icon: FileStack },
  { href: "/upload", label: "Upload", icon: UploadCloud },
  { href: "/chat", label: "Research Chat", icon: MessagesSquare },
  { href: "/evaluation", label: "Evaluation", icon: BarChart3 },
  { href: "/settings", label: "Settings", icon: SettingsIcon },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 shrink-0 border-r border-slate-200 bg-white h-screen sticky top-0 flex flex-col">
      <div className="px-6 py-5 border-b border-slate-200">
        <h1 className="text-lg font-bold text-slate-900">RAG Research Analyst</h1>
        <p className="text-xs text-slate-500 mt-1">Financial Document Intelligence</p>
      </div>
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                active
                  ? "bg-brand-50 text-brand-700"
                  : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
              )}
            >
              <Icon size={18} />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="px-6 py-4 border-t border-slate-200 text-xs text-slate-400">
        Hybrid retrieval + reranking RAG pipeline
      </div>
    </aside>
  );
}
