import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "RAG Financial Research Assistant",
  description: "Domain-specific RAG assistant for financial documents",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="flex-1 p-8 max-w-6xl mx-auto w-full">{children}</main>
        </div>
      </body>
    </html>
  );
}
