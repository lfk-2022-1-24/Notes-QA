"use client";

import { useState } from "react";
import UploadZone from "@/components/UploadZone";
import SourcePanel from "@/components/SourcePanel";
import ChatPanel from "@/components/ChatPanel";

interface HighlightRange {
  noteId: string;
  startChar: number;
  endChar: number;
}

export default function Home() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [highlight, setHighlight] = useState<HighlightRange | null>(null);

  const handleUploadComplete = () => {
    setRefreshKey((k) => k + 1);
  };

  const handleCitationClick = (range: HighlightRange) => {
    setHighlight(range);
    // Clear highlight after a few seconds
    setTimeout(() => setHighlight(null), 8000);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold text-gray-800">Notes QA</span>
          <span className="text-xs text-gray-400">Ask questions grounded in your notes</span>
        </div>
      </header>

      {/* Upload zone */}
      <div className="px-4 pt-3 shrink-0">
        <UploadZone onUploadComplete={handleUploadComplete} />
      </div>

      {/* Main content */}
      <div className="flex-1 flex min-h-0 px-4 pb-4 gap-4">
        {/* Source panel */}
        <div className="w-1/3 bg-white rounded-lg border border-gray-200 overflow-hidden">
          <SourcePanel refreshKey={refreshKey} highlight={highlight} />
        </div>

        {/* Chat panel */}
        <div className="flex-1 bg-white rounded-lg border border-gray-200 overflow-hidden">
          <ChatPanel onCitationClick={handleCitationClick} />
        </div>
      </div>
    </div>
  );
}
