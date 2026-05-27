"use client";

import { useState, useCallback, useRef } from "react";

interface UploadZoneProps {
  onUploadComplete: () => void;
}

type UploadResult = {
  status: "success" | "error";
  summary: string;
  messages: string[];
};

export default function UploadZone({ onUploadComplete }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const lastFilesRef = useRef<File[]>([]);

  const uploadFiles = useCallback(
    async (files: FileList | File[]) => {
      setIsUploading(true);
      setUploadResult(null);

      const formData = new FormData();
      const fileArray = Array.from(files);
      lastFilesRef.current = fileArray;
      for (const file of fileArray) {
        formData.append("files", file);
      }

      try {
        const res = await fetch("/api/notes/upload", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();

        const statuses: string[] = [];
        let okCount = 0;
        let errorCount = 0;
        for (const r of data.results) {
          if (r.error) {
            errorCount += 1;
            statuses.push(`❌ ${r.filename}: ${r.error}`);
          } else {
            okCount += 1;
            statuses.push(`✓ ${r.filename} (${r.chunkCount} chunks)`);
          }
        }
        const status: UploadResult = {
          status: errorCount > 0 ? "error" : "success",
          summary:
            errorCount > 0
              ? `导入完成：成功 ${okCount} 个，失败 ${errorCount} 个。`
              : `导入成功：共 ${okCount} 个文件。`,
          messages: statuses,
        };
        setUploadResult(status);
        if (okCount > 0) {
          onUploadComplete();
        }
      } catch (err) {
        setUploadResult({
          status: "error",
          summary: "导入失败：请求未完成。",
          messages: ["错误原因: " + (err instanceof Error ? err.message : "Unknown error")],
        });
      } finally {
        setIsUploading(false);
      }
    },
    [onUploadComplete]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (e.dataTransfer.files.length > 0) {
        uploadFiles(e.dataTransfer.files);
      }
    },
    [uploadFiles]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        uploadFiles(e.target.files);
      }
    },
    [uploadFiles]
  );

  return (
    <div className="mb-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer ${
          isDragging
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 hover:border-gray-400"
        } ${isUploading ? "opacity-50 pointer-events-none" : ""}`}
        onClick={() => document.getElementById("file-input")?.click()}
      >
        <input
          id="file-input"
          type="file"
          multiple
          accept=".md,.txt,.pdf,.docx"
          className="hidden"
          onChange={handleFileSelect}
        />
        {isUploading ? (
          <div className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-5 w-5 text-blue-500" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span className="text-gray-600">Processing & embedding...</span>
          </div>
        ) : (
          <div>
            <p className="text-gray-600 font-medium">
              Drop files here or click to upload
            </p>
            <p className="text-gray-400 text-sm mt-1">
              Supports .md, .txt, .pdf, .docx
            </p>
          </div>
        )}
      </div>
      {uploadResult && (
        <div
          className={`mt-3 rounded-lg border px-3 py-2 text-sm ${
            uploadResult.status === "success"
              ? "border-green-200 bg-green-50 text-green-700"
              : "border-red-200 bg-red-50 text-red-700"
          }`}
        >
          <div className="font-medium">{uploadResult.summary}</div>
          <div className="mt-1 space-y-1">
            {uploadResult.messages.map((status, i) => (
              <div key={i}>{status}</div>
            ))}
          </div>
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              onClick={() => setUploadResult(null)}
              className={`px-2.5 py-1 rounded text-xs font-medium ${
                uploadResult.status === "success"
                  ? "bg-green-600 text-white hover:bg-green-700"
                  : "bg-red-600 text-white hover:bg-red-700"
              }`}
            >
              确认
            </button>
            {uploadResult.status === "error" && lastFilesRef.current.length > 0 && (
              <button
                type="button"
                onClick={() => uploadFiles(lastFilesRef.current)}
                className="px-2.5 py-1 rounded text-xs font-medium border border-red-200 text-red-700 bg-white hover:bg-red-50"
              >
                重试
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
