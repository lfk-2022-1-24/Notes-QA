"use client";

import { useState, useCallback } from "react";

interface UploadZoneProps {
  onUploadComplete: () => void;
}

export default function UploadZone({ onUploadComplete }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string[]>([]);

  const uploadFiles = useCallback(
    async (files: FileList | File[]) => {
      setIsUploading(true);
      setUploadStatus([]);

      const formData = new FormData();
      for (const file of files) {
        formData.append("files", file);
      }

      try {
        const res = await fetch("/api/notes/upload", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();

        const statuses: string[] = [];
        for (const r of data.results) {
          if (r.error) {
            statuses.push(`❌ ${r.filename}: ${r.error}`);
          } else {
            statuses.push(`✓ ${r.filename} (${r.chunkCount} chunks)`);
          }
        }
        setUploadStatus(statuses);
        onUploadComplete();
      } catch (err) {
        setUploadStatus(["Upload failed: " + (err instanceof Error ? err.message : "Unknown error")]);
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
          accept=".md,.txt,.pdf"
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
              Supports .md, .txt, .pdf
            </p>
          </div>
        )}
      </div>
      {uploadStatus.length > 0 && (
        <div className="mt-2 text-sm space-y-1">
          {uploadStatus.map((status, i) => (
            <div key={i} className={status.startsWith("❌") ? "text-red-500" : "text-green-600"}>
              {status}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
