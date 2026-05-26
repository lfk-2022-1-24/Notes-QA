"use client";

import { useState, useEffect, useCallback, useRef } from "react";

interface Note {
  id: string;
  filename: string;
  file_type: string;
  created_at: string;
  chunk_count: string;
}

interface NoteDetail {
  note: { id: string; filename: string; content: string };
  chunks: { id: string; chunk_index: number; content: string; start_char: number; end_char: number }[];
}

interface HighlightRange {
  noteId: string;
  startChar: number;
  endChar: number;
  anchorText?: string;
  queryText?: string;
}

interface SourcePanelProps {
  refreshKey: number;
  highlight: HighlightRange | null;
}

export default function SourcePanel({ refreshKey, highlight }: SourcePanelProps) {
  const [notes, setNotes] = useState<Note[]>([]);
  const [expandedNote, setExpandedNote] = useState<string | null>(null);
  const [noteDetail, setNoteDetail] = useState<NoteDetail | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  const expandedHeaderRef = useRef<HTMLDivElement | null>(null);
  const highlightRef = useRef<HTMLElement | null>(null);

  const loadNoteDetail = useCallback(async (noteId: string) => {
    try {
      const res = await fetch(`/api/notes/${noteId}`);
      const data = await res.json();
      setNoteDetail(data);
    } catch (err) {
      console.error("Failed to load note detail:", err);
    }
  }, []);

  const fetchNotes = useCallback(async () => {
    try {
      const res = await fetch("/api/notes");
      const data = await res.json();
      setNotes(data.notes || []);
    } catch (err) {
      console.error("Failed to fetch notes:", err);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchNotes();
  }, [fetchNotes, refreshKey]);

  useEffect(() => {
    if (highlight?.noteId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setExpandedNote(highlight.noteId);
      loadNoteDetail(highlight.noteId);
    }
  }, [highlight, loadNoteDetail]);

  // Auto scroll to the expanded note + highlighted range.
  useEffect(() => {
    if (!highlight?.noteId) return;
    if (expandedNote !== highlight.noteId) return;
    if (!noteDetail || noteDetail.note.id !== highlight.noteId) return;

    // 1) Ensure the expanded note header is visible in the list.
    expandedHeaderRef.current?.scrollIntoView({ block: "nearest" });

    // 2) Scroll to highlighted text inside the note content.
    // Wait a tick so <mark> exists in the DOM.
    const t = window.setTimeout(() => {
      highlightRef.current?.scrollIntoView({ block: "center" });
    }, 0);

    return () => window.clearTimeout(t);
  }, [highlight, expandedNote, noteDetail]);

  const toggleNote = (noteId: string) => {
    if (expandedNote === noteId) {
      setExpandedNote(null);
      setNoteDetail(null);
    } else {
      setExpandedNote(noteId);
      loadNoteDetail(noteId);
    }
  };

  const deleteNote = async (noteId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Delete this note and all its chunks?")) return;
    try {
      await fetch(`/api/notes/${noteId}`, { method: "DELETE" });
      if (expandedNote === noteId) {
        setExpandedNote(null);
        setNoteDetail(null);
      }
      fetchNotes();
    } catch (err) {
      console.error("Failed to delete note:", err);
    }
  };

  const extractQueryTokens = (q: string): string[] => {
    const text = (q || "")
      .trim()
      .replace(/\s+/g, " ")
      .replace(/[?？!！。，,;；:："'“”‘’（）()【】\[\]{}<>]/g, " ");
    if (!text) return [];
    const stop = new Set([
      "什么",
      "为什么",
      "怎么",
      "如何",
      "是不是",
      "是否",
      "需要",
      "应该",
      "的",
      "吗",
      "it",
      "they",
      "this",
      "that",
    ]);
    const toks = new Set<string>();
    for (const m of text.matchAll(/[\p{Script=Han}A-Za-z0-9]{2,}/gu)) {
      const t = m[0];
      if (!t || stop.has(t)) continue;
      toks.add(t);
    }
    return Array.from(toks).sort((a, b) => b.length - a.length).slice(0, 8);
  };

  const countTokenHits = (haystack: string, tokens: string[]): number => {
    if (tokens.length === 0) return 0;
    let hits = 0;
    for (const t of tokens) {
      if (haystack.includes(t)) hits++;
    }
    return hits;
  };

  const renderHighlightedContent = (content: string) => {
    // Normalize CRLF -> LF so backend offsets (which are typically computed on LF text)
    // align with what we render/highlight in the UI.
    const raw = content || "";
    const doc = raw.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

    // Only highlight inside the cited note. Prevents unrelated highlights
    // when user expands other notes while a highlight is still active.
    if (!highlight || highlight.noteId !== noteDetail?.note.id) {
      return <pre className="whitespace-pre-wrap text-sm">{doc}</pre>;
    }

    let { startChar, endChar } = highlight;

    // Map backend offsets (raw string indices) to UI indices (LF-normalized).
    // Only CRLF removes one char, so we subtract the number of "\r\n" pairs before the offset.
    const rawToDocIndex = (idxRaw: number) => {
      if (!Number.isFinite(idxRaw) || idxRaw <= 0) return 0;
      const max = Math.min(raw.length, Math.floor(idxRaw));
      let removed = 0;
      for (let i = 0; i < max - 1; i++) {
        if (raw[i] === "\r" && raw[i + 1] === "\n") removed++;
      }
      return Math.max(0, idxRaw - removed);
    };

    startChar = rawToDocIndex(startChar);
    endChar = rawToDocIndex(endChar);

    const isRangeValid =
      Number.isFinite(startChar) &&
      Number.isFinite(endChar) &&
      startChar >= 0 &&
      endChar > startChar &&
      endChar <= doc.length;

    const isDateQuery =
      /\b20\d{2}-\d{2}-\d{2}\b/.test(highlight.queryText || "") ||
      /\b20\d{2}年\d{1,2}月\d{1,2}[日号]\b/.test(highlight.queryText || "") ||
      /\b20\d{6}\b/.test(highlight.queryText || "");
    const anchorLooksLikeLogHeading = (() => {
      const a = (highlight.anchorText || "").trim();
      return (
        /^#?\s*日志\s*20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b/.test(a) ||
        /^#?\s*日志\s*20\d{2}年\d{1,2}月\d{1,2}[日号]\b/.test(a)
      );
    })();

    // Stronger anchor alignment:
    // - only search for anchor within a window around the provided offsets (or start of doc as fallback)
    // - require the matched location to contain at least one query token (prevents unrelated matches)
    //
    // IMPORTANT: for date-scoped log queries, trust backend offsets to avoid "drift" across repeated templates.
    if (highlight.anchorText) {
      if (isDateQuery || anchorLooksLikeLogHeading) {
        // Skip anchor-based realignment for log/date queries.
      } else {
      const anchor = highlight.anchorText.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim().slice(0, 260);
      if (anchor.length >= 24) {
        const tokens = extractQueryTokens(highlight.queryText || "");
        const center = isRangeValid ? Math.floor((startChar + endChar) / 2) : 0;
        const winStart = Math.max(0, center - 3000);
        const winEnd = Math.min(doc.length, center + 3000);
        const windowText = doc.slice(winStart, winEnd);

        const localIdx = windowText.indexOf(anchor);
        if (localIdx >= 0) {
          const absIdx = winStart + localIdx;
          const localSecond = windowText.indexOf(anchor, localIdx + 1);
          const uniqueInWindow = localSecond < 0;

          // Verify token hits around match.
          const verifyStart = Math.max(0, absIdx - 200);
          const verifyEnd = Math.min(doc.length, absIdx + anchor.length + 600);
          const verifyText = doc.slice(verifyStart, verifyEnd);
          const okByTokens = tokens.length === 0 ? true : countTokenHits(verifyText, tokens) >= 1;

          // Hard guard: never allow anchor-based alignment to move far away from backend offsets.
          // This prevents drifting upwards into repeated "template" sections in txt notes.
          const closeToProvided =
            !isRangeValid || Math.abs(absIdx - startChar) <= 600;

          if (uniqueInWindow && okByTokens && closeToProvided) {
            // Only adjust the start position; preserve the intended highlight length
            // from backend to avoid over-highlighting for docx/doc paragraphs.
            const len = Number.isFinite(endChar) && Number.isFinite(startChar) ? Math.max(40, endChar - startChar) : anchor.length;
            startChar = absIdx;
            endChar = Math.min(doc.length, absIdx + Math.min(anchor.length, len));
          }
        }
      }
      }
    }

    if (!Number.isFinite(startChar) || startChar < 0) startChar = 0;
    if (!Number.isFinite(endChar) || endChar <= startChar) endChar = Math.min(doc.length, startChar + 200);
    // Final UI safeguard: never highlight an overly large span.
    if (endChar - startChar > 220) endChar = startChar + 220;
    if (startChar >= doc.length) return <pre className="whitespace-pre-wrap text-sm">{doc}</pre>;

    const before = doc.slice(0, startChar);
    const highlighted = doc.slice(startChar, endChar);
    const after = doc.slice(endChar);

    return (
      <pre className="whitespace-pre-wrap text-sm">
        {before}
        <mark
          ref={(el) => {
            highlightRef.current = el;
          }}
          className="bg-yellow-200 px-0.5 rounded"
        >
          {highlighted}
        </mark>
        {after}
      </pre>
    );
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-3 border-b border-gray-200">
        <h2 className="font-semibold text-gray-700">Notes ({notes.length})</h2>
      </div>
      <div ref={listRef} className="flex-1 overflow-y-auto">
        {notes.length === 0 ? (
          <div className="p-4 text-gray-400 text-center text-sm">
            No notes uploaded yet
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {notes.map((note) => (
              <div key={note.id}>
                <div
                  ref={note.id === expandedNote ? expandedHeaderRef : undefined}
                  className="flex items-center justify-between p-3 hover:bg-gray-50 cursor-pointer"
                  onClick={() => toggleNote(note.id)}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-xs px-1.5 py-0.5 bg-gray-100 rounded text-gray-500 uppercase">
                      {note.file_type}
                    </span>
                    <span className="text-sm truncate">{note.filename}</span>
                    <span className="text-xs text-gray-400">
                      {note.chunk_count} chunks
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      onClick={(e) => deleteNote(note.id, e)}
                      className="text-gray-400 hover:text-red-500 text-xs"
                      title="Delete"
                    >
                      ✕
                    </button>
                    <span className="text-gray-400 text-xs">
                      {expandedNote === note.id ? "▼" : "▶"}
                    </span>
                  </div>
                </div>
                {expandedNote === note.id && noteDetail && noteDetail.note.id === note.id && (
                  <div className="px-3 pb-3 bg-gray-50/50 border-t border-gray-100">
                    {renderHighlightedContent(noteDetail.note.content)}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
