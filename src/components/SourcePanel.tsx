"use client";

import { useState, useEffect, useCallback } from "react";

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
}

interface SourcePanelProps {
  refreshKey: number;
  highlight: HighlightRange | null;
}

export default function SourcePanel({ refreshKey, highlight }: SourcePanelProps) {
  const [notes, setNotes] = useState<Note[]>([]);
  const [expandedNote, setExpandedNote] = useState<string | null>(null);
  const [noteDetail, setNoteDetail] = useState<NoteDetail | null>(null);

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
    fetchNotes();
  }, [fetchNotes, refreshKey]);

  useEffect(() => {
    if (highlight?.noteId) {
      setExpandedNote(highlight.noteId);
      loadNoteDetail(highlight.noteId);
    }
  }, [highlight]);

  const loadNoteDetail = async (noteId: string) => {
    try {
      const res = await fetch(`/api/notes/${noteId}`);
      const data = await res.json();
      setNoteDetail(data);
    } catch (err) {
      console.error("Failed to load note detail:", err);
    }
  };

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

  const renderHighlightedContent = (content: string) => {
    if (!highlight) return <pre className="whitespace-pre-wrap text-sm">{content}</pre>;

    const { startChar, endChar } = highlight;
    if (startChar >= content.length) return <pre className="whitespace-pre-wrap text-sm">{content}</pre>;

    const before = content.slice(0, startChar);
    const highlighted = content.slice(startChar, endChar);
    const after = content.slice(endChar);

    return (
      <pre className="whitespace-pre-wrap text-sm">
        {before}
        <mark className="bg-yellow-200 px-0.5 rounded">{highlighted}</mark>
        {after}
      </pre>
    );
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-3 border-b border-gray-200">
        <h2 className="font-semibold text-gray-700">Notes ({notes.length})</h2>
      </div>
      <div className="flex-1 overflow-y-auto">
        {notes.length === 0 ? (
          <div className="p-4 text-gray-400 text-center text-sm">
            No notes uploaded yet
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {notes.map((note) => (
              <div key={note.id}>
                <div
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
