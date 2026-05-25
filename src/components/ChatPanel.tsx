"use client";

import { useEffect, useMemo, useRef, useState } from "react";

interface Source {
  index: number;
  noteId: string;
  filename: string;
  content: string;
  startChar: number;
  endChar: number;
  similarity: number;
}

interface HighlightRange {
  noteId: string;
  startChar: number;
  endChar: number;
  anchorText?: string;
}

interface ChatPanelProps {
  onCitationClick: (highlight: HighlightRange) => void;
}

interface HistoryTurn {
  question: string;
  answer: string;
}

interface ChatMessage {
  id: string;
  question: string;
  answer: string | null;
  sources: Source[];
  hasAnswer: boolean;
  error: string | null;
  createdAt: number;
}

const STORAGE_KEY = "notes-qa:chat-history:v1";

export default function ChatPanel({ onCitationClick }: ChatPanelProps) {
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const extractCitedSourceIndices = (answerText: string): Set<number> => {
    const set = new Set<number>();
    if (!answerText) return set;
    for (const m of answerText.matchAll(/\[(\d+)\]/g)) {
      const n = Number.parseInt(m[1], 10);
      if (Number.isFinite(n)) set.add(n);
    }
    return set;
  };

  const historyForFollowups = useMemo<HistoryTurn[]>(() => {
    return messages
      .filter((m) => typeof m.answer === "string" && m.answer.trim().length > 0)
      .slice(-6)
      .map((m) => ({ question: m.question, answer: m.answer! }));
  }, [messages]);

  // Load persisted messages on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as ChatMessage[];
      if (!Array.isArray(parsed)) return;
      setMessages(
        parsed
          .filter(
            (m) =>
              m &&
              typeof m.id === "string" &&
              typeof m.question === "string" &&
              typeof m.createdAt === "number"
          )
          .slice(-50)
      );
    } catch {
      // ignore
    }
  }, []);

  // Persist messages
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-50)));
    } catch {
      // ignore
    }
  }, [messages]);

  // Auto-scroll to newest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  const handleAsk = async () => {
    if (!question.trim() || isLoading) return;

    const currentQuestion = question.trim();
    setQuestion("");
    setIsLoading(true);

    const msgId =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

    const pending: ChatMessage = {
      id: msgId,
      question: currentQuestion,
      answer: null,
      sources: [],
      hasAnswer: true,
      error: null,
      createdAt: Date.now(),
    };

    setMessages((ms) => [...ms, pending]);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentQuestion, history: historyForFollowups }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Request failed");
      }

      const data = await res.json();
      setMessages((ms) =>
        ms.map((m) =>
          m.id === msgId
            ? {
                ...m,
                answer: typeof data.answer === "string" ? data.answer : "",
                sources: data.sources || [],
                hasAnswer: Boolean(data.hasAnswer),
                error: null,
              }
            : m
        )
      );
    } catch (err) {
      setMessages((ms) =>
        ms.map((m) =>
          m.id === msgId
            ? {
                ...m,
                answer: "",
                sources: [],
                hasAnswer: true,
                error: err instanceof Error ? err.message : "Failed to get answer",
              }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  const renderAnswer = (text: string, msgSources: Source[]) => {
    if (!text) return null;

    // Light cleanup: if model outputs markdown markers, hide them rather than showing raw '*'.
    // (We still rely on citations like [1] to be clickable.)
    const cleaned = text
      .replace(/\r\n/g, "\n")
      .replace(/^\s*[-*+]\s+/gm, "")
      .replace(/^\s*\d+\.\s+/gm, "")
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/\*([^*]+)\*/g, "$1")
      .replace(/__([^_]+)__/g, "$1")
      .replace(/_([^_]+)_/g, "$1");

    // Split by citation patterns like [1], [2], [1][3] etc.
    const parts = cleaned.split(/(\[\d+\])/g);

    return parts.map((part, i) => {
      const citationMatch = part.match(/^\[(\d+)\]$/);
      if (citationMatch) {
        const sourceIndex = parseInt(citationMatch[1]);
        const source = msgSources.find((s) => s.index === sourceIndex);
        if (source) {
          return (
            <button
              key={i}
              onClick={() =>
                onCitationClick({
                  noteId: source.noteId,
                  startChar: source.startChar,
                  endChar: source.endChar,
                  anchorText: source.content.slice(0, 600),
                })
              }
              className="inline-flex items-center px-1 py-0.5 text-xs font-medium bg-blue-100 text-blue-700 rounded hover:bg-blue-200 cursor-pointer"
              title={`Source: ${source.filename} (similarity: ${(source.similarity * 100).toFixed(1)}%)`}
            >
              [{sourceIndex}]
            </button>
          );
        }
      }
      return <span key={i}>{part}</span>;
    });
  };

  const clearHistory = () => {
    setMessages([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-3 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-gray-700">Ask Questions</h2>
          <button
            type="button"
            onClick={clearHistory}
            className="text-xs text-gray-400 hover:text-gray-600"
            title="Clear chat history"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Answer area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && !isLoading && (
          <div className="text-gray-400 text-center mt-12">
            <p className="text-lg mb-2">Upload notes, then ask questions</p>
            <p className="text-sm">Answers will be grounded in your notes with clickable citations</p>
          </div>
        )}

        {messages.map((m) => (
          <div key={m.id} className="space-y-2">
            <div className="text-sm text-gray-500 font-medium">Question: {m.question}</div>

            {m.error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                {m.error}
              </div>
            )}

            {!m.error && m.answer === null && (
              <div className="flex items-center gap-2 text-gray-500">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span>Searching notes & generating answer...</span>
              </div>
            )}

            {!m.error && typeof m.answer === "string" && (
              <div
                className={`p-4 rounded-lg ${
                  m.hasAnswer ? "bg-white border border-gray-200" : "bg-amber-50 border border-amber-200"
                }`}
              >
                <div className="prose prose-sm max-w-none text-gray-800 leading-relaxed">
                  {renderAnswer(m.answer, m.sources)}
                </div>
                {!m.hasAnswer && (
                  <div className="mt-2 text-xs text-amber-600">
                    No relevant sources found in your notes for this question.
                  </div>
                )}
              </div>
            )}

            {/* Sources */}
            {(() => {
              if (m.sources.length === 0) return null;
              const cited = extractCitedSourceIndices(typeof m.answer === "string" ? m.answer : "");
              const displaySources = m.sources
                .filter((s) => cited.has(s.index))
                .sort((a, b) => a.index - b.index);
              if (displaySources.length === 0) return null;

              return (
              <div className="space-y-2">
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Sources</h3>
                {displaySources.map((source) => (
                  <div
                    key={`${m.id}-${source.index}`}
                    className="p-2 bg-gray-50 rounded border border-gray-100 cursor-pointer hover:bg-gray-100 transition-colors"
                    onClick={() =>
                      onCitationClick({
                        noteId: source.noteId,
                        startChar: source.startChar,
                        endChar: source.endChar,
                        anchorText: source.content.slice(0, 600),
                      })
                    }
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-gray-600">
                        [{source.index}] {source.filename}
                      </span>
                      <span className="text-xs text-gray-400">
                        {(source.similarity * 100).toFixed(1)}% match
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 line-clamp-2">{source.content.slice(0, 200)}...</p>
                  </div>
                ))}
              </div>
              );
            })()}
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="p-3 border-t border-gray-200">
        <div className="flex gap-2">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your notes..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={2}
            disabled={isLoading}
          />
          <button
            onClick={handleAsk}
            disabled={isLoading || !question.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed self-end"
          >
            {isLoading ? "..." : "Ask"}
          </button>
        </div>
      </div>
    </div>
  );
}
