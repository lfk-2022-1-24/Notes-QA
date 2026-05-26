import { NextRequest, NextResponse } from "next/server";
import { getEmbedding } from "@/lib/embedding";
import { askQuestion, HistoryTurn, rewriteStandaloneQuestion, SourceChunk } from "@/lib/llm";
import { query } from "@/lib/db";
import { normalizeForEmbedding } from "@/lib/text-normalize";

// Retrieval tuning:
// - Pull more candidates, then filter with an adaptive threshold.
// - Keep at least 1 result to avoid false "no answer" when the DB is small.
const TOP_K = 20;
const MIN_KEEP = 1;
const BASE_THRESHOLD = 0.2;
const GAP_THRESHOLD = 0.12;
const KEYWORD_K = 60;
const LLM_TOP_K = 8;

type SearchRow = {
  id: string;
  note_id: string;
  content: string;
  start_char: string | number;
  end_char: string | number;
  filename: string;
  similarity: number;
};

function buildKeywordPatterns(question: string): string[] {
  const q = question
    .trim()
    .replace(/\s+/g, "")
    .replace(/[?？!！。，,;；:："'“”‘’（）()【】\[\]{}<>]/g, "");

  // Pull likely "topic words" without requiring Chinese word segmentation.
  const stop = new Set([
    "如何",
    "怎么",
    "怎样",
    "为什么",
    "什么",
    "哪些",
    "是否",
    "可以",
    "需要",
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "一个",
    "保证",
  ]);

  const candidates = new Set<string>();

  // 1) Split by common particle "的"
  const parts = q.split("的").filter(Boolean);

  for (const p of parts) {
    const s = p.trim();
    if (s.length >= 2 && !stop.has(s)) {
      candidates.add(s);
      // If it's long, also add its tail/head for better ILIKE match
      if (s.length >= 6) {
        candidates.add(s.slice(-4));
        candidates.add(s.slice(0, 4));
      }
    }
  }

  // 2) Extract any long Han/alpha-numeric sequences.
  for (const m of q.matchAll(/[\p{Script=Han}A-Za-z0-9]{2,}/gu)) {
    const s = m[0];
    if (s.length >= 2 && !stop.has(s)) {
      candidates.add(s);
      if (s.length >= 6) {
        candidates.add(s.slice(-4));
        candidates.add(s.slice(0, 4));
      }
    }
  }

  // 3) Chinese n-gram (2~3 chars) for better recall on Q/A style notes.
  const hanOnly = (q.match(/\p{Script=Han}+/gu) || []).join("");
  for (let n = 3; n >= 2; n--) {
    for (let i = 0; i + n <= hanOnly.length; i++) {
      const g = hanOnly.slice(i, i + n);
      if (!stop.has(g) && g.length >= 2) candidates.add(g);
    }
  }

  // Prefer longer tokens; cap to avoid overly broad OR queries.
  const sorted = Array.from(candidates).sort((a, b) => b.length - a.length);
  return sorted.slice(0, 10).map((t) => `%${t}%`);
}

function stripPunctuation(s: string): string {
  return s.replace(/[?？!！。，,;；:："'“”‘’（）()【】\[\]{}<>]/g, "");
}

function normalizeLooseContains(s: string): string {
  return (s || "")
    .replace(/\s+/g, "")
    .replace(/[?？!！。，,;；:："'“”‘’（）()【】\[\]{}<>]/g, "")
    .toLowerCase();
}

function includesLoose(haystack: string, needle: string): boolean {
  const n = normalizeLooseContains(needle);
  if (!n) return false;
  const h = normalizeLooseContains(haystack);
  return h.includes(n);
}

type QuestionFocus = {
  topic?: string;
  aspect?: "advantages" | "disadvantages";
  aspectWord?: string; // original (e.g. "优点"/"缺点")
};

function extractDefinitionTopic(question: string): string | null {
  const q = question.trim();
  if (!q) return null;
  // Chinese: "索引是什么" / "索引是什么？" / "什么是索引"
  const m1 = q.match(/^(.+?)\s*(?:是啥|是什么)\s*[?？]*$/);
  if (m1) {
    const topic = m1[1].trim();
    return topic.length >= 2 ? topic : null;
  }
  const m2 = q.match(/(?:什么是)\s*(.+?)\??$/);
  if (m2) {
    const topic = m2[1].trim();
    return topic.length >= 2 ? topic : null;
  }
  // English: "what is X"
  const m3 = q.match(/\bwhat\s+is\s+(.+?)\??$/i);
  if (m3) {
    const topic = m3[1].trim();
    return topic.length >= 2 ? topic : null;
  }
  return null;
}

function hasDefinitionCue(content: string, topic: string): boolean {
  const c = normalizeLooseContains(content);
  const t = normalizeLooseContains(topic);
  if (!t) return false;

  // Direct definition patterns
  if (c.includes(`${t}是`) || c.includes(`${t}指`) || c.includes(`${t}意味着`)) return true;
  if (c.includes(`什么是${t}`) || c.includes(`${t}是什么`)) return true;

  // Meta definition cues (common in docx/notes)
  if (c.includes("定义") || c.includes("含义") || c.includes("内涵") || c.includes("解释") || c.includes("概念")) return true;
  return false;
}

function refineRangeByDefinitionTopic(
  chunkContent: string,
  chunkStartAbs: number,
  topic: string
): { content: string; startChar: number; endChar: number } | null {
  const text = chunkContent.replace(/\r\n/g, "\n");
  const t = topic.trim();
  if (!t || t.length < 2) return null;

  const candidates = [
    `什么是${t}`,
    `${t}是什么`,
    `**什么是${t}**`,
    `**${t}是什么**`,
    `## ${t}`,
    `### ${t}`,
  ];

  let idx = -1;
  for (const c of candidates) {
    const i = text.indexOf(c);
    if (i >= 0 && (idx < 0 || i < idx)) idx = i;
  }
  if (idx < 0) return null;

  // For definition-style queries, prefer a sentence-level highlight around "X是/指/意味着"
  // rather than a whole paragraph (docx paragraphs can be long).
  const matchIdx = (() => {
    const candidates = [`${t}是`, `${t}指`, `${t}意味着`];
    for (const c of candidates) {
      const i = text.indexOf(c, idx);
      if (i >= 0) return i;
    }
    const any = text.indexOf(t, idx);
    return any >= 0 ? any : idx;
  })();

  const { startIdx, endIdx } = getSentenceBounds(text, matchIdx, { maxLen: 220 });
  const snippet = text.slice(startIdx, endIdx);
  if (!snippet.trim()) return null;

  return { content: snippet, startChar: chunkStartAbs + startIdx, endChar: chunkStartAbs + endIdx };
}

function extractQuestionNumber(question: string): number | null {
  const q = question.trim();
  if (!q) return null;
  // Common Chinese phrasing: "第14题" / "第14个问题" / "第14问"
  const m1 = q.match(/第\s*(\d{1,4})\s*(?:题|个问题|问题|问)\b/);
  if (m1) return Number.parseInt(m1[1], 10);
  // Sometimes: "14题是什么" / "14. 什么是..."
  const m2 = q.match(/(?:^|[^\d])(\d{1,4})\s*(?:题|问)\b/);
  if (m2) return Number.parseInt(m2[1], 10);
  return null;
}

function refineRangeByNumber(
  chunkContent: string,
  chunkStartAbs: number,
  n: number
): { content: string; startChar: number; endChar: number } | null {
  const text = chunkContent.replace(/\r\n/g, "\n");

  const startPatterns: RegExp[] = [
    new RegExp(`(^|\\n)\\s*(?:Q\\s*)?${n}\\s*[:：.．、)）]`, "m"),
    new RegExp(`(^|\\n)\\s*第\\s*${n}\\s*(?:题|问|个问题|问题)\\s*[:：.．、)）]?`, "m"),
    // Loose: "14 " (followed by punctuation) in Q/A PDFs
    new RegExp(`(^|\\n)\\s*${n}\\s*[.．、)]`, "m"),
  ];

  let startIdx: number | null = null;
  for (const re of startPatterns) {
    const m = text.match(re);
    if (!m || m.index === undefined) continue;
    const prefix = m[1] ?? "";
    const idx = m.index + prefix.length;
    if (startIdx === null || idx < startIdx) startIdx = idx;
  }
  if (startIdx === null) return null;

  const afterStart = startIdx + 1;
  const nextCandidates: number[] = [];

  // Next Arabic-number question header
  const anyArabic = /(^|\n)\s*(?:Q\s*)?(\d{1,4})\s*[:：.．、)）]/gm;
  for (const m of text.slice(afterStart).matchAll(anyArabic)) {
    const num = Number.parseInt(m[2], 10);
    if (num === n) continue;
    const prefix = m[1] ?? "";
    const idx = afterStart + (m.index ?? 0) + prefix.length;
    nextCandidates.push(idx);
    break; // first is closest
  }

  // Next Chinese "第xx题/问"
  const anyChinese = /(^|\n)\s*第\s*(\d{1,4})\s*(?:题|问|个问题|问题)\b/gm;
  for (const m of text.slice(afterStart).matchAll(anyChinese)) {
    const num = Number.parseInt(m[2], 10);
    if (num === n) continue;
    const prefix = m[1] ?? "";
    const idx = afterStart + (m.index ?? 0) + prefix.length;
    nextCandidates.push(idx);
    break;
  }

  // Paragraph boundary as fallback end marker
  const paraBreak = text.slice(afterStart).search(/\n\s*\n/);
  if (paraBreak >= 0) nextCandidates.push(afterStart + paraBreak);

  let endIdx =
    nextCandidates.length > 0 ? Math.min(...nextCandidates) : Math.min(text.length, startIdx + 700);

  if (endIdx <= startIdx) endIdx = Math.min(text.length, startIdx + 700);

  // Trim whitespace, but keep the leading question marker itself.
  let localStart = startIdx;
  let localEnd = endIdx;
  while (localEnd > localStart && /\s/.test(text[localEnd - 1]!)) localEnd--;
  while (localStart < localEnd && text[localStart] === "\n") localStart++;

  const snippet = text.slice(localStart, localEnd);
  if (!snippet.trim()) return null;

  return {
    content: snippet,
    startChar: chunkStartAbs + localStart,
    endChar: chunkStartAbs + localEnd,
  };
}

function extractMatchTokensFromQuestion(question: string): string[] {
  const q = question
    .trim()
    .replace(/\s+/g, " ")
    .replace(/[?？!！。，,;；:："'“”‘’（）()【】\[\]{}<>]/g, " ");

  const stop = new Set([
    "什么",
    "为什么",
    "怎么",
    "如何",
    "是不是",
    "是否",
    "听到",
    "需要",
    "应该",
    "要不要",
    "要",
    "的",
    "吗",
    // Very generic tokens that often appear everywhere; they skew "earliest match" to the document start.
    "agent",
    "Agent",
    "llm",
    "LLM",
    "模型",
  ]);

  const tokens = new Set<string>();
  for (const m of q.matchAll(/[\p{Script=Han}A-Za-z0-9]{2,}/gu)) {
    const t = m[0].trim();
    if (!t || stop.has(t)) continue;
    tokens.add(t);
  }
  // Prefer longer tokens first (e.g. "国歌" > "什么")
  return Array.from(tokens).sort((a, b) => b.length - a.length).slice(0, 12);
}

function findBestKeywordMatchIndex(text: string, tokens: string[]): number | null {
  if (tokens.length === 0) return null;

  // Collect limited occurrences for each token.
  const positionsByToken = new Map<string, number[]>();
  for (const t of tokens) {
    const pos: number[] = [];
    let from = 0;
    let guard = 0;
    while (from < text.length && guard < 25) {
      const idx = text.indexOf(t, from);
      if (idx < 0) break;
      pos.push(idx);
      from = idx + Math.max(1, Math.floor(t.length / 2));
      guard++;
    }
    if (pos.length > 0) positionsByToken.set(t, pos);
  }

  const candidates: number[] = [];
  for (const pos of positionsByToken.values()) candidates.push(...pos);
  if (candidates.length === 0) return null;

  const uniqCandidates = Array.from(new Set(candidates)).sort((a, b) => a - b).slice(0, 200);

  // Score each candidate by how many (and how "specific") tokens appear in a window around it.
  const windowBefore = 260;
  const windowAfter = 900;
  let bestIdx: number | null = null;
  let bestScore = -Infinity;
  let bestHitCount = -Infinity;

  const tokenFreq = (t: string) => positionsByToken.get(t)?.length ?? 0;

  for (const idx of uniqCandidates) {
    const start = Math.max(0, idx - windowBefore);
    const end = Math.min(text.length, idx + windowAfter);

    let score = 0;
    let hitCount = 0;
    for (const t of tokens) {
      const occurrences = positionsByToken.get(t);
      if (!occurrences) continue;
      // Is there any occurrence within [start, end)?
      let present = false;
      for (const p of occurrences) {
        if (p < start) continue;
        if (p >= end) break;
        present = true;
        break;
      }
      if (!present) continue;
      hitCount++;
      // Longer tokens are more specific; downweight very frequent tokens.
      const freq = tokenFreq(t);
      const freqPenalty = freq >= 8 ? 0.35 : freq >= 4 ? 0.2 : 0;
      score += Math.max(1, Math.min(8, t.length)) * (1 - freqPenalty);
    }

    // Prefer matches that are not at the very start when score ties.
    const startBiasPenalty = idx < 120 ? 0.8 : 0;
    score -= startBiasPenalty;

    if (
      score > bestScore ||
      (score === bestScore && hitCount > bestHitCount) ||
      (score === bestScore && hitCount === bestHitCount && bestIdx !== null && idx > bestIdx)
    ) {
      bestScore = score;
      bestHitCount = hitCount;
      bestIdx = idx;
    }
  }

  return bestIdx;
}

function getTightBounds(text: string, matchIdx: number): { startIdx: number; endIdx: number } {
  const t = text;
  const idx = Math.max(0, Math.min(matchIdx, t.length));

  // Prefer paragraph bounds (docx/raw text uses double newlines per paragraph).
  const paraSep = /\n\s*\n/;
  const before = t.slice(0, idx);
  const after = t.slice(idx);

  const prevPara = before.lastIndexOf("\n\n");
  let paraStart = prevPara >= 0 ? prevPara + 2 : 0;
  const nextParaRel = after.search(paraSep);
  let paraEnd = nextParaRel >= 0 ? idx + nextParaRel : t.length;

  // If paragraph is too long, fall back to sentence-ish bounds.
  const paraLen = paraEnd - paraStart;
  if (paraLen > 480) {
    // Sentence punctuation boundaries (CN + EN) or line breaks.
    const sentStops = /[。！？!?；;]\s|\n/;
    // Find sentence start
    let sStart = Math.max(0, idx - 1);
    while (sStart > 0) {
      const c = t[sStart - 1]!;
      if (c === "\n") break;
      if (/[。！？!?；;]/.test(c)) break;
      sStart--;
      if (idx - sStart > 260) break;
    }
    // Find sentence end
    let sEnd = idx;
    while (sEnd < t.length) {
      const c = t[sEnd]!;
      if (c === "\n" || /[。！？!?；;]/.test(c)) {
        sEnd++;
        break;
      }
      sEnd++;
      if (sEnd - idx > 320) break;
    }

    // Ensure minimum length
    if (sEnd - sStart < 60) {
      sStart = Math.max(0, idx - 140);
      sEnd = Math.min(t.length, idx + 260);
    }
    return { startIdx: sStart, endIdx: sEnd };
  }

  // Trim surrounding whitespace.
  while (paraStart < paraEnd && (t[paraStart] === "\n" || t[paraStart] === " ")) paraStart++;
  while (paraEnd > paraStart && /\s/.test(t[paraEnd - 1]!)) paraEnd--;

  // Keep at least a small window.
  if (paraEnd - paraStart < 40) {
    const s = Math.max(0, idx - 140);
    const e = Math.min(t.length, idx + 260);
    return { startIdx: s, endIdx: e };
  }

  return { startIdx: paraStart, endIdx: paraEnd };
}

function getSentenceBounds(
  text: string,
  matchIdx: number,
  opts?: { maxLen?: number }
): { startIdx: number; endIdx: number } {
  const t = text;
  const idx = Math.max(0, Math.min(matchIdx, t.length));
  const maxLen = opts?.maxLen ?? 280;
  const softLen = Math.min(maxLen, 170);

  // Expand to nearest punctuation / newline boundaries.
  let start = idx;
  while (start > 0) {
    const c = t[start - 1]!;
    if (c === "\n") break;
    if (/[。！？!?；;]/.test(c)) break;
    start--;
    if (idx - start > Math.floor(maxLen * 0.6)) break;
  }

  let end = idx;
  while (end < t.length) {
    const c = t[end]!;
    if (c === "\n" || /[。！？!?；;]/.test(c)) {
      end++;
      break;
    }
    end++;
    if (end - start > maxLen) break;
  }

  // If the sentence is still long, try to cut at a comma near softLen.
  if (end - start > softLen) {
    const window = t.slice(start, Math.min(t.length, start + maxLen));
    const hardStop = window.search(/[。！？!?；;\n]/);
    if (hardStop >= 0 && hardStop >= 30) {
      end = start + hardStop + 1;
    } else {
      const commaIdx = window.slice(0, softLen + 60).search(/[，,]/);
      if (commaIdx >= 0 && commaIdx >= 60) {
        end = start + commaIdx + 1;
      } else {
        end = Math.min(t.length, start + softLen);
      }
    }
  }

  // Minimum window
  if (end - start < 60) {
    start = Math.max(0, idx - 140);
    end = Math.min(t.length, idx + 220);
  }

  // Trim whitespace
  while (start < end && (t[start] === "\n" || t[start] === " ")) start++;
  while (end > start && /\s/.test(t[end - 1]!)) end--;

  return { startIdx: start, endIdx: end };
}

function getAllQuestionHeaderPositions(text: string): number[] {
  const headers: number[] = [];
  const patterns: RegExp[] = [
    /(^|\n)\s*(?:Q\s*)?\d{1,4}\s*[:：.．、)）]/gm, // 14. / 14、 / Q14:
    /(^|\n)\s*第\s*\d{1,4}\s*(?:题|问|个问题|问题)\s*[:：.．、)）]?/gm, // 第14题
    /(^|\n)\s*\(\s*\d{1,4}\s*\)\s*/gm, // (14)
  ];

  for (const re of patterns) {
    for (const m of text.matchAll(re)) {
      if (m.index === undefined) continue;
      const prefix = m[1] ?? "";
      headers.push(m.index + prefix.length);
    }
  }
  return headers.sort((a, b) => a - b);
}

function looksLikeQuestionHeader(line: string): boolean {
  const s = line.trim();
  if (!s) return false;
  // Question mark or common interrogatives.
  return /[?？]/.test(s) || /(什么|为什么|如何|怎么|哪些|是否|几种|多少|区别)/.test(s);
}

function getAllLikelyQuestionHeaderPositions(text: string): number[] {
  const headers: number[] = [];
  const patterns: RegExp[] = [
    /(^|\n)\s*(?:Q\s*)?\d{1,4}\s*[:：.．、)）]/gm,
    /(^|\n)\s*第\s*\d{1,4}\s*(?:题|问|个问题|问题)\s*[:：.．、)）]?/gm,
    /(^|\n)\s*\(\s*\d{1,4}\s*\)\s*/gm,
  ];

  for (const re of patterns) {
    for (const m of text.matchAll(re)) {
      if (m.index === undefined) continue;
      const prefix = m[1] ?? "";
      const idx = m.index + prefix.length;
      const lineEnd = text.indexOf("\n", idx);
      const line = (lineEnd >= 0 ? text.slice(idx, lineEnd) : text.slice(idx)).slice(0, 120);
      // Only treat as a "question header" if it looks like a question line.
      // This avoids mistaking markdown ordered lists ("4. 主从复制...") as question blocks.
      if (!looksLikeQuestionHeader(line)) continue;
      headers.push(idx);
    }
  }

  return headers.sort((a, b) => a - b);
}

function refineRangeByQuestionMatch(
  chunkContent: string,
  chunkStartAbs: number,
  question: string
): { content: string; startChar: number; endChar: number } | null {
  const text = chunkContent.replace(/\r\n/g, "\n");
  const tokens = extractMatchTokensFromQuestion(question);
  if (tokens.length === 0) return null;

  const matchIdx = findBestKeywordMatchIndex(text, tokens);
  if (matchIdx === null) return null;

  // Try to bound by nearest question header before the match, and next header after it.
  const headers = getAllLikelyQuestionHeaderPositions(text);
  if (headers.length < 2) {
    // Not a numbered Q/A style chunk; tighten around match.
    // If we only have 1 token (often very generic, like "诚信"), prefer sentence-level to avoid huge highlights.
    const { startIdx, endIdx } =
      tokens.length <= 1 ? getSentenceBounds(text, matchIdx, { maxLen: 320 }) : getTightBounds(text, matchIdx);
    const snippet = text.slice(startIdx, endIdx);
    if (!snippet.trim()) return null;
    return { content: snippet, startChar: chunkStartAbs + startIdx, endChar: chunkStartAbs + endIdx };
  }
  let startIdx = 0;
  for (const h of headers) {
    if (h <= matchIdx) startIdx = h;
    else break;
  }

  // Avoid jumping too far up if headers are sparse; in that case just start near the match.
  if (matchIdx - startIdx > 1200) startIdx = Math.max(0, matchIdx - 200);

  let endIdx = Math.min(text.length, startIdx + 900);
  for (const h of headers) {
    if (h > startIdx + 1) {
      endIdx = h;
      break;
    }
  }

  // If header-based end is still huge, clamp.
  if (endIdx - startIdx > 1400) endIdx = Math.min(text.length, startIdx + 1400);

  // Prefer to end at a paragraph break if it occurs reasonably soon.
  const para = text.slice(startIdx + 1, endIdx).search(/\n\s*\n/);
  if (para >= 0 && para < 900) endIdx = startIdx + 1 + para;

  // Tighten within the block to the most relevant paragraph/sentence around the match.
  const tight =
    tokens.length <= 1
      ? getSentenceBounds(text.slice(startIdx, endIdx), matchIdx - startIdx, { maxLen: 320 })
      : getTightBounds(text.slice(startIdx, endIdx), matchIdx - startIdx);
  const tightStart = startIdx + tight.startIdx;
  const tightEnd = startIdx + tight.endIdx;

  const snippet = text.slice(tightStart, tightEnd);
  if (!snippet.trim()) return null;

  return {
    content: snippet,
    startChar: chunkStartAbs + tightStart,
    endChar: chunkStartAbs + tightEnd,
  };
}

function extractFocus(question: string): QuestionFocus {
  const q = question.trim();

  // Chinese patterns like: "<topic>的优点是什么" / "<topic>缺点有哪些"
  const m = q.match(/^(.+?)(?:的)?(优点|缺点|好处|坏处)(?:是|有哪些|有什么|分别是什么|是什么|都有哪些)?.*$/);
  if (m) {
    const topic = m[1]?.trim();
    const word = m[2];
    if (topic && topic.length >= 2) {
      return {
        topic,
        aspect: word === "缺点" || word === "坏处" ? "disadvantages" : "advantages",
        aspectWord: word,
      };
    }
  }

  // English patterns: "pros/cons/advantages/disadvantages of X"
  const e = q.match(/\b(advantages|pros|benefits|disadvantages|cons|drawbacks)\b.*?\b(of)\b\s+(.+?)\??$/i);
  if (e) {
    const word = e[1].toLowerCase();
    const topic = e[3]?.trim();
    return {
      topic,
      aspect: /(disadvantages|cons|drawbacks)/.test(word) ? "disadvantages" : "advantages",
      aspectWord: e[1],
    };
  }

  return {};
}

function clipToFocusedSection(content: string, focus: QuestionFocus): string {
  if (!focus.topic) return content;
  const topic = focus.topic;
  const aspectWord = focus.aspectWord;
  const text = content.replace(/\r\n/g, "\n");

  const idxTopic = text.indexOf(topic);
  if (idxTopic < 0) return content;

  // Prefer to clip around aspect ("缺点"/"优点") within the same chunk if present.
  let idx = idxTopic;
  if (aspectWord) {
    const near = text.indexOf(aspectWord, Math.max(0, idxTopic - 200));
    if (near >= 0) idx = near;
  }

  // Keep this window relatively small so citation highlighting is precise.
  // (The raw chunk is still stored in DB; this is only what we return to UI/LLM.)
  const start = Math.max(0, idx - 120);
  const end = Math.min(text.length, idx + 420);
  return text.slice(start, end);
}

export async function POST(req: NextRequest) {
  try {
    const { question, history } = await req.json();

    if (!question || typeof question !== "string" || !question.trim()) {
      return NextResponse.json({ error: "Question is required" }, { status: 400 });
    }

    const safeHistory: HistoryTurn[] = Array.isArray(history)
      ? history
          .filter(
            (t) =>
              t &&
              typeof t.question === "string" &&
              typeof t.answer === "string" &&
              t.question.trim() &&
              t.answer.trim()
          )
          .slice(-6)
      : [];

    // Rewrite for retrieval so pronouns like "it/its/他的/它的" become standalone.
    const retrievalQuestion = await rewriteStandaloneQuestion(question, safeHistory);
    const focus = extractFocus(retrievalQuestion);
    const defTopic = extractDefinitionTopic(retrievalQuestion);
    const qNum = extractQuestionNumber(question) ?? extractQuestionNumber(retrievalQuestion);

    // Embed the question
    const normalizedQuestion = normalizeForEmbedding(retrievalQuestion);
    let vectorStr: string | null = null;
    try {
      const queryEmbedding = await getEmbedding(normalizedQuestion);
      vectorStr = `[${queryEmbedding.join(",")}]`;
    } catch (e) {
      // If embedding service is down, fall back to keyword-only retrieval.
      console.warn("Embedding failed, falling back to keyword-only retrieval:", e instanceof Error ? e.message : e);
      vectorStr = null;
    }
    const rawQ = retrievalQuestion.trim();
    const qNoPunct = stripPunctuation(rawQ);

    let vectorRows: SearchRow[] = [];
    if (vectorStr) {
      // Search for similar chunks using cosine distance
      const searchResult = await query(
        `SELECT
          c.id,
          c.note_id,
          c.content,
          c.start_char,
          c.end_char,
          n.filename,
          1 - (c.embedding <=> $1::vector) AS similarity
        FROM chunks c
        JOIN notes n ON n.id = c.note_id
        ORDER BY c.embedding <=> $1::vector
        LIMIT $2`,
        [vectorStr, TOP_K]
      );

      vectorRows = searchResult.rows.map((r) => ({
        ...(r as Omit<SearchRow, "similarity">),
        similarity: parseFloat(r.similarity),
      }));
    }

    // Filter vector results with an adaptive threshold.
    const topSim = vectorRows[0]?.similarity ?? 0;
    const threshold = Math.max(BASE_THRESHOLD, topSim - GAP_THRESHOLD);
    let vectorRelevant = vectorRows.filter((row) => row.similarity >= threshold);
    if (vectorStr && vectorRelevant.length < MIN_KEEP) vectorRelevant = vectorRows.slice(0, MIN_KEEP);

    // If we have a clear topic, prefer candidates that mention it.
    if (focus.topic) {
      const topic = focus.topic;
      const focusedVector = vectorRows.filter((row) => row.content.includes(topic));
      if (focusedVector.length > 0) {
        vectorRelevant = focusedVector.slice(0, TOP_K);
      }
    }

    // Keyword-constrained recall boost (hybrid search):
    // If the question contains distinctive tokens (e.g. “审核结果”“准确性”), search those tokens directly
    // and then rank those keyword-matched chunks by vector distance.
    let keywordRows: SearchRow[] = [];
    const exactLikes = [
      `%${rawQ}%`,
      `%${qNoPunct}%`,
      `%Q4:%${qNoPunct}%`,
      `%Q：%${qNoPunct}%`,
    ];

    // Exact phrase match is the most reliable for Q/A style notes.
    const exactResult = vectorStr
      ? await query(
          `SELECT
            c.id,
            c.note_id,
            c.content,
            c.start_char,
            c.end_char,
            n.filename,
            1 - (c.embedding <=> $1::vector) AS similarity
          FROM chunks c
          JOIN notes n ON n.id = c.note_id
          WHERE c.content ILIKE $3 OR c.content ILIKE $4 OR c.content ILIKE $5 OR c.content ILIKE $6
          ORDER BY
            ((CASE WHEN c.content ILIKE $3 THEN 1 ELSE 0 END) +
             (CASE WHEN c.content ILIKE $4 THEN 1 ELSE 0 END) +
             (CASE WHEN c.content ILIKE $5 THEN 1 ELSE 0 END) +
             (CASE WHEN c.content ILIKE $6 THEN 1 ELSE 0 END)) DESC,
            c.embedding <=> $1::vector
          LIMIT $2`,
          [vectorStr, KEYWORD_K, ...exactLikes]
        )
      : await query(
          `SELECT
            c.id,
            c.note_id,
            c.content,
            c.start_char,
            c.end_char,
            n.filename,
            0.0 AS similarity
          FROM chunks c
          JOIN notes n ON n.id = c.note_id
          WHERE c.content ILIKE $1 OR c.content ILIKE $2 OR c.content ILIKE $3 OR c.content ILIKE $4
          ORDER BY
            ((CASE WHEN c.content ILIKE $1 THEN 1 ELSE 0 END) +
             (CASE WHEN c.content ILIKE $2 THEN 1 ELSE 0 END) +
             (CASE WHEN c.content ILIKE $3 THEN 1 ELSE 0 END) +
             (CASE WHEN c.content ILIKE $4 THEN 1 ELSE 0 END)) DESC,
            c.id
          LIMIT $5`,
          [...exactLikes, KEYWORD_K]
        );

    const exactRows: SearchRow[] = exactResult.rows.map((r) => ({
      ...(r as Omit<SearchRow, "similarity">),
      similarity: parseFloat(r.similarity),
    }));

    const patterns = buildKeywordPatterns(retrievalQuestion);
    if (patterns.length > 0) {
      const focusLikes =
        focus.topic && focus.aspectWord ? [`%${focus.topic}%`, `%${focus.aspectWord}%`] : focus.topic ? [`%${focus.topic}%`] : [];
      const all = [`%${rawQ}%`, `%${qNoPunct}%`, ...focusLikes, ...patterns];
      const whereOffset = vectorStr ? 3 : 2; // $1 is vector (when present), $1 is LIMIT otherwise
      const where = all.map((_, i) => `c.content ILIKE $${i + whereOffset}`).join(" OR ");

      // Exact-hit score + keyword-hit count; then fall back to vector distance.
      const exactHit =
        `((CASE WHEN c.content ILIKE $${whereOffset} THEN 1 ELSE 0 END) + (CASE WHEN c.content ILIKE $${
          whereOffset + 1
        } THEN 1 ELSE 0 END))`;
      const kwHits = all
        .map((_, i) => `CASE WHEN c.content ILIKE $${i + whereOffset} THEN 1 ELSE 0 END`)
        .join(" + ");

      const keywordResult = vectorStr
        ? await query(
            `SELECT
              c.id,
              c.note_id,
              c.content,
              c.start_char,
              c.end_char,
              n.filename,
              1 - (c.embedding <=> $1::vector) AS similarity
            FROM chunks c
            JOIN notes n ON n.id = c.note_id
            WHERE ${where}
            ORDER BY ${exactHit} DESC, (${kwHits}) DESC, c.embedding <=> $1::vector
            LIMIT $2`,
            [vectorStr, KEYWORD_K, ...all]
          )
        : await query(
            `SELECT
              c.id,
              c.note_id,
              c.content,
              c.start_char,
              c.end_char,
              n.filename,
              0.0 AS similarity
            FROM chunks c
            JOIN notes n ON n.id = c.note_id
            WHERE ${where}
            ORDER BY ${exactHit} DESC, (${kwHits}) DESC, c.id
            LIMIT $1`,
            [KEYWORD_K, ...all]
          );
      keywordRows = keywordResult.rows.map((r) => ({
        ...(r as Omit<SearchRow, "similarity">),
        similarity: parseFloat(r.similarity),
      }));
    }

    // Merge exact match + keyword match
    const mergedKw = new Map<string, SearchRow>();
    for (const r of exactRows) mergedKw.set(r.id, r);
    for (const r of keywordRows) mergedKw.set(r.id, r);
    keywordRows = Array.from(mergedKw.values());

    // Merge: prefer keyword hits first (higher recall), then fill with vector hits.
    // This avoids missing obvious Q/A passages even when the embedding similarity is low.
    const merged = new Map<string, SearchRow>();
    for (const r of keywordRows) merged.set(r.id, r);
    for (const r of vectorRelevant) merged.set(r.id, r);
    const scoreExact = (content: string) => {
      const c = content;
      let s = 0;
      if (focus.topic && includesLoose(c, focus.topic)) s += 6;
      if (focus.aspectWord && includesLoose(c, focus.aspectWord)) s += 3;
      if (defTopic && includesLoose(c, defTopic)) s += 4;
      if (defTopic && includesLoose(c, `${defTopic}是`)) s += 6;
      if (defTopic && (c.includes("目的") || c.includes("作用") || c.includes("用于"))) s += 2;
      if (qNoPunct && c.includes(qNoPunct)) s += 2;
      if (rawQ && c.includes(rawQ)) s += 2;
      if (qNoPunct && c.includes(`Q4: ${qNoPunct}`)) s += 6;
      if (qNoPunct && c.includes(`Q4:${qNoPunct}`)) s += 6;
      return s;
    };

    const relevantChunks = Array.from(merged.values())
      .sort((a, b) => scoreExact(b.content) - scoreExact(a.content) || b.similarity - a.similarity)
      .slice(0, TOP_K);

    // Build base sources (full chunk content) for the LLM.
    const baseSources: SourceChunk[] = relevantChunks.map((row, index) => {
      const chunkStart =
        typeof row.start_char === "number" ? row.start_char : Number.parseInt(row.start_char, 10);
      const chunkEnd = typeof row.end_char === "number" ? row.end_char : Number.parseInt(row.end_char, 10);

      return {
        index: index + 1,
        noteId: row.note_id,
        filename: row.filename,
        content: row.content,
        startChar: chunkStart,
        endChar: chunkEnd,
        similarity: row.similarity,
      };
    });

    // Pick LLM sources.
    // - If focus.topic exists, keep only those mentioning the topic.
    // - If the question is a definition ("X是什么"), prefer sources that mention X to avoid unrelated chunks.
    const focusedSources = (() => {
      if (focus.topic) {
        const arr = baseSources.filter((s) => includesLoose(s.content, focus.topic!));
        return (arr.length > 0 ? arr : baseSources).slice(0, LLM_TOP_K);
      }

      if (defTopic) {
        const byTopic = baseSources.filter((s) => includesLoose(s.content, defTopic));
        const byDefinition = byTopic.filter((s) => hasDefinitionCue(s.content, defTopic));
        const picked = (byDefinition.length > 0 ? byDefinition : byTopic.length > 0 ? byTopic : baseSources).slice(
          0,
          LLM_TOP_K
        );
        return picked;
      }

      return baseSources.slice(0, LLM_TOP_K);
    })();

    // Build UI sources: same indices as LLM sources, but with a refined highlight range and a shorter snippet.
    const uiSources: SourceChunk[] = focusedSources.map((s) => {
      if (!Number.isFinite(s.startChar)) return s;

      if (defTopic) {
        const refined = refineRangeByDefinitionTopic(s.content, s.startChar, defTopic);
        if (refined) {
          return { ...s, content: refined.content, startChar: refined.startChar, endChar: refined.endChar };
        }
      }

      if (qNum) {
        const refined = refineRangeByNumber(s.content, s.startChar, qNum);
        if (refined) {
          return { ...s, content: refined.content, startChar: refined.startChar, endChar: refined.endChar };
        }
      }

      const refined = refineRangeByQuestionMatch(s.content, s.startChar, retrievalQuestion);
      if (refined) {
        return { ...s, content: refined.content, startChar: refined.startChar, endChar: refined.endChar };
      }

      return s;
    });

    // For definition-style questions, also clip the LLM context to the definition sentence/snippet.
    // This prevents the model from citing adjacent, unrelated concepts in the same long paragraph.
    const llmSources: SourceChunk[] =
      defTopic
        ? focusedSources.map((s) => {
            const refined = refineRangeByDefinitionTopic(s.content, 0, defTopic);
            return refined ? { ...s, content: refined.content } : s;
          })
        : focusedSources;

    // Ask LLM with sources
    const result = await askQuestion(question, llmSources, {
      focusTopic: focus.topic ?? defTopic ?? undefined,
      focusAspect: focus.aspectWord,
    });

    return NextResponse.json({
      answer: result.answer,
      sources: uiSources,
      hasAnswer: result.hasAnswer,
      rewrittenQuestion: retrievalQuestion === question ? undefined : retrievalQuestion,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Internal server error";
    console.error("Ask error:", err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
