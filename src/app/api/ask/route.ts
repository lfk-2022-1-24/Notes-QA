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

function normalizeAnswerCitationsToAvailableSources(answer: string, sources: SourceChunk[]): string {
  if (!answer || sources.length === 0) return answer;
  const available = new Set<number>(sources.map((s) => s.index).filter((n) => Number.isFinite(n)));
  if (available.size === 0) return answer;
  const sorted = Array.from(available).sort((a, b) => a - b);
  const maxIdx = sorted[sorted.length - 1]!;

  // If the model emits a citation number that we didn't return (e.g. [3] while we only have [1]),
  // remap it to the closest available index so the UI can still show a source.
  return answer.replace(/\[(\d+)\]/g, (full, g1) => {
    const n = Number.parseInt(String(g1), 10);
    if (!Number.isFinite(n)) return full;
    if (available.has(n)) return full;
    // clamp to closest: below 1 -> first, above max -> last, otherwise nearest lower.
    if (n <= sorted[0]!) return `[${sorted[0]}]`;
    if (n >= maxIdx) return `[${maxIdx}]`;
    let lower = sorted[0]!;
    for (const v of sorted) {
      if (v <= n) lower = v;
      else break;
    }
    return `[${lower}]`;
  });
}

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

function normalizeForLooseSearch(s: string): string {
  return (s || "")
    .toLowerCase()
    .replace(/\s+/g, "")
    .replace(/[?？!！。，,;；:："'“”‘’（）()【】\[\]{}<>]/g, "");
}

function isNearDuplicateQuestion(chunkContent: string, question: string): boolean {
  const q = normalizeForLooseSearch(question);
  if (!q || q.length < 4) return false;
  const c = normalizeForLooseSearch(chunkContent);
  // If the chunk contains the question (or a very close prefix), treat it as an exact/near-exact match.
  return c.includes(q) || (q.length >= 8 && c.includes(q.slice(0, 8)));
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

function trimLeadingToBoldHeading(
  snippet: string,
  preferredToken?: string
): { snippet: string; delta: number } {
  // Fix "highlight drift upward" when multiple items are flattened into one line.
  // Example: `2. ... **常用的分库分表策略**： ...` should start from the bold heading, not the previous list item.
  if (!snippet) return { snippet, delta: 0 };
  const matches: { idx: number; title: string }[] = [];
  for (const m of snippet.matchAll(/\*\*([^*\n]{2,80})\*\*/g)) {
    if (m.index === undefined) continue;
    matches.push({ idx: m.index, title: (m[1] || "").trim() });
    if (matches.length >= 12) break;
  }
  if (matches.length === 0) return { snippet, delta: 0 };

  const looksLikeDriftPrefix = (prefix: string) =>
    /\b\d{1,4}\s*[.．、)]\s+\S/.test(prefix) || prefix.trim().length >= 12;

  const pickIdx = (() => {
    // Prefer a bold heading that matches the user's main token, but only when it appears after a drift-y prefix.
    if (preferredToken) {
      for (const it of matches) {
        if (it.idx <= 0) continue;
        const prefix = snippet.slice(0, it.idx);
        if (!looksLikeDriftPrefix(prefix)) continue;
        if (includesLoose(it.title, preferredToken)) return it.idx;
      }
    }
    // Fallback: first bold heading after a drift-y prefix.
    for (const it of matches) {
      if (it.idx <= 0) continue;
      const prefix = snippet.slice(0, it.idx);
      if (looksLikeDriftPrefix(prefix)) return it.idx;
    }
    return null;
  })();

  if (pickIdx === null) return { snippet, delta: 0 };
  const out = snippet.slice(pickIdx);
  if (!out.trim()) return { snippet, delta: 0 };
  return { snippet: out, delta: pickIdx };
}

function buildLfTextAndMap(orig: string): { text: string; map: number[] } {
  // Normalize CRLF -> LF for matching/regex, while keeping a mapping to original indices.
  // map[normIndex] = origIndex
  const map: number[] = [];
  let out = "";
  for (let i = 0; i < orig.length; i++) {
    const ch = orig[i]!;
    if (ch === "\r" && orig[i + 1] === "\n") continue;
    out += ch;
    map[out.length - 1] = i;
  }
  return { text: out, map };
}

function mapNormRangeToOrig(map: number[], startNorm: number, endNorm: number, origLen: number) {
  const s = Math.max(0, Math.min(startNorm, map.length));
  const e = Math.max(s, Math.min(endNorm, map.length));
  if (map.length === 0) return { start: 0, end: 0 };

  const start = s >= map.length ? origLen : map[s] ?? 0;
  const end =
    e <= 0
      ? 0
      : e >= map.length
        ? origLen
        : ((map[e - 1] ?? map[map.length - 1] ?? 0) + 1);

  return { start: Math.max(0, start), end: Math.max(Math.max(0, start), end) };
}

function anyTokenMatchLoose(content: string, tokens: string[]): boolean {
  if (!tokens || tokens.length === 0) return false;
  for (const t of tokens) {
    if (!t) continue;
    if (includesLoose(content, t)) return true;
  }
  return false;
}

function countTokenMatchesLoose(content: string, tokens: string[]): number {
  if (!tokens || tokens.length === 0) return 0;
  let hits = 0;
  for (const t of tokens) {
    if (!t) continue;
    if (includesLoose(content, t)) hits++;
  }
  return hits;
}

function refineRangeByNumberedQaQuestionText(
  chunkContent: string,
  chunkStartAbs: number,
  question: string,
  maxLen = 520
): { content: string; startChar: number; endChar: number } | null {
  const { text, map } = buildLfTextAndMap(chunkContent);
  const qNorm = normalizeForLooseSearch(question);
  if (!qNorm || qNorm.length < 4) return null;

  // Typical extraction format (pdf/txt/docx flattening):
  // - "33. 问：...？ 答：..."
  // - "9. ...的区别？ 解答：..."
  // NOTE: Many parsers flatten line breaks into spaces, so we cannot rely on '\n'.
  // Use a non-digit boundary instead to find headers in the middle of a long line.
  const headerRe = /(^|[^\d])(\d{1,4})\s*[.．、)]\s*(?:问\s*[:：])?/gm;
  const headers: { idx: number; num: number }[] = [];
  for (const m of text.matchAll(headerRe)) {
    const prefix = m[1] ?? "";
    const idx = (m.index ?? 0) + prefix.length;
    const num = Number.parseInt(m[2] ?? "", 10);
    if (!Number.isFinite(num)) continue;
    headers.push({ idx, num });
  }
  if (headers.length === 0) return null;

  const tokens0 = extractMatchTokensFromQuestion(question);
  const localFocus = extractFocus(question);
  const focusTokens = [localFocus.topic, localFocus.aspectWord].filter((x): x is string => Boolean(x));
  const tokens = filterTokensByRarityInText(text, Array.from(new Set([...focusTokens, ...tokens0])));

  let best: { startIdx: number; endIdx: number; score: number } | null = null;

  for (let i = 0; i < headers.length; i++) {
    const startIdx = headers[i]!.idx;
    const endIdx = i + 1 < headers.length ? headers[i + 1]!.idx : Math.min(text.length, startIdx + 1200);
    if (endIdx <= startIdx + 10) continue;

    const block = text.slice(startIdx, endIdx);
    const qMark = block.search(/问\s*[:：]/);
    const aMark = block.search(/(?:解答|答案|答)\s*[:：]/);

    // Extract a short "question line" for scoring.
    const qStart = qMark >= 0 ? qMark + 2 : 0;
    let qEnd = -1;
    if (aMark > qStart) qEnd = aMark;
    const qm = block.search(/[?？]/);
    if (qm >= 0 && (qEnd < 0 || qm < qEnd) && qm > qStart) qEnd = qm + 1;
    if (qEnd < 0) qEnd = Math.min(block.length, qStart + 180);

    const qTextRaw = block.slice(qStart, qEnd);
    const qText = qTextRaw.replace(/\s+/g, " ").trim();
    if (qText.length < 6) continue;

    const usedTokens = tokens.length > 0 ? tokens : Array.from(new Set([...focusTokens, ...tokens0]));
    const hit = countTokenMatchesLoose(qText, usedTokens);

    // Guard: avoid matching ordered list items that are not Q/A.
    // Accept if:
    // - explicit 问/答 markers exist, OR
    // - question mark + answer marker exist (or the answer is in parentheses right after '?'), OR
    // - the header line is an "aspect/diff" style item (优缺点/区别/是什么...) and matches 2+ query tokens.
    const hasQaMarkers = qMark >= 0 || aMark >= 0;
    const hasQuestionMark = /[?？]/.test(qText);
    const qmInBlock = block.search(/[?？]/);
    const hasParenAnswerNearQ =
      qmInBlock >= 0 && /[（(]/.test(block.slice(qmInBlock + 1, Math.min(block.length, qmInBlock + 12)));
    const looksLikeAspectItem = /(优点|缺点|优缺点|区别|对比|不同|是什么|定义|含义|原理|机制|作用|意义)/.test(qText);
    if (
      !hasQaMarkers &&
      !(hasQuestionMark && (aMark >= 0 || hasParenAnswerNearQ) && hit >= 1) &&
      !(looksLikeAspectItem && hit >= 2)
    )
      continue;

    const qTextNorm = normalizeForLooseSearch(qText);
    let score = 0;

    // Big bonus for near-duplicate question lines.
    if (qTextNorm && (qNorm.includes(qTextNorm) || qTextNorm.includes(qNorm))) score += 12;

    // Token overlap within the question line (not the whole answer) is a strong signal.
    score += hit * 3;

    // Prefer blocks whose question line ends with a question mark.
    if (/[?？]\s*$/.test(qText)) score += 1;

    if (!best || score > best.score) {
      best = { startIdx, endIdx, score };
    }
  }

  if (!best || best.score < 4) return null;

  let startIdx = best.startIdx;
  let endIdx = best.endIdx;

  // Clip to a readable window but never cross into the next question.
  if (endIdx - startIdx > maxLen) endIdx = startIdx + maxLen;

  // Trim whitespace
  while (startIdx < endIdx && (text[startIdx] === "\n" || text[startIdx] === " ")) startIdx++;
  while (endIdx > startIdx && /\s/.test(text[endIdx - 1]!)) endIdx--;

  const mapped = mapNormRangeToOrig(map, startIdx, endIdx, chunkContent.length);
  const snippet = chunkContent.slice(mapped.start, mapped.end);
  if (!snippet.trim()) return null;

  return { content: snippet, startChar: chunkStartAbs + mapped.start, endChar: chunkStartAbs + mapped.end };
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
  const { text, map } = buildLfTextAndMap(chunkContent);
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
  // Fallback: many notes mention the term but don't have "X是什么" titles.
  if (idx < 0) {
    idx = text.indexOf(t);
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

  // IMPORTANT: don't include headings/preamble that often appear before "X是..." in Markdown.
  // Always start at the actual match position (or later) to avoid highlighting unrelated content above.
  const bounds = getSentenceBounds(text, matchIdx, { maxLen: 220 });
  // For "definition topic", never include content before the topic occurrence itself.
  const startIdx = Math.max(matchIdx, bounds.startIdx);
  let endIdx = bounds.endIdx;

  // Many notes are "definition line + numbered list" flattened into one paragraph, e.g.
  // "反射？（...） 12. ... 13. ...". In this case, cut strictly before the next numbered header
  // to avoid drifting into adjacent Q/A items.
  {
    const lookAhead = text.slice(startIdx, Math.min(text.length, startIdx + 900));
    const nextHeader = lookAhead.match(/(^|[^\d])\s*\d{1,4}\s*[.．、)]\s+/m);
    if (nextHeader && nextHeader.index !== undefined) {
      const rel = nextHeader.index + (nextHeader[1]?.length ?? 0);
      // Only treat it as a boundary if it is not immediately at the start (avoid cutting "1. ..." within term text)
      if (rel >= 18) {
        endIdx = Math.min(endIdx, startIdx + rel);
      }
    }
  }

  // Hard cap for definition snippets.
  if (endIdx - startIdx > 240) endIdx = startIdx + 240;
  const mapped = mapNormRangeToOrig(map, startIdx, endIdx, chunkContent.length);
  const snippet = chunkContent.slice(mapped.start, mapped.end);
  if (!snippet.trim()) return null;

  return { content: snippet, startChar: chunkStartAbs + mapped.start, endChar: chunkStartAbs + mapped.end };
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
  const { text, map } = buildLfTextAndMap(chunkContent);

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

  const mapped = mapNormRangeToOrig(map, localStart, localEnd, chunkContent.length);
  const snippet = chunkContent.slice(mapped.start, mapped.end);
  if (!snippet.trim()) return null;

  return {
    content: snippet,
    startChar: chunkStartAbs + mapped.start,
    endChar: chunkStartAbs + mapped.end,
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
    "讲解",
    "解释",
    "说明",
    "介绍",
    "概述",
    "分析",
    "总结",
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
    // File/meta words
    "txt",
    "md",
    "pdf",
    "docx",
    "doc",
    "文件",
    "文档",
    "笔记",
    "内容",
    "这里",
    "里面",
    "提到",
    "说",
    "讲",
    "说明",
    "介绍",
    "问题",
    "答案",
    "引用",
    "溯源",
  ]);

  const stripSuffixParticles = (s: string) => {
    let out = s.trim();
    // Common Chinese particles/prepositions that often attach to nouns in questions.
    out = out.replace(/(上的|里的|中的|内的|外的)$/g, "");
    out = out.replace(/(上|下|中|内|外)$/g, "");
    out = out.replace(/的$/g, "");
    return out;
  };

  const stripInterrogativePrefixes = (s: string) => {
    let out = s.trim();
    // Common leading question phrases in Chinese.
    out = out.replace(/^(怎么才能|如何才能|怎么|如何|怎样|为什么|什么|哪些|是否|能否|可以|应该|要不要|如何去|怎么去)/g, "");
    out = out.replace(/^才能/g, "");
    out = out.replace(/^是/g, "");
    out = out.trim();
    return out;
  };

  const tokens = new Set<string>();
  for (const m of q.matchAll(/[\p{Script=Han}A-Za-z0-9]{2,}/gu)) {
    const t = m[0].trim();
    if (!t || stop.has(t)) continue;
    const base = stripInterrogativePrefixes(stripSuffixParticles(t));
    if (base && !stop.has(base)) tokens.add(base);

    // Heuristics for Chinese: add tail/head substrings to avoid missing matches when
    // the query contains leading verbs like “讲解/解释/介绍...”.
    // Example: “讲解垃圾回收” -> add “垃圾回收”.
    const verbPrefixes = ["讲解", "解释", "说明", "介绍", "概述", "分析", "总结", "聊聊", "说说", "讲讲"];
    for (const vp of verbPrefixes) {
      if (base.startsWith(vp) && base.length > vp.length + 1) {
        const tail = stripInterrogativePrefixes(stripSuffixParticles(base.slice(vp.length)));
        if (tail && !stop.has(tail)) tokens.add(tail);
      }
    }

    // Split by common particle "的" to extract topic words.
    for (const part of base.split("的").filter(Boolean)) {
      const p = stripInterrogativePrefixes(stripSuffixParticles(part));
      if (p.length >= 2 && !stop.has(p)) tokens.add(p);
    }

    if (base.length >= 4) {
      const tail4 = stripInterrogativePrefixes(stripSuffixParticles(base.slice(-4)));
      if (tail4.length >= 2 && !stop.has(tail4)) tokens.add(tail4);
    }
    if (base.length >= 6) {
      const tail3 = stripInterrogativePrefixes(stripSuffixParticles(base.slice(-3)));
      if (tail3.length >= 2 && !stop.has(tail3)) tokens.add(tail3);
    }
  }
  // Add a couple of canonical tokens for "辨别真伪/正确错误" style questions.
  const qLower = q.toLowerCase();
  if (qLower.includes("分辨") || qLower.includes("辨别") || qLower.includes("判断")) {
    tokens.add("分辨");
    tokens.add("辨别");
    tokens.add("判断");
  }
  if (qLower.includes("正确") || qLower.includes("错误") || qLower.includes("是非") || qLower.includes("真伪")) {
    tokens.add("正确");
    tokens.add("错误");
    tokens.add("是非");
    tokens.add("真伪");
  }
  if (qLower.includes("网络") || qLower.includes("网上") || qLower.includes("互联网")) {
    tokens.add("网络");
    tokens.add("网上");
    tokens.add("互联网");
  }
  if (qLower.includes("观点") || qLower.includes("说法")) {
    tokens.add("观点");
    tokens.add("说法");
  }
  // Prefer longer tokens first (e.g. "国歌" > "什么")
  return Array.from(tokens).sort((a, b) => b.length - a.length).slice(0, 12);
}

function filterTokensByRarityInText(text: string, tokens: string[]): string[] {
  // Drop tokens that occur too frequently in the chunk; they are weak anchors for highlighting.
  const kept: Array<{ t: string; freq: number }> = [];
  for (const t of tokens) {
    let from = 0;
    let freq = 0;
    while (from < text.length && freq <= 12) {
      const idx = text.indexOf(t, from);
      if (idx < 0) break;
      freq++;
      from = idx + Math.max(1, Math.floor(t.length / 2));
    }
    // Keep if reasonably rare, or if the token is long enough to be specific.
    if (freq <= 6 || t.length >= 4) kept.push({ t, freq });
  }
  // Prefer longer and rarer tokens.
  return kept
    .sort((a, b) => b.t.length - a.t.length || a.freq - b.freq)
    .map((x) => x.t)
    .slice(0, 10);
}

function isOverviewQuestion(question: string): boolean {
  const q = question.trim();
  if (!q) return false;
  return /(讲解|解释|说明|介绍|概述|分析|总结|聊聊|说说|讲讲)\b/.test(q);
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

function refineRangeBySentenceMatch(
  chunkContent: string,
  chunkStartAbs: number,
  question: string,
  maxLen = 220
): { content: string; startChar: number; endChar: number } | null {
  const { text, map } = buildLfTextAndMap(chunkContent);
  const tokens0 = extractMatchTokensFromQuestion(question);
  const tokens = filterTokensByRarityInText(text, tokens0);
  if (tokens.length === 0) return null;
  const matchIdx = findBestKeywordMatchIndex(text, tokens);
  if (matchIdx === null) return null;

  const { startIdx, endIdx } = getSentenceBounds(text, matchIdx, { maxLen });
  const mapped = mapNormRangeToOrig(map, startIdx, endIdx, chunkContent.length);
  const snippet = chunkContent.slice(mapped.start, mapped.end);
  if (!snippet.trim()) return null;

  return {
    content: snippet,
    startChar: chunkStartAbs + mapped.start,
    endChar: chunkStartAbs + mapped.end,
  };
}

function refineRangeByMarkdownLineMatch(
  chunkContent: string,
  chunkStartAbs: number,
  question: string,
  maxLen = 180
): { content: string; startChar: number; endChar: number } | null {
  const { text, map } = buildLfTextAndMap(chunkContent);
  const tokens0 = extractMatchTokensFromQuestion(question);
  const tokens = filterTokensByRarityInText(text, tokens0);
  if (tokens.length === 0) return null;
  const matchIdx = findBestKeywordMatchIndex(text, tokens);
  if (matchIdx === null) return null;

  const isMdHeading = (s: string) => /^\s*#{1,6}\s+\S+/.test(s.trim());
  const looksLikeQuestionLine = (s: string) => /[?？]/.test(s) || /(怎么|如何|为什么|是什么|有哪些|是否|区别)/.test(s);
  const isMdNoiseLine = (s: string) => {
    const t = s.trim();
    return t.startsWith("![") || /^\s*!\[.*\]\(.*\)\s*$/.test(t) || /^[-*_]{3,}\s*$/.test(t);
  };

  const overview = isOverviewQuestion(question);
  const mainToken = tokens.find((t) => t.length >= 2) || tokens[0];

  // Overview mode: if we can find a heading containing the main token near the match,
  // return a section slice (not just a single line).
  if (overview && mainToken) {
    type Heading = { pos: number; level: number; title: string };
    const headings: Heading[] = [];
    const re = /(^|\n)(#{1,6})\s+([^\n]+)/g;
    for (const m of text.matchAll(re)) {
      if (m.index === undefined) continue;
      const prefix = m[1] ?? "";
      const pos = m.index + prefix.length;
      const level = (m[2] || "#").length;
      const title = (m[3] || "").trim();
      headings.push({ pos, level, title });
    }

    if (headings.length > 0) {
      // Choose the last heading before match that mentions the main token.
      let chosen: Heading | null = null;
      for (const h of headings) {
        if (h.pos > matchIdx) break;
        if (includesLoose(h.title, mainToken)) chosen = h;
      }

      // Fallback: choose the last heading before match.
      if (!chosen) {
        for (const h of headings) {
          if (h.pos > matchIdx) break;
          chosen = h;
        }
      }

      if (chosen) {
        // Start after heading line, then skip empty/noise lines.
        const headingLineEnd = text.indexOf("\n", chosen.pos);
        let startIdx = headingLineEnd >= 0 ? headingLineEnd + 1 : chosen.pos;
        for (let hops = 0; hops < 6 && startIdx < text.length; hops++) {
          const nextEnd = text.indexOf("\n", startIdx);
          const lineEnd = nextEnd >= 0 ? nextEnd : text.length;
          const line = text.slice(startIdx, lineEnd).trim();
          if (!line || isMdNoiseLine(line)) {
            startIdx = lineEnd + 1;
            continue;
          }
          break;
        }

        // End at next heading of same or higher level, or cap by maxLen.
        let endIdx = Math.min(text.length, startIdx + Math.max(400, maxLen));
        for (const h of headings) {
          if (h.pos <= chosen.pos) continue;
          if (h.level <= chosen.level) {
            endIdx = Math.min(endIdx, h.pos);
            break;
          }
        }

        if (endIdx - startIdx > maxLen) endIdx = startIdx + maxLen;
        const mapped = mapNormRangeToOrig(map, startIdx, endIdx, chunkContent.length);
        const snippet = chunkContent.slice(mapped.start, mapped.end);
        if (snippet.trim()) {
          return { content: snippet, startChar: chunkStartAbs + mapped.start, endChar: chunkStartAbs + mapped.end };
        }
      }
    }
  }

  // Line boundaries for match
  const prevNl = text.lastIndexOf("\n", matchIdx);
  const nextNl = text.indexOf("\n", matchIdx);
  let startIdx = prevNl >= 0 ? prevNl + 1 : 0;
  let endIdx = nextNl >= 0 ? nextNl : text.length;

  // Handle cases like "...。## Title" where the heading got glued to the previous sentence.
  // If we see an inline heading marker, cut the line to start from that heading.
  {
    const rawLine = text.slice(startIdx, endIdx);
    const inlineHeadingPos = rawLine.search(/#{1,6}\s+\S+/);
    if (inlineHeadingPos > 0) {
      startIdx = startIdx + inlineHeadingPos;
    }
  }

  let line = text.slice(startIdx, endIdx).trim();

  // If we matched a heading/question line, highlight the next non-empty non-heading line as the "answer" line.
  if (isMdHeading(line) || looksLikeQuestionLine(line)) {
    let cursor = endIdx;
    for (let hops = 0; hops < 6 && cursor < text.length; hops++) {
      // move to start of next line
      cursor = cursor + 1;
      if (cursor >= text.length) break;
      const nextEnd = text.indexOf("\n", cursor);
      const nextLineEnd = nextEnd >= 0 ? nextEnd : text.length;
      const nextLineRaw = text.slice(cursor, nextLineEnd);
      const nextLine = nextLineRaw.trim();
      if (!nextLine) {
        cursor = nextLineEnd;
        continue;
      }
      if (isMdHeading(nextLine)) {
        cursor = nextLineEnd;
        continue;
      }
      if (isMdNoiseLine(nextLine)) {
        cursor = nextLineEnd;
        continue;
      }
      startIdx = cursor;
      endIdx = nextLineEnd;
      line = nextLine;
      break;
    }
  }

  // Cap highlight size hard.
  if (endIdx - startIdx > maxLen) endIdx = startIdx + maxLen;

  // Trim
  while (startIdx < endIdx && (text[startIdx] === " " || text[startIdx] === "\n")) startIdx++;
  while (endIdx > startIdx && /\s/.test(text[endIdx - 1]!)) endIdx--;

  const mapped = mapNormRangeToOrig(map, startIdx, endIdx, chunkContent.length);
  let snippet = chunkContent.slice(mapped.start, mapped.end);
  // If the line was flattened and includes prior numbered items, prefer starting from the first bold heading.
  {
    const trimmed = trimLeadingToBoldHeading(snippet, mainToken);
    if (trimmed.delta > 0) {
      snippet = trimmed.snippet;
      mapped.start = mapped.start + trimmed.delta;
    }
  }
  // If the snippet accidentally includes a prefix before an inline heading marker, drop that prefix.
  const inlineHeading = snippet.search(/#{1,6}\s+\S+/);
  if (inlineHeading > 0) {
    snippet = snippet.slice(inlineHeading);
    mapped.start = mapped.start + inlineHeading;
  }
  if (!snippet.trim()) return null;

  return { content: snippet, startChar: chunkStartAbs + mapped.start, endChar: chunkStartAbs + mapped.end };
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
    // Support both line-start and "inline" (PDF/text flattening) headers by allowing a non-digit boundary.
    /(^|[^\d])\s*(?:Q\s*)?\d{1,4}\s*[:：.．、)）]/gm,
    /(^|[^\d])\s*第\s*\d{1,4}\s*(?:题|问|个问题|问题)\s*[:：.．、)）]?/gm,
    /(^|[^\d])\s*\(\s*\d{1,4}\s*\)\s*/gm,
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
  const { text, map } = buildLfTextAndMap(chunkContent);
  const tokens0 = extractMatchTokensFromQuestion(question);
  const tokens = filterTokensByRarityInText(text, tokens0);
  if (tokens.length === 0) return null;

  const matchIdx = findBestKeywordMatchIndex(text, tokens);
  if (matchIdx === null) return null;

  // Try to bound by nearest question header before the match, and next header after it.
  const headers = getAllLikelyQuestionHeaderPositions(text);
  if (headers.length < 1) {
    // Not a numbered Q/A style chunk; tighten around match.
    // If we only have 1 token (often very generic, like "诚信"), prefer sentence-level to avoid huge highlights.
    const { startIdx, endIdx } =
      tokens.length <= 1 ? getSentenceBounds(text, matchIdx, { maxLen: 320 }) : getTightBounds(text, matchIdx);
    const mapped = mapNormRangeToOrig(map, startIdx, endIdx, chunkContent.length);
    const snippet = chunkContent.slice(mapped.start, mapped.end);
    if (!snippet.trim()) return null;
    return { content: snippet, startChar: chunkStartAbs + mapped.start, endChar: chunkStartAbs + mapped.end };
  }
  // Even if there is only ONE header in this chunk, use it as a hard start boundary.
  let startIdx = headers[0] ?? 0;
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

  const mapped = mapNormRangeToOrig(map, tightStart, tightEnd, chunkContent.length);
  let snippet = chunkContent.slice(mapped.start, mapped.end);
  {
    const mainToken = tokens.find((t) => t.length >= 2) || tokens[0];
    const trimmed = trimLeadingToBoldHeading(snippet, mainToken);
    if (trimmed.delta > 0) {
      snippet = trimmed.snippet;
      mapped.start = mapped.start + trimmed.delta;
    }
  }
  if (!snippet.trim()) return null;

  return {
    content: snippet,
    startChar: chunkStartAbs + mapped.start,
    endChar: chunkStartAbs + mapped.end,
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

    // Post-filter: keep only chunks that contain at least one strong query token.
    // This reduces "answer is correct but citations drift to unrelated blocks" (common for txt notes).
    const qTokens = extractMatchTokensFromQuestion(retrievalQuestion);
    const focusTokens = [focus.topic, focus.aspectWord].filter((x): x is string => Boolean(x)).slice(0, 2);

    const strongTokens = (defTopic ? [defTopic, ...qTokens] : qTokens)
      .filter(Boolean)
      .filter((t) => t.length >= 3)
      .slice(0, 8);
    const fallbackTokens = (defTopic ? [defTopic, ...qTokens] : qTokens).filter(Boolean).slice(0, 10);

    // IMPORTANT: for "topic + aspect" questions, the most important tokens can be 2-char Chinese words
    // (e.g. "反射"/"优点"). Always include them to avoid filtering out the correct chunk.
    const filterTokens = Array.from(
      new Set<string>([...focusTokens, ...(strongTokens.length > 0 ? strongTokens : fallbackTokens)])
    ).slice(0, 12);

    // For longer questions, require 2+ token hits to avoid "vaguely related" chunks.
    const minHits =
      (filterTokens.length >= 4 || retrievalQuestion.trim().length >= 12) && !defTopic && !focus.topic ? 2 : 1;

    const filteredRelevant = relevantChunks.filter((r) => countTokenMatchesLoose(r.content, filterTokens) >= minHits);
    const relaxedRelevant = relevantChunks.filter((r) => anyTokenMatchLoose(r.content, filterTokens));

    const finalRelevantChunks =
      filteredRelevant.length >= 1
        ? filteredRelevant
        : relaxedRelevant.length >= Math.min(LLM_TOP_K, 2)
          ? relaxedRelevant
          : relevantChunks;

    // Build base sources (full chunk content) for the LLM.
    const baseSources: SourceChunk[] = finalRelevantChunks.map((row, index) => {
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
      // If the note contains the user's question as an almost-exact string (common in Q/A txt),
      // aggressively prefer that chunk to avoid citing adjacent template/metadata blocks.
      const nearExact = baseSources.filter((s) => isNearDuplicateQuestion(s.content, retrievalQuestion));
      if (nearExact.length > 0) {
        return nearExact.slice(0, Math.min(LLM_TOP_K, 2));
      }

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

      // Default: prefer sources that match more query tokens.
      const rankTokens = filterTokens.length > 0 ? filterTokens : qTokens;
      return baseSources
        .slice()
        .sort(
          (a, b) =>
            countTokenMatchesLoose(b.content, rankTokens) - countTokenMatchesLoose(a.content, rankTokens) ||
            b.similarity - a.similarity
        )
        .slice(0, LLM_TOP_K);
    })();

    // Build UI sources: same indices as LLM sources, but with a refined highlight range and a shorter snippet.
    const uiSources: SourceChunk[] = focusedSources.map((s) => {
      if (!Number.isFinite(s.startChar)) return s;

      const lowerName = (s.filename || "").toLowerCase();
      const isMarkdown = lowerName.endsWith(".md") || lowerName.endsWith(".markdown");

      if (defTopic) {
        const refined = refineRangeByDefinitionTopic(s.content, s.startChar, defTopic);
        if (refined) {
          return { ...s, content: refined.content, startChar: refined.startChar, endChar: refined.endChar };
        }
      }

      // For numbered Q/A style notes (pdf/txt/docx) where the question text exists verbatim,
      // prefer matching the question line to avoid drifting into adjacent questions.
      const qaRefined = refineRangeByNumberedQaQuestionText(s.content, s.startChar, retrievalQuestion, 520);
      if (qaRefined) {
        return { ...s, content: qaRefined.content, startChar: qaRefined.startChar, endChar: qaRefined.endChar };
      }

      if (qNum) {
        const refined = refineRangeByNumber(s.content, s.startChar, qNum);
        if (refined) {
          return { ...s, content: refined.content, startChar: refined.startChar, endChar: refined.endChar };
        }
      }

      // For markdown notes, always prefer sentence-level highlighting to avoid
      // highlighting adjacent unrelated lines/paragraphs.
      if (isMarkdown) {
        const lineRefined = refineRangeByMarkdownLineMatch(s.content, s.startChar, retrievalQuestion, 180);
        if (lineRefined) {
          return {
            ...s,
            content: lineRefined.content,
            startChar: lineRefined.startChar,
            endChar: lineRefined.endChar,
          };
        }
        const refined = refineRangeBySentenceMatch(s.content, s.startChar, retrievalQuestion, 220);
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

    const isClarifyMarker = (s: string) => s.includes("【需要补充上下文】");
    const lastTurn = safeHistory.at(-1);
    const askedClarifyLastTurn = Boolean(lastTurn?.answer && isClarifyMarker(lastTurn.answer));

    // Two-stage behavior when retrieval/grounding fails:
    // - 1st time: ask user to clarify instead of claiming "no info"
    // - 2nd time (after user follow-up): be explicit that notes don't contain relevant material
    if (!result.hasAnswer) {
      const stage1 = `【需要补充上下文】\n我在当前笔记里暂时没检索到能直接回答你这个问题的内容。你可以补充一下：\n（1）你说的关键术语/对象具体指什么？（可以给全称、同义词、英文缩写）\n或者\n（2）如果你手头有相关段落/关键词，请直接贴出来或上传对应资料。`;

      const stage2 = `我在当前已上传的笔记中仍然没有检索到与该问题直接相关、可用于作答的资料。\n如果你希望我继续回答，请上传/补充相关笔记（或把关键段落贴出来），我再基于新增资料进行检索与问答。`;

      return NextResponse.json({
        answer: askedClarifyLastTurn ? stage2 : stage1,
        sources: [],
        // For stage1 we treat it as a follow-up prompt (avoid "no sources" warning UI);
        // for stage2 we mark as no-answer so UI can highlight the limitation.
        hasAnswer: askedClarifyLastTurn ? false : true,
        rewrittenQuestion: retrievalQuestion === question ? undefined : retrievalQuestion,
      });
    }

    const normalizedAnswer = normalizeAnswerCitationsToAvailableSources(result.answer, uiSources);

    return NextResponse.json({
      answer: normalizedAnswer,
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
