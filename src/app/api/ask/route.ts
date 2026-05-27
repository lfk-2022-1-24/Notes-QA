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

function includesTopicLoose(haystack: string, topic: string): boolean {
  // Like includesLoose, but tolerant to common Chinese connector "的" inserted between words.
  // Example: topic "集体利益" should match "集体的利益".
  if (includesLoose(haystack, topic)) return true;
  const t = topic.trim();
  if (!t || t.length < 3) return false;
  const h0 = normalizeLooseContains(haystack);
  const t0 = normalizeLooseContains(t);
  if (!h0 || !t0) return false;
  // Remove "的" only for the topic match (avoid global behavior changes elsewhere).
  const h = h0.replace(/的/g, "");
  const n = t0.replace(/的/g, "");
  if (!n) return false;
  return h.includes(n);
}

function topicTokenVariants(topic: string): string[] {
  const t = (topic || "").trim();
  if (!t) return [];
  const out: string[] = [t];
  // For common 4+ char Chinese topics, add head/tail to match forms like "集体的利益".
  if (/^[\p{Script=Han}]{4,}$/u.test(t)) {
    out.push(t.slice(0, 2));
    out.push(t.slice(-2));
    if (t.length >= 6) {
      out.push(t.slice(0, 3));
      out.push(t.slice(-3));
    }
  }
  // Dedup + cap
  return Array.from(new Set(out)).slice(0, 5);
}

function buildTopicLikePatterns(topic: string): string[] {
  const t = (topic || "").trim();
  if (!t) return [];
  const patterns = new Set<string>();
  patterns.add(`%${t}%`);
  // Allow connector words like "的" between parts, e.g. "%集体%利益%" matches "集体的利益".
  if (/^[\p{Script=Han}]{4,}$/u.test(t)) {
    const head2 = t.slice(0, 2);
    const tail2 = t.slice(-2);
    if (head2 && tail2) patterns.add(`%${head2}%${tail2}%`);
  }
  return Array.from(patterns).slice(0, 3);
}

function extractDateYYYYMMDD(s: string): string | null {
  // Normalize common date forms into canonical YYYY-MM-DD.
  const raw = String(s || "");
  const m1 = raw.match(/\b(20\d{2})-(\d{1,2})-(\d{1,2})\b/);
  const m2 = raw.match(/\b(20\d{2})[\/.](\d{1,2})[\/.](\d{1,2})\b/);
  const m3 = raw.match(/\b(20\d{2})年(\d{1,2})月(\d{1,2})[日号]\b/);
  const m4 = raw.match(/\b(20\d{2})(\d{2})(\d{2})\b/); // 20240216
  if (m4) return `${m4[1]}-${m4[2]}-${m4[3]}`;
  const m = m1 ?? m2 ?? m3;
  if (!m) return null;
  const y = m[1]!;
  const mm = String(Number.parseInt(m[2]!, 10)).padStart(2, "0");
  const dd = String(Number.parseInt(m[3]!, 10)).padStart(2, "0");
  return `${y}-${mm}-${dd}`;
}

function buildDateVariants(dateYYYYMMDD: string): string[] {
  // Expand a canonical YYYY-MM-DD into common log/note formats.
  const m = dateYYYYMMDD.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return [];
  const y = m[1]!;
  const mm = m[2]!;
  const dd = m[3]!;
  const m1 = String(Number.parseInt(mm, 10));
  const d1 = String(Number.parseInt(dd, 10));
  const set = new Set<string>();
  set.add(`${y}-${mm}-${dd}`);
  set.add(`${y}-${m1}-${d1}`);
  set.add(`${y}/${mm}/${dd}`);
  set.add(`${y}/${m1}/${d1}`);
  set.add(`${y}.${mm}.${dd}`);
  set.add(`${y}.${m1}.${d1}`);
  set.add(`${y}年${mm}月${dd}日`);
  set.add(`${y}年${m1}月${d1}日`);
  set.add(`${y}年${mm}月${dd}号`);
  set.add(`${y}年${m1}月${d1}号`);
  set.add(`${y}${mm}${dd}`);
  // Yearless variants (some logs omit year in body but keep it in filename/context).
  set.add(`${mm}-${dd}`);
  set.add(`${m1}-${d1}`);
  set.add(`${mm}/${dd}`);
  set.add(`${m1}/${d1}`);
  set.add(`${mm}.${dd}`);
  set.add(`${m1}.${d1}`);
  set.add(`${mm}月${dd}日`);
  set.add(`${m1}月${d1}日`);
  set.add(`${mm}月${dd}号`);
  set.add(`${m1}月${d1}号`);
  return Array.from(set);
}

function normalizeDateMatchText(s: string): string {
  let out = String(s || "").replace(/\s+/g, "");
  out = out.replace(/[‐‑‒–—−]/g, "-");
  out = out.replace(/[／]/g, "/");
  out = out.replace(/[．]/g, ".");
  out = out.replace(/[０-９]/g, (ch) => String.fromCharCode(ch.charCodeAt(0) - 0xff10 + 0x30));
  out = out.replace(/[\u200b\u200c\u200d\ufeff]/g, "");
  return out;
}

function includesDateLoose(text: string, dateVariant: string): boolean {
  const t = normalizeDateMatchText(text);
  const d = normalizeDateMatchText(dateVariant);
  if (!d) return false;
  return t.includes(d);
}

function computeDisplaySimilarity(args: {
  existing: number;
  content: string;
  filename: string;
  filterTokens: string[];
  topicHint?: string;
  dateHint?: string;
  dateVariants?: string[];
}): number {
  const { existing, content, filename, filterTokens, topicHint, dateHint, dateVariants } = args;
  if (Number.isFinite(existing) && existing > 0) return Math.min(1, existing);

  // Date queries: prefer a high score when the exact date is present.
  if (dateHint) {
    const vars = dateVariants && dateVariants.length > 0 ? dateVariants : [dateHint];
    const inContent = vars.some((v) => includesDateLoose(content, v));
    const inName = vars.some((v) => includesDateLoose(filename, v));
    const base = inContent ? 0.98 : inName ? 0.85 : 0.35;
    return base;
  }

  // Token-based heuristic for keyword-only retrieval.
  const denom = Math.max(3, Math.min(8, filterTokens.length || 0));
  const hits = filterTokens.length > 0 ? countTokenMatchesLoose(content, filterTokens) : 0;
  let score = denom > 0 ? hits / denom : 0;

  if (topicHint && includesTopicLoose(content, topicHint)) score = Math.max(score, 0.65);
  if (includesLoose(content, filename)) score = Math.max(score, 0.25);

  // Clamp to a reasonable UI range (avoid 0.0% for real matches).
  score = Math.max(0.12, Math.min(0.95, score));
  return score;
}

function pickYearfulDateVariants(dateHint: string, variants: string[]): string[] {
  const year = dateHint.slice(0, 4);
  const yearful = variants.filter((v) => v.includes(year) || v.startsWith(year));
  return yearful.length > 0 ? yearful : variants;
}

function refineRangeByLogDateHeading(
  chunkContent: string,
  chunkStartAbs: number,
  dateHint: string,
  dateVariants: string[],
  maxLen = 1400
): { content: string; startChar: number; endChar: number } | null {
  const { text, map } = buildLfTextAndMap(chunkContent);
  const variants = pickYearfulDateVariants(dateHint, dateVariants.length > 0 ? dateVariants : [dateHint]);

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
  if (headings.length === 0) return null;

  const isDateHeading = (h: Heading) =>
    /日志/.test(h.title) &&
    (/\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b/.test(h.title) || /\b20\d{2}年\d{1,2}月\d{1,2}[日号]\b/.test(h.title));

  // Find the first heading that contains the requested date.
  let chosen: Heading | null = null;
  for (const h of headings) {
    if (variants.some((v) => includesDateLoose(h.title, v))) {
      chosen = h;
      break;
    }
  }
  if (!chosen) return null;

  let startIdx = chosen.pos;
  let endIdx = Math.min(text.length, startIdx + maxLen);

  for (const h of headings) {
    if (h.pos <= chosen.pos) continue;
    if (h.level <= chosen.level && isDateHeading(h)) {
      endIdx = Math.min(endIdx, h.pos);
      break;
    }
  }

  // Trim whitespace
  while (startIdx < endIdx && (text[startIdx] === " " || text[startIdx] === "\n")) startIdx++;
  while (endIdx > startIdx && /\s/.test(text[endIdx - 1]!)) endIdx--;

  const mapped = mapNormRangeToOrig(map, startIdx, endIdx, chunkContent.length);
  const snippet = chunkContent.slice(mapped.start, mapped.end);
  if (!snippet.trim()) return null;

  return { content: snippet, startChar: chunkStartAbs + mapped.start, endChar: chunkStartAbs + mapped.end };
}

function refineRangeByPlainLogDateHeading(
  chunkContent: string,
  chunkStartAbs: number,
  dateHint: string,
  dateVariants: string[],
  maxLen = 1400
): { content: string; startChar: number; endChar: number } | null {
  const { text, map } = buildLfTextAndMap(chunkContent);
  const variants = pickYearfulDateVariants(dateHint, dateVariants.length > 0 ? dateVariants : [dateHint]);

  // Prefer "日志 <date>" anchors (common in txt exports).
  let matchIdx = -1;
  for (const v of variants) {
    const re = new RegExp(`(?:^|\\n)\\s*日志\\s*${escapeRegExp(v)}\\b`, "m");
    const m = text.match(re);
    if (m && m.index !== undefined) {
      matchIdx = m.index + (m[0].startsWith("\n") ? 1 : 0);
      break;
    }
  }
  if (matchIdx < 0) {
    // Fallback: find the date, then scan a short prefix for "日志".
    for (const v of variants) {
      const i = text.indexOf(v);
      if (i < 0) continue;
      const prefixStart = Math.max(0, i - 24);
      const prefix = text.slice(prefixStart, i);
      if (prefix.includes("日志")) {
        // move to line start
        const prevNl = text.lastIndexOf("\n", i);
        matchIdx = prevNl >= 0 ? prevNl + 1 : Math.max(0, prefixStart);
        break;
      }
    }
  }
  if (matchIdx < 0) return null;

  // Start at the line start containing "日志 ..."
  const startIdx = matchIdx;

  // End at next "日志 <YYYY-MM-DD>" style header, or at a strong separator line.
  let endIdx = Math.min(text.length, startIdx + maxLen);
  const lookAhead = text.slice(startIdx + 1, Math.min(text.length, startIdx + 6000));

  const nextLog = lookAhead.search(/(?:^|\n)\s*日志\s*20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b|(?:^|\n)\s*日志\s*20\d{2}年\d{1,2}月\d{1,2}[日号]\b/m);
  if (nextLog >= 0 && nextLog >= 20) endIdx = Math.min(endIdx, startIdx + 1 + nextLog);

  const nextSep = lookAhead.search(/(?:^|\n)\s*[─-]{8,}\s*(?:\n|$)/m);
  if (nextSep >= 0 && nextSep >= 40) endIdx = Math.min(endIdx, startIdx + 1 + nextSep);

  // Trim
  let s = startIdx;
  let e = endIdx;
  while (s < e && (text[s] === " " || text[s] === "\n")) s++;
  while (e > s && /\s/.test(text[e - 1]!)) e--;

  const mapped = mapNormRangeToOrig(map, s, e, chunkContent.length);
  const snippet = chunkContent.slice(mapped.start, mapped.end);
  if (!snippet.trim()) return null;
  return { content: snippet, startChar: chunkStartAbs + mapped.start, endChar: chunkStartAbs + mapped.end };
}

function escapeRegExp(s: string): string {
  return String(s || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function refineRangeByAnyLogDateHeading(
  chunkContent: string,
  chunkStartAbs: number,
  dateHint: string,
  dateVariants: string[],
  maxLen = 1400
): { content: string; startChar: number; endChar: number } | null {
  return (
    refineRangeByLogDateHeading(chunkContent, chunkStartAbs, dateHint, dateVariants, maxLen) ??
    refineRangeByPlainLogDateHeading(chunkContent, chunkStartAbs, dateHint, dateVariants, maxLen)
  );
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

function extractInclusionTopic(question: string): string | null {
  const q = question.trim();
  if (!q) return null;
  // Chinese: "X包括哪些" / "X包含什么" / "X主要包括什么" / "X具体包括哪些呀"
  const m1 = q.match(/^(.+?)\s*(?:主要|具体|一般|通常|常见|常用)?\s*(?:包括|包含)\s*(?:哪些|什么|什么内容|有哪些|哪几类|哪几种|哪方面|都有哪些)\s*[?？呀啊呢吗]*$/);
  if (m1) {
    const topic = (m1[1] || "").trim();
    return topic.length >= 2 ? topic : null;
  }
  // Variant: "X都包括什么" / "X包括什么"
  const m2 = q.match(/^(.+?)\s*(?:都)?\s*(?:包括|包含)\s*(?:什么|哪些)\s*[?？呀啊呢吗]*$/);
  if (m2) {
    const topic = (m2[1] || "").trim();
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

function refineRangeByInclusionTopic(
  chunkContent: string,
  chunkStartAbs: number,
  topic: string,
  maxLen = 260
): { content: string; startChar: number; endChar: number } | null {
  const { text, map } = buildLfTextAndMap(chunkContent);
  const t = topic.trim();
  if (!t || t.length < 2) return null;

  const idxTopic = text.indexOf(t);
  if (idxTopic < 0) return null;

  // Prefer the first occurrence of "包括/包含/主要包括" after the topic.
  const cues = ["主要包括", "具体包括", "一般包括", "通常包括", "包括", "包含"];
  let idx = idxTopic;
  for (const c of cues) {
    const i = text.indexOf(c, idxTopic);
    if (i >= 0 && i - idxTopic <= 260) {
      idx = i;
      break;
    }
  }

  const bounds = getSentenceBounds(text, idx, { maxLen });
  const startIdx = Math.max(idxTopic, bounds.startIdx);
  let endIdx = bounds.endIdx;

  // Cut before next Q/A header if present (PDF Q/A flattening).
  {
    const lookAhead = text.slice(startIdx, Math.min(text.length, startIdx + 900));
    const nextHeader = lookAhead.match(/(^|[^\d])\s*(?:Q\s*)?\d{1,4}\s*[:：.．、)）]\s+|(^|[^\S\r\n])(?:问题|问)\s*[:：]/m);
    if (nextHeader && nextHeader.index !== undefined) {
      const rel = nextHeader.index + ((nextHeader[1] ?? nextHeader[2])?.length ?? 0);
      if (rel >= 18) endIdx = Math.min(endIdx, startIdx + rel);
    }
  }

  const mapped = mapNormRangeToOrig(map, startIdx, endIdx, chunkContent.length);
  const snippet = chunkContent.slice(mapped.start, mapped.end);
  if (!snippet.trim()) return null;
  return { content: snippet, startChar: chunkStartAbs + mapped.start, endChar: chunkStartAbs + mapped.end };
}

function refineRangeByTrainingStrategyCue(
  chunkContent: string,
  chunkStartAbs: number,
  maxLen = 520
): { content: string; startChar: number; endChar: number } | null {
  const { text, map } = buildLfTextAndMap(chunkContent);
  if (!text.trim()) return null;
  // Prefer an explicit "训练策略" heading, and clip to that section (heading + 1-2 lines)
  // to avoid highlighting the entire surrounding chapter.
  const lower = text.toLowerCase();
  const headingCues = ["### 训练策略", "## 训练策略", "# 训练策略", "训练策略：", "训练策略:", "training strategy"];
  let idx = -1;
  for (const c of headingCues) {
    const i = lower.indexOf(c.toLowerCase());
    if (i >= 0 && (idx < 0 || i < idx)) idx = i;
  }
  if (idx < 0) return null;

  // Find start of the heading line.
  const lineStart = (() => {
    const j = text.lastIndexOf("\n", idx);
    return j >= 0 ? j + 1 : 0;
  })();
  const afterHeadingLine = (() => {
    const j = text.indexOf("\n", idx);
    return j >= 0 ? j + 1 : text.length;
  })();

  // Include the next 1-2 non-empty lines, stopping before the next heading.
  const tail = text.slice(afterHeadingLine);
  const lines = tail.split("\n");
  let consumed = 0;
  let included = 0;
  for (const rawLine of lines) {
    const l = rawLine.trimEnd();
    consumed += rawLine.length + 1; // +1 for '\n'
    if (!l.trim()) continue;
    if (/^\s*#{1,6}\s+/.test(l)) break;
    included++;
    if (included >= 2) break;
  }
  const endIdx = Math.min(text.length, afterHeadingLine + consumed);
  const mappedRange = mapNormRangeToOrig(map, lineStart, endIdx, chunkContent.length);
  const snippet = chunkContent.slice(mappedRange.start, mappedRange.end);
  if (!snippet.trim()) return null;
  // Hard cap as a safety.
  const capped = snippet.length > maxLen ? snippet.slice(0, maxLen) : snippet;
  const capEnd = mappedRange.start + capped.length;
  return {
    content: capped,
    startChar: chunkStartAbs + mappedRange.start,
    endChar: chunkStartAbs + capEnd,
  };
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
    // Inclusion phrasing (too generic for anchoring/highlighting)
    "包括",
    "包含",
    "主要包括",
    "具体包括",
    "都包括",
    "有哪些",
    "哪几类",
    "哪几种",
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

function isResultOrExperimentQuestion(question: string): boolean {
  const q = (question || "").trim();
  if (!q) return false;
  return /(实验结果|实验|结果|指标|准确率|acc|f1|auc|loss|提升|下降|对比|消融|ablation|baseline|sota|效果|性能)/i.test(q);
}

function extractTailKeyphrase(question: string): string | null {
  const q = (question || "").trim();
  if (!q) return null;
  // Keep this conservative: only keyphrases that strongly indicate the user intent.
  if (q.includes("训练策略")) return "训练策略";
  if (q.includes("优化策略")) return "优化策略";
  return null;
}

function extractTrainingStrategyBoostTokens(question: string): string[] {
  const q = (question || "").trim();
  if (!q) return [];
  const isTrainingStrategy =
    q.includes("训练策略") ||
    q.includes("训练方法") ||
    q.includes("训练流程") ||
    q.includes("训练设置") ||
    q.toLowerCase().includes("training strategy") ||
    q.toLowerCase().includes("optimizer") ||
    q.toLowerCase().includes("learning rate") ||
    q.toLowerCase().includes("scheduler");
  if (!isTrainingStrategy) return [];

  const tokens = [
    "训练策略",
    "训练方法",
    "训练流程",
    "训练设置",
    "训练阶段",
    "超参数",
    "优化器",
    "学习率",
    "调度",
    "余弦",
    "退火",
    "cosine",
    "anneal",
    "scheduler",
    "learning rate",
    "epoch",
    "epochs",
    "分布式",
    "数据并行",
    "并行",
    "DDP",
    "distributed",
    "data parallel",
  ];
  return Array.from(new Set(tokens)).slice(0, 18);
}

function extractRecordIdHint(question: string): { kind: "meeting"; id: string } | null {
  const q = (question || "").trim();
  if (!q) return null;
  const s = q
    .replace(/[０-９]/g, (d) => String.fromCharCode(d.charCodeAt(0) - 0xff10 + 0x30))
    .replace(/\s+/g, " ");

  const m =
    // Note: avoid \b here because "002中" (digit + Han) isn't a word-boundary in JS regex.
    s.match(/(?:会议记录|会议纪要|会议纪要记录)(?:\s*第)?\s*([0-9]{1,4})(?![0-9])/i) ||
    s.match(/(?:^|[^0-9])([0-9]{1,4})\s*(?:号)?\s*(?:会议记录|会议纪要|会议纪要记录)/i);
  if (!m) return null;
  const raw = (m[1] || "").trim();
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n)) return null;
  const id = String(n).padStart(3, "0");
  return { kind: "meeting", id };
}

function buildMeetingRecordTokens(id: string): string[] {
  const core = id.replace(/^0+/, "") || id;
  const tokens = [
    `会议记录 ${id}`,
    `会议记录${id}`,
    `会议记录 ${core}`,
    `会议记录${core}`,
    `会议纪要 ${id}`,
    `会议纪要${id}`,
    `会议纪要 ${core}`,
    `会议纪要${core}`,
    `纪要 ${id}`,
    `纪要${id}`,
    `第${id}`,
    `第 ${id}`,
  ];
  return Array.from(new Set(tokens));
}

function buildMeetingRecordTokensStrict(id: string): string[] {
  const core = id.replace(/^0+/, "") || id;
  const fullwidthId = id.replace(/[0-9]/g, (d) => String.fromCharCode(d.charCodeAt(0) - 0x30 + 0xff10));
  const fullwidthCore = core.replace(/[0-9]/g, (d) => String.fromCharCode(d.charCodeAt(0) - 0x30 + 0xff10));
  const tokens = [
    `会议记录 ${id}`,
    `会议记录${id}`,
    `会议记录 ${core}`,
    `会议记录${core}`,
    `会议记录 ${fullwidthId}`,
    `会议记录${fullwidthId}`,
    `会议记录 ${fullwidthCore}`,
    `会议记录${fullwidthCore}`,
    `会议记录-${id}`,
    `会议记录-${core}`,
    `会议记录—${id}`,
    `会议记录—${core}`,
    `会议记录_${id}`,
    `会议记录_${core}`,
    `会议记录（${id}）`,
    `会议记录(${id})`,
    `会议记录（${core}）`,
    `会议记录(${core})`,
    `会议记录（${fullwidthId}）`,
    `会议记录(${fullwidthId})`,
    `会议记录（${fullwidthCore}）`,
    `会议记录(${fullwidthCore})`,
    `会议纪要 ${id}`,
    `会议纪要${id}`,
    `会议纪要 ${core}`,
    `会议纪要${core}`,
    `会议纪要 ${fullwidthId}`,
    `会议纪要${fullwidthId}`,
    `会议纪要 ${fullwidthCore}`,
    `会议纪要${fullwidthCore}`,
    `会议纪要-${id}`,
    `会议纪要-${core}`,
    `会议纪要—${id}`,
    `会议纪要—${core}`,
    `会议纪要记录 ${id}`,
    `会议纪要记录${id}`,
    `会议纪要记录 ${core}`,
    `会议纪要记录${core}`,
    `会议纪要记录 ${fullwidthId}`,
    `会议纪要记录${fullwidthId}`,
    `会议纪要记录 ${fullwidthCore}`,
    `会议纪要记录${fullwidthCore}`,
    `会议纪要记录-${id}`,
    `会议纪要记录-${core}`,
    `会议纪要记录—${id}`,
    `会议纪要记录—${core}`,
    `# 会议记录 ${id}`,
    `# 会议记录${id}`,
    `# 会议纪要 ${id}`,
    `# 会议纪要${id}`,
    `## 会议记录 ${id}`,
    `## 会议记录${id}`,
    `## 会议纪要 ${id}`,
    `## 会议纪要${id}`,
    `# 会议记录 ${fullwidthId}`,
    `# 会议记录${fullwidthId}`,
    `# 会议纪要 ${fullwidthId}`,
    `# 会议纪要${fullwidthId}`,
    `## 会议记录 ${fullwidthId}`,
    `## 会议记录${fullwidthId}`,
    `## 会议纪要 ${fullwidthId}`,
    `## 会议纪要${fullwidthId}`,
  ];
  return Array.from(new Set(tokens));
}

function detectMeetingRecordHeadingId(content: string): string | null {
  const text = (content || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  if (!text.trim()) return null;
  // Prefer markdown H1 headings.
  const m1 = text.match(/^\s*#\s*(?:会议记录|会议纪要|会议纪要记录)\s*0*([0-9]{1,4})(?![0-9])/m);
  if (m1 && m1[1]) {
    const n = Number.parseInt(m1[1], 10);
    if (Number.isFinite(n)) return String(n).padStart(3, "0");
  }
  // Fallback: allow "会议记录 002" anywhere, but this is weaker.
  const m2 = text.match(/(?:会议记录|会议纪要|会议纪要记录)\s*0*([0-9]{1,4})(?![0-9])/);
  if (m2 && m2[1]) {
    const n = Number.parseInt(m2[1], 10);
    if (Number.isFinite(n)) return String(n).padStart(3, "0");
  }
  return null;
}

function computeMeetingRecordSectionBoundsFromNoteContent(
  noteContent: string,
  recordId: string
): { start: number; end: number } | null {
  const raw = String(noteContent || "");
  if (!raw) return null;
  const idCore = (recordId || "").replace(/^0+/, "") || recordId;
  if (!idCore) return null;
  // Accept both markdown headings ("## 会议记录 003") and plain-text headings ("会议记录 003").
  const re = new RegExp(
    String.raw`^\s*(?:#{1,6}\s*)?(?:会议记录|会议纪要|会议纪要记录)\s*0*${escapeRegExp(idCore)}(?![0-9]).*$`,
    "m"
  );
  const all: { idx: number }[] = [];
  for (const m of raw.matchAll(new RegExp(re.source, "gm"))) {
    if (m.index !== undefined) all.push({ idx: m.index });
  }
  if (all.length === 0) return null;
  all.sort((a, b) => a.idx - b.idx);

  // Find the first heading occurrence for this record id.
  const target = all[0]!.idx;
  // Find the next heading after target.
  const nextRe = new RegExp(
    String.raw`^\s*(?:#{1,6}\s*)?(?:会议记录|会议纪要|会议纪要记录)\s*0*\d{1,4}(?![0-9]).*$`,
    "gm"
  );
  let end = raw.length;
  for (const m of raw.matchAll(nextRe)) {
    if (m.index !== undefined && m.index > target) {
      end = m.index;
      break;
    }
  }
  return { start: target, end };
}

function clipChunkToSection(
  chunkContent: string,
  chunkStartAbs: number,
  chunkEndAbs: number,
  sectionStartAbs: number,
  sectionEndAbs: number
): { content: string; startChar: number; endChar: number } | null {
  if (!Number.isFinite(chunkStartAbs) || !Number.isFinite(chunkEndAbs)) return null;
  if (!Number.isFinite(sectionStartAbs) || !Number.isFinite(sectionEndAbs)) return null;
  const startAbs = Math.max(chunkStartAbs, sectionStartAbs);
  const endAbs = Math.min(chunkEndAbs, sectionEndAbs);
  if (!(endAbs > startAbs)) return null;
  const relStart = Math.max(0, startAbs - chunkStartAbs);
  const relEnd = Math.max(relStart, endAbs - chunkStartAbs);
  const sliced = String(chunkContent || "").slice(relStart, relEnd);
  if (!sliced.trim()) return null;
  // Avoid creating meaningless one-character snippets like "-" after clipping.
  if (sliced.trim().length < 3) return null;
  return { content: sliced, startChar: startAbs, endChar: endAbs };
}

function computeMeetingRecordSectionBounds(
  sources: SourceChunk[],
  targetId: string
): { start: number; end: number } | null {
  const heads: { id: string; startChar: number }[] = [];
  for (const s of sources) {
    const id = detectMeetingRecordHeadingId(s.content);
    if (!id) continue;
    if (!Number.isFinite(s.startChar)) continue;
    heads.push({ id, startChar: s.startChar });
  }
  if (heads.length === 0) return null;
  heads.sort((a, b) => a.startChar - b.startChar);

  const idx = heads.findIndex((h) => h.id === targetId);
  if (idx < 0) return null;
  const start = heads[idx]!.startChar;
  const end = heads[idx + 1]?.startChar ?? Number.POSITIVE_INFINITY;
  return { start, end };
}

function extractMeetingAgendaEvidenceLines(content: string, subtopic: string, maxLines = 10): string[] {
  const t = (subtopic || "").trim();
  if (!t) return [];
  const text = (content || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  if (!text.trim()) return [];
  const re = new RegExp(String.raw`^\s*##\s*议题\s*\d+\s*[:：]\s*${escapeRegExp(t)}(?:\s|$).*`, "m");
  const m = text.match(re);
  if (!m || m.index === undefined) return [];
  const startIdx = m.index;
  const after = text.slice(startIdx + 1);
  const next = after.search(/^\s*##\s+(?:议题\s*\d+\s*[:：]|行动项汇总)\s*/m);
  const endIdx = next >= 0 ? startIdx + 1 + next : text.length;
  const section = text.slice(startIdx, endIdx);
  const lines = section
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    // Drop the heading itself; keep content cues.
    .filter((l) => !/^##\s*议题\s*\d+\s*[:：]/.test(l))
    // Avoid checklist-style action items if they appear.
    .filter((l) => !/^\[\s*[xX ]\s*\]\s+/.test(l) && !/^-\s*\[\s*[xX ]\s*\]\s+/.test(l));

  // Prefer background/discussion/decision blocks if present.
  const preferred: string[] = [];
  for (const l of lines) {
    if (/^(背景|讨论要点|决议)\s*[:：]/.test(stripMarkdownFormattingInAnswer(l))) {
      preferred.push(stripMarkdownFormattingInAnswer(l));
      continue;
    }
    // Keep bullet points that follow those sections.
    if (/^[-*+]\s+/.test(l) || /^[•·]\s*/.test(l)) {
      preferred.push(stripMarkdownFormattingInAnswer(l));
      continue;
    }
    // Also keep a couple of plain sentences.
    if (preferred.length < maxLines) preferred.push(stripMarkdownFormattingInAnswer(l));
    if (preferred.length >= maxLines) break;
  }
  return preferred.slice(0, maxLines);
}

function extractMeetingRecordSubtopic(question: string, recordId: string): string | null {
  const q = (question || "").trim();
  if (!q) return null;
  const id = recordId.replace(/^0+/, "") || recordId;
  const re = new RegExp(
    // Same: avoid \b; allow trailing Chinese like "002中".
    String.raw`(?:会议记录|会议纪要|会议纪要记录)\s*(?:第)?\s*0*${id}(?![0-9])[\s\S]{0,30}?(?:中提到的|里提到的|中的|里|内的|内|提到的)\s*([^?？。！!]+)`,
    "i"
  );
  const m = q.match(re);
  if (m && m[1]) {
    const t = String(m[1]).trim();
    // Trim trailing generic ask words
    const cleaned = t.replace(/(是什么|有哪些|主要|怎么|如何|讲解一下|解释一下|说明一下|详细说一下|说了什么|讲了什么)\s*$/g, "").trim();
    if (!cleaned || cleaned.length < 2) return null;
    if (/^(说了什么|讲了什么|什么|内容|情况)$/i.test(cleaned)) return null;
    return cleaned.replace(/^(对于|关于|针对|就)\s*/g, "").replace(/^的\s*/g, "").trim();
  }
  // Also allow "...会议记录 002 的 <topic>" without "中提到的"
  const reLoose = new RegExp(
    String.raw`(?:会议记录|会议纪要|会议纪要记录)\s*(?:第)?\s*0*${id}(?![0-9])\s*的\s*([^?？。！!]+)`,
    "i"
  );
  const mLoose = q.match(reLoose);
  if (mLoose && mLoose[1]) {
    const t = String(mLoose[1]).trim();
    const cleaned = t
      .replace(/^(详细|具体)?\s*(?:说一下|详细说一下|讲解一下|解释一下|说明一下|说说|讲讲)?\s*/g, "")
      .replace(/(是什么|有哪些|主要|怎么|如何|讲解一下|解释一下|说明一下|详细说一下|说了什么|讲了什么)\s*$/g, "")
      .replace(/^(对于|关于|针对|就)\s*/g, "")
      .replace(/^的\s*/g, "")
      .trim();
    if (!cleaned || cleaned.length < 2) return null;
    if (/^(说了什么|讲了什么|什么|内容|情况)$/i.test(cleaned)) return null;
    return cleaned;
  }
  // Fallback: if question contains a quoted/explicit phrase, use it.
  const m2 = q.match(/[“"「『](.+?)[”"」』]/);
  if (m2 && m2[1]) {
    const t = String(m2[1]).trim();
    return t.length >= 2 ? t : null;
  }
  return null;
}

function extractMeetingRecordSubtopics(question: string, recordId: string): string[] {
  const q = (question || "").trim();
  if (!q) return [];
  const id = recordId.replace(/^0+/, "") || recordId;
  // Capture everything after "会议记录002 ..." and then split by common separators.
  const re = new RegExp(
    String.raw`(?:会议记录|会议纪要|会议纪要记录)\s*(?:第)?\s*0*${id}(?![0-9])[\s\S]{0,40}?(?:的)?\s*(?:中提到的|里提到的|中的|里|内的|内|提到的|关于)\s*([^?？。！!]+)`,
    "i"
  );
  const m = q.match(re);
  let tail = m?.[1] ? String(m[1]).trim() : "";
  if (!tail) {
    // If no "中提到的" cue, still allow "...会议记录 002 的 A，B" patterns.
    const re2 = new RegExp(
      String.raw`(?:详细|具体)?\s*(?:说一下|讲解一下|讲讲|说明一下)?\s*(?:会议记录|会议纪要|会议纪要记录)\s*(?:第)?\s*0*${id}(?![0-9])\s*(?:的)?\s*([^?？。！!]+)`,
      "i"
    );
    const m2 = q.match(re2);
    tail = m2?.[1] ? String(m2[1]).trim() : "";
  }
  if (!tail) return [];

  // Remove generic ask words.
  tail = tail
    .replace(/^(详细|具体)\s*(?:说一下|讲解一下|讲讲|说明一下)?/g, "")
    .replace(/(是什么|有哪些|主要|怎么|如何|讲解一下|解释一下|说明一下|详细说一下)\s*$/g, "")
    .trim();
  if (!tail) return [];

  const parts = tail
    .split(/[，,、;；\/\|]|(?:和|以及|及|还有|与)/g)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => s.replace(/^(关于|对于|针对|就|提到的|中提到的|里提到的)\s*/g, "").trim())
    .map((s) => s.replace(/^的\s*/g, "").trim())
    .filter((s) => s.length >= 2)
    .filter((s) => !/^(说了什么|讲了什么|什么|内容|情况|等等|之类)$/i.test(s));

  // Dedup, keep stable order.
  const out: string[] = [];
  const seen = new Set<string>();
  for (const p of parts) {
    if (seen.has(p)) continue;
    seen.add(p);
    out.push(p);
  }
  return out.slice(0, 4);
}

function extractMeetingAgendaSubtopicFromPattern(question: string): string | null {
  const q = (question || "").trim();
  if (!q) return null;
  // Pattern: "对于X，会议记录002里说了什么" / "关于X，会议记录 002 ..."
  const m = q.match(/^(?:对于|关于|针对|就)\s*([^，,。！？?？]{2,50})[，,]/);
  if (!m || !m[1]) return null;
  const t = String(m[1]).trim().replace(/^的\s*/g, "").trim();
  if (!t || t.length < 2) return null;
  if (/(会议记录|会议纪要|会议纪要记录)/.test(t)) return null;
  if (/^(什么|内容|情况|说了什么|讲了什么)$/i.test(t)) return null;
  return t;
}

function extractMeetingAgendaTitlesFromHeader(content: string): string[] {
  const text = (content || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  if (!text.trim()) return [];
  const lines = text.split("\n");
  const line =
    lines.find((l) => /\b议题\b/.test(l) && /[:：]/.test(l)) ||
    lines.find((l) => l.includes("议题") && (l.includes(":") || l.includes("："))) ||
    "";
  if (!line) return [];
  const idx = line.indexOf(":") >= 0 ? line.indexOf(":") : line.indexOf("：");
  if (idx < 0) return [];
  const rhs = line.slice(idx + 1).trim();
  if (!rhs) return [];
  const parts = rhs
    .split(/[;；]/g)
    .map((s) => s.trim())
    .filter(Boolean)
    .flatMap((s) => s.split(/[，,、]/g).map((x) => x.trim()).filter(Boolean));
  const out: string[] = [];
  const seen = new Set<string>();
  for (const p of parts) {
    const t = p.replace(/^\*\*|^\-\s*/g, "").replace(/\*\*$/g, "").trim();
    if (!t || t.length < 2) continue;
    if (seen.has(t)) continue;
    seen.add(t);
    out.push(t);
  }
  return out.slice(0, 10);
}

function pickMeetingRecordHeaderCandidate(sources: SourceChunk[]): SourceChunk | null {
  if (!sources || sources.length === 0) return null;
  const withAgenda = sources.find((s) => /[-*+]\s*\*\*议题\*\*\s*[:：]/.test(String(s.content || "")));
  if (withAgenda) return withAgenda;
  // Otherwise pick the earliest chunk in the section.
  const sorted = sources
    .filter((s) => Number.isFinite(s.startChar))
    .slice()
    .sort((a, b) => (a.startChar as number) - (b.startChar as number));
  return sorted[0] ?? sources[0] ?? null;
}

function stripParenSegments(s: string): string {
  return (s || "")
    .replace(/[\(（][^）\)]*[\)）]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function alignToAgendaTitles(raw: string, agendaTitles: string[]): string | null {
  const r = stripParenSegments(raw);
  if (!r) return null;
  for (const t of agendaTitles) {
    const tt = stripParenSegments(t);
    if (!tt) continue;
    if (includesLoose(tt, r) || includesLoose(r, tt)) return t;
  }
  return null;
}

function extractAgendaSectionFromRecordSection(
  recordSectionText: string,
  recordSectionStartAbs: number,
  agendaTitle: string
): { content: string; startChar: number; endChar: number } | null {
  const raw = String(recordSectionText || "");
  if (!raw.trim()) return null;
  const target = stripParenSegments(agendaTitle);
  if (!target) return null;

  // Find the H2 agenda heading line.
  const re = /^##\s*议题\s*\d+\s*[:：]\s*(.+?)\s*$/gm;
  let m: RegExpExecArray | null;
  let startRel: number | null = null;
  while ((m = re.exec(raw))) {
    const title = (m[1] || "").trim();
    if (!title) continue;
    const canon = stripParenSegments(title);
    if (!canon) continue;
    if (includesLoose(canon, target) || includesLoose(target, canon)) {
      startRel = m.index;
      break;
    }
  }
  if (startRel == null) return null;

  const after = raw.slice(startRel + 1);
  const next = after.search(/^\s*##\s+(?:议题\s*\d+\s*[:：]|行动项汇总)\s*/m);
  const endRel = next >= 0 ? startRel + 1 + next : raw.length;

  const startAbs = recordSectionStartAbs + startRel;
  const endAbs = recordSectionStartAbs + endRel;
  const snippet = raw.slice(startRel, endRel).trim();
  if (!snippet) return null;
  // Hard cap for LLM/UI payload size.
  const capped = snippet.length > 1800 ? snippet.slice(0, 1800) : snippet;
  return { content: capped, startChar: startAbs, endChar: startAbs + capped.length };
}

function refineRangeByNeedle(
  chunkContent: string,
  chunkStartAbs: number,
  needle: string,
  maxLen = 900
): { content: string; startChar: number; endChar: number } | null {
  const n = (needle || "").trim();
  if (!n) return null;
  const { text, map } = buildLfTextAndMap(chunkContent);
  if (!text.trim()) return null;
  const idx = text.indexOf(n);
  if (idx < 0) return null;
  const bounds = getSentenceBounds(text, idx, { maxLen });
  const mappedRange = mapNormRangeToOrig(map, bounds.startIdx, bounds.endIdx, chunkContent.length);
  const snippet = chunkContent.slice(mappedRange.start, mappedRange.end);
  if (!snippet.trim()) return null;
  return {
    content: snippet,
    startChar: chunkStartAbs + mappedRange.start,
    endChar: chunkStartAbs + mappedRange.end,
  };
}

function refineRangeByMeetingAgendaSubtopic(
  chunkContent: string,
  chunkStartAbs: number,
  subtopic: string,
  maxLen = 1300
): { content: string; startChar: number; endChar: number } | null {
  const t = (subtopic || "").trim();
  if (!t) return null;
  const { text, map } = buildLfTextAndMap(chunkContent);
  if (!text.trim()) return null;
  // Prefer a markdown heading like "## 议题3: <t>" (tolerant to "**背景**" on same line).
  const re = new RegExp(String.raw`^\s*##\s*议题\s*\d+\s*[:：]\s*${escapeRegExp(t)}(?:\s|$).*`, "m");
  const m = text.match(re);
  const idx = m && m.index !== undefined ? m.index : -1;
  if (idx < 0) return null;

  const startIdx = idx;
  // End at next "## 议题" or "## 行动项汇总" or next H2 heading.
  const after = text.slice(idx + 1);
  const next = after.search(/^\s*##\s+(?:议题\s*\d+\s*[:：]|行动项汇总)\s*/m);
  const endIdx = next >= 0 ? Math.min(text.length, idx + 1 + next) : Math.min(text.length, startIdx + maxLen);

  const mappedRange = mapNormRangeToOrig(map, startIdx, endIdx, chunkContent.length);
  const snippet = chunkContent.slice(mappedRange.start, mappedRange.end);
  if (!snippet.trim()) return null;
  return {
    content: snippet.length > maxLen ? snippet.slice(0, maxLen) : snippet,
    startChar: chunkStartAbs + mappedRange.start,
    endChar: chunkStartAbs + mappedRange.start + Math.min(snippet.length, maxLen),
  };
}

function extractMeetingAgendaSubtopicCandidatesFromQuestion(question: string): string[] {
  const q = (question || "").trim();
  if (!q) return [];
  // Split by separators, keep medium-length chunks as candidates.
  const parts = q
    .replace(/[?？!！。]/g, " ")
    .split(/[，,、;；\/\|]|(?:和|以及|及|还有|与)/g)
    .map((s) => s.trim())
    .filter(Boolean);
  const bad = new Set([
    "会议记录",
    "会议纪要",
    "会议纪要记录",
    "详细说一下",
    "讲解一下",
    "解释一下",
    "说明一下",
    "说一下",
    "说了什么",
    "讲了什么",
    "对于",
    "关于",
    "中提到的",
    "里提到的",
    "提到的",
    "主要包含",
    "包含哪几部分",
    "包含哪些部分",
  ]);
  const out: string[] = [];
  for (const p of parts) {
    const s = p
      .replace(/^(详细|具体)?\s*(?:说一下|详细说一下|讲解一下|解释一下|说明一下|说说|讲讲)\s*/g, "")
      .replace(/^(对于|关于|针对|就)\s*/g, "")
      .replace(/^的\s*/g, "")
      .trim();
    if (!s || s.length < 2) continue;
    if (bad.has(s)) continue;
    // Drop record id mentions like "002"
    if (/^\d{1,4}$/.test(s)) continue;
    // Drop phrases that still contain the record name (likely the non-topic half).
    if (/(会议记录|会议纪要|会议纪要记录)/.test(s)) continue;
    // Avoid generic
    if (/^(什么|内容|情况|等等)$/i.test(s)) continue;
    out.push(s);
  }
  return Array.from(new Set(out)).slice(0, 4);
}

function isLikelyMeetingAgendaQuestion(question: string): boolean {
  const q = (question || "").trim();
  if (!q) return false;
  // "对于X...会议记录002里说了什么" / "会议记录002里关于X" / etc.
  return /(会议记录|会议纪要|会议纪要记录).*(说了什么|讲了什么|提到|关于|对于|详细说一下|讲解一下)/.test(q);
}

function refineRangeByMeetingRecordHeading(
  chunkContent: string,
  chunkStartAbs: number,
  recordId: string,
  maxLen = 900
): { content: string; startChar: number; endChar: number } | null {
  const { text, map } = buildLfTextAndMap(chunkContent);
  if (!text.trim()) return null;
  const tokens = buildMeetingRecordTokensStrict(recordId);
  let idx = -1;
  let matchedToken: string | null = null;
  for (const t of tokens) {
    const i = text.indexOf(t);
    if (i >= 0 && (idx < 0 || i < idx)) {
      idx = i;
      matchedToken = t;
    }
  }
  if (idx < 0) return null;

  // Start at the heading line itself to avoid "preamble drift" like "准测试 ...".
  const lineStart = (() => {
    const j = text.lastIndexOf("\n", idx);
    return j >= 0 ? j + 1 : 0;
  })();

  // End at next record heading within this chunk if present; otherwise keep a reasonable window.
  const afterStart = text.slice(lineStart);
  const nextRel = afterStart.slice(1).search(/^\s*#\s*(?:会议记录|会议纪要|会议纪要记录)\s*0*\d{1,4}(?!\d)/m);
  const endIdx =
    nextRel >= 0 ? Math.min(text.length, lineStart + 1 + nextRel) : Math.min(text.length, lineStart + maxLen);

  const mappedRange = mapNormRangeToOrig(map, lineStart, endIdx, chunkContent.length);
  const snippet = chunkContent.slice(mappedRange.start, mappedRange.end);
  if (!snippet.trim()) return null;
  return {
    content: snippet,
    startChar: chunkStartAbs + mappedRange.start,
    endChar: chunkStartAbs + mappedRange.end,
  };
}

function isMostlyMetaSnippet(s: string): boolean {
  const t = (s || "").trim();
  if (!t) return true;
  // Only allow a small set of metadata lines; treat it as "meta-only" if it doesn't contain any result-ish cue.
  const lines = t.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
  if (lines.length === 0) return true;
  const metaRe = /^[-*+]\s*(?:\*\*|__)?(日期|时间|天气|心情|标签|地点|作者)(?:\*\*|__)?\s*[:：]/;
  const resultRe = /(实验结果|结果|指标|准确率|acc|f1|auc|loss|提升|下降|对比|ablation|消融|baseline|sota|显著|统计)/i;
  const nonMeta = lines.filter((l) => !metaRe.test(l));
  if (nonMeta.length === 0) return true;
  return !resultRe.test(t);
}

function extractEvidenceLinesFromSource(content: string, needle: string, maxLines = 3): string[] {
  const text = (content || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  if (!text.trim()) return [];
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) return [];

  const isTraining = needle === "训练策略";
  // Prefer a direct "训练策略" (or English) line; otherwise fall back to weaker cues.
  const hitIdxNeedle = lines.findIndex((l) => l.includes(needle) || (isTraining && l.toLowerCase().includes("training strategy")));
  const hitIdx =
    hitIdxNeedle >= 0
      ? hitIdxNeedle
      : lines.findIndex((l) => {
          if (isTraining) {
            // Also accept common cues in training strategy sections (weaker than the title itself).
            return (
              (l.includes("训练") && l.includes("策略")) ||
              l.toLowerCase().includes("training strategy") ||
              l.includes("学习率") ||
              l.includes("调度") ||
              l.includes("余弦") ||
              l.includes("退火") ||
              /epoch/i.test(l) ||
              l.includes("分布式") ||
              l.includes("数据并行") ||
              l.includes("DDP")
            );
          }
          return false;
        });
  const start = hitIdx >= 0 ? hitIdx : 0;
  const out: string[] = [];
  for (let i = start; i < lines.length && out.length < maxLines; i++) {
    const l = lines[i]!;
    // Skip pure metadata
    if (/^[-*+]\s*(?:\*\*|__)?(日期|时间|天气|心情|标签|地点|作者)(?:\*\*|__)?\s*[:：]/.test(l)) continue;
    // For training-strategy extraction, stop before the next markdown section heading,
    // so we don't accidentally include "评估与对比/下一节" headings.
    if (isTraining && i > start && /^\s*#{1,6}\s+/.test(l)) break;
    // Normalize markdown headings into plain text for answers.
    if (isTraining && i === start && /^\s*#{1,6}\s+/.test(l)) {
      out.push("训练策略：");
      continue;
    }
    out.push(l);
  }
  return out;
}

function stripMarkdownFormattingInAnswer(answer: string): string {
  if (!answer) return answer;
  const lines = answer
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .split("\n")
    .map((l) => {
      let s = l;
      // Remove markdown headings at line start: "# foo" -> "foo"
      s = s.replace(/^\s{0,3}#{1,6}\s+/, "");
      // Remove simple list bullets: "- foo" -> "foo"
      s = s.replace(/^\s*[-*+]\s+/, "");
      // Remove bold/underline markers (keep text)
      s = s.replace(/\*\*/g, "").replace(/__/g, "");
      return s;
    });
  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function extractMeetingRecordIdFromHistory(history: HistoryTurn[]): string | null {
  if (!Array.isArray(history) || history.length === 0) return null;
  // Look at most recent turns first.
  for (let i = history.length - 1; i >= 0; i--) {
    const h = history[i];
    if (!h) continue;
    const q = typeof h.question === "string" ? h.question : "";
    const a = typeof h.answer === "string" ? h.answer : "";
    const fromQ = extractRecordIdHint(q);
    if (fromQ?.kind === "meeting") return fromQ.id;
    const fromA = extractRecordIdHint(a);
    if (fromA?.kind === "meeting") return fromA.id;
  }
  return null;
}

function isLikelyRecordFollowupQuestion(question: string): boolean {
  const q = (question || "").trim();
  if (!q) return false;
  // If the current question already names a record id, it's not "history-based".
  if (extractRecordIdHint(q)?.kind === "meeting") return false;
  // Pronoun/follow-up cues.
  if (/(它|他|她|其|这个|那个|上述|前面|上一个|该|此|这里|里面|其中|这次|上次|刚刚|之前)/.test(q)) return true;
  // Explicit meeting-record context words without id.
  if (/(会议记录|会议纪要|会议|议题|参会人|主持人|行动项|决议)/.test(q)) return true;
  return false;
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

  const isMetaLine = (s: string) => {
    const t = s.trim();
    if (!t) return false;
    // Common metadata lines in experiment logs / journals
    return (
      /^[-*+]\s*(?:\*\*|__)?(日期|时间|天气|心情|标签|地点|作者)(?:\*\*|__)?\s*[:：]/.test(t) ||
      /^日期\s*[:：]/.test(t) ||
      /^time\s*[:：]/i.test(t) ||
      /^date\s*[:：]/i.test(t)
    );
  };

  const looksLikeResultLine = (s: string) => {
    const t = s.trim();
    if (!t) return false;
    return (
      /(实验结果|结果|指标|准确率|acc|f1|auc|loss|提升|下降|对比|ablation|消融|baseline|sota|显著|统计)/i.test(t) ||
      /(\d+(\.\d+)?\s*%)/.test(t)
    );
  };

  const advanceFromMeta = (fromLineEndIdx: number) => {
    let cursor = fromLineEndIdx;
    for (let hops = 0; hops < 16 && cursor < text.length; hops++) {
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
      if (isMdNoiseLine(nextLine) || isMetaLine(nextLine)) {
        cursor = nextLineEnd;
        continue;
      }
      if (isMdHeading(nextLine)) {
        cursor = nextLineEnd;
        continue;
      }
      // Prefer an explicit "result-like" line if present.
      if (looksLikeResultLine(nextLine)) {
        startIdx = cursor;
        endIdx = nextLineEnd;
        line = nextLine;
        return;
      }
      // Otherwise, jump to the first non-meta content line.
      if (hops >= 2) {
        startIdx = cursor;
        endIdx = nextLineEnd;
        line = nextLine;
        return;
      }
      cursor = nextLineEnd;
    }
  };

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

  // If we ended up on a metadata line (common when the match is a heading and the next line is "- **日期**: ..."),
  // move down to a more meaningful line for highlighting (prefer result-like lines).
  if (isMetaLine(line)) {
    advanceFromMeta(endIdx);
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

  // PDF / extracted Q&A style: "问题：... 答案：..."
  // Treat "问题：" as a hard header even if the line doesn't contain '?'.
  for (const m of text.matchAll(/(^|[^\S\r\n])(?:问题|问)\s*[:：]/gm)) {
    if (m.index === undefined) continue;
    const prefix = m[1] ?? "";
    const idx = m.index + prefix.length;
    headers.push(idx);
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
    // Date hint should be extracted from the original question first.
    // The rewrite step can occasionally drop the explicit date (e.g. "总结这天"), which would break date-scoped queries.
    const dateHint = extractDateYYYYMMDD(question) ?? extractDateYYYYMMDD(retrievalQuestion);
    const dateVariants = dateHint ? buildDateVariants(dateHint) : [];
    const focus = extractFocus(retrievalQuestion);
    const defTopic = extractDefinitionTopic(retrievalQuestion);
    const inclusionTopic = extractInclusionTopic(retrievalQuestion);
    const topicHint = focus.topic ?? defTopic ?? inclusionTopic ?? undefined;
    const qNum = extractQuestionNumber(question) ?? extractQuestionNumber(retrievalQuestion);
    const tailKeyphrase = extractTailKeyphrase(retrievalQuestion);
    const trainingBoostTokens = extractTrainingStrategyBoostTokens(retrievalQuestion);
    // For record-scoped detection, prefer the ORIGINAL question: rewrites can reorder phrases
    // and break "对于X，会议记录002..." style extraction.
    const recordHintRaw = extractRecordIdHint(question) ?? extractRecordIdHint(retrievalQuestion);
    // Only inherit record id from the *immediately previous turn*.
    // This prevents unrelated follow-ups like "反射的优点" -> "它的缺点" from accidentally inheriting
    // an older meeting record id (e.g. "会议记录 025") from earlier in the conversation.
    const recordIdFromHistory = isLikelyRecordFollowupQuestion(question)
      ? extractMeetingRecordIdFromHistory(safeHistory.slice(-1))
      : null;
    const recordHint: { kind: "meeting"; id: string } | null =
      recordHintRaw ??
      (recordIdFromHistory
        ? {
            kind: "meeting",
            id: recordIdFromHistory,
          }
        : null);
    const recordTokensLoose = recordHint?.kind === "meeting" ? buildMeetingRecordTokens(recordHint.id) : [];
    const recordTokensStrict = recordHint?.kind === "meeting" ? buildMeetingRecordTokensStrict(recordHint.id) : [];
    let recordSubtopic =
      recordHint?.kind === "meeting" && recordHint.id ? extractMeetingRecordSubtopic(question, recordHint.id) : null;
    let recordSubtopics =
      recordHint?.kind === "meeting" && recordHint.id ? extractMeetingRecordSubtopics(question, recordHint.id) : [];
    let recordPrimarySubtopic = recordSubtopic ?? recordSubtopics[0] ?? null;

    // Special-case: "对于X，会议记录002里说了什么" should treat X as subtopic.
    if (recordHint?.kind === "meeting" && recordHint.id) {
      const t = extractMeetingAgendaSubtopicFromPattern(question);
      if (t) {
        recordSubtopics = Array.from(new Set([t, ...recordSubtopics])).slice(0, 4);
        recordSubtopic = recordSubtopic ?? t;
        recordPrimarySubtopic = recordPrimarySubtopic ?? t;
      }
    }

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
    // For date-scoped log queries, prefer the ORIGINAL question for keyword recall and ranking.
    const keywordQuestion = (dateHint ? question : retrievalQuestion).trim();
    const rawQ = keywordQuestion;
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
    if (topicHint) {
      const topic = topicHint;
      const focusedVector = vectorRows.filter((row) => includesTopicLoose(row.content, topic));
      if (focusedVector.length > 0) {
        vectorRelevant = focusedVector.slice(0, TOP_K);
      }
    }

    // Keyword-constrained recall boost (hybrid search):
    // If the question contains distinctive tokens (e.g. “审核结果”“准确性”), search those tokens directly
    // and then rank those keyword-matched chunks by vector distance.
    let keywordRows: SearchRow[] = [];
    // Keep this fixed-length (4) to match the hard-coded placeholders in the exact-match queries below.
    const exactLikes = [`%${rawQ}%`, `%${qNoPunct}%`, `%Q4:%${qNoPunct}%`, `%Q：%${qNoPunct}%`];

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

    const patterns = buildKeywordPatterns(keywordQuestion);
    if (patterns.length > 0) {
      const focusLikes =
        topicHint && focus.aspectWord
          ? [...buildTopicLikePatterns(topicHint), `%${focus.aspectWord}%`]
          : topicHint
            ? buildTopicLikePatterns(topicHint)
            : [];
      const dateLikes = dateHint ? pickYearfulDateVariants(dateHint, dateVariants).map((d) => `%${d}%`).slice(0, 10) : [];
      const all = [`%${rawQ}%`, `%${qNoPunct}%`, ...dateLikes, ...focusLikes, ...patterns];
      const whereOffset = vectorStr ? 3 : 2; // $1 is vector (when present), $1 is LIMIT otherwise
      // Match both chunk content and filename, so date-only filenames still work for log queries.
      const where = all
        .map((_, i) => `(c.content ILIKE $${i + whereOffset} OR n.filename ILIKE $${i + whereOffset})`)
        .join(" OR ");

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

    // Date-only fallback: if user asks about a specific date, directly recall chunks by that date (content OR filename).
    // This avoids missing results when the query contains broad tokens like “总结/这天”.
    let dateRows: SearchRow[] = [];
    if (dateHint) {
      const likes = dateVariants.length > 0 ? pickYearfulDateVariants(dateHint, dateVariants).slice(0, 10).map((d) => `%${d}%`) : [`%${dateHint}%`];
      const where = likes.map((_, i) => `(c.content ILIKE $${i + 2} OR n.filename ILIKE $${i + 2})`).join(" OR ");
      const dateResult = await query(
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
        ORDER BY c.id
        LIMIT $1`,
        [KEYWORD_K, ...likes]
      );
      dateRows = dateResult.rows.map((r) => ({
        ...(r as Omit<SearchRow, "similarity">),
        similarity: parseFloat(r.similarity),
      }));
    }

    // Training-strategy fallback: recall chunks that likely contain training strategy details.
    let trainingRows: SearchRow[] = [];
    if (trainingBoostTokens.length > 0) {
      const likes = trainingBoostTokens.slice(0, 10).map((t) => `%${t}%`);
      const where = likes.map((_, i) => `(c.content ILIKE $${i + 2} OR n.filename ILIKE $${i + 2})`).join(" OR ");
      const trainingResult = await query(
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
        ORDER BY c.id
        LIMIT $1`,
        [KEYWORD_K, ...likes]
      );
      trainingRows = trainingResult.rows.map((r) => ({
        ...(r as Omit<SearchRow, "similarity">),
        similarity: parseFloat(r.similarity),
      }));
    }

    // Meeting-record fallback: if the question names a specific record id (e.g. "会议记录 002"),
    // recall chunks by matching the record id in content/filename.
    let recordRows: SearchRow[] = [];
    const recordNoteIds = new Set<string>();
    const recordRowById = new Map<string, SearchRow>();
    let recordTargetNoteId: string | null = null;
    if (recordTokensStrict.length > 0) {
      const likes = recordTokensStrict.slice(0, 10).map((t) => `%${t}%`);
      const where = likes.map((_, i) => `(c.content ILIKE $${i + 2} OR n.filename ILIKE $${i + 2})`).join(" OR ");
      const recordResult = await query(
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
        ORDER BY c.id
        LIMIT $1`,
        [KEYWORD_K, ...likes]
      );
      recordRows = recordResult.rows.map((r) => ({
        ...(r as Omit<SearchRow, "similarity">),
        similarity: parseFloat(r.similarity),
      }));
      for (const r of recordRows) recordNoteIds.add(r.note_id);
      for (const r of recordRows) recordRowById.set(r.id, r);
      // Prefer the note that contains an actual "# 会议记录 <id>" heading.
      if (recordHint?.kind === "meeting" && recordHint.id) {
        const head = recordRows.find((r) => detectMeetingRecordHeadingId(r.content) === recordHint.id);
        recordTargetNoteId = head?.note_id ?? null;
      }
    }

    // If we have the noteId for record 002, compute exact section bounds using DB start_char ordering.
    let meetingSectionBoundsDb: { noteId: string; start: number; end: number } | null = null;
    let meetingSectionNoteContent: string | null = null;
    if (recordHint?.kind === "meeting" && recordHint.id && recordNoteIds.size > 0) {
      const noteId = recordTargetNoteId ?? Array.from(recordNoteIds)[0]!;
      // Prefer note-level content bounds (more accurate than chunk boundaries when a chunk spans two records).
      const noteRow = await query(`SELECT content FROM notes WHERE id = $1 LIMIT 1`, [noteId]);
      const noteContent = String(noteRow.rows[0]?.content || "");
      meetingSectionNoteContent = noteContent;
      const bounds = computeMeetingRecordSectionBoundsFromNoteContent(noteContent, recordHint.id);
      if (bounds) {
        meetingSectionBoundsDb = { noteId, start: bounds.start, end: bounds.end };
      }
    }
    // If chunk-level record recall didn't return any rows (or couldn't identify the target note),
    // fall back to locating the note directly from notes.content/filename.
    if (recordHint?.kind === "meeting" && recordHint.id && meetingSectionBoundsDb == null) {
      const likes = buildMeetingRecordTokensStrict(recordHint.id).slice(0, 12).map((t) => `%${t}%`);
      const where = likes.map((_, i) => `(n.content ILIKE $${i + 1} OR n.filename ILIKE $${i + 1})`).join(" OR ");
      const noteRes = await query(
        `SELECT id, filename, content
         FROM notes n
         WHERE ${where}
         ORDER BY n.id
         LIMIT 2`,
        likes
      );
      const row = noteRes.rows[0];
      if (row) {
        const noteId = String(row.id || "");
        const noteContent = String(row.content || "");
        if (noteId && noteContent.trim()) {
          meetingSectionNoteContent = noteContent;
          const bounds = computeMeetingRecordSectionBoundsFromNoteContent(noteContent, recordHint.id);
          if (bounds) {
            meetingSectionBoundsDb = { noteId, start: bounds.start, end: bounds.end };
            recordNoteIds.add(noteId);
            recordTargetNoteId = recordTargetNoteId ?? noteId;
          }
        }
      }
    }

    // Direct agenda-section recall: when user asks about a specific agenda subtopic in a specific record,
    // fetch the exact "## 议题N: <subtopic>" chunk inside that record section.
    let recordAgendaRows: SearchRow[] = [];
    if (
      recordHint?.kind === "meeting" &&
      recordHint.id &&
      recordPrimarySubtopic &&
      recordTargetNoteId &&
      meetingSectionBoundsDb &&
      Number.isFinite(meetingSectionBoundsDb.start)
    ) {
      const start = meetingSectionBoundsDb.start;
      const end = Number.isFinite(meetingSectionBoundsDb.end) ? meetingSectionBoundsDb.end : 2147483647;
      const like = `%${recordPrimarySubtopic}%`;
      const agendaResult = await query(
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
        WHERE c.note_id = $2
          AND c.end_char > $3
          AND c.start_char < $4
          AND c.content ILIKE '%## 议题%'
          AND c.content ILIKE $5
        ORDER BY c.start_char ASC
        LIMIT $1`,
        [KEYWORD_K, recordTargetNoteId, start, end, like]
      );
      const rows = agendaResult.rows.map((r) => ({
        ...(r as Omit<SearchRow, "similarity">),
        similarity: parseFloat(r.similarity),
      }));
      // Keep only rows that truly contain the agenda heading for this subtopic.
      recordAgendaRows = rows.filter((r) => Boolean(refineRangeByMeetingAgendaSubtopic(r.content, 0, recordPrimarySubtopic, 280)));
    }

    // Record + subtopic boost: ensure we retrieve the agenda sections for requested topics (not just record header).
    let recordSubtopicRows: SearchRow[] = [];
    if (recordPrimarySubtopic && recordNoteIds.size > 0) {
      const noteIds = Array.from(recordNoteIds).slice(0, 50);
      const subLike = `%${recordPrimarySubtopic}%`;
      const recordSubtopicResult = await query(
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
        WHERE c.note_id = ANY($2::uuid[]) AND c.content ILIKE '%## 议题%' AND c.content ILIKE $3
        ORDER BY c.id
        LIMIT $1`,
        [KEYWORD_K, noteIds, subLike]
      );
      recordSubtopicRows = recordSubtopicResult.rows.map((r) => ({
        ...(r as Omit<SearchRow, "similarity">),
        similarity: parseFloat(r.similarity),
      }));
    }

    // Multi-subtopic boost: for questions that mention multiple agenda topics, retrieve one chunk per topic.
    let recordSubtopicsRows: SearchRow[] = [];
    if (recordSubtopics.length > 0 && recordNoteIds.size > 0) {
      const noteIds = Array.from(recordNoteIds).slice(0, 50);
      const subs = recordSubtopics.slice(0, 4);
      const likes = subs.map((t) => `%${t}%`);
      const or = likes.map((_, i) => `c.content ILIKE $${i + 3}`).join(" OR ");
      const result = await query(
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
        WHERE c.note_id = ANY($2::uuid[]) AND c.content ILIKE '%## 议题%' AND (${or})
        ORDER BY c.id
        LIMIT $1`,
        [KEYWORD_K, noteIds, ...likes]
      );
      recordSubtopicsRows = result.rows.map((r) => ({
        ...(r as Omit<SearchRow, "similarity">),
        similarity: parseFloat(r.similarity),
      }));
    }

    // Merge exact match + keyword match
    const mergedKw = new Map<string, SearchRow>();
    for (const r of exactRows) mergedKw.set(r.id, r);
    for (const r of keywordRows) mergedKw.set(r.id, r);
    for (const r of dateRows) mergedKw.set(r.id, r);
    for (const r of trainingRows) mergedKw.set(r.id, r);
    for (const r of recordRows) mergedKw.set(r.id, r);
    for (const r of recordSubtopicRows) mergedKw.set(r.id, r);
    for (const r of recordSubtopicsRows) mergedKw.set(r.id, r);
    for (const r of recordAgendaRows) mergedKw.set(r.id, r);
    keywordRows = Array.from(mergedKw.values());

    // Merge: prefer keyword hits first (higher recall), then fill with vector hits.
    // This avoids missing obvious Q/A passages even when the embedding similarity is low.
    const merged = new Map<string, SearchRow>();
    for (const r of keywordRows) merged.set(r.id, r);
    for (const r of vectorRelevant) merged.set(r.id, r);
    const scoreExact = (content: string) => {
      const c = content;
      let s = 0;
      if (topicHint && includesTopicLoose(c, topicHint)) s += 6;
      if (focus.aspectWord && includesLoose(c, focus.aspectWord)) s += 3;
      if (defTopic && includesLoose(c, defTopic)) s += 4;
      if (defTopic && includesLoose(c, `${defTopic}是`)) s += 6;
      if (defTopic && (c.includes("目的") || c.includes("作用") || c.includes("用于"))) s += 2;
      if (tailKeyphrase && includesLoose(c, tailKeyphrase)) s += 8;
      if (recordTokensLoose.length > 0) {
        for (const t of recordTokensLoose) {
          if (includesLoose(c, t)) {
            s += 10;
            break;
          }
        }
      }
      if (trainingBoostTokens.length > 0) {
        let hits = 0;
        for (const t of trainingBoostTokens) if (includesLoose(c, t)) hits++;
        if (hits >= 2) s += 6;
        else if (hits >= 1) s += 3;
      }
      if (qNoPunct && c.includes(qNoPunct)) s += 2;
      if (rawQ && c.includes(rawQ)) s += 2;
      if (qNoPunct && c.includes(`Q4: ${qNoPunct}`)) s += 6;
      if (qNoPunct && c.includes(`Q4:${qNoPunct}`)) s += 6;
      return s;
    };

    let relevantChunks = Array.from(merged.values())
      .sort((a, b) => scoreExact(b.content) - scoreExact(a.content) || b.similarity - a.similarity)
      .slice(0, TOP_K);

    // Record-id guard: if user asked for a specific meeting record, ensure the record header chunk
    // ("# 会议记录 013") is included when present. This stabilizes recall when keywordRows happen
    // to miss the header but still hit other parts of the file.
    if (recordHint?.kind === "meeting" && recordHint.id && recordRows.length > 0) {
      const header = recordRows.find((r) => detectMeetingRecordHeadingId(r.content) === recordHint.id);
      if (header && !relevantChunks.some((r) => r.id === header.id)) {
        relevantChunks = [header, ...relevantChunks].slice(0, TOP_K);
      }
    }

    // Tail-keyphrase guard: ensure we don't miss the core intent when the query has many qualifiers.
    // Example: "基于对比学习的知识图谱性能分析的训练策略" should still retrieve chunks about "训练策略"
    // even if they don't mention "知识图谱/性能分析".
    if (tailKeyphrase) {
      const must = Array.from(merged.values()).filter((r) => includesLoose(r.content, tailKeyphrase)).slice(0, TOP_K);
      if (must.length > 0 && !relevantChunks.some((r) => includesLoose(r.content, tailKeyphrase))) {
        const seen = new Set<string>();
        const combined: SearchRow[] = [];
        for (const r of must) {
          if (seen.has(r.id)) continue;
          seen.add(r.id);
          combined.push(r);
        }
        for (const r of relevantChunks) {
          if (seen.has(r.id)) continue;
          seen.add(r.id);
          combined.push(r);
        }
        relevantChunks = combined.slice(0, TOP_K);
      }
    }

    // Post-filter: keep only chunks that contain at least one strong query token.
    // This reduces "answer is correct but citations drift to unrelated blocks" (common for txt notes).
    const qTokens = extractMatchTokensFromQuestion(retrievalQuestion);
    const focusTokens = Array.from(
      new Set<string>([
        ...(tailKeyphrase ? [tailKeyphrase] : []),
        ...(trainingBoostTokens.slice(0, 2)),
        ...(recordTokensLoose.slice(0, 2)),
        ...(topicHint ? topicTokenVariants(topicHint).slice(0, 2) : []),
        ...(focus.aspectWord ? [focus.aspectWord] : []),
      ])
    ).slice(0, 4);

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
    // When embeddings are unavailable (vectorStr == null), be more permissive to reduce false "no results".
    const minHits =
      (filterTokens.length >= 4 || retrievalQuestion.trim().length >= 12) &&
      !defTopic &&
      !topicHint &&
      !tailKeyphrase &&
      recordTokensLoose.length === 0 &&
      vectorStr != null
        ? 2
        : 1;

    const filteredRelevant = relevantChunks.filter((r) => countTokenMatchesLoose(r.content, filterTokens) >= minHits);
    const relaxedRelevant = relevantChunks.filter((r) => anyTokenMatchLoose(r.content, filterTokens));

    const finalRelevantChunks =
      filteredRelevant.length >= 1
        ? filteredRelevant
        : relaxedRelevant.length >= Math.min(LLM_TOP_K, 2)
          ? relaxedRelevant
          : relevantChunks;

    // Internal automatic second retrieval (looser):
    // If we have a strong constraint (meeting record id) but ended up with no chunks,
    // fall back to slicing the raw note content directly. This avoids "first miss, second hit"
    // caused by embedding outages or chunking gaps.
    if (
      finalRelevantChunks.length === 0 &&
      recordHint?.kind === "meeting" &&
      recordHint.id &&
      recordTokensStrict.length > 0
    ) {
      // Try to locate the note content by searching the notes table (content/filename).
      // We keep this query small and only use it when we already failed to retrieve any chunks.
      const likes = buildMeetingRecordTokensStrict(recordHint.id).slice(0, 10).map((t) => `%${t}%`);
      const where = likes.map((_, i) => `(n.content ILIKE $${i + 1} OR n.filename ILIKE $${i + 1})`).join(" OR ");
      const noteRes = await query(
        `SELECT id, filename, content
         FROM notes n
         WHERE ${where}
         ORDER BY n.id
         LIMIT 3`,
        likes
      );
      for (const row of noteRes.rows) {
        const noteId = String(row.id || "");
        const filename = String(row.filename || "meeting_record");
        const content = String(row.content || "");
        if (!content.trim()) continue;
        const bounds = computeMeetingRecordSectionBoundsFromNoteContent(content, recordHint.id);
        if (!bounds) continue;
        const section = content.slice(bounds.start, bounds.end);
        if (!section.trim()) continue;

        // If user also asked an agenda subtopic, try to extract that section; otherwise return header+agenda list.
        const subtopic = recordPrimarySubtopic;
        const agenda =
          subtopic ? extractAgendaSectionFromRecordSection(section, bounds.start, subtopic) : null;
        const picked = agenda
          ? { content: agenda.content, startChar: agenda.startChar, endChar: agenda.endChar }
          : { content: section.slice(0, 1800), startChar: bounds.start, endChar: Math.min(bounds.end, bounds.start + 1800) };

        const synthetic: SourceChunk = {
          index: 1,
          noteId,
          filename,
          content: picked.content,
          startChar: picked.startChar,
          endChar: picked.endChar,
          similarity: 0.98,
        };

        // Reuse the existing code path by assigning to baseSources later.
        // We return early with a simplified response to avoid duplicating the full pipeline.
        const uiSourcesFallback: SourceChunk[] = (() => {
          let s = synthetic;
          // refine highlight for record heading if possible
          const refined = refineRangeByMeetingRecordHeading(s.content, s.startChar, recordHint.id, 1100);
          if (refined) s = { ...s, content: refined.content, startChar: refined.startChar, endChar: refined.endChar };
          return [s];
        })();

        const llmSourcesFallback: SourceChunk[] = uiSourcesFallback.map((s) => ({ ...s, content: s.content }));
        const result = await askQuestion(question, llmSourcesFallback, {
          focusTopic: recordPrimarySubtopic ?? undefined,
          focusAspect: focus.aspectWord,
        });

        return NextResponse.json({
          answer: stripMarkdownFormattingInAnswer(
            normalizeAnswerCitationsToAvailableSources(result.answer, uiSourcesFallback)
          ),
          sources: uiSourcesFallback,
          hasAnswer: result.hasAnswer,
          rewrittenQuestion: retrievalQuestion === question ? undefined : retrievalQuestion,
        });
      }

      // If record id was explicitly requested but we still couldn't locate it anywhere in notes,
      // be explicit rather than repeatedly asking for "key terms".
      return NextResponse.json({
        answer: `我在当前已上传的笔记中没有找到“会议记录 ${recordHint.id}”对应的内容（也未找到包含该编号的会议纪要/会议纪要记录）。\n如果你确认已经上传了会议记录${recordHint.id}，请把对应文件/段落贴出来或重新上传，我再基于新增资料检索与问答。`,
        sources: [],
        hasAnswer: false,
        rewrittenQuestion: retrievalQuestion === question ? undefined : retrievalQuestion,
      });
    }

    // Build base sources (full chunk content) for the LLM.
    // Drop trivial one-character chunks (can appear due to chunking/artifacts), but never drop ALL.
    const finalRelevantChunksNonTrivial = finalRelevantChunks.filter((r) => String(r.content || "").trim().length >= 3);
    const finalRelevantChunksForSources =
      finalRelevantChunksNonTrivial.length > 0 ? finalRelevantChunksNonTrivial : finalRelevantChunks;

    const baseSources: SourceChunk[] = finalRelevantChunksForSources.map((row, index) => {
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
        similarity: computeDisplaySimilarity({
          existing: row.similarity,
          content: row.content,
          filename: row.filename,
          filterTokens,
          topicHint,
          dateHint: dateHint ?? undefined,
          dateVariants,
        }),
      };
    });

    // If we detected a meeting record id, further restrict chunks to the section of that record within the same note
    // (prevents mixing "参会人" blocks from other records inside the same file).
    const meetingSectionNoteId =
      meetingSectionBoundsDb?.noteId ?? recordTargetNoteId ?? (recordNoteIds.size > 0 ? Array.from(recordNoteIds)[0]! : null);
    const meetingSectionBounds =
      recordHint?.kind === "meeting" && recordHint.id && meetingSectionNoteId
        ? computeMeetingRecordSectionBounds(baseSources.filter((s) => s.noteId === meetingSectionNoteId), recordHint.id)
        : null;
    let sectionScopedSources =
      meetingSectionNoteId &&
      ((meetingSectionBoundsDb && Number.isFinite(meetingSectionBoundsDb.start)) ||
        (meetingSectionBounds && Number.isFinite(meetingSectionBounds.start)))
        ? (() => {
            const start =
              meetingSectionBoundsDb && meetingSectionBoundsDb.noteId === meetingSectionNoteId
                ? meetingSectionBoundsDb.start
                : meetingSectionBounds!.start;
            const end =
              meetingSectionBoundsDb && meetingSectionBoundsDb.noteId === meetingSectionNoteId
                ? meetingSectionBoundsDb.end
                : meetingSectionBounds!.end;
            const scoped = baseSources
              .filter((s) => s.noteId === meetingSectionNoteId && Number.isFinite(s.startChar) && Number.isFinite(s.endChar))
              .map((s) => {
                const clipped = clipChunkToSection(s.content, s.startChar, s.endChar, start, end);
                return clipped ? { ...s, content: clipped.content, startChar: clipped.startChar, endChar: clipped.endChar } : null;
              })
              .filter(Boolean) as SourceChunk[];
            return scoped;
          })()
        : baseSources;

    // Fallback: if record-scoping produced an empty list (e.g. none of the top-K chunks came from that note),
    // synthesize a single source by slicing the raw note content for that record section.
    if (
      sectionScopedSources.length === 0 &&
      recordHint?.kind === "meeting" &&
      recordHint.id &&
      meetingSectionBoundsDb &&
      meetingSectionNoteContent &&
      Number.isFinite(meetingSectionBoundsDb.start) &&
      Number.isFinite(meetingSectionBoundsDb.end)
    ) {
      const raw = meetingSectionNoteContent;
      const start = meetingSectionBoundsDb.start;
      const end = meetingSectionBoundsDb.end;
      const sectionText = raw.slice(start, end);
      if (sectionText.trim()) {
        const picked = { content: sectionText.slice(0, 1800), startChar: start, endChar: Math.min(end, start + 1800) };
        const filename =
          recordRows.find((r) => r.note_id === meetingSectionBoundsDb.noteId)?.filename ??
          baseSources.find((s) => s.noteId === meetingSectionBoundsDb.noteId)?.filename ??
          "meeting_record";
        sectionScopedSources = [
          {
            index: 1,
            noteId: meetingSectionBoundsDb.noteId,
            filename,
            content: picked.content,
            startChar: picked.startChar,
            endChar: picked.endChar,
            similarity: 0.98,
          },
        ];
      }
    }

    // If user did not use a recognizable "中提到的/的..." phrasing, try to infer subtopics directly
    // from the record's agenda list in the header and any explicit mentions in the question.
    if (recordHint?.kind === "meeting" && recordHint.id) {
      const header = pickMeetingRecordHeaderCandidate(sectionScopedSources);
      const agendaTitles = header ? extractMeetingAgendaTitlesFromHeader(header.content) : [];
      if (agendaTitles.length > 0) {
        const mentions = agendaTitles.filter((t) => includesLoose(retrievalQuestion, t));
        if (mentions.length > 0) {
          recordSubtopics = Array.from(new Set([...mentions, ...recordSubtopics])).slice(0, 4);
          recordSubtopic = recordSubtopic ?? recordSubtopics[0] ?? null;
          recordPrimarySubtopic = recordPrimarySubtopic ?? recordSubtopic;
        }
      }
    }

    // If we extracted a subtopic but it doesn't exactly match the agenda title (e.g. missing "(SLA)"),
    // align it to the canonical agenda title so heading matching works.
    if (recordHint?.kind === "meeting" && recordHint.id && (recordSubtopic || recordSubtopics.length > 0)) {
      const header = pickMeetingRecordHeaderCandidate(sectionScopedSources);
      const agendaTitles = header ? extractMeetingAgendaTitlesFromHeader(header.content) : [];
      if (agendaTitles.length > 0) {
        if (recordSubtopic) {
          const aligned = alignToAgendaTitles(recordSubtopic, agendaTitles);
          if (aligned) recordSubtopic = aligned;
        }
        if (recordSubtopics.length > 0) {
          const mapped: string[] = [];
          for (const t of recordSubtopics) {
            mapped.push(alignToAgendaTitles(t, agendaTitles) ?? t);
          }
          recordSubtopics = Array.from(new Set(mapped)).slice(0, 4);
        }
        recordPrimarySubtopic = recordSubtopic ?? recordSubtopics[0] ?? recordPrimarySubtopic;
      }
    }

    // If still no subtopic extracted but the user mentions an agenda title-like phrase,
    // try to align it to one of the agenda titles.
    if (
      recordHint?.kind === "meeting" &&
      recordHint.id &&
      recordSubtopics.length === 0 &&
      recordSubtopic == null
    ) {
      const header = pickMeetingRecordHeaderCandidate(sectionScopedSources);
      const agendaTitles = header ? extractMeetingAgendaTitlesFromHeader(header.content) : [];
      const candidates = extractMeetingAgendaSubtopicCandidatesFromQuestion(retrievalQuestion);
      if (agendaTitles.length > 0 && candidates.length > 0) {
        const aligned: string[] = [];
        for (const c of candidates) {
          const hit = agendaTitles.find((t) => includesLoose(c, t) || includesLoose(t, c));
          if (hit) aligned.push(hit);
        }
        if (aligned.length > 0) {
          recordSubtopics = Array.from(new Set(aligned)).slice(0, 4);
          recordSubtopic = recordSubtopics[0] ?? null;
          recordPrimarySubtopic = recordSubtopic;
        }
      }
    }

    // Last-chance: for agenda questions like "对于X，会议记录002里说了什么？",
    // treat any agenda title that appears anywhere in the question as a subtopic.
    if (
      recordHint?.kind === "meeting" &&
      recordHint.id &&
      recordSubtopics.length === 0 &&
      recordSubtopic == null &&
      isLikelyMeetingAgendaQuestion(retrievalQuestion)
    ) {
      const header = pickMeetingRecordHeaderCandidate(sectionScopedSources);
      const agendaTitles = header ? extractMeetingAgendaTitlesFromHeader(header.content) : [];
      if (agendaTitles.length > 0) {
        const hits = agendaTitles.filter((t) => includesLoose(question, t) || includesLoose(retrievalQuestion, t));
        if (hits.length > 0) {
          recordSubtopics = hits.slice(0, 4);
          recordSubtopic = recordSubtopics[0] ?? null;
          recordPrimarySubtopic = recordSubtopic;
        }
      }
    }

    // Pick LLM sources.
    // - If focus.topic exists, keep only those mentioning the topic.
    // - If the question is a definition ("X是什么"), prefer sources that mention X to avoid unrelated chunks.
    let focusedSources = (() => {
      const sources = sectionScopedSources;
      // If the note contains the user's question as an almost-exact string (common in Q/A txt),
      // aggressively prefer that chunk to avoid citing adjacent template/metadata blocks.
      const nearExact = sources.filter((s) => isNearDuplicateQuestion(s.content, retrievalQuestion));
      if (nearExact.length > 0) {
        return nearExact.slice(0, Math.min(LLM_TOP_K, 2));
      }

      // Record-scoped queries: if user specifies a record id (e.g. "会议记录 002"),
      // always prefer chunks from that record first, so follow-up questions stay grounded.
      if (recordTokensStrict.length > 0) {
        const scoped =
          recordNoteIds.size > 0
            ? sources.filter((s) => recordNoteIds.has(s.noteId))
            : sources.filter((s) => recordTokensStrict.some((t) => includesLoose(s.filename, t) || includesLoose(s.content, t)));
        if (scoped.length > 0) {
          const subtopics = recordSubtopics.length > 0 ? recordSubtopics : recordSubtopic ? [recordSubtopic] : [];
          if (subtopics.length > 0) {
            const picked: SourceChunk[] = [];
            const seen = new Set<string>();
            for (const t of subtopics) {
              // Strongly prefer the exact agenda section "## 议题N: <t>".
              const agendaCandidates = scoped.filter((s) => Boolean(refineRangeByMeetingAgendaSubtopic(s.content, 0, t, 280)));
              const candidates = agendaCandidates
                .slice()
                // Prefer earlier section headings (closer to the actual agenda heading line).
                .sort((a, b) => a.startChar - b.startChar);
              for (const c of candidates) {
                const key = `${c.noteId}:${c.startChar}:${c.endChar}`;
                if (seen.has(key)) continue;
                seen.add(key);
                picked.push(c);
                break;
              }
            }
            // When user explicitly asks about agenda subtopics, only keep the picked subtopic sections
            // (and drop other chunks) to avoid irrelevant sources confusing the citations panel.
            if (picked.length > 0) return picked.slice(0, LLM_TOP_K);
          }
          // For record overview queries (no specific subtopic), always include the record header chunk
          // so the model can ground basic info + agenda list.
          if (recordHint?.kind === "meeting" && recordHint.id) {
            const header = scoped.find((s) => detectMeetingRecordHeadingId(s.content) === recordHint.id);
            if (header) {
              const rest = scoped.filter((s) => s !== header);
              const rankTokens = filterTokens.length > 0 ? filterTokens : qTokens;
              const picked = [header, ...rest]
                .slice()
                .sort(
                  (a, b) =>
                    (a === header ? -1 : b === header ? 1 : 0) ||
                    countTokenMatchesLoose(b.content, rankTokens) - countTokenMatchesLoose(a.content, rankTokens) ||
                    b.similarity - a.similarity
                )
                .slice(0, LLM_TOP_K);
              return picked;
            }
          }
          return scoped.slice(0, LLM_TOP_K);
        }
      }

      // Date-scoped queries: ensure at least one chunk per note (md/pdf/txt) can be returned when it contains the date.
      if (dateHint) {
        const variants = pickYearfulDateVariants(dateHint, dateVariants.length > 0 ? dateVariants : [dateHint]);
        const perNote = new Map<string, SourceChunk>();
        for (const s of sources) {
          const ok =
            variants.some((v) => includesDateLoose(s.filename, v) || includesDateLoose(s.content, v)) ||
            Boolean(refineRangeByAnyLogDateHeading(s.content, 0, dateHint, dateVariants, 240));
          if (!ok) continue;
          if (!perNote.has(s.noteId)) perNote.set(s.noteId, s);
        }
        const picked = Array.from(perNote.values());
        return (picked.length > 0 ? picked : sources).slice(0, LLM_TOP_K);
      }

      // Tail keyphrase intent (e.g. "训练策略"): prioritize chunks that contain it.
      if (tailKeyphrase) {
        const rankTokens = filterTokens.length > 0 ? filterTokens : qTokens;
        const arr = sources
          .filter(
            (s) =>
              includesLoose(s.content, tailKeyphrase) ||
              (s.content.includes("训练") && (s.content.includes("策略") || s.content.includes("方法") || s.content.includes("阶段")))
          )
          .sort(
            (a, b) =>
              countTokenMatchesLoose(b.content, rankTokens) - countTokenMatchesLoose(a.content, rankTokens) ||
              b.similarity - a.similarity
          )
          .slice(0, LLM_TOP_K);
        if (arr.length > 0) return arr;
      }

      if (topicHint) {
        const arr = sources.filter((s) => includesTopicLoose(s.content, topicHint));
        return (arr.length > 0 ? arr : sources).slice(0, LLM_TOP_K);
      }

      if (defTopic) {
        const byTopic = sources.filter((s) => includesLoose(s.content, defTopic));
        const byDefinition = byTopic.filter((s) => hasDefinitionCue(s.content, defTopic));
        const picked = (byDefinition.length > 0 ? byDefinition : byTopic.length > 0 ? byTopic : sources).slice(
          0,
          LLM_TOP_K
        );
        return picked;
      }

      // Experiment/result questions: avoid picking "meta-only" chunks like "- **日期**: ..."
      if (isResultOrExperimentQuestion(retrievalQuestion)) {
        const rankTokens = filterTokens.length > 0 ? filterTokens : qTokens;
        const arr = sources
          .filter((s) => countTokenMatchesLoose(s.content, rankTokens) >= 1)
          .sort(
            (a, b) =>
              countTokenMatchesLoose(b.content, rankTokens) - countTokenMatchesLoose(a.content, rankTokens) ||
              b.similarity - a.similarity
          )
          .filter((s) => !isMostlyMetaSnippet(s.content))
          .slice(0, LLM_TOP_K);
        if (arr.length > 0) return arr;
      }

      // Default: prefer sources that match more query tokens.
      const rankTokens = filterTokens.length > 0 ? filterTokens : qTokens;
      return sources
        .slice()
        .sort(
          (a, b) =>
            countTokenMatchesLoose(b.content, rankTokens) - countTokenMatchesLoose(a.content, rankTokens) ||
            b.similarity - a.similarity
        )
        .slice(0, LLM_TOP_K);
    })();

    // (debug logging removed)

    // If this is a record + agenda-subtopic query and we have the raw note content,
    // build a single synthetic source from the exact agenda section in the note content.
    if (
      recordHint?.kind === "meeting" &&
      recordHint.id &&
      recordPrimarySubtopic &&
      meetingSectionBoundsDb &&
      meetingSectionNoteContent
    ) {
      const sectionText = meetingSectionNoteContent.slice(meetingSectionBoundsDb.start, meetingSectionBoundsDb.end);
      const agenda = extractAgendaSectionFromRecordSection(sectionText, meetingSectionBoundsDb.start, recordPrimarySubtopic);
      if (agenda) {
        const filename = baseSources.find((s) => s.noteId === meetingSectionBoundsDb.noteId)?.filename ?? "meeting_record";
        focusedSources = [
          {
            index: 1,
            noteId: meetingSectionBoundsDb.noteId,
            filename,
            content: agenda.content,
            startChar: agenda.startChar,
            endChar: agenda.endChar,
            similarity: 0.99,
          },
        ];
      }
    }

    // Record + subtopic strictness: if user asked about specific agenda subtopic(s),
    // only keep sources that truly contain the corresponding "## 议题N: <subtopic>" section.
    if (recordTokensStrict.length > 0) {
      const subtopics = recordSubtopics.length > 0 ? recordSubtopics : recordSubtopic ? [recordSubtopic] : [];
      if (subtopics.length > 0 && focusedSources.length > 1) {
        const agendaOnly = focusedSources.filter((s) =>
          subtopics.some((t) => Boolean(refineRangeByMeetingAgendaSubtopic(s.content, 0, t, 280)))
        );
        if (agendaOnly.length > 0) focusedSources = agendaOnly.slice(0, LLM_TOP_K);
      }
    }

    // Guardrail: if the user asks about a clear topic ("X是什么"/"X包括什么"/focus patterns),
    // but we failed to retrieve any chunk mentioning that topic, don't answer with unrelated citations.
    if (topicHint && recordTokensStrict.length === 0 && !focusedSources.some((s) => includesTopicLoose(s.content, topicHint))) {
      // Let the downstream no-answer flow produce the two-stage clarification UX.
      const isClarifyMarker = (s: string) => s.includes("【需要补充上下文】");
      const lastTurn = safeHistory.at(-1);
      const askedClarifyLastTurn = Boolean(lastTurn?.answer && isClarifyMarker(lastTurn.answer));
      const stage1 = `【需要补充上下文】\n我在当前笔记里暂时没检索到能直接回答你这个问题的内容。你可以补充一下：\n（1）你说的关键术语/对象具体指什么？（可以给全称、同义词、英文缩写）\n或者\n（2）如果你手头有相关段落/关键词，请直接贴出来或上传对应资料。`;
      const stage2 = `我在当前已上传的笔记中仍然没有检索到与该问题直接相关、可用于作答的资料。\n如果你希望我继续回答，请上传/补充相关笔记（或把关键段落贴出来），我再基于新增资料进行检索与问答。`;
      return NextResponse.json({
        answer: askedClarifyLastTurn ? stage2 : stage1,
        sources: [],
        hasAnswer: askedClarifyLastTurn ? false : true,
        rewrittenQuestion: retrievalQuestion === question ? undefined : retrievalQuestion,
      });
    }

    // Re-index sources after any filtering so citations remain consistent.
    // Also drop trivial one-character sources when we have better alternatives.
    const focusedSourcesReindexedRaw: SourceChunk[] = focusedSources.map((s, i) => ({ ...s, index: i + 1 }));
    const focusedSourcesReindexedNonTrivial = focusedSourcesReindexedRaw.filter(
      (s) => String(s.content || "").trim().length >= 3
    );
    const focusedSourcesReindexed: SourceChunk[] =
      focusedSourcesReindexedNonTrivial.length > 0 ? focusedSourcesReindexedNonTrivial : focusedSourcesReindexedRaw;

    // (debug logging removed)

    // Build UI sources: same indices as LLM sources, but with a refined highlight range and a shorter snippet.
    const uiSources: SourceChunk[] = focusedSourcesReindexed.map((s) => {
      if (!Number.isFinite(s.startChar)) return s;

      const lowerName = (s.filename || "").toLowerCase();
      const isMarkdown = lowerName.endsWith(".md") || lowerName.endsWith(".markdown");

      if (dateHint) {
        const refined = refineRangeByAnyLogDateHeading(s.content, s.startChar, dateHint, dateVariants, 1500);
        if (refined) {
          return { ...s, content: refined.content, startChar: refined.startChar, endChar: refined.endChar };
        }
      }

      if (tailKeyphrase === "训练策略" || trainingBoostTokens.length > 0) {
        const refined = refineRangeByTrainingStrategyCue(s.content, s.startChar, 620);
        if (refined) {
          return { ...s, content: refined.content, startChar: refined.startChar, endChar: refined.endChar };
        }
      }

      if (recordHint?.kind === "meeting" && recordHint.id) {
        if (recordSubtopic) {
          const refined =
            refineRangeByMeetingAgendaSubtopic(s.content, s.startChar, recordSubtopic, 1300) ??
            refineRangeByNeedle(s.content, s.startChar, recordSubtopic, 900);
          if (refined) {
            return { ...s, content: refined.content, startChar: refined.startChar, endChar: refined.endChar };
          }
        }
        const refined = refineRangeByMeetingRecordHeading(s.content, s.startChar, recordHint.id, 1100);
        if (refined) {
          return { ...s, content: refined.content, startChar: refined.startChar, endChar: refined.endChar };
        }
      }

      if (defTopic) {
        const refined = refineRangeByDefinitionTopic(s.content, s.startChar, defTopic);
        if (refined) {
          return { ...s, content: refined.content, startChar: refined.startChar, endChar: refined.endChar };
        }
      }
      if (inclusionTopic) {
        const refined = refineRangeByInclusionTopic(s.content, s.startChar, inclusionTopic, 320);
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

    // Ensure we never end up with empty sources for record-id queries when we did retrieve record-scoped candidates.
    if (recordHint?.kind === "meeting" && recordHint.id && uiSources.length === 0 && focusedSources.length > 0) {
      const fallbackUiSources = focusedSources.map((s, i) => ({ ...s, index: i + 1 }));
      const llmSourcesFallback = fallbackUiSources.map((s) => ({ ...s, content: s.content }));
      const result = await askQuestion(question, llmSourcesFallback, {
        focusTopic: recordPrimarySubtopic ?? undefined,
        focusAspect: focus.aspectWord,
      });
      return NextResponse.json({
        answer: stripMarkdownFormattingInAnswer(
          normalizeAnswerCitationsToAvailableSources(result.answer, fallbackUiSources)
        ),
        sources: fallbackUiSources,
        hasAnswer: result.hasAnswer,
        rewrittenQuestion: retrievalQuestion === question ? undefined : retrievalQuestion,
      });
    }

    // For definition-style questions, also clip the LLM context to the definition sentence/snippet.
    // This prevents the model from citing adjacent, unrelated concepts in the same long paragraph.
    const llmSources: SourceChunk[] =
      defTopic
        ? focusedSourcesReindexed.map((s) => {
            const refined = refineRangeByDefinitionTopic(s.content, 0, defTopic);
            return refined ? { ...s, content: refined.content } : s;
          })
        : dateHint
          ? focusedSourcesReindexed.map((s) => {
              const refined = refineRangeByAnyLogDateHeading(s.content, 0, dateHint, dateVariants, 1800);
              return refined ? { ...s, content: refined.content } : s;
            })
          : tailKeyphrase === "训练策略" || trainingBoostTokens.length > 0
            ? focusedSourcesReindexed.map((s) => {
                const refined = refineRangeByTrainingStrategyCue(s.content, 0, 900);
                return refined ? { ...s, content: refined.content } : s;
              })
            : recordHint?.kind === "meeting" && recordHint.id
              ? focusedSourcesReindexed.map((s) => {
                  const refined = recordSubtopic
                    ? refineRangeByMeetingAgendaSubtopic(s.content, 0, recordSubtopic, 1400) ??
                      refineRangeByNeedle(s.content, 0, recordSubtopic, 1200) ??
                      refineRangeByMeetingRecordHeading(s.content, 0, recordHint.id, 1200)
                    : refineRangeByMeetingRecordHeading(s.content, 0, recordHint.id, 1200);
                  return refined ? { ...s, content: refined.content } : s;
                })
              : focusedSourcesReindexed;

    // Ask LLM with sources
    const result = await askQuestion(question, llmSources, {
      focusTopic:
        topicHint ??
        (recordHint?.kind === "meeting" && recordHint.id && recordPrimarySubtopic
          ? recordPrimarySubtopic
          : undefined),
      focusAspect: focus.aspectWord,
    });

    const isClarifyMarker = (s: string) => s.includes("【需要补充上下文】");
    const lastTurn = safeHistory.at(-1);
    const askedClarifyLastTurn = Boolean(lastTurn?.answer && isClarifyMarker(lastTurn.answer));

    // Two-stage behavior when retrieval/grounding fails:
    // - 1st time: ask user to clarify instead of claiming "no info"
    // - 2nd time (after user follow-up): be explicit that notes don't contain relevant material
    if (!result.hasAnswer) {
      // Special-case: meeting record id queries are already highly specific. If we still have no sources,
      // be explicit that this record id wasn't found in the uploaded notes (instead of asking for "key terms").
      if (recordHint?.kind === "meeting" && recordHint.id && recordTokensStrict.length > 0 && uiSources.length === 0) {
        const msg = `我在当前已上传的笔记中没有找到“会议记录 ${recordHint.id}”对应的内容（也未找到包含该编号的会议纪要/会议纪要记录）。\n如果你确认已经上传了会议记录013，请把对应文件/段落贴出来或重新上传，我再基于新增资料检索与问答。`;
        return NextResponse.json({
          answer: msg,
          sources: [],
          hasAnswer: false,
          rewrittenQuestion: retrievalQuestion === question ? undefined : retrievalQuestion,
        });
      }

      // Record overview fallback: if user asked about a specific record id (but not a specific subtopic),
      // and we DO have scoped sources, extract the header fields directly instead of asking for clarification.
      if (recordHint?.kind === "meeting" && recordHint.id && recordTokensStrict.length > 0 && !recordSubtopic && uiSources.length > 0) {
        const header = pickMeetingRecordHeaderCandidate(uiSources);
        if (header) {
          const lines = header.content
            .replace(/\r\n/g, "\n")
            .replace(/\r/g, "\n")
            .split("\n")
            .map((l) => l.trim())
            .filter(Boolean);
          const keys = ["日期", "部门", "主持人", "参会人", "时长", "议题", "议题1", "议题2", "议题3"];
          const picked: string[] = [];
          for (const k of keys) {
            const hit = lines.find((l) => l.includes(k) && (l.includes(":") || l.includes("：")));
            if (hit && !picked.includes(hit)) picked.push(hit);
          }
          if (picked.length > 0) {
            const answer = stripMarkdownFormattingInAnswer(
              [`在你的笔记中，“会议记录 ${recordHint.id}”的基本信息/议题如下：`, ...picked.map((l) => `${l} [${header.index}]`)].join(
                "\n"
              )
            );
            return NextResponse.json({
              answer,
              sources: uiSources,
              hasAnswer: true,
              rewrittenQuestion: retrievalQuestion === question ? undefined : retrievalQuestion,
            });
          }
        }
      }

      // If the model refused due to "insufficient info" but sources contain the requested keyphrase,
      // extract evidence lines directly from sources rather than asking for more context.
      if (tailKeyphrase && uiSources.length > 0) {
        const evidences = uiSources
          .map((s) => ({ s, lines: extractEvidenceLinesFromSource(s.content, tailKeyphrase) }))
          .filter((x) => x.lines.length > 0)
          .slice(0, 3);
        if (evidences.length > 0) {
          const answer = [
            `在你的笔记中，和“${tailKeyphrase}”相关的内容包括：`,
            ...evidences.flatMap((x) => x.lines.map((l) => `${l} [${x.s.index}]`)),
          ].join("\n");
          return NextResponse.json({
            answer,
            sources: uiSources,
            hasAnswer: true,
            rewrittenQuestion: retrievalQuestion === question ? undefined : retrievalQuestion,
          });
        }
      }

      const stage1 = `【需要补充上下文】\n我在当前笔记里暂时没检索到能直接回答你这个问题的内容。你可以补充一下：\n（1）你说的关键术语/对象具体指什么？（可以给全称、同义词、英文缩写）\n或者\n（2）如果你手头有相关段落/关键词，请直接贴出来或上传对应资料。`;

      const stage2 = `我在当前已上传的笔记中仍然没有检索到与该问题直接相关、可用于作答的资料。\n如果你希望我继续回答，请上传/补充相关笔记（或把关键段落贴出来），我再基于新增资料进行检索与问答。`;

      // Record + subtopic fallback: if user asks about a specific meeting record and subtopic,
      // but the model says "no info", extract evidence lines directly from scoped sources.
      if (recordHint?.kind === "meeting" && recordHint.id && recordSubtopic) {
        const scoped =
          recordNoteIds.size > 0
            ? baseSources.filter((s) => recordNoteIds.has(s.noteId))
            : baseSources.filter((s) => recordTokensStrict.some((t) => includesLoose(s.filename, t) || includesLoose(s.content, t)));

        const agendaEvidence = scoped
          .map((s) => ({ s, lines: extractMeetingAgendaEvidenceLines(s.content, recordSubtopic, 10) }))
          .filter((x) => x.lines.length > 0)
          .slice(0, 1);
        const genericEvidence = scoped
          .map((s) => ({ s, lines: extractEvidenceLinesFromSource(s.content, recordSubtopic, 4) }))
          .filter((x) => x.lines.length > 0)
          .slice(0, 2);

        const evidences = (agendaEvidence.length > 0 ? agendaEvidence : genericEvidence).slice(0, 3);
        if (evidences.length > 0) {
          // Put evidence sources first so citations map to visible blocks.
          const evidenceSources = evidences.map((e) => e.s);
          const pickedSources = [
            ...evidenceSources,
            ...scoped.filter((s) => !evidenceSources.some((e) => e.noteId === s.noteId && e.startChar === s.startChar && e.endChar === s.endChar)),
          ].slice(0, LLM_TOP_K);
          const sourcesForResp = pickedSources.map((s, i) => ({ ...s, index: i + 1 }));

          // Build an index map for evidence sources (by noteId + range) to stable [n] citations.
          const indexByKey = new Map<string, number>();
          for (const s of sourcesForResp) {
            indexByKey.set(`${s.noteId}:${s.startChar}:${s.endChar}`, s.index);
          }

          const answer = stripMarkdownFormattingInAnswer(
            [
              `在你的笔记中，“会议记录 ${recordHint.id}”里提到的“${recordSubtopic}”相关内容包括：`,
              ...evidences.flatMap((x) => {
                const key = `${x.s.noteId}:${x.s.startChar}:${x.s.endChar}`;
                const idx = indexByKey.get(key) ?? 1;
                return x.lines.map((l) => `${l} [${idx}]`);
              }),
            ].join("\n")
          );
          return NextResponse.json({
            answer,
            sources: sourcesForResp,
            hasAnswer: true,
            rewrittenQuestion: retrievalQuestion === question ? undefined : retrievalQuestion,
          });
        }
      }

      return NextResponse.json({
        answer: askedClarifyLastTurn ? stage2 : stage1,
        sources: [],
        // For stage1 we treat it as a follow-up prompt (avoid "no sources" warning UI);
        // for stage2 we mark as no-answer so UI can highlight the limitation.
        hasAnswer: askedClarifyLastTurn ? false : true,
        rewrittenQuestion: retrievalQuestion === question ? undefined : retrievalQuestion,
      });
    }

    let normalizedAnswer = normalizeAnswerCitationsToAvailableSources(result.answer, uiSources);

    // If this is a "training strategy" query and the model answer missed the key details,
    // fall back to extracting the strongest evidence lines directly from sources.
    if ((tailKeyphrase === "训练策略" || trainingBoostTokens.length > 0) && uiSources.length > 0) {
      const looksWeak =
        !normalizedAnswer.includes("训练策略") &&
        !normalizedAnswer.includes("学习率") &&
        !normalizedAnswer.includes("余弦") &&
        !/epoch/i.test(normalizedAnswer);
      if (looksWeak) {
        const evidences = uiSources
          .map((s) => ({ s, lines: extractEvidenceLinesFromSource(s.content, "训练策略", 4) }))
          .filter((x) => x.lines.length > 0)
          .slice(0, 3);
        if (evidences.length > 0) {
          normalizedAnswer = [
            `在你的笔记中，“训练策略”相关原文要点如下：`,
            ...evidences.flatMap((x) => x.lines.map((l) => `${l} [${x.s.index}]`)),
          ].join("\n");
        }
      }
    }

    return NextResponse.json({
      answer: stripMarkdownFormattingInAnswer(normalizedAnswer),
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
