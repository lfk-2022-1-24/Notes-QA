import { NextRequest, NextResponse } from "next/server";
import { getEmbedding } from "@/lib/embedding";
import { askQuestion, SourceChunk } from "@/lib/llm";
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

export async function POST(req: NextRequest) {
  try {
    const { question } = await req.json();

    if (!question || typeof question !== "string" || !question.trim()) {
      return NextResponse.json({ error: "Question is required" }, { status: 400 });
    }

    // Embed the question
    const normalizedQuestion = normalizeForEmbedding(question);
    const queryEmbedding = await getEmbedding(normalizedQuestion);
    const vectorStr = `[${queryEmbedding.join(",")}]`;
    const rawQ = question.trim();
    const qNoPunct = stripPunctuation(rawQ);

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

    const vectorRows: SearchRow[] = searchResult.rows.map((r) => ({
      ...(r as Omit<SearchRow, "similarity">),
      similarity: parseFloat(r.similarity),
    }));

    // Filter vector results with an adaptive threshold.
    const topSim = vectorRows[0]?.similarity ?? 0;
    const threshold = Math.max(BASE_THRESHOLD, topSim - GAP_THRESHOLD);
    let vectorRelevant = vectorRows.filter((row) => row.similarity >= threshold);
    if (vectorRelevant.length < MIN_KEEP) vectorRelevant = vectorRows.slice(0, MIN_KEEP);

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
    const exactResult = await query(
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
    );

    const exactRows: SearchRow[] = exactResult.rows.map((r) => ({
      ...(r as Omit<SearchRow, "similarity">),
      similarity: parseFloat(r.similarity),
    }));

    const patterns = buildKeywordPatterns(question);
    if (patterns.length > 0) {
      const all = [`%${rawQ}%`, `%${qNoPunct}%`, ...patterns];
      const where = all.map((_, i) => `c.content ILIKE $${i + 3}`).join(" OR ");

      // Exact-hit score + keyword-hit count; then fall back to vector distance.
      const exactHit =
        `((CASE WHEN c.content ILIKE $3 THEN 1 ELSE 0 END) + (CASE WHEN c.content ILIKE $4 THEN 1 ELSE 0 END))`;
      const kwHits = all
        .map((_, i) => `CASE WHEN c.content ILIKE $${i + 3} THEN 1 ELSE 0 END`)
        .join(" + ");

      const keywordResult = await query(
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
      if (qNoPunct && c.includes(qNoPunct)) s += 2;
      if (rawQ && c.includes(rawQ)) s += 2;
      if (qNoPunct && c.includes(`Q4: ${qNoPunct}`)) s += 6;
      if (qNoPunct && c.includes(`Q4:${qNoPunct}`)) s += 6;
      return s;
    };

    const relevantChunks = Array.from(merged.values())
      .sort((a, b) => scoreExact(b.content) - scoreExact(a.content) || b.similarity - a.similarity)
      .slice(0, TOP_K);

    // Build source objects
    const sources: SourceChunk[] = relevantChunks.map((row, index) => ({
      index: index + 1,
      noteId: row.note_id,
      filename: row.filename,
      content: row.content,
      startChar: parseInt(row.start_char),
      endChar: parseInt(row.end_char),
      similarity: row.similarity,
    }));

    // Ask LLM with sources
    const result = await askQuestion(question, sources.slice(0, LLM_TOP_K));

    return NextResponse.json({
      answer: result.answer,
      sources: result.sources,
      hasAnswer: result.hasAnswer,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Internal server error";
    console.error("Ask error:", err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
