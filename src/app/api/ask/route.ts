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

type QuestionFocus = {
  topic?: string;
  aspect?: "advantages" | "disadvantages";
  aspectWord?: string; // original (e.g. "优点"/"缺点")
};

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

  const start = Math.max(0, idx - 250);
  const end = Math.min(text.length, idx + 900);
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

    // Embed the question
    const normalizedQuestion = normalizeForEmbedding(retrievalQuestion);
    const queryEmbedding = await getEmbedding(normalizedQuestion);
    const vectorStr = `[${queryEmbedding.join(",")}]`;
    const rawQ = retrievalQuestion.trim();
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

    const patterns = buildKeywordPatterns(retrievalQuestion);
    if (patterns.length > 0) {
      const focusLikes =
        focus.topic && focus.aspectWord ? [`%${focus.topic}%`, `%${focus.aspectWord}%`] : focus.topic ? [`%${focus.topic}%`] : [];
      const all = [`%${rawQ}%`, `%${qNoPunct}%`, ...focusLikes, ...patterns];
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
      if (focus.topic && c.includes(focus.topic)) s += 6;
      if (focus.aspectWord && c.includes(focus.aspectWord)) s += 3;
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
      startChar: typeof row.start_char === "number" ? row.start_char : parseInt(row.start_char, 10),
      endChar: typeof row.end_char === "number" ? row.end_char : parseInt(row.end_char, 10),
      similarity: row.similarity,
    }));

    const focusedSources =
      focus.topic
        ? sources
            .filter((s) => s.content.includes(focus.topic!))
            .map((s) => ({
              ...s,
              content: clipToFocusedSection(s.content, focus),
            }))
            .slice(0, LLM_TOP_K)
        : sources.slice(0, LLM_TOP_K);

    // Ask LLM with sources
    const result = await askQuestion(question, focusedSources, {
      focusTopic: focus.topic,
      focusAspect: focus.aspectWord,
    });

    return NextResponse.json({
      answer: result.answer,
      sources: result.sources,
      hasAnswer: result.hasAnswer,
      rewrittenQuestion: retrievalQuestion === question ? undefined : retrievalQuestion,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Internal server error";
    console.error("Ask error:", err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
