import { NextRequest, NextResponse } from "next/server";
import { getEmbedding } from "@/lib/embedding";
import { askQuestion, SourceChunk } from "@/lib/llm";
import { query } from "@/lib/db";

const SIMILARITY_THRESHOLD = 0.3;
const TOP_K = 8;

export async function POST(req: NextRequest) {
  try {
    const { question } = await req.json();

    if (!question || typeof question !== "string" || !question.trim()) {
      return NextResponse.json({ error: "Question is required" }, { status: 400 });
    }

    // Embed the question
    const queryEmbedding = await getEmbedding(question);
    const vectorStr = `[${queryEmbedding.join(",")}]`;

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

    // Filter by similarity threshold
    const relevantChunks = searchResult.rows.filter(
      (row) => parseFloat(row.similarity) >= SIMILARITY_THRESHOLD
    );

    // Build source objects
    const sources: SourceChunk[] = relevantChunks.map((row, index) => ({
      index: index + 1,
      noteId: row.note_id,
      filename: row.filename,
      content: row.content,
      startChar: parseInt(row.start_char),
      endChar: parseInt(row.end_char),
      similarity: parseFloat(row.similarity),
    }));

    // Ask LLM with sources
    const result = await askQuestion(question, sources);

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
