import { NextResponse } from "next/server";
import { query } from "@/lib/db";

export async function GET() {
  try {
    const result = await query(`SELECT COUNT(*) as note_count FROM notes`);
    const chunkResult = await query(`SELECT COUNT(*) as chunk_count FROM chunks`);

    return NextResponse.json({
      status: "ok",
      notes: parseInt(result.rows[0].note_count),
      chunks: parseInt(chunkResult.rows[0].chunk_count),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Database connection failed";
    return NextResponse.json({ status: "error", error: message }, { status: 503 });
  }
}
