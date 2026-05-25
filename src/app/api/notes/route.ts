import { NextResponse } from "next/server";
import { query } from "@/lib/db";

export async function GET() {
  try {
    const notesResult = await query(`
      SELECT n.id, n.filename, n.file_type, n.created_at, COUNT(c.id) as chunk_count
      FROM notes n
      LEFT JOIN chunks c ON c.note_id = n.id
      GROUP BY n.id, n.filename, n.file_type, n.created_at
      ORDER BY n.created_at DESC
    `);

    return NextResponse.json({ notes: notesResult.rows });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
