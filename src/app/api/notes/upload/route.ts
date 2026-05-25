import { NextRequest, NextResponse } from "next/server";
import { parseFile } from "@/lib/parser";
import { chunkText } from "@/lib/chunking";
import { getEmbeddingsBatch } from "@/lib/embedding";
import { query } from "@/lib/db";
import { v4 as uuidv4 } from "uuid";
import { normalizeForEmbedding } from "@/lib/text-normalize";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const files = formData.getAll("files") as File[];

    if (!files || files.length === 0) {
      return NextResponse.json({ error: "No files provided" }, { status: 400 });
    }

    const results = [];

    for (const file of files) {
      try {
        const buffer = Buffer.from(await file.arrayBuffer());
        const { text, fileType } = await parseFile(buffer, file.name);

        if (!text.trim()) {
          results.push({ filename: file.name, error: "File appears to be empty or could not be parsed" });
          continue;
        }

        const noteId = uuidv4();

        // Insert note
        await query(
          `INSERT INTO notes (id, filename, file_type, content) VALUES ($1, $2, $3, $4)`,
          [noteId, file.name, fileType, text]
        );

        // Chunk the ORIGINAL text so start/end offsets match what we display.
        // Only normalize when generating embeddings (below).
        const chunks = chunkText(text);

        if (chunks.length === 0) {
          results.push({ filename: file.name, error: "Could not extract meaningful chunks" });
          continue;
        }

        // Generate embeddings in batches
        const chunkTextsForEmbedding = chunks.map((c) => {
          const cleaned = normalizeForEmbedding(c.content);
          // Add filename context to help retrieval across many notes.
          const body = cleaned.trim() ? cleaned : c.content;
          return `${file.name}\n\n${body}`;
        });
        const embeddings = await getEmbeddingsBatch(chunkTextsForEmbedding);

        // Insert chunks with embeddings
        for (let i = 0; i < chunks.length; i++) {
          const chunk = chunks[i];
          const embedding = embeddings[i];
          const chunkId = uuidv4();
          const vectorStr = `[${embedding.join(",")}]`;

          await query(
            `INSERT INTO chunks (id, note_id, chunk_index, content, start_char, end_char, embedding)
             VALUES ($1, $2, $3, $4, $5, $6, $7::vector)`,
            [chunkId, noteId, i, chunk.content, chunk.startChar, chunk.endChar, vectorStr]
          );
        }

        results.push({
          id: noteId,
          filename: file.name,
          fileType,
          chunkCount: chunks.length,
          charCount: text.length,
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown error";
        results.push({ filename: file.name, error: message });
      }
    }

    return NextResponse.json({ results });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
