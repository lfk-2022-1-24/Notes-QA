import matter from "gray-matter";

// eslint-disable-next-line @typescript-eslint/no-require-imports
const pdfParse = require("pdf-parse");

export interface ParseResult {
  text: string;
  fileType: string;
}

export async function parseFile(
  buffer: Buffer,
  filename: string
): Promise<ParseResult> {
  const ext = filename.split(".").pop()?.toLowerCase() || "";

  switch (ext) {
    case "md":
    case "markdown":
      return parseMarkdown(buffer);
    case "txt":
      return parsePlainText(buffer);
    case "pdf":
      return parsePdf(buffer);
    default:
      throw new Error(`Unsupported file format: .${ext}. Supported: .md, .txt, .pdf`);
  }
}

function parseMarkdown(buffer: Buffer): ParseResult {
  const raw = buffer.toString("utf-8");
  const { content } = matter(raw);
  return { text: content, fileType: "md" };
}

function parsePlainText(buffer: Buffer): ParseResult {
  const text = buffer.toString("utf-8");
  return { text, fileType: "txt" };
}

async function parsePdf(buffer: Buffer): Promise<ParseResult> {
  const data = await pdfParse(buffer);
  return { text: data.text, fileType: "pdf" };
}
