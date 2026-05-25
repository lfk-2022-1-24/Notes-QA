import matter from "gray-matter";
import iconv from "iconv-lite";

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
  const raw = decodeTextBuffer(buffer);
  const { content } = matter(raw);
  return { text: content, fileType: "md" };
}

function parsePlainText(buffer: Buffer): ParseResult {
  const text = decodeTextBuffer(buffer);
  return { text, fileType: "txt" };
}

async function parsePdf(buffer: Buffer): Promise<ParseResult> {
  const data = await pdfParse(buffer);
  return { text: data.text, fileType: "pdf" };
}

function decodeTextBuffer(buffer: Buffer): string {
  // Handle BOM explicitly
  if (buffer.length >= 3 && buffer[0] === 0xef && buffer[1] === 0xbb && buffer[2] === 0xbf) {
    return buffer.slice(3).toString("utf8");
  }
  if (buffer.length >= 2 && buffer[0] === 0xff && buffer[1] === 0xfe) {
    return buffer.slice(2).toString("utf16le");
  }
  if (buffer.length >= 2 && buffer[0] === 0xfe && buffer[1] === 0xff) {
    // Node doesn't support utf16be decode directly; use iconv.
    return iconv.decode(buffer.slice(2), "utf16-be");
  }

  const utf8 = buffer.toString("utf8");
  // If the text contains too many replacement chars, it's likely not UTF-8.
  const replacementCount = (utf8.match(/\uFFFD/g) || []).length;
  const replacementRatio = utf8.length > 0 ? replacementCount / utf8.length : 0;
  if (replacementCount >= 3 && replacementRatio > 0.002) {
    // GB18030 is a superset that covers GBK/GB2312 and is common on Windows.
    return iconv.decode(buffer, "gb18030");
  }
  return utf8;
}
