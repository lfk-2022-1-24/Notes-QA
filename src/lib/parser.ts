import matter from "gray-matter";
import iconv from "iconv-lite";
import mammoth from "mammoth";

type PDFParseInstance = {
  getText: () => Promise<{ text: string }>;
  destroy: () => Promise<void>;
};

type PDFParseCtor = new (options: { data: Buffer | Uint8Array }) => PDFParseInstance;

let cachedPDFParseClass: PDFParseCtor | null = null;

async function getPDFParseClass() {
  if (cachedPDFParseClass) return cachedPDFParseClass;

  // pdf-parse@2.x exports a PDFParse class (not a function).
  // Use dynamic import to work in Next.js bundler/module environments.
  const mod: unknown = await import("pdf-parse");
  const cls =
    (mod as { PDFParse?: unknown }).PDFParse ??
    (mod as { default?: { PDFParse?: unknown } }).default?.PDFParse;
  if (typeof cls !== "function") {
    throw new Error("pdf-parse PDFParse export is not a constructor");
  }
  cachedPDFParseClass = cls as unknown as PDFParseCtor;
  return cachedPDFParseClass;
}

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
    case "docx":
      return parseDocx(buffer);
    default:
      throw new Error(`Unsupported file format: .${ext}. Supported: .md, .txt, .pdf, .docx`);
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
  const PDFParse = await getPDFParseClass();
  const parser = new PDFParse({ data: buffer });
  try {
    const data = await parser.getText();
    return { text: data.text, fileType: "pdf" };
  } finally {
    await parser.destroy();
  }
}

async function parseDocx(buffer: Buffer): Promise<ParseResult> {
  const result = await mammoth.extractRawText({ buffer });
  const text = (result.value || "").replace(/\r\n/g, "\n");
  return { text, fileType: "docx" };
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
