export interface Chunk {
  content: string;
  startChar: number;
  endChar: number;
}

const MAX_CHUNK_TOKENS = 500;
const MIN_CHUNK_TOKENS = 50;
const OVERLAP_TOKENS = 50;

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3.5);
}

function splitByParagraphs(text: string): string[] {
  return text.split(/\n\s*\n/).filter((p) => p.trim().length > 0);
}

function splitBySentences(text: string): string[] {
  const sentences = text.match(/[^.!?。！？]+[.!?。！？]+\s*/g) || [text];
  return sentences.filter((s) => s.trim().length > 0);
}

export function chunkText(text: string): Chunk[] {
  const chunks: Chunk[] = [];
  const paragraphs = splitByParagraphs(text);

  let currentContent = "";
  let currentStart = 0;
  let pos = 0;

  for (const para of paragraphs) {
    const paraStart = text.indexOf(para, pos);
    const paraTokenEstimate = estimateTokens(currentContent + para);

    if (paraTokenEstimate > MAX_CHUNK_TOKENS && currentContent.length > 0) {
      // Flush current chunk
      chunks.push({
        content: currentContent.trim(),
        startChar: currentStart,
        endChar: currentStart + currentContent.length,
      });

      // Start new chunk with overlap
      const overlapText = getOverlapTail(currentContent);
      currentContent = overlapText + para;
      currentStart = paraStart - overlapText.length;
    } else if (paraTokenEstimate > MAX_CHUNK_TOKENS && currentContent.length === 0) {
      // Single paragraph too large, split by sentences
      const sentences = splitBySentences(para);
      let sentContent = "";

      for (const sent of sentences) {
        if (estimateTokens(sentContent + sent) > MAX_CHUNK_TOKENS && sentContent.length > 0) {
          chunks.push({
            content: sentContent.trim(),
            startChar: currentStart || paraStart,
            endChar: (currentStart || paraStart) + sentContent.length,
          });
          const overlapText = getOverlapTail(sentContent);
          sentContent = overlapText + sent;
          currentStart = paraStart + para.indexOf(sent) - overlapText.length;
        } else {
          if (sentContent.length === 0) currentStart = paraStart;
          sentContent += sent;
        }
      }

      if (sentContent.trim().length > 0) {
        currentContent = sentContent;
      }
      pos = paraStart + para.length;
      continue;
    } else {
      if (currentContent.length === 0) currentStart = paraStart;
      currentContent += (currentContent ? "\n\n" : "") + para;
    }

    pos = paraStart + para.length;
  }

  // Flush remaining
  if (currentContent.trim().length > 0) {
    // Merge with previous if too short
    const lastChunk = chunks[chunks.length - 1];
    if (
      lastChunk &&
      estimateTokens(currentContent) < MIN_CHUNK_TOKENS &&
      estimateTokens(lastChunk.content + "\n\n" + currentContent) <= MAX_CHUNK_TOKENS * 1.5
    ) {
      chunks[chunks.length - 1] = {
        content: (lastChunk.content + "\n\n" + currentContent).trim(),
        startChar: lastChunk.startChar,
        endChar: currentStart + currentContent.length,
      };
    } else {
      chunks.push({
        content: currentContent.trim(),
        startChar: currentStart,
        endChar: currentStart + currentContent.length,
      });
    }
  }

  // Re-index positions based on actual content positions
  return chunks.map((chunk) => {
    // The initial startChar is derived from paragraph scanning, which should already be close.
    // When the document contains repeated headings/phrases (common in docx exports),
    // a global-ish indexOf can jump to a different repeated section and break source highlighting.
    // Therefore, only search within a small local window around the expected start.
    const needle = chunk.content.slice(0, 50);
    const expected = Math.max(0, chunk.startChar);
    const windowStart = Math.max(0, expected - 200);
    const windowEnd = Math.min(text.length, expected + 2000);
    const localIdx = text.slice(windowStart, windowEnd).indexOf(needle);
    const actualStart = localIdx >= 0 ? windowStart + localIdx : -1;
    return {
      ...chunk,
      startChar: actualStart >= 0 ? actualStart : chunk.startChar,
      endChar: actualStart >= 0 ? actualStart + chunk.content.length : chunk.endChar,
    };
  });
}

function getOverlapTail(text: string): string {
  const targetChars = OVERLAP_TOKENS * 3.5;
  if (text.length <= targetChars) return "";
  const tail = text.slice(-Math.floor(targetChars));
  const firstSpace = tail.indexOf(" ");
  return firstSpace >= 0 ? tail.slice(firstSpace + 1) : tail;
}
