export function normalizeForEmbedding(input: string): string {
  // Remove common markdown / formatting noise that hurts embedding similarity.
  // Keep the textual content and code blocks (code can be important for technical notes).
  let s = input;

  // Normalize newlines
  s = s.replace(/\r\n/g, "\n");

  // Remove frontmatter blocks
  s = s.replace(/^---\n[\s\S]*?\n---\n/gm, "");

  // Remove HTML tags
  s = s.replace(/<\/?[^>]+>/g, " ");

  // Strip markdown images/links but keep visible text
  s = s.replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1");
  s = s.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");

  // Strip headings/bullets markers while keeping text
  s = s.replace(/^\s{0,3}(#{1,6})\s+/gm, "");
  s = s.replace(/^\s{0,3}>\s?/gm, "");
  s = s.replace(/^\s{0,3}[-*+]\s+/gm, "");
  s = s.replace(/^\s{0,3}\d+\.\s+/gm, "");

  // Emphasis markers
  s = s.replace(/\*\*([^*]+)\*\*/g, "$1");
  s = s.replace(/\*([^*]+)\*/g, "$1");
  s = s.replace(/__([^_]+)__/g, "$1");
  s = s.replace(/_([^_]+)_/g, "$1");
  s = s.replace(/`([^`]+)`/g, "$1");

  // Collapse whitespace
  s = s.replace(/[ \t]+/g, " ");
  s = s.replace(/\n{3,}/g, "\n\n");

  return s.trim();
}

