const API_KEY = process.env.ARK_API_KEY!;
const BASE_URL = process.env.ARK_BASE_URL || "https://ark.cn-beijing.volces.com/api/v3";
const EMBEDDING_MODEL = process.env.ARK_EMBEDDING_MODEL || "doubao-embedding-vision-251215";
const EMBEDDING_DIM = parseInt(process.env.ARK_EMBEDDING_DIMENSION || "2048");
const EMBEDDING_TIMEOUT_MS = parseInt(process.env.ARK_EMBEDDING_TIMEOUT_MS || "8000");

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number) {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: ac.signal });
  } finally {
    clearTimeout(t);
  }
}

async function callMultimodalEmbedding(text: string): Promise<number[]> {
  const url = `${BASE_URL}/embeddings/multimodal`;
  const body = JSON.stringify({
    model: EMBEDDING_MODEL,
    input: [{ type: "text", text }],
    dimensions: EMBEDDING_DIM,
  });

  let lastErr: unknown = null;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await fetchWithTimeout(
        url,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${API_KEY}`,
          },
          body,
        },
        EMBEDDING_TIMEOUT_MS
      );
      if (!res.ok) {
        const err = await res.text();
        throw new Error(`Embedding API error ${res.status}: ${err}`);
      }
      const json = await res.json();
      return json.data.embedding;
    } catch (err) {
      lastErr = err;
      // small backoff for transient network resets
      await new Promise((r) => setTimeout(r, 150 * (attempt + 1)));
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("Embedding request failed");
}

export async function getEmbedding(text: string): Promise<number[]> {
  return callMultimodalEmbedding(text);
}

export async function getEmbeddingsBatch(texts: string[]): Promise<number[][]> {
  const concurrency = 8;
  const results: number[][] = [];
  for (let i = 0; i < texts.length; i += concurrency) {
    const batch = texts.slice(i, i + concurrency);
    const embeddings = await Promise.all(batch.map((t) => callMultimodalEmbedding(t)));
    results.push(...embeddings);
  }
  return results;
}

export { EMBEDDING_DIM };
