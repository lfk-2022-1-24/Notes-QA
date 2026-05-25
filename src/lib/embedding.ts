const API_KEY = process.env.ARK_API_KEY!;
const BASE_URL = process.env.ARK_BASE_URL || "https://ark.cn-beijing.volces.com/api/v3";
const EMBEDDING_MODEL = process.env.ARK_EMBEDDING_MODEL || "doubao-embedding-vision-251215";
const EMBEDDING_DIM = parseInt(process.env.ARK_EMBEDDING_DIMENSION || "2048");

async function callMultimodalEmbedding(text: string): Promise<number[]> {
  const res = await fetch(`${BASE_URL}/embeddings/multimodal`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({
      model: EMBEDDING_MODEL,
      input: [{ type: "text", text }],
      dimensions: EMBEDDING_DIM,
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Embedding API error ${res.status}: ${err}`);
  }
  const json = await res.json();
  return json.data.embedding;
}

export async function getEmbedding(text: string): Promise<number[]> {
  return callMultimodalEmbedding(text);
}

export async function getEmbeddingsBatch(texts: string[]): Promise<number[][]> {
  const concurrency = 20;
  const results: number[][] = [];
  for (let i = 0; i < texts.length; i += concurrency) {
    const batch = texts.slice(i, i + concurrency);
    const embeddings = await Promise.all(batch.map((t) => callMultimodalEmbedding(t)));
    results.push(...embeddings);
  }
  return results;
}

export { EMBEDDING_DIM };
