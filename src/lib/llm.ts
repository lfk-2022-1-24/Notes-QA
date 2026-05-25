import OpenAI from "openai";

const deepseekClient = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: process.env.DEEPSEEK_BASE_URL,
});

const CHAT_MODEL = process.env.DEEPSEEK_CHAT_MODEL || "deepseek-v4-pro";

export interface SourceChunk {
  index: number;
  noteId: string;
  filename: string;
  content: string;
  startChar: number;
  endChar: number;
  similarity: number;
}

export interface AskResult {
  answer: string;
  sources: SourceChunk[];
  hasAnswer: boolean;
}

const SYSTEM_PROMPT = `You are a helpful assistant that answers questions based ONLY on the provided source notes.

Rules:
1. Answer the question using ONLY information from the provided sources.
2. For EVERY claim in your answer, cite the source number in square brackets, like [1] or [2].
3. If multiple sources support a claim, cite all of them: [1][3].
4. If the sources do not contain enough information to answer the question, say: "I could not find relevant information in your notes to answer this question."
5. Do NOT make up information that is not in the sources.
6. Do NOT use external knowledge to supplement the answer.
7. Output as plain text only. Do NOT use markdown formatting such as bullets, *, **, headers, or code fences.`;

export async function askQuestion(
  question: string,
  sources: SourceChunk[]
): Promise<AskResult> {
  if (sources.length === 0) {
    return {
      answer: "I could not find relevant information in your notes to answer this question.",
      sources: [],
      hasAnswer: false,
    };
  }

  const sourceText = sources
    .map((s) => `[Source ${s.index}] (from "${s.filename}"):\n${s.content}`)
    .join("\n\n---\n\n");

  const userMessage = `Here are the source notes:\n\n${sourceText}\n\n---\n\nQuestion: ${question}`;

  const response = await deepseekClient.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userMessage },
    ],
    temperature: 0.1,
    max_tokens: 2048,
  });

  const answer = response.choices[0]?.message?.content || "";

  // Check if the answer indicates no information found
  const hasAnswer = !answer.includes("could not find relevant information");

  return { answer, sources, hasAnswer };
}
