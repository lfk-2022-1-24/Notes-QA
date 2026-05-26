import OpenAI from "openai";

const deepseekClient = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: process.env.DEEPSEEK_BASE_URL,
});

const CHAT_MODEL = process.env.DEEPSEEK_CHAT_MODEL || "deepseek-v4-pro";
const LLM_TIMEOUT_MS = parseInt(process.env.DEEPSEEK_TIMEOUT_MS || "20000");

// Internal sentinel for "no relevant info" detection.
// We never want to show this raw string to end users; the API layer should translate it.
const NO_RELEVANT_INFO_SENTINEL = "__NO_RELEVANT_INFO__";

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

export interface HistoryTurn {
  question: string;
  answer: string;
}

export interface AskOptions {
  focusTopic?: string;
  focusAspect?: string;
}

const SYSTEM_PROMPT = `You are a helpful assistant that answers questions based ONLY on the provided source notes.

Rules:
1. Answer the question using ONLY information from the provided sources.
2. For EVERY claim in your answer, cite the source number in square brackets, like [1] or [2].
3. If multiple sources support a claim, cite all of them: [1][3].
4. If the sources do not contain enough information to answer the question, output EXACTLY: "__NO_RELEVANT_INFO__"
5. Do NOT make up information that is not in the sources.
6. Do NOT use external knowledge to supplement the answer.
7. If the sources contain suggested phrasing, example replies, or a recommended way to respond, include that wording.
8. Respond in the same language as the user's question (Chinese question -> Simplified Chinese answer; English question -> English answer).
9. For definition-style questions ("X是什么/什么是X"), if the sources don't provide a single-sentence definition but do provide what X includes/consists of, answer in the form "在你的笔记中，X主要包括：..." and cite sources.
10. Output as plain text only. Do NOT use markdown formatting such as bullets, *, **, headers, or code fences.`;

const REWRITE_SYSTEM_PROMPT = `You rewrite a user's question into a standalone question using chat history.

Rules:
1. Output ONLY the rewritten standalone question, nothing else.
2. Resolve references like "it", "they", "this", "that", "he/she", "其/它/他/她/他们/这个/那个/上述/该/此" to the most likely subject from the history.
3. Keep the user's language (Chinese stays Chinese, English stays English).
4. If the question is already standalone, output it unchanged.
5. Do NOT add extra questions or commentary.`;

function maybeResolveWithHeuristic(question: string, history: HistoryTurn[]): string {
  const q = question.trim();
  const last = history.at(-1)?.question?.trim();
  if (!last) return q;

  const hasPronoun =
    /(^|\s)(it|they|them|this|that|these|those)\b/i.test(q) ||
    /(它|他|她|他们|其|这个|那个|上述|前面|上一个|该|此|其优点|其缺点|它的|他的|她的)/.test(q);

  if (!hasPronoun) return q;

  // Extract a rough "topic" from the last question by removing common suffixes.
  const topic = last
    .replace(/[？?]\s*$/g, "")
    .replace(/(有哪些|有什么|是什么|分别是什么|主要是|如何|怎么)(.+)?$/g, "")
    .replace(/(的)?(优点|缺点|好处|坏处|作用|意义|风险|问题)\s*$/g, "")
    .trim();

  if (!topic) return q;
  // Example: "它的缺点是什么？" -> "垂直拆分的缺点是什么？"
  return q
    .replace(/^(它|他|她|其|这个|那个|上述|该|此)/, topic)
    .replace(/(它|他|她|其)(的)/g, `${topic}$2`);
}

export async function rewriteStandaloneQuestion(
  question: string,
  history: HistoryTurn[]
): Promise<string> {
  const trimmed = question.trim();
  if (!trimmed) return trimmed;
  if (!history || history.length === 0) return trimmed;

  // Keep the history short to control cost.
  const recent = history.slice(-3);
  const historyText = recent
    .map((t, i) => `Turn ${i + 1}:\nUser: ${t.question}\nAssistant: ${t.answer}`)
    .join("\n\n");

  try {
    const response = await deepseekClient.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: REWRITE_SYSTEM_PROMPT },
        {
          role: "user",
          content: `Chat history:\n${historyText}\n\nUser question:\n${trimmed}\n\nRewrite into a standalone question:`,
        },
      ],
      temperature: 0.0,
      max_tokens: 128,
    });

    const out = (response.choices[0]?.message?.content || "").trim();
    // Basic sanity checks: keep it short and single-line-ish.
    if (!out || out.length > 300) return maybeResolveWithHeuristic(trimmed, recent);
    return out.replace(/\s+/g, " ").trim();
  } catch {
    return maybeResolveWithHeuristic(trimmed, recent);
  }
}

export async function askQuestion(
  question: string,
  sources: SourceChunk[],
  options?: AskOptions
): Promise<AskResult> {
  if (sources.length === 0) {
    return {
      answer: NO_RELEVANT_INFO_SENTINEL,
      sources: [],
      hasAnswer: false,
    };
  }

  const sourceText = sources
    .map((s) => `[Source ${s.index}] (from "${s.filename}"):\n${s.content}`)
    .join("\n\n---\n\n");

  const focusLine =
    options?.focusTopic
      ? `Focus: Answer ONLY about "${options.focusTopic}"${
          options.focusAspect ? ` (${options.focusAspect})` : ""
        }. Do not discuss other topics unless the question explicitly asks.\n\n`
      : "";

  const userMessage = `${focusLine}Here are the source notes:\n\n${sourceText}\n\n---\n\nQuestion: ${question}`;

  const response = await deepseekClient.chat.completions.create(
    {
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userMessage },
      ],
      temperature: 0.1,
      max_tokens: 2048,
    },
    { timeout: LLM_TIMEOUT_MS }
  );

  const answer = response.choices[0]?.message?.content || "";

  // Check if the answer indicates no information found
  const normalized = answer.trim();
  const hasAnswer =
    normalized !== NO_RELEVANT_INFO_SENTINEL &&
    !normalized.includes("could not find relevant information") &&
    !normalized.includes("__NO_RELEVANT_INFO__");

  return { answer, sources, hasAnswer };
}
