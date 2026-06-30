import { streamText, convertToModelMessages, type UIMessage } from 'ai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

// Allow long cloud-model streams.
export const maxDuration = 300;

const OLLAMA = process.env.OLLAMA_HOST || 'http://localhost:11434';

// Ollama exposes an OpenAI-compatible endpoint at /v1 — the AI SDK talks to it
// directly. The same provider serves local AND ':cloud' models (the daemon
// proxies cloud models to ollama.com).
const ollama = createOpenAICompatible({
  name: 'ollama',
  baseURL: `${OLLAMA}/v1`,
  apiKey: 'ollama',
  includeUsage: true, // emit stream_options.include_usage → token counts in the stream
});

type Settings = {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  num_ctx?: number;
  repeat_penalty?: number;
  max_tokens?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  seed?: number;
};

export async function POST(req: Request) {
  const body = (await req.json()) as {
    messages: UIMessage[];
    model?: string;
    system?: string;
    settings?: Settings;
  };

  const s = body.settings || {};
  const num = (v: unknown) =>
    v === undefined || v === null || v === '' || Number.isNaN(Number(v))
      ? undefined
      : Number(v);

  // Native Ollama options (top_k / num_ctx / repeat_penalty) are best-effort:
  // recent Ollama honors them through the OpenAI-compatible body; older builds
  // ignore unknown fields harmlessly.
  // Best-effort native Ollama options.
  const ollamaExtra: Record<string, number> = {};
  if (num(s.top_k) !== undefined) ollamaExtra.top_k = num(s.top_k)!;
  if (num(s.num_ctx) !== undefined) ollamaExtra.num_ctx = num(s.num_ctx)!;
  if (num(s.repeat_penalty) !== undefined)
    ollamaExtra.repeat_penalty = num(s.repeat_penalty)!;

  const modelMessages = await convertToModelMessages(body.messages);

  const result = streamText({
    model: ollama(body.model || 'gpt-oss:120b-cloud'),
    system: body.system || undefined,
    messages: modelMessages,
    temperature: num(s.temperature),
    topP: num(s.top_p),
    maxOutputTokens: num(s.max_tokens),
    frequencyPenalty: num(s.frequency_penalty),
    presencePenalty: num(s.presence_penalty),
    seed: num(s.seed),
    providerOptions:
      Object.keys(ollamaExtra).length > 0 ? { ollama: ollamaExtra } : undefined,
  });

  // Forward real provider errors to the client (otherwise the AI SDK masks them
  // as "An error occurred."). This surfaces e.g. Ollama cloud 403 subscription
  // notices or "model not found" so the participant sees the actual reason.
  return result.toUIMessageStreamResponse({
    // Attach real token usage to the assistant message so the UI can show a
    // token counter (Ollama reports prompt/completion token counts).
    messageMetadata: ({ part }) => {
      if (part.type === 'finish') {
        const u = part.totalUsage;
        const input = u?.inputTokens ?? null;
        const output = u?.outputTokens ?? null;
        const total =
          (u?.totalTokens ?? ((input ?? 0) + (output ?? 0))) || null;
        return { inputTokens: input, outputTokens: output, totalTokens: total };
      }
    },
    onError: (error) => {
      const msg = error instanceof Error ? error.message : String(error);
      const m = /403[^"]*?:?\s*([^"]*requires a subscription[^"]*)/i.exec(msg);
      if (m) return 'Cloud model needs an Ollama subscription — ' + m[1].trim();
      if (/not found|no such model/i.test(msg)) return 'Model not found. Pull it first (Models tab).';
      return msg.slice(0, 300);
    },
  });
}
