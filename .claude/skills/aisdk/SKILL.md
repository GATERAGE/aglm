---
name: aisdk
description: Build and debug the AGLM participant console (aglm-console/) on the Vercel AI SDK v7 — streaming chat with useChat, the streamText route handler, and the OpenAI-compatible Ollama provider (local + :cloud models). Use when editing aglm-console, adding chat/model features, wiring providers, or hitting AI SDK errors. Triggers include "AI SDK", "useChat", "streamText", "ai-sdk", "ollama provider", "openai-compatible", "convertToModelMessages", "aglm-console", "stream not working", "reasoning model".
---

# AI SDK v7 in the AGLM console

The AGLM participant UI lives in `aglm-console/` — a Next.js 15 (App Router) app
on the **Vercel AI SDK v7**. AGLM is not a model; the AI SDK streams from a model
(Ollama) while the **Flask brain** on `:5000` supplies the logic-augmentation.

## Architecture

```
Browser (useChat)  ──POST /api/chat──►  Next route (streamText)  ──/v1──►  Ollama
       │                                                                  (local + :cloud)
       └──POST /aglm/augment-prompt──►  Flask AGLM brain (:5000)  →  system prompt + trace
```

- `/aglm/*` is rewritten to `http://localhost:5000/api/*` in `next.config.mjs`.
- Run both: `python3 aglm_app.py` (brain, :5000) and `npm run dev` in
  `aglm-console/` (UI, :3000).

## Pinned versions (peer-compatible set)

```
ai@7.0.9   @ai-sdk/react@4.0.10   @ai-sdk/openai-compatible@3.0.2
next@^15.5.19   react@19.2.7   react-dom@19.2.7   zod@3.25.76
```

- **`@ai-sdk/react@4.0.10` requires React ≥ 19.2.1** (peer
  `^18 || ~19.0.1 || ~19.1.2 || ^19.2.1`). React 19.2.0 fails install — use 19.2.7.
- Keep `next` on a CVE-patched 15.5.x (≥ 15.5.18).

## Route handler (server) — `app/api/chat/route.ts`

```ts
import { streamText, convertToModelMessages, type UIMessage } from 'ai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

const ollama = createOpenAICompatible({
  name: 'ollama',
  baseURL: 'http://localhost:11434/v1', // Ollama's OpenAI-compatible endpoint
  apiKey: 'ollama',                      // any non-empty string
});

export async function POST(req: Request) {
  const { messages, model, system, settings } = await req.json();
  const modelMessages = await convertToModelMessages(messages); // ⚠️ MUST await
  const result = streamText({
    model: ollama(model || 'qwen3:0.6b'),
    system,
    messages: modelMessages,
    temperature: settings?.temperature,
    topP: settings?.top_p,
    maxOutputTokens: settings?.max_tokens,
    providerOptions: { ollama: { top_k: 40, num_ctx: 4096 } }, // native extras
  });
  return result.toUIMessageStreamResponse();
}
```

### Gotchas that actually bit us
- **`convertToModelMessages` returns a Promise** — `await` it. Forgetting the
  await passes a Promise to `streamText`, which throws
  `messages.some is not a function`.
- `StreamTextResult` has **`.toUIMessageStreamResponse()`** (and
  `.toTextStreamResponse()`, `.pipeUIMessageStreamToResponse()`). The lower-level
  path is `createUIMessageStreamResponse({ stream: toUIMessageStream({ stream: result.stream }) })`.
- Provider param mapping: SDK `topP`/`maxOutputTokens` → OpenAI `top_p`/`max_tokens`.
  Ollama-native options (`top_k`, `num_ctx`, `repeat_penalty`) go in
  `providerOptions.ollama` (best-effort; the `/v1` endpoint ignores unknown keys).

## Client (useChat) — `app/page.tsx`

```ts
'use client';
import { useChat } from '@ai-sdk/react';
const { messages, sendMessage, status, stop, setMessages, error } = useChat();
// default transport posts to /api/chat. Send extra body per turn:
sendMessage({ text }, { body: { model, system, settings } });
```

- `status`: `'submitted' | 'streaming' | 'ready' | 'error'`. Disable Send unless
  `ready`; show Stop while `streaming`.
- Messages are `{ id, role, parts[] }`. Render `part.type === 'text'` as the
  answer; **`part.type === 'reasoning'`** is the thinking stream from reasoning
  models (qwen3, deepseek-r1) — show it collapsibly, not as the answer.
- Reasoning models spend tokens thinking first. Keep `max_tokens` generous
  (≥ ~1500) or the budget is consumed before any `text` part is emitted. The
  `/v1` endpoint does **not** honor `think:false` / `enable_thinking` — only
  Ollama's native `/api/chat` does.
- Restore a saved conversation with `setMessages(session.messages)`.

## Models

Discover live from the brain: `GET /aglm/ollama/models` → `{ local[], cloud[] }`.
`:cloud` models are proxied by the local daemon to ollama.com (needs
`ollama signin`). Default test model on modest hardware: `qwen3:0.6b`.

## Quick checks

```bash
# stream test (mimics useChat payload)
curl -sN localhost:3000/api/chat -H 'content-type: application/json' \
  -d '{"messages":[{"id":"1","role":"user","parts":[{"type":"text","text":"hi"}]}],"model":"qwen3:0.6b"}'
npx tsc --noEmit          # type check (dev mode tolerates type errors)
```
