# AGLM — Technical Reference

This document describes the architecture of AGLM (Autonomous General Learning
Model) and the two front ends in this repository. For a usage walkthrough see
[CONSOLE.md](CONSOLE.md); for the original module notes see [README.md](README.md).

## 1. Thesis

**AGLM is not a language model.** It is a *logic-augmentation layer* that wraps an
existing model with Socratic, epistemic, nonmonotonic, and BDI reasoning. Every
prior UI attempt in this repo (`uiux*.py`, `UIUX*.py`, `main.py`) failed to start
because it loaded a multi-GB model *before* rendering the UI. The current design
inverts that: the cognitive faculties are pure Python (no GPU, no checkpoint), so
the UI **always loads**, and the language model is an optional, pluggable backend.

## 2. Component map

```
                          ┌──────────────────────────────────────────┐
                          │  AGLM brain — Flask (aglm_app.py)  :5000  │
  Browser ───────────────►│  aglm_engine.py: faculties + augmentation │
   (either console)       │  /api/* JSON endpoints                    │
                          └───────────────┬──────────────────────────┘
                                          │  /api/augment-prompt, /api/ollama/*
   ┌──────────────────────────────────┐   │
   │ AI SDK console (aglm-console/)    │   │        ┌───────────────────────────┐
   │ Next.js 15 + Vercel AI SDK v7    │───┼───────►│ Ollama daemon  :11434     │
   │ /api/chat → streamText → Ollama  │   │  /v1   │ local + :cloud models     │
   └──────────────────────────────────┘   │        └───────────────────────────┘
                                          │
   ┌──────────────────────────────────┐   │
   │ Flask console (embedded HTML UI) │───┘
   │ tabs for every faculty           │
   └──────────────────────────────────┘
```

## 3. The cognitive engine (`aglm_engine.py`)

Each faculty is the **original prototype module**, imported behind a guard so the
engine still boots (and reports the faculty as degraded) if one fails.

| Faculty class | Wraps | Responsibility |
|---|---|---|
| `SocraticFaculty` | `reasoning.SocraticReasoning` | premises, challenge, conclude, probing questions |
| `LogicFaculty` | `logic.LogicTables` | truth tables over boolean expressions |
| `NonmonotonicFaculty` | `nonmonotonic.DefaultLogic` | default logic with retractable conclusions |
| `EpistemicFaculty` | `epistemic.AutoepistemicAgent` | belief state + revision |
| `BDIFaculty` | `bdi.*` | Belief / Desire / Intention state |
| `AutonomizeFaculty` | `autonomize.Autonomizer` | self-healing retry loop |
| `MastermindFaculty` | `MASTERMIND.py` | agent load / validate / execute |
| `MemoryFaculty` | `memory.py` | persist + list `./memory/*.json` |
| `PredictionFaculty` | `prediction.Predictor` | optional sklearn-style inference |

Output capture: faculties that only `print`/`log` (Socratic, Epistemic) are run
through `contextlib.redirect_stdout` or an instance-level `log` shim so their
output is returned as JSON rather than lost to stdout.

### Augmentation pipeline

`AugmentationEngine.augment_system(query)` runs the query through Socratic →
Epistemic → Nonmonotonic → BDI and returns:

- `system` — the Professor Codephreak persona (`automind.DEFAULT_SYSTEM_PROMPT`)
  plus a reasoning scaffold (probing questions + defeasibility + BDI plan).
- `trace` — a structured list of what each faculty contributed (shown in the UI).

The system prompt is what gets handed to a model. In *augmentation-only* mode the
augmented prompt **is** the deliverable, which is why the consoles load with no
model attached.

## 4. Model backend (Ollama)

Models are discovered live from a running Ollama daemon and split into local and
`:cloud`. The same OpenAI-compatible `/v1` endpoint serves both — the daemon
proxies `:cloud` models to ollama.com.

- `DEFAULT_OLLAMA_MODEL` (env `AGLM_DEFAULT_MODEL`) — default `gpt-oss:120b-cloud`.
- `OLLAMA_HOST` (env) — default `http://localhost:11434`.
- Flask endpoints: `/api/ollama/models`, `/api/ollama/search` (cloud library),
  `/api/ollama/pull` (auto-pull), `/api/chat` (augment + call).

Cloud entitlements: some `:cloud` models require an Ollama subscription and return
HTTP 403; `gpt-oss:120b-cloud` is free. Both consoles surface the real reason.

## 5. AI SDK console (`aglm-console/`)

Next.js 15 (App Router) + Vercel AI SDK v7. See the **`aisdk`** skill
(`.claude/skills/aisdk/`) for the implementation details and gotchas.

- **Route** `app/api/chat/route.ts`: `createOpenAICompatible({ baseURL: OLLAMA/v1,
  includeUsage: true })` → `streamText({ model, system, messages, … })` →
  `result.toUIMessageStreamResponse({ messageMetadata, onError })`.
  - `convertToModelMessages` **returns a Promise** — it must be awaited.
  - `includeUsage: true` makes Ollama emit `stream_options.include_usage`, so token
    counts arrive in the stream; they are attached to the assistant message via
    `messageMetadata` on the `finish` part (`totalUsage`).
  - `onError` forwards real provider errors (e.g. cloud 403) instead of the SDK's
    masked "An error occurred."
- **Client** `app/page.tsx`: `useChat` from `@ai-sdk/react`. Per-turn settings are
  sent with `sendMessage({ text }, { body: { model, system, settings } })`. The
  augmentation system prompt is fetched from `/aglm/augment-prompt` (a Next rewrite
  to the Flask brain) before each send.
- **Tabs**: Chat (streaming + reasoning collapse + token line), Advanced
  (temperature, top-p, top-k, penalties, max tokens, num_ctx, seed; persisted),
  `.history` (localStorage sessions; reopen / archive-to-memory / delete),
  Faculties (embedded Flask console), About.

### Token counter

- AI SDK console: real `inputTokens` / `outputTokens` / `totalTokens` per message
  (top-bar shows the running session total).
- Flask Ollama backend & automindx app: derived from Ollama's `prompt_eval_count`
  / `eval_count` (+ `eval_duration` for tok/s).

## 6. Pinned versions

```
ai 7.0.9 · @ai-sdk/react 4.0.10 · @ai-sdk/openai-compatible 3.0.2
next ^15.5.19 · react 19.2.7 · zod 3.25.76
@ai-sdk/react ≥ 4.0.10 requires React ≥ 19.2.1 (install fails on 19.2.0).
```

## 7. Run

```bash
./run_console.sh         # Flask cognitive console            → :5000
./run_aisdk_console.sh   # Flask brain :5000 + AI SDK console → :3000
```

No GPU or model checkpoint is required for the UIs to load. Ollama is optional;
without it the consoles run in augmentation-only mode.
