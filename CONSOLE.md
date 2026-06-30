# AGLM Cognitive Console — the UI that actually loads

The original AGLM/UIUX attempts (`uiux*.py`, `UIUX*.py`, `ui.py`, `main.py`) all
tried to spin up a Gradio chat **after loading a 7B language model** through
`torch` / `auto-gptq` / `llama-cpp`. That model-load step is what failed for
weeks — if the checkpoint, CUDA, or a quantization wheel wasn't perfect, the UI
never rendered at all.

This console fixes that by honoring what AGLM actually is. Per `automind.py` and
the README, **AGLM is not a model** — it is the *Autonomous General Learning
Model*, a logic layer that **augments any existing model**. So the UI runs the
**pure-Python cognitive faculties** (zero heavy dependencies, always loads) and
treats the language model as an **optional, pluggable backend**.

## Run it

```bash
./run_console.sh            # or:  python3 aglm_app.py
# open http://localhost:5000
```

The only hard dependency is Flask. With **no model attached** the console runs in
**augmentation-only** mode and still demonstrates every faculty.

## What's included (every aspect of AGLM)

| Faculty | Original module | In the console |
|---|---|---|
| **Augment / Chat** | `automind.py` (system prompt) | Runs your query through all faculties → a *logic-augmented prompt* → optional model |
| **Socratic** | `reasoning.py` `SocraticReasoning` | Add/challenge premises, draw conclusions, generate probing questions |
| **Logic Tables** | `logic.py` `LogicTables` | Truth tables over boolean variables/expressions |
| **Nonmonotonic** | `nonmonotonic.py` `DefaultLogic` | Default logic with retractable conclusions + entailment query |
| **Epistemic** | `epistemic.py` `AutoepistemicAgent` | Belief state + revision under contradicting information |
| **BDI Agent** | `bdi.py` | Beliefs / Desires / Intentions, execute intentions |
| **MASTERMIND** | `MASTERMIND.py` | Loads, validates & executes agents; shows the data store |
| **Autonomize** | `autonomize.py` `Autonomizer` | Resilient self-healing cycle with exponential backoff |
| **Prediction** | `prediction.py` `Predictor` | Optional ML inference (drop a `trained_model.pkl` in `./models`) |
| **Memory** | `memory.py` | Persists chat transcripts to `./memory/*.json`; browse them |

The faculties are the **real prototype modules**, imported directly. Each import
is guarded, so if one ever fails to load the console still starts and reports
that faculty as degraded (see the **About** tab → faculty health).

## Models (Ollama, local + cloud)

The chat backend is discovered live from a running Ollama daemon:

- **Local** models (e.g. `qwen3:0.6b`, `deepseek-r1:1.5b`) for quick tests on
  modest hardware — the default is `qwen3:0.6b`.
- **`:cloud`** models (e.g. `glm-5.2:cloud`, `gpt-oss:120b-cloud`,
  `deepseek-v4-pro:cloud`) proxied by the local daemon to ollama.com for larger
  reasoning. Cloud models require `ollama signin`.

The **Models** tab shows a green **LIVE** light when the daemon is reachable,
lists local + pulled cloud models, and lets you **search the ollama.com cloud
library** and **pull** any model with one click (auto-pull). `glm-5.2:cloud` is
available and can be pulled there or with `ollama pull glm-5.2:cloud`.

Override the daemon URL or default model:

```bash
OLLAMA_HOST=http://localhost:11434 AGLM_DEFAULT_MODEL=qwen3:0.6b python3 aglm_app.py
```

## How augmentation works

For each query the engine:

1. **Socratic** — frames the query as a premise and generates probing questions.
2. **Epistemic** — registers it as a tentative, revisable belief.
3. **Nonmonotonic** — marks conclusions as defeasible.
4. **BDI** — derives a Belief → Desire → Intention scaffold.
5. Builds a **logic-augmented prompt** (Professor Codephreak persona + the
   reasoning scaffold + your query) and hands it to the selected model.

In augmentation-only mode that augmented prompt *is* the deliverable — it's
exactly what AGLM would feed to any model. That's why the UI always loads.

## Files added

- `aglm_engine.py` — wires the original faculties into one augmentation engine.
- `aglm_app.py` — self-contained Flask UI (HTML/CSS/JS embedded, no CDNs).
- `run_console.sh` — launcher.

---

# AI SDK participant console (`aglm-console/`)

A second, "clean-house" front end built on the **Vercel AI SDK v7** for a
polished streaming participant experience. The Flask app above stays the **AGLM
brain** (faculties + augmentation); this Next.js app is the **participant UI**.

```bash
./run_aisdk_console.sh         # starts Flask brain :5000 + Next console :3000
# then open http://localhost:3000   (needs Ollama: `ollama serve`)
```

Manually:

```bash
python3 aglm_app.py                      # brain on :5000
cd aglm-console && npm install && npm run dev   # console on :3000
```

### What it adds

- **Chat** — real token **streaming** via `useChat` (`@ai-sdk/react`). Reasoning
  models (qwen3, deepseek-r1) show a collapsible *thinking* stream, then the
  answer. The **AGLM augmentation** toggle runs each query through the brain
  (`/aglm/augment-prompt`) and shows the Socratic/epistemic/nonmonotonic/BDI
  trace beneath the reply.
- **Advanced** — change **model state** live: temperature, top-p, top-k, repeat
  penalty, frequency/presence penalty, max tokens, context window, seed. Sent
  per-turn into `streamText`. Settings persist in the browser.
- **.history** — every conversation is saved locally; reopen to continue, archive
  to AGLM long-term memory (`./memory/*.json`), or delete.
- **Faculties** — the full Flask cognitive console, embedded.
- **Models** — local + `:cloud` models from the live Ollama list; cloud 403s
  (subscription-gated models) surface their real reason.

### How a turn flows

```
You → (AGLM augmentation on?) → POST /aglm/augment-prompt   [Socratic·Epistemic·Nonmonotonic·BDI → system prompt + trace]
    → useChat → POST /api/chat → streamText( ollama(model), system, messages, {temperature, top_p, …} )
    → tokens stream into the bubble; trace shown beneath.
```

The chat route (`app/api/chat/route.ts`) talks to Ollama's OpenAI-compatible
`/v1` endpoint via `@ai-sdk/openai-compatible`, so the same code serves local and
`:cloud` models. See the **`aisdk`** skill (`.claude/skills/aisdk/`) for the
implementation notes and gotchas.
