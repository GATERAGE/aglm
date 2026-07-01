<div align="center">

# aGLM — Autonomous General Learning Model

**A logic-augmentation layer for language models — not a model itself.**

aGLM wraps any existing model with Socratic, epistemic, nonmonotonic, and BDI
reasoning. Because the reasoning layer is pure Python, the interfaces **load
instantly** and the language model is optional and pluggable.

[![License](https://img.shields.io/badge/license-GPLv3%20%2F%20Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)](pyproject.toml)
[![UI](https://img.shields.io/badge/UI-Flask%20%2B%20Next.js%2015-000)](CONSOLE.md)
[![AI SDK](https://img.shields.io/badge/Vercel%20AI%20SDK-v7-black)](https://ai-sdk.dev)
[![Models](https://img.shields.io/badge/models-Ollama%20local%20%2B%20cloud-4b9)](TECHNICAL.md)

</div>

---

## Overview

AGLM is a hybridization of **MASTERMIND** aGLM with **RAGE**, distilled from the
Professor Codephreak / easyAGI research lineage. Its premise is simple:

> **AGLM is not a language model — it augments one.** Every input can be run
> through a cognitive pipeline (Socratic framing → epistemic belief → nonmonotonic
> defeasibility → BDI plan) that produces a *logic-augmented prompt*. That prompt
> is what gets handed to a model, or, with no model attached, is the deliverable
> itself.

The repository contains three things:

1. **Two consoles** — working user interfaces that actually load.
2. **A modern `aglm/` Python package** — the installable Perceive-Decide-Act core.
3. **The historical research modules** — the cognitive faculties (`socratic.py`,
   `bdi.py`, `logic.py`, `MASTERMIND.py`, …) that AGLM grew out of.

---

## Quickstart — the consoles

Both front ends run with **zero heavy ML dependencies**. The language model runs
in [Ollama](https://ollama.com) (local for modest hardware, `:cloud` for larger
reasoning) and is entirely optional.

```bash
# 1) Flask cognitive console — every faculty, zero heavy deps
./run_console.sh              # → http://localhost:5000

# 2) AI SDK participant console — streaming chat on the Vercel AI SDK v7
./run_aisdk_console.sh        # Flask brain :5000 + Next.js console :3000
```

The **AI SDK console** adds a streaming chat with a model picker (local + `:cloud`),
an **Advanced** model-state panel (temperature, top-p, top-k, penalties, context
window, seed), a **`.history`** tab, and a live **token counter**.

| Faculty | Module | In the console |
|---|---|---|
| Socratic | `reasoning.py` | premises · challenge · conclude · probing questions |
| Logic tables | `logic.py` | truth tables over boolean expressions |
| Nonmonotonic | `nonmonotonic.py` | default logic with retractable conclusions |
| Epistemic | `epistemic.py` | belief state + revision |
| BDI | `bdi.py` | Belief / Desire / Intention agent state |
| Autonomize | `autonomize.py` | self-healing retry loop |
| MASTERMIND | `MASTERMIND.py` | agent load · validate · execute |
| Memory | `memory.py` | persist + browse `./memory/*.json` |
| Prediction | `prediction.py` | optional sklearn-style inference |

📖 **[CONSOLE.md](CONSOLE.md)** — usage walkthrough &nbsp;·&nbsp;
🧭 **[TECHNICAL.md](TECHNICAL.md)** — architecture &nbsp;·&nbsp;
🛠 **`.claude/skills/aisdk/`** — AI SDK v7 reference

---

## The modern `aglm/` package

A standalone, Apache-2.0 Python distribution of aGLM's autonomous-learning loop,
distilled from the [agenticplace/mindX](https://github.com/agenticplace) pattern.

```bash
pip install .                  # core only
pip install ".[rage]"          # with GATERAGE/RAGE
pip install ".[mastermind]"    # with GATERAGE/mastermind
pip install ".[dev]"           # pytest, ruff
```

```python
import asyncio
from aglm import AGLMCore, AutonomousLoop, Decision, PerceptionContext

async def perceive():          return PerceptionContext(facts={"hour": 14})
async def decide(ctx, beliefs): return Decision(action="log", args=ctx.facts)
async def act(decision):        return {"success": True}

async def main():
    core = AGLMCore(perceive=perceive, decide=decide, act=act)
    print(await core.cycle())                              # one cycle
    loop = AutonomousLoop(core, interval_seconds=300.0)    # or a periodic runner
    await loop.start()
    await loop.stop()

asyncio.run(main())
```

| Module | Class | Responsibility |
|---|---|---|
| `aglm/core.py` | `AGLMCore` | Perceive · Orient · Decide · Act cycle |
| `aglm/beliefs.py` | `BeliefSystem` | claim + confidence + source attribution |
| `aglm/cycle.py` | `AutonomousLoop` | periodic runner with circuit breaker |

```bash
pip install ".[dev]" && pytest -v
```

Canonical contract: [`docs/aglm_as_a_service.md`](docs/aglm_as_a_service.md).

---

## Ecosystem

> **RAGE remembers · aGLM decides · MASTERMIND orchestrates.**

- **[GATERAGE/RAGE](https://github.com/GATERAGE/RAGE)** — retrieval substrate (memory)
- **[GATERAGE/mastermind](https://github.com/GATERAGE/mastermind)** — strategic orchestrator (planning)
- **[pythaiml/automindx](https://github.com/pythaiml/automindx)** — Professor Codephreak deployment environment

---

## Historical research modules

The files at the repo root document the philosophical foundation aGLM grew out of
and remain **preserved unchanged**. The cognitive faculties are used directly by
the consoles above.

| Module | Purpose |
|---|---|
| `MASTERMIND.py` | Core controller — loads, validates, and executes agents concurrently |
| `bdi.py` | Beliefs · Desires · Intentions agent framework |
| `socratic.py` / `reasoning.py` | Socratic question-and-answer reasoning over premises |
| `logic.py` | Formal logic operations and truth tables |
| `nonmonotonic.py` | Non-monotonic reasoning — beliefs adapt to contradicting evidence |
| `epistemic.py` | Knowledge/belief tracking and revision over time |
| `autonomize.py` | Self-healing autonomy with exponential backoff |
| `prediction.py` | Forecasting via statistical / ML models |
| `terminai.py` · `terminai_module.py` | OpenAI command-mode (`cmd:`) integration |
| `SimpleCoder.py` | Reusable code snippets and templates |
| `config.json` | Default allowed agency for MASTERMIND |
| `chunk4096.py` | Guards inputs above the 4096-token context of the original model |

> **Note.** The legacy Gradio entrypoints (`uiux*.py`, `main.py`) tried to load a
> multi-GB model *before* rendering and are kept for reference only. Use the
> consoles above instead — they are the working interfaces.

---

## Attribution & license

- **MASTERMIND** — agent creator and control agent · © codephreak, **GPLv3**, 2024
  ([NFT](https://opensea.io/assets/matic/0xf0ba8dcdfba1b5aed0b46acddf7dde97075e97a2/1)).
- **Modern `aglm/` package** — **Apache-2.0**, © 2024–2026 GATERAGE / Professor Codephreak.
- Project details: **[rage.pythai.net](https://rage.pythai.net)**.

See [`LICENSE`](LICENSE) for the full text.
