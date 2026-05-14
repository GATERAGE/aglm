# aGLM as a Service

> *aGLM — Autonomous General Learning Model. The Perceive-Orient-Decide-Act
> loop with a belief substrate. This document is the contract for what aGLM
> offers as a service to multi-agent systems built on top of it.*

Companion specs:

- [RAGE — Retrieval Augmented Generative Engine](https://github.com/GATERAGE/RAGE)
- [MASTERMIND — strategic orchestrator](https://github.com/GATERAGE/mastermind)

Together: **RAGE remembers, aGLM decides, MASTERMIND orchestrates.**

---

## 1. What aGLM is

aGLM is a primitive: a single agent's cognitive cycle. Each cycle, the
agent perceives external state, orients its beliefs to that percept,
decides on an action, executes the action, and records the outcome.

The package distills the autonomous-learning-loop pattern from mindX
([`agents/core/agint.py`](https://github.com/agenticplace) — the
P-O-D-A cognitive core, and `agents/core/mindXagent.py` — the meta-agent
that runs the improvement loop). The mindX versions are
production-scale (174KB+, integrated with LLM router + RAGE + DAIO);
this package is the **agnostic, framework-independent distillation** —
roughly 400 LOC across three modules.

mindX is one consumer of this pattern; this repo is the canonical
agnostic home.

---

## 2. The three primitives

### 2.1 `AGLMCore` — the PODA cycle

```python
from aglm import AGLMCore, BeliefSystem, Decision, PerceptionContext

async def perceive() -> PerceptionContext:
    return PerceptionContext(facts={"temperature": read_sensor()})

async def decide(ctx, beliefs) -> Decision:
    if ctx.facts["temperature"] > 80:
        return Decision(action="alert", args={"level": "hot"})
    return Decision(action="noop")

async def act(d) -> dict:
    return {"success": True, "action": d.action}

core = AGLMCore(perceive=perceive, decide=decide, act=act)
outcome = await core.cycle()
```

Properties:
- Async by default — every callback is awaitable.
- Exception-isolated — a failing perceive/decide/act surfaces as a
  structured outcome dict (`{success: False, stage: 'decide', error:
  '...'}`), never raises out of `cycle()`.
- Belief-updating — every percept becomes a belief; every outcome
  becomes a belief.

### 2.2 `BeliefSystem` — claim + confidence + source

```python
from aglm import Belief, BeliefSystem

bs = BeliefSystem()
bs.add(Belief(claim="user wants coffee", confidence=0.8, source="user.input"))
bs.add(Belief(claim="user wants coffee", confidence=0.4, source="historical.pattern"))

top = bs.top("user wants coffee")
# top.confidence = 0.8, top.source = "user.input"
```

Properties:
- Multiple beliefs about the same claim co-exist (no silent
  overwrite). `top()` returns the highest-confidence one.
- Case-insensitive claim keys (whitespace + casing normalized).
- Serializable to / from dict (round-trip stable).
- `revise()` is the convenience method for "I now think X with
  confidence C" — adds a peer, doesn't displace.

### 2.3 `AutonomousLoop` — periodic runner

```python
from aglm import AutonomousLoop

loop = AutonomousLoop(core, interval_seconds=300.0, max_consecutive_failures=5)
await loop.start()
# … loop runs cycles every 5 minutes …
await loop.stop()
```

Properties:
- Configurable interval, default 300s (same as mindX `mindxagent`
  autonomous mode).
- Circuit breaker — after `max_consecutive_failures` consecutive
  failures, backs off for `backoff_seconds` (default 120s) before
  resuming, then resets the counter.
- Exception-isolated — the loop NEVER kills itself. If `cycle()`
  somehow raises (shouldn't, but defensive), the loop logs and
  continues.
- Stop is graceful — waits for the current cycle to complete, then
  exits cleanly.

---

## 3. Composition with RAGE + MASTERMIND

aGLM is one corner of a triangle:

```
                     ┌──────────────────┐
                     │   MASTERMIND     │  (directive → plan → execute)
                     │  orchestrator    │
                     └────────┬─────────┘
                              │ delegates to
                              ▼
            ┌─────────────────────────────────────┐
            │             aGLM                    │  (PODA cycle + beliefs)
            │   AGLMCore + BeliefSystem           │
            │   AutonomousLoop                    │
            └─────────┬───────────────────────────┘
                      │ retrieves context via
                      ▼
            ┌─────────────────────────────────────┐
            │              RAGE                   │  (retrieval substrate)
            │   IngestionEngine + RetrievalEngine │
            └─────────────────────────────────────┘
```

- A MASTERMIND-orchestrated agent runs an `AGLMCore` for its decision
  loop.
- Inside `decide()`, the agent queries RAGE for relevant memory
  before picking an action.
- Inside `act()`, the agent executes — possibly delegating to other
  MASTERMIND-known agents.
- Outcomes go back into the BeliefSystem; periodic dream cycles
  (mindX-style) consolidate beliefs into skills.

This is the production shape mindX runs. The three repos let you adopt
the pattern without taking mindX wholesale.

---

## 4. Service boundaries

aGLM does **not**:

- Make LLM calls itself. The `decide()` callback may use one; the
  package stays LLM-agnostic.
- Persist beliefs across restarts. Use `BeliefSystem.to_dict()` /
  `BeliefSystem.from_dict()` for serialization; persistence is the
  consumer's job.
- Orchestrate multiple agents. That's MASTERMIND's role.
- Retrieve external knowledge. That's RAGE's role.

aGLM **does**:

- Provide a typed PODA cycle.
- Provide a typed belief store.
- Provide a periodic runner with circuit breaker.
- Stay framework-agnostic — no FastAPI, no LangChain, no LLM SDK.

---

## 5. Roadmap

| Phase | What lands | When |
|---|---|---|
| **aGLM-1** | This spec + the three primitives + tests + examples | Shipped 2026-05-14 |
| **aGLM-2** | Pluggable persistence (JSON, SQLite, pgvector via RAGE) | post-MVP |
| **aGLM-3** | Belief revision strategies (non-monotonic, Bayesian update) | post-MVP |
| **aGLM-4** | Distillation: belief → skill promotion (mindX dream-cycle pattern) | post-MVP |
| **aGLM-5** | Gödel-machine audit trail (decision provenance + replay) | post-MVP |

---

## 6. References

- `aglm/core.py` — `AGLMCore` implementation
- `aglm/beliefs.py` — `BeliefSystem` implementation
- `aglm/cycle.py` — `AutonomousLoop` implementation
- `examples/quickstart.py` — one cycle
- `examples/autonomous.py` — periodic loop
- [RAGE](https://github.com/GATERAGE/RAGE) — memory substrate
- [MASTERMIND](https://github.com/GATERAGE/mastermind) — orchestrator
- [mindX](https://github.com/agenticplace) — production consumer of all three
- [rage.pythai.net](https://rage.pythai.net) — RAGE/aGLM/MASTERMIND docs
