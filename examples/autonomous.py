# SPDX-License-Identifier: Apache-2.0
"""
Run an aGLM core inside an AutonomousLoop that ticks every 2 seconds for
10 seconds, then stops cleanly.

    pip install -e .
    python examples/autonomous.py
"""
from __future__ import annotations

import asyncio
import random

from aglm import AGLMCore, AutonomousLoop, Decision, PerceptionContext


async def perceive() -> PerceptionContext:
    return PerceptionContext(facts={"reading": random.random()}, source="rng")


async def decide(ctx, beliefs) -> Decision:
    r = ctx.facts["reading"]
    return Decision(action="commit" if r > 0.5 else "skip", confidence=r)


async def act(d: Decision) -> dict:
    print(f"  ACT: {d.action}  confidence={d.confidence:.3f}")
    return {"success": True}


async def main():
    core = AGLMCore(perceive=perceive, decide=decide, act=act, agent_id="aglm.demo")
    loop = AutonomousLoop(core, interval_seconds=2.0)
    await loop.start()
    print("loop running for ~10 seconds…")
    await asyncio.sleep(10)
    await loop.stop()
    print()
    print(f"final status: {loop.status()}")


if __name__ == "__main__":
    asyncio.run(main())
