# SPDX-License-Identifier: Apache-2.0
"""
Quick-start for aGLM. Runs one PODA cycle that picks a "time of day"
decision and logs it.

    pip install -e .
    python examples/quickstart.py
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from aglm import AGLMCore, Decision, PerceptionContext


async def perceive() -> PerceptionContext:
    now = datetime.now()
    return PerceptionContext(
        facts={"hour": now.hour, "minute": now.minute},
        source="clock",
    )


async def decide(ctx: PerceptionContext, beliefs) -> Decision:
    hour = ctx.facts["hour"]
    if hour < 6 or hour >= 22:
        return Decision(action="sleep", rationale=f"hour={hour} is outside waking range", confidence=0.9)
    if 6 <= hour < 12:
        return Decision(action="work", rationale=f"morning ({hour}h)", confidence=0.8)
    if 12 <= hour < 18:
        return Decision(action="meet", rationale=f"afternoon ({hour}h)", confidence=0.6)
    return Decision(action="rest", rationale=f"evening ({hour}h)", confidence=0.7)


async def act(d: Decision) -> dict:
    print(f"  ACT: {d.action}  rationale={d.rationale}  confidence={d.confidence:.2f}")
    return {"success": True, "action": d.action}


async def main():
    core = AGLMCore(perceive=perceive, decide=decide, act=act, agent_id="aglm.quickstart")
    outcome = await core.cycle()
    print()
    print(f"outcome: {outcome}")
    print(f"belief count after cycle: {len(core.beliefs)}")


if __name__ == "__main__":
    asyncio.run(main())
