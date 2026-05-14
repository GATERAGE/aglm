# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the AutonomousLoop runner."""
from __future__ import annotations

import asyncio

import pytest

from aglm import AGLMCore, AutonomousLoop, Decision, PerceptionContext


@pytest.mark.asyncio
async def test_loop_start_stop():
    counter = {"n": 0}

    async def perceive():
        counter["n"] += 1
        return PerceptionContext(facts={"n": counter["n"]})

    async def decide(ctx, beliefs):
        return Decision(action="tick")

    async def act(d):
        return {"success": True}

    core = AGLMCore(perceive=perceive, decide=decide, act=act)
    loop = AutonomousLoop(core, interval_seconds=0.05)

    await loop.start()
    assert loop.is_running

    # Let it tick a few times.
    await asyncio.sleep(0.18)
    await loop.stop()

    assert counter["n"] >= 2
    assert loop.is_running is False


@pytest.mark.asyncio
async def test_loop_survives_failing_cycle():
    counter = {"n": 0}

    async def perceive():
        counter["n"] += 1
        if counter["n"] % 2 == 0:
            raise RuntimeError("flaky sensor")
        return PerceptionContext(facts={"n": counter["n"]})

    async def decide(ctx, beliefs):
        return Decision(action="tick")

    async def act(d):
        return {"success": True}

    core = AGLMCore(perceive=perceive, decide=decide, act=act)
    loop = AutonomousLoop(core, interval_seconds=0.03)

    await loop.start()
    await asyncio.sleep(0.20)
    await loop.stop()

    # Should have ticked several times despite the alternating failures.
    assert counter["n"] >= 3
    assert core.cycle_count >= 3
