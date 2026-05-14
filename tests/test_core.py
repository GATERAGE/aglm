# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for AGLMCore — the PODA cycle."""
from __future__ import annotations

import pytest

from aglm import AGLMCore, BeliefSystem, Decision, PerceptionContext


@pytest.mark.asyncio
async def test_cycle_happy_path():
    async def perceive():
        return PerceptionContext(facts={"hour": 14}, source="clock")

    async def decide(ctx, beliefs):
        return Decision(action="log_hour", args={"hour": ctx.facts["hour"]})

    async def act(d):
        return {"success": True, "logged": d.args}

    core = AGLMCore(perceive=perceive, decide=decide, act=act, agent_id="test.aglm")
    outcome = await core.cycle()

    assert outcome["success"] is True
    assert outcome["stage"] == "complete"
    assert outcome["cycle"] == 1
    assert outcome["decision"] == "log_hour"
    assert core.cycle_count == 1


@pytest.mark.asyncio
async def test_cycle_perceive_failure_is_captured():
    async def perceive():
        raise RuntimeError("sensor offline")

    async def decide(ctx, beliefs):
        raise AssertionError("should not be reached")

    async def act(d):
        raise AssertionError("should not be reached")

    core = AGLMCore(perceive=perceive, decide=decide, act=act)
    outcome = await core.cycle()
    assert outcome["success"] is False
    assert outcome["stage"] == "perceive"
    assert "sensor offline" in outcome["error"]


@pytest.mark.asyncio
async def test_cycle_decide_failure_is_captured():
    async def perceive():
        return PerceptionContext(facts={})

    async def decide(ctx, beliefs):
        raise ValueError("no model available")

    async def act(d):
        raise AssertionError("should not be reached")

    core = AGLMCore(perceive=perceive, decide=decide, act=act)
    outcome = await core.cycle()
    assert outcome["success"] is False
    assert outcome["stage"] == "decide"
    assert "no model available" in outcome["error"]


@pytest.mark.asyncio
async def test_cycle_updates_beliefs():
    async def perceive():
        return PerceptionContext(facts={"temperature": 72}, source="sensor.1")

    async def decide(ctx, beliefs):
        return Decision(action="noop")

    async def act(d):
        return {"success": True}

    beliefs = BeliefSystem()
    core = AGLMCore(perceive=perceive, decide=decide, act=act, beliefs=beliefs)
    await core.cycle()

    # Perception was turned into a belief
    assert "temperature" in beliefs
    assert beliefs.top("temperature") is not None
    # Outcome was also recorded
    assert "outcome:noop" in beliefs


@pytest.mark.asyncio
async def test_cycle_counts_increment():
    async def perceive():
        return PerceptionContext(facts={})

    async def decide(ctx, beliefs):
        return Decision(action="x")

    async def act(d):
        return {"success": True}

    core = AGLMCore(perceive=perceive, decide=decide, act=act)
    for i in range(1, 4):
        out = await core.cycle()
        assert out["cycle"] == i
    assert core.cycle_count == 3
