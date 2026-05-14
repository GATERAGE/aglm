# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the belief system."""
from __future__ import annotations

import pytest

from aglm import Belief, BeliefSystem


def test_belief_confidence_validation():
    Belief(claim="x", confidence=0.0, source="t")  # ok
    Belief(claim="x", confidence=1.0, source="t")  # ok
    with pytest.raises(ValueError):
        Belief(claim="x", confidence=-0.1, source="t")
    with pytest.raises(ValueError):
        Belief(claim="x", confidence=1.5, source="t")


def test_belief_system_add_and_top():
    bs = BeliefSystem()
    bs.add(Belief(claim="sky is blue", confidence=0.5, source="agent.a"))
    bs.add(Belief(claim="sky is blue", confidence=0.9, source="agent.b"))
    bs.add(Belief(claim="grass is green", confidence=0.95, source="agent.c"))

    top = bs.top("sky is blue")
    assert top is not None
    assert top.confidence == 0.9
    assert top.source == "agent.b"

    assert len(bs) == 3


def test_belief_system_case_insensitive_keys():
    bs = BeliefSystem()
    bs.add(Belief(claim="The Cat is Black", confidence=0.8, source="t"))
    assert "the cat is black" in bs
    assert "THE CAT IS BLACK" in bs


def test_belief_revision_adds_peer():
    bs = BeliefSystem()
    bs.revise("x", 0.3, "agent.a")
    bs.revise("x", 0.8, "agent.b")
    all_beliefs = bs.all("x")
    assert len(all_beliefs) == 2
    assert bs.top("x").confidence == 0.8


def test_belief_system_round_trip():
    bs = BeliefSystem()
    bs.add(Belief(claim="a", confidence=0.5, source="s", metadata={"key": "value"}))
    bs.add(Belief(claim="b", confidence=0.7, source="s"))

    data = bs.to_dict()
    bs2 = BeliefSystem.from_dict(data)
    assert len(bs2) == 2
    assert bs2.top("a").metadata == {"key": "value"}
