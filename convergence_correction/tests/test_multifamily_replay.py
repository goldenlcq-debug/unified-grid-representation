from types import SimpleNamespace

import numpy as np
import pytest

from rl_poweradj.graph_dqn import GraphTransition
from rl_poweradj.grid_types import FrozenGraph, GraphObservation
from rl_poweradj.multifamily_replay import MultiFamilyReplay, sample_balanced
from rl_poweradj.multifamily_sampler import TRAIN_FAMILIES


def transition(family, index):
    state = SimpleNamespace(nodes=range(10), edges=range(20), targets=range(5))
    return SimpleNamespace(family=family, source=f"{family}-{index}", state=state, next_state=state)


def test_replay_balances_families_under_budgets():
    items = [transition(family, i) for family in TRAIN_FAMILIES for i in range(20)]
    items += [transition("grid_2000", i) for i in range(20, 120)]
    chosen = sample_balanced(SimpleNamespace(items=items), np.random.default_rng(7), 16,
                             node_budget=320, edge_budget=640, action_budget=160)
    assert len(chosen) == 16
    assert {family: sum(item.family == family for item in chosen) for family in TRAIN_FAMILIES} == {
        family: 4 for family in TRAIN_FAMILIES
    }


def test_replay_rejects_oversize_transition():
    replay = SimpleNamespace(items=[transition("grid_118", 1)])
    with pytest.raises(MemoryError):
        sample_balanced(replay, np.random.default_rng(0), 1,
                        node_budget=19, edge_budget=40, action_budget=10)


def test_replay_capacity_is_reserved_per_family():
    graph = FrozenGraph("shared", np.zeros((2, 192), np.float32),
                        np.zeros((1, 192), np.float32), np.zeros((16, 192), np.float32),
                        np.array([[0], [1]], np.int64))
    obs = GraphObservation(graph, np.zeros((2, 8), np.float32), np.zeros((1, 5), np.float32),
                           np.zeros(5, np.float32), np.array([0], np.int64),
                           np.array([1], np.int64), np.array([True]), ("device",))
    replay = MultiFamilyReplay(8)
    for family in TRAIN_FAMILIES:
        replay.add(GraphTransition(obs, 0, -1.0, obs, False, family, f"{family}-first"))
    for index in range(4):
        replay.add(GraphTransition(obs, 0, -1.0, obs, False, "grid_2000", f"source-{index}"))
    assert len(replay.parts["grid_2000"].items) == 2
    assert all(len(replay.parts[family].items) == 1 for family in TRAIN_FAMILIES[1:])
