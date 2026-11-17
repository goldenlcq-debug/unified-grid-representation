import numpy as np
import pytest

from rl_poweradj.graph_dqn import GraphDQN
from rl_poweradj.grid_types import FrozenGraph, GraphObservation


def observation():
    graph = FrozenGraph("exploration", np.zeros((2, 192), np.float32),
                        np.zeros((2, 192), np.float32), np.zeros((16, 192), np.float32),
                        np.array([[0, 1], [0, 1]], np.int64))
    edges = np.zeros((2, 5), np.float32)
    edges[1, 0] = 1
    return GraphObservation(graph, np.zeros((2, 8), np.float32), edges,
                            np.zeros(5, np.float32), np.array([0, 1], np.int64),
                            np.array([1, 0], np.int64), np.array([True, True]),
                            ("off-to-on", "on-to-off"))


def test_exploration_prefers_energizing_off_shunt_without_restricting_legal_mask():
    agent = GraphDQN.__new__(GraphDQN)
    agent.guard = lambda: None
    agent.rng = np.random.default_rng(5)
    state = observation()
    preferred = (state.targets == 1) & (state.edges[state.candidate_edges, 0] == 0)
    assert {agent.choose(state, epsilon=1, exploration_mask=preferred) for _ in range(20)} == {0}
    assert {agent.choose(state, epsilon=1) for _ in range(20)} == {0, 1}
    with pytest.raises(ValueError, match="boolean candidate mask"):
        agent.choose(state, epsilon=1, exploration_mask=np.array([1, 0]))
