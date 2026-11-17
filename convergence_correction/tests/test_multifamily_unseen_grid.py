import numpy as np
import pytest

from rl_poweradj.graph_q_network import CandidateGraphQ
from rl_poweradj.grid_types import FrozenGraph, GraphObservation
from rl_poweradj.multifamily_external import require_unseen_grid


def test_unseen_family_and_node_count_are_required():
    families = ("grid_118", "grid_300", "grid_2000", "grid_3120")
    sizes = (118, 300, 2000, 3120)
    require_unseen_grid("new_grid_157", 157, training_families=families, training_bus_counts=sizes)
    with pytest.raises(ValueError, match="used in training"):
        require_unseen_grid("grid_300", 157, training_families=families, training_bus_counts=sizes)
    with pytest.raises(ValueError, match="not unseen"):
        require_unseen_grid("new_grid_300", 300, training_families=families, training_bus_counts=sizes)


def test_shared_scorer_accepts_unseen_node_count_and_candidate_count():
    n, m, a = 157, 211, 13
    graph = FrozenGraph("synthetic-unseen", np.zeros((n, 192), np.float32),
                        np.zeros((m, 192), np.float32), np.zeros((16, 192), np.float32),
                        np.vstack((np.arange(m) % n, (np.arange(m) + 1) % n)).astype(np.int64))
    obs = GraphObservation(graph, np.zeros((n, 8), np.float32), np.zeros((m, 5), np.float32),
                           np.zeros(5, np.float32), np.arange(a, dtype=np.int64),
                           np.zeros(a, dtype=np.int64), np.ones(a, dtype=np.bool_),
                           tuple(f"device-{i}" for i in range(a)))
    model = CandidateGraphQ(width=16, layers=1)
    assert model(obs).shape == (a,)
