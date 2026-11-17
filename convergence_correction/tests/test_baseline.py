import numpy as np

from rl_poweradj.baseline import beam_search, greedy_search


class SearchEnv:
    action_size = 2

    def __init__(self):
        self.state = 0

    def reset(self, case_index=0):
        self.state = 0
        return np.asarray([0.0]), {
            "converged": False, "error": 10.0,
            "action_mask": np.asarray([True, True]),
        }

    def step(self, action):
        self.state += 1
        converged = action == 1
        return np.asarray([float(self.state)]), 0.0, converged, False, {
            "converged": converged,
            "error": 0.0 if converged else 5.0,
            "action_mask": np.asarray([False, False]),
            "device_id": f"S{action}",
            "target_status": 1,
        }

    def close(self):
        pass


def test_deterministic_baselines_find_best_single_step_action():
    for search in (greedy_search, lambda factory: beam_search(factory, beam_width=2)):
        result = search(SearchEnv)
        assert result.converged
        assert result.actions == (1,)
        assert result.device_actions == (("S1", 1),)
