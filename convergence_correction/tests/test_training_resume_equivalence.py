import numpy as np
import torch

from rl_poweradj.dqn import train, DQNAgent


class RandomObservationEnv:
    action_size = 2

    def __init__(self):
        self._rng = np.random.default_rng()
        self.steps = 0

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.steps = 0
        return self._rng.normal(size=4).astype(np.float32), {
            'converged': False, 'action_mask': np.ones(2, dtype=bool)}

    def step(self, action):
        self.steps += 1
        state = self._rng.normal(size=4).astype(np.float32)
        done = self.steps == 2
        return state, float(action + state[0]), done, False, {
            'converged': done, 'action_mask': np.ones(2, dtype=bool)}


def test_continuous_matches_checkpoint_resume_including_environment_rng(tmp_path):
    continuous, _ = train(RandomObservationEnv(), episodes=4, max_steps=2, batch_size=2, seed=13, device='cpu')
    first, _ = train(RandomObservationEnv(), episodes=2, max_steps=2, batch_size=2, seed=13, device='cpu')
    path = tmp_path / 'resume.pt'
    first.save(path, replay=first.training_replay, metrics=first.training_metrics, episode=2)
    restored, replay, metrics, episode = DQNAgent.load_checkpoint(path, 'cpu')
    resumed, _ = train(RandomObservationEnv(), episodes=2, max_steps=2, batch_size=2, seed=13, device='cpu',
                       agent=restored, replay=replay, metrics=metrics, start_episode=episode)
    assert continuous.training_metrics.to_dict() == resumed.training_metrics.to_dict()
    for key, value in continuous.online.state_dict().items():
        assert torch.equal(value, resumed.online.state_dict()[key])
    assert continuous.runtime_state == resumed.runtime_state
