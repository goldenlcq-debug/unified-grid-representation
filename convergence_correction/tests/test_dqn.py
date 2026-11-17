import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl_poweradj.dqn import (
    DQNAgent, ReplayBuffer, Transition, epsilon_at_step, evaluate, train, train_parallel,
)
from rl_poweradj.model import DETERMINISTIC_OBSERVATION_METADATA


class TinyEnv:
    action_size = 3
    success_reward = 10.0

    def __init__(self):
        self.steps = 0

    def close(self):
        pass

    def reset(self, **_kwargs):
        self.steps = 0
        return np.zeros(4, dtype=np.float32), {
            "converged": False,
            "action_mask": np.asarray([True, True, False]),
        }

    def step(self, _action):
        self.steps += 1
        done = self.steps >= 2
        return (
            np.full(4, self.steps, dtype=np.float32),
            1.0,
            done,
            False,
            {"converged": done, "action_mask": np.asarray([True, not done, False])},
        )


class AlwaysConvergedEnv(TinyEnv):
    def reset(self, **_kwargs):
        state, info = super().reset()
        info["converged"] = True
        return state, info


class Registry:
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        return {"generator_ids": [self.name], "shunt_ids": ["S"], "load_ids": []}


class RegisteredTinyEnv(TinyEnv):
    action_size = 3

    def __init__(self, name):
        super().__init__()
        self.registry = Registry(name)


class RewardRegisteredTinyEnv(RegisteredTinyEnv):
    reward_contract = {"schema_version": 3, "kind": "lcq-test"}


def test_dqn_respects_action_mask_and_learns():
    agent = DQNAgent(4, 3, device="cpu")
    state = np.zeros(4, dtype=np.float32)
    mask = np.asarray([False, True, False])
    assert agent.choose_action(state, mask, epsilon=0.0) == 1
    replay = ReplayBuffer(10)
    for _ in range(4):
        replay.add(Transition(state, 1, 1.0, state, True, mask))
    assert agent.learn(replay, batch_size=4) is not None


def test_epsilon_schedule_depends_only_on_global_environment_steps():
    assert epsilon_at_step(0, 100) == 1.0
    assert np.isclose(epsilon_at_step(50, 100), 0.525)
    assert np.isclose(epsilon_at_step(100, 100), 0.05)
    assert np.isclose(epsilon_at_step(1000, 100), 0.05)


def test_training_rejects_zero_transition_model():
    with pytest.raises(RuntimeError, match="no transitions"):
        train(AlwaysConvergedEnv(), episodes=2, max_steps=2, batch_size=2, device="cpu")


def test_parallel_rollouts_learn_with_isolated_environments():
    agent, rewards = train_parallel(
        lambda _worker: TinyEnv(), episodes=4, workers=2,
        max_steps=2, batch_size=2, device="cpu",
    )
    assert len(rewards) == 4
    assert agent.updates > 0
    assert agent.training_metrics.transitions == 8


def test_registered_training_records_deterministic_observation_contract():
    agent, _rewards = train(
        RegisteredTinyEnv("G1"), episodes=2, max_steps=2, batch_size=2, device="cpu"
    )
    assert agent.metadata["encoder"] == DETERMINISTIC_OBSERVATION_METADATA
    assert agent.metadata["encoder"]["trainable"] is False


def test_training_records_and_validates_reward_contract():
    env = RewardRegisteredTinyEnv("G1")
    agent, _rewards = train(env, episodes=2, max_steps=2, batch_size=2, device="cpu")
    assert agent.metadata["reward"] == env.reward_contract
    incompatible = DQNAgent(
        4, 3, device="cpu",
        metadata={
            "registry": Registry("G1").to_dict(),
            "encoder": DETERMINISTIC_OBSERVATION_METADATA,
            "reward": {"schema_version": 2, "kind": "old"},
        },
    )
    with pytest.raises(ValueError, match="reward contract differs"):
        train(env, episodes=1, max_steps=2, batch_size=2, device="cpu", agent=incompatible)


def test_training_rejects_missing_observation_contract():
    env = RegisteredTinyEnv("G1")
    agent = DQNAgent(
        4, 3, device="cpu", metadata={"registry": Registry("G1").to_dict()}
    )
    with pytest.raises(ValueError, match="observation contract differs"):
        train(env, episodes=1, max_steps=2, batch_size=2, device="cpu", agent=agent)


def test_periodic_checkpoints_keep_only_configured_newest_files(tmp_path):
    train(
        TinyEnv(), episodes=4, max_steps=2, batch_size=2, device="cpu",
        checkpoint_dir=tmp_path, checkpoint_interval=1, checkpoint_keep_last=2,
    )
    assert [path.name for path in sorted(tmp_path.glob("dqn-episode-*.pt"))] == [
        "dqn-episode-3.pt", "dqn-episode-4.pt",
    ]


def test_parallel_rollouts_cycle_across_dataset_cases():
    visited = []

    class IndexedEnv(TinyEnv):
        cases = tuple(range(4))
        current_case_index = 0

        def reset(self, case_index=None, **kwargs):
            if case_index is None:
                case_index = self.current_case_index
            self.current_case_index = (case_index + 1) % len(self.cases)
            visited.append(case_index)
            return super().reset(**kwargs)

    train_parallel(
        lambda _worker: IndexedEnv(), episodes=4, workers=2,
        max_steps=2, batch_size=2, device="cpu",
    )
    # The initial probe visits zero; rollout episodes must still cover all cases.
    assert set(visited) == {0, 1, 2, 3}


def test_serial_training_cycles_cases_and_resume_starts_at_episode_cursor():
    visited = []

    class IndexedEnv(TinyEnv):
        cases = tuple(range(4))
        current_case_index = 0

        def reset(self, case_index=None, **kwargs):
            if case_index is None:
                case_index = self.current_case_index
            self.current_case_index = (case_index + 1) % len(self.cases)
            visited.append(case_index)
            return super().reset(**kwargs)

    train(
        IndexedEnv(), episodes=3, start_episode=2,
        max_steps=2, batch_size=2, device="cpu",
    )
    assert visited == [2, 3, 0]


def test_evaluation_rotates_from_explicit_cursor_and_records_each_case():
    visited = []

    class EvaluationEnv(TinyEnv):
        cases = tuple(range(4))

        def reset(self, case_index=0, **kwargs):
            visited.append(case_index)
            state, info = super().reset(**kwargs)
            info["source_case"] = f"case-{case_index}"
            info["error"] = float(4 - case_index)
            return state, info

        def step(self, action):
            state, reward, terminated, truncated, info = super().step(action)
            info["error"] = 0.0 if terminated else 1.0
            return state, reward, terminated, truncated, info

    env = EvaluationEnv()
    agent = DQNAgent(4, 3, device="cpu")
    result = evaluate(agent, env, episodes=2, max_steps=2, start_case=2)
    assert visited == [2, 3]
    assert [case.case_index for case in result.cases] == [2, 3]
    assert [case.source_case for case in result.cases] == ["case-2", "case-3"]
    assert result.successes == 2
    assert result.worst_final_error == 0.0


def test_best_checkpoint_uses_complete_validation_and_batch_cursor_rotates(tmp_path):
    class ValidationEnv(TinyEnv):
        cases = tuple(range(5))

        def reset(self, case_index=0, **kwargs):
            state, info = super().reset(**kwargs)
            info["source_case"] = f"validation-{case_index}"
            info["error"] = 1.0
            return state, info

        def step(self, action):
            state, reward, terminated, truncated, info = super().step(action)
            info["error"] = 0.0 if terminated else 1.0
            return state, reward, terminated, truncated, info

    best = tmp_path / "best.pt"
    agent, _rewards = train(
        TinyEnv(),
        episodes=4,
        max_steps=2,
        batch_size=2,
        device="cpu",
        evaluation_env=ValidationEnv(),
        evaluation_interval=1,
        evaluation_episodes=2,
        full_evaluation_interval=3,
        best_model_path=best,
    )
    metrics = agent.training_metrics
    assert [item["start_case"] for item in metrics.evaluation_history] == [0, 2]
    assert all(item["episodes"] == 2 for item in metrics.evaluation_history)
    assert len(metrics.full_validation_history) == 2
    assert all(item["episodes"] == 5 for item in metrics.full_validation_history)
    assert all(len(item["cases"]) == 5 for item in metrics.full_validation_history)
    assert best.is_file()
    restored, _replay, restored_metrics, _episode = DQNAgent.load_checkpoint(best, "cpu")
    assert restored_metrics.full_validation_history
    assert restored_metrics.validation_cursor == 4


def test_parallel_training_uses_same_rotating_validation_contract(tmp_path):
    class ValidationEnv(TinyEnv):
        cases = tuple(range(4))

    best = tmp_path / "parallel-best.pt"
    agent, _rewards = train_parallel(
        lambda _worker: TinyEnv(),
        episodes=6,
        workers=2,
        max_steps=2,
        batch_size=2,
        device="cpu",
        evaluation_env=ValidationEnv(),
        evaluation_interval=1,
        evaluation_episodes=2,
        full_evaluation_interval=5,
        best_model_path=best,
    )
    metrics = agent.training_metrics
    assert [item["start_case"] for item in metrics.evaluation_history] == [0, 2]
    assert len(metrics.full_validation_history) == 1
    assert metrics.full_validation_history[0]["episodes"] == 4
    assert best.is_file()


def test_parallel_training_rejects_same_shape_different_registry():
    agent = DQNAgent(
        4, 3, device="cpu",
        metadata={"registry": Registry("different-generator").to_dict()},
    )
    with pytest.raises(ValueError, match="registry differs"):
        train_parallel(
            lambda _worker: RegisteredTinyEnv("expected-generator"),
            episodes=2, workers=2, batch_size=2, device="cpu", agent=agent,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_dqn_full_cuda_path(tmp_path):
    agent = DQNAgent(4, 3, device="cuda")
    assert agent.device.type == "cuda"
    assert all(parameter.is_cuda for parameter in agent.online.parameters())
    state = np.zeros(4, dtype=np.float32)
    mask = np.asarray([True, True, False])
    replay = ReplayBuffer(16)
    for index in range(8):
        replay.add(Transition(state, index % 2, 1.0, state, index == 7, mask))
    loss = agent.learn(replay, batch_size=8)
    assert loss is not None and np.isfinite(loss)
    assert torch.cuda.memory_allocated() > 0
    checkpoint = tmp_path / "cuda-agent.pt"
    agent.save(checkpoint)
    restored = DQNAgent.load(checkpoint, device="cuda")
    assert restored.device.type == "cuda"
    assert restored.choose_action(state, mask) in (0, 1)
    trained, rewards = train(TinyEnv(), episodes=3, max_steps=2, batch_size=2, device="cuda")
    assert trained.device.type == "cuda"
    assert len(rewards) == 3
