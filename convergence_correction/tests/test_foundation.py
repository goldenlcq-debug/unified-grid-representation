import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl_poweradj.dqn import DQNAgent, _validate_metadata_contract, train
from rl_poweradj.foundation import (
    FOUNDATION_FEATURE_SIZE,
    FoundationPowerAdjustmentEnv,
    UnifiedFrozenEncoder,
    verify_foundation_assets,
)
from rl_poweradj.model import DeviceRegistry
from rl_poweradj.paths import ProjectPaths
from rl_poweradj.psasp import SolverResult


class FakeFoundation:
    size = 2
    contract = {
        "schema_version": 3,
        "kind": "test-unified-frozen-encoder",
        "trainable": False,
    }

    def encode(self, _path):
        return np.asarray([3.0, 4.0], dtype=np.float32)


def test_foundation_environment_appends_tokens(monkeypatch):
    import rl_poweradj.foundation as module

    monkeypatch.setattr(
        module, "encode_observation", lambda *args, **kwargs: np.asarray([1.0, 2.0], dtype=np.float32)
    )
    env = object.__new__(FoundationPowerAdjustmentEnv)
    env.registry = SimpleNamespace(observation_size=2)
    env.foundation_encoder = FakeFoundation()
    env.l5_rows = env.l2_rows = env.l6_rows = env.l3_rows = env.dc_plans = []
    env.previous_error = env.prior_error = None
    env.steps, env.max_steps, env.case_dir = 0, 10, Path("case")
    assert env.observation_size == 4
    assert np.array_equal(env._observation(), [1, 2, 3, 4])


def test_foundation_contract_rejects_deterministic_only_checkpoint():
    registry = {"schema": 2}
    agent = DQNAgent(
        4,
        2,
        device="cpu",
        metadata={
            "registry": registry,
            "encoder": {
                "kind": "deterministic_observation_encoder",
                "schema_version": 2,
                "trainable": False,
            },
        },
    )
    with pytest.raises(ValueError, match="observation contract differs"):
        _validate_metadata_contract(
            agent,
            {"registry": registry, "encoder": FakeFoundation.contract},
        )


def test_tampered_foundation_assets_are_rejected(tmp_path):
    (tmp_path / "v1.pt").write_bytes(b"not the canonical encoder")
    (tmp_path / "normalizer.npz").write_bytes(b"not the canonical normalizer")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_foundation_assets(tmp_path)


def test_real_foundation_assets_and_bridge_checkpoint():
    root = Path(__file__).resolve().parents[1]
    encoder = UnifiedFrozenEncoder.from_project(root, device="cpu")
    assert encoder.size == FOUNDATION_FEATURE_SIZE
    assert not any(parameter.requires_grad for parameter in encoder.bundle.encoder.parameters())
    bridge = DQNAgent.load(root / "artifacts/models/dqn-best-foundation.pt", "cpu")
    assert bridge.observation_size == 20168
    assert bridge.action_size == 302
    assert bridge.metadata["encoder"] == encoder.contract
    assert bridge.metadata["foundation_usage"]["foundation_effective"] is False
    weight = bridge.online.state_dict()["layers.0.weight"]
    assert np.count_nonzero(weight[:, 17096:].numpy()) == 0


def test_real_foundation_is_frozen_during_task_network_training():
    root = Path(__file__).resolve().parents[1]
    paths = ProjectPaths.from_root(root)
    encoder = UnifiedFrozenEncoder.from_project(root, device="cpu")
    source = paths.cases("train")[0]
    from rl_poweradj.control import ControlPolicy
    registry = DeviceRegistry.from_cases([source], ControlPolicy.load(paths.control_policy))

    class FakeSolver:
        def __init__(self):
            self.calls = 0

        def run(self, _case_dir):
            self.calls += 1
            return SolverResult(self.calls % 2 == 0, 0.0 if self.calls % 2 == 0 else 1.0, 0)

    before = [parameter.detach().clone() for parameter in encoder.bundle.encoder.parameters()]
    with FoundationPowerAdjustmentEnv(
        paths=paths,
        cases=[source],
        registry=registry,
        foundation_encoder=encoder,
        runner=FakeSolver(),
        max_steps=2,
        enable_qvio_reward=False,
    ) as env:
        agent, rewards = train(env, episodes=1, max_steps=2, batch_size=1, device="cpu", seed=0)
    assert rewards == [499.0]
    assert agent.updates == 1
    assert agent.representation_audit['token_gradient_norm'] > 0
    assert agent.representation_audit['token_update_norm'] > 0
    assert agent.representation_audit['encoder_gradients_absent']
    assert agent.metadata["encoder"] == encoder.contract
    assert not any(parameter.requires_grad for parameter in encoder.bundle.encoder.parameters())
    assert all(torch.equal(left, right) for left, right in zip(before, encoder.bundle.encoder.parameters()))
