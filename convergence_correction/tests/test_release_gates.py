from argparse import Namespace
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from rl_poweradj import cli
from rl_poweradj.dqn import DQNAgent, ReplayBuffer, Transition, TrainMetrics, train
from rl_poweradj.foundation import UnifiedFrozenEncoder, FoundationPowerAdjustmentEnv, default_foundation_paths
from rl_poweradj.paths import ProjectPaths
from rl_poweradj.model import DeviceRegistry


@pytest.mark.parametrize('exception,status', [(RuntimeError('solver fault'), 'failed'), (KeyboardInterrupt(), 'interrupted')])
def test_training_wrapper_records_failure_without_erasing_progress(tmp_path, monkeypatch, exception, status):
    args = Namespace(run_dir=tmp_path)
    def fail(paths, args):
        args._run_accepted = True
        (tmp_path / 'progress.json').write_text('{"completed_episodes":12}')
        raise exception
    monkeypatch.setattr(cli, '_train_command', fail)
    with pytest.raises(type(exception)):
        cli.train_command(None, args)
    record = json.loads((tmp_path / 'progress.json').read_text())
    assert record['status'] == status and record['completed_episodes'] == 12


@pytest.fixture(scope='module')
def encoder():
    return UnifiedFrozenEncoder.from_project(Path(__file__).resolve().parents[1], 'cpu')


def test_same_product_wrong_token_layout_is_rejected(encoder, monkeypatch):
    import gridfm.reason.b_line.encoder_io as interface
    old = (encoder.bundle.config.n_global, encoder.bundle.config.d)
    encoder.bundle.config.n_global, encoder.bundle.config.d = 8, 384
    monkeypatch.setattr(interface, 'FrozenBEncoder', lambda *a, **kw: encoder.bundle)
    assets, source = default_foundation_paths(ProjectPaths.from_root().root)
    try:
        with pytest.raises(ValueError, match='contract size'):
            UnifiedFrozenEncoder(assets, source)
    finally:
        encoder.bundle.config.n_global, encoder.bundle.config.d = old


def test_deleting_bridge_flag_does_not_make_it_resumable(encoder):
    paths = ProjectPaths.from_root()
    agent = DQNAgent.load(paths.model_dir / 'dqn-best-foundation.pt', 'cpu')
    agent.metadata.pop('foundation_usage')
    registry = DeviceRegistry.from_dict(agent.metadata['registry'])
    with FoundationPowerAdjustmentEnv(paths=paths, cases=[paths.cases('train')[0]], registry=registry,
                                      foundation_encoder=encoder, enable_qvio_reward=False) as env:
        env.reset = lambda **kw: pytest.fail('Rejected checkpoint reached PSASP reset')
        with pytest.raises(ValueError, match='freshly trained'):
            train(env, agent=agent, replay=ReplayBuffer(), metrics=TrainMetrics(), start_episode=1)


def test_optimizer_cannot_include_frozen_encoder_parameters(encoder):
    agent = DQNAgent(20168, 302, device='cpu')
    agent._foundation = encoder
    agent.optimizer.add_param_group({'params': list(encoder.bundle.encoder.parameters())})
    replay = ReplayBuffer()
    state = np.zeros(20168, dtype=np.float32)
    replay.add(Transition(state, 0, 1, state, True, np.ones(302, dtype=bool)))
    with pytest.raises(RuntimeError, match='isolation'):
        agent.learn(replay, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_production_cuda_encoder_enables_deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        value = UnifiedFrozenEncoder.from_project(ProjectPaths.from_root().root, 'cuda')
        assert torch.are_deterministic_algorithms_enabled()
        assert not any(p.requires_grad for p in value.bundle.encoder.parameters())
    finally:
        torch.use_deterministic_algorithms(previous)
