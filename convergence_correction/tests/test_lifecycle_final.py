import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_poweradj.dqn import DQNAgent, train
from rl_poweradj.psasp import PSASPRunner, MissingFreshOutputs, SolverResult
from rl_poweradj.lifecycle import request_pause, set_status
from test_training_resume_equivalence import RandomObservationEnv


def test_future_orphan_does_not_displace_fresh_recovery_point(tmp_path):
    from rl_poweradj.dqn import _save_periodic_checkpoint, ReplayBuffer, TrainMetrics
    agent = DQNAgent(4, 2, device='cpu')
    replay = ReplayBuffer()
    agent.save(tmp_path / 'dqn-episode-9.pt', episode=9, replay=replay)
    (tmp_path / 'dqn-episode-99.pt').write_bytes(b'orphan')
    saved = _save_periodic_checkpoint(agent, tmp_path, 10, replay, TrainMetrics(), 1)
    assert saved.is_file()
    assert not (tmp_path / 'dqn-episode-9.pt').exists()
    from rl_poweradj.checkpoints import select_latest
    selected, _ = select_latest(tmp_path)
    assert selected == saved


def test_incomplete_validation_cannot_select_checkpoint(tmp_path):
    from rl_poweradj.dqn import _record_validation, TrainMetrics, EvaluationPaused
    env = RandomObservationEnv()
    resets = []
    original = env.reset
    def reset(**kwargs):
        resets.append(1)
        return original(**kwargs)
    env.reset = reset
    env._pause_requested = lambda: len(resets) >= 1
    metrics = TrainMetrics()
    agent = DQNAgent(4, 2, device='cpu')
    with pytest.raises(EvaluationPaused):
        _record_validation(agent, env, metrics, 1, 2, 3, tmp_path / 'best.pt', full=True)
    assert not metrics.full_validation_history
    assert not (tmp_path / 'best.pt').exists()


def test_last_episode_pause_keeps_complete_replay(tmp_path):
    agent, _ = train(RandomObservationEnv(), episodes=1, max_steps=2, batch_size=2, seed=13, device='cpu',
                      checkpoint_dir=tmp_path, checkpoint_interval=10, stop_requested=lambda: True)
    restored, replay, metrics, episode = DQNAgent.load_checkpoint(tmp_path / 'dqn-episode-1.pt', 'cpu')
    assert agent.training_paused and episode == metrics.completed_episodes == 1
    assert len(replay) == 2 and restored.updates == 1


def test_pause_checkpoint_then_resume_matches_continuous(tmp_path):
    baseline, _ = train(RandomObservationEnv(), episodes=4, max_steps=2, batch_size=2, seed=13, device='cpu')
    paused, rewards = train(RandomObservationEnv(), episodes=4, max_steps=2, batch_size=2, seed=13, device='cpu',
                             checkpoint_dir=tmp_path, checkpoint_interval=10, stop_requested=lambda: True)
    assert paused.training_paused and len(rewards) == 1
    path = tmp_path / 'dqn-episode-1.pt'
    restored, replay, metrics, episode = DQNAgent.load_checkpoint(path, 'cpu')
    resumed, _ = train(RandomObservationEnv(), episodes=3, max_steps=2, batch_size=2, seed=13, device='cpu',
                        agent=restored, replay=replay, metrics=metrics, start_episode=episode)
    assert baseline.training_metrics.to_dict() == resumed.training_metrics.to_dict()
    assert all(torch.equal(p, resumed.online.state_dict()[k]) for k, p in baseline.online.state_dict().items())


def test_status_preserves_progress_and_pause_request(tmp_path):
    (tmp_path / 'run.json').write_text('{}')
    set_status(tmp_path, 'running', completed_episodes=12)
    set_status(tmp_path, 'failed', error='missing output')
    record = json.loads((tmp_path / 'progress.json').read_text())
    assert record['completed_episodes'] == 12 and record['status'] == 'failed'
    request_pause(tmp_path)
    assert (tmp_path / 'pause.request').is_file()


def test_only_missing_outputs_are_retried(tmp_path, monkeypatch):
    executable = tmp_path / 'solver.exe'
    executable.write_bytes(b'stub')
    case = tmp_path / 'case'
    case.mkdir()
    runner = PSASPRunner(executable, audit_dir=tmp_path / 'audit', retries=1)
    calls = []
    def first_missing(directory, record):
        calls.append(1)
        if len(calls) == 1:
            raise MissingFreshOutputs('transient missing outputs')
        return SolverResult(False, 1.0, 0)
    monkeypatch.setattr('rl_poweradj.psasp.solver_processes_present', lambda: False)
    monkeypatch.setattr(runner, '_run_once', first_missing)
    assert not runner.run(case).converged
    assert len(calls) == 2
    def license_error(directory, record):
        calls.append(1)
        raise RuntimeError('license unavailable')
    monkeypatch.setattr(runner, '_run_once', license_error)
    with pytest.raises(RuntimeError, match='license'):
        runner.run(case)
    assert len(calls) == 3
    def process_timeout(directory, record):
        calls.append(1)
        raise TimeoutError('process timeout')
    monkeypatch.setattr(runner, '_run_once', process_timeout)
    with pytest.raises(TimeoutError):
        runner.run(case)
    assert len(calls) == 4


def test_residual_process_blocks_retry(tmp_path, monkeypatch):
    exe = tmp_path / 'solver.exe'
    exe.write_bytes(b'stub')
    case = tmp_path / 'case'
    case.mkdir()
    runner = PSASPRunner(exe, audit_dir=tmp_path / 'audit')
    checks = iter([False, True])
    monkeypatch.setattr('rl_poweradj.psasp.solver_processes_present', lambda: next(checks))
    def missing(*args):
        raise MissingFreshOutputs('missing')
    monkeypatch.setattr(runner, '_run_once', missing)
    with pytest.raises(MissingFreshOutputs):
        runner.run(case)
    assert len(list((tmp_path / 'audit').iterdir())) == 1
