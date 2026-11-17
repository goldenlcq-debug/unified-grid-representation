from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from rl_poweradj.checkpoints import select_latest
from rl_poweradj.dqn import DQNAgent, ReplayBuffer, CaseEvaluation, EvaluationMetrics, _validation_score
from rl_poweradj.representation_audit import token_sensitivity, network_digest
from rl_poweradj.data_preflight import input_card_hashes, verify_audit
from rl_poweradj.run_store import atomic_json


def result(cases):
    return EvaluationMetrics(len(cases), sum(c.success for c in cases), 0, 0, 0, 0, 0, tuple(cases))


def test_lexicographic_selection_prioritizes_required_success_before_steps():
    slow_success = result([CaseEvaluation(0, 'a', True, 350, 150, 1)])
    fast_failure = result([CaseEvaluation(0, 'a', False, -1, 1, 0)])
    fast_success = result([CaseEvaluation(0, 'a', True, 499, 1, 100)])
    assert _validation_score(slow_success) > _validation_score(fast_failure)
    assert _validation_score(fast_success) > _validation_score(slow_success)
    initial = CaseEvaluation(1, 'b', True, 0, 0, 0)
    assert _validation_score(result([initial, fast_failure.cases[0]])) == _validation_score(fast_failure)


def test_token_sensitivity_does_not_consume_rng_or_change_weights():
    network = torch.nn.Linear(4, 2)
    states = torch.tensor([[1., 2., 3., 4.], [1., 2., 8., 9.]])
    masks = torch.ones(2, 2, dtype=torch.bool)
    before = network_digest(network)
    rng = torch.get_rng_state().clone()
    report = token_sensitivity(network, states, masks, token_size=2)
    assert report['zero_q_max_abs'] > 0
    assert report['permuted_q_max_abs'] > 0
    assert network_digest(network) == before
    assert torch.equal(rng, torch.get_rng_state())


def test_partial_checkpoint_manifest_commit_preserves_previous(tmp_path, monkeypatch):
    agent = DQNAgent(4, 2, device='cpu')
    replay = ReplayBuffer()
    old = tmp_path / 'episode-10.pt'
    agent.save(old, episode=10, replay=replay)
    def interrupt(*args, **kwargs):
        raise OSError('injected manifest commit interruption')
    monkeypatch.setattr('rl_poweradj.dqn.atomic_json', interrupt)
    with pytest.raises(OSError):
        agent.save(tmp_path / 'episode-20.pt', episode=20, replay=replay)
    selected, rejected = select_latest(tmp_path)
    assert selected == old
    assert len(rejected) == 1


def test_extended_card_identity_catches_static_card_changes(tmp_path):
    case = tmp_path / 'sample_1_1'
    case.mkdir()
    (case / 'LF.L1').write_bytes(b'input')
    (case / 'ST.S5').write_bytes(b'dynamic input')
    (case / 'LF.LP1').write_bytes(b'result')
    registry = SimpleNamespace(to_dict=lambda: {'id': 1})
    paths = SimpleNamespace(reactive_cases=lambda s: [case] if s == 'train' else [])
    audit = tmp_path / 'audit.json'
    atomic_json(audit, {'schema_version': 1, 'status': 'passed', 'errors': [], 'registry': {'id': 1},
        'dataset_digest': 'digest', 'records': [{'case': case.name, 'split': 'train', 'input_cards': input_card_hashes(case)}]})
    (case / 'LF.LP1').write_bytes(b'another result')
    verify_audit(paths, audit, registry)
    (case / 'ST.S5').write_bytes(b'changed input')
    with pytest.raises(ValueError, match='Inputs changed'):
        verify_audit(paths, audit, registry)
