import json
from pathlib import Path

import pytest
import torch

from rl_poweradj.dqn import DQNAgent, train
from rl_poweradj.environment import PowerAdjustmentEnv
from rl_poweradj.run_store import prepare_run
from rl_poweradj.foundation import UnifiedFrozenEncoder


def test_fresh_run_never_overwrites_and_resume_checks_contract(tmp_path):
    directory = tmp_path / 'run'
    identity = prepare_run(directory, {'seed': 0})
    with pytest.raises(FileExistsError):
        prepare_run(directory, {'seed': 0})
    assert prepare_run(directory, {'seed': 0}, resume=True) == identity
    with pytest.raises(ValueError, match='contract'):
        prepare_run(directory, {'seed': 1}, resume=True)


def test_production_base_env_rejected_before_reset():
    env = object.__new__(PowerAdjustmentEnv)
    env.reset = lambda **kwargs: pytest.fail('reset must not run')
    with pytest.raises(ValueError, match='GridEncoder'):
        train(env)


def test_bridge_rejected_by_low_level_trainer():
    agent = DQNAgent(4, 2, device='cpu', metadata={'foundation_usage': {'inference_only': True}})
    with pytest.raises(ValueError, match='Inference-only'):
        train(object(), agent=agent)


def test_checkpoint_corruption_rejected_before_torch_load(tmp_path, monkeypatch):
    agent = DQNAgent(4, 2, device='cpu')
    checkpoint = tmp_path / 'checkpoint.pt'
    agent.save(checkpoint)
    with checkpoint.open('ab') as stream:
        stream.write(b'corrupt')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('must verify first'))
    with pytest.raises(ValueError, match='integrity'):
        DQNAgent.load(checkpoint, 'cpu')


@pytest.fixture(scope='module')
def real_encoder():
    return UnifiedFrozenEncoder.from_project(Path(__file__).resolve().parents[1], 'cpu')


def test_encoder_train_mode_and_normalizer_drift_rejected(real_encoder):
    encoder = real_encoder
    encoder.bundle.encoder.train()
    try:
        with pytest.raises(RuntimeError, match='eval'):
            encoder.assert_runtime_frozen()
    finally:
        encoder.bundle.encoder.eval()
    mean = encoder.bundle.normalizer.node_mean
    original = mean.copy()
    mean[0] += 1
    try:
        with pytest.raises(RuntimeError, match='normalizer'):
            encoder.verify_frozen()
    finally:
        mean[:] = original
    encoder.verify_frozen()


def test_missing_state_view_ignores_solver_outputs(real_encoder):
    from rl_poweradj.paths import ProjectPaths
    from rl_poweradj.psasp import CaseWorkspace
    from gridfm.reason.b_line.encoder_io import prepare_missing_state_sample
    from gridfm.schema.fields import Group, NODE_TABLE, BRANCH_TABLE
    from gridfm.schema.sample import group_bit_index
    import numpy as np
    source = ProjectPaths.from_root().cases('train')[0]
    with CaseWorkspace(source) as case:
        first = real_encoder.encode(case)
        for name in ('LF.LP1', 'LF.LP2', 'LF.LP5', 'LF.CAL'):
            (case / name).write_text('unparseable solver output', encoding='ascii')
        second = real_encoder.encode(case)
        assert np.array_equal(first, second)
        raw = real_encoder.assemble(case, sample_id='test', family_id='test')
        prepared = prepare_missing_state_sample(raw, real_encoder.bundle.normalizer)
        state = group_bit_index(Group.STATE)
        assert prepared.node_missing[:, state].all()
        assert prepared.branch_missing[:, state].all()
        assert not prepared.node_feat[:, NODE_TABLE.group_slice(Group.STATE)].any()
        assert not prepared.branch_feat[:, BRANCH_TABLE.group_slice(Group.STATE)].any()


def test_shunt_mark_is_task_condition_under_missing_state_protocol(real_encoder):
    from rl_poweradj.paths import ProjectPaths
    from rl_poweradj.psasp import CaseWorkspace
    from rl_poweradj.model import L2_FIELDS, read_table, write_table, is_shunt
    from gridfm.schema.fields import BRANCH_TABLE
    import numpy as np
    source = ProjectPaths.from_root().cases('train')[0]
    with CaseWorkspace(source) as case:
        raw_before = real_encoder.assemble(case, sample_id='same', family_id='same')
        token_before = real_encoder.encode(case)
        rows = read_table(case / 'LF.L2', L2_FIELDS)
        row = next(r for r in rows if is_shunt(r))
        old_mark = int(row['Mark'])
        row['Mark'] = 1 - old_mark
        write_table(case / 'LF.L2', rows)
        assert int(next(r for r in read_table(case / 'LF.L2', L2_FIELDS) if is_shunt(r))['Mark']) != old_mark
        raw_after = real_encoder.assemble(case, sample_id='same', family_id='same')
        assert not raw_before.branch_feat[:, BRANCH_TABLE.offset_of('mark')].any()
        assert not raw_after.branch_feat[:, BRANCH_TABLE.offset_of('mark')].any()
        assert np.array_equal(token_before, real_encoder.encode(case))
