import json
import pytest
from rl_poweradj.run_store import atomic_json, file_hash
from rl_poweradj.readiness import verify_readiness
import rl_poweradj.readiness as module


def test_incomplete_readiness_rejected(tmp_path):
    path = tmp_path / 'ready.json'
    atomic_json(path, {'schema_version': 1, 'status': 'ready_for_training', 'test_count': 70, 'evidence': []})
    with pytest.raises(ValueError, match='Incomplete'):
        verify_readiness(tmp_path, path, None, None, tmp_path / 'config')


@pytest.mark.parametrize('change', ['code', 'config', 'runtime'])
def test_readiness_binds_code_recipe_runtime_and_evidence(tmp_path, monkeypatch, change):
    audit, solver, config, evidence = [tmp_path / name for name in ('audit', 'solver', 'config', 'evidence')]
    for path in (audit, solver, config, evidence):
        path.write_bytes(b'fixed')
    monkeypatch.setattr(module, 'runtime_identity', lambda: {'runtime': 1})
    monkeypatch.setattr(module, 'default_foundation_paths', lambda root: (root, root))
    monkeypatch.setattr(module, 'verify_foundation_assets', lambda path: {'encoder': 'fixed'})
    monkeypatch.setattr(module, '_source_digest', lambda path: module.APPROVED_GRIDFM_SHA256)
    release = {'schema_version': 1, 'status': 'ready_for_training', 'test_count': 70,
               'code': {}, 'runtime': {'runtime': 1}, 'assets': {'encoder': 'fixed'},
               'data_audit_sha256': file_hash(audit), 'solver_sha256': file_hash(solver), 'recipe_sha256': file_hash(config),
               'evidence': [{'kind': kind, 'path': 'evidence', 'sha256': file_hash(evidence)}
                            for kind in ('tests', 'cli_acceptance', 'parity', 'prerun', 'data_audit')]}
    ready = tmp_path / 'ready.json'
    atomic_json(ready, release)
    assert verify_readiness(tmp_path, ready, audit, solver, config)['sha256'] == file_hash(ready)
    if change == 'code':
        (tmp_path / 'rl_poweradj').mkdir()
        (tmp_path / 'rl_poweradj/a.py').write_text('changed')
    elif change == 'config':
        config.write_bytes(b'changed')
    else:
        monkeypatch.setattr(module, 'runtime_identity', lambda: {'runtime': 2})
    with pytest.raises(ValueError, match='Stale'):
        verify_readiness(tmp_path, ready, audit, solver, config)
