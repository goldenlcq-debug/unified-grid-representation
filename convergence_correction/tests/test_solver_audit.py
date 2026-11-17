from types import SimpleNamespace
import json

import pytest

from rl_poweradj.psasp import PSASPRunner
from rl_poweradj.solver_lock import SolverLock


def test_solver_lock_rejects_second_owner_and_releases(tmp_path):
    path = tmp_path / 'license.lock'
    with SolverLock(path):
        with pytest.raises(RuntimeError, match='busy'):
            with SolverLock(path):
                pytest.fail('Concurrent owner accepted')
    with SolverLock(path):
        pass


def test_dongle_error_detected_without_cal_and_failure_is_preserved(tmp_path, monkeypatch):
    case = tmp_path / 'case'
    case.mkdir()
    executable = tmp_path / 'solver.exe'
    executable.write_bytes(b'stub')
    audit = tmp_path / 'audit'
    def run(*args, **kwargs):
        (case / 'lfreport.lis').write_text('软件狗不响应，请检查加密狗!', encoding='gbk')
        return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')
    monkeypatch.setattr('rl_poweradj.psasp.subprocess.run', run)
    with pytest.raises(RuntimeError, match='license unavailable'):
        PSASPRunner(executable, timeout=30, audit_dir=audit).run(case)
    directory = next(audit.iterdir())
    record = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    assert record['status'] == 'solver_error'
    assert record['seconds'] < 5
    assert (directory / 'failure-case/lfreport.lis').is_file()


def test_generic_error_heading_is_not_license_failure(tmp_path, monkeypatch):
    case = tmp_path / 'case'
    case.mkdir()
    executable = tmp_path / 'solver.exe'
    executable.write_bytes(b'stub')
    def run(*args, **kwargs):
        (case / 'lfreport.lis').write_text('Operation Error Information\n0 BUS1 0.0\n', encoding='gbk')
        (case / 'LF.CAL').write_text('-1,\n', encoding='gbk')
        return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')
    monkeypatch.setattr('rl_poweradj.psasp.subprocess.run', run)
    result = PSASPRunner(executable, audit_dir=tmp_path / 'audit').run(case)
    assert result.converged is False
    assert result.error == 0
