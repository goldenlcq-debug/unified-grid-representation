from pathlib import Path

import numpy as np
import pytest

from rl_poweradj.environment import PowerAdjustmentEnv
from rl_poweradj.model import DeviceRegistry
from rl_poweradj.psasp import SolverResult, read_records, write_records
from rl_poweradj.qvio_reward import (
    QViolationDiagnostics,
    QViolationRewardEnhancer,
    QViolationSettings,
    compose_qvio_reward,
)


def _diagnostics(value: float, *, available: bool = True):
    if not available:
        return QViolationDiagnostics.unavailable("synthetic failure", applicable=True)
    return QViolationDiagnostics(
        available=True,
        applicable=True,
        nolimit_converged=True,
        q_excess_total=value,
        q_excess_max=value / 2.0,
        n_violating=2,
        n_voltage_controlled=2,
        n_gen_widened=3,
        normalized_total=value,
        normalized_max=value,
        normalized_rms=value,
        violating_fraction=1.0 if value > 0.0 else 0.0,
        composite_score=value,
        normalized_excesses=(value / 2.0, value / 2.0),
    )


def test_qvio_reward_is_monotonic_in_composite_excess_score():
    improved = compose_qvio_reward(_diagnostics(8.0), _diagnostics(2.0))
    worsened = compose_qvio_reward(_diagnostics(2.0), _diagnostics(8.0))
    assert improved.total > -1.0
    assert worsened.total < -1.0
    assert np.isclose(improved.total, -1.0 + 40.0 * np.log10(4.0))
    assert np.isclose(worsened.total, -1.0 - 40.0 * np.log10(4.0))


def test_qvio_reward_falls_back_to_step_cost_when_metric_unavailable():
    result = compose_qvio_reward(_diagnostics(5.0), _diagnostics(5.0, available=False))
    assert result.total == -1.0


class SequenceRunner:
    def __init__(self, results):
        self.results = iter(results)

    def run(self, _case_dir):
        return next(self.results)


class SequenceEnhancer:
    settings = QViolationSettings()
    contract = {"schema_version": 5, "kind": "synthetic_qvio"}

    def __init__(self, diagnostics):
        self.diagnostics = iter(diagnostics)
        self.calls = 0

    def evaluate(self, _case_dir):
        self.calls += 1
        return next(self.diagnostics)

    def require_compatible_solver(self):
        return None


def _case(path: Path) -> Path:
    path.mkdir()
    write_records(path / "LF.L1", [["BUS1"], ["BUS2"]])
    write_records(path / "LF.L2", [
        [1, 1, 1, 1, 0.0, 0.1, 0.0, 0, 0, 0, 0, 0, 0.0, 0.0,
         100.0, "", 0, "S1"],
        [1, 1, 2, 2, 0.0, 0.2, 0.0, 0, 0, 0, 0, 0, 0.0, 0.0,
         100.0, "", 0, "B1"],
    ])
    write_records(path / "LF.L3", [])
    write_records(path / "LF.L5", [
        [1, 1, -2, 1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 2.0, 0.0,
         0, 0, 0, 0.0, 0, "", 0, "PV1"],
        [1, 2, 0, 0.0, 0.0, 1.0, 0.0, 2.0, -2.0, 2.0, -2.0,
         0, 0, 0, 0.0, 0, "", 0, "SLACK"],
    ])
    write_records(path / "LF.L6", [])
    return path


def test_environment_uses_qvio_only_after_failed_psasp(tmp_path):
    case = _case(tmp_path / "case")
    registry = DeviceRegistry.from_cases([case])
    enhancer = SequenceEnhancer([_diagnostics(8.0), _diagnostics(2.0)])
    runner = SequenceRunner([SolverResult(False, 100.0, 0), SolverResult(False, 1.0, 0)])
    with PowerAdjustmentEnv(
        cases=[case], registry=registry, runner=runner,
        qvio_reward_enhancer=enhancer, max_steps=2,
    ) as env:
        _state, info = env.reset()
        action = int(np.flatnonzero(info["action_mask"])[0])
        _state, reward, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    assert reward > 0.0
    assert info["qvio_q_excess_total"] == 2.0
    assert info["reward_qvio_log_improvement"] == pytest.approx(np.log10(4.0))
    assert enhancer.calls == 2


def test_psasp_success_skips_qvio_and_owns_terminal_reward(tmp_path):
    case = _case(tmp_path / "case")
    registry = DeviceRegistry.from_cases([case])
    enhancer = SequenceEnhancer([_diagnostics(8.0)])
    runner = SequenceRunner([SolverResult(False, 100.0, 0), SolverResult(True, 0.0, 0)])
    with PowerAdjustmentEnv(
        cases=[case], registry=registry, runner=runner,
        qvio_reward_enhancer=enhancer, max_steps=2,
    ) as env:
        _state, info = env.reset()
        action = int(np.flatnonzero(info["action_mask"])[0])
        _state, reward, terminated, _truncated, info = env.step(action)
    assert terminated
    assert reward == 499.0
    assert info["reward_total"] == 499.0
    assert enhancer.calls == 1


def test_disabled_qvio_does_not_fall_back_to_psasp_iteration_error(tmp_path):
    case = _case(tmp_path / "case")
    registry = DeviceRegistry.from_cases([case])
    runner = SequenceRunner([SolverResult(False, 100.0, 0), SolverResult(False, 1.0, 0)])
    with PowerAdjustmentEnv(
        cases=[case], registry=registry, runner=runner,
        enable_qvio_reward=False, max_steps=2,
    ) as env:
        _state, info = env.reset()
        action = int(np.flatnonzero(info["action_mask"])[0])
        _state, reward, _terminated, _truncated, _info = env.step(action)
    assert reward == -1.0


class WideningRunner:
    def __init__(self, executable: Path):
        self.executable = executable

    def run(self, case_dir):
        rows = read_records(case_dir / "LF.L5")
        assert all(float(row[7]) == 9999.0 and float(row[8]) == -9999.0 for row in rows)
        write_records(case_dir / "LF.LP5", [
            [1, 1.0, 2.5, 0, "PV1"],
            [2, 0.0, 0.0, 0, "SLACK"],
        ])
        return SolverResult(True, 0.0, 0)


def test_qvio_enhancer_widens_only_a_copy_and_reads_fresh_lp5(tmp_path):
    case = _case(tmp_path / "case")
    executable = tmp_path / "solver.exe"
    executable.write_bytes(b"synthetic solver")
    settings = QViolationSettings()
    enhancer = QViolationRewardEnhancer(WideningRunner(executable), settings)
    before = (case / "LF.L5").read_bytes()
    result = enhancer.evaluate(case)
    assert result.available and result.applicable and result.nolimit_converged
    assert result.n_voltage_controlled == 1
    assert result.n_violating == 1
    assert result.q_excess_total == 1.5
    assert result.q_excess_max == 1.5
    assert result.normalized_total == 0.75
    assert result.normalized_max == 0.75
    assert result.normalized_rms == 0.75
    assert result.violating_fraction == 1.0
    assert result.composite_score == pytest.approx(0.9 * np.log1p(0.75) + 0.1)
    assert (case / "LF.L5").read_bytes() == before


def test_qvio_enhancer_accepts_any_available_psasp_binary(tmp_path):
    executable = tmp_path / "solver.exe"
    executable.write_bytes(b"any available solver")
    enhancer = QViolationRewardEnhancer(WideningRunner(executable))
    enhancer.require_compatible_solver()
    assert enhancer.solver_compatible
