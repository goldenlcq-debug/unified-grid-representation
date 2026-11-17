import numpy as np
import pytest

from rl_poweradj.model import DeviceRegistry
from rl_poweradj.paths import ProjectPaths
from rl_poweradj.psasp import SolverResult
from rl_poweradj.recovery import ConvergenceRecovery


class SequenceRunner:
    def __init__(self, results):
        self.results = iter(results)

    def run(self, case_dir):
        result = next(self.results)
        (case_dir / "LF.CAL").write_text(
            "1, fresh\n" if result.converged else "0, fresh\n", encoding="gbk"
        )
        return result


class FakeFoundation:
    size = 0
    contract = {"schema_version": 3, "kind": "test-unified-frozen", "trainable": False}

    def encode(self, _case_dir):
        return np.empty(0, dtype=np.float32)


class FirstValidAgent:
    def __init__(self, registry):
        self.metadata = {
            "registry": registry.to_dict(),
            "encoder": FakeFoundation.contract,
        }

    def choose_action(self, _state, action_mask, epsilon=0.0):
        assert epsilon == 0.0
        return int(np.flatnonzero(action_mask)[0])


def test_recovery_rejects_agent_without_unified_foundation():
    source = ProjectPaths.from_root().root / "600nodesample" / "dataset2" / "1"
    registry = DeviceRegistry.from_cases([source])
    with pytest.raises(ValueError, match="Unified frozen encoder is required"):
        ConvergenceRecovery(SequenceRunner([]), FirstValidAgent(registry))


def test_recovery_exports_only_converged_case_without_changing_lf_l5(tmp_path):
    source = ProjectPaths.from_root().root / "600nodesample" / "dataset2" / "1"
    registry = DeviceRegistry.from_cases([source])
    output = tmp_path / "recovered"
    active_inputs = {
        name: (source / name).read_bytes() for name in ("LF.L5", "LF.L6", "LF.NL4")
    }
    result = ConvergenceRecovery(
        SequenceRunner([SolverResult(False, 1.0, 0), SolverResult(True, 0.0, 0)]),
        FirstValidAgent(registry),
        foundation_encoder=FakeFoundation(),
    ).run(source, output_case=output, max_steps=2)
    assert not result.initial_validation.converged
    assert result.final_validation.converged
    assert len(result.steps) == 1
    assert result.output_case == output.resolve()
    assert active_inputs == {name: (output / name).read_bytes() for name in active_inputs}


def test_recovery_failure_never_exports_case(tmp_path):
    source = ProjectPaths.from_root().root / "600nodesample" / "dataset2" / "1"
    registry = DeviceRegistry.from_cases([source])
    output = tmp_path / "must-not-exist"
    recovery = ConvergenceRecovery(
        SequenceRunner([SolverResult(False, 2.0, 0), SolverResult(False, 1.0, 0)]),
        FirstValidAgent(registry),
        foundation_encoder=FakeFoundation(),
    )
    with pytest.raises(RuntimeError, match="no output case was exported"):
        recovery.run(source, output_case=output, max_steps=1)
    assert not output.exists()


def test_recovery_exports_already_converged_case_without_actions(tmp_path):
    source = ProjectPaths.from_root().root / "600nodesample" / "dataset2" / "1"
    registry = DeviceRegistry.from_cases([source])
    output = tmp_path / "already-converged"
    result = ConvergenceRecovery(
        SequenceRunner([SolverResult(True, 0.0, 0)]),
        FirstValidAgent(registry),
        foundation_encoder=FakeFoundation(),
    ).run(source, output_case=output)
    assert result.steps == ()
    assert result.final_validation.converged
    assert output.is_dir()
