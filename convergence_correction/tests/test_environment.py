from pathlib import Path
import numpy as np
import pytest

from rl_poweradj.environment import PowerAdjustmentEnv
from rl_poweradj.model import DeviceRegistry, L2_FIELDS, L5_FIELDS
from rl_poweradj.paths import ProjectPaths
from rl_poweradj.psasp import SolverResult, read_records, write_records


class FakeRunner:
    def __init__(self, results=None):
        self.results = iter(results or [SolverResult(True, 0.0, 0)])

    def run(self, _case_dir):
        return next(self.results)


def test_environment_fixed_registry_dimensions_and_isolation():
    with PowerAdjustmentEnv(runner=FakeRunner()) as env:
        observation, info = env.reset(case_index=0)
        assert observation.shape == (env.registry.observation_size,)
        assert env.action_size == env.registry.action_size
        assert info["converged"] is True
        assert info["action_mask"].shape == (env.action_size,)
        assert info["action_mask"].sum() > 0
        assert env.case_dir != env.cases[0]
        load_offset = 3 + 4 * len(env.registry.generator_ids) + 3 * len(env.registry.shunt_ids)
        assert np.count_nonzero(observation[load_offset:]) > 0


def _case(path: Path, generator: str, shunts: list[str]) -> Path:
    path.mkdir()
    l5 = [[1, 1, 0, 0.0, 0.0, 1.0, 0.0, 2.0, -2.0, 2.0, -2.0,
           0, 0, 0, 0.0, 0, "", 0, generator]]
    l2 = [[1, index + 1, index + 1, index + 1, 0.0, 0.1, 0.2, 0, 0,
           0, 0, 0, 0.0, 0.0, 100.0, "", 0, name]
          for index, name in enumerate(shunts)]
    write_records(path / "LF.L5", l5)
    write_records(path / "LF.L2", l2)
    write_records(path / "LF.L1", [["BUS1"], ["BUS2"]])
    write_records(path / "LF.L3", [])
    write_records(path / "LF.L6", [])
    return path


def test_heterogeneous_cases_keep_fixed_identity_contract(tmp_path):
    first = _case(tmp_path / "first", "G1", ["S1"])
    second = _case(tmp_path / "second", "G2", ["S1", "S2"])
    registry = DeviceRegistry.from_cases([first, second])
    runner = FakeRunner([SolverResult(False, 1.0, 0), SolverResult(False, 2.0, 0)])
    with PowerAdjustmentEnv(
        cases=[first, second], registry=registry, runner=runner,
    ) as env:
        state1, info1 = env.reset(case_index=0)
        state2, info2 = env.reset(case_index=1)
    assert state1.shape == state2.shape == (registry.observation_size,)
    assert info1["action_mask"].shape == info2["action_mask"].shape == (registry.action_size,)
    # S2 is absent in the first case, so neither direction is valid there.
    s2 = registry.shunt_ids.index("S2")
    assert not info1["action_mask"][2 * s2:2 * s2 + 2].any()
    assert info2["action_mask"][2 * s2:2 * s2 + 2].any()


def test_invalid_action_consumes_episode_budget():
    with PowerAdjustmentEnv(runner=FakeRunner([SolverResult(False, 1.0, 0)]), max_steps=1) as env:
        env.reset()
        _state, reward, terminated, truncated, info = env.step(-1)
    assert reward < 0
    assert not terminated and truncated
    assert info["invalid_action"]


def test_non_shunt_l2_topology_is_observed_but_not_actionable(tmp_path):
    first = _case(tmp_path / "first", "G1", ["S1"])
    second = _case(tmp_path / "second", "G1", ["S1"])
    branch = [1, 1, 2, 2, 0.0, 0.1, 0.0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, "", 0, "B1"]
    for case, mark in ((first, 1), (second, 0)):
        rows = read_records(case / "LF.L2")
        row = branch.copy()
        row[0] = mark
        rows.append(row)
        write_records(case / "LF.L2", rows)
    registry = DeviceRegistry.from_cases([first, second])
    assert registry.branch_ids == ("B1",)
    assert registry.shunt_ids == ("S1",)
    runner = FakeRunner([SolverResult(False, 1.0, 0), SolverResult(False, 1.0, 0)])
    with PowerAdjustmentEnv(
        cases=[first, second], registry=registry, runner=runner,
    ) as env:
        state_on, _info = env.reset(case_index=0)
        state_off, _info = env.reset(case_index=1)
    assert env.action_size == 2  # B1 is a disturbance, not an allowed control.
    branch_offset = (
        3 + 4 * len(registry.generator_ids) + 3 * len(registry.shunt_ids)
        + 4 * len(registry.load_ids)
    )
    assert np.allclose(state_on[branch_offset:branch_offset + 2], (1.0, 1.0))
    assert np.allclose(state_off[branch_offset:branch_offset + 2], (1.0, 0.0))
    assert not np.array_equal(state_on, state_off)


def test_environment_masks_every_visited_configuration(tmp_path):
    case = _case(tmp_path / "one-shunt", "G1", ["S1"])
    registry = DeviceRegistry.from_cases([case])
    runner = FakeRunner([
        SolverResult(False, 10.0, 0),
        SolverResult(False, 1.0, 0),
    ])
    with PowerAdjustmentEnv(
        cases=[case], registry=registry, runner=runner,
        max_steps=3,
    ) as env:
        _state, info = env.reset()
        action = int(np.flatnonzero(info["action_mask"])[0])
        _state, _reward, _terminated, _truncated, info = env.step(action)
        assert not info["action_mask"].any()


def test_environment_rejects_active_invalid_case_without_adjusting_it():
    source = ProjectPaths.from_root().root / "600nodesample" / "dataset2" / "112"
    registry = DeviceRegistry.from_cases([source])
    before = {name: (source / name).read_bytes() for name in ("LF.L5", "LF.L6", "LF.NL4")}
    with PowerAdjustmentEnv(
        cases=[source], registry=registry,
        runner=FakeRunner([SolverResult(False, 1.0, 0)]),
    ) as env, pytest.raises(RuntimeError, match="RL-only input"):
        env.reset(case_index=0)
    assert before == {name: (source / name).read_bytes() for name in before}
