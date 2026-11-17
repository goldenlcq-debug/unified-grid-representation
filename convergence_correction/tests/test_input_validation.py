import shutil

import numpy as np

from rl_poweradj.input_validation import InputCaseValidator
from rl_poweradj.model import (
    DeviceRegistry, L2_FIELDS, L3_FIELDS, L5_FIELDS, L6_FIELDS,
    encode_observation, read_dc_plans, read_table,
)
from rl_poweradj.paths import ProjectPaths


def _observation(case, registry):
    l3 = case / "LF.L3"
    l6 = case / "LF.L6"
    return encode_observation(
        registry,
        read_table(case / "LF.L5", L5_FIELDS),
        read_table(case / "LF.L2", L2_FIELDS),
        1.25,
        read_table(l6, L6_FIELDS) if l6.is_file() else [],
        read_table(l3, L3_FIELDS) if l3.is_file() else [],
        read_dc_plans(case),
        previous_error=2.5,
        step_fraction=0.25,
    )


def test_rl_only_validator_is_read_only_and_encoder_contract_stays_frozen(tmp_path):
    source = ProjectPaths.from_root().root / "600nodesample" / "dataset2" / "1"
    case = tmp_path / "case"
    shutil.copytree(source, case)
    registry = DeviceRegistry.from_cases([case])
    before_files = {
        name: (case / name).read_bytes()
        for name in ("LF.L2", "LF.L5", "LF.L6", "LF.NL4")
    }
    before_observation = _observation(case, registry)

    result = InputCaseValidator().validate(case)

    assert result.active_power_valid
    assert before_files == {name: (case / name).read_bytes() for name in before_files}
    assert registry.schema_version == 2
    assert registry.to_dict() == DeviceRegistry.from_dict(registry.to_dict()).to_dict()
    assert np.array_equal(before_observation, _observation(case, registry))


def test_rl_only_validator_rejects_case_that_previously_needed_active_adjustment():
    source = ProjectPaths.from_root().root / "600nodesample" / "dataset2" / "112"
    result = InputCaseValidator().validate(source)
    assert not result.active_power_valid
    assert any("no online Type=0" in error for error in result.errors)


def test_validator_accepts_fixed_generator_schedule(tmp_path):
    from rl_poweradj.psasp import write_records

    case = tmp_path / "fixed"
    case.mkdir()
    write_records(case / "LF.L1", [["B1"]])
    write_records(case / "LF.L2", [])
    write_records(case / "LF.L3", [])
    write_records(case / "LF.L5", [[
        1, 1, 0, 1.5, 0.0, 1.0, 0.0, 1.0, -1.0, 1.5, 1.5,
        0, 0, 0, 0.0, 0, "", 0, "G1",
    ]])
    write_records(case / "LF.L6", [])
    result = InputCaseValidator().validate(case)
    assert result.active_power_valid
