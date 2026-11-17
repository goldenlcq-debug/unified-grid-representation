from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .input_validation import InputCaseValidator
from .model import (
    DeviceRegistry, L2_FIELDS, L3_FIELDS, L5_FIELDS, L6_FIELDS,
    encode_observation, observation_feature_names, read_dc_plans, read_generator_table, read_table,
)
from .psasp import CaseWorkspace


def audit_observations(
    cases: list[Path],
    registry: DeviceRegistry,
    flow_manifest: Path,
) -> dict:
    payload = json.loads(flow_manifest.read_text(encoding="utf-8"))
    errors = {
        str(item["case"]): item.get("first_ac_error") for item in payload["records"]
    }
    observations = []
    validator = InputCaseValidator()
    for source in cases:
        with CaseWorkspace(source) as workspace:
            result = validator.validate(workspace)
            if not result.active_power_valid:
                continue
            l2 = read_table(workspace / "LF.L2", L2_FIELDS)
            l3 = read_table(workspace / "LF.L3", L3_FIELDS)
            l5 = read_generator_table(workspace / "LF.L5")
            l6 = read_table(workspace / "LF.L6", L6_FIELDS)
            observations.append(encode_observation(
                registry, l5, l2, errors.get(source.name), l6, l3,
                read_dc_plans(workspace), previous_error=None, step_fraction=0.0,
            ))
    matrix = np.stack(observations)
    names = observation_feature_names(registry)
    spread = np.ptp(matrix, axis=0)
    varying = np.flatnonzero(spread > 1e-8)
    return {
        "schema_version": registry.schema_version,
        "case_count": len(matrix),
        "observation_size": registry.observation_size,
        "varying_feature_count": int(len(varying)),
        "constant_feature_count": int(registry.observation_size - len(varying)),
        "varying_features": [
            {
                "index": int(index), "name": names[index],
                "minimum": float(matrix[:, index].min()),
                "maximum": float(matrix[:, index].max()),
            }
            for index in varying
        ],
    }
