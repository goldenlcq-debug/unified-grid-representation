import numpy as np
import pytest

from rl_poweradj.control import ControlPolicy
from rl_poweradj.model import DeviceRegistry, encode_observation


def test_control_policy_expands_ranges_and_rejects_unknown_devices():
    policy = ControlPolicy.from_dict({
        "schema_version": 1,
        "ranges": [{"prefix": "S", "start": 1, "end": 2, "allowed_targets": [0, 1]}],
    })
    assert [(item.device_id, item.target_status) for item in policy.actions] == [
        ("S1", 0), ("S1", 1), ("S2", 0), ("S2", 1),
    ]
    policy.validate(["S1", "S2"])
    with pytest.raises(ValueError, match="unknown shunt"):
        policy.validate(["S1"])


def test_observation_v2_log_error_retains_large_error_resolution():
    registry = DeviceRegistry((), (), bus_count=0)
    low = encode_observation(registry, (), (), 20.0)
    high = encode_observation(registry, (), (), 1000.0)
    assert registry.observation_size == 3
    assert 0.0 < low[0] < high[0] < 1.0


def test_observation_v2_checkpoint_registry_rejects_old_schema():
    with pytest.raises(ValueError, match="obsolete observation"):
        DeviceRegistry.from_dict({"generator_ids": [], "shunt_ids": []})
