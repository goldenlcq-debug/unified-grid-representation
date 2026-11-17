"""Maintained runtime APIs for RL_POWERADJ."""

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from .control import ControlPolicy
from .environment import PowerAdjustmentEnv as DeterministicTaskStateEnvironment
from .foundation import FoundationPowerAdjustmentEnv, UnifiedFrozenEncoder
from .input_validation import InputCaseValidator
from .qvio_reward import QViolationRewardEnhancer
from .model import DeviceRegistry
from .paths import (
    ProjectPaths, case_input_fingerprint, deterministic_partition,
    discover_cases, read_case_catalog,
)
from .psasp import PSASPRunner

# The public environment name is the mandatory unified frozen-encoder path.
PowerAdjustmentEnv = FoundationPowerAdjustmentEnv

__all__ = [
    "DeviceRegistry", "PowerAdjustmentEnv", "FoundationPowerAdjustmentEnv", "UnifiedFrozenEncoder",
    "DeterministicTaskStateEnvironment",
    "InputCaseValidator", "QViolationRewardEnhancer", "ControlPolicy", "ProjectPaths", "PSASPRunner", "case_input_fingerprint",
    "deterministic_partition", "discover_cases", "read_case_catalog",
]
__version__ = "1.4.0"
