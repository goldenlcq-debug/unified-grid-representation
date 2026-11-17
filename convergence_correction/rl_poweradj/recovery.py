from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .dqn import DQNAgent
from .environment import PowerAdjustmentEnv
from .input_validation import InputCaseValidator, InputValidationResult
from .model import DeviceRegistry
from .psasp import PSASPRunner, SolverResult, convergence_status


@dataclass(frozen=True)
class RecoveryStep:
    device_id: str
    target_status: int
    error: float
    converged: bool


@dataclass(frozen=True)
class RecoveryResult:
    input_validation: InputValidationResult
    initial_validation: SolverResult
    final_validation: SolverResult
    steps: tuple[RecoveryStep, ...]
    output_case: Path | None


class ConvergenceRecovery:
    """Read-only input validation -> AC -> optional DQN reactive recovery."""

    def __init__(
        self,
        runner: PSASPRunner,
        agent: DQNAgent | None,
        input_validator: InputCaseValidator | None = None,
        foundation_encoder=None,
    ):
        self.runner = runner
        self.agent = agent
        self.input_validator = input_validator or InputCaseValidator()
        self.foundation_encoder = foundation_encoder
        if agent is None:
            self.registry = None
        else:
            if foundation_encoder is None:
                raise ValueError("Unified frozen encoder is required for DQN recovery")
            registry_data = agent.metadata.get("registry")
            if registry_data is None:
                raise ValueError("DQN checkpoint does not contain a device registry")
            self.registry = DeviceRegistry.from_dict(registry_data)
            encoder = agent.metadata.get("encoder")
            expected = foundation_encoder.contract
            if encoder != expected:
                raise ValueError("DQN checkpoint is missing or uses a different observation contract")

    @staticmethod
    def _export(source: Path, output: Path | None) -> Path | None:
        if output is None:
            return None
        output = output.resolve()
        if output.exists():
            raise FileExistsError(f"Output case already exists: {output}")
        shutil.copytree(source, output)
        return output

    def run(
        self,
        source_case: Path | str,
        output_case: Path | str | None = None,
        max_steps: int = 100,
        seed: int = 0,
    ) -> RecoveryResult:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        output = None if output_case is None else Path(output_case)
        if output is not None and output.resolve().exists():
            raise FileExistsError(f"Output case already exists: {output.resolve()}")
        trace: list[RecoveryStep] = []
        source = Path(source_case).resolve()
        validation = self.input_validator.validate(source)
        if not validation.active_power_valid:
            raise RuntimeError(
                "RL-only input violates the active-power contract; no AC solve was "
                "attempted and no output case was exported: " + "; ".join(validation.errors)
            )
        registry = self.registry or DeviceRegistry.from_cases([source])
        if self.agent is None:
            environment_type = PowerAdjustmentEnv
            foundation_kwargs = {}
        else:
            from .foundation import FoundationPowerAdjustmentEnv
            environment_type = FoundationPowerAdjustmentEnv
            foundation_kwargs = {"foundation_encoder": self.foundation_encoder}
        with environment_type(
            runner=self.runner, max_steps=max_steps, registry=registry,
            cases=[source], input_validator=self.input_validator,
            enable_qvio_reward=False,
            **foundation_kwargs,
        ) as env:
            # reset performs the first complete AC calculation on an
            # unchanged copy of the supplied active-power plan.
            state, info = env.reset(seed=seed)
            initial = SolverResult(bool(info["converged"]), float(info["error"]), 0)
            final = initial
            if not initial.converged and self.agent is None:
                raise RuntimeError(
                    "The input remains non-converged and no DQN checkpoint was supplied; "
                    "no output case was exported"
                )
            while not final.converged and env.steps < max_steps and info["action_mask"].any():
                action = self.agent.choose_action(state, info["action_mask"], epsilon=0.0)
                state, _reward, terminated, truncated, info = env.step(action)
                final = SolverResult(bool(info["converged"]), float(info["error"]), 0)
                trace.append(RecoveryStep(
                    device_id=str(info["device_id"]),
                    target_status=int(info["target_status"]),
                    error=final.error,
                    converged=final.converged,
                ))
                if terminated or truncated:
                    break
            if not final.converged or not convergence_status(env.case_dir):
                raise RuntimeError(
                    f"DQN convergence recovery failed after {env.steps} steps "
                    f"(final error={final.error:.6g}); fresh LF.CAL does not confirm "
                    "convergence, so no output case was exported"
                )
            copied = self._export(env.case_dir, output)
        return RecoveryResult(validation, initial, final, tuple(trace), copied)
