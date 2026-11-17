from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .control import ControlPolicy
from .input_validation import InputCaseValidator
from .qvio_reward import (
    QViolationDiagnostics,
    QViolationRewardComponents,
    QViolationRewardEnhancer,
    compose_qvio_reward,
)
from .model import (
    DeviceRegistry,
    L2_FIELDS,
    L3_FIELDS,
    L5_FIELDS,
    L6_FIELDS,
    DCPlan,
    Record,
    encode_observation,
    is_shunt,
    read_dc_plans,
    read_generator_table,
    read_table,
    write_table,
)
from .paths import ProjectPaths
from .psasp import CaseWorkspace, PSASPRunner, write_records


class PowerAdjustmentEnv:
    """Fixed-contract PSASP shunt-control environment.

    Device identities, rather than file row numbers, define observations and
    actions. Every episode operates on an isolated case copy.
    """

    def __init__(
        self,
        split: str = "train",
        paths: ProjectPaths | None = None,
        runner: PSASPRunner | None = None,
        max_steps: int = 100,
        success_reward: float = 500.0,
        invalid_action_penalty: float = -25.0,
        step_cost: float = 1.0,
        initial_perturbations: int = 0,
        registry: DeviceRegistry | None = None,
        cases: list[Path] | None = None,
        input_validator: InputCaseValidator | None = None,
        qvio_reward_enhancer: QViolationRewardEnhancer | None = None,
        enable_qvio_reward: bool = True,
    ):
        self.paths = paths or ProjectPaths.from_root()
        explicit_cases = cases is not None
        self.cases = list(cases if cases is not None else self.paths.cases(split))
        if not self.cases:
            raise ValueError("At least one case is required")
        if registry is None:
            policy = ControlPolicy.load(self.paths.control_policy)
            registry_sources = self.cases if explicit_cases else self.paths.registry_cases(split)
            registry = DeviceRegistry.from_cases(registry_sources, policy)
        self.registry = registry
        self.runner = runner or PSASPRunner(self.paths.solver)
        self.input_validator = input_validator or InputCaseValidator()
        self.qvio_reward_enhancer = qvio_reward_enhancer
        if (
            self.qvio_reward_enhancer is None
            and enable_qvio_reward
            and isinstance(self.runner, PSASPRunner)
        ):
            self.qvio_reward_enhancer = QViolationRewardEnhancer(self.runner)
        self.max_steps = int(max_steps)
        self.success_reward = float(success_reward)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.step_cost = float(step_cost)
        self.initial_perturbations = int(initial_perturbations)
        self._workspace: CaseWorkspace | None = None
        self.case_dir: Path | None = None
        self.l2_rows: list[Record] = []
        self.l3_rows: list[Record] = []
        self.l5_rows: list[Record] = []
        self.l6_rows: list[Record] = []
        self.dc_plans: list[DCPlan] = []
        self.steps = 0
        self.previous_error: float | None = None
        self.prior_error: float | None = None
        self.previous_qvio: QViolationDiagnostics | None = None
        self.current_case_index = 0
        self._visited_configurations: set[tuple[int, ...]] = set()
        self._rng = np.random.default_rng()

    @property
    def observation_size(self) -> int:
        return self.registry.observation_size

    @property
    def action_size(self) -> int:
        return self.registry.action_size

    @property
    def reward_contract(self) -> dict[str, object]:
        if self.qvio_reward_enhancer is None:
            return {"schema_version": 5, "kind": "psasp_terminal_step_cost_only"}
        return self.qvio_reward_enhancer.contract

    def require_reward_preflight(self) -> None:
        if self.qvio_reward_enhancer is not None:
            self.qvio_reward_enhancer.require_compatible_solver()

    def _l2_by_id(self) -> dict[str, Record]:
        return {row.identifier: row for row in self.l2_rows if is_shunt(row)}

    def _observation(self) -> np.ndarray:
        observation = encode_observation(
            self.registry,
            self.l5_rows,
            self.l2_rows,
            self.previous_error,
            self.l6_rows,
            self.l3_rows,
            self.dc_plans,
            previous_error=self.prior_error,
            step_fraction=self.steps / max(self.max_steps, 1),
        )
        if observation.shape != (self.observation_size,):
            raise RuntimeError("Observation encoder violated the registry contract")
        return observation

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=bool)
        by_id = self._l2_by_id()
        controlled = self._controlled_ids()
        for index, (identifier, target) in enumerate(self.registry.action_targets):
            row = by_id.get(identifier)
            if row is None:
                continue
            current = int(row["Mark"])
            if current == target:
                continue
            candidate = tuple(
                target if item == identifier else int(by_id[item]["Mark"])
                for item in controlled
                if item in by_id
            )
            mask[index] = candidate not in self._visited_configurations
        return mask

    def _controlled_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(identifier for identifier, _ in self.registry.action_targets))

    def _configuration(self) -> tuple[int, ...]:
        by_id = self._l2_by_id()
        return tuple(
            int(by_id[identifier]["Mark"])
            for identifier in self._controlled_ids()
            if identifier in by_id
        )

    def _apply_initial_perturbations(self) -> list[str]:
        if self.initial_perturbations <= 0:
            return []
        by_id = self._l2_by_id()
        rows = [by_id[identifier] for identifier in self._controlled_ids() if identifier in by_id]
        count = min(self.initial_perturbations, len(rows))
        if not count:
            return []
        chosen = self._rng.choice(len(rows), size=count, replace=False)
        adjusted: list[str] = []
        records: list[list] = []
        for position in np.atleast_1d(chosen):
            row = rows[int(position)]
            target = 0 if int(row["Mark"]) else 1
            row["Mark"] = target
            adjusted.append(row.identifier)
            records.append([102, row.identifier, "Mark", target, 0])
        write_table(self.case_dir / "LF.L2", self.l2_rows)
        write_records(self.case_dir / "LFadj.txt", records)
        return adjusted

    def reset(self, *, case_index: int | None = None, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if case_index is None:
            case_index = self.current_case_index
            self.current_case_index = (self.current_case_index + 1) % len(self.cases)
        else:
            self.current_case_index = (int(case_index) + 1) % len(self.cases)
        source = self.cases[case_index % len(self.cases)]
        self.close()
        self._workspace = CaseWorkspace(source)
        self.case_dir = self._workspace.open()
        input_validation = self.input_validator.validate(self.case_dir)
        if not input_validation.active_power_valid:
            self.close()
            raise RuntimeError(
                f"RL-only input violates the active-power contract: {source}: "
                + "; ".join(input_validation.errors)
            )
        self.l2_rows = read_table(self.case_dir / "LF.L2", L2_FIELDS)
        l3_path = self.case_dir / "LF.L3"
        self.l3_rows = read_table(l3_path, L3_FIELDS) if l3_path.is_file() else []
        self.l5_rows = read_generator_table(self.case_dir / "LF.L5")
        l6_path = self.case_dir / "LF.L6"
        self.l6_rows = read_table(l6_path, L6_FIELDS) if l6_path.is_file() else []
        self.dc_plans = read_dc_plans(self.case_dir)
        self.steps = 0
        perturbed = self._apply_initial_perturbations()
        self._visited_configurations = {self._configuration()}
        result = self.runner.run(self.case_dir)
        self.previous_error = result.error
        self.prior_error = None
        self.previous_qvio = None
        if not result.converged and self.qvio_reward_enhancer is not None:
            self.previous_qvio = self.qvio_reward_enhancer.evaluate(self.case_dir)
        info = {
            "source_case": str(source),
            "workspace": str(self.case_dir),
            "converged": result.converged,
            "error": result.error,
            "action_mask": self.valid_action_mask(),
            "initial_perturbations": perturbed,
            "active_power_valid": input_validation.active_power_valid,
            "active_power_islands": input_validation.islands,
        }
        if self.previous_qvio is not None:
            info.update(self.previous_qvio.info())
        return self._observation(), info

    def step(self, action: int):
        if self.case_dir is None:
            raise RuntimeError("reset() must be called before step()")
        mask = self.valid_action_mask()
        valid = (
            isinstance(action, (int, np.integer))
            and 0 <= int(action) < self.action_size
            and mask[int(action)]
        )
        if not valid:
            self.steps += 1
            truncated = self.steps >= self.max_steps
            return self._observation(), self.invalid_action_penalty, False, truncated, {
                "invalid_action": True,
                "action_mask": mask,
                "converged": False,
                "error": self.previous_error,
            }

        action = int(action)
        identifier, target_status = self.registry.action_targets[action]
        row = self._l2_by_id()[identifier]
        row["Mark"] = target_status
        write_table(self.case_dir / "LF.L2", self.l2_rows)
        write_records(self.case_dir / "LFadj.txt", [[102, identifier, "Mark", target_status, 0]])

        result = self.runner.run(self.case_dir)
        self.steps += 1
        if result.converged:
            reward = self.success_reward - self.step_cost
            reward_components = QViolationRewardComponents(total=reward)
        else:
            current_qvio = (
                self.qvio_reward_enhancer.evaluate(self.case_dir)
                if self.qvio_reward_enhancer is not None
                else QViolationDiagnostics.unavailable("Q-violation reward is disabled")
            )
            reward_components = compose_qvio_reward(
                self.previous_qvio,
                current_qvio,
                settings=(
                    self.qvio_reward_enhancer.settings
                    if self.qvio_reward_enhancer is not None
                    else None
                ),
                step_cost=self.step_cost,
            )
            reward = reward_components.total
            self.previous_qvio = current_qvio
        self.prior_error = self.previous_error
        self.previous_error = result.error
        self._visited_configurations.add(self._configuration())
        terminated = result.converged
        truncated = self.steps >= self.max_steps and not terminated
        info: dict[str, Any] = {
            "converged": result.converged,
            "error": result.error,
            "action_mask": self.valid_action_mask(),
            "device_id": identifier,
            "target_status": target_status,
            "revisited_configuration": False,
        }
        info.update(reward_components.info())
        if not result.converged and self.previous_qvio is not None:
            info.update(self.previous_qvio.info())
        return self._observation(), float(reward), terminated, truncated, info

    def preview_action(self, action: int):
        """Evaluate one valid action on an isolated copy without mutating the episode."""
        if self.case_dir is None:
            raise RuntimeError("reset() must be called before preview_action()")
        mask = self.valid_action_mask()
        if not isinstance(action, (int, np.integer)) or not (
            0 <= int(action) < self.action_size and mask[int(action)]
        ):
            raise ValueError("preview_action requires a currently valid action")
        identifier, target_status = self.registry.action_targets[int(action)]
        with CaseWorkspace(self.case_dir) as preview_dir:
            rows = read_table(preview_dir / "LF.L2", L2_FIELDS)
            by_id = {row.identifier: row for row in rows if is_shunt(row)}
            by_id[identifier]["Mark"] = target_status
            write_table(preview_dir / "LF.L2", rows)
            return self.runner.run(preview_dir)

    def close(self) -> None:
        if self._workspace is not None:
            self._workspace.close()
        self._workspace = None
        self.case_dir = None
        self.previous_qvio = None

    def __enter__(self) -> "PowerAdjustmentEnv":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
