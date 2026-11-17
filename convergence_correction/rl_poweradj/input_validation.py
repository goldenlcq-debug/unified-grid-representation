from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from .model import (
    L2_FIELDS,
    L3_FIELDS,
    L5_FIELDS,
    L6_FIELDS,
    is_shunt,
    read_dc_plans,
    read_generator_table,
    read_table,
)
from .psasp import read_records


@dataclass(frozen=True)
class IslandActiveStatus:
    index: int
    buses: tuple[int, ...]
    demand: float
    scheduled_generation: float
    dc_injection: float
    scheduled_mismatch: float
    has_online_reference: bool
    energized: bool


@dataclass(frozen=True)
class InputValidationResult:
    """Read-only validation result for the RL-only input contract."""

    active_power_valid: bool
    islands: tuple[IslandActiveStatus, ...]
    errors: tuple[str, ...]


class InputCaseValidator:
    """Validate that an input can go directly to AC and reactive RL control.

    Validation is deliberately read-only. A Type=0 generator may absorb the
    scheduled active-power residual of its island, as it does in PSASP. An
    energized island without an online Type=0 reference is invalid; the RL-only
    pipeline never starts generators, changes Pg/DC schedules, or sheds load.
    """

    def __init__(self, tolerance: float = 1e-6):
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        self.tolerance = float(tolerance)

    @staticmethod
    def _islands(case_dir: Path, bus_count: int) -> list[tuple[int, ...]]:
        graph = nx.Graph()
        graph.add_nodes_from(range(bus_count))
        for row in read_table(case_dir / "LF.L2", L2_FIELDS):
            if int(row["Mark"]) == 1 and not is_shunt(row):
                graph.add_edge(abs(int(row["I_No"])) - 1, abs(int(row["J_No"])) - 1)
        l3_path = case_dir / "LF.L3"
        if l3_path.is_file():
            for row in read_table(l3_path, L3_FIELDS):
                if int(row["Mark"]) == 1:
                    graph.add_edge(abs(int(row["I_No"])) - 1, abs(int(row["J_No"])) - 1)
        return [tuple(sorted(component)) for component in nx.connected_components(graph)]

    def validate(self, case_dir: Path | str) -> InputValidationResult:
        case_dir = Path(case_dir)
        bus_count = len(read_records(case_dir / "LF.L1"))
        islands = self._islands(case_dir, bus_count)
        bus_to_island = {
            bus: index for index, island in enumerate(islands) for bus in island
        }
        demand = np.zeros(len(islands), dtype=float)
        generation = np.zeros(len(islands), dtype=float)
        dc_injection = np.zeros(len(islands), dtype=float)
        has_reference = np.zeros(len(islands), dtype=bool)
        energized = np.zeros(len(islands), dtype=bool)
        errors: list[str] = []

        generators = read_generator_table(case_dir / "LF.L5")
        for row in generators:
            if int(row["Mark"]) != 1:
                continue
            index = bus_to_island[abs(int(row["Bs_No"])) - 1]
            pg = float(row["Pg"])
            pmin, pmax = float(row["Pmin"]), float(row["Pmax"])
            if pmax < pmin - self.tolerance:
                errors.append(
                    f"generator {row.identifier} has invalid bounds [{pmin:.6g}, {pmax:.6g}]"
                )
            # Pg limits are dispatch metadata, not an AC power-flow switching
            # rule. RL-only operation never changes Pg, and PSASP can solve a
            # fixed schedule outside these planning bounds, so it must not be
            # rejected here. Reversed bounds remain a malformed-input error.
            generation[index] += pg
            energized[index] = True
            if int(row["Type"]) == 0:
                has_reference[index] = True

        l6_path = case_dir / "LF.L6"
        loads = read_table(l6_path, L6_FIELDS) if l6_path.is_file() else []
        for row in loads:
            if int(row["Mark"]) != 1:
                continue
            index = bus_to_island[abs(int(row["Bs_No"])) - 1]
            pl, ql = float(row["Pl"]), float(row["Ql"])
            demand[index] += pl
            if abs(pl) > self.tolerance or abs(ql) > self.tolerance:
                energized[index] = True

        for plan in read_dc_plans(case_dir):
            if abs(plan.transfer) <= self.tolerance:
                continue
            from_index = bus_to_island[plan.from_bus]
            to_index = bus_to_island[plan.to_bus]
            dc_injection[from_index] -= plan.transfer
            dc_injection[to_index] += plan.transfer
            energized[from_index] = True
            energized[to_index] = True

        statuses: list[IslandActiveStatus] = []
        for index, buses in enumerate(islands):
            mismatch = generation[index] + dc_injection[index] - demand[index]
            statuses.append(IslandActiveStatus(
                index=index,
                buses=buses,
                demand=float(demand[index]),
                scheduled_generation=float(generation[index]),
                dc_injection=float(dc_injection[index]),
                scheduled_mismatch=float(mismatch),
                has_online_reference=bool(has_reference[index]),
                energized=bool(energized[index]),
            ))
            if energized[index] and not has_reference[index]:
                errors.append(
                    f"energized island {index} has no online Type=0 reference generator"
                )

        return InputValidationResult(not errors, tuple(statuses), tuple(errors))
