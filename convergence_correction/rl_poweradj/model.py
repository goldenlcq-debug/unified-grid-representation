from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .control import ControlPolicy
from .psasp import read_records, write_records


L2_FIELDS = (
    "Mark", "I_No", "J_No", "ID_No", "R", "X", "B_half", "F_T",
    "Type", "PL", "CB", "CL", "VQP", "LC", "LIM", "Name_Ctrl",
    "Type_Ctrl", "IDName",
)
L3_FIELDS = (
    "Mark", "I_No", "J_No", "ID_No", "R1", "X1", "Tk", "Rm", "Xm",
    "F_T", "Type", "PL", "TP", "CB", "CL", "VQP", "theta", "TC", "LIM",
    "ID", "J", "TrsType", "Name_Ctrl", "Type_Ctrl", "IDName", "Un_I",
    "Un_J", "TapMax", "TapMin", "TapUn", "DU", "TapPos",
)
L5_FIELDS = (
    # Older project cases do not contain ID_No. Their third physical column is
    # Type, followed immediately by Pg. Do not insert the newer ID_No field or
    # every electrical quantity will be shifted by one column.
    "Mark", "Bs_No", "Type", "Pg", "Qg", "V0", "theta", "Qmax",
    "Qmin", "Pmax", "Pmin", "PL", "CB", "CL", "VQP", "K",
    "Name_Ctrl", "Type_Ctrl", "IDName",
)
L6_FIELDS = (
    "Mark", "Bs_No", "ID_No", "Type", "Pl", "Ql", "V0", "theta",
    "Qmax", "Qmin", "Pmax", "Pmin", "PL", "CB", "CL", "VQP",
    "Name_Ctrl", "Type_Ctrl", "IDName",
)

# This is the contract for the project's deterministic, non-neural observation
# builder.  Do not describe it as the upstream pretrained unified encoder.
DETERMINISTIC_OBSERVATION_METADATA = {
    "kind": "deterministic_observation_encoder",
    "schema_version": 2,
    "trainable": False,
}


@dataclass(frozen=True)
class DCLink:
    identifier: int
    from_bus: int
    to_bus: int
    from_injection: float
    to_injection: float
    name: str

    @property
    def loss(self) -> float:
        return max(-(self.from_injection + self.to_injection), 0.0)


@dataclass(frozen=True)
class DCPlan:
    """Input-side two-terminal DC schedule parsed exclusively from LF.NL4."""

    identifier: int
    from_bus: int
    to_bus: int
    # Signed I-to-J schedule in p.u.; a negative value means J-to-I.
    transfer: float
    max_transfer: float
    area: int
    operating_mode: int
    name: str
    record_offset: int

    @property
    def from_injection(self) -> float:
        return -self.transfer

    @property
    def to_injection(self) -> float:
        return -self.from_injection


def _system_base(case_dir: Path) -> float:
    records = read_records(case_dir / "LF.L0")
    if len(records) < 2 or not records[1]:
        raise ValueError("LF.L0 does not contain the system base-power record")
    base = float(records[1][0])
    if base <= 0.0:
        raise ValueError("LF.L0 system base power must be positive")
    return base


def read_dc_plans(case_dir: Path | str) -> list[DCPlan]:
    """Read active DC schedules from LF.NL4 without consulting result files.

    Each LF.NL4 system occupies eight physical records.  Pd1/Pd2 are MW;
    the rest of the project uses per-unit active power, so schedules are
    divided by LF.L0.SB.  OP=20 is bipolar and uses both pole schedules;
    the other documented modes have one effective scheduled pole.
    """
    case_dir = Path(case_dir)
    nl4_path = case_dir / "LF.NL4"
    if not nl4_path.is_file():
        return []
    nl4 = read_records(nl4_path)
    if len(nl4) % 8:
        raise ValueError("LF.NL4 must contain eight records per DC link")
    base = _system_base(case_dir)
    plans: list[DCPlan] = []
    for offset in range(0, len(nl4), 8):
        header = nl4[offset]
        if len(header) < 6:
            raise ValueError(f"Invalid LF.NL4 header at record {offset + 1}")
        if int(header[0]) != 1:
            continue
        control = nl4[offset + 5]
        pole1 = nl4[offset + 6]
        pole2 = nl4[offset + 7]
        if len(control) < 1 or len(pole1) < 1 or len(pole2) < 1:
            raise ValueError(f"Invalid LF.NL4 control records at record {offset + 6}")
        mode = int(control[0])
        pole_count = {10: 1, 11: 1, 12: 1, 20: 2}.get(mode)
        if pole_count is None:
            raise ValueError(f"Unsupported LF.NL4 operating mode OP={mode}")
        pole1_mw = float(pole1[0])
        pole2_mw = float(pole2[0])
        if mode == 20 and pole1_mw * pole2_mw < 0.0:
            raise ValueError(
                "LF.NL4 bipolar pole schedules have opposite signs; "
                "a single two-terminal transfer cannot represent that state"
            )
        scheduled_mw = pole1_mw + (pole2_mw if mode == 20 else 0.0)
        from_converter = nl4[offset + 3]
        to_converter = nl4[offset + 4]
        if len(from_converter) < 4 or len(to_converter) < 4:
            raise ValueError(f"Invalid LF.NL4 converter records at record {offset + 4}")
        # Sti/Stj are already documented as *single-pole total* converter-
        # transformer capacities. Bi/Bj are bridge counts and must not be
        # multiplied into the MVA rating a second time.
        single_pole_mva = min(
            abs(float(from_converter[3])), abs(float(to_converter[3]))
        )
        rated_mva = pole_count * single_pole_mva
        if rated_mva > 0.0 and abs(scheduled_mw) > rated_mva + 1e-6:
            raise ValueError(
                f"LF.NL4 schedule {scheduled_mw:.6g} MW exceeds the "
                f"OP={mode} converter rating {rated_mva:.6g} MVA"
            )
        max_transfer = (
            rated_mva / base if rated_mva > 0.0 else abs(scheduled_mw) / base
        )
        plans.append(DCPlan(
            identifier=int(header[3]),
            from_bus=abs(int(header[1])) - 1,
            to_bus=abs(int(header[2])) - 1,
            transfer=scheduled_mw / base,
            max_transfer=max_transfer,
            area=int(header[4]),
            operating_mode=mode,
            name=str(header[5]).strip(),
            record_offset=offset,
        ))
    return plans


def write_dc_plans(case_dir: Path | str, plans: Sequence[DCPlan], transfers: Sequence[float]) -> None:
    """Write signed I-to-J p.u. transfers while preserving pole proportions."""
    case_dir = Path(case_dir)
    if len(plans) != len(transfers):
        raise ValueError("DC transfer vector does not match LF.NL4 plans")
    path = case_dir / "LF.NL4"
    rows = read_records(path)
    base = _system_base(case_dir)
    for plan, transfer in zip(plans, transfers):
        value = float(transfer)
        if abs(value) > plan.max_transfer + 1e-10:
            raise ValueError(
                f"DC transfer for {plan.name} exceeds its OP={plan.operating_mode} rating"
            )
        duplicated_single_pole = plan.operating_mode != 20 and np.isclose(
            float(rows[plan.record_offset + 7][0]),
            float(rows[plan.record_offset + 6][0]),
            atol=1e-9,
        )
        pole_rows = [rows[plan.record_offset + 6]]
        if plan.operating_mode == 20:
            pole_rows.append(rows[plan.record_offset + 7])
        old = np.asarray([abs(float(row[0])) for row in pole_rows], dtype=float)
        weights = old / old.sum() if old.sum() > 0.0 else np.full(len(old), 1.0 / len(old))
        for row, weight in zip(pole_rows, weights):
            row[0] = value * base * float(weight)
        # Keep the inactive/parallel pole consistent when it duplicates pole 1.
        if duplicated_single_pole:
            rows[plan.record_offset + 7][0] = rows[plan.record_offset + 6][0]
    write_records(path, rows)


def read_dc_links(case_dir: Path | str) -> list[DCLink]:
    """Compatibility view of input DC plans; never reads LF.LP4/NP4 results."""
    return [
        DCLink(
            identifier=plan.identifier,
            from_bus=plan.from_bus,
            to_bus=plan.to_bus,
            from_injection=plan.from_injection,
            to_injection=plan.to_injection,
            name=plan.name,
        )
        for plan in read_dc_plans(case_dir)
    ]
@dataclass
class Record:
    """A PSASP row accessed through documented field names."""

    schema: tuple[str, ...]
    values: list

    def __post_init__(self) -> None:
        if len(self.values) < len(self.schema):
            # PSASP 7.91 omits trailing fields that were introduced by later
            # format revisions. Their documented positions remain stable.
            self.values.extend([None] * (len(self.schema) - len(self.values)))

    def __getitem__(self, field: str):
        return self.values[self.schema.index(field)]

    def __setitem__(self, field: str, value) -> None:
        self.values[self.schema.index(field)] = value

    @property
    def identifier(self) -> str:
        return str(self["IDName"]).strip()


def read_table(path: Path | str, schema: Sequence[str]) -> list[Record]:
    normalized = tuple(schema)
    return [Record(normalized, row) for row in read_records(path)]


def read_generator_table(path: Path | str) -> list[Record]:
    """Read both legacy 19-column and PSASP V4.5 20-column LF.L5.

    V4.5 inserts ID_No after Bs_No. The maintained internal generator schema
    intentionally omits that identity column, so it is removed during input
    normalization without changing observation schema 2 field order.
    """
    normalized: list[Record] = []
    for row in read_records(path):
        if len(row) >= 20:
            row = [row[0], row[1], *row[3:20], *row[20:]]
        normalized.append(Record(tuple(L5_FIELDS), row))
    return normalized


def write_table(path: Path | str, rows: Iterable[Record]) -> None:
    write_records(path, [row.values for row in rows])


def is_shunt(row: Record) -> bool:
    return abs(int(row["I_No"])) == abs(int(row["J_No"]))


def transformer_identifier(row: Record) -> str:
    return f"{abs(int(row['I_No']))}:{abs(int(row['J_No']))}:{row.identifier}"


def dc_plan_identifier(plan: DCPlan) -> str:
    return f"{plan.identifier}:{plan.from_bus + 1}:{plan.to_bus + 1}:{plan.name}"


@dataclass(frozen=True)
class DeviceRegistry:
    """Versioned fixed observation/action contract shared by every case."""

    generator_ids: tuple[str, ...]
    shunt_ids: tuple[str, ...]
    branch_ids: tuple[str, ...] = ()
    load_ids: tuple[str, ...] = ()
    transformer_ids: tuple[str, ...] = ()
    dc_ids: tuple[str, ...] = ()
    action_targets: tuple[tuple[str, int], ...] = ()
    bus_count: int = 0
    policy_digest: str = ""
    schema_version: int = 2

    @classmethod
    def from_cases(
        cls,
        cases: Iterable[Path | str],
        control_policy: ControlPolicy | None = None,
    ) -> "DeviceRegistry":
        generators: set[str] = set()
        shunts: set[str] = set()
        branches: set[str] = set()
        loads: set[str] = set()
        transformers: set[str] = set()
        dc_ids: set[str] = set()
        bus_count = 0
        for case in cases:
            case_path = Path(case)
            l5 = read_generator_table(case_path / "LF.L5")
            l2 = read_table(case_path / "LF.L2", L2_FIELDS)
            l3_path = case_path / "LF.L3"
            l3 = read_table(l3_path, L3_FIELDS) if l3_path.is_file() else []
            l6_path = case_path / "LF.L6"
            l6 = read_table(l6_path, L6_FIELDS) if l6_path.is_file() else []
            generators.update(row.identifier for row in l5)
            shunts.update(row.identifier for row in l2 if is_shunt(row))
            branches.update(row.identifier for row in l2 if not is_shunt(row))
            loads.update(row.identifier for row in l6)
            transformers.update(transformer_identifier(row) for row in l3)
            dc_ids.update(dc_plan_identifier(plan) for plan in read_dc_plans(case_path))
            l1_path = case_path / "LF.L1"
            if l1_path.is_file():
                bus_count = max(bus_count, len(read_records(l1_path)))
            else:
                numbered_buses = [
                    abs(int(row[field]))
                    for rows, fields in (
                        (l2, ("I_No", "J_No")),
                        (l3, ("I_No", "J_No")),
                        (l5, ("Bs_No",)),
                        (l6, ("Bs_No",)),
                    )
                    for row in rows
                    for field in fields
                ]
                bus_count = max(bus_count, max(numbered_buses, default=0))
        if not shunts:
            raise ValueError("No controllable shunts were found in the supplied cases")
        policy = control_policy or ControlPolicy.allow_all(shunts)
        policy.validate(shunts)
        return cls(
            tuple(sorted(generators, key=str.casefold)),
            tuple(sorted(shunts, key=str.casefold)),
            tuple(sorted(branches, key=str.casefold)),
            tuple(sorted(loads, key=str.casefold)),
            tuple(sorted(transformers, key=str.casefold)),
            tuple(sorted(dc_ids, key=str.casefold)),
            tuple((action.device_id, action.target_status) for action in policy.actions),
            bus_count,
            policy.digest,
        )

    @property
    def action_size(self) -> int:
        return len(self.action_targets)

    @property
    def observation_size(self) -> int:
        # global error/delta/progress + compact dynamic device state +
        # per-bus island energized/reference flags.
        return (
            3 + 4 * len(self.generator_ids) + 3 * len(self.shunt_ids)
            + 4 * len(self.load_ids) + 2 * len(self.branch_ids)
            + 2 * len(self.transformer_ids) + 2 * len(self.dc_ids)
            + 2 * self.bus_count
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generator_ids": list(self.generator_ids),
            "shunt_ids": list(self.shunt_ids),
            "branch_ids": list(self.branch_ids),
            "load_ids": list(self.load_ids),
            "transformer_ids": list(self.transformer_ids),
            "dc_ids": list(self.dc_ids),
            "action_targets": [list(item) for item in self.action_targets],
            "bus_count": self.bus_count,
            "policy_digest": self.policy_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DeviceRegistry":
        version = int(value.get("schema_version", 1))
        if version != 2:
            raise ValueError(
                "Checkpoint uses an obsolete observation/action registry; "
                "retrain with schema version 2"
            )
        return cls(
            tuple(value["generator_ids"]),  # type: ignore[arg-type]
            tuple(value["shunt_ids"]),  # type: ignore[arg-type]
            tuple(value.get("branch_ids", ())),  # type: ignore[arg-type]
            tuple(value.get("load_ids", ())),  # type: ignore[arg-type]
            tuple(value.get("transformer_ids", ())),  # type: ignore[arg-type]
            tuple(value.get("dc_ids", ())),  # type: ignore[arg-type]
            tuple(
                (str(item[0]), int(item[1]))
                for item in value.get("action_targets", ())  # type: ignore[union-attr]
            ),
            int(value.get("bus_count", 0)),
            str(value.get("policy_digest", "")),
            version,
        )


def _scaled(value, scale: float = 1.0) -> float:
    try:
        number = float(value) / scale
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(number, -10.0, 10.0))


def _log_error(error: float | None) -> float:
    if error is None:
        return 0.0
    value = np.log10(max(abs(float(error)), 1e-12))
    return float((np.clip(value, -12.0, 6.0) + 12.0) / 18.0)


def _log_improvement(error: float | None, previous_error: float | None) -> float:
    if error is None or previous_error is None:
        return 0.0
    ratio = max(abs(float(previous_error)), 1e-12) / max(abs(float(error)), 1e-12)
    return float(np.clip(np.log10(ratio), -6.0, 6.0) / 6.0)


def _bus_island_features(
    bus_count: int,
    generators: Sequence[Record],
    lines: Sequence[Record],
    transformers: Sequence[Record],
    loads: Sequence[Record],
    dc_plans: Sequence[DCPlan],
) -> list[float]:
    parent = list(range(bus_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for row in lines:
        if int(row["Mark"]) == 1 and not is_shunt(row):
            union(abs(int(row["I_No"])) - 1, abs(int(row["J_No"])) - 1)
    for row in transformers:
        if int(row["Mark"]) == 1:
            union(abs(int(row["I_No"])) - 1, abs(int(row["J_No"])) - 1)

    energized: set[int] = set()
    referenced: set[int] = set()
    for row in generators:
        if int(row["Mark"]) == 1:
            root = find(abs(int(row["Bs_No"])) - 1)
            energized.add(root)
            if int(row["Type"]) == 0:
                referenced.add(root)
    for row in loads:
        if int(row["Mark"]) == 1 and (
            abs(float(row["Pl"])) > 1e-12 or abs(float(row["Ql"])) > 1e-12
        ):
            energized.add(find(abs(int(row["Bs_No"])) - 1))
    for plan in dc_plans:
        if abs(plan.transfer) > 1e-12:
            energized.add(find(plan.from_bus))
            energized.add(find(plan.to_bus))

    features: list[float] = []
    for bus in range(bus_count):
        root = find(bus)
        features.extend((float(root in energized), float(root in referenced)))
    return features


def encode_observation(
    registry: DeviceRegistry,
    generators: Iterable[Record],
    lines: Iterable[Record],
    error: float | None,
    loads: Iterable[Record] = (),
    transformers: Iterable[Record] = (),
    dc_plans: Iterable[DCPlan] = (),
    previous_error: float | None = None,
    step_fraction: float = 0.0,
) -> np.ndarray:
    generators = list(generators)
    lines = list(lines)
    loads = list(loads)
    transformers = list(transformers)
    dc_plans = list(dc_plans)
    generator_map = {row.identifier: row for row in generators}
    shunt_map = {row.identifier: row for row in lines if is_shunt(row)}
    branch_map = {row.identifier: row for row in lines if not is_shunt(row)}
    load_map = {row.identifier: row for row in loads}
    transformer_map = {transformer_identifier(row): row for row in transformers}
    dc_map = {dc_plan_identifier(plan): plan for plan in dc_plans}
    features: list[float] = [
        _log_error(error),
        _log_improvement(error, previous_error),
        float(np.clip(step_fraction, 0.0, 1.0)),
    ]
    for identifier in registry.generator_ids:
        row = generator_map.get(identifier)
        if row is None:
            features.extend((0.0,) * 4)
        else:
            features.extend(
                (
                    1.0,
                    _scaled(row["Mark"]),
                    _scaled(row["Pg"]),
                    float(int(row["Mark"]) == 1 and int(row["Type"]) == 0),
                )
            )
    for identifier in registry.shunt_ids:
        row = shunt_map.get(identifier)
        if row is None:
            features.extend((0.0,) * 3)
        else:
            features.extend(
                (
                    1.0,
                    _scaled(row["Mark"]),
                    _scaled(row["B_half"]),
                )
            )
    for identifier in registry.load_ids:
        row = load_map.get(identifier)
        if row is None:
            features.extend((0.0,) * 4)
        else:
            features.extend(
                (
                    1.0,
                    _scaled(row["Mark"]),
                    _scaled(row["Pl"]),
                    _scaled(row["Ql"]),
                )
            )
    for identifier in registry.branch_ids:
        row = branch_map.get(identifier)
        if row is None:
            features.extend((0.0, 0.0))
        else:
            features.extend((1.0, _scaled(row["Mark"])))
    for identifier in registry.transformer_ids:
        row = transformer_map.get(identifier)
        features.extend((0.0, 0.0) if row is None else (1.0, _scaled(row["Mark"])))
    for identifier in registry.dc_ids:
        plan = dc_map.get(identifier)
        if plan is None:
            features.extend((0.0, 0.0))
        else:
            scale = max(plan.max_transfer, 1e-12)
            features.extend((1.0, float(np.clip(plan.transfer / scale, -1.0, 1.0))))
    island_features = _bus_island_features(
        registry.bus_count, generators, lines, transformers, loads, dc_plans
    )
    features.extend(island_features)
    return np.asarray(features, dtype=np.float32)


def observation_feature_names(registry: DeviceRegistry) -> tuple[str, ...]:
    names = ["global.log_error", "global.log_improvement", "global.step_fraction"]
    for identifier in registry.generator_ids:
        names.extend((
            f"generator.{identifier}.present",
            f"generator.{identifier}.mark",
            f"generator.{identifier}.pg",
            f"generator.{identifier}.online_type0",
        ))
    for identifier in registry.shunt_ids:
        names.extend((
            f"shunt.{identifier}.present",
            f"shunt.{identifier}.mark",
            f"shunt.{identifier}.b_half",
        ))
    for identifier in registry.load_ids:
        names.extend((
            f"load.{identifier}.present",
            f"load.{identifier}.mark",
            f"load.{identifier}.pl",
            f"load.{identifier}.ql",
        ))
    for identifier in registry.branch_ids:
        names.extend((f"branch.{identifier}.present", f"branch.{identifier}.mark"))
    for identifier in registry.transformer_ids:
        names.extend((
            f"transformer.{identifier}.present", f"transformer.{identifier}.mark"
        ))
    for identifier in registry.dc_ids:
        names.extend((f"dc.{identifier}.present", f"dc.{identifier}.transfer_ratio"))
    for bus in range(registry.bus_count):
        names.extend((f"bus.{bus + 1}.island_energized", f"bus.{bus + 1}.island_type0"))
    if len(names) != registry.observation_size:
        raise RuntimeError("Observation feature-name contract is inconsistent")
    return tuple(names)
