from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .psasp import CaseWorkspace, PSASPRunner, read_records, write_records


REWARD_SCHEMA_VERSION = 5
@dataclass(frozen=True)
class QViolationSettings:
    widened_q_limit: float = 9999.0
    log_clip_decades: float = 2.0
    improvement_weight: float = 40.0
    q_capability_floor: float = 0.1
    total_weight: float = 0.50
    maximum_weight: float = 0.25
    rms_weight: float = 0.15
    count_weight: float = 0.10

    def __post_init__(self) -> None:
        if self.widened_q_limit <= 0.0:
            raise ValueError("widened_q_limit must be positive")
        if self.log_clip_decades <= 0.0 or self.improvement_weight <= 0.0:
            raise ValueError("Q-violation reward scaling must be positive")
        if self.q_capability_floor <= 0.0:
            raise ValueError("q_capability_floor must be positive")
        weights = (
            self.total_weight,
            self.maximum_weight,
            self.rms_weight,
            self.count_weight,
        )
        if any(value < 0.0 for value in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("Q-violation composite weights must be nonnegative and sum to one")


@dataclass(frozen=True)
class QViolationDiagnostics:
    available: bool
    applicable: bool = False
    nolimit_converged: bool = False
    q_excess_total: float | None = None
    q_excess_max: float | None = None
    n_violating: int | None = None
    n_voltage_controlled: int = 0
    n_gen_widened: int = 0
    worst_bus: int | None = None
    worst_excess: float | None = None
    normalized_total: float | None = None
    normalized_max: float | None = None
    normalized_rms: float | None = None
    violating_fraction: float | None = None
    composite_score: float | None = None
    normalized_excesses: tuple[float, ...] = ()
    error: str | None = None

    @classmethod
    def unavailable(
        cls,
        error: str,
        *,
        applicable: bool = False,
        n_voltage_controlled: int = 0,
        n_gen_widened: int = 0,
    ) -> "QViolationDiagnostics":
        return cls(
            False,
            applicable=applicable,
            n_voltage_controlled=n_voltage_controlled,
            n_gen_widened=n_gen_widened,
            error=error,
        )

    def info(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("normalized_excesses")
        return {f"qvio_{key}": item for key, item in value.items()}


@dataclass(frozen=True)
class QViolationRewardComponents:
    log_improvement: float = 0.0
    total: float = -1.0

    def info(self) -> dict[str, float]:
        return {
            "reward_qvio_log_improvement": float(self.log_improvement),
            "reward_total": float(self.total),
        }


def compose_qvio_reward(
    previous: QViolationDiagnostics | None,
    current: QViolationDiagnostics,
    *,
    settings: QViolationSettings | None = None,
    step_cost: float = 1.0,
) -> QViolationRewardComponents:
    """Turn the PSASP-native Q-limit violation metric into dense reward.

    The composite combines normalized total/max/RMS excess and violation count;
    lower is better. No PSASP iteration error or independent LCQ state is used.
    """
    settings = settings or QViolationSettings()
    if (
        previous is None
        or not previous.available
        or not current.available
        or previous.composite_score is None
        or current.composite_score is None
    ):
        return QViolationRewardComponents(total=-float(step_cost))
    before = max(float(previous.composite_score), 1e-12)
    after = max(float(current.composite_score), 1e-12)
    log_improvement = float(np.log10(before / after))
    bounded = float(np.clip(
        log_improvement,
        -settings.log_clip_decades,
        settings.log_clip_decades,
    ))
    total = -float(step_cost) + settings.improvement_weight * bounded
    lower = -float(step_cost) - settings.improvement_weight * settings.log_clip_decades
    upper = -float(step_cost) + settings.improvement_weight * settings.log_clip_decades
    return QViolationRewardComponents(
        log_improvement=log_improvement,
        total=float(np.clip(total, lower, upper)),
    )


def _solver_sha256(executable: Path) -> str:
    return hashlib.sha256(executable.read_bytes()).hexdigest()


def _l5_layout(row: list) -> tuple[int, int, int]:
    """Return Type/Qmax/Qmin positions for supported 19/20-column LF.L5."""
    if len(row) >= 20:
        return 3, 8, 9
    if len(row) >= 19:
        return 2, 7, 8
    raise ValueError("LF.L5 must use the supported 19- or 20-column layout")


class QViolationRewardEnhancer:
    """PSASP-native dense signal for Type=-2/-3 reactive-limit cases.

    A temporary copy receives widened Q limits and is solved by the same PSASP
    engine. Fresh LF.LP5 Qg values are compared with the untouched original
    limits. The original case is never modified.
    """

    def __init__(
        self,
        runner: PSASPRunner,
        settings: QViolationSettings | None = None,
    ):
        self.runner = runner
        self.settings = settings or QViolationSettings()

    @property
    def solver_sha256(self) -> str:
        return _solver_sha256(self.runner.executable)

    @property
    def solver_compatible(self) -> bool:
        return self.runner.executable.is_file()

    def require_compatible_solver(self) -> None:
        if not self.solver_compatible:
            raise RuntimeError(
                f"Q-violation reward requires an available PSASP executable: "
                f"{self.runner.executable}"
            )

    def is_applicable(self, case_dir: Path | str) -> bool:
        try:
            return bool(self._read_limits(Path(case_dir)))
        except Exception:
            return False

    @property
    def contract(self) -> dict[str, object]:
        source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return {
            "schema_version": REWARD_SCHEMA_VERSION,
            "kind": "psasp_terminal_q_limit_violation_composite_reward",
            "implementation_sha256": source_digest,
            "solver_sha256": self.solver_sha256,
            "solver_compatible": self.solver_compatible,
            "settings": asdict(self.settings),
            "formula": "J=.50*log1p(E1)+.25*log1p(Einf)+.15*log1p(E2)+.10*EN; "
            "reward=clip(-step_cost+weight*log10(Jprev/Jcur),bounds)",
        }

    @staticmethod
    def _read_limits(case_dir: Path) -> list[dict[str, float | int | str]]:
        rows = read_records(case_dir / "LF.L5")
        limits: list[dict[str, float | int | str]] = []
        controlled_buses: set[int] = set()
        for row in rows:
            type_i, qmax_i, qmin_i = _l5_layout(row)
            if int(row[0]) != 1 or int(row[type_i]) not in (-2, -3):
                continue
            bus = abs(int(row[1]))
            if bus in controlled_buses:
                raise ValueError(
                    f"Multiple active Type=-2/-3 generators share bus {bus}; "
                    "LF.LP5 bus aggregation cannot recover per-unit violations"
                )
            controlled_buses.add(bus)
            limits.append({
                "bus": bus,
                "type": int(row[type_i]),
                "qmax": float(row[qmax_i]),
                "qmin": float(row[qmin_i]),
                "name": str(row[-1]).strip(),
            })
        return limits

    def _widen_limits(self, case_dir: Path) -> int:
        path = case_dir / "LF.L5"
        rows = read_records(path)
        count = 0
        for row in rows:
            _type_i, qmax_i, qmin_i = _l5_layout(row)
            row[qmax_i] = self.settings.widened_q_limit
            row[qmin_i] = -self.settings.widened_q_limit
            count += 1
        write_records(path, rows)
        return count

    @staticmethod
    def _read_lp5_qg(case_dir: Path) -> dict[int, float]:
        path = case_dir / "LF.LP5"
        if not path.is_file():
            raise FileNotFoundError(f"PSASP did not create fresh LF.LP5: {path}")
        result: dict[int, float] = {}
        for row in read_records(path):
            if len(row) < 3:
                continue
            try:
                bus = abs(int(row[0]))
                qg = float(row[2])
            except (TypeError, ValueError):
                continue
            if bus > 0 and np.isfinite(qg):
                result[bus] = result.get(bus, 0.0) + qg
        return result

    def evaluate(self, case_dir: Path | str) -> QViolationDiagnostics:
        source = Path(case_dir)
        try:
            limits = self._read_limits(source)
        except Exception as exc:
            return QViolationDiagnostics.unavailable(f"{type(exc).__name__}: {exc}")
        if not limits:
            return QViolationDiagnostics.unavailable(
                "No active Type=-2/-3 generator; Q-violation metric is out of scope",
                applicable=False,
            )
        try:
            with CaseWorkspace(source) as work:
                n_widened = self._widen_limits(work)
                solved = self.runner.run(work)
                if not solved.converged:
                    return QViolationDiagnostics.unavailable(
                        "PSASP still fails after widening Q limits; cause is outside "
                        "the Q-limit-switching reward scope",
                        applicable=True,
                        n_voltage_controlled=len(limits),
                        n_gen_widened=n_widened,
                    )
                qg_by_bus = self._read_lp5_qg(work)
        except Exception as exc:
            return QViolationDiagnostics.unavailable(
                f"{type(exc).__name__}: {exc}",
                applicable=True,
                n_voltage_controlled=len(limits),
            )

        missing = [int(item["bus"]) for item in limits if int(item["bus"]) not in qg_by_bus]
        if missing:
            return QViolationDiagnostics.unavailable(
                f"Fresh LF.LP5 misses controlled generator buses: {missing[:5]}",
                applicable=True,
                n_voltage_controlled=len(limits),
                n_gen_widened=n_widened,
            )

        excesses: list[tuple[float, int, float]] = []
        for item in limits:
            bus = int(item["bus"])
            qg = qg_by_bus[bus]
            excess = max(0.0, qg - float(item["qmax"])) + max(
                0.0, float(item["qmin"]) - qg
            )
            capability = max(
                float(item["qmax"]) - float(item["qmin"]),
                self.settings.q_capability_floor,
            )
            excesses.append((float(excess), bus, float(excess) / capability))
        violating = [(value, bus) for value, bus, _normalized in excesses if value > 1e-6]
        worst_value, worst_bus, _worst_normalized = max(
            excesses, default=(0.0, 0, 0.0), key=lambda item: item[0]
        )
        normalized = np.asarray([item[2] for item in excesses], dtype=float)
        e1 = float(normalized.sum()) if normalized.size else 0.0
        einf = float(normalized.max()) if normalized.size else 0.0
        e2 = float(np.sqrt(np.mean(normalized * normalized))) if normalized.size else 0.0
        en = float(len(violating) / len(excesses)) if excesses else 0.0
        composite = float(
            self.settings.total_weight * np.log1p(e1)
            + self.settings.maximum_weight * np.log1p(einf)
            + self.settings.rms_weight * np.log1p(e2)
            + self.settings.count_weight * en
        )
        return QViolationDiagnostics(
            available=True,
            applicable=True,
            nolimit_converged=True,
            q_excess_total=float(sum(value for value, _bus, _normalized in excesses)),
            q_excess_max=float(worst_value),
            n_violating=len(violating),
            n_voltage_controlled=len(limits),
            n_gen_widened=n_widened,
            worst_bus=worst_bus or None,
            worst_excess=float(worst_value),
            normalized_total=e1,
            normalized_max=einf,
            normalized_rms=e2,
            violating_fraction=en,
            composite_score=composite,
            normalized_excesses=tuple(float(value) for value in normalized),
        )
