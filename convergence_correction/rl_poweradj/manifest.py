from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .input_validation import InputCaseValidator
from .paths import case_input_fingerprint
from .psasp import CaseWorkspace, PSASPRunner, convergence_status


@dataclass(frozen=True)
class CaseFlowRecord:
    case: str
    input_fingerprint: str
    raw_converged: bool
    active_power_valid: bool
    active_power_errors: tuple[str, ...]
    first_ac_converged: bool | None
    first_ac_error: float | None
    category: str

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_case_flow(
    source: Path | str,
    runner: PSASPRunner,
    validator: InputCaseValidator | None = None,
) -> CaseFlowRecord:
    source = Path(source).resolve()
    validator = validator or InputCaseValidator()
    raw_converged = convergence_status(source)
    with CaseWorkspace(source) as workspace:
        validation = validator.validate(workspace)
        if not validation.active_power_valid:
            return CaseFlowRecord(
                source.name, case_input_fingerprint(source).hex(), raw_converged,
                False, validation.errors, None, None, "active_unbalanced",
            )
        result = runner.run(workspace)
    category = (
        "raw_converged" if raw_converged
        else "first_ac_converged" if result.converged
        else "requires_reactive_recovery"
    )
    return CaseFlowRecord(
        source.name,
        case_input_fingerprint(source).hex(),
        raw_converged,
        True,
        (),
        result.converged,
        result.error,
        category,
    )
