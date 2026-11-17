from __future__ import annotations

import ast
import re
import shutil
import subprocess
import tempfile
import time
import math
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from .solver_lock import SolverLock
from .run_store import atomic_json, file_hash


def read_records(path: Path | str, encoding: str = "gbk") -> list[list]:
    """Read comma-terminated PSASP records without executing input text."""
    rows: list[list] = []
    with Path(path).open(encoding=encoding) as stream:
        for line_number, raw in enumerate(stream, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = ast.literal_eval(f"({text})")
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Cannot parse PSASP record {path}:{line_number}") from exc
            rows.append(list(value if isinstance(value, tuple) else (value,)))
    return rows


def write_records(path: Path | str, rows: list[list], encoding: str = "gbk") -> None:
    with Path(path).open("w", encoding=encoding, newline="") as stream:
        for row in rows:
            stream.write(",  ".join(repr(value) for value in row) + ",\n")


def convergence_status(case_dir: Path | str) -> bool:
    """Read the authoritative PSASP convergence flag from LF.CAL."""
    cal_file = Path(case_dir) / "LF.CAL"
    if not cal_file.is_file():
        return False
    first = cal_file.read_text(encoding="gbk", errors="replace").splitlines()
    return bool(first and re.match(r"\s*1\s*,", first[0]))


def convergence_error(case_dir: Path | str) -> float | None:
    report = Path(case_dir) / "lfreport.lis"
    if not report.is_file():
        return None
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
    pattern = re.compile(rf"^\s*\d+\s+\S+\s+({number})(?:\s+.*)?$")
    errors: list[float] = []
    for line in report.read_text(encoding="gbk", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            errors.append(float(match.group(1)))
    return errors[-1] if errors else None


@dataclass(frozen=True)
class SolverResult:
    converged: bool
    error: float
    return_code: int


class MissingFreshOutputs(TimeoutError):
    """A completed invocation failed to deliver the required fresh files."""


def solver_processes_present():
    # Conservative: never retry while another solver/worker could still be alive.
    import psutil
    return any((p.info.get('name') or '').lower() in ('wmlfrtmsg.exe', 'wmudrt.exe')
               for p in psutil.process_iter(['name']))


class PSASPRunner:
    """Run PSASP and accept only fresh, parseable output from this invocation."""

    def __init__(self, executable: Path | str, timeout: float = 60.0, *, audit_dir=None, retries=1):
        self.executable = Path(executable).resolve()
        self.timeout = timeout
        if timeout <= 0 or retries not in (0, 1, 2):
            raise ValueError('Positive timeout and retries in 0..2 required')
        self.retries = retries
        if not self.executable.is_file():
            raise FileNotFoundError(f"PSASP solver does not exist: {self.executable}")
        self.audit_dir = Path(audit_dir) if audit_dir else Path(__file__).resolve().parents[1] / 'artifacts' / 'solver-audit'
        self.executable_sha256 = file_hash(self.executable)

    @staticmethod
    def _archive_outputs(case_dir: Path) -> None:
        # LP/NP files are solver results.  Removing them before every run
        # prevents a failed invocation from making stale results look fresh.
        outputs = [case_dir / "LF.CAL", case_dir / "lfreport.lis", case_dir / 'LFERR.LIS']
        outputs.extend(sorted(case_dir.glob("LF.LP*")))
        outputs.extend(sorted(case_dir.glob("LF.NP*")))
        for output in dict.fromkeys(outputs):
            if output.name.endswith(".previous"):
                continue
            previous = output.with_name(f"{output.name}.previous")
            if previous.exists():
                previous.unlink()
            if output.exists():
                output.replace(previous)

    def run(self, case_dir: Path | str) -> SolverResult:
        case_dir = Path(case_dir).resolve()
        with SolverLock():
            if solver_processes_present():
                raise RuntimeError('PSASP solver process already exists; refusing overlapping invocation')
            call_id = uuid.uuid4().hex
            for attempt in range(self.retries + 1):
                try:
                    return self._audited_attempt(case_dir, call_id, attempt)
                except MissingFreshOutputs:
                    if attempt >= self.retries or solver_processes_present():
                        raise

    def _audited_attempt(self, case_dir, call_id, attempt):
            run_dir = self.audit_dir / uuid.uuid4().hex
            run_dir.mkdir(parents=True, exist_ok=False)
            record = {'started_at': datetime.now(timezone.utc).isoformat(),
                      'call_id': call_id, 'attempt': attempt,
                      'solver': str(self.executable), 'solver_sha256': self.executable_sha256,
                      'case_dir': str(case_dir), 'status': 'running'}
            atomic_json(run_dir / 'run.json', record)
            started = time.monotonic()
            try:
                result = self._run_once(case_dir, record)
                record.update(status='success' if result.converged else 'nonconverged',
                              converged=result.converged, error=result.error)
                return result
            except Exception as exc:
                if 'license unavailable' in str(exc):
                    record['category'] = 'license_unavailable'
                record.update(status='solver_error', exception_type=type(exc).__name__, message=str(exc))
                if case_dir.is_dir():
                    shutil.copytree(case_dir, run_dir / 'failure-case')
                raise
            finally:
                record['seconds'] = time.monotonic() - started
                record['finished_at'] = datetime.now(timezone.utc).isoformat()
                record['outputs'] = {}
                for name in ('LF.CAL', 'lfreport.lis', 'LFERR.LIS'):
                    output = case_dir / name
                    if output.is_file():
                        shutil.copy2(output, run_dir / name)
                        record['outputs'][name] = {'sha256': file_hash(output), 'size': output.stat().st_size}
                atomic_json(run_dir / 'run.json', record)

    @staticmethod
    def _check_license(case_dir):
        for name in ('lfreport.lis', 'LFERR.LIS'):
            path = case_dir / name
            if path.is_file():
                text = path.read_text(encoding='gbk', errors='replace')
                if '软件狗不响应' in text or '许可证不可用' in text:
                    raise RuntimeError('PSASP 许可证不可用 / license unavailable: 软件狗不响应')

    def _run_once(self, case_dir, record):
        case_dir = Path(case_dir).resolve()
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
        self._archive_outputs(case_dir)
        report_file = case_dir / "lfreport.lis"
        cal_file = case_dir / "LF.CAL"

        started = time.monotonic()
        try:
            completed = subprocess.run(
                [str(self.executable), "-nw"],
                cwd=case_dir,
                check=False,
                timeout=self.timeout,
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            record['category'] = 'process_timeout'
            raise TimeoutError(f"PSASP calculation exceeded {self.timeout:g}s: {case_dir}") from exc
        record['return_code'] = completed.returncode
        record['stdout'] = (getattr(completed, 'stdout', b'') or b'').decode('gbk', errors='replace')
        record['stderr'] = (getattr(completed, 'stderr', b'') or b'').decode('gbk', errors='replace')
        self._check_license(case_dir)
        if completed.returncode != 0:
            record['category'] = 'nonzero_exit'
            raise RuntimeError(f"PSASP exited with code {completed.returncode}: {case_dir}")

        deadline = started + self.timeout
        while time.monotonic() < deadline:
            self._check_license(case_dir)
            if (
                report_file.is_file() and report_file.stat().st_size > 0
                and cal_file.is_file() and cal_file.stat().st_size > 0
            ):
                break
            time.sleep(0.2)
        else:
            record['category'] = 'missing_fresh_outputs'
            raise MissingFreshOutputs(
                f"PSASP did not create fresh LF.CAL and lfreport.lis within "
                f"{self.timeout:g}s: {case_dir}"
            )

        report_text = report_file.read_text(encoding="gbk", errors="replace")
        self._check_license(case_dir)
        error = convergence_error(case_dir)
        if error is None or not math.isfinite(error):
            record['category'] = 'invalid_error_value'
            raise RuntimeError(f"Unable to parse convergence error from fresh report: {report_file}")
        cal_text = cal_file.read_text(encoding='gbk', errors='replace')
        if not re.match(r'\s*(?:-1|0|1)\s*,', cal_text):
            record['category'] = 'invalid_convergence_flag'
            raise RuntimeError('Malformed fresh LF.CAL convergence flag')
        # LF.CAL owns the acceptance decision.  The report residual remains a
        # diagnostic/reward signal and is never allowed to override LF.CAL.
        return SolverResult(convergence_status(case_dir), error, completed.returncode)


class CaseWorkspace:
    """Isolated mutable copy of a source case."""

    def __init__(self, source: Path | str):
        self.source = Path(source).resolve()
        self._temp: tempfile.TemporaryDirectory | None = None
        self.path: Path | None = None

    def open(self) -> Path:
        self.close()
        self._temp = tempfile.TemporaryDirectory(prefix="rl_poweradj_")
        self.path = Path(self._temp.name) / self.source.name
        shutil.copytree(self.source, self.path)
        return self.path

    def close(self) -> None:
        if self._temp is not None:
            self._temp.cleanup()
        self._temp = None
        self.path = None

    def __enter__(self) -> Path:
        return self.open()

    def __exit__(self, *_args) -> None:
        self.close()
