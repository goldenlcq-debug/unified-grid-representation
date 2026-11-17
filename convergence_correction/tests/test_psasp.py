import pytest
from types import SimpleNamespace

from rl_poweradj.paths import ProjectPaths
from rl_poweradj.psasp import CaseWorkspace, PSASPRunner, convergence_error, read_records


def test_safe_record_parser_reads_sample():
    case = ProjectPaths.from_root().cases("train")[0]
    l2 = read_records(case / "LF.L2")
    assert len(l2) > 0
    assert all(len(row) >= 3 for row in l2)
    assert any(abs(int(row[1])) == abs(int(row[2])) for row in l2)


def test_workspace_does_not_modify_source():
    case = ProjectPaths.from_root().cases("train")[0]
    original = (case / "LF.L2").read_bytes()
    with CaseWorkspace(case) as workspace:
        (workspace / "LF.L2").write_bytes(b"changed")
    assert (case / "LF.L2").read_bytes() == original


def test_report_parser_accepts_scientific_and_extra_columns(tmp_path):
    (tmp_path / "lfreport.lis").write_text(
        " 0 BUS1 2.5E-2 diagnostic\n 1 BUS2 .000001 extra\n", encoding="gbk"
    )
    assert convergence_error(tmp_path) == 0.000001


def test_runner_archives_stale_outputs(tmp_path):
    (tmp_path / "LF.CAL").write_text("1, stale", encoding="gbk")
    (tmp_path / "lfreport.lis").write_text("0 BUS1 0.0", encoding="gbk")
    PSASPRunner._archive_outputs(tmp_path)
    assert not (tmp_path / "LF.CAL").exists()
    assert not (tmp_path / "lfreport.lis").exists()
    assert (tmp_path / "LF.CAL.previous").exists()


@pytest.mark.parametrize(
    ("cal_status", "residual", "expected"),
    [(0, 0.0, False), (1, 1.0, True)],
)
def test_runner_uses_fresh_cal_as_authoritative_status(
    tmp_path, monkeypatch, cal_status, residual, expected
):
    executable = tmp_path / "solver.exe"
    executable.write_bytes(b"stub")

    def fake_run(*_args, **_kwargs):
        (tmp_path / "LF.CAL").write_text(f"{cal_status}, fresh\n", encoding="gbk")
        (tmp_path / "lfreport.lis").write_text(
            f"0 BUS1 {residual}\n", encoding="gbk"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("rl_poweradj.psasp.subprocess.run", fake_run)
    result = PSASPRunner(executable, timeout=1).run(tmp_path)
    assert result.converged is expected
    assert result.error == residual


def test_solver_smoke_on_temporary_case():
    paths = ProjectPaths.from_root()
    source = paths.cases("train")[0]
    with CaseWorkspace(source) as workspace:
        try:
            result = PSASPRunner(paths.solver, timeout=30).run(workspace)
        except RuntimeError as exc:
            if "许可证不可用" in str(exc):
                pytest.skip("PSASP executable is present but its hardware/software license is unavailable")
            raise
        assert result.return_code == 0
        assert result.error >= 0.0
