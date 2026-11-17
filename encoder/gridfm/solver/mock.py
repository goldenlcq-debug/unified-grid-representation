"""MockSolver(方案 7.0 / T2.1):从 golden fixture 回放结果,使无 PSASP 环境可开发/单测/CI。

潮流:读 case_dir 已有的 LF.CAL/LP1/LP2/LP5 回放(fixture 已含求解结果)。
暂态/CCT:按 fixture 中预置的 verdict 表回放,或给出确定性占位结论。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..psasp_io import cards
from ..psasp_io.lowlevel import Card
from .base import CCTResult, FaultSpec, Job, PFResult, SolverRunner, TSResult


def _resolve(case_dir: Path, name: str) -> Optional[Path]:
    for cand in (case_dir / name, case_dir / f"{name}.txt"):
        if cand.exists():
            return cand
    return None


class MockSolver(SolverRunner):
    def __init__(self, force_converged: Optional[bool] = None,
                 force_stable: Optional[bool] = None) -> None:
        self.force_converged = force_converged
        self.force_stable = force_stable

    def run_powerflow(self, case_dir: Path, overrides: Optional[dict] = None) -> PFResult:
        case_dir = Path(case_dir)
        cal = _resolve(case_dir, "LF.CAL")
        if self.force_converged is not None:
            converged = self.force_converged
        elif cal is not None:
            converged = cards.parse_cal_converged(Card.load(cal)) or False
        else:
            converged = False
        bus = _resolve(case_dir, "LF.LP1")
        br = _resolve(case_dir, "LF.LP2")
        gen = _resolve(case_dir, "LF.LP5")
        return PFResult(
            converged=converged,
            iterations=0,
            bus_voltage=cards.parse_lp1(Card.load(bus)) if bus else None,
            branch_flow=cards.parse_lp2(Card.load(br)) if br else None,
            gen_output=cards.parse_lp5(Card.load(gen)) if gen else None,
            raw_dir=case_dir,
        )

    def run_transient(self, case_dir: Path, fault: FaultSpec, t_end: float) -> TSResult:
        stable = True if self.force_stable is None else self.force_stable
        return TSResult(
            feasible=True,
            stable=stable,
            verdict_detail={"angle": stable, "voltage": stable, "freq": stable, "mock": True},
            extrema={},
        )

    def search_cct(self, case_dir: Path, fault: FaultSpec, lo: float, hi: float,
                   tol: float = 0.01) -> CCTResult:
        # 占位:返回区间中点,标未删失;真实实现见 psasp.py
        return CCTResult(cct_value=round((lo + hi) / 2, 2), is_censored=False, n_sims=0)

    def batch_run(self, jobs: List[Job], n_workers: int, serial: bool = False) -> List[object]:
        out: List[object] = []
        for j in jobs:
            if j.kind == "pf":
                out.append(self.run_powerflow(j.case_dir, j.overrides))
            elif j.kind == "ts":
                out.append(self.run_transient(j.case_dir, j.fault, t_end=10.0))
            elif j.kind == "cct":
                out.append(self.search_cct(j.case_dir, j.fault, 0.0, 0.5))
            else:
                raise ValueError(f"未知作业类型 {j.kind}")
        return out
