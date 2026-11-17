"""SolverRunner 抽象接口与结果类型(方案 7.0 / D16)。

全部上层代码只依赖本接口;PSASP 实现封装在 psasp.py,Mock 实现在 mock.py。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple


@dataclass
class FaultSpec:
    """故障描述(→ ST.S11)。支路三元组定位 + 相别/接地/短路断线 + 时序(方案 4.2)。"""
    i_no: int
    j_no: int
    id_no: int          # 回路号(双回消歧,必填)
    k_pct: float = 50.0  # 故障点距 I 侧百分比
    phases: Tuple[int, int, int] = (1, 1, 1)  # A,B,C
    ground: int = 1
    short: int = 1
    open_: int = 0
    ts: float = 0.0
    te: float = 0.1     # 切除时刻
    r: float = 0.0
    x: float = 0.0


@dataclass
class PFResult:
    converged: bool
    iterations: int = 0
    bus_voltage: Optional[object] = None   # DataFrame/记录列表(LF.LP1)
    branch_flow: Optional[object] = None   # LF.LP2/LP3
    gen_output: Optional[object] = None     # LF.LP5
    raw_dir: Optional[Path] = None


@dataclass
class TSResult:
    feasible: bool                          # STERR 无 init 错
    stable: bool
    verdict_detail: Dict = field(default_factory=dict)  # 三判据逐项
    extrema: Dict = field(default_factory=dict)          # STANA.DAT 摘要
    traj_dir: Optional[Path] = None


@dataclass
class CCTResult:
    cct_value: float                        # infeasible 时为 nan
    is_censored: bool                       # 上界处仍稳定
    n_sims: int = 0
    infeasible: bool = False                # 初始化失败/无产物:单独打标,不入稳/失稳统计(7.3)


@dataclass
class Job:
    case_dir: Path
    kind: str                               # 'pf' | 'ts' | 'cct'
    fault: Optional[FaultSpec] = None
    overrides: Optional[dict] = None
    tag: str = ""


class SolverRunner(Protocol):
    def run_powerflow(self, case_dir: Path, overrides: Optional[dict] = None) -> PFResult: ...

    def run_transient(self, case_dir: Path, fault: FaultSpec, t_end: float) -> TSResult: ...

    def search_cct(self, case_dir: Path, fault: FaultSpec, lo: float, hi: float,
                   tol: float = 0.01) -> CCTResult:
        """二分**故障持续时间**(切除时长 = Te − Ts);cct_value 即临界持续时间。"""
        ...

    def batch_run(self, jobs: List[Job], n_workers: int, serial: bool = False) -> List[object]: ...
