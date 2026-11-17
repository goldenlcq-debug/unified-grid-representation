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
    is_censored: bool                       # **上界**处仍稳定(CCT > hi)
    n_sims: int = 0
    infeasible: bool = False                # 初始化失败/无产物:单独打标,不入稳/失稳统计(7.3)
    # **下侧删失**(2026-07-27 对抗审查 critical):二分区间内**没有一个 mid 稳定**时,
    # `last_stable` 仍停在初值 `lo`,而 **`lo` 从未被仿真过** —— 旧码把它当成一个精确点值
    # `cct=0.0` 落档(`is_censored=False`),于是:①G2 可用率把它算成"可用";
    # ②`--verify` 会跑 `te = ts + 0` 的**零时长故障卡**,引擎判"无扰动→稳" → 双验**空洞
    # 通过**、白烧 2 次仿真并灌水 G1;③`ts_task` 按非删失走 L1 精确回归,把"CCT=0"教给
    # `head_cct`。故与上侧删失对称地单独打标。**上下删失一律不入点值统计。**
    is_censored_low: bool = False           # 下界处已失稳(CCT < lo,真值未测)


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
