"""numpy 牛顿-拉夫逊潮流(白盒诊断/快速可行性筛查;PSASP 仍是标签与验收的终审)。

从 PsaspCase 构建 Y 阵(交流线 π 模型 + 变压器变比支路 + 并补),平启动 NR,
支持 PV→PQ 无功越限切换。返回完整诊断:逐母线 V/θ、失配、雅可比条件数。

用途:①合成算例的发散根因定位;②合成修复内环(比整进程调用 PSASP 快百倍);
③与 LP1 对照可交叉验证解析器与引擎口径。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..psasp_io.case import PsaspCase


@dataclass
class NRResult:
    converged: bool
    iters: int
    max_mismatch: float
    v: np.ndarray
    theta: np.ndarray
    diag: Dict = field(default_factory=dict)


def build_ybus(case: PsaspCase, n: int) -> np.ndarray:
    Y = np.zeros((n, n), dtype=complex)
    for l in case.aclines():
        if l.mark != 1:
            continue
        i, j = l.i_no - 1, l.j_no - 1
        if l.branch_type == "shunt_comp":
            # 并补:挂 I 侧母线的对地导纳(B/2 字段语义按线对待:jB)
            Y[i, i] += 1j * l.b_half * 2
            continue
        z = complex(l.r, l.x)
        if abs(z) < 1e-9:
            continue
        y = 1.0 / z
        Y[i, i] += y + 1j * l.b_half
        Y[j, j] += y + 1j * l.b_half
        Y[i, j] -= y
        Y[j, i] -= y
    for t in case.xfmrs():
        if t.mark != 1:
            continue
        i, j = t.i_no - 1, t.j_no - 1
        z = complex(t.r1, t.x1)
        if abs(z) < 1e-9:
            continue
        y = 1.0 / z
        k = t.tk if abs(t.tk) > 1e-6 else 1.0
        # 变比在 I 侧的标准 π 等效
        Y[i, i] += y / (k * k)
        Y[j, j] += y
        Y[i, j] -= y / k
        Y[j, i] -= y / k
    return Y


def solve_nr(case: PsaspCase, max_iter: int = 40, tol: float = 1e-6,
             enforce_qlim: bool = True) -> NRResult:
    n = case.n_buses
    Y = build_ybus(case, n)
    G, B = Y.real, Y.imag

    Pinj = np.zeros(n); Qinj = np.zeros(n)
    is_gen = np.zeros(n, bool); is_slack = np.zeros(n, bool)
    vset = np.ones(n); qmax = np.full(n, np.inf); qmin = np.full(n, -np.inf)
    for g in case.gens():
        if g.mark != 1:
            continue
        b = g.bs_no - 1
        if g.type_code == 0:
            is_slack[b] = True; vset[b] = g.v0
        else:
            is_gen[b] = True; vset[b] = g.v0
            Pinj[b] += g.pg
            qmax[b], qmin[b] = g.qmax, g.qmin
    for ld in case.loads():
        if ld.mark != 1:
            continue
        b = ld.bs_no - 1
        Pinj[b] -= ld.pl; Qinj[b] -= ld.ql

    # 孤立母线(NULL/无连接)按平电压处理,不进方程
    deg = (np.abs(Y) > 1e-12).sum(1) - 1
    active = deg > 0
    pv = is_gen & active & ~is_slack
    slack = is_slack & active
    pq = active & ~pv & ~slack

    v = vset.copy(); v[pq] = 1.0
    th = np.zeros(n)
    flip_at = np.zeros(n, dtype=np.int8)   # PV→PQ 翻转记录:+1 卡 Qmax,-1 卡 Qmin

    def calc_pq():
        vv = v * np.exp(1j * th)
        s = vv * np.conj(Y @ vv)
        return s.real, s.imag

    hist = []
    for it in range(max_iter):
        Pc, Qc = calc_pq()
        dP = Pinj - Pc
        dQ = Qinj - Qc
        # PV 无功越限切换(机组需发 Q = 计算注入 - 负荷注入)+ 回翻(教科书要求:
        # 卡上限后 V 升回设定值以上 → 机组必然减 Q,应回 PV;否则解的是错题)。
        # 15 步后冻结切换状态:消灭 flip/unflip 极限环(标准工程做法)
        if enforce_qlim and 2 <= it <= 15:
            for b in np.where(pv)[0]:
                qg = Qc[b] - Qinj[b]
                if qg > qmax[b] + 1e-6:
                    pv[b] = False; pq[b] = True; flip_at[b] = 1; Qinj[b] += qmax[b]
                elif qg < qmin[b] - 1e-6:
                    pv[b] = False; pq[b] = True; flip_at[b] = -1; Qinj[b] += qmin[b]
            for b in np.where(flip_at != 0)[0]:
                if (flip_at[b] == 1 and v[b] > vset[b] + 1e-6) or \
                   (flip_at[b] == -1 and v[b] < vset[b] - 1e-6):
                    Qinj[b] -= qmax[b] if flip_at[b] == 1 else qmin[b]
                    flip_at[b] = 0
                    pq[b] = False; pv[b] = True; v[b] = vset[b]
        mis = np.concatenate([dP[pv | pq], dQ[pq]])
        mx = float(np.max(np.abs(mis))) if mis.size else 0.0
        hist.append(mx)
        if mx < tol:
            return NRResult(True, it, mx, v, th, {"hist": hist})
        if it > 8 and mx > 1e4:            # 提前放弃:已进非物理域,继续迭代无意义
            return NRResult(False, it, mx, v, th, {"hist": hist, "bailout": True})
        # 雅可比(向量化,MATPOWER 式复数恒等式)
        idx_a = np.where(pv | pq)[0]      # dθ 变量
        idx_v = np.where(pq)[0]           # dV 变量
        na, nv = len(idx_a), len(idx_v)
        Vc = v * np.exp(1j * th)
        Ibus = Y @ Vc
        Vn = Vc / np.maximum(v, 1e-9)
        # 广播替代 diag 矩阵乘(N=1000 时快一个量级)
        YdV = Y * Vc[None, :]                       # Y @ diag(V)
        M = -YdV
        M[np.arange(n), np.arange(n)] += Ibus       # diag(I) - Y·diag(V)
        dS_dVa = 1j * Vc[:, None] * np.conj(M)
        dS_dVm = Vc[:, None] * np.conj(Y * Vn[None, :])
        dS_dVm[np.arange(n), np.arange(n)] += np.conj(Ibus) * Vn
        J = np.block([
            [dS_dVa.real[np.ix_(idx_a, idx_a)], dS_dVm.real[np.ix_(idx_a, idx_v)]],
            [dS_dVa.imag[np.ix_(idx_v, idx_a)], dS_dVm.imag[np.ix_(idx_v, idx_v)]],
        ])
        try:
            dx = np.linalg.solve(J, mis)
        except np.linalg.LinAlgError:
            return NRResult(False, it, mx, v, th,
                            {"hist": hist, "singular": True,
                             "cond": float(np.linalg.cond(J))})
        # 阻尼牛顿:失配变大则回退半步(最多 5 次)——区分"真不可行"与"步长过冲"
        th0, v0 = th.copy(), v.copy()
        norm0 = float(np.linalg.norm(mis))
        step = 1.0
        improved = False
        for _bt in range(6):
            th = th0.copy(); v = v0.copy()
            th[idx_a] += step * dx[:na]
            v[idx_v] += step * dx[na:]          # 绝对 dV(与向量化雅可比一致)
            Pc2, Qc2 = calc_pq()
            mis2 = np.concatenate([(Pinj - Pc2)[pv | pq], (Qinj - Qc2)[pq]])
            if mis2.size and float(np.linalg.norm(mis2)) < norm0 * 0.9999:
                improved = True                 # 2 范数验收(max 范数验收会在切换边界脆性卡死)
                break
            step *= 0.5
        if not improved:                        # 全部步长都不降:回退并停(避免硬走坏步震荡)
            th, v = th0, v0
            return NRResult(False, it, mx, v, th, {"hist": hist, "stalled": True})
    Pc, Qc = calc_pq()
    dP = Pinj - Pc; dQ = Qinj - Qc
    worst = int(np.argmax(np.abs(np.where(pv | pq, dP, 0) )))
    return NRResult(False, max_iter, hist[-1] if hist else np.inf, v, th,
                    {"hist": hist, "worst_p_bus": worst + 1,
                     "worst_p": float(dP[worst])})
