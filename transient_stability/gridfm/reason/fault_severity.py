"""故障电气严重度侧通道特征(W2;推理器侧 D13,不进编码器)。

从**已解方式标准档案 + 故障参数**(故障支路行、K% 位置)纯 CPU 自算一批不用时域仿真的
静态电气严重度标量:故障点戴维南阻抗、三相短路电流/容量、近区惯量、短路容量/惯量比。
合规先例 = DC-θ 侧通道(pf_physics.dc_theta):只读拓扑/静态阻抗/机组内电抗/故障前电压
(任务输入既有量),不碰任何暂态产物(标签结果字段)→ 无泄漏(单测 test_fault_severity 锁)。

**近似(如实,与 DC-θ 同级)**:正序/三相对称;变压器变比 tk≈1(档案无 tk);只有暂态电抗
x'd(xdp),无零/负序、无次暂态。这些是输入特征非真值,近似可接受。

**物理性质提示**:zth/zline/near_inertia 等由拓扑+静态参数定 → 同一拓扑跨方式恒定(结构量,
随故障位置变、不随运行点变);scc_current/scc_capacity 经故障前电压 |V| 引入运行点依赖。
翻转切面(组内同线)只检验运行点信号 → 结构量在其上必近零,这是设计使然,如实测定。
"""
from __future__ import annotations

import numpy as np

from ..schema.fields import BRANCH_TABLE, BRANCH_TYPES, NODE_TABLE

_EPS = 1e-9
_XFMR = {"xfmr_2w", "phase_shifter", "xfmr_3w"}


def build_ybus(raw) -> np.ndarray:
    """从原始档案构 AC Ybus [n,n] 复数:在运支路 π 模型(变压器以 r1/x1、tk≈1 纳入)+
    并补对地 + 机组内电抗 1/(j·xdp) 加对角(has_gen 节点)。零阻抗/未投运支路跳过。"""
    boff, noff = BRANCH_TABLE.offset_of, NODE_TABLE.offset_of
    n = raw.n_nodes
    Y = np.zeros((n, n), dtype=complex)
    if raw.n_branches:
        r = raw.branch_feat[:, boff("r")].astype(np.float64)
        x = raw.branch_feat[:, boff("x")].astype(np.float64)
        bh = raw.branch_feat[:, boff("b_half")].astype(np.float64)
        mark = raw.branch_feat[:, boff("mark")]
        onehot = raw.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
        bt = np.asarray(BRANCH_TYPES)[np.argmax(onehot, axis=1)]
        ei, ej = raw.edge_index
        z = r + 1j * x
        for k in range(len(r)):
            if mark[k] != 1:
                continue
            i, j = int(ei[k]), int(ej[k])
            if bt[k] == "shunt_comp":
                Y[i, i] += 1j * 2.0 * bh[k]          # 并补:I 侧对地(与 pf_phys_arrays 同)
                continue
            if abs(z[k]) <= _EPS:
                continue
            y = 1.0 / z[k]                             # 变压器 tk≈1 近似(档案无 tk)
            Y[i, i] += y + 1j * bh[k]
            Y[j, j] += y + 1j * bh[k]
            Y[i, j] -= y
            Y[j, i] -= y
    # 机组内电抗(暂态):has_gen 且 xdp>0 → 机端对地 1/(j·xdp)
    has_gen = raw.node_feat[:, noff("has_gen")] > 0.5
    null = raw.node_feat[:, noff("is_null")] > 0.5
    xdp = raw.node_feat[:, noff("xdp")].astype(np.float64)
    for g in np.where(has_gen & ~null & (xdp > _EPS))[0]:
        Y[g, g] += 1.0 / (1j * xdp[g])
    return Y


def _zbus(Y: np.ndarray) -> np.ndarray:
    """Zbus = inv(Ybus);奇异(孤岛/无参考)时加微对角正则。"""
    n = Y.shape[0]
    try:
        return np.linalg.inv(Y + 1e-9 * np.eye(n))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(Y)


def _z_th_at_fault(Y, Zbus, i, j, z_L, kf):
    """故障点(线 i-j 上距 i 端 kf 处)戴维南阻抗 Z_th。

    kf≈0 → 母线 i(Z_th=Zbus[i,i]);kf≈1 → 母线 j;否则拆线加内节点 F(去掉 i-j 原支路,
    接 i-F=kf·z_L 与 F-j=(1-kf)·z_L)重求 Z_th=inv[F,F]。z_L 无效则退回母线 i。
    """
    if abs(z_L) <= _EPS or i == j:
        return Zbus[i, i]
    if kf <= 1e-6:
        return Zbus[i, i]
    if kf >= 1 - 1e-6:
        return Zbus[j, j]
    n = Y.shape[0]
    Ym = np.zeros((n + 1, n + 1), dtype=complex)
    Ym[:n, :n] = Y
    y_L = 1.0 / z_L
    Ym[i, i] -= y_L; Ym[j, j] -= y_L; Ym[i, j] += y_L; Ym[j, i] += y_L   # 去掉原 i-j 支路
    y1 = 1.0 / (kf * z_L)             # i - F
    y2 = 1.0 / ((1.0 - kf) * z_L)     # F - j
    f = n
    Ym[i, i] += y1; Ym[f, f] += y1; Ym[i, f] -= y1; Ym[f, i] -= y1
    Ym[j, j] += y2; Ym[f, f] += y2; Ym[j, f] -= y2; Ym[f, j] -= y2
    return _zbus(Ym)[f, f]


def compute_fault_severity_features(raw, fault_branch_row: int, k_frac: float) -> dict:
    """已解档案 + 故障(支路行 + K% 位置)→ 故障电气严重度标量集(纯静态,无暂态仿真)。

    fault_branch_row 索引 branch_feat / edge_index;k_frac = K/100(0=线首 i 端)。
    返回标量 dict(全部为合法输入的确定函数;泄漏单测锁其对标签结果字段不变)。
    """
    boff, noff = BRANCH_TABLE.offset_of, NODE_TABLE.offset_of
    ei, ej = raw.edge_index
    i = int(ei[fault_branch_row]); j = int(ej[fault_branch_row])
    z_L = complex(raw.branch_feat[fault_branch_row, boff("r")],
                  raw.branch_feat[fault_branch_row, boff("x")])
    Y = build_ybus(raw)
    Zbus = _zbus(Y)
    z_th = _z_th_at_fault(Y, Zbus, i, j, z_L, float(k_frac))
    zth_mag = float(abs(z_th))

    # 故障前电压(节点 i 的 |v|;缺失/0 退回 1.0)——引入运行点依赖
    v_i = float(raw.node_feat[i, noff("v")])
    if not np.isfinite(v_i) or v_i <= 0.01:
        v_i = 1.0

    scc_current = v_i / max(zth_mag, _EPS)
    scc_capacity = v_i * v_i / max(zth_mag, _EPS)

    # 机组惯量:全网 tj 和 + 近区(按到故障母线 i 的电气距离 |Zbus[i,g]| 倒数加权)
    has_gen = raw.node_feat[:, noff("has_gen")] > 0.5
    null = raw.node_feat[:, noff("is_null")] > 0.5
    tj = raw.node_feat[:, noff("tj")].astype(np.float64)
    gens = np.where(has_gen & ~null & (tj > _EPS))[0]
    total_inertia = float(tj[gens].sum()) if len(gens) else 0.0
    near_inertia = 0.0
    for g in gens:
        near_inertia += tj[g] / (abs(Zbus[i, g]) + 1e-6)
    scc_over_near_inertia = scc_capacity / (near_inertia + _EPS)

    return {
        "fs_v_fault": v_i,           # 故障母线故障前电压(归因用:scc 组内信号是否即此平凡读数)
        "fs_zth_mag": zth_mag,
        "fs_zth_x": float(z_th.imag),
        "fs_scc_current": scc_current,
        "fs_scc_capacity": scc_capacity,
        "fs_zline_mag": float(abs(z_L)),
        "fs_total_inertia": total_inertia,
        "fs_near_inertia": float(near_inertia),
        "fs_scc_over_inertia": float(scc_over_near_inertia),
    }


FEATURE_NAMES = ("fs_v_fault", "fs_zth_mag", "fs_zth_x", "fs_scc_current", "fs_scc_capacity",
                 "fs_zline_mag", "fs_total_inertia", "fs_near_inertia", "fs_scc_over_inertia")
