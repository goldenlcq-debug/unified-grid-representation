"""动态参数/能量函数侧通道特征(审查 A2;D13 推理器侧,不进编码器)。

**动因(信息上限第五因子)**:方案 D8 宣称"网络 + 动态参数 + 故障事件"直接判稳,但
生产判稳模型从无路径读到 Tj/X'd——fault_severity(W2)读的是 v0 legacy tj/xdp 列,
而 cepri36/ieee39 的该两列是 ST.S5 内联假 0(fields.py v0.3 注释)。本模块从 v0.3
档案列读 **DATALIB 正确值**(tj_lib/xdp_lib;118 无 DATALIB 段,该列为 ST.S5 内联
真值、dyn_resolved=0,如实),另配能量函数代理特征(加速能量 ~ Pm·dur²/2Tj)。

**输入合法性(泄漏红线同 fault_severity)**:只消费 v0.3 DYNAMIC 给定列 + 故障参数
(支路行、切除时长 dur)+ 故障前档案给定量(pg);不读任何标签结果字段(单测锁定)。
电气距离沿用 fault_severity 先例 |Zbus[i,g]|(戴维南,故障支路 i 端),但 Ybus 机组
内电抗用 xdp_lib 正确值(xdp_lib≤0 时退 legacy xdp,如实计数由调用方负责)。

**要求 full_schema 档案**(GridSample.load_npz(full_schema=True),43 列);v0 视图
25 列样本直接 raise——静默退化成假 0 比报错更危险。
"""
from __future__ import annotations

import numpy as np

from ..schema.fields import BRANCH_TABLE, BRANCH_TYPES, NODE_TABLE_V03

_EPS = 1e-9

# 特征清单(实现即冻结;改动须记变更记录)
DYN_FEATURE_NAMES = (
    "dyn_tj_sum",       # 全网机组惯量 ΣTj(系统惯量,DATALIB 正确值)
    "dyn_tj_mean",      # 机组惯量均值
    "dyn_tj_min",       # 最小机组惯量(最易失步机的惯量下限)
    "dyn_xdp_mean",     # X'd 均值
    "dyn_tjw_xdp",      # 惯量加权 X'd = Σ(Tj·X'd)/ΣTj
    "dyn_near_zdist",   # 故障点到最近机组的电气距离 min|Zbus[i,g]|
    "dyn_near_tj",      # 该最近机组的 Tj
    "dyn_near_xdp",     # 该最近机组的 X'd
    "dyn_prox_tj",      # 故障近区惯量 ΣTj_g/(|Zbus[i,g]|+1e-6)(修正参数版 fs_near_inertia)
    "dyn_acc_proxy",    # 能量函数代理:pg_near·dur²/(2·Tj_near)(等面积加速能量近似)
)


def _build_ybus_v03(full) -> np.ndarray:
    """v0.3 档案 → AC Ybus(与 fault_severity.build_ybus 同构;机组内电抗取 xdp_lib
    正确值,≤0 退 legacy xdp)。"""
    boff, noff = BRANCH_TABLE.offset_of, NODE_TABLE_V03.offset_of
    n = full.n_nodes
    Y = np.zeros((n, n), dtype=complex)
    if full.n_branches:
        r = full.branch_feat[:, boff("r")].astype(np.float64)
        x = full.branch_feat[:, boff("x")].astype(np.float64)
        bh = full.branch_feat[:, boff("b_half")].astype(np.float64)
        mark = full.branch_feat[:, boff("mark")]
        onehot = full.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
        bt = np.asarray(BRANCH_TYPES)[np.argmax(onehot, axis=1)]
        ei, ej = full.edge_index
        z = r + 1j * x
        for k in range(len(r)):
            if mark[k] != 1:
                continue
            i, j = int(ei[k]), int(ej[k])
            if bt[k] == "shunt_comp":
                Y[i, i] += 1j * 2.0 * bh[k]
                continue
            if abs(z[k]) <= _EPS:
                continue
            y = 1.0 / z[k]
            Y[i, i] += y + 1j * bh[k]
            Y[j, j] += y + 1j * bh[k]
            Y[i, j] -= y
            Y[j, i] -= y
    has_gen = full.node_feat[:, noff("has_gen")] > 0.5
    null = full.node_feat[:, noff("is_null")] > 0.5
    xdp = full.node_feat[:, noff("xdp_lib")].astype(np.float64)
    xdp_leg = full.node_feat[:, noff("xdp")].astype(np.float64)
    xdp = np.where(xdp > _EPS, xdp, xdp_leg)
    for g in np.where(has_gen & ~null & (xdp > _EPS))[0]:
        Y[g, g] += 1.0 / (1j * xdp[g])
    return Y


def dyn_zbus(full) -> np.ndarray:
    """Zbus = inv(Ybus)。**只依赖档案**(拓扑 + 运行点 + 机组内电抗),与故障支路/切除
    时长无关——调用方按 sample_id 缓存一次,即可让同一方式的所有故障共用,避免逐故障
    重复求逆(compute_dyn_side_features 的热点)。奇异时退 pinv,与内联口径逐位一致。"""
    if full.node_feat.shape[1] != NODE_TABLE_V03.total_dim:
        raise ValueError(
            f"dyn_side 要求 v0.3 全 schema 档案({NODE_TABLE_V03.total_dim} 列),"
            f"得到 {full.node_feat.shape[1]} 列——请用 load_npz(full_schema=True)")
    Y = _build_ybus_v03(full)
    n = Y.shape[0]
    try:
        return np.linalg.inv(Y + 1e-9 * np.eye(n))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(Y)


def compute_dyn_side_features(full, fault_branch_row: int, dur: float,
                              zbus: "np.ndarray | None" = None) -> dict:
    """v0.3 全 schema 档案 + 故障(支路行 + 切除时长)→ 动态参数/能量函数标量集。

    全部为合法输入(给定列 + 故障参数)的确定函数;不读标签结果字段。
    机组集 = has_gen & ~is_null & tj_lib>0;无机组(异常)返回全 0。

    zbus 给定时跳过 Ybus 构建 + 求逆直接复用(Zbus 只依赖 full,与故障无关);
    None 则内部现算(dyn_zbus),两条路径数值逐位一致(单测锁)。
    """
    noff = NODE_TABLE_V03.offset_of
    if full.node_feat.shape[1] != NODE_TABLE_V03.total_dim:
        raise ValueError(
            f"dyn_side 要求 v0.3 全 schema 档案({NODE_TABLE_V03.total_dim} 列),"
            f"得到 {full.node_feat.shape[1]} 列——请用 load_npz(full_schema=True)")
    has_gen = full.node_feat[:, noff("has_gen")] > 0.5
    null = full.node_feat[:, noff("is_null")] > 0.5
    tj = full.node_feat[:, noff("tj_lib")].astype(np.float64)
    xdp = full.node_feat[:, noff("xdp_lib")].astype(np.float64)
    xdp_leg = full.node_feat[:, noff("xdp")].astype(np.float64)
    xdp = np.where(xdp > _EPS, xdp, xdp_leg)
    pg = full.node_feat[:, noff("pg")].astype(np.float64)
    gens = np.where(has_gen & ~null & (tj > _EPS))[0]
    if len(gens) == 0:
        return {n: 0.0 for n in DYN_FEATURE_NAMES}

    tj_g, xdp_g = tj[gens], xdp[gens]
    tj_sum = float(tj_g.sum())
    Zbus = dyn_zbus(full) if zbus is None else zbus
    i = int(full.edge_index[0][fault_branch_row])
    zdist = np.abs(Zbus[i, gens])
    k_near = int(np.argmin(zdist))                       # 平局取行序最小,确定性
    g_near = int(gens[k_near])
    near_tj = float(tj_g[k_near])
    prox_tj = float((tj_g / (zdist + 1e-6)).sum())
    acc = float(abs(pg[g_near]) * float(dur) ** 2 / (2.0 * max(near_tj, _EPS)))
    return {
        "dyn_tj_sum": tj_sum,
        "dyn_tj_mean": float(tj_g.mean()),
        "dyn_tj_min": float(tj_g.min()),
        "dyn_xdp_mean": float(xdp_g.mean()),
        "dyn_tjw_xdp": float((tj_g * xdp_g).sum() / max(tj_sum, _EPS)),
        "dyn_near_zdist": float(zdist[k_near]),
        "dyn_near_tj": near_tj,
        "dyn_near_xdp": float(xdp_g[k_near]),
        "dyn_prox_tj": prox_tj,
        "dyn_acc_proxy": acc,
    }
