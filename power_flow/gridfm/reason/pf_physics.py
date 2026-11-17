"""潮流物理残差损失(P2 θ 突破杠杆 a;方案 D13 注记 / 架构问答问 4)。

预测 V/θ(与 qg/平衡机 pg)代回基尔霍夫节点功率方程算失配进损失。方程为极坐标
节点功率方程 + 支路 π 模型的直接实现,**非仿真器、无迭代、无裁决权**,不碰 9b
(PSASP 唯一仿真平台);只在损失侧使用,不进编码器输入,不改 D15 输入协议。

**逐节点 Ybus 行完整性口径(v0 决策,2026-07-11)**:GridSample schema 无变压器
变比 tk(v0 字段表;编码器 v1 输入维度已冻结,不可加列),故与在运变压器
(xfmr_2w/xfmr_3w/phase_shifter)或零阻抗支路相邻的节点,其 Ybus 行不完整,
**该节点方程不进残差**;其余节点方程精确(118 golden fixture 实测真值失配 1e-5 级,
单测锁定)。合成家族无变压器 → 全节点可用;三方式库大部分节点可用。

**方程分两类(权重独立)**:
- 干净方程(θ 杠杆主体):PQ 节点 P/Q + PV 节点 P——注入侧全部为给定量
  (机组 pg 卡值 = 调度指定、负荷 pl/ql),只约束预测 V/θ;
- 定义方程:PV 节点 Q + 平衡机 P/Q——把预测 qg/pg 与网络方程绑定,即这些字段的
  **物理定义**。注意:样本存储的 PV/平衡机 qg 与平衡机 pg 为 LF.L5 卡值(解前值),
  与 LP5 解值系统性不符(2026-07-11 实测,账目见变更记录)——定义方程与 L1 监督
  在这些字段上会**相互拉扯**,v0 默认权重 0,待用户裁决数据修复后再启用。

真值失配≈0 的正确性单测:tests/test_pf_physics.py(方案执行清单 1a 强制配套)。
CUDA 上 index_add_ 为原子加,浮点求和次序不定(训练可接受;评估诊断用 CPU 或容忍)。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .. import perf_cache
from ..schema.fields import assert_branch_mark, BRANCH_TABLE, BRANCH_TYPES, NODE_TABLE, Group
from ..schema.sample import GridSample, group_bit_index
from .pf_task import PF_FIELDS

# 残差所需的 6 个节点状态字段(前 4 个与 PF_FIELDS 同序,pred 列直接对应)
PHYS_FIELDS = ("pg", "qg", "v", "theta", "pl", "ql")
assert PHYS_FIELDS[:4] == PF_FIELDS

_XFMR = {"xfmr_2w", "phase_shifter", "xfmr_3w"}
_PQ, _PV, _SLACK = 0, 1, 2


def _ybus_terms(raw: GridSample, tap: Optional[np.ndarray] = None,
                shift: Optional[np.ndarray] = None, shunt_zero_harmless: bool = False) -> Dict[str, np.ndarray]:
    """Ybus 的 COO 项与"方程可建"掩码(物理损失与电压提示共用,口径唯一)。

    `shunt_zero_harmless`(E15,预注册 §19.2 修法 A;默认关 = 逐位兼容):**在运并补记录一律不牵连相邻节点方程**,
    导纳按记录能给的精度计——r=0 → B=−1/x(旧路径,逐位不变);r≠0 → y=1/(r+jx) 的 G、B 全计;r=x=0 才是真空记录,忽略。
    口径修正依据(2026-08-23 数据核实):mp_case300 每方式 15 条 x=0 的并补 r=714~5000,是 MATPOWER 的纯电导并补(Gs),
    存档 pi 与 V²/r 吻合到 5e-6(吃 2e-4~1.2e-3 pu 有功);若按 §18.1 原文当"空记录"忽略,真值代入 |ΔP| 过不了 1e-4。
    另 2 条 r≠0、x≠0 的并补(r≈2.4、x≈−41.5)旧口径忽略 r,真值 ΔP 差 1.3e-3,开关开时一并按 1/z 计。
    影响面:全语料只有 mp_case300(15+2 条/方式)与 lib_cepri36(2 条 r=x=999999 占位,导纳 1e-6 级)有 r≠0 的并补,其余家族逐位不变。

    2026-08-22 修正(结果档 §3.19,全部用真值代入自检过):
      * **变压器**:变比 t 来自家族 sidecar `family_extra.npz`(`branch_tap`,行序 = branch_feat),放 **j 侧**:
        Yii += y,Yjj += y/t²,Yij = Yji = −y/t(真值 ΔP → 0.00000;放 i 侧 0.07~0.28)。
        没给 tap(家族无 sidecar)或移相角 ≠ 0 的变压器 → 与此前一样整支路排除、相邻节点行作废(保守)。
      * **并补**(shunt_comp,I==J):归档把值存在 `x` 列(b_half 恒 0),正确项 **B = −1/x**
        (真值 ΔQ 从 0.03~0.13 → 1e-5;此前按 2·b_half 读 = 恒 0,即漏掉在运并补;118 golden 夹具并补全退运故单测未抓到)。
      * 交流线 π 模型不变:两端各 +j·b_half。
    返回:rows/cols/g/b(COO)、li/lj(串联支路两端,供邻居有效性判断)、incomplete(在运但建不出的支路)。
    """
    boff = BRANCH_TABLE.offset_of
    r = raw.branch_feat[:, boff("r")].astype(np.float64)
    # 归一化样本(log_z)的 r 约半数为负;原始样本中个别负电阻支路是合法等值表示
    # (2026-08-21:ACTIVSg25k 有 447 台 r<0 的两绕组变压器),按多数判
    if r.size and (r < 0).mean() > 0.5:
        raise ValueError("branch r 多数为负——疑似传入了归一化样本,应传原始样本")
    x = raw.branch_feat[:, boff("x")].astype(np.float64)
    bh = raw.branch_feat[:, boff("b_half")].astype(np.float64)
    mark = raw.branch_feat[:, boff("mark")]
    assert_branch_mark(mark, "pf_physics._ybus_terms")
    onehot = raw.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
    bt = np.asarray(BRANCH_TYPES)[np.argmax(onehot, axis=1)] if len(onehot) else np.array([])
    ei, ej = (raw.edge_index if raw.n_branches else (np.zeros(0, np.int64),) * 2)
    m = len(r)
    on = mark == 1
    is_shunt = bt == "shunt_comp"
    is_xfmr = np.isin(bt, list(_XFMR))
    z = r + 1j * x
    nz = np.abs(z) > 1e-9
    if tap is not None:
        tap = np.asarray(tap, dtype=np.float64)
        if tap.shape[0] != m:
            raise ValueError(f"branch_tap 长度 {tap.shape[0]} 与支路数 {m} 不符(sidecar 行序错位?)")
        tap_ok = np.isfinite(tap) & (tap > 1e-6)
        if shift is not None:
            tap_ok &= np.abs(np.asarray(shift, dtype=np.float64)) < 1e-9   # 移相 ≠ 0 → 保守作废
    else:
        tap_ok = np.zeros(m, dtype=bool)
    series = on & ~is_shunt & nz & (~is_xfmr | tap_ok)      # π 模型精确可建的串联支路
    if shunt_zero_harmless:
        shunt = on & is_shunt & (np.abs(x) > 1e-9) & (np.abs(r) <= 1e-9)   # 纯电抗并补:旧路径 B=−1/x(逐位不变)
        shunt_rz = on & is_shunt & (np.abs(r) > 1e-9)                        # 含电阻的并补(含 x=0 纯电导):y=1/(r+jx)
        incomplete = on & ~series & ~is_shunt                                # 并补记录一律不作废相邻行
    else:
        shunt = on & is_shunt & (np.abs(x) > 1e-9)
        shunt_rz = np.zeros(m, dtype=bool)
        incomplete = on & ~series & ~shunt                   # 在运但建不出 → 相邻节点行不完整

    y = np.zeros(m, dtype=complex)
    y[nz] = 1.0 / z[nz]
    li, lj = ei[series].astype(np.int64), ej[series].astype(np.int64)
    ys, bhl = y[series], bh[series]
    t = np.ones(len(ys))
    if tap is not None:
        sel = is_xfmr[series] & tap_ok[series]
        t[sel] = tap[series][sel]
    yij = ys / t                         # 线路 t=1 → 与旧公式逐位相同
    yjj = ys / (t * t)
    si = ei[shunt].astype(np.int64)
    bsh = -1.0 / x[shunt]
    sz = ei[shunt_rz].astype(np.int64)   # 开关关或家族无 r≠0 并补时为空 → 下面的拼接逐位不变
    yz = y[shunt_rz]
    rows = np.concatenate([li, lj, li, lj, si, sz])
    cols = np.concatenate([lj, li, li, lj, si, sz])
    gv = np.concatenate([-yij.real, -yij.real, ys.real, yjj.real, np.zeros(len(si)), yz.real])
    bv = np.concatenate([-yij.imag, -yij.imag, ys.imag + bhl, yjj.imag + bhl, bsh, yz.imag])
    return {"rows": rows, "cols": cols, "g": gv, "b": bv, "li": li, "lj": lj,
            "incomplete": incomplete, "n_xfmr_used": int((series & is_xfmr).sum()),
            "n_shunt_used": int(shunt.sum()), "n_shunt_rz": int(shunt_rz.sum())}


def has_in_service_xfmr(raw: GridSample) -> bool:
    """样本是否含在运变压器(xfmr_2w / xfmr_3w / phase_shifter 且 mark==1)。只读静态给定量。"""
    if raw.n_branches == 0:
        return False
    boff = BRANCH_TABLE.offset_of
    mark = raw.branch_feat[:, boff("mark")]
    onehot = raw.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
    bt = np.asarray(BRANCH_TYPES)[np.argmax(onehot, axis=1)]
    return bool(np.any((mark == 1) & np.isin(bt, list(_XFMR))))


def pf_phys_arrays(raw: GridSample, tap: Optional[np.ndarray] = None,
                   shift: Optional[np.ndarray] = None, shunt_zero_harmless: bool = False,
                   ybus_t: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """从**原始(未归一化)**样本预计算失配所需常量(损失侧,不进编码器)。

    tap / shift:家族 sidecar 的变比与移相角(行序 = branch_feat);None = 无 sidecar,
    变压器按旧的保守口径排除(见 `_ybus_terms`)。shunt_zero_harmless:E15 并补口径(见 `_ybus_terms`;默认关)。

    返回(重复项由 index_add 自动求和):
      index [2,nnz] int64:Ybus COO(行,列)——命名含 index,PyG Batch 自动加节点偏移;
      g / b [nnz] float32:COO 实部/虚部值;
      eqp_clean / eqq_clean / eqp_def / eqq_def [N] bool:四类方程的节点掩码。
    """
    noff = NODE_TABLE.offset_of
    n = raw.n_nodes
    T = _ybus_terms(raw, tap, shift, shunt_zero_harmless) if ybus_t is None else ybus_t
    ei, ej = (raw.edge_index if raw.n_branches else (np.zeros(0, np.int64),) * 2)
    li, lj, incomplete = T["li"], T["lj"], T["incomplete"]

    # 节点方程有效性:非 NULL、STATE 非 missing、Ybus 行完整、有在运支路、
    # 且全部串联邻居的 V/θ 可得(邻居非 NULL/missing——其 V/θ 由给定或预测提供)
    null = raw.node_feat[:, noff("is_null")] > 0.5
    smiss = raw.node_missing[:, group_bit_index(Group.STATE)]
    bad_neighbor = null | smiss
    invalid = np.zeros(n, dtype=bool)
    invalid[ei[incomplete].astype(np.int64)] = True
    invalid[ej[incomplete].astype(np.int64)] = True
    invalid[lj[bad_neighbor[li]]] = True
    invalid[li[bad_neighbor[lj]]] = True
    deg = np.bincount(np.concatenate([li, lj]), minlength=n)
    active = ~null & ~smiss & ~invalid & (deg > 0)

    tcol = noff("node_type")
    ntype = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    cand = ~null & ~smiss                      # "本该有方程"的节点(E12 覆盖率闸的分母;不看 Ybus 行完整性)
    return {
        "index": np.stack([T["rows"], T["cols"]]).astype(np.int64),
        "g": T["g"].astype(np.float32),
        "b": T["b"].astype(np.float32),
        "eqp_clean": active & (ntype != _SLACK),
        "eqq_clean": active & (ntype == _PQ),
        "eqp_def": active & (ntype == _SLACK),
        "eqq_def": active & (ntype != _PQ),
        "eqp_cand": cand & (ntype != _SLACK),
        "eqq_cand": cand & (ntype == _PQ),
    }


def pf_phys_state(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor,
                  denorm: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """(pg, qg, v, θ, pl, ql) 的物理量纲合成(与 `pf_phys_residual` 同一合成规则:类型化未知量位置取 pred,其余取给定真值)。
    E12 修正步用它取预测电压 V(FDPF 的右端 ΔP/V、ΔQ/V 与 V ← V + β·ΔV)。D15:被监督位置一律来自 pred。"""
    vals = []
    for k, f in enumerate(PHYS_FIELDS):
        c = NODE_TABLE.offset_of(f)
        col = target[:, c]
        if f in PF_FIELDS:
            col = torch.where(loss_mask[:, c], pred[:, k], col)
        vals.append(col * denorm[k, 1] + denorm[k, 0])
    return tuple(vals)


def qv_hint(raw: GridSample, tap: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None,
            iters: int = 4, dv_cap: float = 0.3, dc: Optional[np.ndarray] = None,
            shunt_zero_harmless: bool = False, ybus_t: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """电压物理提示 [N,3] = (v̂, valid, fallback):快速解耦潮流的 Q–V 半步(预注册 §14.2,用户 2026-08-22 批准)。

    与 `dc_theta_lossy` 同类的**计算型特征**:PQ 节点电压从 1.0 起步,θ 取带网损 DC,每轮用当前 (V, θ) 算
    PQ 行的无功失配 ΔQ,解 B″·ΔV = ΔQ/V(B″ = −Im Y 的 PQ 子块,稀疏 LU 一次分解复用),更新 V;
    (a) 任一轮 max|ΔV| > dv_cap 或非有限 → 停在上一轮并置 fallback=1(模型可见);(b) 无 PQ 行 → valid=0。
    零训练精度(结果档 §3.20):118 0.00035、2000 留出 0.00101、10k 留出 0.00091 pu。

    **无泄漏声明**:只读拓扑 / 静态 r,x,b_half / 投退 / sidecar 变比 / PQ 节点的 pg,pl,qg,ql(给定量)/
    PV 与平衡机的 v(给定量)/ 平衡机 θ(经 dc_theta_lossy)。**不读** PQ 的 v、任何非平衡机的 θ、PV 的 qg、
    平衡机的 pg/qg(这些是被监督量,在进入计算前显式清零);单测锁定(扰动被监督量不改变输出)。
    """
    import scipy.sparse as sp
    from ..train.masking import PF_UNKNOWN_FIELDS

    noff = NODE_TABLE.offset_of
    n = raw.n_nodes
    out = np.zeros((n, 3), dtype=np.float32)
    out[:, 0] = 1.0
    if raw.n_branches == 0:
        return out
    if tap is None and has_in_service_xfmr(raw):
        # E10b(预注册 §15.1,2026-08-22):没有 sidecar 变比、却有在运变压器的样本(三个老方式库),
        # 此前只在"变压器相邻行作废"后的残缺子图上解 Q–V,提示反而比不给还差(IEEE-39 v 0.018 → 0.029)。
        # 改为整样本提示作废:v̂=1、valid=0、fallback=1(模型可见"提示缺席",退回直接头)。有 sidecar 的家族不受影响。
        out[:, 2] = 1.0
        return out
    # E15:并补口径与物理损失同源(默认关 = 逐位不变);提速二期 B:调用方可传入已算好的同输入 _ybus_terms(纯去重)
    T = _ybus_terms(raw, tap, shift, shunt_zero_harmless) if ybus_t is None else ybus_t
    Y = sp.coo_matrix((T["g"] + 1j * T["b"], (T["rows"], T["cols"])), shape=(n, n)).tocsr()
    tcol = noff("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    null = raw.node_feat[:, noff("is_null")] > 0.5
    smiss = raw.node_missing[:, group_bit_index(Group.STATE)]
    ok = ~null & ~smiss
    # 方程可用的节点(与 pf_phys_arrays 同口径)
    ei, ej = raw.edge_index
    bad_neighbor = null | smiss
    invalid = np.zeros(n, dtype=bool)
    invalid[ei[T["incomplete"]].astype(np.int64)] = True
    invalid[ej[T["incomplete"]].astype(np.int64)] = True
    li, lj = T["li"], T["lj"]
    invalid[lj[bad_neighbor[li]]] = True
    invalid[li[bad_neighbor[lj]]] = True
    deg = np.bincount(np.concatenate([li, lj]), minlength=n)
    pq = (types == _PQ) & ok
    solve = pq & ~invalid & (deg > 0)
    # 给定量:D15 未知位置显式清零(pg@平衡机、qg@PV/平衡机、v@PQ、θ@非平衡机)
    feat = raw.node_feat.astype(np.float64).copy()
    for t_, fields in PF_UNKNOWN_FIELDS.items():
        rows = types == t_
        for f in fields:
            feat[rows, noff(f)] = 0.0
    qinj = feat[:, noff("qg")] - feat[:, noff("ql")]
    v = np.ones(n)
    v[~pq & ok] = feat[~pq & ok, noff("v")]                    # PV / 平衡机电压 = 给定量
    if dc is None:
        dc = dc_theta_lossy(raw)
    th = np.where(dc[:, 1] > 0.5, dc[:, 0], 0.0).astype(np.float64)
    idx = np.where(solve)[0]
    if idx.size == 0:
        return out
    try:
        # LU 因子缓存(预注册 §2 C 项;默认关 = 原地 splu 逐位同路):键 = Ybus 的 COO 内容 + 求解节点集
        lu = perf_cache.cached_splu(
            "qv_bpp", (T["rows"], T["cols"], T["b"], np.int64(n), idx),
            lambda: (-Y.imag)[idx][:, idx].tocsc() + 1e-8 * sp.identity(idx.size, format="csc"))
    except Exception:
        return out
    G, B = Y.real, Y.imag
    fallback = False
    for k in range(int(iters)):
        vre, vim = v * np.cos(th), v * np.sin(th)
        ire = G @ vre - B @ vim
        iim = G @ vim + B @ vre
        q_calc = vim * ire - vre * iim
        dv = lu.solve((qinj - q_calc)[idx] / np.maximum(v[idx], 0.5))
        step = float(np.abs(dv).max()) if dv.size else 0.0
        if not np.isfinite(step) or step > dv_cap:
            fallback = True
            if k == 0:
                return out                                  # 第一步就发散:v̂=1、valid=0
            break
        v[idx] = np.clip(v[idx] + dv, 0.5, 1.5)
    out[idx, 0] = v[idx]
    out[idx, 1] = 1.0
    out[idx, 2] = 1.0 if fallback else 0.0
    return out


def pf_phys_denorm(norm) -> np.ndarray:
    """PHYS_FIELDS 逐字段 (mean, std) 仿射反归一化常量 [6,2](rad 恒等 → (0,1))。

    残差必须在物理量纲(标幺/弧度)上算;仅支持仿射范式(pu_z/z)与恒等(rad/raw),
    schema 若把这些字段改成 log_z 等非仿射范式,此处显式报错而非静默算错。
    """
    codes = {f.name: f.norm for f in NODE_TABLE.fields}
    out = np.zeros((len(PHYS_FIELDS), 2), dtype=np.float32)
    for k, f in enumerate(PHYS_FIELDS):
        c = NODE_TABLE.offset_of(f)
        if codes[f] in ("pu_z", "z"):
            out[k] = (norm.node_mean[c], norm.node_std[c])
        elif codes[f] in ("rad", "raw", "bool"):
            out[k] = (0.0, 1.0)
        else:
            raise ValueError(f"字段 {f} 范式 {codes[f]} 非仿射,物理残差不支持")
    return out


def pf_phys_residual(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor,
                     phys_index: torch.Tensor, phys_g: torch.Tensor, phys_b: torch.Tensor,
                     denorm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """基尔霍夫失配 (ΔP, ΔQ)[N](标幺,fp32)。

    预测/给定合成:pg/qg/v/θ 在类型化未知量位置取 pred、其余取 target(给定真值);
    pl/ql 恒为给定。真值只以给定量身份进入(D15:被监督位置一律用预测,无泄漏)。
    调用方保证全程 fp32(AMP 下先 pred.float())。
    """
    n = target.shape[0]
    vals = []
    for k, f in enumerate(PHYS_FIELDS):
        c = NODE_TABLE.offset_of(f)
        col = target[:, c]
        if f in PF_FIELDS:
            col = torch.where(loss_mask[:, c], pred[:, k], col)
        vals.append(col * denorm[k, 1] + denorm[k, 0])
    pg, qg, v, th, pl, ql = vals

    vre, vim = v * torch.cos(th), v * torch.sin(th)
    row, col = phys_index[0], phys_index[1]
    ire = torch.zeros(n, dtype=pred.dtype, device=pred.device)
    iim = torch.zeros(n, dtype=pred.dtype, device=pred.device)
    ire.index_add_(0, row, phys_g * vre[col] - phys_b * vim[col])
    iim.index_add_(0, row, phys_g * vim[col] + phys_b * vre[col])
    dp = (pg - pl) - (vre * ire + vim * iim)
    dq = (qg - ql) - (vim * ire - vre * iim)
    return dp, dq


def pf_phys_row_scale(phys_index: torch.Tensor, phys_b: torch.Tensor, n: int,
                      floor: float = 1.0) -> torch.Tensor:
    """逐行归一化尺度 [N] = max(|B_ii|, floor)(E15b,预注册 §19.2 `--phys-rownorm`;结果档 §3.22(4) 的 E9b)。

    B_ii = 该节点 Ybus 对角虚部(COO 中 row==col 的项求和:各串联支路 Im(y)+b_half、并补项)。
    机理:ΔP_i ≈ −V_i² B_ii·δθ_i,|B_ii| 在 ACTIVSg10k 中位 863 pu、2000 约 80 pu——同样的相角误差在高导纳行产生几百倍的失配,
    物理项梯度被这些行主导、压过 θ 监督;除以 |B_ii| 后失配回到"角误差 × V²"量级,各行同阶。floor=1 只防 |B_ii|≈0 的行被放大,不改其它行。
    只在训练损失里用;评估打印的 |ΔPQ| 仍是原始标幺值(与历史可比)。
    """
    row, col = phys_index[0], phys_index[1]
    diag = row == col
    b_ii = torch.zeros(n, dtype=phys_b.dtype, device=phys_b.device)
    b_ii.index_add_(0, row[diag], phys_b[diag])
    return b_ii.abs().clamp_min(floor)


def pf_phys_loss(dp: torch.Tensor, dq: torch.Tensor, masks: Dict[str, torch.Tensor],
                 w_clean: float, w_def: float = 0.0, row_scale: Optional[torch.Tensor] = None,
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """L1 失配损失(标幺)与诊断量。clean/def 语义见模块 docstring。
    row_scale [N](E15b):给定时 ΔP、ΔQ 先逐行除以它(见 `pf_phys_row_scale`);None = 旧行为逐位不变。"""
    if row_scale is not None:
        dp, dq = dp / row_scale, dq / row_scale

    def _mean_abs(terms):
        tot = dp.new_zeros(())
        cnt = 0
        for x, m in terms:
            if m.any():
                tot = tot + x[m].abs().sum()
                cnt += int(m.sum())
        return tot / max(cnt, 1)

    clean = _mean_abs([(dp, masks["eqp_clean"]), (dq, masks["eqq_clean"])])
    deff = _mean_abs([(dp, masks["eqp_def"]), (dq, masks["eqq_def"])])
    loss = w_clean * clean + w_def * deff
    return loss, {"phys_clean": float(clean.detach()), "phys_def": float(deff.detach())}


def _slack_dirichlet(B, nodes, refs, p_inj, theta_ref, n):
    """把**全部**平衡机按给定角做 Dirichlet 边界,解自由节点:B_ff θ_f = P_f − B_fr θ_r。

    2026-08-21 第 2 轮修法(结果档 §3.11 第 2 项):原实现只取首个平衡机为参考、其余平衡机
    按卡值 pg 当普通注入,与 `pf_theta_ref` 的"全平衡机 Dirichlet"口径不一致。单平衡机时
    两者数学等价(利用 B 行和为零),现语料 10,538 个含平衡机分量**全部是单平衡机**,
    实测逐位差 ≤ 1e-12(见交接文档;差异只来自 1e-10 微岭破坏行和为零)。
    返回 (rest_idx, θ_rest)。
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    rest = np.setdiff1d(nodes, refs)
    if rest.size == 0:
        return rest, np.zeros(0)
    Brr = B[rest][:, rest].tocsc() + 1e-10 * sp.identity(rest.size, format="csc")
    Brf = B[rest][:, refs]
    rhs = p_inj[rest] - np.asarray(Brf @ theta_ref[refs]).ravel()
    return rest, spsolve(Brr, rhs)


def dc_theta_lossy(raw: GridSample, iters: int = 3, loss_cap: float = 0.5,
                   loss_frac_max: float = 0.3, final_check: bool = False) -> np.ndarray:
    """带网损的 DC-θ 特征 [N,3](θ 弧度 + valid 位 + fallback 位)——2026-08-21 5060 线,**须用户批准后方可进训练**。

    机理:D15 下平衡机 P 未知,普通 DC 只能按无损假设让平衡机吸收 −Σ(其余注入),而真解中平衡机还要
    额外供出全网网损,差额全部压在平衡机的几条支路上 → 一个局部跳变 + 全网常量偏移
    (ACTIVSg2000 实测 +0.59 rad;合成 val 0.58)。本函数做经典"带损 DC":
      θ⁰ = DC;每轮用当前流 P_k=(θ_i−θ_j)/x_k 估支路损耗 r_k·P_k²,半分到两端作附加负荷,再解一次。
    **发散保护**:任一轮 (a) 单支路损耗超过 loss_cap·|P_k|(线性化失效)、(b) 总损耗超过总负荷的 loss_frac_max、
    (c) 迭代不收缩(max|Δθ| 不降)——立即退回普通 DC 并置 fallback=1(模型可见该位)。
    无泄漏声明同 dc_theta:只读拓扑/静态 r,x/给定注入/平衡机给定角;不碰被监督量与支路潮流;单测锁定。

    `final_check`(2026-08-21 第 2 轮新增,结果档 §3.11 第 1 项):(a)(b) 两道闸原本只检查**更新前**的
    迭代点,**最后一轮解出的 θ 从未被复核过**就返回了。开启后对末轮结果补一次 (a)(b)。
    影响面实测(全语料逐分量):合成 train 8718 个分量新增 36 个回退(0.4%)、val 448 个新增 1 个(0.2%)、
    合成 transfer 0、真实尺度 large/ext/25k/70k **全部 0**。因为非零,它**改变数值语义** → 不作默认,
    由 `--dc-encoding lossyg` 显式开启(旧值 `lossy` 与全部归档结果逐位一致)。
    (同轮量到但**未采纳**的一项:闸门 (c) 只要求步长单调、不要求收敛。若再要求"末步 ≤ 5%·|θ|",
    合成 train/val/transfer 会新增 38%/30%/56% 的回退、真实尺度仍是 0 —— 说明带网损 DC 在合成
    700/1000 档本就不收敛(与 §7.1 探针一致)。这是**发现**不是修法,阈值也无先验依据,
    故另立杠杆预注册后再试,不塞进本次合入。)
    """
    import scipy.sparse as sp

    boff, noff = BRANCH_TABLE.offset_of, NODE_TABLE.offset_of
    n = raw.n_nodes
    out = np.zeros((n, 3), dtype=np.float32)
    x = raw.branch_feat[:, boff("x")].astype(np.float64)
    r = raw.branch_feat[:, boff("r")].astype(np.float64)
    mark = raw.branch_feat[:, boff("mark")]
    onehot = raw.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
    bt = np.asarray(BRANCH_TYPES)[np.argmax(onehot, axis=1)] if raw.n_branches else np.array([])
    use = (mark == 1) & (bt != "shunt_comp") & (np.abs(x) > 1e-9)
    if not use.any():
        return out
    ei, ej = raw.edge_index
    i, j = ei[use].astype(np.int64), ej[use].astype(np.int64)
    xu, ru = x[use], np.maximum(r[use], 0.0)          # 负电阻等值支路不产生"损耗",按 0 计
    b = 1.0 / xu
    B = sp.coo_matrix((np.r_[b, b, -b, -b], (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n)).tocsr()
    tcol = noff("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    null = raw.node_feat[:, noff("is_null")] > 0.5
    p_inj = (raw.node_feat[:, noff("pg")] - raw.node_feat[:, noff("pl")]).astype(np.float64)
    p_load = np.abs(raw.node_feat[:, noff("pl")].astype(np.float64))
    theta_slack = raw.node_feat[:, noff("theta")].astype(np.float64)
    from scipy.sparse.csgraph import connected_components
    _nc, comp = connected_components(B, directed=False)
    for c in np.unique(comp):
        nodes = np.where((comp == c) & ~null)[0]
        if nodes.size < 2:
            continue
        slacks = nodes[types[nodes] == _SLACK]
        if slacks.size == 0:
            continue
        # 全部平衡机做 Dirichlet 参考(修法 2);LU 只分解一次,迭代复用
        rest = np.setdiff1d(nodes, slacks)
        if rest.size == 0:
            out[slacks, 0] = theta_slack[slacks]
            out[nodes, 1] = 1.0
            continue
        # LU 因子缓存(预注册 §2 B 项;默认关 = 原地 splu 逐位同路):键 = B′ 的 COO 内容 + 自由节点集
        lu = perf_cache.cached_splu(
            "dcp_bp", (i, j, b, np.int64(n), rest),
            lambda: B[rest][:, rest].tocsc() + 1e-10 * sp.identity(rest.size, format="csc"))
        bnd = np.asarray(B[rest][:, slacks] @ theta_slack[slacks]).ravel()   # −B_fr θ_r 项
        th0 = np.zeros(n); th0[slacks] = theta_slack[slacks]
        th0[rest] = lu.solve(p_inj[rest] - bnd)
        th, prev_step, fallback = th0.copy(), np.inf, False
        in_c = comp[i] == c
        tot_load = max(p_load[nodes].sum(), 1e-9)

        def _violates(theta):
            """(a) 单支路损耗 > loss_cap·|P|(线性化失效);(b) 总损耗 > loss_frac_max·总负荷。"""
            P_ = (theta[i] - theta[j]) / xu
            loss_ = ru * P_ * P_
            return (bool(np.any(loss_[in_c] > loss_cap * np.abs(P_[in_c]) + 1e-12))
                    or bool(loss_[in_c].sum() > loss_frac_max * tot_load), P_, loss_)

        for _ in range(int(iters)):
            bad, P, loss = _violates(th)
            if bad:
                fallback = True
                break
            pin = p_inj.copy()
            np.add.at(pin, i[in_c], -0.5 * loss[in_c]); np.add.at(pin, j[in_c], -0.5 * loss[in_c])
            th_new = np.zeros(n); th_new[slacks] = theta_slack[slacks]
            th_new[rest] = lu.solve(pin[rest] - bnd)
            step = float(np.abs(th_new[rest] - th[rest]).max())
            if not np.isfinite(step) or step > prev_step:
                fallback = True
                break
            prev_step, th = step, th_new
        if not fallback and final_check and _violates(th)[0]:
            fallback = True          # 修法 1:末轮解出的 θ 也要过 (a)(b),此前它从未被复核
        if fallback:
            th = th0
        out[rest, 0] = th[rest]
        out[slacks, 0] = theta_slack[slacks]
        out[nodes, 1] = 1.0
        out[nodes, 2] = 1.0 if fallback else 0.0
    return out


def dc_theta_base_delta(raw: GridSample, th_b: np.ndarray, pinj_b: np.ndarray) -> np.ndarray:
    """"基态 + ΔDC + 一轮损耗增量"θ 提示 [N,3](θ_bΔ, valid, fallback≡0)——E18(预注册 §21;
    用户 2026-08-24 裁决"基态解可作为推理输入"后启用)。

    th_b / pinj_b = 本家族 `_base_case`(PSASP 已解)的节点相角与给定注入(经 build_sample 提取,
    机器级缓存 `runs/base_cache/<family>.npz`,见 `scripts/probes_5060/build_base_cache.py`)。
    θ_hint = θ_base + Δθ:按分量解 B'·Δθ = ΔP − 0.5·分摊(Δloss),ΔP = 方式注入 − 基态注入,
    Δloss = r·((P_b+ΔP_flow)² − P_b²)(一轮增量,半分两端);**平衡机 Dirichlet Δθ_s = θ_s^方式 − θ_s^基态**
    (给定角一致时为 0;不一致时自动换参考系,不做断言)。无平衡机分量 valid=0(与 dc_theta 同)。
    零训练裸值(结果档 §3.32 前置探针,2026-08-24):25k 0.0124 / 70k 0.1240(带网损 DC 0.0229 / 0.2128)。

    **无泄漏声明**:输入 = 拓扑 / 静态 r,x / 投退 / 本方式给定注入与平衡机给定角(dc_theta 同款)+
    **另一工况(基态)的解**——基态是独立于本方式的常备输入(用户协议裁决,2026-08-24;
    已核基态与全部考卷方式注入距离 ≥1%、无一相同),不含本方式任何被监督量;单测锁定。
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    boff, noff = BRANCH_TABLE.offset_of, NODE_TABLE.offset_of
    n = raw.n_nodes
    out = np.zeros((n, 3), dtype=np.float32)
    th_b = np.asarray(th_b, dtype=np.float64)
    pinj_b = np.asarray(pinj_b, dtype=np.float64)
    if th_b.shape[0] != n or pinj_b.shape[0] != n or raw.n_branches == 0:
        return out                                            # 基态与方式不匹配:valid 全 0,调用方退回
    x = raw.branch_feat[:, boff("x")].astype(np.float64)
    r = np.maximum(raw.branch_feat[:, boff("r")].astype(np.float64), 0.0)
    mark = raw.branch_feat[:, boff("mark")]
    onehot = raw.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
    bt = np.asarray(BRANCH_TYPES)[np.argmax(onehot, axis=1)]
    use = (mark == 1) & (bt != "shunt_comp") & (np.abs(x) > 1e-9)
    if not use.any():
        return out
    ei, ej = raw.edge_index
    i, j = ei[use].astype(np.int64), ej[use].astype(np.int64)
    xu = x[use]
    ru = r[use]
    b = 1.0 / xu
    B = sp.coo_matrix((np.r_[b, b, -b, -b], (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n)).tocsr()
    tcol = noff("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    null = raw.node_feat[:, noff("is_null")] > 0.5
    pinj_m = (raw.node_feat[:, noff("pg")] - raw.node_feat[:, noff("pl")]).astype(np.float64)
    theta_slack = raw.node_feat[:, noff("theta")].astype(np.float64)   # 仅平衡机行被读(给定角)
    dP = pinj_m - pinj_b
    _nc, comp = connected_components(B, directed=False)
    dth = np.zeros(n)
    done = np.zeros(n, dtype=bool)
    for c in np.unique(comp):
        nodes = np.where((comp == c) & ~null)[0]
        if nodes.size < 2:
            continue
        slacks = nodes[types[nodes] == _SLACK]
        if slacks.size == 0:
            continue
        rest = np.setdiff1d(nodes, slacks)
        d_sl = theta_slack[slacks] - th_b[slacks]             # 参考系差(通常 0)
        if rest.size == 0:
            dth[slacks] = d_sl
            done[nodes] = True
            continue
        try:
            # 提速二期 C(E22 定档后补上 §3.37 的遗留):矩阵与 dc_theta_lossy 的 B′ 同构同值(同 use 掩码、
            # 同 1/x、同 rest、同 1e-10 岭)→ 共用 tag "dcp_bp",70k 门内直接命中已有因子,零新增内存
            lu = perf_cache.cached_splu(
                "dcp_bp", (i, j, b, np.int64(n), rest),
                lambda: B[rest][:, rest].tocsc() + 1e-10 * sp.identity(rest.size, format="csc"))
        except Exception:
            continue
        bnd = np.asarray(B[rest][:, slacks] @ d_sl).ravel()
        d1 = np.zeros(n)
        d1[slacks] = d_sl
        d1[rest] = lu.solve(dP[rest] - bnd)
        # 一轮损耗增量:ΔP_flow 用 Δθ¹,基态流量用 θ_b
        in_c = comp[i] == c
        P0 = (th_b[i] - th_b[j]) / xu
        P1 = ((th_b[i] + d1[i]) - (th_b[j] + d1[j])) / xu
        dloss = ru * (P1 ** 2 - P0 ** 2)
        pin2 = dP.copy()
        np.add.at(pin2, i[in_c], -0.5 * dloss[in_c])
        np.add.at(pin2, j[in_c], -0.5 * dloss[in_c])
        d2 = np.zeros(n)
        d2[slacks] = d_sl
        d2[rest] = lu.solve(pin2[rest] - bnd)
        if not np.all(np.isfinite(d2[nodes])):
            continue
        dth[nodes] = d2[nodes]
        done[nodes] = True
    out[done, 0] = (th_b[done] + dth[done]).astype(np.float32)
    out[done, 1] = 1.0
    return out


def dc_theta_lossy_v(raw: GridSample, tap: Optional[np.ndarray] = None,
                     shift: Optional[np.ndarray] = None, shunt_zero_harmless: bool = False,
                     seed: Optional[np.ndarray] = None, d_lossy: Optional[np.ndarray] = None,
                     ybus_t: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """"AC 一阶修正 DC"θ [N,3](θ_acv, valid, fallback)——E16(预注册 §20,用户 2026-08-23 批准;
    配方于 2026-08-23 按离线探针修订,见 §20.1 修订记录)。

    用途(修订后):**只作边侧 Δθ 基线与 Q–V 提示的 θ**;侧通道的节点级 θ 仍用带网损 DC(`_pf_data` 双轨)。
    机理(结果档 §3.31(d)/(g)):DC 逐边 Δθ 漏了交流一阶关系里的 V_iV_j 因子、电阻项与变比。逐边解一阶支路方程
        P_send = g·V̂_i² − V̂_i·V̂_j'·(g·cosΔ + b·sinΔ),   V̂_j' = V̂_j / t(j 侧变比,与 Ybus 口径一致),
    其中 y = 1/(r+jx) = g + jb,P_send = 带网损 DC 的支路流量 + 半损耗(r·P²/2),Δ⁰ = DC 边差,定点迭代 2 次;
    |sin| 参数超界或非有限的边退回 DC 边差。再把逐边 Δθ 最小二乘积分(图拉普拉斯 + 平衡机给定角 Dirichlet,权重 1,
    与推理器 potential_reconstruct 同一数学)回节点。
    离线实测(探针 `probe_dcv_offline.py`):逐边 Δθ 与重算的 v 提示在**所有**家族改善;节点级积分在
    2000/10k/300 变差(全网系统性偏移,V/AC 修正不治)——故侧通道不用本输出。
    **Q–V 提示整体无效的样本(三个老库:无 sidecar + 在运变压器,E10b 规则)原样返回带网损 DC,逐位不变。**

    **无泄漏声明**:输入 = dc_theta_lossy 的输入(拓扑/静态 r,x/给定注入/平衡机给定角)+ qv_hint 的输入
    (给定量,含 sidecar 变比)+ PV/平衡机的给定 V。PQ 节点电压只用 Q–V 提示 V̂,提示无效处取 1.0,
    **不读 PQ 真值 V**;单测锁定(扰动全部被监督量输出不变)。
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    boff, noff = BRANCH_TABLE.offset_of, NODE_TABLE.offset_of
    n = raw.n_nodes
    # 提速二期 A:同输入的带网损 DC 调用方已算过时直接传入(纯函数,逐位同值;默认 None = 原路)
    d = dc_theta_lossy(raw) if d_lossy is None else d_lossy
    if seed is not None:
        # E18(§21):AC 逐边修正的种子换成"基态 + Δ"提示(流量估计与 Δ⁰ 都更准);
        # valid = 两者交集;seed 无效处沿用带网损 DC(θ 列同缩)。fallback 位沿用 lossy 的。
        sd = np.asarray(seed, dtype=np.float32)
        both = (d[:, 1] > 0.5) & (sd[:, 1] > 0.5)
        d = d.copy()
        d[both, 0] = sd[both, 0]
    valid = d[:, 1] > 0.5
    if raw.n_branches == 0 or not valid.any():
        return d.copy()
    vh = qv_hint(raw, tap=tap, shift=shift, dc=d, shunt_zero_harmless=shunt_zero_harmless, ybus_t=ybus_t)
    if not (vh[:, 1] > 0.5).any():
        return d.copy()                                       # 提示整体无效(老库):不修正,逐位退回带网损 DC
    tcol = noff("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    is_pq = types == _PQ
    # D15:PQ 的 V 是被监督量,只能用提示;提示无效的 PQ 取 1.0;PV / 平衡机用给定 V
    v_hat = np.where(is_pq, np.where(vh[:, 1] > 0.5, vh[:, 0].astype(np.float64), 1.0),
                     raw.node_feat[:, noff("v")].astype(np.float64))
    v_hat = np.where(np.isfinite(v_hat) & (v_hat > 0.5), v_hat, 1.0)

    x = raw.branch_feat[:, boff("x")].astype(np.float64)
    r = raw.branch_feat[:, boff("r")].astype(np.float64)
    mark = raw.branch_feat[:, boff("mark")]
    onehot = raw.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
    bt = np.asarray(BRANCH_TYPES)[np.argmax(onehot, axis=1)]
    ei, ej = raw.edge_index
    use = (mark == 1) & (bt != "shunt_comp") & (np.abs(x) > 1e-9)      # 与 dc_theta 同一支路口径
    t = np.ones(raw.n_branches)
    if tap is not None:
        tp = np.asarray(tap, dtype=np.float64)
        sel = np.isin(bt, list(_XFMR)) & np.isfinite(tp) & (tp > 1e-6)
        t[sel] = tp[sel]
    ok = use & valid[ei] & valid[ej]
    i, j = ei[ok].astype(np.int64), ej[ok].astype(np.int64)
    if i.size == 0:
        return d.copy()
    xo = x[ok]
    ro = np.maximum(r[ok], 0.0)                               # 负电阻等值支路按 0 计损(与 dc_theta_lossy 同)
    di, dj = d[i, 0].astype(np.float64), d[j, 0].astype(np.float64)
    Vi, Vj = v_hat[i], v_hat[j] / t[ok]
    z2 = r[ok] ** 2 + xo ** 2
    g = r[ok] / z2
    bb = -xo / z2                                             # y = g + jb,b = −x/|z|²
    p_avg = (di - dj) / xo                                    # 带网损 DC 的支路流量(损耗已半分到两端)
    p_send = p_avg + 0.5 * ro * p_avg ** 2
    delta = di - dj                                           # θ_i − θ_j,DC 起点
    good = np.ones(i.size, dtype=bool)
    for _ in range(2):
        arg = (p_send - g * Vi ** 2 + g * Vi * Vj * np.cos(delta)) / (-bb * Vi * Vj)
        good &= np.isfinite(arg) & (np.abs(arg) < 0.995)
        delta = np.where(good, np.arcsin(np.clip(arg, -0.995, 0.995)), di - dj)
    dth = -np.where(good, delta, di - dj)                     # 逐边 Δθ = θ_j − θ_i(解不动的边退回 DC 边差)

    out = d.copy()
    null = raw.node_feat[:, noff("is_null")] > 0.5
    theta_slack = raw.node_feat[:, noff("theta")].astype(np.float64)   # 仅平衡机行被读(给定角)
    one = np.ones(i.size)
    # 图拉普拉斯(权重 1,对角 +1 / 非对角 −1)——与推理器 potential_reconstruct 的法方程同一矩阵
    L = sp.coo_matrix((np.r_[one, one, -one, -one], (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n)).tocsr()
    b = np.zeros(n)
    np.add.at(b, j, dth)
    np.add.at(b, i, -dth)
    _nc, comp = connected_components(L, directed=False)
    for c in np.unique(comp[valid & ~null]):
        nodes = np.where((comp == c) & ~null)[0]
        if nodes.size < 2:
            continue
        slacks = nodes[types[nodes] == _SLACK]
        if slacks.size == 0:
            continue                                          # 无参考角分量:保持 θ_dcL
        rest = np.setdiff1d(nodes, slacks)
        if rest.size == 0:
            continue
        try:
            # LU 因子缓存(预注册 §2 B 项之 dcv 积分拉普拉斯;键 = 边表 + 自由节点集,b 是右端不进键)
            lu = perf_cache.cached_splu(
                "dcv_lap", (i, j, np.int64(n), rest),
                lambda: L[rest][:, rest].tocsc() + 1e-10 * sp.identity(rest.size, format="csc"))
            th = lu.solve(b[rest] - np.asarray(L[rest][:, slacks] @ theta_slack[slacks]).ravel())
        except Exception:
            out[nodes, 2] = 1.0                               # LS 解不出:该分量退回 θ_dcL 并亮 fallback
            continue
        if not np.all(np.isfinite(th)):
            out[nodes, 2] = 1.0
            continue
        out[rest, 0] = th.astype(np.float32)
        out[slacks, 0] = theta_slack[slacks].astype(np.float32)
    return out


def dc_theta(raw: GridSample) -> np.ndarray:
    """DC-θ 特征 [N,2](θ_dc 弧度 + valid 位)——θ 杠杆 d(2026-07-12 用户批准 B 方案)。

    直流潮流线性解 B'θ = P:B' 由在运支路 1/x 构建(变压器以 x1 计,变比≈1 近似——
    这是**输入特征**非真值,近似可接受;串补 x<0 如实计入),P = 给定注入(pg−pl,
    机组取卡值调度、负荷取聚合值),平衡机参考角 = 样本给定 θ_slack。
    **无泄漏声明**:只读拓扑/静态 x/给定注入/平衡机给定 θ——全部是任务输入既有信息,
    不碰任何解算产物(PQ/PV 的 V/θ、支路潮流);单测锁定(扰动被监督 θ 不改变本特征)。
    含平衡机的连通分量外(孤岛/NULL)θ_dc=0 且 valid=0。
    评估纪律:报告须并列"DC 裸值"基线行(模型对 DC 的增量透明),门禁仍挂 L2(8.9)。
    """
    import scipy.sparse as sp

    boff, noff = BRANCH_TABLE.offset_of, NODE_TABLE.offset_of
    n = raw.n_nodes
    out = np.zeros((n, 2), dtype=np.float32)
    x = raw.branch_feat[:, boff("x")].astype(np.float64)
    mark = raw.branch_feat[:, boff("mark")]
    onehot = raw.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
    bt = np.asarray(BRANCH_TYPES)[np.argmax(onehot, axis=1)] if raw.n_branches else np.array([])
    use = (mark == 1) & (bt != "shunt_comp") & (np.abs(x) > 1e-9)
    if not use.any():
        return out
    ei, ej = raw.edge_index
    i, j = ei[use].astype(np.int64), ej[use].astype(np.int64)
    b = 1.0 / x[use]
    B = sp.coo_matrix((np.r_[b, b, -b, -b], (np.r_[i, j, i, j], np.r_[i, j, j, i])),
                      shape=(n, n)).tocsr()

    tcol = noff("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    null = raw.node_feat[:, noff("is_null")] > 0.5
    p_inj = (raw.node_feat[:, noff("pg")] - raw.node_feat[:, noff("pl")]).astype(np.float64)
    theta_slack = raw.node_feat[:, noff("theta")].astype(np.float64)   # 仅平衡机行被读

    # 连通分量(只在使用的支路上);逐含平衡机分量求解
    from scipy.sparse.csgraph import connected_components
    _nc, comp = connected_components(B, directed=False)
    for c in np.unique(comp):
        nodes = np.where((comp == c) & ~null)[0]
        if nodes.size < 2:
            continue
        slacks = nodes[types[nodes] == _SLACK]
        if slacks.size == 0:
            continue                       # 无参考角孤岛:valid=0
        # 全部平衡机按给定角做 Dirichlet(修法 2;单平衡机时与"解完再平移"数学等价)
        rest, th = _slack_dirichlet(B, nodes, slacks, p_inj, theta_slack, n)
        out[rest, 0] = th
        out[slacks, 0] = theta_slack[slacks]
        out[nodes, 1] = 1.0
    return out
