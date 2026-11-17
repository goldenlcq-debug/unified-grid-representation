"""装配器:PsaspCase → GridSample(方案 T1.1 / 3.3)。存原始值,归一化在 T1.6 施加。

另提供潮流任务输入掩码 powerflow_mask_inputs(D15/5.2):按节点类型将未知量置缺失,
供防真值泄漏单测。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..psasp_io.cards import XfmrView
from ..psasp_io.case import PsaspCase
from ..psasp_io.catalog import Catalog
from ..psasp_io.datalib import SENTINEL, gen_params
from .fields import BRANCH_TABLE, BRANCH_TYPES, Group, NODE_TABLE, NODE_TABLE_V03
from .sample import GridSample, group_bit_index

# DATALIB #,GEN, 字段 → v0.3 新列名(哨兵 ≥SENTINEL 置 0:如 Tq0'=9999 表示不建模)
_GEN_LIB_TO_FIELD = {"xd": "xd", "xdp": "xdp_lib", "xdpp": "xdpp", "xq": "xq",
                     "xqp": "xqp", "xqpp": "xqpp", "x2": "x2", "td0p": "td0p",
                     "td0pp": "td0pp", "tq0p": "tq0p", "tq0pp": "tq0pp",
                     "tj": "tj_lib", "damp": "damp_d"}

# 节点类型 onehot 索引
PQ, PV, SLACK = 0, 1, 2


def _gen_node_type(type_code: int) -> int:
    if type_code == 0:
        return SLACK
    if type_code in (-1, -2):
        return PV
    if type_code in (1, -3):
        return PQ
    return PV  # -4..-8 特殊控制型,v0 归 PV 并由调用方记录


def build_sample(case: PsaspCase, catalog: Catalog, sample_id: str = "", family_id: str = "",
                 schema: str = "v0") -> GridSample:
    """schema="v0"(默认,25 节点列,行为逐位与 v0.3 前一致)| "v0.3"(S1:+18 动态列,
    DATALIB 解引用同步机全量参数;v0 的 25 列填法**不变**——legacy tj/xdp 仍取 ST.S5 内联值,
    正确值只进新列 tj_lib/xdp_lib 等,保 v1 栈可复现)。"""
    if schema not in ("v0", "v0.3"):
        raise ValueError(f"未知 schema {schema}")
    table = NODE_TABLE if schema == "v0" else NODE_TABLE_V03
    n = catalog.n_nodes
    m = catalog.n_branches
    Fn = table.total_dim
    Fb = BRANCH_TABLE.total_dim

    node_feat = np.zeros((n, Fn), dtype=np.float32)
    branch_feat = np.zeros((m, Fb), dtype=np.float32)
    node_missing = np.zeros((n, 4), dtype=bool)
    node_masked = np.zeros((n, 4), dtype=bool)
    branch_missing = np.zeros((m, 4), dtype=bool)
    branch_masked = np.zeros((m, 4), dtype=bool)

    # 按母线号索引附属设备
    gens = {g.bs_no: g for g in case.gens()}
    dyns = {d.bs_no: d for d in case.gen_dyns()}
    loads_by_bus: Dict[int, list] = {}
    for ld in case.loads():
        if ld.mark != 1:
            continue    # 2026-07-12 修:退出运行的负荷不得计入母线聚合(曾致 KCL 偏 0.04 pu)
        loads_by_bus.setdefault(ld.bs_no, []).append(ld)
    # 状态组有效性判在**是否收敛**上,不是"结果文件是否存在"(2026-07-09 实测教训):
    # PSASP 发散时仍会往 LF.LP1 写溢出值(实测 v≈9997.8),照单全收会把 9997.8 伏当成
    # 重构目标,并把归一化器的 std_v 从 0.026 推到 200.9 —— 训练里电压这一列等于被抹平。
    # converged() 为 None(无 LF.CAL)时沿用旧行为:有结果即视为有效。
    converged = case.converged()
    results = {r.bus_no: r for r in case.bus_results()}
    has_results = len(results) > 0 and (converged is not False)
    # 机组 P/Q 解值(LF.LP5,按母线聚合;2026-07-12 用户裁决"现在直接修"):
    # 此前状态组 pg/qg 读 LF.L5 卡值 = 解前值,PV 机 qg 与平衡机 pg/qg 系统性非解
    # (val 实测 PV dQ 中位 0.51 pu;LP5 代入定义方程 4.6e-6,变更记录 07-11/12)。
    solved_pq: Dict[int, Tuple[float, float]] = {}
    if has_results:
        for gr in case.gen_results():
            if not (np.isfinite(gr.pg) and np.isfinite(gr.qg)):
                continue
            p0, q0 = solved_pq.get(gr.bus_no, (0.0, 0.0))
            solved_pq[gr.bus_no] = (p0 + gr.pg, q0 + gr.qg)

    off = table.offset_of
    dlib = case.datalib() if schema == "v0.3" else {}
    all_buses = case.buses()   # 循环外预取:循环内逐节点重建全表在万节点级是 O(N²)
                               # (2026-08-05 实测 12,842 节点装配 381s→秒级,profile 在案)
    for e in catalog.nodes:
        row = e.token_row - 1  # token_row 从 1 起
        bus_no = e.bus_no
        g = gens.get(bus_no)
        lds = loads_by_bus.get(bus_no, [])
        has_gen = g is not None
        has_load = len(lds) > 0

        # 拓扑组
        ntype = _gen_node_type(g.type_code) if has_gen else PQ
        node_feat[row, off("node_type") + ntype] = 1.0
        node_feat[row, off("is_null")] = 1.0 if e.is_null else 0.0
        node_feat[row, off("has_gen")] = 1.0 if has_gen else 0.0
        node_feat[row, off("has_load")] = 1.0 if has_load else 0.0
        node_feat[row, off("vbase")] = e.vbase
        if e.is_null:
            node_missing[row, group_bit_index(Group.TOPOLOGY)] = True
            node_missing[row, group_bit_index(Group.STATIC)] = True

        # 静态组
        bus = all_buses[row]
        node_feat[row, off("vmax")] = bus.vmax
        node_feat[row, off("vmin")] = bus.vmin
        if has_gen:
            node_feat[row, off("pmax")] = g.pmax
            node_feat[row, off("pmin")] = g.pmin
            node_feat[row, off("qmax")] = g.qmax
            node_feat[row, off("qmin")] = g.qmin

        # 动态组(v0 四列填法**永不变**:legacy tj/xdp = ST.S5 内联值,含 cepri36/ieee39 的假 0)
        d = dyns.get(bus_no)
        if d is not None and d.mark == 1:
            node_feat[row, off("tj")] = d.tj
            node_feat[row, off("xdp")] = d.xdp
            node_feat[row, off("sh")] = d.sh
            node_feat[row, off("ph")] = d.ph
            if schema == "v0.3":
                # v0.3 扩展列:DATALIB #,GEN, 解引用全量参数(ST.S5.Lg → Par_No)。
                # dyn_resolved=1 ⇔ 全 13 参数来自 DATALIB;解引用失败(如 lib118 的
                # #,GEN, 段为空)→ 退 ST.S5 内联子集(tj/xdp/xdpp/x2,118 已填、
                # cepri36/ieee39 为 0)+ dyn_resolved=0——修"假 0 当有效值"的语义坑。
                prm = gen_params(dlib, d.lg)
                if prm is not None:
                    for k, fname in _GEN_LIB_TO_FIELD.items():
                        v = prm[k]
                        node_feat[row, off(fname)] = 0.0 if v >= SENTINEL else v
                    node_feat[row, off("dyn_resolved")] = 1.0
                else:
                    node_feat[row, off("tj_lib")] = d.tj
                    node_feat[row, off("xdp_lib")] = d.xdp
                    node_feat[row, off("xdpp")] = d.xdpp
                    node_feat[row, off("x2")] = d.x2v
                node_feat[row, off("gen_model")] = float(d.tg)
                node_feat[row, off("has_avr")] = 1.0 if d.lvr > 0 else 0.0
                node_feat[row, off("has_gov")] = 1.0 if d.lgo > 0 else 0.0
                node_feat[row, off("has_pss")] = 1.0 if d.lpss > 0 else 0.0
        else:
            node_missing[row, group_bit_index(Group.DYNAMIC)] = True

        # 运行状态组(机组 P/Q:LP5 解值优先;无解值时退卡值,退运机组注入为 0)
        if has_gen:
            if bus_no in solved_pq:
                node_feat[row, off("pg")], node_feat[row, off("qg")] = solved_pq[bus_no]
            elif g.mark == 1:
                node_feat[row, off("pg")] = g.pg
                node_feat[row, off("qg")] = g.qg
            node_feat[row, off("v_set")] = g.v0
        if has_load:
            node_feat[row, off("pl")] = sum(x.pl for x in lds)
            node_feat[row, off("ql")] = sum(x.ql for x in lds)
        r = results.get(bus_no) if has_results else None
        # 防线二:即便标称收敛,单点溢出值也不得入库(v_pu 物理上不可能 >3;NaN/inf 同理)
        if r is not None and (not np.isfinite(r.v_pu) or not np.isfinite(r.theta_deg)
                              or abs(r.v_pu) > 3.0):
            r = None
        if r is not None:
            node_feat[row, off("v")] = r.v_pu
            node_feat[row, off("theta")] = np.deg2rad(r.theta_deg)
        else:
            node_missing[row, group_bit_index(Group.STATE)] = True

        # 元信息组
        node_feat[row, off("adjustable")] = 1.0 if has_gen else 0.0

    # 支路
    boff = BRANCH_TABLE.offset_of
    aclines = case.aclines()
    xfmrs = case.xfmrs()
    views = aclines + xfmrs  # 与 catalog.branches 同序
    flows = {(f.i_no, f.j_no, f.id_no): f for f in case.branch_flows()}
    has_flows = len(flows) > 0 and (converged is not False)   # 同上:发散潮流的支路流是垃圾
    ei = np.zeros((2, m), dtype=np.int64)

    for e in catalog.branches:
        row = e.token_row
        v = views[row]
        # 母线号取绝对值:LF.L3 三绕组记录的 I 侧用**负号**标记星型中点侧(CEPRI-36 实测
        # 2026-07-10:-12/-28/-30 即 L1 第 12/28/30 行的真实中点母线,LP1 有解),
        # 行号本身 = |bus_no|。两绕组/交流线均为正,abs 为无操作。
        ei[0, row] = abs(e.i_no) - 1
        ei[1, row] = abs(e.j_no) - 1
        # 拓扑:类型 onehot
        branch_feat[row, boff("branch_type") + BRANCH_TYPES.index(e.branch_type)] = 1.0
        # 静态:按视图真实类型读阻抗(变压器用 r1/x1,交流线/串补/并补用 r/x/b)
        if isinstance(v, XfmrView):
            branch_feat[row, boff("r")] = v.r1
            branch_feat[row, boff("x")] = v.x1
        else:
            branch_feat[row, boff("r")] = v.r
            branch_feat[row, boff("x")] = v.x
            branch_feat[row, boff("b_half")] = v.b_half
        # 状态
        branch_feat[row, boff("mark")] = v.mark
        f = flows.get((e.i_no, e.j_no, e.id_no)) if has_flows else None
        if f is not None and not all(np.isfinite(x) for x in (f.pi, f.qi, f.pj, f.qj)):
            f = None
        if f is not None:
            branch_feat[row, boff("pi")] = f.pi
            branch_feat[row, boff("qi")] = f.qi
            branch_feat[row, boff("pj")] = f.pj
            branch_feat[row, boff("qj")] = f.qj
        else:
            branch_missing[row, group_bit_index(Group.STATE)] = True

    # 防线:任何越界/负索引直接报错,不得入库(2026-07-10 教训:CEPRI-36 三绕组负号
    # 约定曾把 -13 写进 edge_index,批产 100 样本全是坏图,训练期才炸)
    if m > 0 and (ei.min() < 0 or ei.max() >= n):
        bad = ei[:, (ei[0] < 0) | (ei[1] < 0) | (ei[0] >= n) | (ei[1] >= n)]
        raise ValueError(f"edge_index 越界(n={n}):{bad[:, :5].tolist()} —— 检查算例支路卡母线号约定")

    gs = GridSample(
        node_feat=node_feat, branch_feat=branch_feat, edge_index=ei,
        node_missing=node_missing, node_masked=node_masked,
        branch_missing=branch_missing, branch_masked=branch_masked,
        sample_id=sample_id, family_id=family_id,
        meta={"n_gens": len(gens), "converged": case.converged()},
    )
    gs.random_features()   # 生成时抽取一次并持久化(方案 3.2-3)
    return gs


# 潮流未知量(按节点类型),对应 NODE_TABLE 状态字段名
_PF_UNKNOWNS = {
    PQ: ["v", "theta"],
    PV: ["qg", "theta"],
    SLACK: ["pg", "qg"],
}


def powerflow_mask_inputs(sample: GridSample) -> Tuple[np.ndarray, List[str]]:
    """潮流任务输入协议(D15/5.2):按节点类型将未知量在 node_feat 中置 0,并置 STATE masked 位。

    返回 (supervised_mask[N, K], field_names):supervised_mask[i,k]=True 表示节点 i 的第 k 个
    状态字段是该节点未知量(值已被清零、不得被模型读到)。用于损失计权与防泄漏单测。
    """
    field_names = ["pg", "qg", "pl", "ql", "v", "theta", "v_set"]
    off = NODE_TABLE.offset_of
    tcol = off("node_type")
    n = sample.n_nodes
    supervised = np.zeros((n, len(field_names)), dtype=bool)
    state_bit = group_bit_index(Group.STATE)

    for i in range(n):
        onehot = sample.node_feat[i, tcol:tcol + 3]
        ntype = int(np.argmax(onehot))
        for fname in _PF_UNKNOWNS[ntype]:
            col = off(fname)
            sample.node_feat[i, col] = 0.0                # 置缺失(清零)
            supervised[i, field_names.index(fname)] = True
        sample.node_masked[i, state_bit] = True

    return supervised, field_names
