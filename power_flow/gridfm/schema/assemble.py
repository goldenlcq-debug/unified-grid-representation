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
from .fields import assert_branch_mark, BRANCH_TABLE, BRANCH_TYPES, Group, NODE_TABLE, NODE_TABLE_V03
from .sample import GridSample, group_bit_index

# DATALIB #,GEN, 字段 → v0.3 新列名(哨兵 ≥SENTINEL 置 0:如 Tq0'=9999 表示不建模)
_GEN_LIB_TO_FIELD = {"xd": "xd", "xdp": "xdp_lib", "xdpp": "xdpp", "xq": "xq",
                     "xqp": "xqp", "xqpp": "xqpp", "x2": "x2", "td0p": "td0p",
                     "td0pp": "td0pp", "tq0p": "tq0p", "tq0pp": "tq0pp",
                     "tj": "tj_lib", "damp": "damp_d"}

# 节点类型 onehot 索引
PQ, PV, SLACK = 0, 1, 2

# 直流侧传输效率(E27/§30;解码规格 §3.5 逐点扫定:K=1.0→0.985 把 131 方式母线级
# Σ|err| 65.5→45.0 pu;34 方式相对误差 0.6%~2.2%)
DC_LOSS_K = 0.985


def dc_equiv_loads(case: PsaspCase) -> Tuple[Dict[int, float], Dict[int, float], Dict]:
    """LF.ML4/NL4 → 换流站交流侧等值注入(负荷正方向,pu@SB;E27/预注册 §30)。

    算法 = 解码规格 §3.4:可控站(VSC PQ非平衡 / LCC 定直流功率)抄输入卡整定值;
    不可控站按直流系统(输入卡内并查集重建)功率闭合反解 —— 损耗取 DC_LOSS_K,
    残差按额定容量加权分摊;NL4 两端直流整流侧 npole·Pd、逆变侧扣线损 I²R。
    只读输入卡(G3 无泄漏闸);Q 按预注册 §30.2 只用 VSC 给定无功(在运实测恒 0,
    LCC/两端直流输入卡无 Q 字段 → 置 0 并入档 = 解码规格 §3.6 选项(a),缺口如实记)。
    返回 (pl_pu, ql_pu, audit):pl_pu>0 = 从交流网抽取(整流),<0 = 倒送(逆变)。
    """
    ml4 = case.ml4()
    nl4 = case.nl4()
    nb = case.n_buses
    sb = case.sb_mva()
    P: Dict[int, float] = {}
    Q: Dict[int, float] = {}
    n_contrib = 0

    def _addP(bus: int, mw: float) -> None:
        # += 累加,禁覆盖赋值(R5:换流站交流母线重数实测 {1:116, 2:42, 4:1},覆盖会静默丢台)
        nonlocal n_contrib
        P[bus] = P.get(bus, 0.0) + mw
        n_contrib += 1

    def _ck_bus(b: int, ctx: str) -> None:
        if not (1 <= b <= nb):
            raise ValueError(f"直流卡母线号越界 {b}(1..{nb}):{ctx}(R11)")

    audit: Dict = {"sb_mva": sb, "n_valid_vsc": 0, "n_valid_lcc": 0, "n_valid_dcline": 0,
                   "n_valid_nl4": 0, "n_vsc_qset_nonzero": 0, "dc_bus_set": set(),
                   "n_dc_systems": 0,
                   "q_option": "a-只补P(VSC整定Q实测恒0,LCC/NL4输入卡无Q字段,缺口如实记)"}

    if ml4 is not None:
        for b in ml4.dc_buses:
            _ck_bus(b, "ML4 #2 直流母线")
        audit["dc_bus_set"] = set(ml4.dc_buses)

        # 直流系统分区(§3.3):有效直流线 + 有效换流器高低压端,输入卡内并查集
        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for ln in ml4.dc_lines:
            if ln.valid == 1:
                union(ln.i_dc_bus, ln.j_dc_bus)
                audit["n_valid_dcline"] += 1
        for v in ml4.vscs:
            if v.valid == 1:
                union(v.hv_dc_bus, v.lv_dc_bus)
        for l in ml4.lccs:
            if l.valid == 1:
                union(l.hv_dc_bus, l.lv_dc_bus)

        fixed: Dict[int, float] = {}
        free: Dict[int, list] = {}
        for v in ml4.vscs:
            if v.valid != 1:
                continue    # 有效位过滤(R1:不滤 = 凭空注入,y51_101 实测 572.4 pu)
            audit["n_valid_vsc"] += 1
            _ck_bus(v.ac_bus, f"VSC {v.name}")
            s = find(v.hv_dc_bus)
            if v.ctrl_mode == 1 and v.balance == 0:
                # 定站(整定值兑现精确,实测 max|err| 0 MW);B[1] 流入换流器为正 = 负荷正方向,不翻号
                _addP(v.ac_bus, v.p_set_mw)
                fixed[s] = fixed.get(s, 0.0) + v.p_set_mw
                if v.q_set_mvar != 0.0:
                    audit["n_vsc_qset_nonzero"] += 1
                Q[v.ac_bus] = Q.get(v.ac_bus, 0.0) + v.q_set_mvar
            else:
                # 平衡站/PV/自由站:交流侧有功是直流网未知量(照抄整定值 max 偏 1500 MW,R2)
                free.setdefault(s, []).append((v.ac_bus, v.sn_mva, v.p_set_mw))
        for l in ml4.lccs:
            if l.valid != 1:
                continue
            audit["n_valid_lcc"] += 1
            _ck_bus(l.ac_bus, f"LCC {l.name}")
            s = find(l.hv_dc_bus)
            p = l.p_dc_mw if l.conv_type == 0 else -l.p_dc_mw   # 整流抽取 / 逆变倒送
            if l.dc_ctrl == 2:
                _addP(l.ac_bus, p)
                fixed[s] = fixed.get(s, 0.0) + p
            else:
                free.setdefault(s, []).append((l.ac_bus, l.xfmr_mva, p))

        audit["n_dc_systems"] = len({find(x) for x in parent})

        # 逐直流系统功率闭合(§3.4):余量摊给自由站;无定站系统以 |整定| 最大的自由站兜底
        for s, lst in free.items():
            if s not in fixed:
                k = max(range(len(lst)), key=lambda i: abs(lst[i][2]))
                _addP(lst[k][0], lst[k][2])
                fixed[s] = lst[k][2]
                lst = lst[:k] + lst[k + 1:]
                if not lst:
                    continue
            total_out = -DC_LOSS_K * fixed[s]
            W = sum(w for _, w, _ in lst)
            for ac, w, _ in lst:
                _addP(ac, total_out * (w / W) if W > 0 else total_out / len(lst))

    for rec in nl4:
        if rec.valid != 1:
            continue
        audit["n_valid_nl4"] += 1
        _ck_bus(rec.i_ac_bus, f"NL4 {rec.name} 整流侧")
        _ck_bus(rec.j_ac_bus, f"NL4 {rec.name} 逆变侧")
        pd = rec.pd_mw
        if pd == 0.0:
            _addP(rec.i_ac_bus, 0.0)
            _addP(rec.j_ac_bus, 0.0)
            continue
        if pd < 0.0:
            import warnings
            warnings.warn(f"NL4 {rec.name} 给定功率为负({pd} MW),方向语义存疑,按公式照算")
        if rec.vd_kv == 0.0:
            raise ValueError(f"NL4 {rec.name} Pd≠0 但 Vd=0,线损不可算")
        loss = rec.npole * (pd / rec.vd_kv) ** 2 * rec.rl_ohm
        # sanity 上界 10%:规格 R3 原写 5%,但其自身实测 max 528 MW(天中直流 2×4000 MW)
        # = 6.6% 已超 5%,原界与自身数据矛盾 → 取 10%(长距离特高压线损物理上限量级)
        if not (0.0 <= loss <= 0.10 * rec.npole * abs(pd)):
            raise ValueError(f"NL4 {rec.name} 线损 {loss:.2f} MW 越界 [0, 10%×{rec.npole}×{pd}](R3)")
        _addP(rec.i_ac_bus, rec.npole * pd)              # 整流侧抽取
        _addP(rec.j_ac_bus, -(rec.npole * pd - loss))    # 逆变侧倒送,幅值必小于整流侧

    # R5 审计:每台在运换流器恰好贡献一次、NL4 两端各一次,不多不少
    want = audit["n_valid_vsc"] + audit["n_valid_lcc"] + 2 * audit["n_valid_nl4"]
    if n_contrib != want:
        raise ValueError(f"等值注入贡献次数 {n_contrib} != {want}(R5 重复/漏计)")
    # 换流站交流母线与直流母线两个集合不得重叠(实测 34 方式 5992 vs 19324 交集 0)
    bad = set(P) & audit["dc_bus_set"]
    if bad:
        raise ValueError(f"换流站交流母线落在直流母线声明集内:{sorted(bad)[:5]}(数据异常)")

    pl_pu = {b: mw / sb for b, mw in P.items() if mw != 0.0}
    ql_pu = {b: mv / sb for b, mv in Q.items() if mv != 0.0}
    audit["n_inj_buses"] = len(pl_pu)
    audit["sum_p_pu"] = float(sum(pl_pu.values()))
    audit["sum_abs_p_pu"] = float(sum(abs(x) for x in pl_pu.values()))
    audit["sum_abs_q_pu"] = float(sum(abs(x) for x in ql_pu.values()))
    return pl_pu, ql_pu, audit


def _gen_node_type(type_code: int) -> int:
    if type_code == 0:
        return SLACK
    if type_code in (-1, -2):
        return PV
    if type_code in (1, -3):
        return PQ
    return PV  # -4..-8 特殊控制型,v0 归 PV 并由调用方记录


def build_sample(case: PsaspCase, catalog: Catalog, sample_id: str = "", family_id: str = "",
                 schema: str = "v0", dc_equiv: bool = False) -> GridSample:
    """schema="v0"(默认,25 节点列,行为逐位与 v0.3 前一致)| "v0.3"(S1:+18 动态列,
    DATALIB 解引用同步机全量参数;v0 的 25 列填法**不变**——legacy tj/xdp 仍取 ST.S5 内联值,
    正确值只进新列 tj_lib/xdp_lib 等,保 v1 栈可复现)。

    dc_equiv(E27/§30,默认关 = 逐位旧行为):True 时读 LF.ML4/NL4,把换流站交流侧
    P/Q 折成等值注入并入 pl/ql(dc_equiv_loads),直流母线(ML4 #2 声明集)标 is_null;
    零新列、节点类型不变、编码器零改动。纯交流语料 ML4 为空模板 → 开关开也是空操作。"""
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

        # 静态组。输入卫生(E27/§30.1,依据 §3.50(e2)):测试数据 L1 卡 VMAX/VMIN 填 kV
        # 有名值(schema 范式 = 语义标幺),值 > 5 显非 pu → 按 kV/|vbase| 折 pu;
        # 语料该两列恒零不触发(G0/G1 逐位闸为证);vbase 取绝对值(直流负极母线 vbase<0)。
        bus = all_buses[row]
        vmax_, vmin_ = bus.vmax, bus.vmin
        vb_ = abs(e.vbase)
        if vmax_ > 5.0 and vb_ > 0:
            vmax_ = vmax_ / vb_
        if vmin_ > 5.0 and vb_ > 0:
            vmin_ = vmin_ / vb_
        node_feat[row, off("vmax")] = vmax_
        node_feat[row, off("vmin")] = vmin_
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
        # 状态(Mark 合法域在源头就拦:2/3 = 单端开断,本项目未建模,见 fields.assert_branch_mark)
        assert_branch_mark([v.mark], f"assemble:{getattr(e, 'idname', '?')}")
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

    # 直流等值注入(E27/§30;默认关 = 上面全部行为逐位不变)
    dc_meta = None
    if dc_equiv:
        dc_pl, dc_ql, dc_audit = dc_equiv_loads(case)
        # 等值机双计排除(预注册 §30.2):换流站交流母线带在运 L5 机组 → 疑似已建等值机,
        # 跳过该母线并如实记(实测 0/159 命中;L6 负荷是同址真负荷,必须叠加不跳,解码规格 §4.2-3)
        gens_on = {b for b, g_ in gens.items() if g_.mark == 1}
        l5_hits = sorted(set(dc_pl) & gens_on)
        l6_hits = sorted(b for b in dc_pl if b in loads_by_bus)
        skipped_mw = 0.0
        for bus_no, p in dc_pl.items():
            if bus_no in l5_hits:
                skipped_mw += p * dc_audit["sb_mva"]
                continue
            node_feat[bus_no - 1, off("pl")] += p
        for bus_no, q in dc_ql.items():
            if bus_no in l5_hits:
                continue
            node_feat[bus_no - 1, off("ql")] += q
        # 直流母线标 is_null:判据只准用 ML4 #2 声明集(R10 禁母线名/vbase 符号);
        # 带在运交流支路的"直流母线"= 数据异常,置 null 会切活网 → 硬拒绝
        touched_on = set()
        for v in views:
            if v.mark == 1:
                touched_on.add(abs(v.i_no))
                touched_on.add(abs(v.j_no))
        dcbus = dc_audit["dc_bus_set"]
        bad_dc = sorted(dcbus & touched_on)
        if bad_dc:
            raise ValueError(f"直流母线带在运交流支路:{bad_dc[:5]}(共 {len(bad_dc)}),拒绝 is_null 标记")
        tbit, sbit = group_bit_index(Group.TOPOLOGY), group_bit_index(Group.STATIC)
        for bus_no in dcbus:
            r_ = bus_no - 1
            node_feat[r_, off("is_null")] = 1.0
            node_missing[r_, tbit] = True
            node_missing[r_, sbit] = True
        dc_meta = {k: (sorted(v) if isinstance(v, set) else v) for k, v in dc_audit.items()
                   if k != "dc_bus_set"}
        dc_meta.update({"n_dc_bus_marked": len(dcbus), "l5_hits": l5_hits,
                        "n_l6_hits": len(l6_hits), "skipped_mw_l5": skipped_mw})

    gs = GridSample(
        node_feat=node_feat, branch_feat=branch_feat, edge_index=ei,
        node_missing=node_missing, node_masked=node_masked,
        branch_missing=branch_missing, branch_masked=branch_masked,
        sample_id=sample_id, family_id=family_id,
        meta={"n_gens": len(gens), "converged": case.converged(),
              **({"dc_equiv": dc_meta} if dc_meta is not None else {})},
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
