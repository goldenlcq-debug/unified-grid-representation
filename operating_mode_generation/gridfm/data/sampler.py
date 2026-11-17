"""方式采样器(方案 T3.1 / 7.2)。在骨架上采样运行方式变量,经真实求解器过滤收敛样本。

v0 方式变量:负荷水平(整体缩放 0.8~1.15 + 分区扰动)、发电有功再分配(按同因子缩放,
平衡机吸收失配)、机端电压设定微扰。新能源等值(D18)为配置侧改造清单,不改卡。
不收敛样本按比例保留作可解性负类(3.4.3 / 8.5)。
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..fsutil import robust_copytree
from ..psasp_io.case import PsaspCase
from ..psasp_io.catalog import Catalog
from ..schema.assemble import build_sample
from ..schema.sample import GridSample
from ..solver.base import SolverRunner


@dataclass
class SampleSpec:
    load_factor: float
    v_jitter: float
    seed: int


# 多样方式的离散负荷档(标签):同档多方式 = 真"一对多"(2026-07-14 用户裁决④)
DIVERSE_LF_LEVELS = (0.95, 1.05, 1.15)


def sample_mode_diverse(case: PsaspCase, rng: np.random.Generator,
                        amp: float = 1.0,
                        lf_levels: tuple = DIVERSE_LF_LEVELS,
                        decommit_frac: float = 0.3,
                        dispatch: str = "dirichlet",
                        only_online: bool = False) -> SampleSpec:
    """多模态方式采样(2026-07-14 用户裁决④:G1"平均化失败"演示的数据前提)。

    与 sample_mode 的本质区别:**同一标签(离散负荷档)对应多种真正不同的方式**——
    - 负荷档从 lf_levels 离散取(标签 = 档位值;逐负荷 ±3%·amp 抖动保多样不改档);
    - **开机组合**:非平衡机按 decommit_frac 概率"停机"(Pg=0,mark 不动 = 空载并网,
      不改拓扑语义;PV 母线电压支撑保留,v0 简化如实入档);
    - **出力分配(dispatch)**:目标总出力 = 档位 × 基态总出力,分配给开机机组后
      [Pmin,Pmax] 截断、残差按剩余裕度轮次消化(消化不完留给平衡机):
      - "dirichlet"(默认,lib118_diverse 口径,**同 seed 逐位不变**):Dirichlet(α=1)
        均匀单纯形随机参与度——真实 118 网收得率 ~64%,**合成网 270 节点实测 10.7% /
        400+ 节点 0%(2026-07-18 G4 首批证伪,勿用于合成家族)**;
      - "proportional"(diverse-lite,2026-07-18 用户批准 C 案):基态出力比例 ×
        逐机 U(1±0.15·amp) 抖动 → 归一到档位目标——sample_mode 同族的可行调度
        (118/2000 双端实证),同档多方式由开机组合 + 抖动承载;
    - 机端电压 ±0.03·amp(比基础采样器 ±0.02 略宽,画像多样性)。

    同档两次采样的逐机出力向量天差地别(单测锁定)——确定性回归对同档条件均值化,
    输出"哪个模式都不是"的平均调度,这才是 6.6 第一档要演示的对象。
    不动 sample_mode(同 seed 复现逐位不变)。
    """
    lf = float(lf_levels[int(rng.integers(0, len(lf_levels)))])
    for ld in case.loads():
        j = float(rng.uniform(1.0 - 0.03 * amp, 1.0 + 0.03 * amp))
        ld.pl = ld.pl * lf * j
        ld.ql = ld.ql * lf * j

    # only_online(F1,第十二任审查 2026-08-08):原实现把 mark=0 停运机也计入
    # base_total 与调度池——万节点 1,932 非平衡机仅 519 在线,base_total 虚高 33%
    # (1729.3 vs 1297.7 pu),目标出力失真且给停运机派出力。默认 False = 历史行为
    # 逐位不变(lib118_diverse 同 seed 复现红线);万节点/新生产一律显式 True。
    gens = [g for g in case.gens()
            if g.type_code != 0 and (not only_online or g.mark == 1)]
    base_total = sum(g.pg for g in gens)
    target = base_total * lf
    on = [g for g in gens if rng.random() >= decommit_frac]
    if not on:                                   # 全停保护:至少留一台
        on = [gens[int(rng.integers(0, len(gens)))]]
    for g in gens:
        if g not in on:
            g.pg = 0.0
    if dispatch == "dirichlet":
        w = rng.dirichlet(np.ones(len(on)))
    elif dispatch == "proportional":
        base = np.array([g.pg for g in on], dtype=float)   # ON 机组此刻仍持基态 Pg
        jit = rng.uniform(1.0 - 0.15 * amp, 1.0 + 0.15 * amp, size=len(on))
        w = base * jit
        w = w / w.sum() if w.sum() > 1e-9 else np.full(len(on), 1.0 / len(on))
    else:
        raise ValueError(f"未知 dispatch={dispatch}(可选 dirichlet/proportional)")
    alloc = w * target
    for g, p in zip(on, alloc):
        g.pg = float(p)
    # [Pmin,Pmax] 截断 + 残差按剩余裕度轮次消化(≤4 轮,消化不完平衡机吸收)
    for _ in range(4):
        resid = target - sum(min(max(g.pg, g.pmin), g.pmax if g.pmax > 0 else g.pg)
                             for g in on)
        for g in on:
            g.pg = min(max(g.pg, g.pmin), g.pmax if g.pmax > 0 else g.pg)
        if abs(resid) < 1e-6:
            break
        room = [max((g.pmax if g.pmax > 0 else g.pg + abs(resid)) - g.pg, 0.0)
                if resid > 0 else max(g.pg - g.pmin, 0.0) for g in on]
        tot = sum(room)
        if tot <= 1e-9:
            break
        for g, r in zip(on, room):
            g.pg = g.pg + resid * (r / tot)

    v_jit = 0.0
    for g in gens:
        dv = float(rng.uniform(-0.03 * amp, 0.03 * amp))
        g.v0 = g.v0 + dv
        v_jit = max(v_jit, abs(dv))
    return SampleSpec(load_factor=lf, v_jitter=v_jit, seed=int(rng.integers(0, 2**31)))


def sample_mode(case: PsaspCase, rng: np.random.Generator, amp: float = 1.0,
                v_band=None, vset_guard: float = 0.004, v_couple=None) -> SampleSpec:
    """在 case(已加载)上原地施加一组方式变量。返回本次采样规格(供标签/复现)。

    amp = 采样幅度系数(2026-07-11 用户批复:大网按规模收紧,防收敛率崩塌吃掉批产预算;
    实测方式收敛率 86%@150 → 59%@700)。amp=1.0 时各区间与旧实现完全相同(逐位兼容);
    amp<1 等比收窄:负荷整体缩放 U(1−0.2a, 1+0.15a)、逐负荷扰动 ±5%a、机端电压 ±0.02a。

    v_band(外协包 E,2026-08-25):可选 (vmin_arr, vmax_arr) 逐母线电压带。给出时,
    机端电压 v0 抖动后钳入自身母线带 [vmin+vset_guard, vmax−vset_guard]——纯随机抖动
    会把贴上限的机组母线(如 3120 网 1.05 带的机组)直接抖出带外,属自伤型越限。
    v_couple:可选 {机组母线: 簇id}(见 _vfix_build_ctx)。同簇机组共用首个成员的抖动值
    ——孪生母线对(|x|≤0.0008 pu 同厂母联)独立抖动会激起 Δv/x 级无功环流(实测
    ±60~140 pu 的非物理 Qg)。随机数消耗条数不变(每机仍抽一次,同簇仅覆写取值)。
    v_band=None(缺省)与旧行为逐位一致(RNG 消耗序列不变,钳位不消耗随机数)。
    """
    lf = float(rng.uniform(1.0 - 0.20 * amp, 1.0 + 0.15 * amp))
    # 负荷:整体缩放 + 每负荷扰动
    for ld in case.loads():
        j = float(rng.uniform(1.0 - 0.05 * amp, 1.0 + 0.05 * amp))
        ld.pl = ld.pl * lf * j
        ld.ql = ld.ql * lf * j
    # 发电:非平衡机 Pg 按同因子缩放,截断于 [Pmin, Pmax];机端电压微扰
    v_jit = 0.0
    cluster_dv: dict = {}
    for g in case.gens():
        if g.type_code == 0:      # 平衡机不设 Pg(由潮流决定)
            continue
        newp = g.pg * lf
        if g.pmax > 0:
            newp = min(newp, g.pmax)
        newp = max(newp, g.pmin)
        g.pg = newp
        dv = float(rng.uniform(-0.02 * amp, 0.02 * amp))
        if v_couple is not None:
            cid = v_couple.get(g.bs_no)
            if cid is not None:
                dv = cluster_dv.setdefault(cid, dv)
        g.v0 = _clamp_v0(g.v0 + dv, g.bs_no, v_band, vset_guard)
        v_jit = max(v_jit, abs(dv))
    return SampleSpec(load_factor=lf, v_jitter=v_jit, seed=int(rng.integers(0, 2**31)))


# ---------------- 电压带闭环校正(外协包 E · 2026-08-25) ----------------
# 病灶(3120 底图实测):底图自带 3 个低压越限母线(477/478/2530,最深差带 1.33%),
# 另有 13 个贴线母线(1.05 带机组区 + 488)。旧采样器 v0 纯随机抖动与负荷方向无关,
# 扰动方式在底图固有越限上再添 1~6 个 → 全部方式带越限、中位 4 个。
# 对策(全确定性、不依赖种子):
#   ① 采样时 v0 自钳(见 sample_mode v_band 参数)+ 孪生机组母线抖动同步:
#      case3120sp 存在大量 |x|≤0.0008 pu 的机组母线对(同厂母联),逐机独立抖动会在
#      小阻抗联络上激起 Q=Δv/x 级别的无功环流(实测 ±60~140 pu),抖动按簇同值消除;
#   ② 变压器末端母线走分接头:度=1 且唯一馈路为变压器 j 侧的越限母线(477/478/488/
#      2530 全属此类),Δtk≈err 一比一直达(实测灵敏度 ~1.05,零旁路副作用),
#      机端电压隔 3~4 跳对这类母线杠杆不足(实测 2 轮仅 +0.4 个千分点);
#   ③ 底图静态先验:库启动时从底图 LF.LP1 一次算出 v0/tk 修正,每方式预置;
#   ④ 解算后闭环:读 LP1 找越限/贴线母线,tap 末端母线走 Δtk,其余误差按电气跳数
#      加权分摊到邻近机组簇 v0,重解,至多 vfix_iters 轮。簇级 Q 裕度门(簇内求和,
#      环流分量自动相消)防无效调整;机组自身母线带钳位防自伤。

_VFIX_RHO = 0.55          # 跳数衰减:hop h 的机组权重 = rho^h / rho^min_hop
_VFIX_MAX_HOPS = 5        # 只动 5 跳内机组(问题母线实测 1~5 跳内均有带裕度机组)
_VFIX_GAIN = 1.5          # 过冲增益:母线对邻近机组 v0 的响应 <1,配合迭代收敛
_VFIX_STEP_CAP = 0.03     # 单轮单簇 v0 修正上限(pu)
_VFIX_ABS_LO, _VFIX_ABS_HI = 0.90, 1.15   # v0 绝对保底(防收敛性破坏)
_VFIX_X_TWIN = 0.002      # 孪生耦合阈:|x|≤此值的机组母线对同簇(3120 实测断档 0.00083→0.00317)
_VFIX_TK_STEP = 0.025     # 单轮单台变压器 tk 修正上限
_VFIX_TK_DRIFT = 0.05     # tk 距底图值的累计漂移上限(≈实际 OLTC ±8×1.25% 档距)


def _clamp_v0(v0: float, bus_no: int, v_band, guard: float) -> float:
    """把机组 v0 钳入自身母线电压带(带内留 guard 裕度)+ 绝对保底。

    v_band=None 时纯透传——保证旧调用(不传 v_band)同 seed 复现逐位不变。
    """
    if v_band is None:
        return v0
    vmin_a, vmax_a = v_band
    if 1 <= bus_no <= len(vmin_a):
        lo, hi = float(vmin_a[bus_no - 1]), float(vmax_a[bus_no - 1])
        if hi > lo > 0:
            v0 = min(max(v0, lo + guard), hi - guard)
    return min(max(v0, _VFIX_ABS_LO), _VFIX_ABS_HI)


def _vband_adjacency(case: PsaspCase) -> dict:
    """在线串联支路(交流线+变压器)的母线邻接表。并补(自环/无后缀)不入图。"""
    adj: dict = {}

    def _add(i: int, j: int) -> None:
        if i <= 0 or j <= 0 or i == j:
            return
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    for ln in case.aclines():
        if ln.mark == 1 and ln.branch_type != "shunt_comp":
            _add(ln.i_no, ln.j_no)
    for xf in case.xfmrs():
        if xf.mark == 1:
            _add(xf.i_no, xf.j_no)
    return adj


def _v_from_lp1(records, n_bus: int) -> np.ndarray:
    """LP1 记录 → 电压数组(与 eval_quality 同口径:有限值、<100、按 bus_no-1 落位)。"""
    v = np.zeros(n_bus)
    for r in records or []:
        if 1 <= r.bus_no <= n_bus and np.isfinite(r.v_pu) and r.v_pu < 100:
            v[r.bus_no - 1] = r.v_pu
    return v


def _vfix_build_ctx(base_case: PsaspCase, v_band, with_prior: bool = True) -> dict:
    """库级一次性上下文(拓扑/底图不随方式变):

    adj        在线串联支路邻接表;
    cluster    {机组母线: 簇id}(|x|≤_VFIX_X_TWIN 的机组母线对并查集,孪生同厂母线);
    members    {簇id: [机组母线...]};
    tap_rows   {母线: [LF.L3 行号...]}——度=1 且唯一馈路为变压器 j 侧的"变压器末端"
               母线(477/478/488/2530 类),分接头一比一直达,不走机端电压;
    base_tk    {L3 行号: 底图 tk}(累计漂移上限的锚);
    prior_v0   {L5 行号: Δv0}、prior_tk {L3 行号: Δtk}:底图静态先验。
    """
    adj = _vband_adjacency(base_case)
    gen_buses = {g.bs_no for g in base_case.gens() if g.mark == 1}
    parent = {b: b for b in gen_buses}

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ln in base_case.aclines():
        if (ln.mark == 1 and ln.branch_type == "acline" and abs(ln.x) <= _VFIX_X_TWIN
                and ln.i_no in gen_buses and ln.j_no in gen_buses and ln.i_no != ln.j_no):
            parent[_find(ln.i_no)] = _find(ln.j_no)
    for xf in base_case.xfmrs():
        if (xf.mark == 1 and abs(xf.x1) <= _VFIX_X_TWIN
                and xf.i_no in gen_buses and xf.j_no in gen_buses):
            parent[_find(xf.i_no)] = _find(xf.j_no)
    cluster = {b: _find(b) for b in gen_buses}
    members: dict = {}
    for b, c in cluster.items():
        members.setdefault(c, []).append(b)

    tap_rows: dict = {}
    base_tk: dict = {}
    deg: dict = {}
    for b, nbrs in adj.items():
        deg[b] = len(set(nbrs))
    for xf in base_case.xfmrs():
        if xf.mark == 1:
            base_tk[xf.row] = xf.tk
            jb = xf.j_no
            if deg.get(jb, 0) == 1 and jb not in gen_buses:
                tap_rows.setdefault(jb, []).append(xf.row)
    # 末端母线若还有交流线并联(度>1 已排除);同母线多台并联变压器全收

    ctx = {"adj": adj, "cluster": cluster, "members": members,
           "tap_rows": tap_rows, "base_tk": base_tk,
           "prior_v0": {}, "prior_tk": {}}
    if with_prior:
        try:
            v = _v_from_lp1(base_case.bus_results(), len(v_band[0]))
        except Exception:
            return ctx
        if (v > 0.01).sum() == 0:
            return ctx
        q_room = _cluster_q_room(base_case, base_case.gen_results(), cluster)
        # 先验档:探测收严到带界+6/4 个千分位(底图贴线母线全纳入),目标取中档水位
        new_v0, new_tk = _vfix_correction(base_case, v, v_band, ctx,
                                          detect_lo=0.006, target_lo=0.008,
                                          detect_hi=0.004, target_hi=0.004,
                                          q_room=q_room, gain=1.2)
        base_v0 = {g.row: g.v0 for g in base_case.gens()}
        ctx["prior_v0"] = {row: min(max(nv - base_v0[row], -0.025), 0.025)
                           for row, nv in new_v0.items()}
        ctx["prior_tk"] = {row: nt - base_tk[row] for row, nt in new_tk.items()}
    return ctx


def _cluster_q_room(case: PsaspCase, gen_results, cluster: dict) -> dict:
    """簇级无功裕度 {簇id: (可升压, 可降压)}。按簇求和——孪生对间的无功环流分量
    在求和中相消,门判不再被环流噪声干扰。限值缺失/哨兵值(≥9000、qmax≤qmin、
    全 0)按放行处理;LP5 缺失同放行。"""
    qmax_c: dict = {}
    qmin_c: dict = {}
    nolim: dict = {}
    for g in case.gens():
        if g.mark != 1 or g.bs_no not in cluster:
            continue
        c = cluster[g.bs_no]
        qmx, qmn = g.qmax, g.qmin
        if (not np.isfinite(qmx)) or (not np.isfinite(qmn)) or qmx >= 9000 \
                or qmx <= qmn or (qmx == 0.0 and qmn == 0.0):
            nolim[c] = True
            continue
        qmax_c[c] = qmax_c.get(c, 0.0) + qmx
        qmin_c[c] = qmin_c.get(c, 0.0) + qmn
    qg_c: dict = {}
    for r in gen_results or []:
        if r.bus_no in cluster and np.isfinite(r.qg):
            c = cluster[r.bus_no]
            qg_c[c] = qg_c.get(c, 0.0) + r.qg
    room: dict = {}
    for c in set(qmax_c) | set(nolim):
        if nolim.get(c) or c not in qg_c:
            room[c] = (True, True)
        else:
            room[c] = (qg_c[c] < qmax_c[c] - 0.10, qg_c[c] > qmin_c[c] + 0.10)
    return room


def _vfix_correction(case: PsaspCase, v: np.ndarray, v_band, ctx: dict,
                     detect_lo: float = 0.003, target_lo: float = 0.008,
                     detect_hi: float = 0.002, target_hi: float = 0.004,
                     q_room: Optional[dict] = None,
                     gain: float = _VFIX_GAIN):
    """由当前解算越限/贴线母线的修正计划。返回 (new_v0 {L5行: v0}, new_tk {L3行: tk})。

    探测阈(detect)与目标水位(target)分离:只有贴到 带界+detect 以内的母线才触发修正,
    但修正把它送到 带界+target——target 逐方式随机(generate_library 抽取),让修正后的
    电压下/上包络在方式间保持展开,不塌缩到同一条线上(多样性护栏)。
    变压器末端母线(ctx.tap_rows)走 Δtk = err 一比一(实测灵敏度 ~1.05、无旁路副作用),
    其余母线 BFS 找 _VFIX_MAX_HOPS 跳内机组,gain·err 按 rho^hop 归一派发,**按簇聚合**
    (同向取最大、异向相抵,簇内所有机组同步动,防孪生失配激起无功环流)。
    簇 Δv0 受簇内最紧母线带钳位;两计划全空 = 无需修正。
    """
    vmin_a, vmax_a = v_band
    n_bus = len(vmin_a)
    targets = []
    for i in range(n_bus):
        if v[i] <= 0.01:
            continue
        lo, hi = float(vmin_a[i]), float(vmax_a[i])
        if not (hi > lo > 0):
            continue
        if v[i] < lo + detect_lo:
            targets.append((i + 1, (lo + target_lo) - v[i]))
        elif v[i] > hi - detect_hi:
            targets.append((i + 1, (hi - target_hi) - v[i]))
    if not targets:
        return {}, {}

    gen_rows_by_bus: dict = {}
    for g in case.gens():
        if g.mark == 1:
            gen_rows_by_bus.setdefault(g.bs_no, []).append(g)
    xfmrs = case.xfmrs()

    from collections import deque
    new_tk: dict = {}
    up_d: dict = {}
    dn_d: dict = {}
    adj = ctx["adj"]
    cluster = ctx["cluster"]
    for src, err in targets:
        # 变压器末端:分接头一比一,不再派发机端电压需求
        if src in ctx["tap_rows"]:
            for xr in ctx["tap_rows"][src]:
                cur = xfmrs[xr].tk
                b0 = ctx["base_tk"].get(xr, cur)
                step = min(max(err, -_VFIX_TK_STEP), _VFIX_TK_STEP)
                nt = min(max(cur + step, b0 - _VFIX_TK_DRIFT), b0 + _VFIX_TK_DRIFT)
                if abs(nt - cur) >= 5e-4:
                    new_tk[xr] = nt
            continue
        seen = {src: 0}
        q = deque([src])
        found = []
        while q:
            u = q.popleft()
            h = seen[u]
            if h >= _VFIX_MAX_HOPS:
                continue
            for w in adj.get(u, ()):
                if w not in seen:
                    seen[w] = h + 1
                    if w in gen_rows_by_bus:
                        found.append((w, h + 1))
                    q.append(w)
        if src in gen_rows_by_bus:
            found.insert(0, (src, 0))
        if not found:
            continue
        min_hop = min(h for _b, h in found)
        for gb, h in found:
            c = cluster.get(gb, gb)
            d = gain * err * (_VFIX_RHO ** (h - min_hop))
            if d > 0:
                up_d[c] = max(up_d.get(c, 0.0), d)
            else:
                dn_d[c] = min(dn_d.get(c, 0.0), d)

    new_v0: dict = {}
    for c in set(up_d) | set(dn_d):
        net = up_d.get(c, 0.0) + dn_d.get(c, 0.0)
        if abs(net) < 5e-4:
            continue
        if q_room is not None and c in q_room:
            can_up, can_dn = q_room[c]
            if (net > 0 and not can_up) or (net < 0 and not can_dn):
                continue
        net = min(max(net, -_VFIX_STEP_CAP), _VFIX_STEP_CAP)
        # 簇内最紧母线带决定可动余量(孪生带一致时等价于普通钳位)
        mem = ctx["members"].get(c, [c])
        lo_room = None
        hi_room = None
        for gb in mem:
            for g in gen_rows_by_bus.get(gb, ()):
                lo, hi = _band_of(gb, v_band)
                up_room = hi - 0.004 - g.v0
                dn_room = g.v0 - (lo + 0.004)
                hi_room = up_room if hi_room is None else min(hi_room, up_room)
                lo_room = dn_room if lo_room is None else min(lo_room, dn_room)
        if hi_room is None:
            continue
        dv = min(net, max(hi_room, 0.0)) if net > 0 else max(net, -max(lo_room, 0.0))
        if abs(dv) < 5e-4:
            continue
        for gb in mem:
            for g in gen_rows_by_bus.get(gb, ()):
                nv = min(max(g.v0 + dv, _VFIX_ABS_LO), _VFIX_ABS_HI)
                if abs(nv - g.v0) >= 1e-4:
                    new_v0[g.row] = nv
    return new_v0, new_tk


def _band_of(bus_no: int, v_band):
    """母线自身电压带(无带/带异常回退到绝对保底区间)。"""
    vmin_a, vmax_a = v_band
    if 1 <= bus_no <= len(vmin_a):
        lo, hi = float(vmin_a[bus_no - 1]), float(vmax_a[bus_no - 1])
        if hi > lo > 0:
            return lo, hi
    return _VFIX_ABS_LO, _VFIX_ABS_HI


def generate_library(base_case_dir: Path, solver: SolverRunner, n: int, out_dir: Path,
                     work_dir: Path, seed: int = 0, family_id: str = "ieee118",
                     keep_divergent_frac: float = 0.08,
                     max_attempts: Optional[int] = None,
                     v_floor: Optional[float] = 0.85,
                     amp: float = 1.0,
                     keep_cases: bool = False,
                     id_start: int = 0,
                     diverse: bool = False,
                     diverse_dispatch: str = "dirichlet",
                     diverse_decommit=0.3,
                     schema: str = "v0",
                     only_online: bool = False,
                     v_band=None,
                     vfix_iters: int = 2,
                     vfix_prior: bool = True) -> dict:
    """生成方式库,直到收集到 **n 个收敛样本**(或尝试次数超 max_attempts,默认 3n)。

    实测教训(2026-07-09):旧实现只跑固定 n 次迭代,发散即丢——合成 500 节点网的方式
    发散率约 20%,库大小随规模缩水,家族间样本数不齐。改为"目标收敛数"语义。

    收敛样本正常入库;按 keep_divergent_frac 比例额外保留不收敛样本作负类(状态量置
    missing、可解性=0),负类不计入 n。keep_divergent_frac=0 时纯收敛库(冻结测试集用)。

    **电压带验收(2026-07-11 用户批准,阈值 0.85)**:收敛但存在有效母线 v < v_floor 的
    样本(合成器病态弱电压口袋 / 失电孤岛 v=0)**不入主库、不计入 n**,封存到
    `out_dir/_lowv_quarantine/`(带独立 manifest,留档供困难样本集/热启动窄门,7.2)。
    v_floor=None 关闭验收(既有库复现用)。现有冻结集不受影响(本函数只管新生产)。

    **keep_cases(2026-07-11,T4.2 前置)**:入主库的每个样本把**已解算例目录**整目录
    留存到 `out_dir/cases/<sample_id>/`(含 LF.* 输入卡与 LP*/CAL 结果卡)——T4.2 批量
    判稳需要每方式原始算例文件(交接文档实现路径 2);不影响 RNG 消耗序列,同 seed 复现
    的 .npz 与不开 keep_cases 逐位一致。低压封存/丢弃样本不留存。

    **id_start(2026-07-14,外协包 C 号段隔离护栏)**:sample_id 序号 = id_start + 尝试序
    (`{family_id}_{id_start+i:05d}`)。外协扩产按任务书指派号段(如包 C = 20000 起),
    与主线 0 段物理不相交;默认 0 与旧命名逐位一致,**不消耗 RNG**,同 seed 复现不受影响。

    **diverse_decommit 自适应阶梯(2026-07-18 用户批准 S2;2026-07-19 审查修:窗口率)**:
    传 float = 固定停机率(历史行为);传 tuple/list(如 (0.15, 0.08, 0.0))= 逐家族自适应
    ——每档内每 40 次尝试查点,**当前档窗口内**收敛率 < 0.25 即降到下一档。窗口率而非
    全程累计率是修复:累计率把前档积累的失败带进新档,会让已好转的家族被误判继续降档
    (审查:205)。背景:合成拓扑"体质"方差大,任何固定值都有长尾;阶梯保证交付满额,
    代价(各家族停机多样性异质)**全记录**:manifest 每行记该样本 decommit,返回含
    decommit_final / decommit_steps。同 seed 确定可复现(降档由求解器确定性结果驱动)。
    返回 {n_converged, n_divergent_kept, n_lowv_quarantined, n_total, n_attempts,
    decommit_final, decommit_steps, out_dir}。

    **v_band 电压带闭环校正(外协包 E,2026-08-25)**:传 (vmin_arr, vmax_arr) 逐母线带
    (与验收计分同源,即 MATPOWER .m 的 VMIN/VMAX 列)即启用:采样时机组 v0 自钳 +
    孪生簇抖动同步、底图静态先验(vfix_prior,v0 与末端变压器 tk)、解算后闭环修正
    (至多 vfix_iters 轮重解,修正解算发散则回滚重解一次,保证入库状态始终为收敛解)。
    修正的探测阈固定(带界内 3/2 个千分位),目标水位逐方式从 RNG 抽取
    (target_lo~U(0.004,0.016)、target_hi~U(0.003,0.006))——修正后的电压包络在方式间
    保持展开,不塌缩多样性;每次尝试固定消耗 2 个额外随机数,同 seed 确定可复现。
    v_band=None(缺省)与旧行为逐位一致(不建上下文、不加随机数消耗)。
    返回 dict 增加 {n_fix_solves, n_fix_reverts, n_modes_vfixed}(v_band=None 时全 0)。
    """
    base_case_dir = Path(base_case_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir)
    rng = np.random.default_rng(seed)
    cap = max_attempts if max_attempts is not None else max(3 * n, n + 10)

    # 停机阶梯(S2):float = 固定;序列 = 自适应降档(查点每 40 尝试,收敛率 < 0.25)
    ladder = (list(diverse_decommit)
              if isinstance(diverse_decommit, (list, tuple)) else [float(diverse_decommit)])
    li = 0
    decommit_steps = []
    tier_att0 = tier_conv0 = 0        # 当前档起点的累计尝试/收敛(算窗口率用)
    _CHECK_EVERY, _MIN_KEEP = 40, 0.25

    # 电压带校正上下文(库级一次:邻接/孪生簇/末端变压器/底图先验;拓扑不随方式变)
    vctx = None
    if v_band is not None:
        vctx = _vfix_build_ctx(PsaspCase(base_case_dir), v_band, with_prior=vfix_prior)

    n_conv = 0
    n_div_kept = 0
    n_lowv = 0
    n_fix_solves = 0
    n_fix_reverts = 0
    n_modes_vfixed = 0
    attempts = 0
    manifest = []
    lowv_manifest = []
    lowv_dir = out_dir / "_lowv_quarantine"
    while n_conv < n and attempts < cap:
        # 自适应查点(先于本次采样;仅阶梯模式且未到底档;用**当前档窗口**收敛率)
        win_att = attempts - tier_att0
        if (len(ladder) > 1 and li < len(ladder) - 1 and win_att > 0
                and win_att % _CHECK_EVERY == 0
                and (n_conv - tier_conv0) / win_att < _MIN_KEEP):
            decommit_steps.append({"attempt": attempts,
                                   "from": ladder[li], "to": ladder[li + 1],
                                   "window_keep_rate": round((n_conv - tier_conv0) / win_att, 3)})
            li += 1
            tier_att0, tier_conv0 = attempts, n_conv    # 新档窗口清零
        i = attempts
        attempts += 1
        # 独立工作目录(单库内串行;并行只在家族级,7.0 纪律 7)
        # 家族级并行下 Windows 句柄异步释放会让 rmtree/copytree 偶发瞬态失败
        # (07-09 批产丢家族)→ 重试兜底,耗尽仍失败即抛(不静默跳过)
        robust_copytree(base_case_dir, work_dir)
        case = PsaspCase(work_dir)
        # diverse = 多模态方式(裁决④,离散负荷档 × 随机开机/参与度);默认走基础采样器。
        # 停机率剂量效应(2026-07-18 三臂标定@270 合成网):0→100%、0.15→60%、0.30→10%
        # 收得率;家族方差大 → S2 自适应阶梯(见 docstring)。默认 0.3 = lib118_diverse。
        if diverse:
            spec = sample_mode_diverse(case, rng, amp=amp, dispatch=diverse_dispatch,
                                       decommit_frac=ladder[li],
                                       only_online=only_online)
        else:
            spec = sample_mode(case, rng, amp=amp, v_band=v_band,
                               v_couple=(vctx["cluster"] if vctx else None))
        if vctx is not None:
            # 逐方式目标水位(探测阈固定、修正落点随机——校正后的电压包络在方式间保持
            # 展开,不塌缩;每尝试固定 2 draw,序列确定)
            t_lo = float(rng.uniform(0.004, 0.016))
            t_hi = float(rng.uniform(0.003, 0.006))
            if vctx["prior_v0"]:
                for g in case.gens():
                    dv = vctx["prior_v0"].get(g.row)
                    if dv:
                        g.v0 = _clamp_v0(g.v0 + dv, g.bs_no, v_band, 0.004)
            if vctx["prior_tk"]:
                xfs = case.xfmrs()
                for xr, dtk in vctx["prior_tk"].items():
                    xfs[xr].tk = xfs[xr].tk + dtk
        case.save_inplace()

        pf = solver.run_powerflow(work_dir)
        converged = pf.converged
        if vctx is not None and converged and vfix_iters > 0:
            mode_fixed = False
            for _fx in range(vfix_iters):
                v_now = _v_from_lp1(pf.bus_voltage, len(v_band[0]))
                q_room = _cluster_q_room(case, pf.gen_output, vctx["cluster"])
                new_v0, new_tk = _vfix_correction(case, v_now, v_band, vctx,
                                                  detect_lo=0.003, target_lo=t_lo,
                                                  detect_hi=0.002, target_hi=t_hi,
                                                  q_room=q_room)
                if not new_v0 and not new_tk:
                    break
                gens_now = case.gens()
                xfs_now = case.xfmrs()
                snap_v = {row: gens_now[row].v0 for row in new_v0}
                snap_t = {row: xfs_now[row].tk for row in new_tk}
                for row, nv in new_v0.items():
                    gens_now[row].v0 = nv
                for row, nt in new_tk.items():
                    xfs_now[row].tk = nt
                case.save_inplace()
                pf2 = solver.run_powerflow(work_dir)
                n_fix_solves += 1
                mode_fixed = True
                if not pf2.converged:
                    # 修正致发散:回滚 v0/tk 重解,恢复到修正前的收敛解再入库
                    gens_now = case.gens()
                    xfs_now = case.xfmrs()
                    for row, ov in snap_v.items():
                        gens_now[row].v0 = ov
                    for row, ot in snap_t.items():
                        xfs_now[row].tk = ot
                    case.save_inplace()
                    pf = solver.run_powerflow(work_dir)
                    n_fix_solves += 1
                    n_fix_reverts += 1
                    converged = pf.converged
                    break
                pf = pf2
            if mode_fixed:
                n_modes_vfixed += 1
        keep = converged or (rng.random() < keep_divergent_frac)
        if not keep:
            continue

        # 重新加载已解算例(结果卡已更新)装配档案
        solved = PsaspCase(work_dir)
        cat = Catalog(solved)
        sid = f"{family_id}_{id_start + i:05d}"
        # schema 贯通(F2,第十二任审查 2026-08-08):原实现不传 → 恒出 25 列,
        # v2 烧机产物进不了 43 列管线。默认 "v0" = 既有库同 seed 复现逐位不变;
        # 万节点/v2 扩增一律显式 "v0.3"。
        gs = build_sample(solved, cat, sample_id=sid, family_id=family_id, schema=schema)
        if not converged:
            # 负类:状态量置 missing、可解性标签=0(装配已按无结果处理;此处显式标注)
            gs.meta["solvable"] = 0
        else:
            gs.meta["solvable"] = 1
        gs.meta["load_factor"] = spec.load_factor

        # 电压带验收(只对收敛样本;发散负类状态本就 missing,不适用)
        if converged and v_floor is not None:
            vmin = _min_valid_voltage(gs)
            if np.isfinite(vmin) and vmin < v_floor:
                lowv_dir.mkdir(parents=True, exist_ok=True)
                gs.save_npz(lowv_dir / f"{sid}.npz")
                lowv_manifest.append({"sample_id": sid, "family_id": family_id,
                                      "solvable": 1, "load_factor": spec.load_factor,
                                      "min_v_pu": float(vmin)})
                n_lowv += 1
                continue                      # 封存留档,不入主库、不计入 n

        gs.save_npz(out_dir / f"{sid}.npz")
        if keep_cases:
            robust_copytree(work_dir, out_dir / "cases" / sid)
        row = {"sample_id": sid, "family_id": family_id,
               "solvable": gs.meta["solvable"], "load_factor": spec.load_factor}
        if diverse:
            row["decommit"] = ladder[li]        # S2 全记录:该样本采样时的停机率
        manifest.append(row)
        if converged:
            n_conv += 1
        else:
            n_div_kept += 1

    _write_manifest(out_dir / "manifest.csv", manifest)
    if lowv_manifest:
        _write_manifest(lowv_dir / "manifest.csv", lowv_manifest)
    return {"n_converged": n_conv, "n_divergent_kept": n_div_kept,
            "n_lowv_quarantined": n_lowv,
            "n_total": len(manifest), "n_attempts": attempts,
            "n_fix_solves": n_fix_solves, "n_fix_reverts": n_fix_reverts,
            "n_modes_vfixed": n_modes_vfixed,
            "decommit_final": (ladder[li] if diverse else None),
            "decommit_steps": decommit_steps,
            "out_dir": str(out_dir)}


def _min_valid_voltage(gs) -> float:
    """样本内有效母线(非 NULL、状态非 missing)的最低电压(pu)。无有效母线返回 nan。"""
    from ..schema.fields import NODE_TABLE, Group
    from ..schema.sample import group_bit_index
    off_v = NODE_TABLE.offset_of("v")
    off_null = NODE_TABLE.offset_of("is_null")
    sbit = group_bit_index(Group.STATE)
    ok = (~gs.node_missing[:, sbit]) & (gs.node_feat[:, off_null] < 0.5)
    if not ok.any():
        return float("nan")
    return float(gs.node_feat[ok, off_v].min())


def _write_manifest(path: Path, rows: List[dict]) -> None:
    import csv
    if not rows:
        path.write_text("sample_id,family_id,solvable,load_factor\n", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
