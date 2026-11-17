# -*- coding: utf-8 -*-
r"""调整器跨规模验证语料生成(2026-08-21;README §10.2-B;样本生成主线)。

在六档健康库(libs/mp_*)的**健康源方式**上造"收敛且电压越限"的病例,PSASP 唯一裁判、串行:
  无功病种(--disease q):按 scripts/reactive_evalgen.py 机制——加大无功负荷 QL 造电压越限
      global:逐负荷独立 ql×U(f_lo,f_hi)(IEEE118 原配方 1.1~1.6);
      local :随机锚点母线 ≤hops 跳内负荷 ql×U(local_f_lo,local_f_hi)(大网用,造局部低压口袋);
  有功病种(--disease p):按 scripts/adjust_evalgen.py 机制——出力再分配造低压
      transfer:k 台机减 ρ×(p−pmin),同量按余量分给 ≥min_hops 跳外机组(零和,平衡机只吃网损变化;
               大网默认——2000/70k 档平衡机 Pmax 仅 13~14 pu,不平衡 >10 pu 即发散);
      band    :逐台非平衡机在 base±band×幅度内独立重抽(adjust_evalgen 原机制,平衡机吸收)。
只留"收敛且越限"的病例;每例落完整卡片目录(含病态 PSASP 解)+ perturb.json(逐台扰动参数)
+ 源方式哈希;jsonl/summary 沿 evalgen 格式。新目录(libs/viol_<q|p>_<tier>)、s92xx 种子段。

电压带(档自适应,写进 eval_summary 供阅卷臂沿用):
  vio_low  = min(0.93, floor01(底图 vmin))   —— case300 底图 vmin=0.9288,0.93 线把底图本身判病;
  vio_high = max(1.08, ceil01(库内最大 vmax)) —— 3120/25k/70k 底图 vmax 1.10~1.14(MATPOWER 1.1 上限口径),
                                                1.08 双边带会把这三档 100% 判"本来就越限"。
源方式筛选:q 需双边健康(vmin≥low 且 vmax≤high),p 需 vmin≥low(口径同 run_adjust 的 healthy 判据)。
70k 档卡片 L0 Meq=1:每次试算前把源方式 LF.P1D 复位(热启动基准=健康解,不用上一抽的解)。

  python viol_evalgen_mp.py --tier 118 --disease q --seed 9201
  python viol_evalgen_mp.py --tier 2000 --disease p --seed 9214 --p-mode transfer
  python viol_evalgen_mp.py --tier 118 --disease q --seed 9201 --mock --out-name _mock_q_118  # 无狗冒烟
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

REPO = Path(r"E:\统一计算大模型-迁移包\增量-20260902\N-k其他推理器实验-20260905\grid-foundation")
LINE = Path(r"E:\统一计算大模型-迁移包\增量-20260902\大算例样本线-20260820")
sys.path.insert(0, str(REPO))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.data.repair_perturb import _clear_stale_pf_outputs  # noqa: E402
from gridfm.fsutil import robust_copytree  # noqa: E402
from gridfm.psasp_io.cards import GenView, LoadView  # noqa: E402
from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402

TIER_LIBS = {"118": ["mp_case118"], "300": ["mp_case300"], "3120": ["mp_case3120sp_fix"],
             "2000": ["mp_activsg2000", "mp_activsg2000_b2"],
             "25k": ["mp_activsg25k", "mp_activsg25k_b2"],
             "70k": ["mp_activsg70k", "mp_activsg70k_b2"],
             # pegase 陌生网泛化档(2026-08-27):池由 adj3_pool_register 注册,底图 validate.json 已随库留档
             "1354": ["mp_case1354pegase_pool"], "2869": ["mp_case2869pegase_pool"],
             "9241": ["mp_case9241pegase_pool"]}
N_BUS = {"118": 118, "300": 300, "3120": 3120, "2000": 2000, "25k": 25000, "70k": 70000,
         "1354": 1354, "2869": 2869, "9241": 9241}
PROFILE = LINE / "调整器跨规模验证-20260821" / "healthy_lib_vprofile.json"
EPS = 5e-4          # 与 reactive_labels._is_violated_2s 同口径
INPUT_CARDS = ("LF.L0", "LF.L1", "LF.L2", "LF.L3", "LF.L5", "LF.L6", "LF.L7")

# ---- 自动升级机制阶梯(2026-08-21 11:50;118 档试水:q 原配方 1.1~1.6 仅 2/300,1.6~2.6 为 25/40;p 零和配对 0/40)----
# --auto 时每 AUTO_CHECK_EVERY 抽检查一次累计病率,低于 AUTO_MIN_RATE 就升到下一阶;已造病例保留,逐例记 mechanism/stage。
AUTO_CHECK_EVERY, AUTO_MIN_RATE = 40, 0.15
# 窗口判向(2026-08-21 12:50 增补;300 档教训:QL×2~3 造出 vmin 0.53~0.74 的崩溃病,三臂 0/30):每窗口 40 抽,
#   病率 <15% 时——健康抽占多数 → 加码(升阶);发散+过深抽占多数 → 减码(降阶);同一阶只降一次(防反复)。
#   过深 = 病例 vmin < low − max_depth(或 vmax > high + max_depth),不入库,计 n_too_deep(可修性口径:与库验收 0.85 底同源)。
Q_STAGES = [dict(q_mode="global", f_lo=1.1, f_hi=1.6),                                  # 0 IEEE118 原配方(118 档 2/300)
            dict(q_mode="global", f_lo=1.6, f_hi=2.6),                                  # 1 (118 档 25/40)
            dict(q_mode="global", f_lo=2.0, f_hi=3.0),                                  # 2 (118 档 30/36;300 档过深)
            dict(q_mode="global", f_lo=3.0, f_hi=5.0),                                  # 3
            dict(q_mode="local", local_hops=2, local_f_lo=4.0, local_f_hi=8.0)]         # 4
P_STAGES = [dict(p_mode="band", band=0.3),                                              # 0 独立限带重抽(原机制;118 试水 0/40)
            dict(p_mode="transfer", k_lo=4, k_hi=8, rho_lo=1.0, rho_hi=1.0, min_hops=4),    # 1 强零和配对(118 1/40)
            dict(p_mode="transfer", k_lo=10, k_hi=25, rho_lo=1.0, rho_hi=1.0, min_hops=3),  # 2 更强零和配对
            dict(p_mode="loadgrowth", lg_lo=1.05, lg_hi=1.15, band=0.0),                    # 3 负荷增长+出力同比上调(截断 Pmax)
            dict(p_mode="loadgrowth", lg_lo=1.10, lg_hi=1.30, band=0.0),                    # 4 (118 0/40)
            dict(p_mode="loadgrowth", lg_lo=1.40, lg_hi=1.80, band=0.0),                    # 5 (118 5/40 → 正式 30/61)
            dict(p_mode="loadgrowth", lg_lo=1.70, lg_hi=2.20, band=0.0)]                    # 6 最重


# ---------------------------------------------------------------- 小工具
def sha256_files(cdir: Path, names=INPUT_CARDS) -> str:
    h = hashlib.sha256()
    for n in names:
        p = cdir / n
        if p.is_file():
            h.update(n.encode("utf-8")); h.update(b"\x00"); h.update(p.read_bytes()); h.update(b"\n")
    return h.hexdigest()


def floor01(x: float) -> float:
    return math.floor(x * 100 + 1e-9) / 100.0


def ceil01(x: float) -> float:
    return math.ceil(x * 100 - 1e-9) / 100.0


def tier_band(tier: str, prof: dict) -> dict:
    """档自适应电压带(见模块 docstring)。"""
    lib0 = TIER_LIBS[tier][0]
    val = json.loads((LINE / "libs" / lib0 / "validate.json").read_text(encoding="utf-8"))
    base_vmin, base_vmax = float(val["psasp_vmin"]), float(val["psasp_vmax"])
    if tier in prof.get("tiers", {}):
        lib_vmax_max = float(prof["tiers"][tier]["vmax_min_med_max"][2])
    else:
        # 新档(pegase 1354/2869)未进 healthy_lib_vprofile:改从审计 JSON 该档逐方式 vmax 取最大,
        # 口径与 profile 完全一致(都是"库内最大 vmax"),不新增自由度。
        aud = json.loads((LINE / "调整器跨规模验证-20260821" / "audit_static_feasibility.json")
                         .read_text(encoding="utf-8"))
        rows = [r for r in aud["tiers"][tier]["rows"] if r["lib"] in TIER_LIBS[tier]]
        assert rows, "审计 JSON 无 %s 档方式行,先跑 adj3_pool_register" % tier
        lib_vmax_max = float(max(r["vmax"] for r in rows))
    low = min(0.93, floor01(base_vmin))
    high = max(1.08, ceil01(lib_vmax_max))
    return {"vio_low": low, "vio_high": high, "base_vmin": base_vmin, "base_vmax": base_vmax,
            "lib_vmax_max": lib_vmax_max, "rule": "low=min(0.93,floor01(base_vmin)); high=max(1.08,ceil01(max lib vmax))"}


def bus_adjacency(work: Path) -> Dict[int, Set[int]]:
    """交流线 + 变压器(mark==1)的母线邻接(比 reactive_apply_eval._bus_adjacency 多含变压器)。"""
    case = PsaspCase(work)
    adj: Dict[int, Set[int]] = {}
    for ln in case.aclines():
        if ln.mark == 1 and ln.i_no != ln.j_no:
            adj.setdefault(ln.i_no, set()).add(ln.j_no)
            adj.setdefault(ln.j_no, set()).add(ln.i_no)
    for tr in case.xfmrs():
        if tr.mark == 1 and tr.i_no != tr.j_no:
            adj.setdefault(tr.i_no, set()).add(tr.j_no)
            adj.setdefault(tr.j_no, set()).add(tr.i_no)
    return adj


def hops_from(adj: Dict[int, Set[int]], seeds: List[int], max_h: Optional[int] = None) -> Dict[int, int]:
    dist = {s: 0 for s in seeds}
    q = deque(seeds)
    while q:
        u = q.popleft()
        if max_h is not None and dist[u] >= max_h:
            continue
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def read_loads(l6: Card) -> Dict[int, dict]:
    out = {}
    for i, r in enumerate(l6.records):
        lv = LoadView(r, i)
        if lv.mark == 1:
            out[i] = {"bus": lv.bs_no, "pl": lv.pl, "ql": lv.ql, "view": lv}
    return out


def read_gens(l5: Card):
    slack, gens = None, {}
    for i, r in enumerate(l5.records):
        g = GenView(r, i)
        if g.mark != 1:
            continue
        if g.type_code == 0:
            slack = g.bs_no
            continue
        gens[i] = {"bus": g.bs_no, "p": g.pg, "pmin": g.pmin, "pmax": g.pmax, "view": g}
    return slack, gens


def span_of(g: dict) -> float:
    return max(g["pmax"] - g["pmin"], abs(g["p"]), 1e-3)


# ---------------------------------------------------------------- 扰动机制
def q_global(loads: Dict[int, dict], rng, f_lo: float, f_hi: float) -> dict:
    fac = {i: float(rng.uniform(f_lo, f_hi)) for i in loads}
    for i, f in fac.items():
        loads[i]["view"].ql = loads[i]["ql"] * f
    return {"mode": "global", "factors": {str(i): round(f, 5) for i, f in fac.items()},
            "n_loads_scaled": len(fac), "mean_factor": round(float(np.mean(list(fac.values()))), 4)}


def q_local(loads: Dict[int, dict], adj, rng, hops: int, f_lo: float, f_hi: float) -> dict:
    cand = sorted({ld["bus"] for ld in loads.values() if abs(ld["ql"]) > 1e-6 or abs(ld["pl"]) > 1e-6})
    anchor = int(cand[int(rng.integers(0, len(cand)))])
    region = set(hops_from(adj, [anchor], hops).keys())
    fac = {}
    for i, ld in loads.items():
        if ld["bus"] in region:
            f = float(rng.uniform(f_lo, f_hi))
            ld["view"].ql = ld["ql"] * f
            fac[i] = f
    return {"mode": "local", "anchor_bus": anchor, "hops": hops, "region_n_bus": len(region),
            "factors": {str(i): round(f, 5) for i, f in fac.items()},
            "n_loads_scaled": len(fac), "mean_factor": round(float(np.mean(list(fac.values()))), 4) if fac else None}


def p_band(gens: Dict[int, dict], rng, band: float) -> dict:
    """adjust_evalgen.redraw_dispatch(限带)逐位同语义:base±band×幅度内独立均匀重抽。"""
    out = {}
    for gi, g in gens.items():
        sp = span_of(g)
        lo = max(g["pmin"], g["p"] - band * sp)
        hi = min(g["pmax"] if g["pmax"] > g["pmin"] + 1e-9 else g["p"] + band * sp, g["p"] + band * sp)
        p = float(rng.uniform(lo, hi)) if hi > lo + 1e-9 else float(lo)
        g["view"].pg = p
        out[gi] = p
    net = sum(out[gi] - gens[gi]["p"] for gi in gens)
    return {"mode": "band", "band": band, "n_gens_changed": len(out),
            "net_delta_pu": round(net, 4),
            "dispatch": {str(gens[gi]["bus"]): round(p, 4) for gi, p in out.items()}}


def p_loadgrowth(loads: Dict[int, dict], gens: Dict[int, dict], rng, lg_lo: float, lg_hi: float,
                 band: float = 0.0) -> dict:
    """负荷增长 + 出力同比上调:逐负荷 PL×U(lg_lo,lg_hi)(QL 不动);非平衡机 PG×(全网负荷增量比)截断 [Pmin,Pmax]
    (截断造成的差额由平衡机吸收);band>0 时再叠加 base±band×幅度的独立抖动。低压来自重潮流,修法仍是有功再分配。"""
    pl0 = sum(ld["pl"] for ld in loads.values())
    fac = {}
    for i, ld in loads.items():
        f = float(rng.uniform(lg_lo, lg_hi))
        ld["view"].pl = ld["pl"] * f
        fac[i] = f
    pl1 = sum(ld["pl"] * fac[i] for i, ld in loads.items())
    ratio = pl1 / pl0 if pl0 > 1e-9 else 1.0
    pg0 = sum(g["p"] for g in gens.values())
    out = {}
    for gi, g in gens.items():
        p = g["p"] * ratio
        if band > 0:
            sp = span_of(g)
            p += float(rng.uniform(-band * sp, band * sp))
        hi = g["pmax"] if g["pmax"] > g["pmin"] + 1e-9 else p
        p = float(min(max(p, g["pmin"]), hi))
        g["view"].pg = p
        out[gi] = p
    pg1 = sum(out.values())
    return {"mode": "loadgrowth", "lg_lo": lg_lo, "lg_hi": lg_hi, "band": band,
            "load_ratio": round(ratio, 4), "pl_total_before": round(pl0, 2), "pl_total_after": round(pl1, 2),
            "pg_total_before": round(pg0, 2), "pg_total_after": round(pg1, 2),
            "net_delta_pu": round((pl1 - pl0) - (pg1 - pg0), 4),
            "n_loads_scaled": len(fac), "n_gens_changed": len(out),
            "factors": {str(i): round(f, 5) for i, f in fac.items()},
            "dispatch": {str(gens[gi]["bus"]): round(p, 4) for gi, p in out.items()}}


def p_lg_local(loads: Dict[int, dict], adj, rng, hops: int, lg_lo: float, lg_hi: float) -> dict:
    """只放大 PL 的局部负荷口袋(锚点 ≤hops 跳内 PL×U(lg_lo,lg_hi),QL 与机组不动;2026-08-23 C 项立项机制,
    对应 adj2_labelgen --mech loadgrowth_local --no-scale-q)。低压来自口袋内重潮流,修法 = 就近机组加出力。"""
    cand = sorted({ld["bus"] for ld in loads.values() if abs(ld["pl"]) > 1e-6})
    anchor = int(cand[int(rng.integers(0, len(cand)))])
    region = set(hops_from(adj, [anchor], hops).keys())
    fac = {}
    for i, ld in loads.items():
        if ld["bus"] in region:
            f = float(rng.uniform(lg_lo, lg_hi))
            ld["view"].pl = ld["pl"] * f
            fac[i] = f
    dpl = sum(loads[i]["pl"] * (f - 1.0) for i, f in fac.items())
    return {"mode": "lg_local", "anchor_bus": anchor, "hops": hops, "region_n_bus": len(region), "lg": [lg_lo, lg_hi],
            "factors": {str(i): round(f, 5) for i, f in fac.items()}, "n_loads_scaled": len(fac),
            "mean_factor": (round(float(np.mean(list(fac.values()))), 4) if fac else None), "dpl_total_pu": round(dpl, 4)}


def p_transfer(gens: Dict[int, dict], adj, rng, k_lo: int, k_hi: int,
               rho_lo: float, rho_hi: float, min_hops: int) -> Optional[dict]:
    """零和配对再分配:k 台施主减 ρ×(p−pmin);同量按上行余量比例分给 ≥min_hops 跳外的受主。"""
    donors_pool = [gi for gi, g in gens.items() if g["p"] - g["pmin"] > 1e-3 and g["pmax"] > g["pmin"] + 1e-9]
    if not donors_pool:
        return None
    k = int(rng.integers(k_lo, k_hi + 1))
    k = min(k, len(donors_pool))
    donors = [donors_pool[int(c)] for c in rng.choice(len(donors_pool), size=k, replace=False)]
    red = {gi: float(rng.uniform(rho_lo, rho_hi)) * (gens[gi]["p"] - gens[gi]["pmin"]) for gi in donors}
    total = sum(red.values())
    dist = hops_from(adj, [gens[gi]["bus"] for gi in donors])
    recv = [gi for gi, g in gens.items()
            if gi not in red and dist.get(g["bus"], 10 ** 9) >= min_hops
            and g["pmax"] > g["pmin"] + 1e-9 and g["pmax"] - g["p"] > 1e-3]
    room = {gi: gens[gi]["pmax"] - gens[gi]["p"] for gi in recv}
    cap = sum(room.values())
    if cap < 1e-6:
        return None
    if cap < total:                        # 受主余量不够:施主按比例少减,保持零和
        scale = cap / total
        red = {gi: v * scale for gi, v in red.items()}
        total = cap
    inc = {gi: total * r / cap for gi, r in room.items()}
    for gi, v in red.items():
        gens[gi]["view"].pg = gens[gi]["p"] - v
    for gi, v in inc.items():
        gens[gi]["view"].pg = gens[gi]["p"] + v
    return {"mode": "transfer", "k_donors": k, "min_hops": min_hops,
            "donors": {str(gens[gi]["bus"]): round(-v, 4) for gi, v in red.items()},
            "n_receivers": len(inc), "transfer_pu": round(total, 4),
            "receivers_top": {str(gens[gi]["bus"]): round(v, 4)
                              for gi, v in sorted(inc.items(), key=lambda t: -t[1])[:20]},
            "receivers_all": {str(gens[gi]["bus"]): round(v, 5) for gi, v in inc.items()},
            "net_delta_pu": round(sum(inc.values()) - sum(red.values()), 6)}


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=list(TIER_LIBS))
    ap.add_argument("--disease", required=True, choices=["q", "p"])
    ap.add_argument("--seed", type=int, required=True, help="s92xx 段")
    ap.add_argument("--n-target", type=int, default=30)
    ap.add_argument("--tries-per-mode", type=int, default=6)
    ap.add_argument("--max-draws", type=int, default=400)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--vio-low", type=float, default=None, help="缺省档自适应")
    ap.add_argument("--vio-high", type=float, default=None, help="缺省档自适应")
    # q
    ap.add_argument("--q-mode", default="global", choices=["global", "local"])
    ap.add_argument("--f-lo", type=float, default=1.1)
    ap.add_argument("--f-hi", type=float, default=1.6)
    ap.add_argument("--local-hops", type=int, default=2)
    ap.add_argument("--local-f-lo", type=float, default=1.5)
    ap.add_argument("--local-f-hi", type=float, default=3.0)
    # p
    ap.add_argument("--p-mode", default="transfer", choices=["transfer", "band", "loadgrowth", "lg_local"])
    ap.add_argument("--exclude-from", nargs="*", default=[],
                    help="排除源方式:目录列表,取其 adjust_labels.jsonl / eval_cases.jsonl 里的 sid(防训练/旧考卷泄漏;2026-08-23)")
    ap.add_argument("--p-slack-in-band", action="store_true",
                    help="有功:只用平衡机解得出力在 [Pmin,Pmax] 内的源方式(与 adj2_labelgen 有功筛同口径;2026-08-23)")
    ap.add_argument("--lg-lo", type=float, default=1.10)
    ap.add_argument("--lg-hi", type=float, default=1.30)
    ap.add_argument("--auto-start-stage", type=int, default=0, help="--auto 时从第几阶起(大档跳过注定失败的低阶)")
    ap.add_argument("--band", type=float, default=0.3)
    ap.add_argument("--k-lo", type=int, default=1)
    ap.add_argument("--k-hi", type=int, default=3)
    ap.add_argument("--rho-lo", type=float, default=0.5)
    ap.add_argument("--rho-hi", type=float, default=1.0)
    ap.add_argument("--min-hops", type=int, default=3)
    ap.add_argument("--mock", action="store_true", help="无 PSASP 冒烟(假裁判,奇数抽判病)")
    ap.add_argument("--auto", action="store_true",
                    help="自动升级机制阶梯(Q_STAGES/P_STAGES):每 40 抽查累计病率,<15%% 升一阶;逐例记 stage")
    ap.add_argument("--auto-min-rate", type=float, default=AUTO_MIN_RATE, help="升级阈值(窗口病率),测试用")
    ap.add_argument("--max-depth", type=float, default=0.08,
                    help="病例深度上限:vmin < low-max_depth(或 vmax > high+max_depth)判'过深'不入库(可修性口径)")
    a = ap.parse_args()
    stages = (Q_STAGES if a.disease == "q" else P_STAGES) if a.auto else None
    stage_i = min(a.auto_start_stage, len(stages) - 1) if stages else 0

    def apply_stage(i: int) -> None:
        for k, v in stages[i].items():
            setattr(a, k, v)
    if stages:
        apply_stage(stage_i)

    prof = json.loads(PROFILE.read_text(encoding="utf-8"))
    band = tier_band(a.tier, prof)
    low = a.vio_low if a.vio_low is not None else band["vio_low"]
    high = a.vio_high if a.vio_high is not None else band["vio_high"]

    out_name = a.out_name or ("viol_%s_%s" % (a.disease, a.tier))
    out = LINE / "libs" / out_name
    if (out / "eval_summary.json").exists():
        print("[viol] %s 已有 eval_summary.json,拒绝覆盖(新产物新目录纪律)" % out, flush=True)
        return 3
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases").mkdir(exist_ok=True)
    work = LINE / "_work" / out_name
    work.parent.mkdir(parents=True, exist_ok=True)

    # 源方式池:按档自适应带筛健康源,再按 seed 洗牌
    pool = []
    for m in prof["tiers"][a.tier]["modes"]:
        ok = (m["vmin"] >= low - EPS) and (a.disease == "p" or m["vmax"] <= high + EPS)
        if ok:
            pool.append(m)
    excluded: Set[str] = set()
    for d in a.exclude_from:
        for fn in ("adjust_labels.jsonl", "eval_cases.jsonl"):
            fp = Path(d) / fn
            if fp.is_file():
                for ln in fp.read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        excluded.add(json.loads(ln)["sid"])
    if excluded:
        n_before = len(pool)
        pool = [m for m in pool if m["sid"] not in excluded]
        print("[viol] 排除 %d 个源方式(来自 %s):%d → %d" % (len(excluded), a.exclude_from, n_before, len(pool)), flush=True)
    if a.disease == "p" and a.p_slack_in_band:
        sys.path.insert(0, str(LINE / "scripts"))
        from adj2_common import slack_info as _slack_info
        keep = []
        for m in pool:
            si = _slack_info(LINE / "libs" / m["lib"] / "cases" / m["sid"])
            if si.get("dist") is not None and si["dist"] <= 1e-9:
                keep.append(m)
        print("[viol] 平衡机在限内筛:%d → %d" % (len(pool), len(keep)), flush=True)
        pool = keep
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(pool))
    pool = [pool[int(i)] for i in order]
    print("[viol] tier=%s disease=%s band=[%.2f,%.2f] 源池 %d/%d(健康筛后) seed=%d out=%s"
          % (a.tier, a.disease, low, high, len(pool), len(prof["tiers"][a.tier]["modes"]), a.seed, out), flush=True)

    if a.mock:
        from gridfm.solver.mock import MockSolver
        solver = MockSolver(force_converged=True)
        _try_counter = {"n": 0}

        def referee(wdir):                 # 假裁判:奇数抽判"病"(vmin 0.90),偶数抽健康;只验管线
            _try_counter["n"] += 1
            sick = _try_counter["n"] % 2 == 1
            return {"converged": True, "vmin": 0.90 if sick else 0.96, "vmax": 1.05,
                    "v": np.full(N_BUS[a.tier], 0.90 if sick else 0.96)}
    else:
        from gridfm.data.synth import psasp_referee
        from gridfm.solver.psasp import PsaspSolver
        solver = PsaspSolver()

        def referee(wdir):
            return psasp_referee(wdir, solver)

    from gridfm.solver.psasp import LicenseError

    rows: List[dict] = []
    n_sick = n_healthy = n_div = n_draw = 0
    n_src_used = 0
    t0 = time.time()
    fused = False
    stage_log: List[dict] = [{"stage": stage_i, "params": dict(stages[stage_i]), "from_draw": 0}] if stages else []
    sick_by_stage: Dict[int, int] = {}
    n_deep = 0
    win = {"sick": 0, "healthy": 0, "div": 0, "deep": 0, "start": 0}     # 当前窗口计数
    deesc_done: Set[int] = set()                                         # 已降过的阶(防反复)

    def flush():
        (out / "eval_cases.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
            encoding="utf-8")

    try:
        for m in pool:
            if n_sick >= a.n_target or n_draw >= a.max_draws:
                break
            src = LINE / "libs" / m["lib"] / "cases" / m["sid"]
            if not src.is_dir():
                continue
            n_src_used += 1
            robust_copytree(src, work)
            src_hash = sha256_files(src)
            adj = bus_adjacency(work) if (a.disease == "p" and a.p_mode in ("transfer", "lg_local")) or \
                (a.disease == "q" and a.q_mode == "local") else None
            for t in range(a.tries_per_mode):
                if n_sick >= a.n_target or n_draw >= a.max_draws:
                    break
                # 自动判向:每窗口 40 抽查窗口病率;太轻→升阶,太重→降阶
                wn = win["sick"] + win["healthy"] + win["div"] + win["deep"]
                if stages and wn >= AUTO_CHECK_EVERY and stage_log[-1]["from_draw"] != n_draw:
                    rate = win["sick"] / wn
                    if rate < a.auto_min_rate:
                        too_strong = (win["div"] + win["deep"]) > win["healthy"]
                        new_i = None
                        if too_strong and stage_i > 0 and stage_i not in deesc_done:
                            new_i = stage_i - 1
                            deesc_done.add(stage_i)
                        elif not too_strong and stage_i < len(stages) - 1:
                            new_i = stage_i + 1
                        if new_i is not None:
                            direction = "down" if new_i < stage_i else "up"
                            stage_log.append({"stage": new_i, "params": dict(stages[new_i]), "from_draw": n_draw,
                                              "window": dict(win), "direction": direction})
                            print("[viol] 窗口病率 %.1f%% (健康 %d/发散 %d/过深 %d) -> %s 到阶 %d %s" % (
                                100 * rate, win["healthy"], win["div"], win["deep"], direction, new_i, stages[new_i]),
                                flush=True)
                            stage_i = new_i
                            apply_stage(stage_i)
                            adj = bus_adjacency(work) if (a.disease == "p" and a.p_mode in ("transfer", "lg_local")) or \
                                (a.disease == "q" and a.q_mode == "local") else adj
                    win = {"sick": 0, "healthy": 0, "div": 0, "deep": 0, "start": n_draw}
                # 每抽从源方式重置:先清旧结果(PF_OUTPUT_FILES 含 LF.P1D!),再复位改动卡 + 热启动基准 P1D
                # (2026-08-21 15:55 修:原先先拷 P1D 再清 → 70k 退回平启动 120/120 发散)
                if not a.mock:
                    _clear_stale_pf_outputs(work)
                for name in ("LF.L5", "LF.L6", "LF.P1D"):
                    if (src / name).is_file():
                        shutil.copy2(src / name, work / name)
                if a.disease == "q":
                    l6 = Card.load(work / "LF.L6")
                    loads = read_loads(l6)
                    pert = (q_global(loads, rng, a.f_lo, a.f_hi) if a.q_mode == "global"
                            else q_local(loads, adj, rng, a.local_hops, a.local_f_lo, a.local_f_hi))
                    l6.save(work / "LF.L6")
                else:
                    l5 = Card.load(work / "LF.L5")
                    _slack, gens = read_gens(l5)
                    if a.p_mode == "transfer":
                        pert = p_transfer(gens, adj, rng, a.k_lo, a.k_hi, a.rho_lo, a.rho_hi, a.min_hops)
                    elif a.p_mode == "band":
                        pert = p_band(gens, rng, a.band)
                    elif a.p_mode == "lg_local":
                        l6p = Card.load(work / "LF.L6")
                        pert = p_lg_local(read_loads(l6p), adj, rng, a.local_hops, a.lg_lo, a.lg_hi)
                        l6p.save(work / "LF.L6")
                    else:
                        l6p = Card.load(work / "LF.L6")
                        pert = p_loadgrowth(read_loads(l6p), gens, rng, a.lg_lo, a.lg_hi, a.band)
                        l6p.save(work / "LF.L6")
                    if pert is None:
                        continue
                    l5.save(work / "LF.L5")
                n_draw += 1
                rf = referee(work)
                if not rf.get("converged"):
                    n_div += 1
                    win["div"] += 1
                    continue
                vmin, vmax = float(rf["vmin"]), float(rf["vmax"])
                if a.disease == "q":
                    violated = vmin < low - EPS or vmax > high + EPS
                    healthy = vmin >= low - EPS and vmax <= high + EPS
                    too_deep = vmin < low - a.max_depth or vmax > high + a.max_depth
                else:
                    violated = vmin < low - EPS
                    healthy = vmin >= low - EPS
                    too_deep = vmin < low - a.max_depth
                if violated and too_deep:
                    n_deep += 1
                    win["deep"] += 1
                    continue
                if violated:
                    win["sick"] += 1
                    n_sick += 1
                    sick_by_stage[stage_i] = sick_by_stage.get(stage_i, 0) + 1
                    pert["stage"] = stage_i
                    cid = "%s__%s%02d" % (m["sid"], a.disease, t + 1)
                    v = np.asarray(rf["v"])
                    live = v > 0.01
                    buses_low = [int(i) + 1 for i in np.where(live & (v < low - EPS))[0]]
                    buses_high = [int(i) + 1 for i in np.where(live & (v > high + EPS))[0]] if a.disease == "q" else []
                    robust_copytree(work, out / "cases" / cid)
                    (out / "cases" / cid / "perturb.json").write_text(
                        json.dumps({"case_id": cid, "source_sid": m["sid"], "source_lib": m["lib"],
                                    "source_hash_sha256": src_hash, "disease": a.disease,
                                    "try": t + 1, "seed": a.seed, "perturb": pert},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
                    row = {"case_id": cid, "sid": m["sid"], "source_lib": m["lib"], "tier": a.tier,
                           "disease": a.disease, "try": t + 1,
                           "source_hash_sha256": src_hash,
                           "source_vmin": m["vmin"], "source_vmax": m["vmax"],
                           "vmin": round(vmin, 4), "vmax": round(vmax, 4),
                           "n_bus_low": len(buses_low), "n_bus_high": len(buses_high),
                           "buses_low": buses_low[:50], "buses_high": buses_high[:50],
                           "perturb_summary": {k: v for k, v in pert.items()
                                               if k not in ("factors", "dispatch", "receivers_all")}}
                    rows.append(row)
                    print("[viol] %s 第%d抽: 病态 vmin=%.4f vmax=%.4f 低压母线 %d 高压母线 %d (%d/%d, %.0fs)"
                          % (m["sid"], t + 1, vmin, vmax, len(buses_low), len(buses_high),
                             n_sick, a.n_target, time.time() - t0), flush=True)
                    flush()                 # 逐例落盘:任意时刻暂停不丢已落病例(病例目录已整拷)
                elif healthy:
                    n_healthy += 1
                    win["healthy"] += 1
    except LicenseError as e:
        fused = True
        print("[viol] 加密狗失效熔断,立即停止不重试: %s" % e, flush=True)
    finally:
        flush()
        shutil.rmtree(work, ignore_errors=True)

    secs = round(time.time() - t0, 1)
    sz = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    summary = {"tier": a.tier, "disease": a.disease, "family": ("A_qload" if a.disease == "q" else "P_redispatch"),
               "mechanism": (a.q_mode if a.disease == "q" else a.p_mode),
               "params": ({"f_lo": a.f_lo, "f_hi": a.f_hi} if a.disease == "q" and a.q_mode == "global" else
                          {"hops": a.local_hops, "f_lo": a.local_f_lo, "f_hi": a.local_f_hi} if a.disease == "q" else
                          {"k": [a.k_lo, a.k_hi], "rho": [a.rho_lo, a.rho_hi], "min_hops": a.min_hops}
                          if a.p_mode == "transfer" else {"lg": [a.lg_lo, a.lg_hi], "band": a.band}
                          if a.p_mode == "loadgrowth" else {"hops": a.local_hops, "lg": [a.lg_lo, a.lg_hi], "pl_only": True}
                          if a.p_mode == "lg_local" else {"band": a.band}),
               "excluded_sids": sorted(excluded), "p_slack_in_band": bool(a.disease == "p" and a.p_slack_in_band),
               "source_libs": TIER_LIBS[a.tier], "n_source_pool": len(pool), "n_source_used": n_src_used,
               "n_base_modes": len(set(r["sid"] for r in rows)),
               "n_draws": n_draw, "n_sick": n_sick, "n_healthy_draw": n_healthy, "n_diverged": n_div,
               "n_too_deep": n_deep, "max_depth": a.max_depth,
               "sick_rate": round(n_sick / max(n_draw, 1), 3), "diverge_rate": round(n_div / max(n_draw, 1), 3),
               "criterion": ("bilateral" if a.disease == "q" else "vmin_line"),
               "vio_band": [low, high], "band_rule": band, "eps": EPS,
               "auto_escalate": bool(stages), "stage_log": stage_log,
               "sick_by_stage": {str(k): v for k, v in sorted(sick_by_stage.items())},
               "final_stage_params": (dict(stages[stage_i]) if stages else None),
               "seed": a.seed, "tries_per_mode": a.tries_per_mode, "max_draws": a.max_draws,
               "hit_target": n_sick >= a.n_target, "license_fused": fused, "mock": a.mock,
               "wall_secs": secs, "secs_per_draw": round(secs / max(n_draw, 1), 2), "lib_mb": round(sz, 1),
               "note": "非逆扰动独立来源病例;无修复标签;阅卷走功能性判据(run_adjust 三臂,PSASP 裁判)"}
    (out / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[viol] 收尾:病态 %d/%d 抽(源方式 %d,健康 %d,发散 %d,过深 %d) %.0fs %.1fMB 熔断=%s"
          % (n_sick, n_draw, summary["n_base_modes"], n_healthy, n_div, n_deep, secs, sz, fused), flush=True)
    return 1 if fused else 0


if __name__ == "__main__":
    raise SystemExit(main())
