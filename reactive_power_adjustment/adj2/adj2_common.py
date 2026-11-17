# -*- coding: utf-8 -*-
r"""调整器效果提升线(2026-08-22)公共层:路径/档配置/合格口径闸/平衡机裕度/热稳定审计/侧通道特征。

只读主仓库 gridfm(冻结面),本线一切新代码放本目录 adj2_*.py,产物进 libs/labels_* 与 调整器效果提升-20260822/。
口径(用户 2026-08-22 裁决):静稳电压 = 算例自带逐母线 VMIN/VMAX;热稳定 = S/rateA ≤ 1;暂稳不做。
标签合格闸 = 修后 (a) 电压:全母线在自带带内 且 vmin≥考卷低线 且 vmax≤考卷高线(3120 档放宽为 BVR≤0.1% 且不新增越限母线,须声明);
           (b) 热:不新增过载支路(有额定值的档;118 无额定值标"不可审");
           (c) 有功臂加平衡机合理性:修后平衡机到 [Pmin,Pmax] 带的距离 ≤ 修前 + 0.1 pu(B 段 §4b-4 相对判据)。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

REPO = Path(r"E:\统一计算大模型-迁移包\增量-20260902\N-k其他推理器实验-20260905\grid-foundation")
LINE = Path(r"E:\统一计算大模型-迁移包\增量-20260902\大算例样本线-20260820")
PKG = Path(r"E:\统一计算大模型-迁移包\增量-20260902\内网部署包-Linux-20260827")
MP = Path(r"E:\统一计算大模型-迁移包\增量-20260902\MATPOWER转换-20260818\matpower_cases")
OUT = Path(r"E:\统一计算大模型-迁移包\增量-20260902\N-k其他推理器实验-20260905")   # 2026-09-05 N-k 线：产物落本线目录
AUDIT_JSON = LINE / "调整器跨规模验证-20260821" / "audit_static_feasibility.json"
B_RUNS = LINE / "调整器跨规模验证-20260821" / "runs"
PY = Path(sys.executable)
PSASP_BIN = Path(r"D:\simulation\DISK1中国电力科学研究院有限公司7.95.07.10758冬滚正式版\PSASP")
for _p in (str(REPO), str(Path(__file__).resolve().parent)):   # 2026-09-05: 同目录补丁副本优先，不再挂原线 scripts
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.psasp_io import cards  # noqa: E402
from gridfm.psasp_io.cards import GenView, LoadView  # noqa: E402
from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402
from gridfm.schema.fields import NODE_TABLE  # noqa: E402
from gridfm.schema.sample import GridSample  # noqa: E402

EPS = 5e-4
MAX_DEPTH = 0.08            # 病例深度上限(与 B 段"过深不入库"同口径:vmin ≥ low−0.08)
SLACK_TOL = 0.1             # 平衡机相对判据容差(pu)
RATE_SUSPECT = 10.0         # 额定值 <10 MVA 视为可疑不计(与审计脚本一致)
INPUT_CARDS = ("LF.L0", "LF.L1", "LF.L2", "LF.L3", "LF.L5", "LF.L6", "LF.L7")
KEEP_FILES = INPUT_CARDS + ("DATALIB.DAT", "LF.AREA", "LF.ZONE", "PSASPDATA.INI", "LF.BUS", "LF.L20",
                            "LF.ML4", "LF.PSTATION", "LF.UDP", "PHYBUS.LFI", "convert_info.json")
PF_OUTPUT_FILES = ("LF.CAL", "LF.LP1", "LF.LP2", "LF.LP3", "LF.LP5", "LF.LP6", "LF.P1D")

# 档配置:考卷带沿 libs/viol_*/eval_summary.json(118/2000 = 0.93/1.08;3120 = 0.93/1.12);自带带来自 MATPOWER bus 表。
TIERS = {
    "118": dict(libs=["mp_case118"], mfile="case118.m", battery="mp_case118",
                exam_low=0.93, exam_high=1.08, relaxed=False, psasp_secs=1.8),
    "2000": dict(libs=["mp_activsg2000", "mp_activsg2000_b2"], mfile="case_ACTIVSg2000.m",
                 battery="mp_activsg2000", exam_low=0.93, exam_high=1.08, relaxed=False, psasp_secs=2.5),
    "3120": dict(libs=["mp_case3120sp_fix"], mfile="case3120sp.m", battery="mp_case3120sp",
                 exam_low=0.93, exam_high=1.12, relaxed=True, psasp_secs=2.9),
    # pegase 两档(2026-08-27 泛化线新增;口径依据见 pegase-20260827/结果档 §A1/§A2):
    #   底图自带带(VMIN 0.9/0.7, VMAX 1.1/1.3)逐母线越限 = 0 → 电压照 2000 严格口径 relaxed=False;
    #   exam_high 按本线既定规则 max(1.08, ceil01(库内最大 vmax)):1354 池 vmax≤1.1179 → 1.12;
    #   2869 池 vmax≤1.1500 → 1.15(两网底图 vmax 1.108/1.1412 本就 >1.08,不放宽则基线自身出带、全部方式被跳过);
    #   底图自带热稳过载 10 条(1354, max 1.093) / 2 条(2869, max 1.026) → pool_thermal_zero=False
    #   (沿用项目既定"相对口径、不修容量数据";合格闸 label_gate 本就按"不新增过载"判,不受影响)。
    "1354": dict(libs=["mp_case1354pegase_pool"], mfile="case1354pegase.m", battery="mp_case1354pegase",
                 exam_low=0.93, exam_high=1.12, relaxed=False, pool_thermal_zero=False, psasp_secs=1.8),
    "2869": dict(libs=["mp_case2869pegase_pool"], mfile="case2869pegase.m", battery="mp_case2869pegase",
                 exam_low=0.93, exam_high=1.15, relaxed=False, pool_thermal_zero=False, psasp_secs=2.0),
    # 9241pegase(2026-08-27 长线新增):同为 PEGASE 族,移相器 66 条但角度 ≤0.489°(ref_nr 3.65e-4 pu)风险同 1354/2869;
    #   ⚠ 底图自带一个结构性低压区——VMIN=0.9 组里 42 条母线 <0.93、PSASP 实测全网 vmin 0.8235,
    #   故考卷线按本线既定规则(照 case300 先例)只能是 min(0.93, floor01(0.8235)) = 0.82,
    #   **成功率不可与 1354/2869 横比,只能横比"开预筛前后的相对增益"**;
    #   exam_high 0.82/1.20 为暂定值,真正的考卷带由 viol_evalgen_mp.tier_band 从方式池实测算出并写进 eval_summary。
    "9241": dict(libs=["mp_case9241pegase_pool"], mfile="case9241pegase.m", battery="mp_case9241pegase",
                 exam_low=0.82, exam_high=1.20, relaxed=False, pool_thermal_zero=False, psasp_secs=2.7),
}
EXAM_LIBS = {"118": ["viol_q_118", "viol_p_118"], "2000": ["viol_q_2000"],
             "3120": ["viol_q_3120", "viol_p_3120", "viol_t_3120"], "25k": ["viol_q_25k"], "70k": ["viol_q_70k"],
             "1354": ["viol_q_1354_local"], "2869": ["viol_q_2869_local"],
             "9241": ["viol_q_9241_local"], "300": ["viol_q_300", "viol_p_300", "viol_q_300_local"]}


# ---------------------------------------------------------------- MATPOWER 自带限值 / 额定值
_MP_CACHE: Dict[str, tuple] = {}


def mp_limits(tier: str):
    """(baseMVA, vmin_own[N], vmax_own[N], rate[M], has_rate[M]);母线序 = 转换器 L1 行序(与审计脚本同源 read_m)。"""
    if tier in _MP_CACHE:
        return _MP_CACHE[tier]
    from audit_static_feasibility import read_m
    base, bus, br = read_m(MP / TIERS[tier]["mfile"])
    rate = br[:, 5]
    has_rate = rate >= RATE_SUSPECT
    _MP_CACHE[tier] = (base, bus[:, 12].copy(), bus[:, 11].copy(), rate, has_rate)
    return _MP_CACHE[tier]


# ---------------------------------------------------------------- 结果卡读数
def bus_voltage(case_dir: Path, n_bus: int) -> np.ndarray:
    v = np.zeros(n_bus)
    if not (case_dir / "LF.LP1").is_file():
        return v
    for r in cards.parse_lp1(Card.load(case_dir / "LF.LP1")):
        if 1 <= r.bus_no <= n_bus and np.isfinite(r.v_pu) and r.v_pu < 100:
            v[r.bus_no - 1] = r.v_pu
    return v


def voltage_audit(v: np.ndarray, tier: str, low_exam: float, high_exam: float) -> dict:
    """电压审计:自带带逐母线越限数/BVR + 考卷线。v 为 0 基(失电 v<=0.01 不计)。"""
    _b, vmin_own, vmax_own, _r, _h = mp_limits(tier)
    n = min(len(v), len(vmin_own))
    vv, lo, hi = v[:n], vmin_own[:n], vmax_own[:n]
    live = vv > 0.01
    viol = live & ((vv < lo - 1e-4) | (vv > hi + 1e-4))
    nlive = int(live.sum())
    return {"vmin": float(vv[live].min()) if nlive else None, "vmax": float(vv[live].max()) if nlive else None,
            "n_live": nlive, "n_viol_own": int(viol.sum()), "bvr_own": float(viol.sum() / max(nlive, 1)),
            "viol_own_buses": [int(i) + 1 for i in np.where(viol)[0][:50]],
            "in_exam": bool(nlive and vv[live].min() >= low_exam - EPS and vv[live].max() <= high_exam + EPS)}


def thermal_audit(case_dir: Path, tier: str) -> dict:
    """热稳定审计:交流线 LP2 + 变压器 LP3,idname L{k}_ac / T{k}_2w / T{k}_sf → rateA[k](可疑额定值不计)。"""
    base, _lo, _hi, rate, has_rate = mp_limits(tier)
    if not has_rate.any():
        return {"available": False, "n_checked": 0, "n_over": None, "over_ids": [], "max_loading": None}
    ratios, over = [], []
    for fname in ("LF.LP2", "LF.LP3"):
        p = case_dir / fname
        if not p.is_file():
            continue
        for fl in cards.parse_lp2(Card.load(p)):
            m = re.match(r"[LT](\d+)_", (fl.name or "").strip())
            if not m:
                continue
            k = int(m.group(1))
            if k >= len(rate) or not has_rate[k]:
                continue
            s = max(abs(complex(fl.pi, fl.qi)), abs(complex(fl.pj, fl.qj))) * base
            r = s / rate[k]
            ratios.append(r)
            if r > 1.0 + 1e-6:
                over.append(k)
    return {"available": True, "n_checked": len(ratios), "n_over": len(over), "over_ids": sorted(over),
            "max_loading": (round(float(max(ratios)), 4) if ratios else None)}


def slack_info(case_dir: Path) -> dict:
    """平衡机:母线号、[Pmin,Pmax]、卡片 Pg、LP5 解得 Pg、到带距离(带内 0)。"""
    l5 = Card.load(case_dir / "LF.L5")
    sl = None
    for i, r in enumerate(l5.records):
        g = GenView(r, i)
        if g.mark == 1 and g.type_code == 0:
            sl = g
            break
    if sl is None:
        return {"bus": None, "dist": None}
    p_solved = None
    if (case_dir / "LF.LP5").is_file():
        for gr in cards.parse_lp5(Card.load(case_dir / "LF.LP5")):
            if gr.bus_no == sl.bs_no and np.isfinite(gr.pg):
                p_solved = (p_solved or 0.0) + float(gr.pg)
    pmin, pmax = float(sl.pmin), float(sl.pmax)
    limited = pmax > pmin + 1e-9
    dist = None
    if p_solved is not None and limited:
        dist = max(0.0, pmin - p_solved, p_solved - pmax)
    return {"bus": int(sl.bs_no), "pmin": pmin, "pmax": pmax, "limited": limited, "p_card": float(sl.pg),
            "p_solved": (round(p_solved, 4) if p_solved is not None else None),
            "dist": (round(dist, 4) if dist is not None else None),
            "up_headroom": (round(pmax - p_solved, 4) if (p_solved is not None and limited) else None),
            "down_headroom": (round(p_solved - pmin, 4) if (p_solved is not None and limited) else None)}


def slack_plausible(before: dict, after: dict, tol: float = SLACK_TOL) -> Optional[bool]:
    d0, d1 = before.get("dist"), after.get("dist")
    if d0 is None or d1 is None:
        return None
    return bool(d1 <= d0 + tol)


def full_audit(case_dir: Path, tier: str, n_bus: int, low_exam: float, high_exam: float) -> dict:
    v = bus_voltage(case_dir, n_bus)
    return {"voltage": voltage_audit(v, tier, low_exam, high_exam), "thermal": thermal_audit(case_dir, tier),
            "slack": slack_info(case_dir)}


def label_gate(after: dict, source: dict, tier: str, task: str, ref: Optional[dict] = None,
               level: str = "strict") -> Tuple[bool, List[str]]:
    """合格闸:修后 after 的 full_audit 对照。返回 (通过, 未通过原因列表)。
    level="strict"(缺省,合格口径):考卷带 + 自带带全在带内(3120 放宽:BVR≤0.1% 且越限母线数 ≤ 源方式)
                                 + 不新增过载支路(过载集合 ⊆ 修前)+ 有功平衡机相对判据;
    level="exam"(次级,须声明):考卷带 + 自带带越限母线数不多于修前(病态)+ 过载支路数不多于修前 + 平衡机相对判据。
      动因(2026-08-22 试产):118 负荷增长病态里 345 kV 输电母线 38 有功再分配抬不到 0.94,strict 在该档有功任务几乎不可达;
      3120 病态过载 ~260 条,任何再调度都会有个别支路新跨 100%,按集合判"新增"必败,按条数判才有意义。
    "修前"参照 ref = 病态(缺省 ref=source,逆扰动修法=源身份时二者相同)。"""
    cfg = TIERS[tier]
    ref = ref if ref is not None else source
    why: List[str] = []
    va, vs, vr = after["voltage"], source["voltage"], ref["voltage"]
    if not va["in_exam"]:
        why.append("exam_band")
    if level == "strict":
        if cfg["relaxed"]:
            if va["bvr_own"] > 0.001 + 1e-9:
                why.append("bvr_own>0.1%")
            if va["n_viol_own"] > vs["n_viol_own"]:
                why.append("new_own_viol")
        elif va["n_viol_own"] > 0:
            why.append("own_band")
    else:
        if va["n_viol_own"] > vr["n_viol_own"]:
            why.append("more_own_viol")
    ta, ts = after["thermal"], ref["thermal"]
    if ta["available"]:
        if level == "strict":
            new_over = set(ta["over_ids"]) - set(ts["over_ids"])
            if new_over:
                why.append("new_overload:%d" % len(new_over))
        elif (ta["n_over"] or 0) > (ts["n_over"] or 0):
            why.append("more_overload")
    if task == "active":
        pl = slack_plausible(ref["slack"], after["slack"])
        if pl is False:
            why.append("slack_farther")
    return (not why), why


# ---------------------------------------------------------------- 方式池
def exam_source_sids(tier: str) -> Set[str]:
    out: Set[str] = set()
    for lib in EXAM_LIBS.get(tier, []):
        p = LINE / "libs" / lib / "eval_cases.jsonl"
        if p.is_file():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    out.add(json.loads(ln)["sid"])
    return out


def qualified_modes(tier: str, exclude_exam: bool = True) -> List[dict]:
    """按审计 JSON 取合格方式(118/2000:自带带全在带内且无过载(有额定值档);3120:放宽,BVR≤0.1% 即纳入,须声明底图带过载)。"""
    d = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    rows = d["tiers"][tier]["rows"]
    ex = exam_source_sids(tier) if exclude_exam else set()
    out = []
    for r in rows:
        if r["sid"] in ex:
            continue
        if TIERS[tier]["relaxed"]:
            ok = r["bvr_own"] <= 0.001 + 1e-9
        else:
            ok = r["n_viol_own"] == 0
            if TIERS[tier].get("pool_thermal_zero", True):   # 缺省 True = 既有各档行为逐位不变
                ok = ok and ((r.get("thermal_n_over") or 0) == 0)
        if ok:
            out.append(r)
    return out


# ---------------------------------------------------------------- 卡片小工具
def read_gens(l5: Card):
    slack, gens = None, {}
    for i, r in enumerate(l5.records):
        g = GenView(r, i)
        if g.mark != 1:
            continue
        if g.type_code == 0:
            slack = g.bs_no
            continue
        gens[i] = {"bus": g.bs_no, "p": g.pg, "pmin": g.pmin, "pmax": g.pmax, "v0": g.v0, "view": g}
    return slack, gens


def read_loads(l6: Card) -> Dict[int, dict]:
    out = {}
    for i, r in enumerate(l6.records):
        lv = LoadView(r, i)
        if lv.mark == 1:
            out[i] = {"bus": lv.bs_no, "pl": lv.pl, "ql": lv.ql, "view": lv}
    return out


def span_of(g: dict) -> float:
    return max(g["pmax"] - g["pmin"], abs(g["p"]), 1e-3)


def eff_bounds(pmin: float, pmax: float, cur: float):
    if pmax > pmin + 1e-9:
        return pmin, pmax
    return min(0.0, cur), float("inf")


def bus_adjacency(case: PsaspCase) -> Dict[int, Set[int]]:
    """交流线 + 变压器(mark==1,非自环)的母线邻接。"""
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


def clear_outputs(case_dir: Path) -> None:
    for n in PF_OUTPUT_FILES:
        p = case_dir / n
        if p.is_file():
            p.unlink()


def sha256_files(cdir: Path, names=INPUT_CARDS) -> str:
    h = hashlib.sha256()
    for n in names:
        p = cdir / n
        if p.is_file():
            h.update(n.encode("utf-8")); h.update(b"\x00"); h.update(p.read_bytes()); h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------- 侧通道特征(纯输入函数;D13 计算型)
def active_side_feats(raw: GridSample, vio_low: float = 0.93, pu_scale: float = 10.0) -> np.ndarray:
    """有功 v0.7 侧通道 [N,5] = [低压深度, 机组上行余量比, 机组下行余量比, 平衡机上行裕度(pu/10, 全图广播), 平衡机下行裕度(pu/10, 广播)]。
    全部只读档案给定列(v/pmax/pmin/pg/node_type/has_gen/is_null),不读任何标签字段。"""
    off = NODE_TABLE.offset_of
    nf = raw.node_feat.astype(np.float64)
    v = nf[:, off("v")]
    null = nf[:, off("is_null")] > 0.5
    has_gen = (nf[:, off("has_gen")] > 0.5) & ~null
    dead = (v <= 0.01) | null
    depth = np.maximum(0.0, vio_low - v)
    depth[dead] = 0.0
    pmax, pmin, pg = nf[:, off("pmax")], nf[:, off("pmin")], nf[:, off("pg")]
    span = np.maximum(np.maximum(pmax - pmin, np.abs(pg)), 1e-3)
    limited = pmax > pmin + 1e-9
    up = np.where(limited, (pmax - pg) / span, 1.0)
    dn = np.where(limited, (pg - pmin) / span, 1.0)
    up = np.clip(up, -1.0, 1.0)
    dn = np.clip(dn, -1.0, 1.0)
    up[~has_gen] = 0.0
    dn[~has_gen] = 0.0
    slack = (nf[:, off("node_type") + 2] > 0.5) & has_gen
    s_up = s_dn = 0.0
    if slack.any():
        i = int(np.where(slack)[0][0])
        if limited[i]:
            s_up = (pmax[i] - pg[i]) / pu_scale
            s_dn = (pg[i] - pmin[i]) / pu_scale
    n = raw.n_nodes
    return np.stack([depth, up, dn, np.full(n, s_up), np.full(n, s_dn)], axis=1).astype(np.float32)


def slack_headroom_from_raw(raw: GridSample) -> Tuple[float, float]:
    """(上行裕度 pu, 下行裕度 pu);无平衡机/未设限 → (inf, inf)。"""
    off = NODE_TABLE.offset_of
    nf = raw.node_feat.astype(np.float64)
    null = nf[:, off("is_null")] > 0.5
    has_gen = (nf[:, off("has_gen")] > 0.5) & ~null
    slack = (nf[:, off("node_type") + 2] > 0.5) & has_gen
    if not slack.any():
        return float("inf"), float("inf")
    i = int(np.where(slack)[0][0])
    pmax, pmin, pg = nf[i, off("pmax")], nf[i, off("pmin")], nf[i, off("pg")]
    if pmax <= pmin + 1e-9:
        return float("inf"), float("inf")
    return float(pmax - pg), float(pg - pmin)


def project_action_to_slack(action: Dict[int, float], up_headroom: float, down_headroom: float,
                            tol: float = SLACK_TOL) -> Tuple[Dict[int, float], float]:
    """平衡机裕度投影:机组净增 net=Σ ΔP 由平衡机反向吸收(ΔP_slack≈−net)。若 net 超出 [−up, down] 带(含容差),等比例缩放动作。
    返回 (缩放后动作, 缩放系数)。对已越限的平衡机(裕度为负)只允许"往回走"的方向。"""
    net = float(sum(action.values()))
    if not action or abs(net) < 1e-9:
        return dict(action), 1.0
    lo, hi = -(max(up_headroom, 0.0) + tol), (max(down_headroom, 0.0) + tol)
    if up_headroom < 0 and net < 0:       # 平衡机已超 Pmax:机组再减出力只会更糟
        lo = 0.0
    if down_headroom < 0 and net > 0:     # 平衡机已低于 Pmin:机组再加出力更糟
        hi = 0.0
    if lo <= net <= hi:
        return dict(action), 1.0
    target = hi if net > hi else lo
    s = max(0.0, target / net)
    return {b: d * s for b, d in action.items()}, s


def violation_proximity(raw: GridSample, low: float = 0.93, high: Optional[float] = None,
                        scale: float = 3.0, eps: float = EPS) -> np.ndarray:
    """逐节点"到最近越限母线的图跳数"→ exp(−d/scale) [N] float32(越限母线自身 1,远处 → 0;无越限/不连通 → 0)。
    纯输入函数(电压列 + 拓扑),D13 计算型侧通道;是老师"就近机组"启发式的特征化。BFS 多源,70k 节点亦可。"""
    off = NODE_TABLE.offset_of
    nf = raw.node_feat
    v = nf[:, off("v")].astype(np.float64)
    null = nf[:, off("is_null")] > 0.5
    live = (v > 0.01) & ~null
    viol = live & (v < low - eps)
    if high is not None:
        viol |= live & (v > high + eps)
    n = raw.n_nodes
    out = np.zeros(n, dtype=np.float32)
    if not viol.any() or raw.edge_index.size == 0:
        return out
    ei = raw.edge_index
    src = np.concatenate([ei[0], ei[1]])
    dst = np.concatenate([ei[1], ei[0]])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    indptr = np.searchsorted(src, np.arange(n + 1))
    dist = np.full(n, -1, dtype=np.int64)
    frontier = np.where(viol)[0]
    dist[frontier] = 0
    d = 0
    while frontier.size:
        d += 1
        nxt = []
        for u in frontier:
            nb = dst[indptr[u]:indptr[u + 1]]
            nb = nb[dist[nb] < 0]
            if nb.size:
                dist[nb] = d
                nxt.append(nb)
        frontier = np.unique(np.concatenate(nxt)) if nxt else np.zeros(0, dtype=np.int64)
    ok = dist >= 0
    out[ok] = np.exp(-dist[ok] / scale).astype(np.float32)
    return out


# ---------------------------------------------------------------- 动态列填法开关(2026-08-22 用户裁决:数据级规则,不动编码器)
DYN_LEGACY_COLS = ("tj", "xdp", "sh", "ph")
DYN_FILL_MODES = ("zero", "mean", "mean_all")


def apply_dyn_fill(normed: GridSample, raw: GridSample, mode: str = "zero") -> int:
    """归一化之后的档案上,把"动态组缺失"节点的四列 legacy 发电机动态参数改填归一化均值(z=0)。

    背景(结果档 §4.6):转换算例 ST.S5 为空 → build_sample 填 0 并打 DYNAMIC 组 missing 位,但 missing 位不进编码器,
    编码器看到的是 0 归一后的常量伪值 tj z≈−6.46 / xdp −4.79 / sh −8.39 / ph −1.45(lib118 真值 z 在 ±1 内)。
    mode:
      zero     = 现状不动(与部署包 run_adjust 同口径);
      mean     = 只对机组节点(has_gen & ~is_null)且动态组缺失者填 z=0(= 归一化均值 = "平均机组")。
                 非机组节点在全部预训练家族里本就是 0+missing → z≈−6.5,编码器见惯,不动;
      mean_all = 所有动态组缺失节点都填 z=0(编码器没见过"非机组节点 z=0",只作对照项)。
    返回改动的节点数。raw 只读(取 missing 位与 has_gen);normed 原地改。"""
    if mode in (None, "zero"):
        return 0
    if mode not in DYN_FILL_MODES:
        raise ValueError("dyn_fill 只能是 %s" % (DYN_FILL_MODES,))
    from gridfm.schema.fields import Group
    from gridfm.schema.sample import group_bit_index
    miss = raw.node_missing[:, group_bit_index(Group.DYNAMIC)].astype(bool)
    if mode == "mean":
        off = NODE_TABLE.offset_of
        gm = (raw.node_feat[:, off("has_gen")] > 0.5) & ~(raw.node_feat[:, off("is_null")] > 0.5)
        rows = miss & gm
    else:
        rows = miss
    if not rows.any():
        return 0
    cols = [NODE_TABLE.offset_of(c) for c in DYN_LEGACY_COLS]
    for c in cols:
        normed.node_feat[rows, c] = 0.0
    return int(rows.sum())


# ---------------------------------------------------------------- 模型引导的阶梯(纯函数;PSASP 调用在跑分器里)
def ladder_rungs(ks, mags, skip: Optional[Tuple[int, float]] = None, k_cap: Optional[int] = None) -> List[Tuple[int, float]]:
    """阶梯档位列表,老师同款顺序(台数外层、量级内层):[(k, mag), ...]。
    k_cap = 候选机组数上限(k 截到它,重复档位去重);skip = 与模型首步相同的档位(首步已试过,不重复)。"""
    out: List[Tuple[int, float]] = []
    for k in ks:
        kk = min(int(k), int(k_cap)) if k_cap else int(k)
        if kk <= 0:
            continue
        for mag in mags:
            r = (kk, float(mag))
            if skip is not None and r[0] == int(skip[0]) and abs(r[1] - float(skip[1])) < 1e-12:
                continue
            if r in out:
                continue
            out.append(r)
    return out


def rank_by_prob(prob: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """候选节点行按指针概率降序(稳定排序,同分按行号)。"""
    rows = np.where(cand)[0]
    return rows[np.argsort(-prob[rows], kind="stable")]


def reactive_ladder_action(order: np.ndarray, k: int, step: float, sign: float) -> dict:
    """无功阶梯动作:排序前 k 台机 v_set 各 ±step(低压 +、过压 −)。母线号 = 行 + 1。"""
    return {"dvset": {int(r) + 1: round(float(sign) * float(step), 4) for r in order[:int(k)]}, "shunt": []}


def active_ladder_action(order: np.ndarray, k: int, amount: float, up_room: np.ndarray,
                         dn_room: Optional[np.ndarray] = None, zero_net: bool = False) -> Dict[int, float]:
    """有功阶梯动作(老师 teacher_active 同款分摊,只换排序):排序前 k 台有上行余量的机组按余量比例分摊 amount(封顶 Σ余量);
    zero_net(平衡机已低于 Pmin)时另取排序末端 max(4,k) 台有下行余量的机组等量减出力(零和对调)。母线号 = 行 + 1。"""
    sel = [int(r) for r in order if up_room[r] > 1e-3][:int(k)]
    if not sel:
        return {}
    cap = float(sum(up_room[r] for r in sel))
    amt = min(float(amount), cap)
    alloc = {r + 1: amt * float(up_room[r]) / cap for r in sel}
    if zero_net and dn_room is not None:
        chosen = set(sel)
        far = [int(r) for r in order[::-1] if int(r) not in chosen and dn_room[r] > 1e-3][:max(4, int(k))]
        dcap = float(sum(dn_room[r] for r in far))
        if dcap > 1e-6:
            give = min(amt, dcap)
            for r in far:
                alloc[r + 1] = -give * float(dn_room[r]) / dcap
    return {b: round(v, 4) for b, v in alloc.items() if abs(v) > 1e-9}


def gen_rooms_from_raw(raw: GridSample) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(上行余量 pmax−pg, 下行余量 pg−pmin, 可调位 = 机组 & pmax>pmin)[N];固定出力机组(pmax<=pmin)余量记 0、可调位 False。"""
    off = NODE_TABLE.offset_of
    nf = raw.node_feat.astype(np.float64)
    null = nf[:, off("is_null")] > 0.5
    gm = (nf[:, off("has_gen")] > 0.5) & ~null
    pmax, pmin, pg = nf[:, off("pmax")], nf[:, off("pmin")], nf[:, off("pg")]
    limited = gm & (pmax > pmin + 1e-9)
    up = np.where(limited, pmax - pg, 0.0)
    dn = np.where(limited, pg - pmin, 0.0)
    return up, dn, limited


# ---------------------------------------------------------------- v6e 档位头(2026-08-24 A 项)
# 档位表与 ladder_rungs 同序(台数外层、量级内层);类 0 = 纯并补/无法解析,1..12 = 档位。
RUNG_KS = (1, 2, 4, 8, 16, 32)
RUNG_MAGS = (0.02, 0.05)
RUNG_TABLE = tuple((k, m) for k in RUNG_KS for m in RUNG_MAGS)   # 12 档


def rung_index(source: str) -> int:
    """老师标签 canonical_fix.source('vset_near{k}_{mag}')→ 档位类 1..12;纯并补/其他机制 → 0。
    k 不在标准档时就近向上取档(保守:宁大勿小,与阶梯"由小到大试"的语义一致)。"""
    if not source or not source.startswith("vset_near"):
        return 0
    try:
        body = source[len("vset_near"):]
        k_s, mag_s = body.split("_", 1)
        k, mag = int(k_s), float(mag_s)
    except (ValueError, IndexError):
        return 0
    kk = next((x for x in RUNG_KS if x >= k), RUNG_KS[-1])
    mm = min(RUNG_MAGS, key=lambda m: abs(m - mag))
    return 1 + RUNG_TABLE.index((kk, mm))


def clip_dvset_to_cap(action: dict, raw: GridSample, cap: float) -> dict:
    """v_set 上限守门(与部署包 --vset-cap 同语义,跑分器侧实现):正向 Δv_set 裁到 cap − v_set_now,
    裁后 ≤1e-6 的项丢弃;负向不动。只改 dvset,shunt 原样。"""
    dv = action.get("dvset") or {}
    if not dv:
        return action
    nf = raw.node_feat
    off = NODE_TABLE.offset_of
    vset_now = nf[:, off("v_set")]
    out = {}
    for bus, d in dv.items():
        if d > 0:
            room = float(cap) - float(vset_now[int(bus) - 1])
            d = min(float(d), max(room, 0.0))
            if d <= 1e-6:
                continue
        out[int(bus)] = round(float(d), 4)
    return {"dvset": out, "shunt": list(action.get("shunt") or [])}
