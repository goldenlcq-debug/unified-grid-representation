# -*- coding: utf-8 -*-
r"""调整器扩展层(2026-08-23 合入;来源 = 大算例样本线 scripts/adj2_common.py / adj2_train.py / adj2_eval.py 已验收代码)。

内容(全部纯函数或小模型类,不改 gridfm 包;PSASP 调用仍只在 run_adjust.py 里):
  1. 固定出力机组判定(MATPOWER 口径 PMAX==PMIN 不可调;CEPRI36 类"全库 pmax=pmin=0 = 未设限"保留遗留约定);
  2. 侧通道特征:有功 v0.7 五维(低压深度、机组上/下行余量比、平衡机上/下行裕度)、越限邻近度(v6d);
  3. 平衡机裕度投影(有功动作净增量不超出平衡机裕度带);
  4. 模型引导的阶梯(首步失败后按指针概率排序扩台数;无功 k×Δv_set、有功 k×平衡机超额分摊);
  5. 无功 v6d 推理器类(v5 三头 + 图级台数头)与按检查点 config 建推理器的工厂;
  6. 档案 sample_id = 输入卡片内容哈希(随机特征列不再随目录名变);动态列填法开关(缺省 zero = 现状);
  7. 实际落地干预(改机数 / Σ|Δ| / 推满台数 / 投切数 / 平衡机修前后距限)。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import nn

from gridfm.psasp_io import cards
from gridfm.psasp_io.cards import AclineView, GenView
from gridfm.psasp_io.lowlevel import Card
from gridfm.reason.adjust_task import AdjustReasoner, viol_depth
from gridfm.reason.reactive_task import DV_LEVELS, ReactiveShuntReasoner, build_reactive_reasoner, react_side
from gridfm.schema.fields import NODE_TABLE
from gridfm.schema.sample import GridSample

EPS_DEFAULT = 5e-4          # 裁判容差(与规则臂 _is_healthy_2s、病态判定同一把尺子;2026-08-23 用户裁决模型臂统一到此)
SLACK_TOL = 0.1             # 平衡机相对判据容差(pu)
INPUT_CARDS = ("LF.L0", "LF.L1", "LF.L2", "LF.L3", "LF.L5", "LF.L6", "LF.L7")
DYN_LEGACY_COLS = ("tj", "xdp", "sh", "ph")
DYN_FILL_MODES = ("zero", "mean", "mean_all")
LADDER_DEFAULT = {"reactive": ((1, 2, 4, 8), (0.02, 0.05)), "active": ((2, 4, 8, 16, 32), (1.0,))}
FIXED_MODES = ("auto", "fixed", "unlimited")


# ---------------------------------------------------------------- 1. 固定出力机组
def case_has_limits(gens: Dict[int, dict]) -> bool:
    """算例是否真填了出力限值(存在任何 pmax>pmin 的机组)。全库 pmax=pmin=0(CEPRI36 类)= 未设限约定。"""
    return any(g["pmax"] > g["pmin"] + 1e-9 for g in gens.values())


def is_matpower_case(case_dir) -> bool:
    """MATPOWER 转换算例(转换器在目录里写 convert_info.json)→ 限值口径 = MATPOWER:PMAX==PMIN 固定出力,含 0/0。
    在库 / 内网 PSASP 卡片(无此文件)→ 遗留约定:pmax=pmin=0 = 未设限(CEPRI36 即为混合:部分机组有限值、部分 0/0)。"""
    return (Path(case_dir) / "convert_info.json").is_file()


def fixed_unit_gis(gens: Dict[int, dict], mode: str = "auto", matpower: bool = False) -> Set[int]:
    """不可调(固定出力)机组记录序集合。
    mode=auto(缺省):MATPOWER 转换算例(matpower=True)→ pmax<=pmin 一律固定(含 0/0);
                    其他算例 → 0/0 = 未设限(遗留约定,可调),只有 pmax<=pmin 且不为 0/0 的视为固定;
    fixed = 一律 pmax<=pmin 固定;unlimited = 一律可调(= 2026-08-21 包的行为)。"""
    if mode not in FIXED_MODES:
        raise ValueError("fixed_units 只能是 %s" % (FIXED_MODES,))
    if mode == "unlimited":
        return set()
    if mode == "fixed" or matpower:
        return {gi for gi, g in gens.items() if g["pmax"] <= g["pmin"] + 1e-9}
    return {gi for gi, g in gens.items()
            if g["pmax"] <= g["pmin"] + 1e-9 and not (abs(g["pmax"]) < 1e-9 and abs(g["pmin"]) < 1e-9)}


def fixed_node_mask(raw: GridSample, mode: str = "auto", matpower: bool = False) -> np.ndarray:
    """档案节点级固定出力位 [N](has_gen 且按 mode 判固定;口径同 fixed_unit_gis)。"""
    off = NODE_TABLE.offset_of
    nf = raw.node_feat.astype(np.float64)
    null = nf[:, off("is_null")] > 0.5
    gm = (nf[:, off("has_gen")] > 0.5) & ~null
    pmax, pmin = nf[:, off("pmax")], nf[:, off("pmin")]
    if mode == "unlimited":
        return np.zeros(raw.n_nodes, dtype=bool)
    le = gm & (pmax <= pmin + 1e-9)
    if mode == "fixed" or matpower:
        return le
    return le & ~((np.abs(pmax) < 1e-9) & (np.abs(pmin) < 1e-9))


# ---------------------------------------------------------------- 2. 侧通道特征(纯输入函数)
def active_side_feats(raw: GridSample, vio_low: float = 0.93, pu_scale: float = 10.0) -> np.ndarray:
    """有功 v0.7 侧通道 [N,5] = [低压深度, 机组上行余量比, 机组下行余量比, 平衡机上行裕度(pu/10 广播), 平衡机下行裕度(广播)]。"""
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
    up = np.clip(np.where(limited, (pmax - pg) / span, 1.0), -1.0, 1.0)
    dn = np.clip(np.where(limited, (pg - pmin) / span, 1.0), -1.0, 1.0)
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


def violation_proximity(raw: GridSample, low: float = 0.93, high: Optional[float] = None,
                        scale: float = 3.0, eps: float = EPS_DEFAULT) -> np.ndarray:
    """逐节点"到最近越限母线的图跳数"→ exp(−d/scale) [N](越限母线自身 1;无越限/不连通 → 0)。多源 BFS。"""
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


def side_feats(task: str, variant: str, raw: GridSample, low: float, high: float, with_shunt: bool = True) -> np.ndarray:
    """按变体取侧通道(与训练侧 adj2_train.side_feats 同口径)。"""
    if task == "active":
        if variant in ("v06", "v06re"):
            return viol_depth(raw, low).astype(np.float32)                    # [N] 一维,推理器内 unsqueeze
        base = active_side_feats(raw, low)
        if variant == "v07d":
            base = np.concatenate([base, violation_proximity(raw, low)[:, None]], axis=1)
        return base.astype(np.float32)
    base = react_side(raw, low, high, with_shunt=with_shunt)
    if variant in ("v6d",):
        base = np.concatenate([base, violation_proximity(raw, low, high)[:, None]], axis=1)
    return base.astype(np.float32)


# ---------------------------------------------------------------- 3. 平衡机裕度
def slack_headroom_from_raw(raw: GridSample) -> Tuple[float, float]:
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
    """机组净增 net=Σ ΔP 由平衡机反向吸收;net 超出 [−up, down] 带(含容差)则等比例缩放;已越限的平衡机只许往回走。"""
    net = float(sum(action.values()))
    if not action or abs(net) < 1e-9:
        return dict(action), 1.0
    lo, hi = -(max(up_headroom, 0.0) + tol), (max(down_headroom, 0.0) + tol)
    if up_headroom < 0 and net < 0:
        lo = 0.0
    if down_headroom < 0 and net > 0:
        hi = 0.0
    if lo <= net <= hi:
        return dict(action), 1.0
    target = hi if net > hi else lo
    s = max(0.0, target / net)
    return {b: d * s for b, d in action.items()}, s


def slack_info(case_dir: Path) -> dict:
    """平衡机:母线、[Pmin,Pmax]、LP5 解得 Pg、到带距离(带内 0)、上/下行裕度。须在该目录已有潮流结果(LF.LP5)时调用。"""
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


# ---------------------------------------------------------------- 4. 模型引导的阶梯(纯函数)
def ladder_rungs(ks, mags, first: Optional[Tuple[int, float]] = None, k_cap: Optional[int] = None,
                 skip_dominated: bool = True) -> List[Tuple[int, float]]:
    """阶梯档位(台数外层、量级内层 = 老师同款顺序)。first = 首步若是"k 台同幅度"则 (k, 幅度):
    跳过与首步相同的档,以及被首步支配的档(台数 ≤ 首步台数且幅度 ≤ 首步幅度:同排序下是首步的子集、量又不比它大,
    必然也修不好;2026-08-23 验收实跑里这些档全部白费 1 次 PSASP)。k_cap = 候选机组数。"""
    out: List[Tuple[int, float]] = []
    for k in ks:
        kk = min(int(k), int(k_cap)) if k_cap else int(k)
        if kk <= 0:
            continue
        for mag in mags:
            r = (kk, float(mag))
            if first is not None:
                if r[0] == int(first[0]) and abs(r[1] - float(first[1])) < 1e-12:
                    continue
                if skip_dominated and r[0] <= int(first[0]) and r[1] <= float(first[1]) + 1e-12:
                    continue
            if r in out:
                continue
            out.append(r)
    return out


def rank_by_prob(prob: np.ndarray, cand: np.ndarray) -> np.ndarray:
    rows = np.where(cand)[0]
    return rows[np.argsort(-prob[rows], kind="stable")]


def reactive_ladder_action(order: np.ndarray, k: int, step: float, sign: float) -> dict:
    return {"dvset": {int(r) + 1: round(float(sign) * float(step), 4) for r in order[:int(k)]}, "shunt": []}


def gen_rooms_from_raw(raw: GridSample, fixed_mode: str = "auto", matpower: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(上行余量 pmax−pg, 下行余量 pg−pmin, 可调位)[N];固定出力机组余量 0、可调位 False;未设限机组余量取 max(|pg|,1)。"""
    off = NODE_TABLE.offset_of
    nf = raw.node_feat.astype(np.float64)
    null = nf[:, off("is_null")] > 0.5
    gm = (nf[:, off("has_gen")] > 0.5) & ~null
    pmax, pmin, pg = nf[:, off("pmax")], nf[:, off("pmin")], nf[:, off("pg")]
    fixed = fixed_node_mask(raw, fixed_mode, matpower)
    adj = gm & ~fixed
    limited = adj & (pmax > pmin + 1e-9)
    up = np.where(limited, pmax - pg, np.where(adj, np.maximum(np.abs(pg), 1.0), 0.0))
    dn = np.where(limited, pg - pmin, np.where(adj, np.maximum(pg, 0.0), 0.0))
    return up, dn, adj


def active_ladder_action(order: np.ndarray, k: int, amount: float, up_room: np.ndarray,
                         dn_room: Optional[np.ndarray] = None, zero_net: bool = False) -> Dict[int, float]:
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


# ---------------------------------------------------------------- 5. 推理器
class ReactiveCountReasoner(ReactiveShuntReasoner):
    """v6d:v5 三头(指针 / 4 档 Δv_set 分类 / 并补指针)+ 图级台数头 head_k(0..K_MAX 分类;0 = 纯并补)。"""

    def __init__(self, d, n_queries=16, n_layers=2, n_heads=4, ffn_mult=2, side_dim=4, n_dv_levels=4, n_k_max=12):
        super().__init__(d=d, n_queries=n_queries, n_layers=n_layers, n_heads=n_heads, ffn_mult=ffn_mult,
                         side_dim=side_dim, n_dv_levels=n_dv_levels)
        self.n_k_max = int(n_k_max)
        self.head_k = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, self.n_k_max + 1))

    def forward(self, h, g, e, node_batch, edge_batch, depth, n_graphs=None):
        feat = self.node_feat(h, g, e, node_batch, edge_batch, depth)
        dv = self.head_dp(feat)
        nb = node_batch
        B = int(n_graphs) if n_graphs is not None else (int(nb.max()) + 1 if nb.numel() else 0)
        f32 = feat.float()
        pooled = torch.zeros(B, f32.shape[-1], dtype=torch.float32, device=feat.device)
        pooled.index_add_(0, nb, f32)
        cnt = torch.zeros(B, dtype=torch.float32, device=feat.device)
        cnt.index_add_(0, nb, torch.ones_like(nb, dtype=torch.float32))
        pooled = (pooled / cnt.clamp_min(1.0).unsqueeze(-1)).to(feat.dtype)
        return (self.head_ptr(feat).squeeze(-1), dv, self.head_shunt(feat).squeeze(-1), self.head_k(pooled))


def build_reasoner(task: str, rcfg: dict):
    """按检查点 config 建推理器;返回 (模型, 变体名)。v06 / v5 = 2026-08-21 包的两个检查点(config 无 variant 字段)。"""
    variant = rcfg.get("variant") or ("v06" if task == "active" else "v5")
    d, q, L = rcfg["d"], rcfg["queries"], rcfg["layers"]
    if task == "active":
        if variant in ("v06", "v06re", "v07"):
            return AdjustReasoner(d=d, n_queries=q, n_layers=L, side_dim=int(rcfg.get("side_dim", 1))), variant
        raise ValueError("有功变体 %s 未在部署包实现(v07d 台数头版未交付)" % variant)
    n_dv = int(rcfg.get("n_dv_levels", 0))
    if variant == "v5":
        return build_reactive_reasoner(d=d, n_queries=q, n_layers=L, with_shunt=bool(rcfg.get("with_shunt", False)),
                                       n_dv_levels=n_dv), variant
    if variant == "v5re":
        return ReactiveShuntReasoner(d=d, n_queries=q, n_layers=L, side_dim=int(rcfg.get("side_dim", 3)), n_dv_levels=n_dv), variant
    if variant in ("v6", "v6d"):
        return ReactiveCountReasoner(d=d, n_queries=q, n_layers=L, side_dim=int(rcfg.get("side_dim", 4)), n_dv_levels=n_dv,
                                     n_k_max=int(rcfg.get("k_max", 12))), variant
    raise ValueError("无功变体 %s 未在部署包实现" % variant)


# ---------------------------------------------------------------- 6. sample_id / 动态列填法
def sha256_cards(case_dir: Path, names=INPUT_CARDS) -> str:
    h = hashlib.sha256()
    for n in names:
        p = case_dir / n
        if p.is_file():
            h.update(n.encode("utf-8")); h.update(b"\x00"); h.update(p.read_bytes()); h.update(b"\n")
    return h.hexdigest()


def apply_dyn_fill(normed: GridSample, raw: GridSample, mode: str = "zero") -> int:
    """归一化之后,把动态组缺失节点的四列 legacy 动态参数改填归一化均值(z=0)。zero = 不动(缺省,与 2026-08-21 包同口径)。"""
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
    for c in (NODE_TABLE.offset_of(c) for c in DYN_LEGACY_COLS):
        normed.node_feat[rows, c] = 0.0
    return int(rows.sum())


# ---------------------------------------------------------------- 7. 实际落地干预
def _gens_of(l5: Card):
    out = {}
    for i, r in enumerate(l5.records):
        g = GenView(r, i)
        if g.mark == 1 and g.type_code != 0:
            out[i] = (g.bs_no, g.pg, g.pmin, g.pmax, g.v0)
    return out


def _shunt_marks(l2: Card):
    out = {}
    for i, r in enumerate(l2.records):
        a = AclineView(r, i)
        if a.i_no == a.j_no:
            out[i] = a.mark
    return out


def realized(work: Path, bak: Path, task: str, slack_before: Optional[dict] = None, fixed_mode: str = "auto") -> dict:
    """病态起点(bak)与修后(work)卡片逐台对比得实际干预;平衡机修前距限由调用方传入(初判后的 LP5),修后从 work/LF.LP5 读。"""
    g0, g1 = _gens_of(Card.load(bak / "LF.L5")), _gens_of(Card.load(work / "LF.L5"))
    s_after = slack_info(work)
    d0 = (slack_before or {}).get("dist")
    d1 = s_after.get("dist")
    out = {"slack_dist_before": d0, "slack_dist_after": d1, "slack_p_before": (slack_before or {}).get("p_solved"),
           "slack_p_after": s_after.get("p_solved"),
           "plausible": (bool(d1 <= d0 + SLACK_TOL) if (d0 is not None and d1 is not None) else None)}
    if task == "active":
        dps = {gi: g1[gi][1] - g0[gi][1] for gi in g0 if gi in g1 and abs(g1[gi][1] - g0[gi][1]) > 1e-6}
        at_hi = sum(1 for gi in dps if g0[gi][3] > g0[gi][2] + 1e-9 and abs(g1[gi][1] - g0[gi][3]) < 1e-6)
        gens_d = {gi: {"pmax": g0[gi][3], "pmin": g0[gi][2]} for gi in g0}
        fixed = fixed_unit_gis(gens_d, "fixed")         # 报告一律按物理口径统计落在固定机组上的量(与 mode 无关,便于报警)
        onf = sum(abs(v) for gi, v in dps.items() if gi in fixed)
        tot = sum(abs(v) for v in dps.values())
        out.update({"n_changed": len(dps), "sum_abs_dp": round(tot, 3), "net_dp": round(sum(dps.values()), 3),
                    "n_pushed_to_pmax": at_hi, "sum_abs_dp_on_fixed": round(onf, 3),
                    "frac_dp_on_fixed": (round(onf / tot, 3) if tot > 1e-9 else 0.0),
                    "n_fixed_units_changed": sum(1 for gi in dps if gi in fixed)})
        return out
    dvs = {gi: g1[gi][4] - g0[gi][4] for gi in g0 if gi in g1 and abs(g1[gi][4] - g0[gi][4]) > 1e-6}
    n_sw = 0
    if (bak / "LF.L2").is_file() and (work / "LF.L2").is_file():
        s0, s1 = _shunt_marks(Card.load(bak / "LF.L2")), _shunt_marks(Card.load(work / "LF.L2"))
        n_sw = sum(1 for i in s0 if i in s1 and s0[i] != s1[i])
    out.update({"n_changed": len(dvs), "sum_abs_dv": round(sum(abs(v) for v in dvs.values()), 4),
                "max_abs_dv": round(max([abs(v) for v in dvs.values()] or [0.0]), 4), "n_shunt_switched": n_sw})
    out["plausible"] = None
    return out


def judge_ok(rf: dict, task: str, low: float, high: float, eps: float = EPS_DEFAULT) -> bool:
    """统一裁判(模型臂、阶梯、规则臂同一把尺子):收敛 且 vmin ≥ low−eps(无功再要求 vmax ≤ high+eps)。"""
    if not rf.get("converged"):
        return False
    if rf.get("vmin") is None:
        return False
    if task == "active":
        return bool(rf["vmin"] >= low - eps)
    return bool(rf["vmin"] >= low - eps and rf.get("vmax", 9.9) <= high + eps)
