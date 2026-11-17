# -*- coding: utf-8 -*-
"""解析反算基线臂(泄漏修复线任务 A;预注册 docs/预注册-泄漏修复线-解析基线与支路遮蔽分叉-20260827 §2,
含 2026-08-27 对抗审查 B1/B2/C1/C2 修订)。

在与迁移门禁完全相同的掩码流(默认 MaskConfig、seed 0+10007、同样本序)与相同评估面上,
从可见量(支路流+静态+拓扑可见位+未遮节点)解析反算被遮 STATE 列,产出技巧分三臂表的解析臂。

协议要点(逐条对预注册):
- 证书内建(B2):正式跑同一循环抓取 lm,以 ladder_baselines 逐字镜像算式复算 L0/L1/L2 12 项,
  逐位等于 p1_full_report.json 存值才准出表;
- 在跑断言(A1/B1/C3):is_null 不入评估/锚;branch_missing[STATE] 全 False;
  逐集 KCL 双侧中位 <1e-5(支路数组对位级证明不变的节点注入闭合);
- 支路类型适用域(C1):自环/shunt_comp 一律排除;xfmr/phase_shifter 按训练侧分型标定取舍
  (KEEP_V/KEEP_TH 表,冻结常数);
- 主/副口径(C2):门禁字段 v/θ 主数字 = 全评估位分母(不可判回填 L2 同款零知识值 θ→0/v→1);
  副表 = 可判子集 × 遮蔽档 × 锚定层;pg/qg 维持"不可判不填数"(任务书口径);
- 遮蔽档重建(G1):镜像 rng 平行流重建 u 抽签,与 lm 形态互验。

符号约定(裁定档 §3 实测):pi = i 端流入支路,pj = 到达 j 端;KCL 取 +pi/−pj;损耗 = pi−pj。
两端四量联立:k=|I|²=(pi−pj)/r;送端 b²u²+(2·qi·b−k)u+(pi²+qi²)=0;受端 b²w²−(2·qj·b+k)w+(pj²+qj²)=0;
根取距 1.0 近者;δij = −angle(Vi − I·(r+jx)),Vi=√u∠0,I=(pi−j(qi+b·u))/√u;
r≤EPS_R 支路走无功损耗迭代路(定点 5 步+终检自洽);x 侧 k 交叉检验(CONS_TOL)剔噪声支配支路。

用法:
  --selfcheck   训练侧标定+自检门(lib118 前 5 + synth train 抽样;含分型标定表;全过才准 --run)
  --run         正式跑 val/transfer 三臂表(内建证书+在跑断言,写 JSON 结果)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridfm.data.dataset import library_files, load_families  # noqa: E402
from gridfm.schema.fields import BRANCH_TABLE, BRANCH_TYPES, NODE_TABLE, Group  # noqa: E402
from gridfm.schema.sample import GridSample, group_bit_index  # noqa: E402
from gridfm.train.masking import MaskConfig, apply_node_mask  # noqa: E402
from gridfm.train.pretrain import _VAL_MASK_SEED_OFFSET  # noqa: E402

ROOT = Path(r"E:\AI-2\统一计算大模型\主线机器归档-20260721\gridfm-data")
V1_REPORT = Path(r"E:\AI-2\统一计算大模型\任务分包-机器B回传\C包_G线-20260717"
                 r"\grid-foundation\runs\encoder_v1\p1_full_report.json")
SEED = 0                       # 与验收 cfg.seed 同

# ---- 标定常数(训练侧标定后冻结;触 val/transfer 前不得再改;预注册 §2.3 全清单)----
EPS_R = 1e-5                   # r ≤ 此值走无功损耗迭代路
EPS_K = 1e-9                   # |I|² 下限,更小视为零流支路 → 弃用
CONS_TOL = 0.05                # k_r 与 k_x 相对不一致上限(四量联立自洽滤波)
EPS_X = 1e-8                   # |x| ≤ 此值无法做 x 侧检验 → 弃用
RIDGE = 1e-8                   # θ 差场 LS 的岭(与 transfer.py L2 同量级)
QUAD_DEGEN = 1e-30             # 二次系数退化阈(b²≈0 → 线性式)
FIX_ITERS = 5                  # r≈0 迭代路定点步数
# 分型取舍表(审查 C1;训练侧分型标定后冻结——见 --selfcheck 输出与预注册 §2.3):
# 规则:|V| 端点误差中位 <1e-3 才入 KEEP_V;δ 误差中位 <1e-4 才入 KEEP_TH。
# 自环与 shunt_comp 无条件排除(π 模型无意义);标定实测(2026-08-27,13 样本,--selfcheck):
#   acline    候选 2460/可用 1897:δ 1.89e-5 / |V| 2.80e-4 → 双入
#   xfmr_2w   候选   45/可用   45:δ 1.32e-6(变比为实数,不移相 → θ 入) /
#             |V| 7.47e-3(变比缺列的系统偏差,自洽滤波不可见 → |V| 弃,审查 C1 命中)
#   series_comp / phase_shifter / xfmr_3w:标定面 0 支路 → 保守排除(遇到即弃用,如实计数)
KEEP_V = frozenset({"acline"})
KEEP_TH = frozenset({"acline", "xfmr_2w"})
KCL_GATE = 1e-5                # 在跑一致性门:逐集 KCL 双侧中位阈(审查 B1)

OFF = NODE_TABLE.offset_of
BOFF = BRANCH_TABLE.offset_of
SBIT = group_bit_index(Group.STATE)
TBIT = group_bit_index(Group.TOPOLOGY)
LADDER_FALLBACK = {"theta": 0.0, "v": 1.0}     # 与 transfer.py 同源的零知识常数


# ---------------- 逐支路两端四量联立 ----------------

def _solve_quad_near1(a, b, c):
    """解 a·t² + b·t + c = 0,取正根中距 1.0 近者;无实正根返回 nan。"""
    if abs(a) < QUAD_DEGEN:
        if abs(b) < QUAD_DEGEN:
            return float("nan")
        t = -c / b
        return t if t > 0 else float("nan")
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return float("nan")
    sq = np.sqrt(disc)
    roots = [t for t in ((-b + sq) / (2 * a), (-b - sq) / (2 * a)) if t > 0]
    if not roots:
        return float("nan")
    return min(roots, key=lambda t: abs(t - 1.0))


def solve_branch(pi, qi, pj, qj, r, x, b):
    """两端四量联立。返回 (u, w, delta, ok)。u=|Vi|², w=|Vj|², delta=θi−θj。

    超定自洽滤波:r 路解出 (u,w) 后用 k_x=(qA−qB)/x 与 k_r 交叉检验;
    迭代路收敛后检幅值方程相对残差。不一致 > CONS_TOL 判不可靠弃用。
    """
    if abs(x) <= EPS_X:
        return np.nan, np.nan, np.nan, False
    if r > EPS_R:
        k = (pi - pj) / r
        if k <= EPS_K:
            return np.nan, np.nan, np.nan, False
        u = _solve_quad_near1(b * b, 2.0 * qi * b - k, pi * pi + qi * qi)
        w = _solve_quad_near1(b * b, -(2.0 * qj * b + k), pj * pj + qj * qj)
        if not (np.isfinite(u) and np.isfinite(w)) or u <= 0 or w <= 0:
            return np.nan, np.nan, np.nan, False
        k_x = ((qi + b * u) - (qj - b * w)) / x
        if k_x <= 0 or abs(k - k_x) / max(k, k_x) > CONS_TOL:
            return np.nan, np.nan, np.nan, False
    else:
        u = w = 1.0
        k = np.nan
        for _ in range(FIX_ITERS):
            k = ((qi + b * u) - (qj - b * w)) / x
            if not np.isfinite(k) or k <= EPS_K:
                return np.nan, np.nan, np.nan, False
            u = (pi * pi + (qi + b * u) ** 2) / k
            w = (pj * pj + (qj - b * w) ** 2) / k
            if not (np.isfinite(u) and np.isfinite(w)) or u <= 0 or w <= 0:
                return np.nan, np.nan, np.nan, False
        res_u = abs((pi * pi + (qi + b * u) ** 2) / k - u) / max(u, 1e-12)
        res_w = abs((pj * pj + (qj - b * w) ** 2) / k - w) / max(w, 1e-12)
        if max(res_u, res_w) > CONS_TOL:
            return np.nan, np.nan, np.nan, False
    if not (np.isfinite(u) and np.isfinite(w)) or u <= 0 or w <= 0:
        return np.nan, np.nan, np.nan, False
    vi = np.sqrt(u)
    I = (pi - 1j * (qi + b * u)) / vi
    vj_ph = vi - I * (r + 1j * x)
    delta = -np.angle(vj_ph)
    return float(u), float(w), float(delta), True


def branch_estimates(s: GridSample):
    """对样本内全部候选支路解 (u,w,δ)。返回 [(a,c,u,w,delta,btype), ...](联立成功者)。

    候选 = mark>0.5 且非自环 且两端非 null 且 btype≠shunt_comp。类型取舍由消费方按
    KEEP_V/KEEP_TH 过滤(标定期需要全类型测量,故此处不滤 KEEP)。
    """
    bf = s.branch_feat
    ei = s.edge_index
    null = s.node_feat[:, OFF("is_null")] > 0.5
    mark = bf[:, BOFF("mark")] > 0.5
    btypes = np.argmax(bf[:, 0:6], axis=1)
    out = []
    for b_idx in range(s.n_branches):
        if not mark[b_idx]:
            continue
        a, c = int(ei[0][b_idx]), int(ei[1][b_idx])
        bt = BRANCH_TYPES[int(btypes[b_idx])]
        if a == c or null[a] or null[c] or bt == "shunt_comp":
            continue
        u, w, d, ok = solve_branch(
            float(bf[b_idx, BOFF("pi")]), float(bf[b_idx, BOFF("qi")]),
            float(bf[b_idx, BOFF("pj")]), float(bf[b_idx, BOFF("qj")]),
            float(bf[b_idx, BOFF("r")]), float(bf[b_idx, BOFF("x")]),
            float(bf[b_idx, BOFF("b_half")]))
        if ok:
            out.append((a, c, u, w, d, bt))
    return out


# ---------------- 节点级反算 ----------------

def kcl_net(s: GridSample):
    """P_net = pg−pl、Q_net = qg−ql 的 KCL 重建(全部 mark>0.5 支路,含自环 shunt)。"""
    bf, ei = s.branch_feat, s.edge_index
    mark = bf[:, BOFF("mark")] > 0.5
    p = np.zeros(s.n_nodes)
    q = np.zeros(s.n_nodes)
    np.add.at(p, ei[0][mark], bf[mark, BOFF("pi")])
    np.add.at(p, ei[1][mark], -bf[mark, BOFF("pj")])
    np.add.at(q, ei[0][mark], bf[mark, BOFF("qi")])
    np.add.at(q, ei[1][mark], -bf[mark, BOFF("qj")])
    return p, q


def theta_field(s: GridSample, ests, anchor_mask, theta_anchor_vals):
    """θ 差场加权 LS(权=1)。返回 (theta_pred[N], anchored[N], covered[N])。

    图 = KEEP_TH 类型的可用支路诱导图;连通片按该图;STATE-missing 节点留作自由变量;
    无锚片解出后片内预测均值置 0(零知识约定,预注册 §2.2-4)。
    """
    n = s.n_nodes
    ests = [e for e in ests if e[5] in KEEP_TH]
    if not ests:
        return np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
    ia = np.array([e[0] for e in ests])
    ib = np.array([e[1] for e in ests])
    dd = np.array([e[4] for e in ests])

    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(ia, ib):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comp = np.array([find(i) for i in range(n)])
    covered = np.zeros(n, bool)
    covered[ia] = True
    covered[ib] = True

    ones = np.ones(len(ia))
    A = sp.coo_matrix((ones, (ia, ib)), shape=(n, n))
    A = A + A.T
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    div = np.zeros(n)
    np.add.at(div, ia, dd)
    np.add.at(div, ib, -dd)

    theta = np.zeros(n)
    anchored = np.zeros(n, bool)
    for cid in np.unique(comp[covered]):
        nodes = np.where((comp == cid) & covered)[0]
        if nodes.size < 2:
            continue
        anch = nodes[anchor_mask[nodes]]
        if anch.size > 0:
            unk = nodes[~anchor_mask[nodes]]
            theta[anch] = theta_anchor_vals[anch]
            if unk.size > 0:
                LII = L[unk][:, unk].tocsc() + RIDGE * sp.identity(unk.size, format="csc")
                rhs = div[unk] - L[unk][:, anch] @ theta_anchor_vals[anch]
                theta[unk] = spsolve(LII, rhs)
            anchored[nodes] = True
        else:
            unk = nodes[1:]
            theta[nodes[0]] = 0.0
            if unk.size > 0:
                LII = L[unk][:, unk].tocsc() + RIDGE * sp.identity(unk.size, format="csc")
                theta[unk] = spsolve(LII, div[unk])
            theta[nodes] -= theta[nodes].mean()
    return theta, anchored, covered


def analytic_reconstruct(s: GridSample, lm: np.ndarray):
    """对单样本反算被遮 STATE 列。可见性与阶梯基线同惯例(掩码后可见量+loss_mask 指示)。"""
    n = s.n_nodes
    nf = s.node_feat
    miss_state = s.node_missing[:, SBIT]
    miss_topo = s.node_missing[:, TBIT]
    m_theta = lm[:, OFF("theta")]
    m_topo = lm[:, OFF("has_gen")]        # TOPOLOGY 组遮蔽在列上的投影
    topo_vis = (~m_topo) & (~miss_topo)

    ests = branch_estimates(s)

    p_net, q_net = kcl_net(s)
    hg = nf[:, OFF("has_gen")] > 0.5
    hl = nf[:, OFF("has_load")] > 0.5
    pg_pred = np.zeros(n)
    pg_det = np.zeros(n, bool)
    qg_pred = np.zeros(n)
    qg_det = np.zeros(n, bool)
    pure_gen = topo_vis & hg & ~hl
    no_gen = topo_vis & ~hg
    pg_pred[pure_gen] = p_net[pure_gen]
    qg_pred[pure_gen] = q_net[pure_gen]
    pg_det[pure_gen | no_gen] = True
    qg_det[pure_gen | no_gen] = True
    coexist = topo_vis & hg & hl

    v_pred = np.full(n, np.nan)
    v_lists: list = [[] for _ in range(n)]
    for a, c, u, w, _d, bt in ests:
        if bt in KEEP_V:
            v_lists[a].append(np.sqrt(u))
            v_lists[c].append(np.sqrt(w))
    for i in range(n):
        if v_lists[i]:
            v_pred[i] = float(np.median(v_lists[i]))
    v_det = np.isfinite(v_pred)

    anchor_mask = (~m_theta) & (~miss_state)
    null = nf[:, OFF("is_null")] > 0.5
    anchor_mask &= ~null                   # 在跑断言 A1 的锚侧保证(null 不作锚)
    th_pred, th_anchored, th_cov = theta_field(s, ests, anchor_mask, nf[:, OFF("theta")])
    return {
        "pg": (pg_pred, pg_det), "qg": (qg_pred, qg_det),
        "v": (v_pred, v_det), "theta": (th_pred, th_cov),
        "theta_anchored": th_anchored, "coexist": coexist, "topo_masked_det": ~topo_vis,
    }


# ---------------- 证书:ladder_baselines 逐字镜像(消费捕获的 lm) ----------------

def mirror_ladder(samples, lms):
    """与 gridfm/eval/transfer.py::ladder_baselines 逐字镜像的算式,唯一差别 = lm 由外部捕获传入。

    正式跑用它 + 捕获 lm 复算 12 项并断言逐位等于报告存值(审查 B2:掩码流任何错位 L2 必变)。
    """
    errs = {f: {"L0": [], "L1": [], "L2": []} for f in LADDER_FALLBACK}
    pooled = {f: [] for f in LADDER_FALLBACK}
    per_sample = []
    for s, lm in zip(samples, lms):
        per_sample.append((s, lm))
        ok = ~s.node_missing[:, SBIT]
        for f in LADDER_FALLBACK:
            c = OFF(f)
            pooled[f].append(s.node_feat[ok, c].astype(np.float64))
    pool_mean = {f: np.concatenate(v).mean() if v else 0.0 for f, v in pooled.items()}

    for s, lm in per_sample:
        n = s.n_nodes
        i, j = s.edge_index
        A = sp.coo_matrix((np.ones(i.size * 2), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
        A.data[:] = 1.0
        L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
        ok = ~s.node_missing[:, SBIT]
        for f, fb in LADDER_FALLBACK.items():
            c = OFF(f)
            m = lm[:, c] & ok
            if not m.any():
                continue
            truth = s.node_feat[m, c].astype(np.float64)
            vis = ok & ~m
            vis_vals = s.node_feat[vis, c].astype(np.float64)
            errs[f]["L0"].append(np.abs(truth - pool_mean[f]))
            errs[f]["L1"].append(np.abs(truth - (vis_vals.mean() if vis_vals.size else fb)))
            x = np.full(n, fb, dtype=np.float64)
            x[vis] = s.node_feat[vis, c]
            if vis.any():
                I = np.where(m)[0]; B = np.where(vis)[0]  # noqa: E702
                LII = L[I][:, I].tocsc() + 1e-8 * sp.identity(I.size, format="csc")
                x[I] = spsolve(LII, -L[I][:, B] @ x[B])
            errs[f]["L2"].append(np.abs(x[m] - truth))
    return {f: {k: float(np.concatenate(v).mean()) if v else float("nan")
                for k, v in d.items()} for f, d in errs.items()}


# ---------------- 评估(与门禁同掩码流;内建证书+在跑断言) ----------------

def run_arm(samples, want_ladder, seed=SEED):
    """正式评估一集。返回 (结果 dict);内建证书或在跑断言不过 → 直接抛错停。"""
    rng = np.random.default_rng(seed + _VAL_MASK_SEED_OFFSET)
    rng2 = np.random.default_rng(seed + _VAL_MASK_SEED_OFFSET)   # 平行流:重建 u 抽签(G1)

    # ---- pass 1:捕获 lm + 遮蔽档 + 在跑断言 ----
    lms, strata = [], []
    kcl_p_all, kcl_q_all = [], []
    n_isolated = 0
    n_empty_state = 0
    for s in samples:
        _mf, _tgt, lm = apply_node_mask(s.node_feat, s.node_missing, MaskConfig(), rng)
        u2 = rng2.random()
        if not (u2 < 0.20):
            rng2.random((s.n_nodes, 4))   # 镜像 apply_node_mask 的 (n, ng=4) 消耗(仅高/基础档)
        stratum = "full" if u2 < 0.20 else ("high" if u2 < 0.30 else "base")
        null = s.node_feat[:, OFF("is_null")] > 0.5
        # 断言 A1:is_null 节点不入评估位
        assert not lm[null].any(), "is_null 节点出现在 loss_mask,违反预注册 §2.2-7①,停"
        # 断言 C3(按语义收窄,2026-08-27 实查):branch_missing[STATE] 仅允许出现在停运支路
        # (assemble.py:249:LP2 无流记录 → missing;实测 val/transfer 1485 条全部 mark=0 四流全零)。
        # 毒格 = 在运支路 missing(0 会被当真流),该格必须为空。
        bmark = s.branch_feat[:, BOFF("mark")] > 0.5
        assert not (s.branch_missing[:, SBIT] & bmark).any(), \
            "在运支路存在 branch_missing[STATE],0 将被当真流,停(§2.2-7②)"
        # 互验 G1:全遮档 ⇔ 全部有效节点 STATE 被遮。
        # 边格(2026-08-27 实跑发现):val 存在整样本 STATE 缺失的样本(n_ok=0,无可遮节点、
        # 评估零贡献、掩码流仍只耗 1 抽)——此类样本跳过互验、如实计数。
        oks = (~s.node_missing[:, SBIT]) & ~null
        if oks.any():
            full_masked = bool(lm[oks, OFF("theta")].all())
            assert (stratum == "full") == full_masked, \
                f"遮蔽档互验失败(u2={u2:.4f} vs lm 形态),掩码流错位,停"
        else:
            n_empty_state += 1
        # KCL 一致性门样本级积累(B1)
        p_net, q_net = kcl_net(s)
        okn = oks
        kcl_p_all.append(np.abs(p_net - (s.node_feat[:, OFF("pg")] - s.node_feat[:, OFF("pl")]))[okn])
        kcl_q_all.append(np.abs(q_net - (s.node_feat[:, OFF("qg")] - s.node_feat[:, OFF("ql")]))[okn])
        deg = np.zeros(s.n_nodes, int)
        mark = s.branch_feat[:, BOFF("mark")] > 0.5
        np.add.at(deg, s.edge_index[0][mark], 1)
        np.add.at(deg, s.edge_index[1][mark], 1)
        n_isolated += int(((deg == 0) & okn).sum())
        lms.append(lm)
        strata.append(stratum)

    kcl_p = float(np.median(np.concatenate(kcl_p_all)))
    kcl_q = float(np.median(np.concatenate(kcl_q_all)))
    assert kcl_p < KCL_GATE and kcl_q < KCL_GATE, \
        f"在跑 KCL 一致性门不过(有功 {kcl_p:.2e} / 无功 {kcl_q:.2e} ≥ {KCL_GATE}),支路数组存疑,停"

    # ---- 内建证书(B2):镜像阶梯 12 项逐位断言 ----
    got = mirror_ladder(samples, lms)
    for f in ("theta", "v"):
        for lvl in ("L0", "L1", "L2"):
            a, b = got[f][lvl], want_ladder[f][lvl]
            assert abs(a - b) <= abs(b) * 1e-12, \
                f"内建证书失败:{f}/{lvl} 复算 {a!r} ≠ 存值 {b!r},掩码流/数据错位,停"

    # ---- pass 2:解析臂 ----
    fields = ("v", "theta", "pg", "qg")
    prim_err = {f: [] for f in ("v", "theta")}            # 主口径:全评估位(零知识回填)
    sub = {}                                              # 副表:可判子集分层

    def _acc(key, arr):
        sub.setdefault(key, []).append(arr)

    cover = {f: [0, 0] for f in fields}                   # [n_det, n_pos]
    pg_coexist = pg_topomask = 0
    for s, lm, st in zip(samples, lms, strata):
        rec = analytic_reconstruct(s, lm)
        truth = s.node_feat.astype(np.float64)
        for f in fields:
            c = OFF(f)
            pos = lm[:, c]
            pred, det = rec[f]
            cover[f][0] += int((pos & det).sum())
            cover[f][1] += int(pos.sum())
            if f in ("v", "theta"):
                fill = LADDER_FALLBACK[f]
                pfull = np.where(det, pred, fill)
                prim_err[f].append(np.abs(pfull[pos] - truth[pos, c]))
                m = pos & det
                if m.any():
                    _acc((f, st, "det"), np.abs(pred[m] - truth[m, c]))
                if f == "theta":
                    for tag, sel in (("anch", rec["theta_anchored"]), ("noanch", ~rec["theta_anchored"])):
                        mm = m & sel
                        if mm.any():
                            _acc((f, st, tag), np.abs(pred[mm] - truth[mm, c]))
            else:
                m = pos & det
                if m.any():
                    _acc((f, "all", "det"), np.abs(pred[m] - truth[m, c]))
        pg_coexist += int((lm[:, OFF("pg")] & rec["coexist"]).sum())
        pg_topomask += int((lm[:, OFF("pg")] & rec["topo_masked_det"]).sum())

    def _stat(arrs):
        e = np.concatenate(arrs) if arrs else np.array([])
        return {"mae": float(e.mean()) if e.size else float("nan"),
                "median": float(np.median(e)) if e.size else float("nan"),
                "n": int(e.size)}

    out = {
        "primary_full_denominator": {f: _stat(prim_err[f]) for f in ("v", "theta")},
        "coverage": {f: {"n_det": cover[f][0], "n_pos": cover[f][1],
                         "frac": cover[f][0] / cover[f][1] if cover[f][1] else float("nan")}
                     for f in fields},
        "secondary_strata": {"|".join(k): _stat(v) for k, v in sorted(sub.items())},
        "pg_qg_covered": {f: _stat(sub.get((f, "all", "det"), [])) for f in ("pg", "qg")},
        "stratum_share": {st: strata.count(st) for st in ("full", "high", "base")},
        "kcl_gate": {"p_median": kcl_p, "q_median": kcl_q, "gate": KCL_GATE},
        "n_isolated_nodes": n_isolated,
        "n_empty_state_samples": n_empty_state,
        "mirror_ladder": got,
    }
    return out


# ---------------- 自检(训练侧;预注册 §2.3,含分型标定) ----------------

def _probe_paths():
    paths = library_files(ROOT / "lib118")[:5]
    fam_root = ROOT / "families" / "train"
    for fam in sorted(d for d in fam_root.iterdir() if d.is_dir())[:8]:
        fs = library_files(fam)
        if fs:
            paths.append(fs[0])
    return paths


def selfcheck():
    print("== 自检门(训练侧:lib118 前 5 + synth train 8 族首样本;含分型标定)==")
    paths = _probe_paths()
    kcl_p, kcl_q = [], []
    by_type = {t: {"d": [], "v": [], "n_cand": 0} for t in BRANCH_TYPES}
    for p in paths:
        s = GridSample.load_npz(p)
        nf = s.node_feat
        null = nf[:, OFF("is_null")] > 0.5
        miss = s.node_missing[:, SBIT]
        okn = ~null & ~miss
        p_net, q_net = kcl_net(s)
        kcl_p.append(np.abs(p_net - (nf[:, OFF("pg")] - nf[:, OFF("pl")]))[okn])
        kcl_q.append(np.abs(q_net - (nf[:, OFF("qg")] - nf[:, OFF("ql")]))[okn])
        mark = s.branch_feat[:, BOFF("mark")] > 0.5
        btypes = np.argmax(s.branch_feat[:, 0:6], axis=1)
        ei = s.edge_index
        for b_idx in range(s.n_branches):
            if mark[b_idx] and ei[0][b_idx] != ei[1][b_idx]:
                bt = BRANCH_TYPES[int(btypes[b_idx])]
                if bt != "shunt_comp":
                    by_type[bt]["n_cand"] += 1
        vt = nf[:, OFF("v")]
        tt = nf[:, OFF("theta")]
        for a, c, u, w, d, bt in branch_estimates(s):
            if okn[a] and okn[c]:
                by_type[bt]["d"].append(abs(d - (tt[a] - tt[c])))
                by_type[bt]["v"].append(abs(np.sqrt(u) - vt[a]))
                by_type[bt]["v"].append(abs(np.sqrt(w) - vt[c]))
    kcl_p = np.concatenate(kcl_p)
    kcl_q = np.concatenate(kcl_q)

    print(f"  分型标定表(取舍规则:|V| 中位<1e-3 入 KEEP_V;δ 中位<1e-4 入 KEEP_TH):")
    for t, d in by_type.items():
        if d["n_cand"] == 0 and not d["d"]:
            print(f"    {t:14s} 候选 0(标定面无此类型 → 保守排除)")
            continue
        dm = np.median(d["d"]) if d["d"] else float("nan")
        vm = np.median(d["v"]) if d["v"] else float("nan")
        print(f"    {t:14s} 候选 {d['n_cand']:5d} 可用 {len(d['d']):5d}  "
              f"δ中位 {dm:.2e}  |V|中位 {vm:.2e}  当前表:V={'入' if t in KEEP_V else '弃'}"
              f"/θ={'入' if t in KEEP_TH else '弃'}")

    verr = np.concatenate([np.array(by_type[t]["v"]) for t in KEEP_V if by_type[t]["v"]])
    derr = np.concatenate([np.array(by_type[t]["d"]) for t in KEEP_TH if by_type[t]["d"]])
    gates = [
        ("KCL 有功中位 < 1e-5", float(np.median(kcl_p)), np.median(kcl_p) < 1e-5),
        ("KCL 无功中位 < 1e-5", float(np.median(kcl_q)), np.median(kcl_q) < 1e-5),
        ("θ δ 误差中位 < 1e-4(入选类型)", float(np.median(derr)), np.median(derr) < 1e-4),
        ("|V| 误差中位 < 5e-4(硬门,入选类型)", float(np.median(verr)), np.median(verr) < 5e-4),
        ("|V| 误差 P99 < 0.05(入选类型)", float(np.percentile(verr, 99)), np.percentile(verr, 99) < 0.05),
    ]
    print(f"  |V| P95 = {np.percentile(verr, 95):.2e}  最大 = {verr.max():.2e}"
          f"  δ P95 = {np.percentile(derr, 95):.2e}")
    allpass = True
    for name, val, ok in gates:
        print(f"  [{'过' if ok else '不过'}] {name}:{val:.3e}")
        allpass &= ok
    print("自检门 =", "全过" if allpass else "不过(停,不得触 val/transfer)")
    return allpass


# ---------------- 正式跑 ----------------

def run_main():
    rep = json.load(open(V1_REPORT, encoding="utf-8"))
    g3 = rep["gate3_transfer"]
    val_raw = load_families(ROOT, "val")          # 加载唯一入口(审查 B3)
    xfer_raw = load_families(ROOT, "transfer")
    assert len(val_raw) == g3["n_in"] and len(xfer_raw) == g3["n_out"], "集合计数与验收不符,停"

    res = {"in": run_arm(val_raw, g3["ladder_in"]),
           "out": run_arm(xfer_raw, g3["ladder_out"])}
    out = {
        "prereg": "docs/预注册-泄漏修复线-解析基线与支路遮蔽分叉-20260827.md",
        "seed": SEED, "mask_seed_offset": _VAL_MASK_SEED_OFFSET,
        "constants": {"EPS_R": EPS_R, "EPS_K": EPS_K, "CONS_TOL": CONS_TOL, "EPS_X": EPS_X,
                      "RIDGE": RIDGE, "QUAD_DEGEN": QUAD_DEGEN, "FIX_ITERS": FIX_ITERS,
                      "KEEP_V": sorted(KEEP_V), "KEEP_TH": sorted(KEEP_TH), "KCL_GATE": KCL_GATE},
        "analytic": res,
        "l4_model_pu": {"in": g3["mae_in_pu"], "out": g3["mae_out_pu"]},
        "ladder": {"in": g3["ladder_in"], "out": g3["ladder_out"]},
        "skill_v1": {"in": g3["skill_in"], "out": g3["skill_out"]},
    }
    dst = Path(__file__).resolve().parent.parent / "runs" / "analytic_baseline_arm_20260827.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[写出] {dst}")
    for setn in ("in", "out"):
        a = res[setn]
        print(f"\n[{setn}] 证书内建 ✓  KCL 门 p={a['kcl_gate']['p_median']:.1e} "
              f"q={a['kcl_gate']['q_median']:.1e}  遮蔽档样本 {a['stratum_share']}")
        for f in ("v", "theta"):
            p = a["primary_full_denominator"][f]
            c = a["coverage"][f]
            print(f"  {f:6s} 主口径 mae={p['mae']:.6g} (全位 {p['n']})  覆盖 {c['frac']:.3f}")
        for f in ("pg", "qg"):
            p = a["pg_qg_covered"][f]
            c = a["coverage"][f]
            print(f"  {f:6s} 可判子集 mae={p['mae']:.6g} (n={p['n']})  覆盖 {c['frac']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        sys.exit(0 if selfcheck() else 2)
    elif a.run:
        run_main()
    else:
        ap.print_help()
