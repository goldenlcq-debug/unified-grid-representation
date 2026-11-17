"""对称孪生用例集构造 + 五判据孪生探针(方案 8.3,2026-07-09 FABLE 评审修订版)。

**为什么在 GridSample 层构造,而非 PSASP 重解固化**(偏离交接文档的说明):
FABLE 评审谱分析已证——真孪生的可分性来自反对称模,而**边权/迭代求解会破坏严格自同构**
(线路导纳的微小不等即让"孪生"不再是数学孪生)。要检验"LaPE 在严格孪生上不可分、身份只能
由随机通道承载",孪生两侧的全部输入通道必须**逐字节相同**——这只能靠构造保证,不能靠重解。
故本模块直接在档案层复制节点/支路造严格孪生:除持久化随机通道外所有输入通道逐字节相同。

孪生类型:
- 节点孪生:新增节点 b,连到节点 a 的**全部邻居**(同边特征),node_feat[b]=node_feat[a]。
  swap(a,b) 是图自同构(a 的边集 ↔ b 的边集,邻居全不动),a、b 非相邻 → 严格真孪生。
  LaPE 在 a、b 逐位相同(反对称模 λ=1 落在 λ₁₆ 之外,见 test_lape_true_twins_*)。
- 支路孪生:复制支路成并联同型回路(同 i、j、同 branch_feat)。两条支路 token 输入除
  branch_random 外逐字节相同(2026-07-09 评审:这是 token 级区分"1 号/2 号回路"的唯一手段)。

五判据(§8.3):①可分性(相对裕度) ②跨视图检索 ②'状态扰动 ③机制消融 ④指针探针(在报告脚本)。
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch

from ..data.graph import sample_to_data
from ..schema.fields import MASKABLE_GROUPS, NODE_TABLE, Group
from ..schema.sample import GridSample, group_bit_index
from ..train.masking import COL_GROUP


def _blank_masked(n_or_m: int) -> np.ndarray:
    return np.zeros((n_or_m, len(MASKABLE_GROUPS)), dtype=bool)


def add_node_twin(sample: GridSample, node_idx: int) -> Tuple[GridSample, Tuple[int, int]]:
    """在 node_idx 上造一个严格真孪生节点(接入其全部邻居,同边特征)。返回 (新样本, (a, b))。

    新节点 b = 原 n_nodes。node_feat/node_missing 从 a 复制;边(a,nb)→(b,nb) 同 branch_feat。
    随机通道清空 → 由新 n_nodes 重新派生(a、b 各得独立随机行 = 唯一破缺手段)。
    """
    a = int(node_idx)
    n = sample.n_nodes
    b = n
    ei = sample.edge_index
    # a 的关联边(无向图:edge_index 只存一条方向,两端都要查)
    inc = np.where((ei[0] == a) | (ei[1] == a))[0]
    new_cols, new_bfeat = [], []
    for e in inc:
        i, j = int(ei[0, e]), int(ei[1, e])
        nb = j if i == a else i          # a 的邻居
        new_cols.append((b, nb))
        new_bfeat.append(sample.branch_feat[e])
    add_ei = np.array(new_cols, dtype=ei.dtype).T.reshape(2, -1)

    node_feat = np.vstack([sample.node_feat, sample.node_feat[a:a + 1]])
    node_missing = np.vstack([sample.node_missing, sample.node_missing[a:a + 1]])
    branch_feat = np.vstack([sample.branch_feat, np.array(new_bfeat, dtype=sample.branch_feat.dtype)])
    edge_index = np.concatenate([ei, add_ei], axis=1)

    out = GridSample(
        node_feat=node_feat, branch_feat=branch_feat, edge_index=edge_index,
        node_missing=node_missing, node_masked=_blank_masked(b + 1),
        branch_missing=np.vstack([sample.branch_missing,
                                  np.zeros((add_ei.shape[1], len(MASKABLE_GROUPS)), dtype=bool)]),
        branch_masked=_blank_masked(branch_feat.shape[0]),
        sample_id=f"{sample.sample_id}+ntwin{a}", family_id=sample.family_id,
    )
    out.random_features(); out.branch_random_features()   # 按新尺寸重新派生并持久化
    return out, (a, b)


def add_branch_twin(sample: GridSample, branch_idx: int) -> Tuple[GridSample, Tuple[int, int]]:
    """复制 branch_idx 成并联同型回路。返回 (新样本, (原支路 token, 新支路 token))。"""
    e = int(branch_idx)
    edge_index = np.concatenate([sample.edge_index, sample.edge_index[:, e:e + 1]], axis=1)
    branch_feat = np.vstack([sample.branch_feat, sample.branch_feat[e:e + 1]])
    out = GridSample(
        node_feat=sample.node_feat, branch_feat=branch_feat, edge_index=edge_index,
        node_missing=sample.node_missing, node_masked=_blank_masked(sample.n_nodes),
        branch_missing=np.vstack([sample.branch_missing, sample.branch_missing[e:e + 1]]),
        branch_masked=_blank_masked(branch_feat.shape[0]),
        sample_id=f"{sample.sample_id}+btwin{e}", family_id=sample.family_id,
    )
    out.random_features(); out.branch_random_features()
    return out, (e, sample.n_branches)


def verify_true_twin(sample: GridSample, a: int, b: int) -> bool:
    """校验 a、b 是严格真孪生:去掉彼此后邻居集相同,且档案节点特征逐字节相同。"""
    ei = sample.edge_index
    def nbrs(x):
        s = set(ei[1, ei[0] == x].tolist()) | set(ei[0, ei[1] == x].tolist())
        return s - {a, b}
    return nbrs(a) == nbrs(b) and np.array_equal(sample.node_feat[a], sample.node_feat[b])


def pick_twin_anchors(sample: GridSample, k: int) -> List[int]:
    """确定性多锚点选择(探针加密,交接债①):有效节点、度≥2,按 (度, 编号) 排序后
    等距抽取——覆盖度谱两端与中段,发电/负荷母线自然混入。单锚点粒度 12.5% 的统计
    盲区由此消除;LaPE 前提(twin_probe guard)不满足时由调用方顺延到下一候选。
    """
    off = NODE_TABLE.offset_of("is_null")
    deg = np.bincount(sample.edge_index.ravel(), minlength=sample.n_nodes)
    cand = [a for a in range(sample.n_nodes)
            if sample.node_feat[a, off] < 0.5 and deg[a] >= 2]
    cand.sort(key=lambda a: (int(deg[a]), a))
    if not cand:
        return []
    idx = np.unique(np.linspace(0, len(cand) - 1, num=min(k, len(cand)), dtype=int))
    return [cand[i] for i in idx]


def pick_branch_anchors(sample: GridSample, k: int) -> List[int]:
    """确定性多支路锚点:全支路按编号等距抽取(支路孪生构造对任意支路合法)。"""
    m = sample.n_branches
    if m == 0:
        return []
    return sorted(set(np.linspace(0, m - 1, num=min(k, m), dtype=int).tolist()))


def _branch_col_group_index() -> np.ndarray:
    from ..schema.fields import BRANCH_TABLE
    gi = {g: i for i, g in enumerate(MASKABLE_GROUPS)}
    idx: List[int] = []
    for f in BRANCH_TABLE.fields:
        idx.extend([gi.get(f.group, -1)] * f.dim)
    return np.array(idx, dtype=np.int64)


BRANCH_COL_GROUP = _branch_col_group_index()


def transplant_state(sample: GridSample, donor: GridSample) -> GridSample:
    """②' 状态扰动视图 B'(§8.3-②',持久化口径):同拓扑同随机特征,运行状态组
    (节点 + 支路)整体移植自**同家族另一已收敛方式**——"限内再调度"的最诚实来源是
    求解器真实解出的另一个运行点,不是人造噪声。replace() 保持 sample_id/family_id 与
    随机特征缓存 → 两视图持久化标签逐位相同(②' 前提)。
    """
    if (sample.n_nodes != donor.n_nodes or sample.n_branches != donor.n_branches
            or not np.array_equal(sample.edge_index, donor.edge_index)):
        raise ValueError("②' 状态移植要求同家族同拓扑方式(节点/支路/edge_index 一致)")
    sbit = group_bit_index(Group.STATE)
    non_state_n = np.array([g != sbit for g in COL_GROUP])
    if not np.allclose(sample.node_feat[:, non_state_n], donor.node_feat[:, non_state_n],
                       atol=1e-6):
        raise ValueError("②' 前提破坏:两方式非状态列不一致(不是同一母网?)")
    nf = sample.node_feat.copy()
    nf[:, ~non_state_n] = donor.node_feat[:, ~non_state_n]
    non_state_b = np.array([g != sbit for g in BRANCH_COL_GROUP])
    bf = sample.branch_feat.copy()
    bf[:, ~non_state_b] = donor.branch_feat[:, ~non_state_b]
    nm = sample.node_missing.copy(); nm[:, sbit] = donor.node_missing[:, sbit]
    bm = sample.branch_missing.copy(); bm[:, sbit] = donor.branch_missing[:, sbit]
    return replace(sample, node_feat=nf, branch_feat=bf, node_missing=nm, branch_missing=bm)


def state_perturb_probe(encoder, twin_view_a: GridSample, twin_view_b: GridSample,
                        node_twins: List[Tuple[int, int]],
                        random_dim: int = 8, branch_random_dim: int = 8,
                        lape_k: int = 16, rwse_steps: int = 16) -> dict:
    """§8.3-②' 状态扰动判据(补实现,交接债①):同拓扑同随机特征、扰动运行点后
    交叉检索,孪生 top-1 ≥99%——防身份与状态值非线性纠缠(②的遮蔽视图测不出)。

    视图 A/B' 须由 transplant_state 构造(同 sample_id → 持久化标签逐位相同,入口校验)。
    双向四查询(A[a]→B'、B'[a]→A、a/b 对称);并列(差为 0)计错,与 8.2 一致。
    主判据 = 方案文本的全库 top-1;另报**孪生对内**命中率(②同哲学)与状态漂移诊断——
    若全库失败而对内完好,即 ② 修订时"被无关第三节点截胡"的同款情形,留给用户裁决。
    """
    if not np.array_equal(twin_view_a.random_features(random_dim),
                          twin_view_b.random_features(random_dim)):
        raise ValueError("②' 前提破坏:两视图持久化随机特征不一致")
    encoder.eval()
    tA, _ = _encode(encoder, twin_view_a, lape_k, rwse_steps, random_dim, branch_random_dim)
    tB, _ = _encode(encoder, twin_view_b, lape_k, rwse_steps, random_dim, branch_random_dim)
    full_hits, pair_hits = [], []
    for a, b in node_twins:
        for q, other in ((a, b), (b, a)):
            for src, dst in ((tA, tB), (tB, tA)):
                sim = src[q] @ dst.T
                full_hits.append(int(sim.argmax()) == q)
                pair_hits.append(bool(float(sim[q]) > float(sim[other])))
    drift = 1.0 - (tA * tB).sum(-1)
    full_rate = float(np.mean(full_hits)) if full_hits else float("nan")
    pair_rate = float(np.mean(pair_hits)) if pair_hits else float("nan")
    return {
        "n_twins": len(node_twins),
        "perturb_hit_rate": full_rate, "perturb_ok": bool(full_hits and full_rate >= 0.99),
        "perturb_pair_rate": pair_rate,      # 诊断:孪生对内(②修订同哲学)
        "state_drift_median": float(drift.median()),
    }


# ---------------- 编码工具 ----------------

def _mask_state_view(sample: GridSample) -> GridSample:
    """视图 B:运行状态组全遮蔽(c_topo 生产视图)。状态列置 0,不改随机通道/拓扑/静态。"""
    nf = sample.node_feat.copy()
    sbit = group_bit_index(Group.STATE)
    for c in range(nf.shape[1]):
        if COL_GROUP[c] == sbit:
            nf[:, c] = 0.0
    return replace(sample, node_feat=nf)


def _encode(encoder, sample: GridSample, lape_k: int, rwse_steps: int,
            random_dim: int, branch_random_dim: int, ablate: bool = False):
    """持久化口径编码 → 归一化 (node_tokens, edge_tokens)。

    ablate=True:把随机通道**置零**(而非删列)——保持输入维度不变(编码器 node_proj 宽度固定),
    检验"抹掉随机通道后孪生是否塌回不可分"(机制消融,方案 8.3③)。
    """
    rf = np.zeros((sample.n_nodes, random_dim), dtype=np.float32) if ablate else None
    brf = np.zeros((sample.n_branches, branch_random_dim), dtype=np.float32) if ablate else None
    data = sample_to_data(sample, lape_k, rwse_steps, random_dim=random_dim,
                          random_feat=rf, branch_random_dim=branch_random_dim,
                          branch_random_feat=brf)
    with torch.no_grad():
        h, _g, e = encoder(Batch.from_data_list([data]))
    return (torch.nn.functional.normalize(h, dim=-1),
            torch.nn.functional.normalize(e, dim=-1))


# ---------------- 五判据探针 ----------------

def _ridge_readout(X: np.ndarray, Y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """轻量线性读出 W[d,k](岭回归闭式解)。X = 归一化 token,Y = 持久化随机标签。

    n<d 时欠定,lam 正则保证可解;孪生对不进拟合集(见调用方),预测孪生是真泛化。"""
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def twin_probe(encoder, sample: GridSample, node_twins: List[Tuple[int, int]],
               random_dim: int = 8, branch_random_dim: int = 8,
               lape_k: int = 16, rwse_steps: int = 16, alpha: float = 1.0) -> dict:
    """节点孪生五判据(持久化口径)。返回逐项指标 + 门禁布尔。

    **2026-07-11 用户批准修订(方案 §8.3)**:①②改**读出口径**——四点操作带扫描实测
    (w=0/0.02/0.05/0.15,val)证明裸 token 余弦裕度/裸 xview 与 8.2 重采样检索结构性互斥
    (瓶颈=状态遮蔽自漂移 0.16~0.22,w=0 即存在),而身份已可线性解码(tag MSE 0.008)。
    ①可解码区分:岭回归读出(孪生对不进拟合集)下,孪生各自的读出在 {自己,孪生} 标签中
      最近邻命中自己,≥99%;另报重采样 token 移动量中位数 ≤0.3(过度依赖上界,护 8.2)。
    ②读出空间跨视图身份(2026-07-11 二次修订):**孪生对内**——视图 B(状态遮蔽)token 的
      读出在 {自己, 孪生} 中命中自己,≥99%;全库标签寻址降诊断项(decode_bank_rate)。
    ③机制消融:随机通道置零 → 孪生对裸 sep 必须塌到 ~0(证明身份靠随机通道,非 LaPE 简并数值)。
    原裸口径(sep/裕度比/裸 xview)降为诊断项,继续报告。

    **前提守卫** lape_identical:"严格真孪生 ⇒ LaPE 两点逐位相同"是图条件性的——孪生所在
    分量若不足 lape_k 个 <1 的特征值(小稠密分量,如 K5),λ=1 反对称模会进入保留窗口,
    lape[a] = −lape[b],消融后仍可分 → ③会**保守地误判失败**(不会假通过)。故先验证前提并
    随报告输出;前提不成立时 mechanism_ok 不可解读,调用方应换孪生锚点。电网类稀疏图不会触发。
    """
    from ..data.posenc import compute_lape
    lape, _valid, _c = compute_lape(sample.edge_index, sample.n_nodes, k=lape_k)
    lape_diff = max((float(np.abs(lape[a] - lape[b]).max()) for a, b in node_twins), default=0.0)
    lape_identical = lape_diff < 1e-6

    encoder.eval()
    tA, _ = _encode(encoder, sample, lape_k, rwse_steps, random_dim, branch_random_dim)
    B = _mask_state_view(sample)
    tB, _ = _encode(encoder, B, lape_k, rwse_steps, random_dim, branch_random_dim)

    # 自漂移:每个节点自身 A↔B 的余弦距离(跨视图基线)
    self_drift = 1.0 - (tA * tB).sum(-1)
    drift_med = float(self_drift.median())

    seps, xview_hits = [], []
    for a, b in node_twins:
        seps.append(float(1.0 - (tA[a] * tA[b]).sum()))
        # 跨视图检索:query = A 的 a token,候选 = 全部 B token,top-1 应命中 a
        sim_a = tA[a] @ tB.T
        sim_b = tB[b] @ tA.T
        xview_hits.append(int(sim_a.argmax()) == a)
        xview_hits.append(int(sim_b.argmax()) == b)
    sep_min = float(np.min(seps)) if seps else float("nan")

    # ③机制消融:随机通道**置零**(不删列)重编码
    z, _ = _encode(encoder, sample, lape_k, rwse_steps, random_dim, branch_random_dim, ablate=True)
    abl_seps = [float(1.0 - (z[a] * z[b]).sum()) for a, b in node_twins]
    abl_max = float(np.max(abl_seps)) if abl_seps else float("nan")

    # ---- ①②读出口径(2026-07-11 用户批准修订,方案 §8.3):裸余弦裕度/裸 xview 与 8.2
    # 结构性互斥(操作带四点扫描为空,见变更记录 2026-07-11),降为诊断项;门禁改为
    # "轻量线性读出下身份可解码"——与 §8.3-⑤ 指针探针同哲学:下游头可用才是生产能力。----
    tags = sample.random_features(random_dim).astype(np.float64)      # 持久化标签
    valid = sample.node_feat[:, NODE_TABLE.offset_of("is_null")] < 0.5
    fit_mask = valid.copy()
    for a, b in node_twins:
        fit_mask[a] = fit_mask[b] = False                             # 孪生对不进拟合集
    A_np, B_np = tA.numpy(), tB.numpy()
    W = _ridge_readout(A_np[fit_mask], tags[fit_mask])
    dec_twin_hits, dec_xview_hits, bank_hits = [], [], []
    valid_idx = np.where(valid)[0]
    tag_bank = tags[valid_idx]                                        # 全库寻址(诊断)候选
    for a, b in node_twins:
        for q, other in ((a, b), (b, a)):
            # 视图 A 孪生区分:读出结果在 {自己, 孪生} 标签里最近邻命中自己
            p = A_np[q] @ W
            dec_twin_hits.append(float(np.linalg.norm(p - tags[q]))
                                 < float(np.linalg.norm(p - tags[other])))
            # ②跨视图身份(2026-07-11 二次修订,用户批准):**孪生对内**口径——视图 B
            # (状态遮蔽)token 的读出在 {自己, 孪生} 中命中自己。链式闭环需要的是孪生
            # 不互换;全库标签寻址属 8.2/目录页(4.4)职责,降为诊断项(实测:对内身份
            # 全部完好时仍可能被无关第三节点截胡,见变更记录)。
            pb = B_np[q] @ W
            dec_xview_hits.append(float(np.linalg.norm(pb - tags[q]))
                                  < float(np.linalg.norm(pb - tags[other])))
            hit = valid_idx[int(np.argmin(np.linalg.norm(tag_bank - pb, axis=1)))]
            bank_hits.append(hit == q)                                # 诊断:全库寻址
    dec_twin_rate = float(np.mean(dec_twin_hits)) if dec_twin_hits else float("nan")
    dec_xview_rate = float(np.mean(dec_xview_hits)) if dec_xview_hits else float("nan")
    bank_rate = float(np.mean(bank_hits)) if bank_hits else float("nan")

    # 过度依赖上界(§8.3-① 保留项):重采样随机通道,token 移动量中位数 ≤0.3(初值,待校准)
    rng = np.random.default_rng(0)
    rf2 = rng.standard_normal((sample.n_nodes, random_dim)).astype(np.float32)
    brf2 = rng.standard_normal((sample.n_branches, branch_random_dim)).astype(np.float32)
    d2 = sample_to_data(sample, lape_k, rwse_steps, random_dim=random_dim, random_feat=rf2,
                        branch_random_dim=branch_random_dim, branch_random_feat=brf2)
    with torch.no_grad():
        h2, _g2, _e2 = encoder(Batch.from_data_list([d2]))
    t2 = torch.nn.functional.normalize(h2, dim=-1)
    rand_share = float((tA[valid] - t2[valid]).norm(dim=-1).median())

    margin_ok = bool(seps and sep_min >= alpha * drift_med)
    xview_rate = float(np.mean(xview_hits)) if xview_hits else float("nan")
    ablated_collapsed = bool(abl_seps and abl_max < 1e-3)
    return {
        "n_twins": len(node_twins),
        # 门禁判据(读出口径,2026-07-11)
        "decode_twin_rate": dec_twin_rate, "decode_twin_ok": bool(dec_twin_hits and dec_twin_rate >= 0.99),
        "decode_xview_rate": dec_xview_rate, "decode_xview_ok": bool(dec_xview_hits and dec_xview_rate >= 0.99),
        "decode_bank_rate": bank_rate,        # 诊断项:全库标签寻址(不裁决)
        "rand_share_median": rand_share, "overuse_ok": bool(rand_share <= 0.3),
        "ablated_sep_max": abl_max, "mechanism_ok": ablated_collapsed,
        # 诊断项(原裸余弦口径,保留观测)
        "sep_min": sep_min, "self_drift_median": drift_med, "alpha": alpha,
        "separability_ok": margin_ok,
        "xview_hit_rate": xview_rate, "xview_ok": bool(xview_hits and xview_rate >= 0.99),
        "lape_twin_max_diff": lape_diff, "lape_identical": lape_identical,
    }


def branch_twin_separability(encoder, sample: GridSample, branch_twins: List[Tuple[int, int]],
                             random_dim: int = 8, branch_random_dim: int = 8,
                             lape_k: int = 16, rwse_steps: int = 16) -> dict:
    """支路孪生可分性:并联同型回路两条支路 token 除 branch_random 外输入逐字节相同,
    编码后必须可分(sep>0);随机通道置零则必须塌到 0。"""
    encoder.eval()
    _h, e = _encode(encoder, sample, lape_k, rwse_steps, random_dim, branch_random_dim)
    m = sample.n_branches
    seps = [float(1.0 - (e[a] * e[b]).sum()) for a, b in branch_twins]   # e 前 m 行 = 正向边

    _h0, e0 = _encode(encoder, sample, lape_k, rwse_steps, random_dim, branch_random_dim, ablate=True)
    abl = [float(1.0 - (e0[a] * e0[b]).sum()) for a, b in branch_twins]
    return {
        "n_branch_twins": len(branch_twins),
        "sep_min": float(np.min(seps)) if seps else float("nan"),
        "ablated_sep_max": float(np.max(abl)) if abl else float("nan"),
        "separable": bool(seps and np.min(seps) > 1e-3),
        "mechanism_ok": bool(abl and np.max(abl) < 1e-3),
        "_m": m,
    }
