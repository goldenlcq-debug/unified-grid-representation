"""潮流计算任务推理器(方案 5.1/5.2/5.3;D13/D15;P2 载体任务之一)。

**输入协议(D15,防真值泄漏,强制)**:推理输入档案由"未解方式"编码得到——
1. 节点:按类型将 PF 未知量置缺失并清零(PQ:V/θ;PV:Q/θ;平衡机:P/Q),
   注入量保持可见。字段映射复用 masking.PF_UNKNOWN_FIELDS(pf_typed 课程同款,单测锁定);
2. 支路:pi/qi/pj/qj(潮流解算产物)一并清零——未解方式没有支路潮流,留着即真值泄漏
   (v0 决策:方案 5.2 文本只点名节点未知量,支路潮流按"未解方式"语义从严处理,
   mark 投退是给定量保留;见变更记录)。
真值只出现在损失监督侧。配强制泄漏单测(tests/test_pf_reasoner.py)。

**口径 A/B 对照(2026-07-11 用户裁决:先做对照再定,均不动编码器)**:
- view="field"(B,现行):上述字段级视图。该花样在 v1 预训练课程中结构性缺席
  (评审第 4 项,OOD 混杂);
- view="group"(A):节点侧整 STATE 组清零(**节点列输入花样**与预训练组缺失课程
  c_topo 逐位同款;逐列语义与 masking.apply_node_mask 组缺失分支单测锁定),给定量(注入/
  负荷/设定值)改由 pf_side_features 从推理器侧原始并读(D13:任务参数在推理器侧;
  known 位显式携带 = D15"missing 位"半边协议在推理器侧的落地)。
  支路潮流仍清零(泄漏红线;预训练从未遮支路,此半边两口径同为 OOD 残留,如实入档)。
**作用域限定(2026-07-11 评审)**:"逐位同款"仅在编码器输入张量的节点特征列层面成立;
编码器是消息传递 GNN + 全局 token,支路清零会经消息传递污染每个节点 token,故编码器输入
整体(含支路)仍含两口径共享的 OOD 残留,"节点 token 表征分布内"不因此成立。残留在 A/B
对照中相互抵消(两口径共享),不影响对照结论;但不得据此把 A 口径结果与预训练 c_topo
重构指标直接比较。两口径 target/loss_mask 逐位一致 → 结果直接可比。

**推理器基准结构(5.3)**:任务侧 Q 个可学习查询 token → n 层[交叉注意力读档案
(节点+支路+全局 token 全集)+ 自注意力 + FFN] → 逐节点共享回归头(节点 token
交叉读回任务 token 作上下文)。输出 (P, Q, V, θ) 四元组,损失按节点类型只对未知量计
(5.2 节点类型损失掩码)。参数 ~1e6(D9);全部参数在推理器侧,编码器冻结不动(D13)。
红线 4:交叉注意力的键 = 编码器 token,LaPE/RWSE 拓扑注入已含其中。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

from ..schema.fields import BRANCH_TABLE, NODE_TABLE, Group
from ..schema.sample import GridSample, group_bit_index
from ..train.masking import COL_GROUP, PF_UNKNOWN_FIELDS, _pf_typed_mask

# 推理器监督/输出的 4 个状态字段(5.2 四元组;列序即头的输出通道序)
PF_FIELDS = ("pg", "qg", "v", "theta")
BRANCH_SOLVED_FIELDS = ("pi", "qi", "pj", "qj")
# STATE 组全列(A 口径整组遮蔽作用域;pl/ql/v_set 亦属 STATE,一并遮蔽后从侧通道供)
STATE_FIELDS = tuple(f.name for f in NODE_TABLE.fields if f.group == Group.STATE)
PF_VIEWS = ("field", "group")
PF_SIDE_DIM = 2 * len(STATE_FIELDS)   # 给定量值 + known 位,各 |STATE|


def pf_input_view(sample: GridSample, view: str = "field",
                  ) -> Tuple[GridSample, np.ndarray, np.ndarray]:
    """D15 未解方式编码。返回 (任务视图, target[N,F], loss_mask[N,F])。

    输入应为**已归一化**样本;清零即"标准化空间置 0"(与掩码课程同约定)。
    视图保持 sample_id/family_id → 持久化随机特征同源(7.4 逐位一致)。
    view 语义见模块 docstring(A/B 口径对照);两口径监督逐位一致。
    """
    if view not in PF_VIEWS:
        raise ValueError(f"未知口径 {view!r},应为 {PF_VIEWS}")
    masked_feat, target, loss_mask = _pf_typed_mask(sample.node_feat, sample.node_missing)
    if view == "group":
        sbit = group_bit_index(Group.STATE)
        rows = ~sample.node_missing[:, sbit]          # 组缺失分支同款:仅豁免本就 missing
        masked_feat[np.ix_(rows, np.where(COL_GROUP == sbit)[0])] = 0.0
    bf = sample.branch_feat.copy()
    for f in BRANCH_SOLVED_FIELDS:
        bf[:, BRANCH_TABLE.offset_of(f)] = 0.0
    view_s = replace(sample, node_feat=masked_feat, branch_feat=bf)
    return view_s, target, loss_mask


def pf_side_features(sample: GridSample) -> np.ndarray:
    """A 口径推理器侧给定量通道 [N, PF_SIDE_DIM](D13,不进编码器)。

    前 |STATE| 列 = STATE 组原始值(类型化未知量清零),后 |STATE| 列 = known 位。
    未知位置值与位恒 0(泄漏单测锁定);STATE missing / NULL 占位行全 0。
    known 位显式区分"给定 0"与"未知置 0"——D15 missing 位半边协议的推理器侧落地。
    """
    n = sample.n_nodes
    k = len(STATE_FIELDS)
    out = np.zeros((n, 2 * k), dtype=np.float32)
    sbit = group_bit_index(Group.STATE)
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(sample.node_feat[:, tcol:tcol + 3], axis=1)
    null = sample.node_feat[:, NODE_TABLE.offset_of("is_null")] > 0.5
    active = ~sample.node_missing[:, sbit] & ~null            # 携带给定量的行
    for j, f in enumerate(STATE_FIELDS):
        unknown = np.zeros(n, dtype=bool)                    # 该字段对本类型是否 PF 未知量
        for t, unk in PF_UNKNOWN_FIELDS.items():
            if f in unk:
                unknown |= types == t
        known = active & ~unknown
        out[known, j] = sample.node_feat[known, NODE_TABLE.offset_of(f)]
        out[known, k + j] = 1.0
    return out


class _MPLayer(nn.Module):
    """推理器侧薄消息传递(θ 杠杆 c;D13 注记:任务物理属推理器侧自由)。

    红线自查(2.4):①权重 d×d 级共享,无 N 依赖;②消息沿真实电网边(edge_index),
    边特征 = 编码器支路 token(含拓扑注入);③无按编号查表;④输出逐 token。
    动机(2026-07-11 诊断):A 口径注入量只经侧通道进推理器,原推理器无图局部传播
    算力,θ 误差随离平衡机图距离线性增长(+0.01~0.07 rad/跳)——补消息传递即补积分。
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, h, e, edge_index):
        src, dst = edge_index[0], edge_index[1]
        x = self.ln(h)
        m = self.msg(torch.cat([x[src], x[dst], e], dim=-1))
        # index_add_ 不在 autocast 白名单:AMP 下 m 为 fp16 而累加器随 h 为 fp32,显式对齐
        agg = torch.zeros_like(h).index_add_(0, dst, m.to(h.dtype))
        return h + self.upd(torch.cat([x, agg], dim=-1))


class _ReasonerLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.ln_c, self.ln_s, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(),
                                 nn.Linear(ffn_mult * d, d))

    def forward(self, q, arch, kpm):
        x = self.ln_c(q)
        q = q + self.cross(x, arch, arch, key_padding_mask=kpm, need_weights=False)[0]
        x = self.ln_s(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        return q + self.ffn(self.ln_f(q))


class PFReasoner(nn.Module):
    def __init__(self, d: int, n_queries: int = 16, n_layers: int = 2,
                 n_heads: int = 4, ffn_mult: int = 2, side_dim: int = 0,
                 gnn_layers: int = 0, theta_head: bool = False) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.layers = nn.ModuleList(_ReasonerLayer(d, n_heads, ffn_mult)
                                    for _ in range(n_layers))
        self.ln_node = nn.LayerNorm(d)
        self.node_cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                  nn.Linear(d, len(PF_FIELDS)))
        # A 口径给定量侧通道(D13 推理器侧参数);0 = B 口径,结构与旧检查点逐位兼容
        self.side_proj = nn.Linear(side_dim, d) if side_dim else None
        # θ 杠杆 c:侧通道注入后的薄消息传递(0 = 关,旧检查点逐位兼容)
        self.mp = nn.ModuleList(_MPLayer(d) for _ in range(gnn_layers))
        # θ 杠杆 b:独立 θ 头(防 θ 梯度被 pg/qg/v 在共享头末层挤压)
        self.th_head = (nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
                        if theta_head else None)

    def forward(self, h: torch.Tensor, g: torch.Tensor, e: torch.Tensor,
                node_batch: torch.Tensor, edge_batch: torch.Tensor,
                side: torch.Tensor = None, edge_index: torch.Tensor = None) -> torch.Tensor:
        """h[N,d] 节点 token、g[B,K,d] 全局、e[E,d] 支路(对称化两份);返回 pred[N,4]。

        side[N,side_dim]:A 口径给定量(pf_side_features)。配置与输入必须成对——
        防止 group 视图漏供给定量 / field 视图重复供,两种静默口径混用都判错。
        edge_index[2,E]:gnn_layers>0 时必供(与 e 行对齐的对称化边);同样成对判错。
        """
        if (side is None) != (self.side_proj is None):
            raise ValueError("side 输入与 side_dim 配置不成对(A/B 口径混用)")
        if (edge_index is None) == bool(len(self.mp)):
            raise ValueError("edge_index 输入与 gnn_layers 配置不成对")
        if self.side_proj is not None:
            h = h + self.side_proj(side)
        for mp in self.mp:
            h = mp(h, e, edge_index)
        hd, hmask = to_dense_batch(h, node_batch)
        ed, emask = to_dense_batch(e, edge_batch)
        ones = torch.ones(g.shape[:2], dtype=torch.bool, device=g.device)
        arch = torch.cat([hd, ed, g], dim=1)
        kpm = ~torch.cat([hmask, emask, ones], dim=1)
        q = self.queries.unsqueeze(0).expand(hd.size(0), -1, -1)
        for layer in self.layers:
            q = layer(q, arch, kpm)
        ctx = self.node_cross(self.ln_node(hd), q, q, need_weights=False)[0]
        feat = torch.cat([h, ctx[hmask]], dim=-1)
        pred = self.head(feat)
        if self.th_head is not None:
            k = PF_FIELDS.index("theta")
            pred = torch.cat([pred[:, :k], self.th_head(feat)], dim=-1)
        return pred

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def pf_supervised_cols() -> Dict[str, int]:
    return {f: NODE_TABLE.offset_of(f) for f in PF_FIELDS}


def pf_loss(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor,
            field_weights: Dict[str, float] = None) -> torch.Tensor:
    """L1(归一化空间),只在类型化未知量位置计(5.2 节点类型损失掩码)。

    field_weights:字段级损失权重(θ 杠杆 b,防 θ 被 pg/qg/v 挤压;已实测损失竞争)。
    None = 全 1,与旧行为逐位一致;加权按 Σ w_f·|err| / Σ w_f·n_f(权重同乘分母,
    保持损失量纲,不等效于改学习率)。
    """
    cols = pf_supervised_cols()
    tot, cnt = pred.new_zeros(()), 0.0
    for k, f in enumerate(PF_FIELDS):
        m = loss_mask[:, cols[f]]
        w = 1.0 if field_weights is None else float(field_weights.get(f, 1.0))
        if m.any() and w > 0:
            tot = tot + w * (pred[m, k] - target[m, cols[f]]).abs().sum()
            cnt += w * int(m.sum())
    return tot / max(cnt, 1e-9)


def pf_zero_learning_baseline(raw_samples: Sequence[GridSample]) -> Dict[str, Dict[str, float]]:
    """任务掩码下的零学习阶梯(D20 同款 L0/L1/L2,掩码 = D15 任务协议而非随机课程)。

    与 8.4 阶梯不可直接混排(掩码流不同);单独成表,用于"注入量可见后 θ 是否突破
    同信息零学习"的裁决。输入 = **原始(未归一化)**样本,MAE 即标幺。
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    sbit = group_bit_index(Group.STATE)
    fallback = {"pg": 0.0, "qg": 0.0, "v": 1.0, "theta": 0.0}
    errs = {f: {"L0": [], "L1": [], "L2": []} for f in PF_FIELDS}
    pooled = {f: [] for f in PF_FIELDS}
    per_sample = []
    for s in raw_samples:
        _mf, _tgt, lm = _pf_typed_mask(s.node_feat, s.node_missing)
        per_sample.append((s, lm))
        ok = ~s.node_missing[:, sbit]
        for f in PF_FIELDS:
            c = NODE_TABLE.offset_of(f)
            pooled[f].append(s.node_feat[ok, c].astype(np.float64))
    pool_mean = {f: (np.concatenate(v).mean() if v else 0.0) for f, v in pooled.items()}

    for s, lm in per_sample:
        n = s.n_nodes
        i, j = s.edge_index
        A = sp.coo_matrix((np.ones(i.size * 2), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
        A.data[:] = 1.0
        L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
        ok = ~s.node_missing[:, sbit]
        for f in PF_FIELDS:
            c = NODE_TABLE.offset_of(f)
            m = lm[:, c] & ok
            if not m.any():
                continue
            truth = s.node_feat[m, c].astype(np.float64)
            vis = ok & ~m
            vis_vals = s.node_feat[vis, c].astype(np.float64)
            errs[f]["L0"].append(np.abs(truth - pool_mean[f]))
            errs[f]["L1"].append(np.abs(truth - (vis_vals.mean() if vis_vals.size else fallback[f])))
            x = np.full(n, fallback[f], dtype=np.float64)
            x[vis] = s.node_feat[vis, c]
            if vis.any():
                I = np.where(m)[0]; B = np.where(vis)[0]
                LII = L[I][:, I].tocsc() + 1e-8 * sp.identity(I.size, format="csc")
                x[I] = spsolve(LII, -L[I][:, B] @ x[B])
            errs[f]["L2"].append(np.abs(x[m] - truth))
    return {f: {k: (float(np.concatenate(v).mean()) if v else float("nan"))
                for k, v in d.items()} for f, d in errs.items()}
