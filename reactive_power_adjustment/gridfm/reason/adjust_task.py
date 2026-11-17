"""有功调整推理器 v0(方向二 R3;方案 5.2 有功潮流调整;D13;宪法多解处理原文)。

**输入协议(与判稳任务同口径,泄漏红线不同)**:病态方式是**已解方式**(收敛但电压
越限)——解后状态(V/θ/Pg/支路潮流)是合法给定量,档案 = 标准档案不掩码。**泄漏红线 =
修复标签**:调整对象集合与 ΔP 只准出现在损失侧;输入侧唯一的任务条件 = 逐节点越限深度
viol_depth = max(0, vio_low − V_i)(输入自身的确定函数,D13 计算型特征;单测锁定
"打乱标签字段输入张量逐位不变")。

**输出(5.2 原文:指针 + 数值头)**:
- 指针头:逐机组 token 打分(共享线性,红线 3)→ multi-hot BCE,正标签 = **全部已验证
  可行修法的机组并集**(宪法 5.2 多解处理:"指针损失允许多正标签");非机组 token 掩掉;
- 数值头:逐机组 token 回归 ΔP(按机组容量幅度归一化)→ L1 只在**规范解**(最小 ‖ΔP‖)
  的机组位计损(宪法:"训练标签唯一化为规范解")。

评估侧(另脚本 adjust_apply_eval):功能性判据——施加预测 → PSASP 复核越限消除且收敛;
不比对构造标签(7.3 评估侧免疫)。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch

from ..schema.fields import NODE_TABLE
from ..schema.sample import GridSample
from .pf_task import _ReasonerLayer
from .ts_task import gen_node_mask

VIO_LOW = 0.93                      # 与 adjust_labels 单边线口径一致(D-1 细调)

# 标签行里除输入定位键外的一切修复字段都视为"结果字段"(泄漏单测据此打乱)
ADJ_INPUT_KEYS = ("sid", "battery", "tag")


def viol_depth(raw_sample: GridSample, vio_low: float = VIO_LOW) -> np.ndarray:
    """逐节点越限深度 [N]:max(0, vio_low − V_i);失电/NULL 行 0。
    输入自身的确定函数(读档案电压列,不读任何标签字段)。"""
    v = raw_sample.node_feat[:, NODE_TABLE.offset_of("v")].astype(np.float64)
    null = raw_sample.node_feat[:, NODE_TABLE.offset_of("is_null")] > 0.5
    d = np.maximum(0.0, vio_low - v)
    d[(v <= 0.01) | null] = 0.0
    return d.astype(np.float32)


def unit_span(raw_sample: GridSample) -> np.ndarray:
    """逐节点机组容量幅度 [N](ΔP 归一化分母):max(pmax−pmin, |pg|, 1e-3);
    非机组行 1(不参与,防除零)。全部为档案给定列。"""
    noff = NODE_TABLE.offset_of
    pmax = raw_sample.node_feat[:, noff("pmax")].astype(np.float64)
    pmin = raw_sample.node_feat[:, noff("pmin")].astype(np.float64)
    pg = raw_sample.node_feat[:, noff("pg")].astype(np.float64)
    span = np.maximum(np.maximum(pmax - pmin, np.abs(pg)), 1e-3)
    gm = gen_node_mask(raw_sample)
    span[~gm] = 1.0
    return span.astype(np.float32)


def build_adjust_example(label_row: Dict, raw_sample: GridSample,
                         tag: str) -> Optional[Dict]:
    """一行调整标签 + 病态方式档案 → 训练例。输入侧只消费档案;标签只落 y_*。

    y_units [N] = 全部可行修法机组并集的 multi-hot(节点行 = 母线号 − 1,须 has_gen,
    错位即 raise);y_dp [N] = 规范解 ΔP / span(仅规范解机组位有效,y_dp_mask 标注)。
    """
    if not label_row.get("feasible"):
        return None
    n = raw_sample.n_nodes
    gm = gen_node_mask(raw_sample)
    span = unit_span(raw_sample)
    y_units = np.zeros(n, dtype=np.float32)
    for fx in label_row["fixes"]:
        for bus_s in fx["deltas"]:
            row = int(bus_s) - 1
            if row < 0 or row >= n or not gm[row]:
                raise ValueError(f"调整标签错位:{tag} 母线 {bus_s} → 行 {row} 非机组节点")
            y_units[row] = 1.0
    canon = label_row["canonical_fix"]
    y_dp = np.zeros(n, dtype=np.float32)
    y_dp_mask = np.zeros(n, dtype=bool)
    for bus_s, dp in canon["deltas"].items():
        row = int(bus_s) - 1
        y_dp[row] = float(dp) / span[row]
        y_dp_mask[row] = True
    return {"tag": tag, "sid": label_row["sid"], "battery": label_row["battery"],
            "y_units": y_units, "y_dp": y_dp, "y_dp_mask": y_dp_mask,
            "n_fixes": int(label_row.get("n_fixes", 1))}


class AdjustReasoner(nn.Module):
    """查询 token 交叉读档案(节点+支路+全局)→ 机组 token 双头(指针 + ΔP)。

    越限深度经 side_proj 加进节点 token(推理器侧 Adapter,D13;冻结编码器输出不动)。
    结构随 5.3 基准(_ReasonerLayer 复用);输出逐 token 共享头(红线 3)。
    """

    def __init__(self, d: int, n_queries: int = 16, n_layers: int = 2,
                 n_heads: int = 4, ffn_mult: int = 2, side_dim: int = 1) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        # side_dim=1 = 有功单边越限深度(缺省,与 v0.6 检查点逐位兼容,先例 PFReasoner side_dim);
        # side_dim=2 = 无功双边深度(低压+过压),reactive_task 用
        self.side_proj = nn.Linear(side_dim, d)   # viol_depth → 节点 token 加性注入
        self.layers = nn.ModuleList(_ReasonerLayer(d, n_heads, ffn_mult)
                                    for _ in range(n_layers))
        self.ln_node = nn.LayerNorm(d)
        self.node_cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.head_ptr = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.head_dp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def node_feat(self, h: torch.Tensor, g: torch.Tensor, e: torch.Tensor,
                  node_batch: torch.Tensor, edge_batch: torch.Tensor,
                  depth: torch.Tensor) -> torch.Tensor:
        """共享主干:档案 + 侧通道 → 逐节点特征 [N,2d](= 节点 token ⊕ 查询上下文)。
        与 forward 原实现逐位一致;拆出来是为了子类(如无功 shunt 版)加第三个头而不改
        本类参数集合——v0.6 检查点必须继续 strict load(单测锁定)。"""
        dep = depth if depth.dim() > 1 else depth.unsqueeze(-1)
        h = h + self.side_proj(dep.to(h.dtype))
        hd, hmask = to_dense_batch(h, node_batch)
        ed, emask = to_dense_batch(e, edge_batch)
        ones_g = torch.ones(g.shape[:2], dtype=torch.bool, device=g.device)
        arch = torch.cat([hd, ed, g], dim=1)
        kpm = ~torch.cat([hmask, emask, ones_g], dim=1)
        q = self.queries.unsqueeze(0).expand(hd.size(0), -1, -1)
        for layer in self.layers:
            q = layer(q, arch, kpm)
        ctx = self.node_cross(self.ln_node(hd), q, q, need_weights=False)[0]
        return torch.cat([h, ctx[hmask]], dim=-1)

    def forward(self, h: torch.Tensor, g: torch.Tensor, e: torch.Tensor,
                node_batch: torch.Tensor, edge_batch: torch.Tensor,
                depth: torch.Tensor):
        """h[N,d]/g[B,K,d]/e[E,d] 冻结档案;depth[N] 或 depth[N,side_dim] 逐节点侧通道。
        返回 (ptr_logit[N], dp_pred[N])。"""
        feat = self.node_feat(h, g, e, node_batch, edge_batch, depth)
        return (self.head_ptr(feat).squeeze(-1), self.head_dp(feat).squeeze(-1))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def adjust_loss(ptr_logit: torch.Tensor, dp_pred: torch.Tensor, batch,
                w_ptr: float = 1.0, w_dp: float = 1.0):
    """指针 BCE(仅机组 token;多正标签 multi-hot)+ ΔP L1(仅规范解机组位)。"""
    gm = batch.gen_mask
    ptr = F.binary_cross_entropy_with_logits(ptr_logit[gm], batch.y_units[gm])
    m = batch.y_dp_mask
    dp = (F.l1_loss(dp_pred[m], batch.y_dp[m]) if m.any()
          else torch.zeros((), device=ptr_logit.device))
    total = w_ptr * ptr + w_dp * dp
    return total, {"total": float(total.detach()), "ptr": float(ptr.detach()),
                   "dp": float(dp.detach()) if m.any() else 0.0}
