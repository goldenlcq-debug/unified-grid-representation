"""G1.5 读出解码器(方案 3.4.2 规格;2026-07-17 诊断 A 触发,用户批准方案 (a))。

与 v1 ReconDecoder 的差异(诊断 A + 归因探针的结论落地):
- **监督口径 = 全位置读出**:v1 的损失只在被掩位置计,解码器只学过"从上下文填充",
  从未学过"读出自己看见的值"——而生成(扩散产物 z = 完整方式潜码)恰恰要后者。
  实测 v1 readout pg MAE 0.97 pu / 探针证明潜码里信息线性可读(0.12)。
- **带 2 层 MPNN**(3.4.2 原文"与编码器同构的 MPNN 层,无全局支路";v1 实现是纯
  逐 token MLP):复用 GPSLayer 的 MPNN 段 + FFN,去掉全局写/读。
- **输入 = 节点 token + c_topo 编码的支路 token + edge_index**:生成期节点 token 是
  扩散产物,支路 token 来自骨架(c_topo)编码——骨架已知,训练/生成分布一致。
  (v1 解码器无支路输出头;扩散状态已按用户裁决收缩为节点 token 场,支路 token
  只作为条件/上下文进解码器,不被生成。)
- **潜码噪声增广**(3.4.2 三前提②,v1 缺席,本次补上):训练时对节点潜码加相对
  尺度高斯噪声(σ_eff = σ × batch 潜码逐通道 std,σ∈{0,0.1,0.2} 采样);支路 token
  不加噪(生成期它来自干净的 c_topo 编码,非采样产物)。re-mask 语义:解码器只见
  token 不见原始特征,结构性满足(同 v1 注记)。

2.4 四红线自查:①权重共享(逐 token/逐边算子,无 N 依赖)✓;②无任务参数进编码器 ✓;
③PSASP 唯一裁决不变 ✓;④跨规模可复用(与编码器同构)✓。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...schema.fields import MASKABLE_GROUPS, NODE_TABLE
from ..encoder.gps import _mlp, scatter_mean
from .recon import GROUP_RANGES


class ReadoutMPNNLayer(nn.Module):
    """GPSLayer 的 MPNN 段 + FFN(无全局支路,3.4.2)。"""

    def __init__(self, d: int, ffn_mult: int = 2) -> None:
        super().__init__()
        h = d * ffn_mult
        self.ln_mp = nn.LayerNorm(d)
        self.ln_e = nn.LayerNorm(d)
        self.msg = _mlp(3 * d, d, h)
        self.upd = _mlp(2 * d, d, h)
        self.edge = _mlp(3 * d, d, h)
        self.ln_ff = nn.LayerNorm(d)
        self.ffn = _mlp(d, d, h)

    def forward(self, h, e, edge_index):
        src, dst = edge_index[0], edge_index[1]
        hln = self.ln_mp(h)
        eln = self.ln_e(e)
        m = self.msg(torch.cat([hln[src], hln[dst], eln], dim=-1))
        agg = scatter_mean(m, dst, h.size(0))
        h = h + self.upd(torch.cat([hln, agg], dim=-1))
        e = e + self.edge(torch.cat([eln, hln[src], hln[dst]], dim=-1))
        h = h + self.ffn(self.ln_ff(h))
        return h, e


class ReadoutDecoder(nn.Module):
    def __init__(self, d: int, n_layers: int = 2, ffn_mult: int = 2,
                 hidden: int = None) -> None:
        super().__init__()
        hidden = hidden or 2 * d
        self.layers = nn.ModuleList(ReadoutMPNNLayer(d, ffn_mult) for _ in range(n_layers))
        self.heads = nn.ModuleDict()
        for g in MASKABLE_GROUPS:
            s, e = GROUP_RANGES[g]
            self.heads[g.value] = nn.Sequential(
                nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, e - s))

    def forward(self, h: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor,
                total_cols: int = None) -> torch.Tensor:
        """h[N,d] 节点 token(生成期=扩散产物)、e[2M,d] 支路 token(c_topo 编码)、
        edge_index[2,2M](对称化)。返回 [N,total_cols],仅填可遮蔽组列,META 列 0。"""
        total_cols = total_cols or NODE_TABLE.total_dim
        for layer in self.layers:
            h, e = layer(h, e, edge_index)
        out = h.new_zeros((h.size(0), total_cols))
        for g in MASKABLE_GROUPS:
            s, ee = GROUP_RANGES[g]
            out[:, s:ee] = self.heads[g.value](h)
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
