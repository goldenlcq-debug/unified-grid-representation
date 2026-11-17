"""GPS 式混合编码器(方案 3.1 / D4)。每层:MPNN(节点+边更新)→ 全局写 → 全局读 → FFN,Pre-LN。

规模无关:所有权重为 d×d 量级,作用于单 token/单边;N 只决定套用次数(2.3)。
按图隔离:MPNN 仅沿真实边(不跨图);全局注意力经 to_dense_batch 按图分行 + key_padding_mask
(不跨图)——保证 batch 前向 == 各图单独前向(跨图泄漏单测 test_encoder)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch_geometric.utils import to_dense_batch


def _mlp(din: int, dout: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, dout))


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """按 index 求均值聚合,native index_add(免 torch-scatter 扩展,D17)。"""
    out = src.new_zeros((n, src.size(-1)))
    out.index_add_(0, index, src)
    deg = src.new_zeros(n)
    deg.index_add_(0, index, torch.ones(index.size(0), dtype=src.dtype, device=src.device))
    return out / deg.clamp(min=1.0).unsqueeze(-1)


class GPSLayer(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, ffn_mult: int = 2) -> None:
        super().__init__()
        h = d * ffn_mult
        # MPNN
        self.ln_mp = nn.LayerNorm(d)
        self.ln_e = nn.LayerNorm(d)
        self.msg = _mlp(3 * d, d, h)
        self.upd = _mlp(2 * d, d, h)
        self.edge = _mlp(3 * d, d, h)
        # 全局写/读
        self.ln_gw = nn.LayerNorm(d)
        self.ln_gr = nn.LayerNorm(d)
        self.attn_w = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.attn_r = nn.MultiheadAttention(d, n_heads, batch_first=True)
        # FFN
        self.ln_ff = nn.LayerNorm(d)
        self.ffn = _mlp(d, d, h)

    def forward(self, h, e, edge_index, batch, g):
        src, dst = edge_index[0], edge_index[1]
        n = h.size(0)

        # --- MPNN(节点+边更新)---
        hln = self.ln_mp(h)
        eln = self.ln_e(e)
        m = self.msg(torch.cat([hln[src], hln[dst], eln], dim=-1))
        agg = scatter_mean(m, dst, n)
        h = h + self.upd(torch.cat([hln, agg], dim=-1))
        e = e + self.edge(torch.cat([eln, hln[src], hln[dst]], dim=-1))

        # --- 全局写:global 从本图节点聚合 ---
        hd, mask = to_dense_batch(self.ln_gw(h), batch)          # [B, Nmax, d], [B, Nmax]
        kpm = ~mask                                              # True = padding(屏蔽)
        gw, _ = self.attn_w(g, hd, hd, key_padding_mask=kpm)
        g = g + gw

        # --- 全局读:节点从更新后的 global 读取 ---
        hd2, mask2 = to_dense_batch(self.ln_gr(h), batch)
        gr, _ = self.attn_r(hd2, g, g)                          # query=节点(dense),key/val=global
        h = h + gr[mask2]                                       # 回散到 [n, d](丢弃 padding)

        # --- FFN ---
        h = h + self.ffn(self.ln_ff(h))
        return h, e, g


class GridEncoder(nn.Module):
    def __init__(self, node_in: int, edge_in: int, d: int = 96, n_layers: int = 5,
                 n_global: int = 12, n_heads: int = 4, ffn_mult: int = 2,
                 node_rand_dim: int = 0, edge_rand_dim: int = 0,
                 grad_checkpoint: bool = False) -> None:
        """node_rand_dim/edge_rand_dim > 0 时启用 D19-A2:输入**末 rand_dim 列**(随机身份通道,
        3.2-3 约定拼接在末端)走独立投影(bias=False)+ 可学习标量增益。独立张量使 AdamW 能按
        参数组把它排除出 weight decay(单 Linear 的列级排除做不到)——没有显式梯度(D19-A1)时
        decay 会把随机通道投影单调压向零(P1-全量实测 2.40→2.11)。默认 0 = 旧单投影结构,
        旧检查点 state_dict 原样兼容。

        grad_checkpoint(编码器 v2 A5 前置件,2026-08-09):True 时训练态逐层
        torch.utils.checkpoint(激活不驻留,反向重算;万节点单图显存件)。**默认 False =
        v1 路径逐位不变**;纯 Python 属性不进 state_dict,检查点兼容双向。只在
        `training and torch.is_grad_enabled()` 时生效——eval/no_grad 流(验证/门禁/
        推理)恒走原路径,逐位性由单测 test_grad_checkpoint 锁定。"""
        super().__init__()
        self.d = d
        self.n_global = n_global
        self.node_rand_dim = node_rand_dim
        self.edge_rand_dim = edge_rand_dim
        self.grad_checkpoint = grad_checkpoint
        self.node_proj = nn.Linear(node_in - node_rand_dim, d)
        self.edge_proj = nn.Linear(edge_in - edge_rand_dim, d)
        if node_rand_dim > 0:
            self.node_rand_proj = nn.Linear(node_rand_dim, d, bias=False)
            self.node_rand_gain = nn.Parameter(torch.ones(()))
        if edge_rand_dim > 0:
            self.edge_rand_proj = nn.Linear(edge_rand_dim, d, bias=False)
            self.edge_rand_gain = nn.Parameter(torch.ones(()))
        self.global_token = nn.Parameter(torch.randn(n_global, d) * 0.02)
        self.layers = nn.ModuleList([GPSLayer(d, n_heads, ffn_mult) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d)

    def no_decay_parameters(self):
        """D19-A2:应排除出 weight decay 的参数(随机通道投影 + 增益)。"""
        out = []
        if self.node_rand_dim > 0:
            out += [self.node_rand_proj.weight, self.node_rand_gain]
        if self.edge_rand_dim > 0:
            out += [self.edge_rand_proj.weight, self.edge_rand_gain]
        return out

    def forward(self, batch):
        """batch:PyG Batch(x, edge_index, edge_attr, batch)。返回 (node_tokens[N,d], global[B,K,d])。"""
        if self.node_rand_dim > 0:
            rd = self.node_rand_dim
            h = self.node_proj(batch.x[:, :-rd]) \
                + self.node_rand_gain * self.node_rand_proj(batch.x[:, -rd:])
        else:
            h = self.node_proj(batch.x)
        if self.edge_rand_dim > 0:
            rd = self.edge_rand_dim
            e = self.edge_proj(batch.edge_attr[:, :-rd]) \
                + self.edge_rand_gain * self.edge_rand_proj(batch.edge_attr[:, -rd:])
        else:
            e = self.edge_proj(batch.edge_attr)
        b = batch.batch
        num_graphs = int(b.max().item()) + 1 if b.numel() > 0 else 1
        g = self.global_token.unsqueeze(0).expand(num_graphs, -1, -1).contiguous()
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for layer in self.layers:
            if use_ckpt:
                h, e, g = torch.utils.checkpoint.checkpoint(
                    layer, h, e, batch.edge_index, b, g, use_reentrant=False)
            else:
                h, e, g = layer(h, e, batch.edge_index, b, g)
        return self.ln_out(h), g, e

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
