"""扩散去噪网络 v0 + 潜空间回归对照(6.2/6.4;2026-07-17 用户批准的账)。

架构口径:
- **AdaLN-MPNN 层**(GPS 式去全局):MPNN 段 + FFN 形状同 ReadoutMPNNLayer;
  t 正弦嵌入经 MLP 与 c_label 嵌入合并,走 AdaLN 调制(DiT 做法),采用
  **adaLN-Zero 初始化**(调制层与输出头零初始化,起点 = 恒等/零输出,小数据稳定);
- **c_topo 注入 = 拼接**(6.2:与 z 同形的逐 token 场,按行对齐沿特征维拼进输入
  投影,永不丢弃);支路 token e_ct 作边上下文(内部隐状态,不出 ε);
- **z = 节点 token 场**(用户裁决;标准化空间,latent_stats_v1),ε 头只出节点侧;
- **c_label v0** = load_factor 档 one-hot(3) + 缺失位(1) + 4 列预留 = 8 维;
  整体丢弃即缺失位置 1、one-hot 清零(CFG 无条件分支同款)。启停注记(2026-07-17
  审查更正):本库 has_gen/支路 mark 恒定,但**每方式有 2~5 台机组 pg≈0,开机组合
  逐方式变化**——启停并非退化量;v0 将其并入连续 pg 场生成(pg≈0 即近停机),
  6.3② 两阶段(启停分类头 + 0/1 通道拼入 c_topo)列为 v0.1 后续项,不是不需要。
- **LatentRegressorV0(单变量对照,6.6 第二档)**:同一网络类,z_t≡0、t≡0,
  确定性回归 z——与扩散唯一差异 = 有无扩散采样,归因干净。

2.4 四红线自查:①逐 token/逐边算子,无 N 依赖 ✓;②无任务参数进编码器(编码器
只读)✓;③PSASP 唯一裁决不变 ✓;④跨规模可复用 ✓。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..encoder.gps import _mlp, scatter_mean

LABEL_DIM = 8          # lf 档 one-hot(3) + 缺失位(1) + 预留(4)
LF_TIERS = (0.95, 1.05, 1.15)
LABEL_MISSING_BIT = 3


def label_vec(tier_idx: int | None, device=None) -> torch.Tensor:
    """tier_idx ∈ {0,1,2} 或 None(=整体缺失/CFG 无条件分支)。"""
    v = torch.zeros(LABEL_DIM, device=device)
    if tier_idx is None:
        v[LABEL_MISSING_BIT] = 1.0
    else:
        v[tier_idx] = 1.0
    return v


class AdaLNMPNNLayer(nn.Module):
    """MPNN 段 + FFN(同 ReadoutMPNNLayer 形状)+ AdaLN-Zero 调制。"""

    def __init__(self, d: int, ffn_mult: int = 2) -> None:
        super().__init__()
        h = d * ffn_mult
        self.ln_mp = nn.LayerNorm(d, elementwise_affine=False)
        self.ln_e = nn.LayerNorm(d)
        self.msg = _mlp(3 * d, d, h)
        self.upd = _mlp(2 * d, d, h)
        self.edge = _mlp(3 * d, d, h)
        self.ln_ff = nn.LayerNorm(d, elementwise_affine=False)
        self.ffn = _mlp(d, d, h)
        self.mod = nn.Linear(d, 6 * d)     # scale/shift/gate × (MPNN, FFN)
        nn.init.zeros_(self.mod.weight)    # adaLN-Zero:gate=0 → 起点恒等
        nn.init.zeros_(self.mod.bias)

    def forward(self, h, e, edge_index, cond):
        s1, b1, g1, s2, b2, g2 = self.mod(cond).chunk(6, dim=-1)
        src, dst = edge_index[0], edge_index[1]
        hln = self.ln_mp(h) * (1 + s1) + b1
        eln = self.ln_e(e)
        m = self.msg(torch.cat([hln[src], hln[dst], eln], dim=-1))
        agg = scatter_mean(m, dst, h.size(0))
        h = h + g1 * self.upd(torch.cat([hln, agg], dim=-1))
        e = e + self.edge(torch.cat([eln, hln[src], hln[dst]], dim=-1))
        h = h + g2 * self.ffn(self.ln_ff(h) * (1 + s2) + b2)
        return h, e


class DenoiserV0(nn.Module):
    """ε̂(z_t, t, c_topo, c_label):z_t 与 c_topo 节点 token 拼接进投影;
    t+label 走 AdaLN;e_ct 作边上下文。输出 [N,d] 节点场 ε(零初始化头)。"""

    def __init__(self, d: int = 192, n_layers: int = 8,
                 label_dim: int = LABEL_DIM) -> None:
        super().__init__()
        self.d = d
        self.in_proj = nn.Linear(2 * d, d)
        self.t_mlp = _mlp(d, d, 2 * d)
        self.label_emb = nn.Linear(label_dim, d)
        self.layers = nn.ModuleList(AdaLNMPNNLayer(d) for _ in range(n_layers))
        self.out_ln = nn.LayerNorm(d, elementwise_affine=False)
        self.out_mod = nn.Linear(d, 2 * d)
        self.head = nn.Linear(d, d)
        for lin in (self.out_mod, self.head):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    @staticmethod
    def t_embed(t: torch.Tensor, d: int) -> torch.Tensor:
        half = d // 2
        freqs = torch.exp(-math.log(10000.0)
                          * torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, z_t, ct_node, e_ct, edge_index, t, label, nodes_per_graph):
        """z_t/ct_node [N,d]、e_ct [2M,d]、t [B] long、label [B,label_dim]、
        nodes_per_graph [B] long。"""
        cond_g = self.t_mlp(self.t_embed(t, self.d)) + self.label_emb(label)
        cond = cond_g.repeat_interleave(nodes_per_graph, dim=0)
        h = self.in_proj(torch.cat([z_t, ct_node], dim=-1))
        e = e_ct
        for layer in self.layers:
            h, e = layer(h, e, edge_index, cond)
        s, b = self.out_mod(cond).chunk(2, dim=-1)
        return self.head(self.out_ln(h) * (1 + s) + b)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LatentRegressorV0(DenoiserV0):
    """单变量对照(6.6 第二档):同类同参数量网络,z_t≡0、t≡0,确定性回归 z。
    与扩散生成器条件输入完全相同(c_topo 拼接 + c_label + e_ct),唯一差异 =
    确定性回归 vs 扩散采样。"""

    def forward(self, ct_node, e_ct, edge_index, label, nodes_per_graph):  # noqa: D102
        z0 = torch.zeros_like(ct_node)
        t = torch.zeros(nodes_per_graph.numel(), dtype=torch.long,
                        device=ct_node.device)
        return super().forward(z0, ct_node, e_ct, edge_index, t, label,
                               nodes_per_graph)
