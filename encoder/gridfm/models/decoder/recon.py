"""重构解码器(方案 3.4.2)。轻量(2 层)、按通道组分头、只消费逐条目 token(不依赖全局 token)。

解码器只见编码器输出 token h(不见原始特征)→ 天然 re-mask,无法绕过编码器偷看邻居原始字段。
损失:连续量 Huber、类别/布尔量 BCE-with-logits,只在被遮蔽位置计,按通道组加权。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...schema.fields import MASKABLE_GROUPS, NODE_TABLE, Group


def _group_col_ranges() -> Dict[Group, Tuple[int, int]]:
    """每个 maskable 组的连续列区间 [start, end)。"""
    ranges: Dict[Group, Tuple[int, int]] = {}
    off = 0
    for f in NODE_TABLE.fields:
        if f.group in MASKABLE_GROUPS:
            s, e = ranges.get(f.group, (off, off))
            ranges[f.group] = (min(s, off), off + f.dim)
        off += f.dim
    return ranges


def _col_loss_is_bce() -> List[bool]:
    """逐列:onehot/bool 用 BCE,其余用 Huber。"""
    flags: List[bool] = []
    for f in NODE_TABLE.fields:
        flags.extend([f.norm in ("onehot", "bool")] * f.dim)
    return flags


GROUP_RANGES = _group_col_ranges()
COL_IS_BCE = _col_loss_is_bce()


class ReconDecoder(nn.Module):
    def __init__(self, d: int, hidden: int = None) -> None:
        super().__init__()
        hidden = hidden or 2 * d
        self.heads = nn.ModuleDict()
        for g in MASKABLE_GROUPS:
            s, e = GROUP_RANGES[g]
            self.heads[g.value] = nn.Sequential(
                nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, e - s)
            )

    def forward(self, h: torch.Tensor, total_cols: int) -> torch.Tensor:
        """h:[N,d] → 重构 [N, total_cols](仅填充可遮蔽组的列,META 列留 0)。"""
        out = h.new_zeros((h.size(0), total_cols))
        for g in MASKABLE_GROUPS:
            s, e = GROUP_RANGES[g]
            out[:, s:e] = self.heads[g.value](h)
        return out


def recon_loss(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor,
               group_weight: Dict[str, float]) -> torch.Tensor:
    """只在 loss_mask 位置计损失,逐列 Huber/BCE,按通道组加权。"""
    device = pred.device
    bce_col = torch.tensor(COL_IS_BCE, device=device)
    # 逐列权重(按组)
    col_w = torch.ones(pred.size(1), device=device)
    for g in MASKABLE_GROUPS:
        s, e = GROUP_RANGES[g]
        col_w[s:e] = group_weight.get(g.value, 1.0)

    total = pred.new_tensor(0.0)
    count = pred.new_tensor(0.0)
    for c in range(pred.size(1)):
        m = loss_mask[:, c]
        if not torch.any(m):
            continue
        p, t = pred[m, c], target[m, c]
        if bce_col[c]:
            l = F.binary_cross_entropy_with_logits(p, t, reduction="sum")
        else:
            l = F.huber_loss(p, t, delta=1.0, reduction="sum")
        total = total + col_w[c] * l
        count = count + m.sum()
    return total / count.clamp(min=1.0)
