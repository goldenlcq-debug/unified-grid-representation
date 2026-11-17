"""随机标签重构辅助头(D19-A1,方案 v2.1 §3.4.3)。

从每个节点/支路 token 经线性头(v0 选线性,"线性或 2 层 MLP"取最轻)还原**该条目本次
前向输入所用的随机标签**,MSE 损失。这是随机身份通道的唯一梯度来源——掩码重构对它
零梯度(结构等价条目的重构目标相同),没有此项时 weight decay 把随机通道投影单调压向零、
孪生门禁 8.3 结构性不可满足(P1-全量实测)。

目标口径:训练期若启用随机特征重采样增广(方案 3.2),目标即本次重采样值;验证/推理期
增广关闭,目标即持久化标签(3.2-3)。两种情形下目标都等于**输入里给这个条目的随机向量**
——A1 训练的是"把身份通道原样运过编码器"的能力,与输入用哪份随机值无关。

训练期专用组件(同 ReconDecoder):不进冻结产物,档案任务无关性(3.4.3 设计原则)不受影响。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TagReconHead(nn.Module):
    def __init__(self, d: int, node_rand_dim: int = 8, edge_rand_dim: int = 8) -> None:
        super().__init__()
        self.node_head = nn.Linear(d, node_rand_dim) if node_rand_dim > 0 else None
        self.edge_head = nn.Linear(d, edge_rand_dim) if edge_rand_dim > 0 else None

    def loss(self, h: torch.Tensor, e: torch.Tensor,
             node_tag: torch.Tensor, edge_tag: torch.Tensor) -> torch.Tensor:
        """整体逐元素 MSE(节点 + 支路合并计数)。

        h:[N,d] 节点 token;e:[2M,d] 对称化支路 token(正反向共享同一回路标签,
        graph.sample_to_data 拼接约定);node_tag:[N,rd];edge_tag:[2M,brd]。
        """
        total = h.new_tensor(0.0)
        count = 0
        if self.node_head is not None and node_tag is not None and node_tag.numel():
            total = total + F.mse_loss(self.node_head(h), node_tag, reduction="sum")
            count += node_tag.numel()
        if self.edge_head is not None and edge_tag is not None and edge_tag.numel():
            total = total + F.mse_loss(self.edge_head(e), edge_tag, reduction="sum")
            count += edge_tag.numel()
        return total / max(count, 1)
