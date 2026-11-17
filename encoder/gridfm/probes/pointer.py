"""§8.3-⑤ 指针探针(P1-全量门禁第五判据,补实现;交接债①)。

判据:冻结编码器 + 轻量指针头按目录 ID 选中指定孪生设备,留出方式上 ≥99%——
token 几何可检索不等于下游头可用,此项直接测生产能力(方案 4.3 输出寻址)。

"目录 ID" 的可运输载体 = 持久化随机标签(3.2-3;目录页 4.4 维护 名称↔行号↔标签
的对应,神经网络内部从不出现绝对名字,红线 2):指针头把目标设备的标签映射为查询
向量,与全部节点 token 点积打分(4.3 指针原语),softmax 分布定义在样本自身 token
集上——规模无关;非候选(NULL/孤立)节点打分时 mask(4.3)。

探针纪律:头只在**训练方式**上拟合,门禁只看**留出方式 + 留出锚点**;头是一次性
探针产物,不入库(编码器冻结不动,D11/D13)。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..eval.twins import _encode
from ..schema.fields import NODE_TABLE
from ..schema.sample import GridSample


def _candidate_mask(sample: GridSample) -> np.ndarray:
    """指针候选 = 非 NULL 且有连接(双端计度;8.2 的 _valid_node_mask 只数源端,
    属冻结门禁协议不动,指针探针为新协议自带口径)。"""
    is_null = sample.node_feat[:, NODE_TABLE.offset_of("is_null")] > 0.5
    deg = np.bincount(sample.edge_index.ravel(), minlength=sample.n_nodes)
    return (~is_null) & (deg > 0)


def encode_token_set(encoder, sample: GridSample, twins: Tuple[int, int],
                     random_dim: int = 8, branch_random_dim: int = 8,
                     lape_k: int = 16, rwse_steps: int = 16) -> Dict:
    """一份探针数据 = 归一化节点 token + 持久化标签 + 有效位 + 孪生对行号。"""
    encoder.eval()
    tokens, _ = _encode(encoder, sample, lape_k, rwse_steps, random_dim, branch_random_dim)
    return {
        "tokens": tokens.detach(),
        "tags": torch.from_numpy(sample.random_features(random_dim).copy()),
        "valid": torch.from_numpy(_candidate_mask(sample)),
        "twins": tuple(twins),
        "sample_id": sample.sample_id,
    }


class PointerHead(torch.nn.Module):
    """轻量指针头(4.3 指针原语,键投影 + L2 计分):score = −‖K·h − tag‖²。

    展开即 2·tag·(K·h) − ‖K·h‖² − ‖tag‖²  = 查询·键点积 + 逐候选范数偏置(查询侧常数
    不影响 argmax)——标准注意力的键投影形态。裸内积计分偏爱大范数标签,v1 实测同样本
    命中率仅 0.57,L2 计分 1.00(2026-07-11 标定,只在 val 方式上)。参数 = d×tag_dim+tag_dim。
    """

    def __init__(self, tag_dim: int, d: int) -> None:
        super().__init__()
        self.key = torch.nn.Linear(d, tag_dim)

    def scores(self, tags: torch.Tensor, tokens: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
        s = -torch.cdist(tags, self.key(tokens)) ** 2    # [B, N]
        return s.masked_fill(~valid.unsqueeze(0), float("-inf"))


def pointer_probe(train_sets: List[Dict], eval_sets: List[Dict],
                  steps: int = 200, lr: float = 1e-3, seed: int = 0,
                  head: Optional[PointerHead] = None) -> Dict:
    """训练(可选)+ 评估指针头。

    训练:每个训练集的全部有效节点作目标(query=自身标签,label=自身行号),
    交叉熵全批迭代——孪生行也在其中,头必须学会读身份通道。
    评估:留出方式上 (a) 孪生行号选中率(门禁,≥99%,并列计 argmax 首个即错方向)
    与 (b) 全节点寻址率(诊断);head 传入时跳过训练(零样本迁移诊断用)。
    """
    if head is None:
        if not train_sets:
            raise ValueError("无训练集且未提供已训头")
        torch.manual_seed(seed)
        d = train_sets[0]["tokens"].shape[1]
        tag_dim = train_sets[0]["tags"].shape[1]
        head = PointerHead(tag_dim, d)
        # 闭式初始化:训练集上池化岭回归 W(token→tag),键投影 = Wᵀ——读出方向已由
        # §8.3-① 证明线性可解,随机初始化的 CE 找不到这一解(实测卡 0.45);
        # 初始化只用训练方式,协议合规。CE 精调在其上进行(解已近最优时梯度≈0,安全)。
        X = torch.cat([ts["tokens"][ts["valid"]] for ts in train_sets]).double().numpy()
        Y = torch.cat([ts["tags"][ts["valid"]] for ts in train_sets]).double().numpy()
        A = X.T @ X + 1e-3 * np.eye(X.shape[1])
        W = np.linalg.solve(A, X.T @ Y)                    # [d, tag_dim]
        with torch.no_grad():
            head.key.weight.copy_(torch.from_numpy(W.T).float())
            head.key.bias.zero_()
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.zeros(())
            for ts in train_sets:
                rows = torch.where(ts["valid"])[0]
                s = head.scores(ts["tags"][rows], ts["tokens"], ts["valid"])
                loss = loss + torch.nn.functional.cross_entropy(s, rows)
            (loss / len(train_sets)).backward()
            opt.step()
    head.eval()

    twin_hits: List[bool] = []
    all_hits: List[bool] = []
    chances: List[float] = []
    with torch.no_grad():
        for ts in eval_sets:
            rows = torch.where(ts["valid"])[0]
            s = head.scores(ts["tags"][rows], ts["tokens"], ts["valid"])
            pred = s.argmax(dim=1)
            hit = (pred == rows)
            all_hits.extend(hit.tolist())
            chances.append(1.0 / max(1, int(ts["valid"].sum())))
            row_pos = {int(r): i for i, r in enumerate(rows.tolist())}
            for t in ts["twins"]:
                twin_hits.append(bool(hit[row_pos[int(t)]]))
    twin_acc = float(np.mean(twin_hits)) if twin_hits else float("nan")
    return {
        "twin_pointer_acc": twin_acc,
        "pointer_ok": bool(twin_hits and twin_acc >= 0.99),
        "all_node_acc": float(np.mean(all_hits)) if all_hits else float("nan"),
        "n_twin_targets": len(twin_hits), "n_eval_sets": len(eval_sets),
        "chance": float(np.mean(chances)) if chances else float("nan"),
        "head": head,
    }
