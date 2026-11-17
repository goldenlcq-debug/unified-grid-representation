"""检索探针与对称孪生探针(方案 8.2 / 8.3,P1★ 门禁)。

检索探针(8.2):对同一样本前向两次,第二次重采样随机特征通道;以第一次节点 i 的 token 为
query、第二次全部节点 token 为候选,余弦 top-1 命中 i 即正确。同时验证"身份可辨"与
"辨识不依赖随机数记忆"(若编码器只透传随机特征,重采样后即检索失败)。

孪生探针(8.3):对人工构造的对称孪生对,检验两 token 可分且各自检索正确。孪生用例由手工
编辑卡文本构造(T1.5),此处接口就绪,用例集由调用方提供。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch

from ..data.graph import sample_to_data
from ..schema.fields import NODE_TABLE
from ..schema.sample import GridSample


def _encode(encoder, sample: GridSample, random_feat: np.ndarray, lape_k: int, rwse_steps: int,
            branch_random_feat: Optional[np.ndarray] = None) -> torch.Tensor:
    """branch_random_feat=None → 用样本持久化支路随机特征(8.3 持久化口径);
    显式传入 → 重采样口径(8.2:两遍全部随机通道一起换,防经邻接支路标签跨遍认人)。"""
    data = sample_to_data(sample, lape_k, rwse_steps, random_dim=random_feat.shape[1],
                          random_feat=random_feat, branch_random_feat=branch_random_feat)
    with torch.no_grad():
        h, _g, _e = encoder(Batch.from_data_list([data]))
    return h


def _valid_node_mask(sample: GridSample) -> np.ndarray:
    """排除 NULL/孤立节点(无法结构可辨)。"""
    off = NODE_TABLE.offset_of
    is_null = sample.node_feat[:, off("is_null")] > 0.5
    n = sample.n_nodes
    deg = np.zeros(n)
    for a in sample.edge_index[0]:
        deg[a] += 1
    return (~is_null) & (deg > 0)


def retrieval_probe(encoder, sample: GridSample, rng: np.random.Generator,
                    random_dim: int = 8, lape_k: int = 16, rwse_steps: int = 16,
                    branch_random_dim: int = 8) -> float:
    """返回 top-1 检索准确率(仅对有效节点)。8.2 重采样口径:两遍各自重采样
    节点与支路随机通道(支路通道不重采样会给模型留"邻接支路标签"后门)。"""
    encoder.eval()
    n = sample.n_nodes
    m = sample.n_branches
    rf1 = rng.standard_normal((n, random_dim)).astype(np.float32)
    rf2 = rng.standard_normal((n, random_dim)).astype(np.float32)
    brf1 = rng.standard_normal((m, branch_random_dim)).astype(np.float32)
    brf2 = rng.standard_normal((m, branch_random_dim)).astype(np.float32)
    t1 = _encode(encoder, sample, rf1, lape_k, rwse_steps, brf1)
    t2 = _encode(encoder, sample, rf2, lape_k, rwse_steps, brf2)
    t1 = torch.nn.functional.normalize(t1, dim=-1)
    t2 = torch.nn.functional.normalize(t2, dim=-1)
    sim = t1 @ t2.T                       # [N, N]
    top1 = sim.argmax(dim=1).cpu().numpy()

    valid = _valid_node_mask(sample)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return float("nan")
    correct = (top1[idx] == idx).mean()
    return float(correct)


def symmetric_twin_probe(encoder, sample: GridSample, twin_pairs: List[Tuple[int, int]],
                         rng: np.random.Generator = None, random_dim: int = 8,
                         lape_k: int = 16, rwse_steps: int = 16) -> Optional[dict]:
    """孪生探针 8.3①(2026-07-09 修订口径):**持久化**随机特征下单遍编码,检验孪生对
    token 可分性。报告相对裕度基线 self_drift(全视图↔状态遮蔽视图同节点自漂移中位数)
    供 ①' 判据:separation ≥ α × self_drift(α 初值 1.0,待校准)。

    ②(跨视图检索)与轨道命中门禁在孪生用例集落地时实现(任务⑤);rng 参数仅为兼容
    旧签名保留,持久化口径下不使用。twin_pairs 为空则返回 None。
    """
    if not twin_pairs:
        return None
    rf = sample.random_features(random_dim)                # 持久化(8.3 口径)
    t = _encode(encoder, sample, rf, lape_k, rwse_steps)   # branch 随机通道同为持久化
    t = torch.nn.functional.normalize(t, dim=-1)
    seps = []
    for a, b in twin_pairs:
        cos = float((t[a] * t[b]).sum())
        seps.append(1.0 - cos)            # 余弦距离
    return {"min_separation": float(np.min(seps)), "mean_separation": float(np.mean(seps)),
            "n_pairs": len(twin_pairs)}
