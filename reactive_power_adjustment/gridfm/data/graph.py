"""GridSample → PyG Data(方案 T5.1)。输入节点特征 = 档案节点特征 ⊕ LaPE ⊕ 有效位 ⊕ RWSE。

边对称化(每条支路加正反向,同一 edge_attr),供 MPNN 无向消息传递。
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import torch
from torch_geometric.data import Data

from ..schema.sample import GridSample
from pathlib import Path

from .posenc import compute_lape, compute_rwse


# 拓扑级 posenc 缓存(方案 3.2/7.5:数据生成阶段按拓扑算一次)。
# 同拓扑的所有方式样本共享 LaPE/RWSE(仅状态不同)→ 训练时命中缓存,消除每步重算。
#
# 容量必须 ≥ 家族总数,否则多家族混训每步随机采样跨家族 → 逐出即重算,缓存等于失效。
# (P1-全量规划:118 + CEPRI-36 + 60~80 合成家族 + 10 留出家族 ≈ 90 个拓扑。)
# 单拓扑占用 ≈ N×(16+16)×4B + N×16B ≈ 150KB @ N=1000 → 512 档约 75MB,可接受。
_POSENC_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_POSENC_CACHE_MAX = int(os.environ.get("GRIDFM_POSENC_CACHE", "512"))


# 2026-09-05（N-k 其他推理器线）：可选的磁盘级 posenc 缓存，环境变量 GRIDFM_POSENC_DISK_CACHE=<目录> 时启用。
# 动因：N-1/N-2 考卷每个病例拓扑都不同，进程内缓存对跨进程（诊断 → 跑分）无效，70k 每次 LaPE 约 6 分钟。
# 缓存内容 = 首次计算的同一组数组（逐位相同），不设该变量时行为与原实现逐位一致。
_POSENC_DISK = os.environ.get("GRIDFM_POSENC_DISK_CACHE")


def _disk_key(n: int, lape_k: int, rwse_steps: int, edge_index: np.ndarray) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(f"{n}|{lape_k}|{rwse_steps}|{edge_index.dtype}|{edge_index.shape}".encode()); h.update(edge_index.tobytes())
    return h.hexdigest()


def _posenc(edge_index: np.ndarray, n: int, lape_k: int, rwse_steps: int):
    key = (n, lape_k, rwse_steps, edge_index.tobytes())
    hit = _POSENC_CACHE.get(key)
    if hit is not None:
        _POSENC_CACHE.move_to_end(key)          # LRU:命中即刷新
        return hit
    dpath = None
    if _POSENC_DISK:
        dpath = Path(_POSENC_DISK) / (_disk_key(n, lape_k, rwse_steps, edge_index) + ".npz")
        if dpath.is_file():
            try:
                z = np.load(dpath, allow_pickle=False)
                hit = (z["lape"], z["valid"], z["rwse"])
                while len(_POSENC_CACHE) >= _POSENC_CACHE_MAX:
                    _POSENC_CACHE.popitem(last=False)
                _POSENC_CACHE[key] = hit
                return hit
            except Exception:                   # 损坏/半写文件：当未命中，重算并覆盖
                pass
    lape, valid, _comp = compute_lape(edge_index, n, k=lape_k)
    rwse = compute_rwse(edge_index, n, steps=rwse_steps)
    if dpath is not None:
        try:
            dpath.parent.mkdir(parents=True, exist_ok=True)
            tmp = dpath.with_name(dpath.name + f".tmp{os.getpid()}")
            np.savez(tmp, lape=lape, valid=valid, rwse=rwse)
            os.replace(str(tmp) + (".npz" if not str(tmp).endswith(".npz") else ""), dpath)
        except Exception:
            pass
    while len(_POSENC_CACHE) >= _POSENC_CACHE_MAX:
        _POSENC_CACHE.popitem(last=False)       # 逐出最久未用
    _POSENC_CACHE[key] = (lape, valid, rwse)
    return _POSENC_CACHE[key]


def feature_slices(node_feat_dim: int, lape_k: int = 16, rwse_steps: int = 16,
                   random_dim: int = 8) -> dict:
    """输入特征 x 的列块切片:feat | lape | valid | rwse | random(与 sample_to_data 拼接顺序一致)。"""
    o = node_feat_dim
    sl = {"feat": slice(0, o)}
    sl["lape"] = slice(o, o + lape_k); o += lape_k
    sl["valid"] = slice(o, o + lape_k); o += lape_k
    sl["rwse"] = slice(o, o + rwse_steps); o += rwse_steps
    sl["random"] = slice(o, o + random_dim)
    return sl


def sample_to_data(sample: GridSample, lape_k: int = 16, rwse_steps: int = 16,
                   random_dim: int = 8, random_feat: np.ndarray = None,
                   branch_random_dim: int = 8, branch_random_feat: np.ndarray = None) -> Data:
    n = sample.n_nodes
    lape, valid, rwse = _posenc(sample.edge_index, n, lape_k, rwse_steps)
    if random_feat is None:
        # 推理/缓存期:用样本持久化随机特征(方案 3.2-3 / 7.4 逐位一致)
        rf = sample.random_features(random_dim)
    else:
        rf = random_feat.astype(np.float32)
    if branch_random_feat is None:
        brf = sample.branch_random_features(branch_random_dim)
    else:
        brf = branch_random_feat.astype(np.float32)

    x = np.concatenate(
        [sample.node_feat, lape, valid.astype(np.float32), rwse, rf], axis=1
    ).astype(np.float32)

    ei = sample.edge_index
    # 对称化:正反向边 + 支路特征各复制一份。支路随机特征在复制前拼接,
    # 保证同一物理回路的正反两条有向边共享同一随机标签(是"回路身份",非"方向身份")。
    ei_sym = np.concatenate([ei, ei[::-1]], axis=1)
    ea = np.concatenate([sample.branch_feat, brf], axis=1).astype(np.float32)
    eattr = np.concatenate([ea, ea], axis=0)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(ei_sym.astype(np.int64)),
        edge_attr=torch.from_numpy(eattr),
    )
    data.num_nodes = n
    return data


def node_input_dim(node_feat_dim: int, lape_k: int = 16, rwse_steps: int = 16,
                   random_dim: int = 8) -> int:
    return node_feat_dim + lape_k + lape_k + rwse_steps + random_dim  # feat+lape+valid+rwse+rand


def edge_input_dim(branch_feat_dim: int, branch_random_dim: int = 8) -> int:
    return branch_feat_dim + branch_random_dim  # feat+rand(回路身份标签,2026-07-09 评审)
