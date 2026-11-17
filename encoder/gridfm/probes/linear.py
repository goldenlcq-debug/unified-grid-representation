"""线性探针(方案 8.5 / T5.5)。冻结编码器的全局 token 线性预测物理量(网损/可解性),
对照手工特征基线。判据:显著优于手工特征与随机(交叉验证均值差 > 1 个标准差)。

numpy 闭式 ridge + K 折 CV(免 sklearn 依赖)。回归报 R²;可解性(需负类)报准确率占位。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Batch

from ..data.graph import sample_to_data
from ..schema.fields import NODE_TABLE
from ..schema.sample import GridSample


def ridge_fit_predict(Xtr, ytr, Xte, alpha=1.0):
    Xtr1 = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xte1 = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)
    d = Xtr1.shape[1]
    reg = alpha * np.eye(d); reg[-1, -1] = 0.0        # 不正则截距
    w = np.linalg.solve(Xtr1.T @ Xtr1 + reg, Xtr1.T @ ytr)
    return Xte1 @ w


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def ridge_cv_r2(X: np.ndarray, y: np.ndarray, folds: int = 5, alpha: float = 1.0,
                seed: int = 0) -> float:
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    # 标准化特征(按训练折统计更严谨,此处全局标准化足够作探针)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    r2s = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if len(te) == 0 or len(tr) < 2:
            continue
        pred = ridge_fit_predict(Xs[tr], y[tr], Xs[te], alpha)
        r2s.append(_r2(y[te], pred))
    return float(np.mean(r2s)) if r2s else 0.0


@torch.no_grad()
def extract_global_features(encoder, samples: List[GridSample], device="cpu",
                            lape_k: int = 16, rwse_steps: int = 16,
                            random_dim: int = 8, branch_random_dim: int = 8) -> np.ndarray:
    """冻结编码器 → 每样本 K×d 全局 token 拼接特征。维度参数须与检查点训练配置一致。"""
    encoder.eval()
    dev = torch.device(device)
    feats = []
    for s in samples:
        data = sample_to_data(s, lape_k, rwse_steps,
                              random_dim=random_dim, branch_random_dim=branch_random_dim)
        _h, g, _e = encoder(Batch.from_data_list([data]).to(dev))
        feats.append(g.reshape(-1).cpu().numpy())
    return np.stack(feats)


def hand_features(sample: GridSample) -> np.ndarray:
    """手工特征基线(约 20 维统计;从原始 node_feat 计算,探针前不应归一化)。"""
    off = NODE_TABLE.offset_of
    nf = sample.node_feat
    def col(name):
        return nf[:, off(name)]
    pg, qg, pl, ql = col("pg"), col("qg"), col("pl"), col("ql")
    v = col("v")
    return np.array([
        pg.sum(), qg.sum(), pl.sum(), ql.sum(),
        pg.sum() - pl.sum(),                        # 近似净注入
        col("has_gen").sum(), col("has_load").sum(),
        v.mean(), v.min(), v.max(), v.max() - v.min(),
        pg.max(), pl.max(), col("pmax").sum(),
    ], dtype=np.float64)


def linear_probe(encoder, enc_samples: List[GridSample], raw_samples: List[GridSample],
                 targets: Dict[str, np.ndarray], device="cpu",
                 random_dim: int = 8, branch_random_dim: int = 8) -> Dict[str, Dict[str, float]]:
    """对每个目标量,比较'全局 token 特征'与'手工特征'的 ridge CV R²。

    enc_samples:归一化样本(编码用,与训练一致);raw_samples:原始样本(手工特征/标签用)。"""
    gfeat = extract_global_features(encoder, enc_samples, device=device,
                                    random_dim=random_dim, branch_random_dim=branch_random_dim)
    hfeat = np.stack([hand_features(s) for s in raw_samples])
    out = {}
    for name, y in targets.items():
        y = np.asarray(y, dtype=np.float64)
        out[name] = {
            "global_r2": ridge_cv_r2(gfeat, y),
            "hand_r2": ridge_cv_r2(hfeat, y),
        }
    return out
