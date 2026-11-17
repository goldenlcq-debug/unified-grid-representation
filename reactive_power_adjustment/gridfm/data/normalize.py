"""规范化(方案 7.4 / T1.6)。fit-transform-inverse 三段,随 schema 版本固化,禁训练/推理不一致。

按字段 norm 码处理:
- onehot / bool / raw:透传;
- rad:透传(θ 用弧度、不 z-score);
- z / pu_z:标幺后按训练集统计 z-score(标幺换算在解析时已按 100MVA 基准完成);
- log_z:先 sign(x)*log1p(|x|),再 z-score(长尾字段如 R/X)。

统计按列进行,排除该列所属通道组被标记 missing 的行(避免结构性零污染统计)。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..schema.fields import BRANCH_TABLE, FieldTable, NODE_TABLE, SCHEMA_VERSION
from ..schema.sample import GridSample, group_bit_index

_ZTYPES = {"z", "pu_z", "log_z"}


def _expand_norm(table: FieldTable):
    """返回逐列 (norm_code, group) 列表。"""
    cols = []
    for f in table.fields:
        cols.extend([(f.norm, f.group)] * f.dim)
    return cols


def _pre(x: np.ndarray, norm: str) -> np.ndarray:
    if norm == "log_z":
        return np.sign(x) * np.log1p(np.abs(x))
    return x


def _post_inv(x: np.ndarray, norm: str) -> np.ndarray:
    if norm == "log_z":
        return np.sign(x) * (np.expm1(np.abs(x)))
    return x


class Normalizer:
    def __init__(self) -> None:
        self.version = SCHEMA_VERSION
        self.node_mean: np.ndarray = None
        self.node_std: np.ndarray = None
        self.branch_mean: np.ndarray = None
        self.branch_std: np.ndarray = None
        self._node_cols = _expand_norm(NODE_TABLE)
        self._branch_cols = _expand_norm(BRANCH_TABLE)

    def _fit_one(self, feats: List[np.ndarray], missings: List[np.ndarray], cols):
        F = len(cols)
        for feat in feats:      # A1(2026-08-06):v0.3 43 列喂 v1 归一化器会从 DYN_EXT 插入点起
            assert feat.shape[1] == F, \
                f"特征 {feat.shape[1]} 列与归一化表 {F} 列不符(v0.3 样本喂 v1 Normalizer?静默错位)"
        mean = np.zeros(F, dtype=np.float64)
        std = np.ones(F, dtype=np.float64)
        for c, (norm, group) in enumerate(cols):
            if norm not in _ZTYPES:
                continue
            gbit = group_bit_index(group) if group.value in ("topology", "static", "dynamic", "state") else None
            vals = []
            for feat, miss in zip(feats, missings):
                col = _pre(feat[:, c].astype(np.float64), norm)
                if gbit is not None:
                    keep = ~miss[:, gbit]
                    col = col[keep]
                vals.append(col)
            allv = np.concatenate(vals) if vals else np.array([0.0])
            if allv.size > 0:
                mean[c] = allv.mean()
                s = allv.std()
                std[c] = s if s > 1e-8 else 1.0
        return mean, std

    def fit(self, samples: List[GridSample]) -> "Normalizer":
        self.node_mean, self.node_std = self._fit_one(
            [s.node_feat for s in samples], [s.node_missing for s in samples], self._node_cols)
        self.branch_mean, self.branch_std = self._fit_one(
            [s.branch_feat for s in samples], [s.branch_missing for s in samples], self._branch_cols)
        return self

    def _apply(self, feat, cols, mean, std, inverse=False):
        assert feat.shape[1] == len(cols), \
            f"特征 {feat.shape[1]} 列与归一化表 {len(cols)} 列不符(v0.3 样本喂 v1 Normalizer?静默错位)"
        out = feat.astype(np.float64).copy()
        for c, (norm, _g) in enumerate(cols):
            if norm not in _ZTYPES:
                continue
            if inverse:
                out[:, c] = _post_inv(out[:, c] * std[c] + mean[c], norm)
            else:
                out[:, c] = (_pre(out[:, c], norm) - mean[c]) / std[c]
        return out.astype(np.float32)

    def transform(self, sample: GridSample) -> GridSample:
        sample.node_feat = self._apply(sample.node_feat, self._node_cols, self.node_mean, self.node_std)
        sample.branch_feat = self._apply(sample.branch_feat, self._branch_cols, self.branch_mean, self.branch_std)
        return sample

    def inverse_transform_node(self, node_feat: np.ndarray) -> np.ndarray:
        return self._apply(node_feat, self._node_cols, self.node_mean, self.node_std, inverse=True)

    def save(self, path) -> None:
        np.savez(path, version=self.version, node_mean=self.node_mean, node_std=self.node_std,
                 branch_mean=self.branch_mean, branch_std=self.branch_std)

    @classmethod
    def load(cls, path) -> "Normalizer":
        z = np.load(path, allow_pickle=False)
        o = cls()
        o.version = str(z["version"])
        o.node_mean, o.node_std = z["node_mean"], z["node_std"]
        o.branch_mean, o.branch_std = z["branch_mean"], z["branch_std"]
        return o
