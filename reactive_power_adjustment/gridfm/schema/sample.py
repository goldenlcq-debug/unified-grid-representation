"""GridSample:档案的样本级容器(numpy 存储,契合 .npz;方案 7.5 / 2.2)。

missing 位 = 数据本身缺失;masked 位 = 训练人为遮蔽。两者形状均为 [N/M, 4]
(对应 MASKABLE_GROUPS 四组),语义独立、通道分离(D15)。

随机特征 node_random / branch_random(方案 3.2 第 3 条,branch 扩展经 2026-07-09 FABLE 评审):
每样本每节点/每支路 8 维,**样本生成时抽取一次并持久化**,用于打破 LaPE/RWSE 消不掉的残余
图对称(节点孪生 / 并联双回同型线路——后者两条支路输入逐字节相同,支路随机特征是唯一破缺手段)。
为兼容既有 .npz(无此字段)并保证跨机可复现,缺省时按 blake2b 稳定种子确定性派生——
等价于"固定种子抽一次"。同一 (family_id, sample_id) 重装配逐位复现同一批随机特征(生成闭环
"改卡→重解→重装配"天然保持孪生对应)。约定:token 嵌入是**样本内**标识;跨样本寻址一律走
目录页 catalog ID(4.4),不得依赖嵌入对应。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .fields import MASKABLE_GROUPS, Group

RANDOM_FEAT_DIM = 8


def stable_seed(*parts: str) -> int:
    """跨进程/跨平台稳定的种子(禁用 python hash():带随机盐)。"""
    key = "/".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")

N_MASK_GROUPS = len(MASKABLE_GROUPS)
_GROUP_INDEX: Dict[Group, int] = {g: i for i, g in enumerate(MASKABLE_GROUPS)}


def group_bit_index(group: Group) -> int:
    return _GROUP_INDEX[group]


@dataclass
class GridSample:
    # 特征矩阵(原始值,归一化在 T1.6 施加)
    node_feat: np.ndarray            # [N, Fn] float32
    branch_feat: np.ndarray          # [M, Fb] float32
    edge_index: np.ndarray           # [2, M] int64,0-based (bus_no-1)
    # 缺失/遮蔽位(四组)
    node_missing: np.ndarray         # [N, 4] bool
    node_masked: np.ndarray          # [N, 4] bool
    branch_missing: np.ndarray       # [M, 4] bool
    branch_masked: np.ndarray        # [M, 4] bool
    # 随机特征(方案 3.2-3):[N, 8]/[M, 8] float32,持久化;None = 按稳定种子惰性派生
    node_random: Optional[np.ndarray] = None
    branch_random: Optional[np.ndarray] = None
    # 元信息
    sample_id: str = ""
    family_id: str = ""
    meta: Dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.node_feat.shape[0]

    @property
    def n_branches(self) -> int:
        return self.branch_feat.shape[0]

    def set_node_masked(self, group: Group, node_rows, value: bool = True) -> None:
        self.node_masked[node_rows, group_bit_index(group)] = value

    def set_node_missing(self, group: Group, node_rows, value: bool = True) -> None:
        self.node_missing[node_rows, group_bit_index(group)] = value

    def random_features(self, dim: int = RANDOM_FEAT_DIM) -> np.ndarray:
        """持久化节点随机特征 [N, dim](方案 3.2-3)。缺失或维度不符时按稳定种子确定性派生并缓存。

        推理/缓存期一律用本函数返回值(7.4 逐位一致);训练期可在其之上做重采样增广。
        """
        if dim <= 0:
            return np.zeros((self.n_nodes, 0), dtype=np.float32)
        cur = self.node_random
        if cur is not None and cur.shape == (self.n_nodes, dim):
            return cur
        rng = np.random.default_rng(stable_seed(self.family_id, self.sample_id))
        self.node_random = rng.standard_normal((self.n_nodes, dim)).astype(np.float32)
        return self.node_random

    def branch_random_features(self, dim: int = RANDOM_FEAT_DIM) -> np.ndarray:
        """持久化支路随机特征 [M, dim]。独立随机流("branch" 盐),与节点流不相关。

        并联双回同型线路的两条支路除本特征外输入逐字节相同——这是 token 级区分
        "1 号回路/2 号回路"的唯一手段(2026-07-09 评审)。
        """
        if dim <= 0:
            return np.zeros((self.n_branches, 0), dtype=np.float32)
        cur = self.branch_random
        if cur is not None and cur.shape == (self.n_branches, dim):
            return cur
        rng = np.random.default_rng(stable_seed(self.family_id, self.sample_id, "branch"))
        self.branch_random = rng.standard_normal((self.n_branches, dim)).astype(np.float32)
        return self.branch_random

    def save_npz(self, path) -> None:
        # schema_version 键(v0.3/S1 起):按节点宽度如实标注;旧读取方按键名取数组,多一键无害
        from .fields import NODE_TABLE, NODE_TABLE_V03
        w = self.node_feat.shape[1]
        ver = "v0.3" if w == NODE_TABLE_V03.total_dim else \
              "v0.2" if w == NODE_TABLE.total_dim else f"unknown_w{w}"
        np.savez_compressed(
            path,
            node_feat=self.node_feat, branch_feat=self.branch_feat, edge_index=self.edge_index,
            node_missing=self.node_missing, node_masked=self.node_masked,
            branch_missing=self.branch_missing, branch_masked=self.branch_masked,
            node_random=self.random_features(), branch_random=self.branch_random_features(),
            sample_id=self.sample_id, family_id=self.family_id,
            schema_version=ver,
        )

    @classmethod
    def load_npz(cls, path, full_schema: bool = False) -> "GridSample":
        """默认返回 **v0 列视图**(25 节点列)——v1 栈全部消费方零改动、逐位不变
        (数据级规则开关,架构规则 5)。v2 栈显式 full_schema=True 取全量 v0.3 列。
        旧 25 宽 npz 两种调用均原样返回;未知宽度直接报错(防半截 schema 静默入训)。
        """
        from .fields import NODE_TABLE, NODE_TABLE_V03, V03_TO_V0_COLS
        z = np.load(path, allow_pickle=False)
        nf = z["node_feat"]
        w = nf.shape[1]
        if w == NODE_TABLE_V03.total_dim and not full_schema:
            nf = np.ascontiguousarray(nf[:, V03_TO_V0_COLS])
        elif w not in (NODE_TABLE.total_dim, NODE_TABLE_V03.total_dim):
            raise ValueError(f"{path}: 未知 node_feat 宽度 {w}"
                             f"(既非 v0 {NODE_TABLE.total_dim} 也非 v0.3 {NODE_TABLE_V03.total_dim})")
        return cls(
            node_feat=nf, branch_feat=z["branch_feat"], edge_index=z["edge_index"],
            node_missing=z["node_missing"], node_masked=z["node_masked"],
            branch_missing=z["branch_missing"], branch_masked=z["branch_masked"],
            node_random=z["node_random"] if "node_random" in z.files else None,
            branch_random=z["branch_random"] if "branch_random" in z.files else None,
            sample_id=str(z["sample_id"]), family_id=str(z["family_id"]),
        )
