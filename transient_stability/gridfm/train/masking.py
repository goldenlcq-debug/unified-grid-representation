"""掩码课程(方案 3.4.1 / D15)。通道组级遮蔽:基础随机 + 结构化(整组缺失/按类型/高掩码率)。

masked 与 missing 独立:只遮蔽本就有数据(未 missing)的 (node, group)。被遮蔽字段在标准化空间
置 0,masked 位置 1;目标 = 遮蔽前的值。损失只在被遮蔽位置计。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..schema.fields import MASKABLE_GROUPS, NODE_TABLE, NODE_TABLE_V03, Group
from ..schema.sample import group_bit_index

# 每个节点列所属的 maskable 组索引(-1 = META/不可遮蔽)
def _col_group_index(table=NODE_TABLE):
    idx = []
    gi = {g: i for i, g in enumerate(MASKABLE_GROUPS)}
    for f in table.fields:
        gg = gi.get(f.group, -1)
        idx.extend([gg] * f.dim)
    return np.array(idx, dtype=np.int64)


COL_GROUP = _col_group_index()
# 编码器 v2(schema v0.3,43 列;2026-08-06 A1):v1 消费方逐位不变,新路径显式传参
COL_GROUP_V03 = _col_group_index(NODE_TABLE_V03)


@dataclass
class MaskConfig:
    base_rate: float = 0.30           # 基础课程:遮蔽 (node,group) 比例
    p_group_dropout: float = 0.10     # 结构化:整运行状态组缺失(c_topo 模式)
    p_typed: float = 0.10             # 结构化:按节点类型定向(潮流模式,占位=遮蔽 STATE)
    p_high: float = 0.10              # 高掩码率档
    # 潮流定向缺失档(方案 3.4.1"按节点类型定向缺失"的字段级实现;2026-07-11 用户批准实验)。
    # 遮各节点的 PF 未知量字段、**注入量保持可见**(=D15 未解方式编码)。默认 0:训练课程
    # 专用开关;验证/门禁掩码流一律走默认 MaskConfig(),逐位不变(8.9)。
    p_pf_typed: float = 0.0
    # **连通片电压/相角课程**(v2 S2,2026-07-16 用户点火;立项书 §2.2"状态通道课程"):
    # 从随机种子节点沿边 BFS 扩成连通片(占有效节点 15%~45%),只遮片内 v/theta 两列、
    # **注入量与片外全部可见**——片内邻居同被遮,邻域插值失效,补全唯一途径 = 边界条件 +
    # 注入 + 潮流一致性(全局推理)。默认 0:验证/门禁掩码流逐位不变(与 p_pf_typed 同纪律)。
    p_region: float = 0.0
    region_frac_lo: float = 0.15
    region_frac_hi: float = 0.45


# 潮流未知量字段(按 node_type onehot 序 PQ/PV/SLACK = 0/1/2)。
# 必须与 schema.assemble._PF_UNKNOWNS / PQ / PV / SLACK 一致(单测锁定;
# 不直接 import 是为避免 psasp_io 依赖链进训练模块)。
PF_UNKNOWN_FIELDS = {0: ("v", "theta"), 1: ("qg", "theta"), 2: ("pg", "qg")}


def _region_vtheta_mask(node_feat: np.ndarray, node_missing: np.ndarray,
                        edge_index: np.ndarray, cfg: MaskConfig, rng: np.random.Generator,
                        table=NODE_TABLE):
    """连通片 v/theta 掩码(S2 新目标课程):BFS 连通片内只遮电压/相角,注入可见。

    片内节点的邻居也在片内(除边界)→ 局部插值不可行;有效补全须用全网潮流关系。
    跳过 STATE 组本就 missing 或 NULL 占位的节点(与 _pf_typed_mask 同约定)。
    v0.3 下 STATE 组列偏移 +18(DYN_EXT 插入),offset 一律走 table。
    """
    n, F = node_feat.shape
    masked_feat = node_feat.copy()
    loss_mask = np.zeros((n, F), dtype=bool)
    state_bit = group_bit_index(Group.STATE)
    null_col = table.offset_of("is_null")
    valid = (~node_missing[:, state_bit]) & (node_feat[:, null_col] <= 0.5)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2 or edge_index is None or edge_index.size == 0:
        return masked_feat, node_feat.copy(), loss_mask     # 无图/无有效节点:本档退空掩码
    # 邻接表(无向)
    adj = [[] for _ in range(n)]
    for a, b in edge_index.T:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    frac = rng.uniform(cfg.region_frac_lo, cfg.region_frac_hi)
    target = max(2, int(round(frac * len(valid_idx))))
    seed = int(valid_idx[rng.integers(len(valid_idx))])
    region, seen, queue = [], {seed}, [seed]
    while queue and len(region) < target:
        cur = queue.pop(0)
        region.append(cur)
        # 只经有效节点扩散 → 片 = 有效诱导子图内的连通块(语义清晰、单测可锁);
        # 邻居随机序入队,片形状不偏向固定编号序
        nbrs = [x for x in adj[cur] if x not in seen and valid[x]]
        rng.shuffle(nbrs)
        for x in nbrs:
            seen.add(x)
            queue.append(x)
    cols = (table.offset_of("v"), table.offset_of("theta"))
    for i in region:
        for c in cols:
            masked_feat[i, c] = 0.0
            loss_mask[i, c] = True
    return masked_feat, node_feat.copy(), loss_mask


def _pf_typed_mask(node_feat: np.ndarray, node_missing: np.ndarray, table=NODE_TABLE):
    """潮流模式字段级掩码:全部有效节点按类型遮 PF 未知量(注入量可见,θ 可学 DC 物理)。"""
    n, F = node_feat.shape
    masked_feat = node_feat.copy()
    loss_mask = np.zeros((n, F), dtype=bool)
    state_bit = group_bit_index(Group.STATE)
    tcol = table.offset_of("node_type")
    types = np.argmax(node_feat[:, tcol:tcol + 3], axis=1)
    null_col = table.offset_of("is_null")
    for i in range(n):
        if node_missing[i, state_bit] or node_feat[i, null_col] > 0.5:
            continue                                  # 本就 missing / NULL 占位不遮
        for fname in PF_UNKNOWN_FIELDS[int(types[i])]:
            c = table.offset_of(fname)
            masked_feat[i, c] = 0.0
            loss_mask[i, c] = True
    return masked_feat, node_feat.copy(), loss_mask


def apply_node_mask(node_feat: np.ndarray, node_missing: np.ndarray, cfg: MaskConfig,
                    rng: np.random.Generator,
                    edge_index: np.ndarray = None,
                    table=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (masked_feat, target, loss_mask[N,F])。loss_mask=True 处为被遮蔽、计损失。

    edge_index 仅 p_region>0 的课程档需要(连通片 v/theta);p_region=0(含全部验证/门禁流)
    时各分支阈值与 rng 消耗与旧实现逐位相同,传不传 edge_index 均不改掩码流。
    """
    n, F = node_feat.shape
    tab = table if table is not None else NODE_TABLE
    cg = COL_GROUP if tab is NODE_TABLE else _col_group_index(tab)
    assert F == tab.total_dim, f"node_feat {F} 列与 schema 表 {tab.total_dim} 不符"
    ng = len(MASKABLE_GROUPS)
    target = node_feat.copy()
    masked_feat = node_feat.copy()
    loss_mask = np.zeros((n, F), dtype=bool)

    # 决定每个 (node, group) 是否遮蔽。p_pf_typed=0 且 p_region=0 时各分支阈值与旧实现
    # 数值相同、rng 消耗相同 → 掩码流逐位不变(门禁/阶梯依赖此保证)。
    u = rng.random()
    if u < cfg.p_group_dropout:
        state_bit = group_bit_index(Group.STATE)
        group_masked = np.zeros((n, ng), dtype=bool)
        group_masked[:, state_bit] = True          # 整运行状态组缺失
    elif u < cfg.p_group_dropout + cfg.p_typed:
        state_bit = group_bit_index(Group.STATE)
        group_masked = np.zeros((n, ng), dtype=bool)
        group_masked[:, state_bit] = True          # v0 简化:定向掩码归入 STATE 组
    elif u < cfg.p_group_dropout + cfg.p_typed + cfg.p_region:
        return _region_vtheta_mask(node_feat, node_missing, edge_index, cfg, rng, tab)
    elif u < cfg.p_group_dropout + cfg.p_typed + cfg.p_region + cfg.p_pf_typed:
        return _pf_typed_mask(node_feat, node_missing, tab)  # 潮流定向缺失(字段级)
    else:
        rate = (0.7 if u < (cfg.p_group_dropout + cfg.p_typed + cfg.p_region
                            + cfg.p_pf_typed + cfg.p_high)
                else cfg.base_rate)
        group_masked = rng.random((n, ng)) < rate

    # 不遮蔽本就 missing 的组
    group_masked &= ~node_missing

    # 展开到列
    for c in range(F):
        g = cg[c]
        if g < 0:
            continue
        sel = group_masked[:, g]
        masked_feat[sel, c] = 0.0
        loss_mask[sel, c] = True

    return masked_feat, target, loss_mask
