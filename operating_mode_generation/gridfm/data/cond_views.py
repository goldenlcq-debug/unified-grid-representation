"""条件视图(G 线扩散/读出解码器的骨架条件构造;2026-07-18 审查修复集中化)。

历史:c_topo 视图此前在 6 个 G 线脚本各拷一份 `_ctopo_view`,只清节点 STATE;
2026-07-17 五维审查发现**支路 STATE 组的 pi/qi/pj/qj 是该方式潮流真解**,经
sample_to_data 全量拼进 edge_attr(graph.py)→ 条件通路泄漏答案。本模块集中定义
两个视图,去泄漏逻辑只此一处(消除重复口径漂移风险):

- `ctopo_view`:节点 STATE 组整体置缺失(标准化空间清零;豁免本就 missing 行)。
  = 预训练课程 (a) 的骨架条件,节点侧口径。**保留供 v1 ReconDecoder 历史路径**
  (纯 MLP 无支路输入,诊断 A 历史数字口径不变)。
- `cgen_view`:在 ctopo_view 基础上,**额外**清零支路 STATE 解量 pi/qi/pj/qj,
  **保留 mark**(在运/开断骨架,实测全库恒定)。用于一切消费支路 token 的生成
  管线(G1.5 读出解码器 e_ct、扩散条件 c_topo 支路 token、回归对照)。

口径注记(随修订方案入档):
- 支路 STATE 组混装骨架量(mark)与解量(pi/qi/pj/qj),故按列去泄漏而非整组置缺失
  ——整组会把 mark 一并抹掉,把全网读成开断,失真更大(6.3② "组整体置缺失" 的列级
  合理偏离,已备案)。
- **风险**:预训练掩码课程只掩节点侧(masking.py 无支路分支),支路 token 置 0 对
  冻结编码器是**分布外新输入**(D11 不解冻,以下游实证 = 诊断 A 复验 + 漂移检查验收)。
- 编码器不消费 missing 位(gps.py 只读 node_feat/edge_attr),掩码信号 = 标准化
  空间值置 0.0;故本视图只改特征列、不动 missing 位,与节点侧 ctopo_view 对称。
"""
from __future__ import annotations

import copy

import numpy as np

from ..schema.fields import BRANCH_TABLE, Group
from ..schema.sample import group_bit_index
from ..train.masking import COL_GROUP

# 支路 STATE 解量列(去泄漏目标;mark 不在内,保留为骨架条件)
BRANCH_FLOW_COLS = [BRANCH_TABLE.offset_of(c) for c in ("pi", "qi", "pj", "qj")]


def ctopo_view(s_norm):
    """节点 STATE 组整体置缺失(标准化空间清零;仅豁免本就 missing 行)。"""
    v = copy.deepcopy(s_norm)
    sbit = group_bit_index(Group.STATE)
    rows = ~v.node_missing[:, sbit]
    cols = np.where(COL_GROUP == sbit)[0]
    v.node_feat[np.ix_(rows, cols)] = 0.0
    return v


def cgen_view(s_norm):
    """节点 STATE 清零(同 ctopo)+ 支路解量 pi/qi/pj/qj 清零、mark 保留(去泄漏)。"""
    v = ctopo_view(s_norm)
    sbit = group_bit_index(Group.STATE)
    brows = ~v.branch_missing[:, sbit]
    v.branch_feat[np.ix_(brows, BRANCH_FLOW_COLS)] = 0.0
    return v
