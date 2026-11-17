"""档案 schema 字段表与通道组(方案 附录 C / 2.2 / D15)。

四通道组:拓扑/静态/动态/运行状态;每组配 missing 位(数据本身缺失)与 masked 位
(训练人为遮蔽)各 1 位,两者语义独立、通道分离。位置编码(LaPE/RWSE/随机特征)
在数据生成阶段离线算,单独存储,不在此字段表内。

v0 说明:本表为可行性期实现子集(涵盖四组关键字段);完整字段与归一化随实验扩充,
扩充须升 SCHEMA_VERSION 并记入变更记录。归一化在 T1.6 单独模块施加,本层存原始值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

SCHEMA_VERSION = "v0.3"   # v0.3(2026-07-16/S1):节点动态组扩 18 列(DATALIB 解引用的同步机全量
#   参数 + 装置存在位 + 解引用有效位,见 DYN_EXT_FIELDS)。**追加不改旧**:v0 的 25 列
#   (含 cepri36/ieee39 内联假 0 的 tj/xdp)字节语义不变;正确值只进新列。v1 栈经
#   load_npz 默认 v0 列视图消费,逐位不变。
# v0.2(2026-07-09):样本增持久化随机特征 node_random/branch_random(方案 3.2-3)


class Group(str, Enum):
    TOPOLOGY = "topology"
    STATIC = "static"
    DYNAMIC = "dynamic"
    STATE = "state"
    META = "meta"


# 四个带 missing/masked 位的通道组(顺序固定,决定位索引)
MASKABLE_GROUPS: List[Group] = [Group.TOPOLOGY, Group.STATIC, Group.DYNAMIC, Group.STATE]


@dataclass
class FieldSpec:
    name: str
    group: Group
    dim: int
    norm: str  # 'onehot' | 'bool' | 'pu_z' | 'z' | 'log_z' | 'rad' | 'raw'


@dataclass
class FieldTable:
    fields: List[FieldSpec] = field(default_factory=list)

    @property
    def total_dim(self) -> int:
        return sum(f.dim for f in self.fields)

    def group_slice(self, group: Group):
        start = 0
        for f in self.fields:
            if f.group == group:
                return slice(start, start + self._group_span(group))
            start += f.dim
        return slice(0, 0)

    def _group_span(self, group: Group) -> int:
        return sum(f.dim for f in self.fields if f.group == group)

    def offset_of(self, name: str) -> int:
        off = 0
        for f in self.fields:
            if f.name == name:
                return off
            off += f.dim
        raise KeyError(name)


# ---------------- 节点字段表(v0) ----------------
NODE_TABLE = FieldTable([
    # 拓扑组
    FieldSpec("node_type", Group.TOPOLOGY, 3, "onehot"),   # [PQ, PV, Slack]
    FieldSpec("is_null", Group.TOPOLOGY, 1, "bool"),
    FieldSpec("has_gen", Group.TOPOLOGY, 1, "bool"),
    FieldSpec("has_load", Group.TOPOLOGY, 1, "bool"),
    FieldSpec("vbase", Group.TOPOLOGY, 1, "z"),
    # 静态组
    FieldSpec("vmax", Group.STATIC, 1, "pu_z"),
    FieldSpec("vmin", Group.STATIC, 1, "pu_z"),
    FieldSpec("pmax", Group.STATIC, 1, "pu_z"),
    FieldSpec("pmin", Group.STATIC, 1, "pu_z"),
    FieldSpec("qmax", Group.STATIC, 1, "pu_z"),
    FieldSpec("qmin", Group.STATIC, 1, "pu_z"),
    # 动态组(无机组则置 0 + missing)
    FieldSpec("tj", Group.DYNAMIC, 1, "log_z"),
    FieldSpec("xdp", Group.DYNAMIC, 1, "z"),
    FieldSpec("sh", Group.DYNAMIC, 1, "pu_z"),
    FieldSpec("ph", Group.DYNAMIC, 1, "pu_z"),
    # 运行状态组
    FieldSpec("pg", Group.STATE, 1, "pu_z"),
    FieldSpec("qg", Group.STATE, 1, "pu_z"),
    FieldSpec("pl", Group.STATE, 1, "pu_z"),
    FieldSpec("ql", Group.STATE, 1, "pu_z"),
    FieldSpec("v", Group.STATE, 1, "pu_z"),
    FieldSpec("theta", Group.STATE, 1, "rad"),
    FieldSpec("v_set", Group.STATE, 1, "pu_z"),
    # 元信息组(无 missing/masked)
    FieldSpec("adjustable", Group.META, 1, "bool"),
])

# ---------------- 支路字段表(v0) ----------------
BRANCH_TABLE = FieldTable([
    # 拓扑组:类型 onehot(acline/series_comp/xfmr_2w/phase_shifter/xfmr_3w/shunt_comp)
    FieldSpec("branch_type", Group.TOPOLOGY, 6, "onehot"),
    # 静态组
    FieldSpec("r", Group.STATIC, 1, "log_z"),
    FieldSpec("x", Group.STATIC, 1, "z"),
    FieldSpec("b_half", Group.STATIC, 1, "z"),
    # 运行状态组
    FieldSpec("mark", Group.STATE, 1, "raw"),
    FieldSpec("pi", Group.STATE, 1, "pu_z"),
    FieldSpec("qi", Group.STATE, 1, "pu_z"),
    FieldSpec("pj", Group.STATE, 1, "pu_z"),
    FieldSpec("qj", Group.STATE, 1, "pu_z"),
])

BRANCH_TYPES = ["acline", "series_comp", "xfmr_2w", "phase_shifter", "xfmr_3w", "shunt_comp"]


# ---------------- v0.3 扩展(S1,2026-07-16):同步机全量动态参数 ----------------
# 来源 = ST.S5 指派卡 + DATALIB.DAT #,GEN, 型号库解引用(gridfm/psasp_io/datalib.py)。
# **追加不改旧**:legacy tj/xdp(cepri36/ieee39 内联假 0)原样保留在 v0 列,正确值进新列
# (tj_lib/xdp_lib),v1 栈可复现性逐位不破。全部入 DYNAMIC 组(插在 ph 之后,组保持连续,
# 共享 DYNAMIC 的组 missing 位;非机组节点新列 = 0 + 组 missing 照旧)。
# dyn_resolved = DATALIB 解引用成功位(修"假 0 被当有效值"的语义坑:v2 侧可据此区分)。
DYN_EXT_FIELDS: List[FieldSpec] = [
    FieldSpec("xd", Group.DYNAMIC, 1, "z"),          # 直轴同步电抗
    FieldSpec("xdp_lib", Group.DYNAMIC, 1, "z"),     # X'd(DATALIB 正确值;legacy xdp 为内联)
    FieldSpec("xdpp", Group.DYNAMIC, 1, "z"),        # X''d
    FieldSpec("xq", Group.DYNAMIC, 1, "z"),
    FieldSpec("xqp", Group.DYNAMIC, 1, "z"),
    FieldSpec("xqpp", Group.DYNAMIC, 1, "z"),
    FieldSpec("x2", Group.DYNAMIC, 1, "z"),          # 负序电抗
    FieldSpec("td0p", Group.DYNAMIC, 1, "log_z"),    # T'd0
    FieldSpec("td0pp", Group.DYNAMIC, 1, "log_z"),
    FieldSpec("tq0p", Group.DYNAMIC, 1, "log_z"),    # 哨兵 9999(不建模)→ 装配置 0
    FieldSpec("tq0pp", Group.DYNAMIC, 1, "log_z"),
    FieldSpec("tj_lib", Group.DYNAMIC, 1, "log_z"),  # 惯量 Tj(DATALIB 正确值)
    FieldSpec("damp_d", Group.DYNAMIC, 1, "z"),      # 阻尼 D
    FieldSpec("gen_model", Group.DYNAMIC, 1, "z"),   # 机型号 Tg
    FieldSpec("has_avr", Group.DYNAMIC, 1, "bool"),  # 励磁装置存在(Lvr>0)
    FieldSpec("has_gov", Group.DYNAMIC, 1, "bool"),  # 调速装置存在(Lgo>0)
    FieldSpec("has_pss", Group.DYNAMIC, 1, "bool"),  # PSS 存在(Lpss>0)
    FieldSpec("dyn_resolved", Group.DYNAMIC, 1, "bool"),  # DATALIB 解引用成功
]


def _build_node_table_v03() -> FieldTable:
    fields = list(NODE_TABLE.fields)
    ip = next(i for i, f in enumerate(fields) if f.name == "ph") + 1   # DYNAMIC 组尾
    return FieldTable(fields[:ip] + DYN_EXT_FIELDS + fields[ip:])


NODE_TABLE_V03 = _build_node_table_v03()
V0_NODE_DIM = NODE_TABLE.total_dim        # 25


def _v0_cols_in_v03() -> List[int]:
    """v0 的 25 列在 v0.3 表中的列索引(按 v0 字段序;供 v0 列视图切片)。"""
    cols: List[int] = []
    for f in NODE_TABLE.fields:
        start = NODE_TABLE_V03.offset_of(f.name)
        cols.extend(range(start, start + f.dim))
    return cols


V03_TO_V0_COLS = _v0_cols_in_v03()
