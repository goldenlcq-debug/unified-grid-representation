"""目录页(方案 4.4 / T1.5):PSASP 名称 ↔ token 行号双向索引,不进神经网络。

- 节点主键:计算母线编号 Bs_No(= LF.L1 行号,从 1 起)与母线名 Bs_Name(双键)。
- 支路主键:三元组 (I_No, J_No, ID_No) 与全局唯一 IDName(双键)——并联双回线仅凭两侧母线名
  无法唯一定位,回路号 ID_No 必须进主键。
- 支路 token 全局顺序 = 交流线(LF.L2 顺序)后接变压器(LF.L3 顺序)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .case import PsaspCase


@dataclass
class NodeEntry:
    token_row: int      # 节点 token 行号 = 计算母线编号(从 1 起)
    bus_no: int
    name: str
    vbase: float
    zone: int
    is_null: bool


@dataclass
class BranchEntry:
    token_row: int      # 支路 token 全局行号(从 0 起)
    i_no: int
    j_no: int
    id_no: int
    idname: str
    branch_type: str


class Catalog:
    def __init__(self, case: PsaspCase) -> None:
        self.nodes: List[NodeEntry] = []
        self.branches: List[BranchEntry] = []
        self._name2node: Dict[str, NodeEntry] = {}
        self._busno2node: Dict[int, NodeEntry] = {}
        self._key2branch: Dict[Tuple[int, int, int], BranchEntry] = {}
        self._idname2branch: Dict[str, BranchEntry] = {}
        self._build(case)

    def _build(self, case: PsaspCase) -> None:
        for i, b in enumerate(case.buses()):
            bus_no = i + 1
            e = NodeEntry(bus_no, bus_no, b.name, b.vbase, b.zone, b.is_null)
            self.nodes.append(e)
            self._busno2node[bus_no] = e
            if not b.is_null:
                self._name2node[b.name] = e

        row = 0
        for ln in case.aclines():
            e = BranchEntry(row, ln.i_no, ln.j_no, ln.id_no, ln.idname, ln.branch_type)
            self._add_branch(e)
            row += 1
        for tr in case.xfmrs():
            e = BranchEntry(row, tr.i_no, tr.j_no, tr.id_no, tr.idname, tr.branch_type)
            self._add_branch(e)
            row += 1

    def _add_branch(self, e: BranchEntry) -> None:
        self.branches.append(e)
        key = (e.i_no, e.j_no, e.id_no)
        if key in self._key2branch:
            raise ValueError(f"支路主键冲突 {key}(回路号未唯一化?)")
        self._key2branch[key] = e
        if e.idname:
            self._idname2branch[e.idname] = e

    # ---- 节点翻译 ----
    def node_row_by_name(self, name: str) -> int:
        return self._name2node[name].token_row

    def node_name_by_row(self, token_row: int) -> str:
        return self._busno2node[token_row].name

    # ---- 支路翻译(双回线用回路号消歧)----
    def branch_row_by_key(self, i_no: int, j_no: int, id_no: int) -> int:
        return self._key2branch[(i_no, j_no, id_no)].token_row

    def branch_row_by_idname(self, idname: str) -> int:
        return self._idname2branch[idname].token_row

    def branch_key_by_row(self, token_row: int) -> Tuple[int, int, int]:
        e = self.branches[token_row]
        return (e.i_no, e.j_no, e.id_no)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_branches(self) -> int:
        return len(self.branches)

    def parallel_branch_groups(self) -> List[List[int]]:
        """返回同 (I,J) 但不同回路号的并联支路组(每组 token_row 列表,长度>1)。"""
        groups: Dict[Tuple[int, int], List[int]] = {}
        for e in self.branches:
            groups.setdefault((e.i_no, e.j_no), []).append(e.token_row)
        return [rows for rows in groups.values() if len(rows) > 1]
