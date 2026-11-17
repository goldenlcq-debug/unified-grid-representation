"""PSASP 51 格式卡片的类型化视图(方案 T1.2 / 附录 D)。

每个视图持有底层 Record 引用与行号,读为类型化属性,改经 Record.set_* 保留字节结构。
列索引依据 IEEE-118 真实数据(2026-07-08 勘察)与附录 D 核实。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .lowlevel import Card, Record


def _find_idname_index(rec: Record) -> int:
    """返回 IDName 单元格下标 = 内容非空的引号单元格(name_ctrl 为全空格,跳过);取最后一个。"""
    found = -1
    for i, cell in enumerate(rec.cells):
        s = cell.strip()
        if len(s) >= 2 and s[0] == "'" and s[-1] == "'" and s[1:-1].strip() != "":
            found = i
    return found


def branch_type_from_idname(idname: str) -> str:
    """按 IDName 后缀判支路类型(方案 §4.4 / 附录 C.2)。"""
    n = idname
    if n.endswith("_ac"):
        return "acline"
    if n.endswith("_sc"):
        return "series_comp"
    if n.endswith("_2w"):
        return "xfmr_2w"
    if n.endswith("_sf"):
        return "phase_shifter"
    if n.endswith("_3w1") or n.endswith("_3w2") or n.endswith("_3w3"):
        return "xfmr_3w"
    return "shunt_comp"  # LF.L2 中无后缀记录 = 并联电容/电抗器


# ---------------- LF.L1 母线 ----------------
class BusView:
    NAME, VBASE, ZONE, VMAX, VMIN, CB1, CB3, BUS_NAME = 0, 1, 2, 3, 4, 5, 6, 7

    def __init__(self, rec: Record, row: int) -> None:
        self.rec, self.row = rec, row

    @property
    def name(self) -> str:
        return self.rec.get_str(self.NAME)

    @property
    def is_null(self) -> bool:
        return self.name.upper().startswith("NULL")

    @property
    def vbase(self) -> float:
        return self.rec.get_float(self.VBASE)

    @property
    def zone(self) -> int:
        return self.rec.get_int(self.ZONE)

    @property
    def vmax(self) -> float:
        return self.rec.get_float(self.VMAX)

    @property
    def vmin(self) -> float:
        return self.rec.get_float(self.VMIN)


# ---------------- LF.L2 交流线/串补/并补 ----------------
class AclineView:
    MARK, I, J, ID, R, X, B2 = 0, 1, 2, 3, 4, 5, 6

    def __init__(self, rec: Record, row: int) -> None:
        self.rec, self.row = rec, row
        self._id_idx = _find_idname_index(rec)

    @property
    def mark(self) -> int:
        return self.rec.get_int(self.MARK)

    @mark.setter
    def mark(self, v: int) -> None:
        self.rec.set_int(self.MARK, v)

    @property
    def i_no(self) -> int:
        return self.rec.get_int(self.I)

    @property
    def j_no(self) -> int:
        return self.rec.get_int(self.J)

    @property
    def id_no(self) -> int:
        return self.rec.get_int(self.ID)

    @property
    def r(self) -> float:
        return self.rec.get_float(self.R)

    @property
    def x(self) -> float:
        return self.rec.get_float(self.X)

    @property
    def b_half(self) -> float:
        return self.rec.get_float(self.B2)

    @property
    def idname(self) -> str:
        return self.rec.get_str(self._id_idx)

    @property
    def branch_type(self) -> str:
        # 交流线卡(L2)的类型域。带后缀记录走后缀规则(自产语料约定,v1 逐位不变)。
        # **无后缀记录改按连接/电气特征判**(F4,第十二任审查 2026-08-08):原实现
        # 一律兜底并补,把万节点(原始工程数据,IDName 如 'AC1')7,304 条交流线全判
        # shunt_comp——Ybus 丢串联阻抗、branch_type 恒常数成域判别捷径。特征判据:
        # i==j 自环 = 并联元件(自产语料的并补正是此形,实扫 122 条全自环,零改判);
        # x<0 = 串补;其余 = 交流线。
        n = self.idname
        if n.endswith(("_ac", "_sc", "_2w", "_sf", "_3w1", "_3w2", "_3w3")):
            t = branch_type_from_idname(n)
            return t if t in ("acline", "series_comp", "shunt_comp") else "acline"
        if self.i_no == self.j_no:
            return "shunt_comp"
        return "series_comp" if self.x < 0 else "acline"


# ---------------- LF.L3 变压器 ----------------
class XfmrView:
    MARK, I, J, ID, R1, X1, TK = 0, 1, 2, 3, 4, 5, 6

    def __init__(self, rec: Record, row: int) -> None:
        self.rec, self.row = rec, row
        self._id_idx = _find_idname_index(rec)

    @property
    def mark(self) -> int:
        return self.rec.get_int(self.MARK)

    @mark.setter
    def mark(self, v: int) -> None:
        self.rec.set_int(self.MARK, v)

    @property
    def i_no(self) -> int:
        return self.rec.get_int(self.I)

    @property
    def j_no(self) -> int:
        return self.rec.get_int(self.J)

    @property
    def id_no(self) -> int:
        return self.rec.get_int(self.ID)

    @property
    def r1(self) -> float:
        return self.rec.get_float(self.R1)

    @property
    def x1(self) -> float:
        return self.rec.get_float(self.X1)

    @property
    def tk(self) -> float:
        return self.rec.get_float(self.TK)

    @property
    def idname(self) -> str:
        return self.rec.get_str(self._id_idx)

    @property
    def branch_type(self) -> str:
        # 变压器卡(L3):后缀不匹配则默认两绕组(而非 shunt_comp 兜底)
        t = branch_type_from_idname(self.idname)
        return t if t in ("xfmr_2w", "phase_shifter", "xfmr_3w") else "xfmr_2w"


# ---------------- LF.L5 发电机 ----------------
def _gen_layout_offset(rec: Record, base_ncells: int, card_name: str) -> int:
    """发电机表版式探测(2026-07-11 GenView 字段错位修复,见变更记录/任务笔记)。

    LF.L5 与 ST.S5 存在两种字段版式:基准版(ieee118/合成器,L5=20、S5=18 单元格,
    含行尾空单元格)与"带机组序号列"版(ieee39/cepri36,idx2 多一列 GenNo,L5=21、
    S5=19)。返回 idx≥2 数值列的整体右移量(0 或 1);未知版式**直接报错**,
    禁止静默回退——此前硬编码索引在 21 单元格版上整列错位(平衡机误标 PV、
    pg/qg/v_set 错读,写入路径连带把方式库物理写坏)。
    """
    off = rec.n_fields() - base_ncells
    if off not in (0, 1):
        raise ValueError(f"{card_name} 未知版式:{rec.n_fields()} 单元格"
                         f"(基准 {base_ncells},仅接受 +0/+1)")
    return off


class GenView:
    # 基准版(20 单元格)列位;实际访问经 _c() 加版式偏移(idx≥2)
    MARK, BS, TYPE, PG, QG, V0, THETA, QMAX, QMIN, PMAX, PMIN = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    BASE_NCELLS = 20

    def __init__(self, rec: Record, row: int) -> None:
        self.rec, self.row = rec, row
        self._off = _gen_layout_offset(rec, self.BASE_NCELLS, "LF.L5")
        self._id_idx = _find_idname_index(rec)
        # 后验校验:Type 必须是整数单元格、Pg 必须是小数单元格(51 格式浮点恒带小数点)
        if "." in rec.raw(self._c(self.TYPE)) or "." not in rec.raw(self._c(self.PG)):
            raise ValueError(f"LF.L5 版式校验失败(off={self._off}):"
                             f"Type={rec.raw(self._c(self.TYPE))!r} "
                             f"Pg={rec.raw(self._c(self.PG))!r}")

    def _c(self, i: int) -> int:
        return i + self._off if i >= 2 else i

    @property
    def mark(self) -> int:
        return self.rec.get_int(self.MARK)

    @mark.setter
    def mark(self, v: int) -> None:
        self.rec.set_int(self.MARK, v)

    @property
    def bs_no(self) -> int:
        return self.rec.get_int(self.BS)

    @property
    def type_code(self) -> int:
        return self.rec.get_int(self._c(self.TYPE))

    @property
    def pg(self) -> float:
        return self.rec.get_float(self._c(self.PG))

    @pg.setter
    def pg(self, v: float) -> None:
        self.rec.set_float(self._c(self.PG), v)

    @property
    def qg(self) -> float:
        return self.rec.get_float(self._c(self.QG))

    @qg.setter
    def qg(self, v: float) -> None:
        self.rec.set_float(self._c(self.QG), v)

    @property
    def v0(self) -> float:
        return self.rec.get_float(self._c(self.V0))

    @v0.setter
    def v0(self, v: float) -> None:
        self.rec.set_float(self._c(self.V0), v)

    @property
    def pmax(self) -> float:
        return self.rec.get_float(self._c(self.PMAX))

    @property
    def pmin(self) -> float:
        return self.rec.get_float(self._c(self.PMIN))

    @property
    def qmax(self) -> float:
        return self.rec.get_float(self._c(self.QMAX))

    @property
    def qmin(self) -> float:
        return self.rec.get_float(self._c(self.QMIN))

    @property
    def idname(self) -> str:
        return self.rec.get_str(self._id_idx)


# ---------------- LF.L6 负荷 ----------------
class LoadView:
    MARK, BS, ID, TYPE, PL, QL = 0, 1, 2, 3, 4, 5

    def __init__(self, rec: Record, row: int) -> None:
        self.rec, self.row = rec, row
        self._id_idx = _find_idname_index(rec)

    @property
    def mark(self) -> int:
        return self.rec.get_int(self.MARK)

    @property
    def bs_no(self) -> int:
        return self.rec.get_int(self.BS)

    @property
    def id_no(self) -> int:
        return self.rec.get_int(self.ID)

    @property
    def pl(self) -> float:
        return self.rec.get_float(self.PL)

    @pl.setter
    def pl(self, v: float) -> None:
        self.rec.set_float(self.PL, v)

    @property
    def ql(self) -> float:
        return self.rec.get_float(self.QL)

    @ql.setter
    def ql(self, v: float) -> None:
        self.rec.set_float(self.QL, v)

    @property
    def idname(self) -> str:
        return self.rec.get_str(self._id_idx)


# ---------------- 结果卡 ----------------
@dataclass
class BusResult:
    bus_no: int
    v_pu: float
    theta_deg: float
    name: str


@dataclass
class BranchFlow:
    i_no: int
    j_no: int
    id_no: int
    pi: float
    qi: float
    pj: float
    qj: float
    name: str


@dataclass
class GenResult:
    bus_no: int
    pg: float
    qg: float
    name: str


def parse_lp1(card: Card) -> List[BusResult]:
    """LF.LP1 母线电压:首条为状态行(跳过);其后 bus_no, V(pu), θ(deg), 'name'。
    发散解可能打出 Fortran 溢出星号(***),此类行跳过(该母线无有效解)。"""
    out: List[BusResult] = []
    for rec in card.records[1:]:
        try:
            out.append(BusResult(rec.get_int(0), rec.get_float(1), rec.get_float(2), rec.get_str(3)))
        except (ValueError, IndexError):
            continue
    return out


def parse_lp2(card: Card) -> List[BranchFlow]:
    """LF.LP2 交流线潮流:I, J, ID, Pi, Qi, Pj, Qj, ..., 'name'。溢出星号行跳过。"""
    out: List[BranchFlow] = []
    for rec in card.records:
        try:
            out.append(
                BranchFlow(
                    rec.get_int(0), rec.get_int(1), rec.get_int(2),
                    rec.get_float(3), rec.get_float(4), rec.get_float(5), rec.get_float(6),
                    rec.get_str(_find_idname_index(rec)),
                )
            )
        except (ValueError, IndexError):
            continue
    return out


def parse_lp5(card: Card) -> List[GenResult]:
    """LF.LP5 发电机出力:bus_no, Pg, Qg, ?, 'name'。溢出星号行跳过。"""
    out: List[GenResult] = []
    for rec in card.records:
        try:
            out.append(GenResult(rec.get_int(0), rec.get_float(1), rec.get_float(2),
                                 rec.get_str(_find_idname_index(rec))))
        except (ValueError, IndexError):
            continue
    return out


def parse_cal_converged(card: Card) -> Optional[bool]:
    """LF.CAL 收敛信息:首条记录首字段(1=收敛)。格式假设,勘察后可校准。"""
    recs = card.records
    if not recs:
        return None
    try:
        return recs[0].get_int(0) == 1
    except (ValueError, IndexError):
        return None


# ---------------- ST.S11 故障卡 ----------------
class FaultView:
    MARK, I, J, ID, KPCT, ADD_NO, A, B, C, GROUND, SHORT, OPEN, TS, TE, R, X = range(16)

    def __init__(self, rec: Record) -> None:
        self.rec = rec

    @property
    def i_no(self) -> int:
        return self.rec.get_int(self.I)

    @property
    def j_no(self) -> int:
        return self.rec.get_int(self.J)

    @property
    def id_no(self) -> int:
        return self.rec.get_int(self.ID)

    @property
    def k_pct(self) -> float:
        return self.rec.get_float(self.KPCT)

    @property
    def add_no(self) -> int:
        """新增计算母线(故障点位置)编号 = ST.S1/LF.L1 行号;K%=0 端部故障为 0。"""
        return self.rec.get_int(self.ADD_NO)

    @property
    def phases(self) -> tuple:
        return (self.rec.get_int(self.A), self.rec.get_int(self.B), self.rec.get_int(self.C))

    @property
    def ground(self) -> int:
        return self.rec.get_int(self.GROUND)

    @property
    def short(self) -> int:
        return self.rec.get_int(self.SHORT)

    @property
    def open_(self) -> int:
        return self.rec.get_int(self.OPEN)

    @property
    def ts(self) -> float:
        return self.rec.get_float(self.TS)

    @property
    def te(self) -> float:
        return self.rec.get_float(self.TE)


# ---------------- ST.S5 发电机动态模型 ----------------
class GenDynView:
    # 基准版(18 单元格)列位;带机组序号列版(19 单元格)idx≥2 全部 +1(同 GenView 修复)
    MARK, BS = 0, 1
    # 型号/参数组号列(schema v0.3/S1):Tg=机型,Lg=DATALIB #,GEN, 参数组号;
    # Tvr/Lvr、Tgo/Lgo、Tpss/Lpss = 励磁/调速/PSS 的型号+组号(0 = 未装该装置)
    TG, LG, TVR, LVR, TGO, LGO, TPSS, LPSS = 2, 3, 4, 5, 6, 7, 8, 9
    XDP, XDPP, X2, TJ, SH, PH = 10, 11, 12, 13, 14, 15
    BASE_NCELLS = 18

    def __init__(self, rec: Record) -> None:
        self.rec = rec
        self._off = _gen_layout_offset(rec, self.BASE_NCELLS, "ST.S5")
        self._id_idx = _find_idname_index(rec)

    def _c(self, i: int) -> int:
        return i + self._off if i >= 2 else i

    @property
    def mark(self) -> int:
        return self.rec.get_int(self.MARK)

    @mark.setter
    def mark(self, v: int) -> None:
        self.rec.set_int(self.MARK, v)

    @property
    def bs_no(self) -> int:
        return self.rec.get_int(self.BS)

    @property
    def tj(self) -> float:
        return self.rec.get_float(self._c(self.TJ))

    @property
    def xdp(self) -> float:
        return self.rec.get_float(self._c(self.XDP))

    @property
    def sh(self) -> float:
        return self.rec.get_float(self._c(self.SH))

    @property
    def ph(self) -> float:
        return self.rec.get_float(self._c(self.PH))

    @property
    def xdpp(self) -> float:
        return self.rec.get_float(self._c(self.XDPP))

    @property
    def x2v(self) -> float:
        return self.rec.get_float(self._c(self.X2))

    @property
    def tg(self) -> int:
        return self.rec.get_int(self._c(self.TG))

    @property
    def lg(self) -> int:
        return self.rec.get_int(self._c(self.LG))

    @property
    def lvr(self) -> int:
        return self.rec.get_int(self._c(self.LVR))

    @property
    def lgo(self) -> int:
        return self.rec.get_int(self._c(self.LGO))

    @property
    def lpss(self) -> int:
        return self.rec.get_int(self._c(self.LPSS))

    @property
    def idname(self) -> str:
        return self.rec.get_str(self._id_idx)
