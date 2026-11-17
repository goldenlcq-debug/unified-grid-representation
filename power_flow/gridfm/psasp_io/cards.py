"""PSASP 51 格式卡片的类型化视图(方案 T1.2 / 附录 D)。

每个视图持有底层 Record 引用与行号,读为类型化属性,改经 Record.set_* 保留字节结构。
列索引依据 IEEE-118 真实数据(2026-07-08 勘察)与附录 D 核实。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

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


# ---------------- LF.ML4 / LF.NL4 直流输入卡(E27,预注册 §30)----------------
# 版式与全部断言依据 docs/参考-直流卡版式解码规格-20260828.md(21 智能体解码 + 34 方式全扫实证)。
# 纪律:固定索引、禁按表头名切列(表头括号内有半角逗号,R7);只读输入卡,
# 解析器不得引用任何结果卡文件名(LF.MP4/LP4/NP4/dcreport,G3 泄漏闸)。

_ML4_HDR_RE = None  # 延迟编译(见 parse_ml4)

# ML4 表头行数签名(34/34 零变体;v2.3 空卡同):每设备占用行数 = 该 tag 连续表头行条数
_ML4_HDR_SIG = {0: 1, 1: 1, 2: 1, 3: 2, 4: 3, 5: 1, 6: 1}
# v2.8 数据行字段数签名(丢行尾空 token 后)
_ML4_NFIELDS = {2: (7,), 3: (14, 15), 4: (10, 12, 12), 5: (10,)}
# NL4 每条 8 行的字段数(986/986 无例外)
_NL4_NFIELDS = (6, 10, 2, 11, 11, 3, 10, 10)
# NL4 运行方式码 → 极数(显式映射,未知码报错;禁写 OP//10)
_NL4_OP_NPOLE = {20: 2, 10: 1, 11: 1, 12: 1}


def _dc_fields(rec: Record) -> List[str]:
    """数据行字段列表:丢弃行尾逗号产生的最后一个空 token。"""
    cells = rec.cells
    if len(cells) > 1 and cells[-1].strip() == "":
        return cells[:-1]
    return list(cells)


@dataclass
class Ml4Vsc:
    valid: int
    ac_bus: int          # A[1] 交流母线号(副边侧 = 阀侧)
    hv_dc_bus: int
    lv_dc_bus: int
    dev_no: int
    sn_mva: float        # A[6] 额定容量(自由站分摊权重)
    name: str
    npole: int           # B[0](在运恒 1,R17 断言)
    p_set_mw: float      # B[1] 给定有功(MW,流入换流器为正 = 负荷正方向,符号直接可用)
    q_set_mvar: float    # B[2] 给定无功(在运实测恒 0,R9)
    ctrl_mode: int       # B[4] 1PQ / -1PV / 0平衡
    balance: int         # B[7] 平衡站优先级


@dataclass
class Ml4Lcc:
    valid: int
    ac_bus: int          # A[1] 交流母线号(真电网母线,联结变在 LCC 模型内部)
    hv_dc_bus: int
    lv_dc_bus: int
    dev_no: int
    xfmr_mva: float      # B[1] 联结变压器单极总容量(自由站分摊权重)
    name: str
    conv_type: int       # C[0] 0整流 / 1逆变
    dc_ctrl: int         # C[2] 2定功率 / 3定压(其余码未见,见到即报错,R2)
    p_dc_mw: float       # C[4] 给定直流功率(非负幅值,方向靠 conv_type)


@dataclass
class Ml4DcLine:
    valid: int
    i_dc_bus: int
    j_dc_bus: int
    dev_no: int


@dataclass
class Ml4:
    version: str
    n_dc_bus: int
    n_vsc: int
    n_lcc: int
    n_dc_line: int
    n_dcdc: int
    dc_buses: List[int]          # #2[0] 声明集(直流母线判据只准用它,R10)
    vscs: List[Ml4Vsc]
    lccs: List[Ml4Lcc]
    dc_lines: List[Ml4DcLine]


@dataclass
class Nl4Rec:
    valid: int
    i_ac_bus: int        # 整流侧交流母线号
    j_ac_bus: int        # 逆变侧交流母线号
    dev_no: int
    name: str
    op_code: int         # L5[0] 给定运行方式(20双极/10地回/11金属回/12双导线并联)
    npole: int
    pd_mw: float         # L6[0] 给定极1直流运行功率(单极,MW)
    vd_kv: float         # L6[1]
    rl_ohm: float        # L1[4] 单极直流线路电阻(线损用)


def _ml4_int(cells: List[str], i: int, ctx: str) -> int:
    try:
        return int(cells[i].strip())
    except (ValueError, IndexError) as e:
        raise ValueError(f"LF.ML4 {ctx} 第 {i} 字段取整失败:{e}")


def parse_ml4(card: Card) -> Ml4:
    """解析 LF.ML4(#0 版本 / #1 计数 / #2 直流母线 / #3 VSC×2行 / #4 LCC×3行 / #5 直流线 / #6 DC/DC)。

    分段规则(最关键):扫描时维护"当前 tag",表头行(str 元素,#N 开头)只切换 tag 不产出数据,
    设备数为 0 时表头连片不塌陷(v2.3 空卡实测)。行数硬校验 + 字段数硬校验 + 值域断言,
    任何未定版式(v2.3 带数据 / DC/DC 非零 / 未知控制码)一律报错,不猜。
    """
    import re
    hdr_re = re.compile(r"^\s*#(\d+)")
    seg: Dict[int, List[Record]] = {}
    hdr_count: Dict[int, int] = {}
    tag: Optional[int] = None
    for el in card.elements:
        if isinstance(el, str):
            m = hdr_re.match(el)
            if m and el.lstrip().startswith("#"):
                tag = int(m.group(1))
                hdr_count[tag] = hdr_count.get(tag, 0) + 1
                seg.setdefault(tag, [])
            continue  # 空行/非表头说明行忽略
        if tag is None:
            raise ValueError("LF.ML4 表头之前出现数据行")
        seg[tag].append(el)

    if hdr_count != _ML4_HDR_SIG:
        raise ValueError(f"LF.ML4 表头行数签名异常:{hdr_count}(期望 {_ML4_HDR_SIG})")
    if len(seg[0]) != 1:
        raise ValueError(f"LF.ML4 版本行数 {len(seg[0])} != 1")
    version = seg[0][0].cells[0].strip().rstrip(",")
    if version not in ("2.3", "2.8"):
        raise ValueError(f"LF.ML4 未知版本 {version!r}(只接受 2.3 空卡 / 2.8)")
    if len(seg[1]) != 1:
        raise ValueError(f"LF.ML4 计数行数 {len(seg[1])} != 1")
    cnt_cells = _dc_fields(seg[1][0])
    if len(cnt_cells) != 6:
        raise ValueError(f"LF.ML4 计数行字段数 {len(cnt_cells)} != 6")
    counts = [_ml4_int(cnt_cells, i, "#1") for i in range(5)]
    n_dc_bus, n_vsc, n_lcc, n_dc_line, n_dcdc = counts
    if n_dcdc != 0:
        raise ValueError(f"LF.ML4 DC/DC 计数 {n_dcdc} != 0:#6 数据行版式未定(文档 8 字段 vs 导出器 7),不实现不猜(R16)")
    if version == "2.3" and any(c != 0 for c in counts):
        raise ValueError(f"LF.ML4 v2.3 带数据({counts}):v2.3 数据行版式无实样支撑,拒绝解析(R8)")
    # 行数硬校验:每设备行数从表头行数读,不写死
    for k, cnt in ((2, n_dc_bus), (3, n_vsc), (4, n_lcc), (5, n_dc_line), (6, n_dcdc)):
        want = cnt * _ML4_HDR_SIG[k]
        if len(seg.get(k, [])) != want:
            raise ValueError(f"LF.ML4 #{k} 行数 {len(seg.get(k, []))} != 计数 {cnt}×{_ML4_HDR_SIG[k]}")

    def _chk_nf(k: int, per_dev: List[List[str]]) -> None:
        want = _ML4_NFIELDS[k]  # 逐行显式判,只用前向索引(R8)
        for j, row in enumerate(per_dev):
            exp = want[j % len(want)]
            if len(row) != exp:
                raise ValueError(f"LF.ML4 #{k} 第 {j} 行字段数 {len(row)} != {exp}(版式漂移,拒绝猜列)")

    dc_buses: List[int] = []
    if n_dc_bus:
        rows2 = [_dc_fields(r) for r in seg[2]]
        _chk_nf(2, rows2)
        dc_buses = [_ml4_int(c, 0, "#2") for c in rows2]

    vscs: List[Ml4Vsc] = []
    if n_vsc:
        rows3 = [_dc_fields(r) for r in seg[3]]
        _chk_nf(3, rows3)
        for k in range(n_vsc):
            A, B = rows3[2 * k], rows3[2 * k + 1]
            valid = _ml4_int(A, 0, "#3A")
            if valid not in (0, 1):
                raise ValueError(f"LF.ML4 #3 第 {k} 台有效位 {valid} 不在 {{0,1}}(R1)")
            if _ml4_int(A, 13, "#3A") != 0:
                raise ValueError(f"LF.ML4 #3 第 {k} 台 A[13](已删除占位)非 0,版式存疑(R12)")
            npole = _ml4_int(B, 0, "#3B")
            ctrl = _ml4_int(B, 4, "#3B")
            bal = _ml4_int(B, 7, "#3B")
            if valid == 1:
                if npole != 1:
                    raise ValueError(f"LF.ML4 #3 第 {k} 台在运极数 {npole} != 1:B[1] 整站/每极口径未定(R17)")
                if ctrl not in (1, -1, 0):
                    raise ValueError(f"LF.ML4 #3 第 {k} 台控制方式 {ctrl} 未知(R2)")
                if bal not in (0, 1):
                    raise ValueError(f"LF.ML4 #3 第 {k} 台平衡站标志 {bal} 未知(R2)")
            vscs.append(Ml4Vsc(
                valid=valid, ac_bus=_ml4_int(A, 1, "#3A"), hv_dc_bus=_ml4_int(A, 2, "#3A"),
                lv_dc_bus=_ml4_int(A, 3, "#3A"), dev_no=_ml4_int(A, 4, "#3A"),
                sn_mva=float(A[6].strip()), name=A[12].strip().strip("'").rstrip(),
                npole=npole, p_set_mw=float(B[1].strip()), q_set_mvar=float(B[2].strip()),
                ctrl_mode=ctrl, balance=bal))

    lccs: List[Ml4Lcc] = []
    if n_lcc:
        rows4 = [_dc_fields(r) for r in seg[4]]
        _chk_nf(4, rows4)
        for k in range(n_lcc):
            A, B, C = rows4[3 * k], rows4[3 * k + 1], rows4[3 * k + 2]
            valid = _ml4_int(A, 0, "#4A")
            if valid not in (0, 1):
                raise ValueError(f"LF.ML4 #4 第 {k} 台有效位 {valid} 不在 {{0,1}}(R1)")
            if _ml4_int(A, 9, "#4A") != 0:
                raise ValueError(f"LF.ML4 #4 第 {k} 台 A[9](已删除占位)非 0,版式存疑(R12)")
            conv_type = _ml4_int(C, 0, "#4C")
            dc_ctrl = _ml4_int(C, 2, "#4C")
            if valid == 1:
                if conv_type not in (0, 1):
                    raise ValueError(f"LF.ML4 #4 第 {k} 台换流器类型 {conv_type} 未知")
                if dc_ctrl not in (2, 3):
                    raise ValueError(f"LF.ML4 #4 第 {k} 台直流控制方式 {dc_ctrl}:语料只出现 2/3,"
                                     f"1定流/4定角无实样,取数口径未定,拒绝猜(R2)")
            lccs.append(Ml4Lcc(
                valid=valid, ac_bus=_ml4_int(A, 1, "#4A"), hv_dc_bus=_ml4_int(A, 2, "#4A"),
                lv_dc_bus=_ml4_int(A, 3, "#4A"), dev_no=_ml4_int(A, 4, "#4A"),
                xfmr_mva=float(B[1].strip()), name=A[8].strip().strip("'").rstrip(),
                conv_type=conv_type, dc_ctrl=dc_ctrl, p_dc_mw=float(C[4].strip())))

    dc_lines: List[Ml4DcLine] = []
    if n_dc_line:
        rows5 = [_dc_fields(r) for r in seg[5]]
        _chk_nf(5, rows5)
        for k, row in enumerate(rows5):
            valid = _ml4_int(row, 0, "#5")
            if valid not in (0, 1):
                raise ValueError(f"LF.ML4 #5 第 {k} 条有效位 {valid} 不在 {{0,1}}(R1)")
            dc_lines.append(Ml4DcLine(valid=valid, i_dc_bus=_ml4_int(row, 1, "#5"),
                                      j_dc_bus=_ml4_int(row, 2, "#5"), dev_no=_ml4_int(row, 3, "#5")))

    return Ml4(version=version, n_dc_bus=n_dc_bus, n_vsc=n_vsc, n_lcc=n_lcc,
               n_dc_line=n_dc_line, n_dcdc=n_dcdc, dc_buses=dc_buses,
               vscs=vscs, lccs=lccs, dc_lines=dc_lines)


def parse_nl4(card: Card) -> List[Nl4Rec]:
    """解析 LF.NL4(两端直流):无表头无计数行,每条恰 8 行。

    记录切分用独立结构锚点:只有每条记录第 0 行含 ASCII 单引号(恰 2 个),其余 7 行 0 个
    (比 len%8 稳,R15)。L6≡L7 断言(单极取哪极未定):不等则告警并按 L6 取值。
    """
    recs = card.records
    if not recs:
        return []
    groups: List[List[Record]] = []
    for r in recs:
        nq = r.text().count("'")
        if nq == 2:
            groups.append([r])
        elif nq == 0:
            if not groups:
                raise ValueError("LF.NL4 首行不含引号锚点,版式异常")
            groups[-1].append(r)
        else:
            raise ValueError(f"LF.NL4 行引号数 {nq} 既非 2 也非 0,版式异常:{r.text()[:60]!r}")
    if len(recs) % 8 != 0 or any(len(g) != 8 for g in groups):
        raise ValueError(f"LF.NL4 记录切分异常:总行 {len(recs)},组长 {[len(g) for g in groups][:8]}(每条应恰 8 行)")

    out: List[Nl4Rec] = []
    for gi, g in enumerate(groups):
        rows = [_dc_fields(r) for r in g]
        for li, exp in enumerate(_NL4_NFIELDS):
            if len(rows[li]) != exp:
                raise ValueError(f"LF.NL4 第 {gi} 条第 {li} 行字段数 {len(rows[li])} != {exp}(版式漂移)")
        valid = int(rows[0][0].strip())
        if valid not in (0, 1):
            raise ValueError(f"LF.NL4 第 {gi} 条有效位 {valid} 不在 {{0,1}}(R1)")
        op = int(float(rows[5][0].strip()))
        if op not in _NL4_OP_NPOLE:
            raise ValueError(f"LF.NL4 第 {gi} 条运行方式码 {op} 未知(已知 20/10/11/12,不猜)")
        if [c.strip() for c in rows[6]] != [c.strip() for c in rows[7]]:
            import warnings
            warnings.warn(f"LF.NL4 第 {gi} 条 L6 != L7(单极在运极口径未定),按 L6 取值")
        out.append(Nl4Rec(
            valid=valid, i_ac_bus=int(rows[0][1].strip()), j_ac_bus=int(rows[0][2].strip()),
            dev_no=int(rows[0][3].strip()), name=rows[0][5].strip().strip("'").rstrip(),
            op_code=op, npole=_NL4_OP_NPOLE[op],
            pd_mw=float(rows[6][0].strip()), vd_kv=float(rows[6][1].strip()),
            rl_ohm=float(rows[1][4].strip())))
    return out


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
