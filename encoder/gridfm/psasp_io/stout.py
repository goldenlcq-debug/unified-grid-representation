"""STOUT.DDD 逐机轨迹解析(T4.3 前置;方案 10.3 M4)+ ST.SME 曲线坐标写卡(阶段 2)。

**格式出处备案(docs/任务笔记-M4-手册出处.md)**:SOT 文本报表(手册 5.1/5.5)在
CLI(wmudrt)运行下不产出;轨迹本体是 **STOUT.DDD——无格式文档,按方案 T4.3
条款"手册不覆盖时以样例反推格式并记录"逆向**(CEPRI-36 实算 2026-07-11;
ieee39 混类型样例 2026-07-13,43 列 = 1 + 32×Type3 + 10×Type12 唯一自洽解):

- 纯 float64 小端平铺,无文件头;行 = 时间步,列 = [t, 坐标1, 坐标2, …];
- 列序 = ST.SME 中 Mark∈{1,2} 且 Coord_Type∈{1,2,3,12}(曲线类)记录的槽位顺序,
  Coord_Type=99 为上一记录的续行;Type 8/11 为监视/自动分析类,不产列;
- 各类型槽位排布(ST.SME.txt 字段表 + 样例实证):
  * Type=1 发电机功角(°):(母线号, 机组序) 对;GUI 样例逐对重复两次,按序去重
    (16 槽 8 对 = 4 坐标 ×2,cepri36 夹具锚定);
  * Type=2 母线电压(pu):每槽位一个母线号,一记录至多 16 条曲线(手册字段表;
    本项目自写,逐槽一号不重复;锚 = 逐步 min/max 对 STANA col8/col14);
  * Type=3 发电机变量:一记录一列,槽位 = (母线号, 机组序, 子变量);子变量
    1=功角(°) 2=转速(pu) 3=机端电压(pu)(手册子表;ieee39 留档实证:var1
    spread↔STANA col6 差 7e-4,var2 min/max↔col12/18 差 5e-6,var3 电压量纲);
  * Type=12 母线电压相角/频率:(I,J) 对,I==J 时为该母线频率(Hz,t=0 恒 50.0,
    ieee39 实证);I≠J(相角差)未经样例验证,出现即拒解。
- 验证锚:功角列逐步 max−min 与 STANA col6 逐步吻合(样例 t=0:70.758° 精确
  一致);单测锁定。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ENC = "gb18030"
# 引擎功角输出的封顶哨兵(2026-07-16 118 实测):逐机 δ 与 STANA 的角差列**各自**打到
# 99999 后恒定 —— 两边封的是不同的量(STANA 封"角差",STOUT 封"每台机的 δ"),故一旦
# 触顶,`max_angle_spread(δ)` 与 STANA 角差列**必然不等**(实测差 2k~22k°),锚点在封顶
# 段本就不可比。凡消费功角真值的口径(体检锚点、机群末态聚类)都必须先排除封顶段。
ANGLE_SENTINEL_DEG = 99998.0
_CURVE_TYPES = {1, 2, 3, 12}
_NONCURVE_TYPES = {8, 11}      # 监视信息/自动分析输出:实证不产列(cepri36+ieee39)
# 手册还有 4-7/13-29 等曲线类型与 9/10 信息类型,均无样例验证——出现即拒解不猜列数


@dataclass
class SmeCoord:
    coord_type: int
    pairs: List[tuple]     # 逐列描述,按列序:T1/T12=(i,j);T2=(bus,);T3=(bus,机组序,子变量)

    @property
    def n_cols(self) -> int:
        return len(self.pairs)


def parse_sme_coords(path: Path) -> List[SmeCoord]:
    """解析 ST.SME 的曲线类输出坐标记录(Mark=1/2,含 99 续行归并)。

    逐类型槽位语义见模块 docstring;未验证类型出现即 ValueError(拒解不猜)。
    """
    out: List[SmeCoord] = []
    cur: SmeCoord = None
    for line in Path(path).read_text(encoding=ENC, errors="replace").splitlines():
        cells = [c.strip() for c in line.split(",") if c.strip() != ""]
        nums = []
        for c in cells:
            try:
                nums.append(int(float(c)))
            except ValueError:
                break                                  # 尾部 'name' 引号列
        if len(nums) < 2 or nums[0] not in (1, 2):
            cur = None
            continue
        ctype = nums[1]
        slots = nums[2:18]
        if ctype == 99:
            if cur is None:
                raise ValueError("ST.SME:99 续行无宿主记录")
            if cur.coord_type == 3:
                raise ValueError("ST.SME:Type=3 的 99 续行未经样例验证,拒解")
            tgt = cur
        elif ctype in _NONCURVE_TYPES:
            cur = None                                 # 不产列
            continue
        elif ctype in _CURVE_TYPES:
            tgt = SmeCoord(ctype, [])
            out.append(tgt)
            cur = tgt
        else:
            raise ValueError(f"ST.SME:Coord_Type={ctype} 无样例验证,拒解不猜列数")
        if tgt.coord_type == 2:
            # 每槽位一个母线号(手册字段表);0 为补位
            tgt.pairs.extend((b,) for b in slots if b != 0)
        elif tgt.coord_type == 3:
            # 一记录一列:(母线号, 机组序, 子变量)
            if len(slots) < 3 or slots[0] == 0:
                raise ValueError(f"ST.SME:Type=3 记录槽位不足:{slots[:3]}")
            tgt.pairs.append((slots[0], slots[1], slots[2]))
        else:                                          # Type 1 / 12:(i,j) 对
            for a, b in zip(slots[0::2], slots[1::2]):
                if a == 0 and b == 0:
                    continue
                if tgt.coord_type == 12 and a != b:
                    raise ValueError(f"ST.SME:Type=12 两侧母线不同 ({a},{b}) = 相角差"
                                     "曲线,未经样例验证,拒解")
                if tgt.coord_type == 1:
                    # GUI 样例逐对重复两次,按序去重(cepri36 夹具锚定)
                    if tgt.pairs and tgt.pairs[-1] == (a, b):
                        continue
                tgt.pairs.append((a, b))
    return out


def parse_stout_blocks(ddd_path: Path, sme_path: Path,
                       ) -> Tuple[np.ndarray, List[Tuple[SmeCoord, np.ndarray]]]:
    """通用多类型列块解析:返回 (t[N], [(SmeCoord, 块[N, n_cols]), …] 按记录序)。

    文件尺寸/列数不自洽即报错不静默(格式假设失效须重新反推)。
    """
    coords = parse_sme_coords(Path(sme_path))
    if not coords:
        raise ValueError("ST.SME 无曲线类输出坐标")
    ncol = 1 + sum(c.n_cols for c in coords)
    raw = np.fromfile(Path(ddd_path), dtype="<f8")
    if raw.size == 0 or raw.size % ncol != 0:
        raise ValueError(f"STOUT.DDD 尺寸 {raw.size} 与列数 {ncol} 不自洽"
                         f"(SME 坐标 {sum(c.n_cols for c in coords)});"
                         "格式假设失效,须重新反推")
    m = raw.reshape(-1, ncol)
    t = m[:, 0]
    # 非降(故障投/切瞬间输出两行同刻,仿真器惯例)且整体前进
    if not (np.all(np.diff(t) >= -1e-12) and abs(float(t[0])) < 1e-9
            and t[-1] > t[0]):
        raise ValueError("STOUT.DDD 首列非非降时间轴;格式假设失效")
    blocks, k = [], 1
    for c in coords:
        blocks.append((c, m[:, k:k + c.n_cols]))
        k += c.n_cols
    return t, blocks


def extract_angles(blocks) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """逐机功角(°)列:Type=1 全部列 + Type=3 子变量=1 列;(母线,机组序) 去重先见先得
    (ieee39 GUI 模板存在整条重复记录,重复列会使等权 COI 双计该机)。"""
    cols, pairs, seen = [], [], set()
    for c, arr in blocks:
        if c.coord_type == 1:
            it = [(p, i) for i, p in enumerate(c.pairs)]
        elif c.coord_type == 3:
            it = [((p[0], p[1]), i) for i, p in enumerate(c.pairs) if p[2] == 1]
        else:
            continue
        for p, i in it:
            if p in seen:
                continue
            seen.add(p)
            pairs.append(p)
            cols.append(arr[:, i])
    if not cols:
        raise ValueError("STOUT:无逐机功角列(Type=1 或 Type=3 子变量=1)")
    return np.stack(cols, axis=1), pairs


def extract_bus_voltage(blocks) -> Tuple[np.ndarray, List[int]]:
    """母线电压(pu)列:Type=2 全部列;返回 (V[N,B], 母线号列表,重复列先见先得)。"""
    cols, buses, seen = [], [], set()
    for c, arr in blocks:
        if c.coord_type != 2:
            continue
        for i, p in enumerate(c.pairs):
            if p[0] in seen:
                continue
            seen.add(p[0])
            buses.append(p[0])
            cols.append(arr[:, i])
    if not cols:
        return np.empty((0, 0)), []
    return np.stack(cols, axis=1), buses


def extract_terminal_voltage(blocks) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
    """机端电压(pu)。返回 (V[N,G], (母线,机组序) 列表, 全网机组数 n_gen)。

    优先 Type=2 列中机组端母线子集(端母线集 = 功角列的母线号);Type=2 缺失时回退
    Type=3 子变量=3(ieee39 GUI 模板自带)。**覆盖率必须回传**:该模板实测只配 9/10
    机(缺 (17,2)=发电机_31),部分集的逐步最小值 ≥ 全机集,单调朝"判稳"放松——
    调用方(judge)据 len(pairs) vs n_gen 决定是否够格作精确口径,不足则保留全网
    vmin 保守代理(2026-07-13 审查确认的假稳通道)。
    """
    _, ang_pairs = extract_angles(blocks)
    n_gen = len(ang_pairs)
    vbus, buses = extract_bus_voltage(blocks)
    if buses:
        idx = {b: i for i, b in enumerate(buses)}
        hit = [(b, g) for b, g in ang_pairs if b in idx]
        if hit:
            cols = np.stack([vbus[:, idx[b]] for b, _ in hit], axis=1)
            return cols, hit, n_gen
    cols, pairs, seen = [], [], set()
    for c, arr in blocks:
        if c.coord_type != 3:
            continue
        for i, p in enumerate(c.pairs):
            if p[2] == 3 and (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                pairs.append((p[0], p[1]))
                cols.append(arr[:, i])
    if not cols:
        return np.empty((0, 0)), [], n_gen
    return np.stack(cols, axis=1), pairs, n_gen


def parse_stout(ddd_path: Path, sme_path: Path,
                ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """兼容入口:返回 (t[N], angles[N,G](°), 机组 (母线号,机组序) 列表)。

    v0 只支持全 Type=1;阶段 2 起支持混类型列块(ieee39 GUI 模板此前拒解,其
    COI 频率一直走单机极值退化口径——变更记录 2026-07-13)。
    """
    t, blocks = parse_stout_blocks(ddd_path, sme_path)
    angles, pairs = extract_angles(blocks)
    return t, angles, pairs


def max_angle_spread(angles: np.ndarray) -> np.ndarray:
    """逐时间步最大功角差(°)= max−min;与 STANA col6 同义(验证锚)。"""
    return angles.max(axis=1) - angles.min(axis=1)


def coi_frequency(t: np.ndarray, angles: np.ndarray, weights: np.ndarray = None,
                  f0: float = 50.0) -> np.ndarray:
    """COI 系统频率 f_sys(t)[Hz]:逐机功角(°)对时间求导的加权平均折算。

    "系统频率高于51/低于49"判据(2026-07-12 用户裁决)的实现口径:单机转速极值
    (STANA col12/18)在近端故障首摆期间瞬时越 ±1 Hz 很常见(实测 318 例仅此翻案,
    fmax 中位 51.3Hz 后回摆),不构成系统频率越限;COI(惯量加权平均)才是"系统"
    语义。weights 缺省等权(cepri36/39 交付卡 Tj 为 0,惯量权不可得时的合理退化)。

    数值:重复时刻行(投切瞬间)先合并;梯度 = np.gradient(非均匀网格安全);
    360°/s = 1 Hz。
    """
    t = np.asarray(t, dtype=float)
    a = np.asarray(angles, dtype=float)
    keep = np.concatenate([[True], np.diff(t) > 1e-12])
    t, a = t[keep], a[keep]
    if len(t) < 3:
        return np.full(len(t), f0)
    w = np.ones(a.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    coi = (a * w).sum(axis=1) / w.sum()
    dcoi = np.gradient(coi, t)
    return f0 + dcoi / 360.0


def add_bus_voltage_records(sme_path: Path, buses: Sequence[int],
                            name: str = "母线电压") -> int:
    """ST.SME 追加 Coord_Type=2 母线电压曲线记录(阶段 2:机端/逐母线精确口径)。

    手册字段表(ST.SME.txt):每槽位一个计算母线编号,一记录至多 16 条曲线;
    >16 母线拆多条主记录(不用 99 续行——续行头部约束条文对母线清单语义含混,
    多主记录逐条自洽,列序 = 记录序,锚定验证于单测/冒烟)。幂等:既有 Type=2
    记录先整体移除再写(重复列会漂移列序)。插入位置 = 首条非曲线记录(8/11)前
    (样例惯例:曲线记录在前)。返回写入的曲线条数。
    """
    sme_path = Path(sme_path)
    buses = list(dict.fromkeys(int(b) for b in buses))
    if not buses:
        raise ValueError("母线清单为空")
    lines = sme_path.read_text(encoding=ENC, errors="replace").splitlines()
    kept, in_t2 = [], False
    for ln in lines:
        cells = [c.strip() for c in ln.split(",") if c.strip() != ""]
        nums = []
        for c in cells:
            try:
                nums.append(int(float(c)))
            except ValueError:
                break
        is_rec = len(nums) >= 2 and nums[0] in (1, 2)
        if is_rec and nums[1] == 2:
            in_t2 = True
            continue
        if is_rec and nums[1] == 99 and in_t2:
            continue                                   # Type=2 的续行一并移除
        if is_rec and nums[1] != 99:
            in_t2 = False
        if ln.strip():
            kept.append(ln)
    new_recs = []
    for k in range(0, len(buses), 16):
        chunk = buses[k:k + 16]
        slots = chunk + [0] * (17 - len(chunk))        # I1..I16 + I17=0
        new_recs.append("  1, 2," + "".join(f"{v:6d}," for v in slots)
                        + f"'{name}',")
    # 插入到首条 8/11(非曲线)记录之前;找不到则追加到末尾
    ins = len(kept)
    for i, ln in enumerate(kept):
        cells = [c.strip() for c in ln.split(",") if c.strip() != ""]
        try:
            if int(float(cells[0])) in (1, 2) and int(float(cells[1])) in _NONCURVE_TYPES:
                ins = i
                break
        except (ValueError, IndexError):
            continue
    out = kept[:ins] + new_recs + kept[ins:]
    sme_path.write_bytes(("\r\n".join(out) + "\r\n").encode(ENC))
    return len(buses)


def gen_pairs_from_l5(l5_path: Path) -> List[Tuple[int, int]]:
    """SME Type=1 功角曲线的 (计算母线号, 全局机组序) 对,只出 mark=1 在运机。

    机组序 = LF.L5 全局行序(1 起,停运机占位计入序号)——cepri36_golden 交付 SME
    的 8 对 (2,1)(10,2)…(34,8) 即此口径(单测锚定);手册把第二个数写作"另一侧
    发电机所连计算母线编号",样例实证为机组序(118 探针同口径引擎接受)。
    """
    from .cards import GenView
    from .lowlevel import Card
    out: List[Tuple[int, int]] = []
    for i, rec in enumerate(Card.load(Path(l5_path)).records):
        g = GenView(rec, i)
        if g.mark == 1:
            out.append((g.bs_no, i + 1))
    return out


def build_sme_from_scratch(sme_path: Path, l5_path: Path) -> int:
    """从零构造 ST.SME(118 类无 SME 交付工程;2026-07-16 双网实算定稿)。

    **功角曲线的寻址语义随工程 LF.L5 版式二分**(与 STANA 列布局二分同源,实算破译):
    - 带机组序版(L5 21 格,cepri36/39 类):Type=1 槽位 = **4 槽一组
      (I母线, I机组序, J母线, J机组序) 的机组对相对角**——GUI 交付样例的"逐对
      重复两次"实为**自对 (bs,seq,bs,seq) = 绝对功角**(把 8 台机打包成 8 个
      2 槽对只产 4 条相对角曲线,实算 41 列反例)→ 逐机写自对,4 机/行,
      首行头 `  1, 1,`、续行头 `  1,99,`,与 GUI 交付字节逐位同款。解析侧的
      "连续重复对去重"恰与自对语义一致(每机一列绝对角,spread↔STANA 锚)。
    - 基准版(L5 20 格无机组序列,118 类):引擎按**手册原义**读 Type=3 槽位 =
      (母线, 变量码清单)——(bs, seq, 1) 会被读成变量 [seq,1] 产两列(实算
      156 列),Type=1 记录整体被无视(实算 119 列纯电压)→ 写 Coord_Type=3
      `(bs, 1, 1)`:变量清单 [1,1] 引擎去重为 δ 一列(实算 138 列整除 + 判据
      精确口径全通)。

    另写 Coord_Type=8/11 两条——**STANA 极值摘要的产出开关**(只写曲线记录时
    STOUT 正常而 STANA 缺失,实证),逐字照抄 cepri36_golden 末两行。
    Type=2 全母线电压不在此写:调用方随后走 add_bus_voltage_records(其插入位置
    逻辑自会把 Type=2 放到 8/11 之前,列序 = 功角 → 电压)。
    目标已存在且非空即拒绝(从零语义;防覆盖 GUI 模板的既有曲线配置)。
    返回功角曲线条数。
    """
    from .cards import GenView, _gen_layout_offset
    from .lowlevel import Card

    sme_path = Path(sme_path)
    if sme_path.exists() and sme_path.read_bytes().strip():
        raise FileExistsError(f"{sme_path} 已存在且非空:从零构造仅限无 SME 工程"
                              "(既有配置请用 add_bus_voltage_records 追加)")
    l5_recs = Card.load(Path(l5_path)).records
    if not l5_recs:
        raise ValueError(f"{l5_path} 无发电机记录")
    off = _gen_layout_offset(l5_recs[0], GenView.BASE_NCELLS, "LF.L5")
    pairs = gen_pairs_from_l5(l5_path)
    if not pairs:
        raise ValueError("LF.L5 无 mark=1 在运机可配功角曲线")
    recs = []
    if off == 1:                       # 带机组序版:Type=1 自对(绝对功角),4 机/行
        quads = [x for (bs, seq) in pairs for x in (bs, seq, bs, seq)]
        for k in range(0, len(quads), 16):
            slots = quads[k:k + 16]
            slots += [0] * (16 - len(slots))
            head = "  1, 1," if k == 0 else "  1,99,"
            recs.append(head + "".join(f"{v:6d}," for v in slots)
                        + "     0,'发电机功角',")
    else:                              # 基准版:Type=3 (母线, 变量清单 [1,1]→δ)
        for bs, _seq in pairs:
            recs.append(f"  1, 3,{bs:6d},{1:6d},{1:6d}," + "     0," * 14
                        + "'发电机功角',")
    recs.append("  1, 8," + f"{101:6d},{1101:6d}," + "     0," * 15 + "'',")
    recs.append("  1,11," + f"{1:6d}," + "     0," * 16 + "'',")
    sme_path.write_bytes(("\r\n".join(recs) + "\r\n").encode(ENC))
    return len(pairs)
